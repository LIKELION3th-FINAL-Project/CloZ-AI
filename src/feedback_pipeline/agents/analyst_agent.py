"""
Analyst Agent

세션 종료 후 사용자 취향 분석 담당

역할:
1. 세션 로그에서 패턴 추출
2. 승인된 코디에서 취향 학습
3. UserProfile.biases 업데이트
4. context_summary 생성

LLM: gpt-4o (고품질 분석)
실행: 비동기/배치 (세션 종료 후)
"""

import os
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from datetime import datetime
from dotenv import load_dotenv

from ..models import (
    UserProfile,
    UserBias,
    ColorPreferences,
    FitPreferences,
    SessionLog,
    SessionStatus,
    KeywordMention,
    MIN_MENTIONS_FOR_PROFILE,
)
from ..models.feedback import FeedbackScope, OutfitSet
from ..llm import LLMFactory, BaseLLM
from ..storage import JsonStorage

# .env 로드
load_dotenv()


@dataclass
class AnalysisResult:
    """분석 결과"""
    success: bool
    updated_biases: Optional[UserBias] = None
    context_summary: str = ""
    insights: List[str] = None
    confidence: float = 0.0
    error: Optional[str] = None

    def __post_init__(self):
        if self.insights is None:
            self.insights = []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "updated_biases": self.updated_biases.to_dict() if self.updated_biases else None,
            "context_summary": self.context_summary,
            "insights": self.insights,
            "confidence": self.confidence,
            "error": self.error,
        }


@dataclass
class FeedbackAnalysisResult:
    """
    피드백 분석 결과 (BUYING 추천용)

    LLM이 피드백 텍스트를 분석하여 추출한 정보
    """
    extracted_keywords: List[str] = None      # 원하는 키워드
    avoid_keywords: List[str] = None          # 피하고 싶은 키워드
    style_direction: str = ""                 # 스타일 방향 요약
    category_sub_hints: List[str] = None      # 추천 세부 카테고리
    subjective_mappings: Dict[str, str] = None  # 주관적 표현 매핑
    profile_updates: Optional[Dict] = None    # UserProfile에 반영할 내용

    def __post_init__(self):
        if self.extracted_keywords is None:
            self.extracted_keywords = []
        if self.avoid_keywords is None:
            self.avoid_keywords = []
        if self.category_sub_hints is None:
            self.category_sub_hints = []
        if self.subjective_mappings is None:
            self.subjective_mappings = {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "extracted_keywords": self.extracted_keywords,
            "avoid_keywords": self.avoid_keywords,
            "style_direction": self.style_direction,
            "category_sub_hints": self.category_sub_hints,
            "subjective_mappings": self.subjective_mappings,
            "profile_updates": self.profile_updates,
        }


@dataclass
class AnalystConfig:
    """Analyst Agent 설정"""
    min_sessions_for_analysis: int = 3      # 분석에 필요한 최소 세션 수
    max_sessions_to_analyze: int = 20       # 분석할 최대 세션 수
    min_approved_outfits: int = 2           # 분석에 필요한 최소 승인 코디 수
    style_keywords_limit: int = 20          # 최대 스타일 키워드 수
    raw_insights_limit: int = 10            # 최대 인사이트 수
    context_summary_max_length: int = 2000  # context_summary 최대 길이


class AnalystAgent:
    """
    Analyst Agent

    세션 종료 후 사용자 취향 분석

    사용 예시:
        analyst = AnalystAgent()

        # 단일 세션 분석
        result = analyst.analyze_session(
            user_id="user_123",
            session_id="session_001"
        )

        # 전체 히스토리 기반 분석
        result = analyst.analyze_user_history(user_id="user_123")

        # 프로필 업데이트
        if result.success:
            analyst.update_user_profile(
                user_id="user_123",
                analysis_result=result
            )
    """

    def __init__(
        self,
        config: Optional[AnalystConfig] = None,
        llm: Optional[BaseLLM] = None,
        storage: Optional[JsonStorage] = None,
    ):
        self.config = config or AnalystConfig()
        self.storage = storage or JsonStorage()

        # LLM 초기화
        self.llm = llm or LLMFactory.create_analyst_agent_llm()

    # ==================== 분석 ====================

    def analyze_session(
        self,
        user_id: str,
        session_id: str
    ) -> AnalysisResult:
        """
        단일 세션 분석

        Args:
            user_id: 사용자 ID
            session_id: 세션 ID

        Returns:
            AnalysisResult
        """
        session = self.storage.load_session(user_id, session_id)
        if not session:
            return AnalysisResult(
                success=False,
                error=f"Session not found: {session_id}"
            )

        return self._analyze_sessions([session], user_id)

    def analyze_user_history(
        self,
        user_id: str,
        include_approved_outfits: bool = True
    ) -> AnalysisResult:
        """
        사용자 전체 히스토리 분석

        Args:
            user_id: 사용자 ID
            include_approved_outfits: 승인된 코디 포함 여부

        Returns:
            AnalysisResult
        """
        # 최근 세션 로드
        sessions = self.storage.get_recent_sessions(
            user_id=user_id,
            limit=self.config.max_sessions_to_analyze
        )

        if len(sessions) < self.config.min_sessions_for_analysis:
            return AnalysisResult(
                success=False,
                error=f"Not enough sessions. Need at least {self.config.min_sessions_for_analysis}, got {len(sessions)}",
                confidence=0.0
            )

        # 승인된 코디 로드
        approved_outfits = []
        if include_approved_outfits:
            approved_outfits = self.storage.load_approved_outfits(user_id)

        return self._analyze_sessions(sessions, user_id, approved_outfits)

    def _analyze_sessions(
        self,
        sessions: List[SessionLog],
        user_id: str,
        approved_outfits: List[Dict[str, Any]] = None
    ) -> AnalysisResult:
        """세션들 분석하여 취향 추출"""
        if approved_outfits is None:
            approved_outfits = []

        # 프롬프트 구성
        prompt = self._build_analysis_prompt(sessions, approved_outfits)
        system_prompt = self._get_system_prompt()

        try:
            # LLM 호출 (JSON 모드)
            result = self.llm.generate_json(
                prompt=prompt,
                system_prompt=system_prompt
            )

            if result:
                return self._parse_analysis_result(result)

        except Exception as e:
            print(f"분석 실패: {e}")
            return AnalysisResult(
                success=False,
                error=str(e)
            )

        return AnalysisResult(
            success=False,
            error="LLM analysis returned no result"
        )

    def _build_analysis_prompt(
        self,
        sessions: List[SessionLog],
        approved_outfits: List[Dict[str, Any]]
    ) -> str:
        """분석용 프롬프트 생성"""
        # 세션 요약 정보
        session_summaries = []
        for session in sessions:
            feedbacks = []
            for entry in session.entries:
                if entry.entry_type == "feedback":
                    is_positive = entry.content.get("is_positive", False)
                    text = entry.content.get("feedback_text", "")
                    scope = entry.content.get("feedback_scope", "")
                    if is_positive:
                        feedbacks.append("[승인]")
                    elif text:
                        feedbacks.append(f"[거절] {text} ({scope})")

            session_summaries.append({
                "prompt": session.context.get("original_prompt", ""),
                "feedbacks": feedbacks,
                "status": session.status.value
            })

        # 승인된 코디 요약
        approved_summary = []
        for outfit_entry in approved_outfits[:10]:  # 최대 10개
            outfit = outfit_entry.get("outfit", {})
            products = outfit.get("products", [])
            items = [f"{p.get('product_name')} ({p.get('category_main')})" for p in products]
            approved_summary.append({
                "items": items,
                "context": outfit_entry.get("context", {}).get("original_prompt", "")
            })

        prompt = f"""
사용자의 패션 취향을 분석하세요.

## 세션 히스토리 ({len(sessions)}개)
{self._format_sessions(session_summaries)}

## 승인된 코디 ({len(approved_summary)}개)
{self._format_approved_outfits(approved_summary)}

## 분석 항목
1. color_preferences: 선호/회피 색상
2. fit_preferences: 핏 선호도 (oversized, regular, slim)
3. style_keywords: 스타일 키워드 (최대 {self.config.style_keywords_limit}개)
4. avoid_keywords: 회피 키워드
5. subjective_mappings: 주관적 표현 → 구체적 매핑
6. raw_insights: 분석 인사이트 (최대 {self.config.raw_insights_limit}개)
7. context_summary: LLM 컨텍스트용 요약 (최대 {self.config.context_summary_max_length}자)
8. confidence: 분석 신뢰도 (0.0 ~ 1.0)

## 응답 형식 (JSON)
{{
    "color_preferences": {{
        "warm": ["베이지", "브라운"],
        "cool": ["네이비", "그레이"],
        "avoid": ["핫핑크", "형광"]
    }},
    "fit_preferences": {{
        "top": "oversized",
        "bottom": "regular",
        "outer": "regular",
        "overall": "전반적으로 편안한 핏 선호"
    }},
    "style_keywords": ["캐주얼", "미니멀", "클린"],
    "avoid_keywords": ["화려한", "패턴"],
    "subjective_mappings": {{
        "차가운 색": "네이비, 그레이, 블랙",
        "편한 핏": "오버핏 상의 + 와이드 하의"
    }},
    "raw_insights": [
        "상의는 밝은 색을 선호하는 경향",
        "하의는 무난한 색상 선호"
    ],
    "context_summary": "20대 여성, 미니멀 캐주얼 스타일 선호...",
    "confidence": 0.85
}}

세션 데이터를 기반으로 취향을 분석하고 JSON으로 응답하세요.
"""
        return prompt

    def _format_sessions(self, session_summaries: List[Dict]) -> str:
        """세션 요약 포맷팅"""
        lines = []
        for i, s in enumerate(session_summaries[:15], 1):  # 최대 15개
            lines.append(f"### 세션 {i}")
            lines.append(f"요청: {s['prompt']}")
            lines.append(f"피드백: {', '.join(s['feedbacks']) or '없음'}")
            lines.append(f"상태: {s['status']}")
            lines.append("")
        return "\n".join(lines)

    def _format_approved_outfits(self, approved_summary: List[Dict]) -> str:
        """승인 코디 요약 포맷팅"""
        if not approved_summary:
            return "승인된 코디 없음"

        lines = []
        for i, a in enumerate(approved_summary, 1):
            lines.append(f"{i}. {a['context']}")
            lines.append(f"   아이템: {', '.join(a['items'])}")
        return "\n".join(lines)

    def _get_system_prompt(self) -> str:
        """시스템 프롬프트"""
        return """당신은 패션 취향 분석 전문가입니다.

사용자의 피드백 히스토리를 분석하여 취향을 추출합니다.

핵심 원칙:
1. NO 피드백에서 회피 패턴을 추출하세요.
2. YES(승인) 피드백에서 선호 패턴을 추출하세요.
3. 반복되는 패턴에 높은 가중치를 두세요.
4. 모호한 표현은 구체적으로 매핑하세요 (subjective_mappings).
5. confidence는 데이터 양과 일관성에 따라 결정하세요.
6. context_summary는 향후 코디 생성에 사용될 수 있도록 명확하게 작성하세요.
7. 항상 유효한 JSON 형식으로 응답하세요."""

    def _parse_analysis_result(self, result: Dict[str, Any]) -> AnalysisResult:
        """LLM 응답 파싱"""
        try:
            color_prefs = result.get("color_preferences", {})
            fit_prefs = result.get("fit_preferences", {})

            updated_biases = UserBias(
                color_preferences=ColorPreferences(
                    warm=color_prefs.get("warm", []),
                    cool=color_prefs.get("cool", []),
                    avoid=color_prefs.get("avoid", []),
                ),
                fit_preferences=FitPreferences(
                    top=fit_prefs.get("top"),
                    bottom=fit_prefs.get("bottom"),
                    outer=fit_prefs.get("outer"),
                    overall=fit_prefs.get("overall", ""),
                ),
                style_keywords=result.get("style_keywords", [])[:self.config.style_keywords_limit],
                avoid_keywords=result.get("avoid_keywords", []),
                subjective_mappings=result.get("subjective_mappings", {}),
                raw_insights=result.get("raw_insights", [])[:self.config.raw_insights_limit],
            )

            return AnalysisResult(
                success=True,
                updated_biases=updated_biases,
                context_summary=result.get("context_summary", "")[:self.config.context_summary_max_length],
                insights=result.get("raw_insights", []),
                confidence=result.get("confidence", 0.5),
            )

        except Exception as e:
            return AnalysisResult(
                success=False,
                error=f"Failed to parse analysis result: {e}"
            )

    # ==================== 프로필 업데이트 ====================

    def update_user_profile(
        self,
        user_id: str,
        analysis_result: AnalysisResult
    ) -> Optional[UserProfile]:
        """
        분석 결과로 프로필 업데이트

        Args:
            user_id: 사용자 ID
            analysis_result: 분석 결과

        Returns:
            업데이트된 UserProfile
        """
        if not analysis_result.success or not analysis_result.updated_biases:
            return None

        # 기존 프로필 로드 또는 생성
        profile = self.storage.get_or_create_profile(user_id)

        # biases 업데이트 (머지)
        profile.biases = self._merge_biases(
            existing=profile.biases,
            new=analysis_result.updated_biases,
            confidence=analysis_result.confidence
        )

        # context_summary 업데이트
        if analysis_result.context_summary:
            profile.context_summary = analysis_result.context_summary

        # 타임스탬프 업데이트
        profile.update_timestamp()

        # 저장
        self.storage.save_user_profile(profile)

        return profile

    def _merge_biases(
        self,
        existing: UserBias,
        new: UserBias,
        confidence: float
    ) -> UserBias:
        """기존 biases와 새 biases 머지"""
        # 신뢰도가 높으면 새 값으로 대체, 낮으면 기존 유지하면서 추가

        def merge_list(old: List, new: List, limit: int = None) -> List:
            """리스트 머지 (중복 제거, 새 값 우선)"""
            merged = list(new)  # 새 값이 앞에
            for item in old:
                if item not in merged:
                    merged.append(item)
            return merged[:limit] if limit else merged

        def merge_dict(old: Dict, new: Dict) -> Dict:
            """딕셔너리 머지 (새 값 우선)"""
            merged = dict(old)
            merged.update(new)
            return merged

        return UserBias(
            color_preferences=ColorPreferences(
                warm=merge_list(existing.color_preferences.warm, new.color_preferences.warm),
                cool=merge_list(existing.color_preferences.cool, new.color_preferences.cool),
                avoid=merge_list(existing.color_preferences.avoid, new.color_preferences.avoid),
            ),
            fit_preferences=FitPreferences(
                top=new.fit_preferences.top or existing.fit_preferences.top,
                bottom=new.fit_preferences.bottom or existing.fit_preferences.bottom,
                outer=new.fit_preferences.outer or existing.fit_preferences.outer,
                overall=new.fit_preferences.overall or existing.fit_preferences.overall,
            ),
            style_keywords=merge_list(
                existing.style_keywords,
                new.style_keywords,
                limit=self.config.style_keywords_limit
            ),
            avoid_keywords=merge_list(existing.avoid_keywords, new.avoid_keywords),
            subjective_mappings=merge_dict(existing.subjective_mappings, new.subjective_mappings),
            confidence_scores=merge_dict(existing.confidence_scores, new.confidence_scores),
            raw_insights=merge_list(
                existing.raw_insights,
                new.raw_insights,
                limit=self.config.raw_insights_limit
            ),
        )

    # ==================== 배치 처리 ====================

    def run_batch_analysis(
        self,
        user_ids: Optional[List[str]] = None
    ) -> Dict[str, AnalysisResult]:
        """
        여러 사용자 배치 분석

        Args:
            user_ids: 분석할 사용자 ID 리스트 (None이면 전체)

        Returns:
            {user_id: AnalysisResult} 딕셔너리
        """
        # TODO: 전체 사용자 목록 조회 기능 필요
        if user_ids is None:
            return {}

        results = {}
        for user_id in user_ids:
            result = self.analyze_user_history(user_id)
            results[user_id] = result

            if result.success:
                self.update_user_profile(user_id, result)

        return results

    # ==================== BUYING용 피드백 분석 ====================

    def analyze_feedback_for_recommendation(
        self,
        feedback_text: str,
        feedback_scopes: List[FeedbackScope],
        current_outfit: Optional[OutfitSet] = None,
        user_profile: Optional[UserProfile] = None,
        session_id: Optional[str] = None,
        context: Optional[Dict] = None
    ) -> FeedbackAnalysisResult:
        """
        BUYING 시 피드백 분석

        LLM을 사용해 피드백 텍스트에서:
        1. 원하는 스타일 키워드 추출
        2. 회피 키워드 추출
        3. 적합한 세부 카테고리 추천
        4. 신뢰도 기반 Profile 업데이트 내용 생성

        Args:
            feedback_text: 사용자 피드백 텍스트
            feedback_scopes: 피드백 범위 (TOP, BOTTOM, OUTER, FULL)
            current_outfit: 현재 코디 (선택)
            user_profile: 사용자 프로필 (기존 취향 참조용)
            session_id: 세션 ID (keyword mention 추적용)
            context: 추가 컨텍스트

        Returns:
            FeedbackAnalysisResult
        """
        # 프롬프트 구성
        prompt = self._build_feedback_analysis_prompt(
            feedback_text=feedback_text,
            feedback_scopes=feedback_scopes,
            current_outfit=current_outfit,
            user_profile=user_profile,
            context=context
        )

        system_prompt = self._get_feedback_analysis_system_prompt()

        try:
            # LLM 호출 (JSON 모드)
            result = self.llm.generate_json(
                prompt=prompt,
                system_prompt=system_prompt
            )

            if result:
                analysis = self._parse_feedback_analysis_result(result)

                # 신뢰도 기반 Profile 업데이트 내용 생성
                if user_profile and session_id:
                    analysis.profile_updates = self._generate_profile_updates(
                        analysis=analysis,
                        user_profile=user_profile,
                        session_id=session_id
                    )

                return analysis

        except Exception as e:
            print(f"피드백 분석 실패: {e}")

        return FeedbackAnalysisResult()

    def _build_feedback_analysis_prompt(
        self,
        feedback_text: str,
        feedback_scopes: List[FeedbackScope],
        current_outfit: Optional[OutfitSet],
        user_profile: Optional[UserProfile],
        context: Optional[Dict]
    ) -> str:
        """피드백 분석용 프롬프트 생성"""
        scope_str = ", ".join([s.value for s in feedback_scopes])

        # 현재 코디 정보
        outfit_info = ""
        if current_outfit and current_outfit.products:
            items = [f"- {p.product_name} ({p.category_main}/{p.category_sub})"
                     for p in current_outfit.products]
            outfit_info = f"\n## 현재 코디\n" + "\n".join(items)

        # 기존 프로필 정보
        profile_info = ""
        if user_profile and user_profile.biases:
            biases = user_profile.biases
            if biases.style_keywords:
                profile_info += f"\n선호 스타일: {', '.join(biases.style_keywords[:10])}"
            if biases.avoid_keywords:
                profile_info += f"\n회피 키워드: {', '.join(biases.avoid_keywords[:5])}"

        # 원본 요청
        original_prompt = ""
        if context and context.get("original_prompt"):
            original_prompt = f"\n## 원본 요청\n{context['original_prompt']}"

        prompt = f"""
사용자의 피드백을 분석하여 상품 추천에 사용할 키워드를 추출하세요.

## 피드백
"{feedback_text}"

## 변경 범위
{scope_str}
{outfit_info}
{original_prompt}

## 기존 사용자 취향
{profile_info if profile_info else "정보 없음"}

## 분석 항목
1. extracted_keywords: 사용자가 원하는 스타일/특성 (예: 캐주얼, 편한, 밝은)
2. avoid_keywords: 사용자가 피하고 싶은 스타일/특성 (예: 딱딱한, 포멀)
3. style_direction: 원하는 방향 요약 (1-2문장)
4. category_sub_hints: 추천할 세부 카테고리 (예: 맨투맨/스웨트셔츠, 후드 티셔츠)

## 세부 카테고리 목록
상의: 반소매 티셔츠, 긴소매 티셔츠, 니트/스웨터, 맨투맨/스웨트셔츠, 민소매 티셔츠, 셔츠/블라우스, 피케/카라 티셔츠, 후드 티셔츠
바지: 데님 팬츠, 코튼 팬츠, 슈트 팬츠/슬랙스, 트레이닝/조거 팬츠, 숏 팬츠
아우터: 카디건, 플리스/뽀글이, 후드 집업, 블루종/MA-1, 트러커 재킷, 코트

## 응답 형식 (JSON)
{{
    "extracted_keywords": ["캐주얼", "편한", "부드러운"],
    "avoid_keywords": ["포멀", "딱딱한"],
    "style_direction": "캐주얼하고 편안한 데일리 스타일",
    "category_sub_hints": ["맨투맨/스웨트셔츠", "후드 티셔츠"],
    "subjective_mappings": {{
        "딱딱한": "정장 느낌의 포멀한 핏"
    }}
}}

피드백을 분석하고 JSON으로 응답하세요.
"""
        return prompt

    def _get_feedback_analysis_system_prompt(self) -> str:
        """피드백 분석 시스템 프롬프트"""
        return """당신은 패션 피드백 분석 전문가입니다.

사용자의 피드백 텍스트를 분석하여 상품 추천에 사용할 키워드를 추출합니다.

핵심 원칙:
1. 피드백에서 명확한 의도를 추출하세요.
2. "딱딱한", "무거운" 등 주관적 표현을 구체적인 패션 용어로 매핑하세요.
3. 변경 범위(TOP, BOTTOM 등)를 고려하여 해당 카테고리 위주로 분석하세요.
4. category_sub_hints는 실제 존재하는 카테고리명만 사용하세요.
5. 항상 유효한 JSON 형식으로 응답하세요."""

    def _parse_feedback_analysis_result(self, result: Dict[str, Any]) -> FeedbackAnalysisResult:
        """피드백 분석 결과 파싱"""
        return FeedbackAnalysisResult(
            extracted_keywords=result.get("extracted_keywords", []),
            avoid_keywords=result.get("avoid_keywords", []),
            style_direction=result.get("style_direction", ""),
            category_sub_hints=result.get("category_sub_hints", []),
            subjective_mappings=result.get("subjective_mappings", {}),
        )

    def _generate_profile_updates(
        self,
        analysis: FeedbackAnalysisResult,
        user_profile: UserProfile,
        session_id: str
    ) -> Dict[str, Any]:
        """
        신뢰도 기반 Profile 업데이트 내용 생성

        3회 이상 언급된 키워드만 실제 profile에 저장

        Returns:
            {
                "style_keywords_to_add": [...],      # 3회 달성하여 추가될 키워드
                "avoid_keywords_to_add": [...],      # 3회 달성하여 추가될 키워드
                "mentions_updated": {               # 언급 횟수 업데이트 정보
                    "style": [{"keyword": "캐주얼", "count": 2, "threshold": 3}],
                    "avoid": [{"keyword": "딱딱한", "count": 1, "threshold": 3}]
                }
            }
        """
        profile_updates = {
            "style_keywords_to_add": [],
            "avoid_keywords_to_add": [],
            "mentions_updated": {
                "style": [],
                "avoid": []
            }
        }

        # 스타일 키워드 처리
        for keyword in analysis.extracted_keywords:
            # 기존 mention 찾기
            existing = next(
                (m for m in user_profile.biases.style_keyword_mentions if m.keyword == keyword),
                None
            )
            current_count = existing.mention_count if existing else 0
            new_count = current_count + 1

            # 3회 달성 여부 확인
            if new_count >= MIN_MENTIONS_FOR_PROFILE and keyword not in user_profile.biases.style_keywords:
                profile_updates["style_keywords_to_add"].append(keyword)

            profile_updates["mentions_updated"]["style"].append({
                "keyword": keyword,
                "count": new_count,
                "threshold": MIN_MENTIONS_FOR_PROFILE,
                "will_be_added": new_count >= MIN_MENTIONS_FOR_PROFILE
            })

        # 회피 키워드 처리
        for keyword in analysis.avoid_keywords:
            existing = next(
                (m for m in user_profile.biases.avoid_keyword_mentions if m.keyword == keyword),
                None
            )
            current_count = existing.mention_count if existing else 0
            new_count = current_count + 1

            if new_count >= MIN_MENTIONS_FOR_PROFILE and keyword not in user_profile.biases.avoid_keywords:
                profile_updates["avoid_keywords_to_add"].append(keyword)

            profile_updates["mentions_updated"]["avoid"].append({
                "keyword": keyword,
                "count": new_count,
                "threshold": MIN_MENTIONS_FOR_PROFILE,
                "will_be_added": new_count >= MIN_MENTIONS_FOR_PROFILE
            })

        return profile_updates

    def apply_profile_updates(
        self,
        user_id: str,
        session_id: str,
        analysis: FeedbackAnalysisResult
    ) -> Optional[UserProfile]:
        """
        세션 종료 후 Profile 업데이트 실행

        신뢰도 기반으로 keyword mention을 업데이트하고,
        3회 이상 언급된 키워드를 실제 profile에 추가

        Args:
            user_id: 사용자 ID
            session_id: 세션 ID
            analysis: 피드백 분석 결과

        Returns:
            업데이트된 UserProfile
        """
        profile = self.storage.get_or_create_profile(user_id)

        promoted_style = []
        promoted_avoid = []

        # 스타일 키워드 mention 업데이트
        for keyword in analysis.extracted_keywords:
            was_promoted = profile.biases.add_style_keyword_mention(keyword, session_id)
            if was_promoted:
                promoted_style.append(keyword)

        # 회피 키워드 mention 업데이트
        for keyword in analysis.avoid_keywords:
            was_promoted = profile.biases.add_avoid_keyword_mention(keyword, session_id)
            if was_promoted:
                promoted_avoid.append(keyword)

        # 주관적 매핑 추가
        if analysis.subjective_mappings:
            profile.biases.subjective_mappings.update(analysis.subjective_mappings)

        # 타임스탬프 업데이트
        profile.update_timestamp()

        # 저장
        self.storage.save_user_profile(profile)

        # 로깅
        if promoted_style or promoted_avoid:
            print(f"[AnalystAgent] Profile 업데이트 (user: {user_id})")
            if promoted_style:
                print(f"  - 새 선호 키워드: {promoted_style}")
            if promoted_avoid:
                print(f"  - 새 회피 키워드: {promoted_avoid}")

        return profile
