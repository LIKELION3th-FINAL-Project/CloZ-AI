"""
Manager Agent

실시간 YES/NO 피드백 처리 담당

역할:
- 피드백 분석 → ActionType 결정
- BUYING 판단 시 Wardrobe Checker 연동
- REGENERATE 시 Generation Model 호출
- 세션 로그 기록

LLM: gpt-4o-mini (빠른 응답, 결정론적)
"""

import os
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from datetime import datetime
from dotenv import load_dotenv

from ..models import (
    FeedbackInput,
    FeedbackScope,
    OutfitSet,
    ActionType,
    ManagerDecision,
    SessionLog,
    SessionEntry,
    SessionStatus,
    UserProfile,
)
from ..llm import LLMFactory, BaseLLM
from ..storage import JsonStorage
from ..interfaces import (
    WardrobeCheckerInterface,
    WardrobeCheckResult,
    GenerationModelInterface,
    GenerationResult,
    BuyingTriggerInterface,
    BuyingRecommendation,
)
from ..checkers.embedding_wardrobe_checker import EmbeddingWardrobeChecker
from ..checkers.embedding_buying_trigger import EmbeddingBuyingTrigger
from ..utils.context_analyzer import ContextAnalyzer
from ..utils.query_builder import QueryBuilder
from ..interfaces.wardrobe_checker import DummyWardrobeChecker
from ..interfaces.generation_model import DummyGenerationModel
from ..interfaces.buying_trigger import DummyBuyingTrigger
from .analyst_agent import AnalystAgent




# .env 로드
load_dotenv()


@dataclass
class ManagerConfig:
    """Manager Agent 설정"""
    max_regenerate_count: int = 1           # 최대 재생성 횟수 (1회로 제한)
    enable_wardrobe_check: bool = True      # 옷장 체크 활성화
    enable_buying_recommendation: bool = True  # 구매 추천 활성화


class ManagerAgent:
    """
        session = manager.start_session(
            user_id="user_123",
            original_prompt="오피스룩 추천해줘"
        )

        # 피드백 처리
        decision = manager.process_feedback(
            feedback=FeedbackInput(
                session_id=session.session_id,
                user_id="user_123",
                is_positive=False,
                current_outfit=outfit,
                feedback_text="상의가 너무 어두워요",
                feedback_scope=FeedbackScope.TOP
            )
        )

        # 결정에 따른 처리
        if decision.action == ActionType.APPROVED:
            # 승인됨 → 세션 종료
            manager.end_session(session.session_id)
        elif decision.action == ActionType.BUYING:
            # 구매 추천 표시
            print(decision.buying_recommendations)
    """

    def __init__(
        self,
        config: Optional[ManagerConfig] = None,
        llm: Optional[BaseLLM] = None,
        storage: Optional[JsonStorage] = None,
        wardrobe_checker: Optional[WardrobeCheckerInterface] = None,
        generation_model: Optional[GenerationModelInterface] = None,
        buying_trigger: Optional[BuyingTriggerInterface] = None,
    ):
        self.config = config or ManagerConfig()
        self.storage = storage or JsonStorage()

        # LLM 초기화 (gpt-4o-mini)
        self.llm = llm or LLMFactory.create_manager_agent_llm()

        # 외부 인터페이스 (임베딩 기반 구현체 우선 사용)
        self.wardrobe_checker = wardrobe_checker or EmbeddingWardrobeChecker()
        self.generation_model = generation_model or DummyGenerationModel()
        self.buying_trigger = buying_trigger or EmbeddingBuyingTrigger()
        
        # 지능형 컨텍스트 분석기
        self.context_analyzer = ContextAnalyzer(llm=self.llm)
        
        # 쿼리 빌더 (재생성 시 임베딩용 쿼리 정제)
        self.query_builder = QueryBuilder()
        
        # 후처리 에이전트 (세션 종료 후 UserProfile 업데이트)
        self.analyst_agent = AnalystAgent(storage=self.storage)






        # 활성 세션 캐시
        self._active_sessions: Dict[str, SessionLog] = {}

    # ==================== 세션 관리 ====================

    def start_session(
        self,
        user_id: str,
        original_prompt: str,
        initial_outfit: Optional[OutfitSet] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> SessionLog:
        """
        새 피드백 세션 시작

        Args:
            user_id: 사용자 ID
            original_prompt: 원본 코디 요청
            initial_outfit: 초기 생성된 코디
            context: 추가 컨텍스트

        Returns:
            SessionLog
        """
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{user_id}"

        session = SessionLog(
            session_id=session_id,
            user_id=user_id,
            context={
                "original_prompt": original_prompt,
                **(context or {})
            }
        )

        # 초기 코디가 있으면 기록
        if initial_outfit:
            entry = SessionEntry(
                entry_id=f"entry_{len(session.entries) + 1:03d}",
                entry_type="outfit_generation",
                timestamp=datetime.now().isoformat(),
                content={"outfit": initial_outfit.to_dict()}
            )
            session.add_entry(entry)

        # 캐시에 저장
        self._active_sessions[session_id] = session

        # 스토리지에 저장
        self.storage.save_session(session)

        return session

    def get_session(self, session_id: str) -> Optional[SessionLog]:
        """세션 조회"""
        # 캐시에서 먼저 확인
        if session_id in self._active_sessions:
            return self._active_sessions[session_id]

        # 스토리지에서 로드 (user_id 추출 필요)
        # session_id 형식: session_YYYYMMDD_HHMMSS_user_id
        parts = session_id.split("_")
        if len(parts) >= 4:
            user_id = "_".join(parts[3:])
            session = self.storage.load_session(user_id, session_id)
            if session:
                self._active_sessions[session_id] = session
            return session

        return None

    def end_session(
        self,
        session_id: str,
        status: SessionStatus = SessionStatus.COMPLETED
    ) -> Optional[SessionLog]:
        """
        세션 종료

        Args:
            session_id: 세션 ID
            status: 종료 상태

        Returns:
            종료된 SessionLog
        """
        session = self.get_session(session_id)
        if not session:
            return None

        session.close(status)
        self.storage.save_session(session)

        # 캐시에서 제거
        if session_id in self._active_sessions:
            del self._active_sessions[session_id]

        # COMPLETED 또는 BUYING_REDIRECT 시 AnalystAgent 트리거
        if status in [SessionStatus.COMPLETED, SessionStatus.BUYING_REDIRECT]:
            self._trigger_analyst_agent(session)

        return session

    def _trigger_analyst_agent(self, session: SessionLog) -> None:
        """
        세션 종료 후 AnalystAgent 트리거
        
        - 피드백 히스토리 분석
        - UserProfile 업데이트
        """
        try:
            # 단일 세션 분석
            analysis_result = self.analyst_agent.analyze_session(
                user_id=session.user_id,
                session_id=session.session_id
            )
            
            # 분석 성공 시 UserProfile 업데이트
            if analysis_result.success:
                self.analyst_agent.update_user_profile(
                    user_id=session.user_id,
                    analysis_result=analysis_result
                )
                print(f"[ManagerAgent] AnalystAgent 분석 완료: user={session.user_id}")
            else:
                print(f"[ManagerAgent] AnalystAgent 분석 실패: {analysis_result.error}")
                
        except Exception as e:
            # 후처리 실패해도 세션 종료는 성공으로 처리
            print(f"[ManagerAgent] AnalystAgent 트리거 예외: {e}")


    # ==================== 피드백 처리 ====================

    def process_feedback(self, feedback: FeedbackInput) -> ManagerDecision:
        """
        피드백 처리 메인 로직

        Args:
            feedback: 사용자 피드백

        Returns:
            ManagerDecision (액션 + 메시지 + 부가 정보)
        """
        session = self.get_session(feedback.session_id)
        if not session:
            return ManagerDecision(
                action=ActionType.REGENERATE,
                message="세션을 찾을 수 없습니다.",
                reasoning="Session not found"
            )

        # YES 피드백 → APPROVED
        if feedback.is_positive:
            return self._handle_positive_feedback(session, feedback)

        # NO 피드백 → LLM 분석
        return self._handle_negative_feedback(session, feedback)

    def _handle_positive_feedback(
        self,
        session: SessionLog,
        feedback: FeedbackInput
    ) -> ManagerDecision:
        """YES 피드백 처리 → APPROVED"""
        # 세션에 기록
        entry = SessionEntry(
            entry_id=f"entry_{len(session.entries) + 1:03d}",
            entry_type="feedback",
            timestamp=datetime.now().isoformat(),
            content={
                "is_positive": True,
                "outfit_id": feedback.current_outfit.outfit_id if feedback.current_outfit else None
            }
        )
        session.add_entry(entry)

        # 승인된 코디 저장
        if feedback.current_outfit:
            self.storage.save_approved_outfit(
                user_id=session.user_id,
                outfit=feedback.current_outfit,
                context={
                    "session_id": session.session_id,
                    "original_prompt": session.context.get("original_prompt", "")
                }
            )

        self.storage.save_session(session)

        return ManagerDecision(
            action=ActionType.APPROVED,
            message="코디가 승인되었습니다! 마음에 드셨다니 기쁩니다",
            reasoning="User approved the outfit"
        )

    def _handle_negative_feedback(
        self,
        session: SessionLog,
        feedback: FeedbackInput
    ) -> ManagerDecision:
        """
        NO 피드백 처리

        플로우:
        - 재생성 횟수 체크 → 초과 시 히스토리 종합 BUYING
        - LLM이 피드백 명확/모호 판단
        - 모호하면 → ASK_MORE (추가 질문)
        - 명확하면 → WardrobeChecker로 옷장 확인
           - 옷장에 있음 → REGENERATE (1회만)
           - 옷장에 없음 → BUYING (상품 추천)
        """
        # 피드백 기록
        entry = SessionEntry(
            entry_id=f"entry_{len(session.entries) + 1:03d}",
            entry_type="feedback",
            timestamp=datetime.now().isoformat(),
            content={
                "is_positive": False,
                "feedback_text": feedback.feedback_text,
                "feedback_scopes": [s.value for s in feedback.feedback_scopes] if feedback.feedback_scopes else ["FULL"],
                "outfit_id": feedback.current_outfit.outfit_id if feedback.current_outfit else None
            }
        )
        session.add_entry(entry)

        # 재생성 횟수 체크 (먼저)
        regenerate_count = self._get_regenerate_count(session)
        if regenerate_count >= self.config.max_regenerate_count:
            # 히스토리 종합해서 바로 BUYING (LLM 판단 스킵)
            decision = self._create_buying_decision_with_history(session, feedback)
        else:
            # 지능형 컨텍스트 분석 (ContextAnalyzer 사용)
            analysis = self.context_analyzer.analyze_session_context(session, feedback)

            if not analysis["is_clear"]:
                # 모호함 → ASK_MORE
                decision = ManagerDecision(
                    action=ActionType.ASK_MORE,
                    message=analysis.get("question", "조금 더 구체적으로 어떤 점이 마음에 안 드셨나요?"),
                    reasoning=analysis.get("reasoning", "Feedback is ambiguous"),
                    extracted_requirements=analysis.get("constraints", [])
                )
            else:
                # 명확함 → WardrobeChecker로 옷장 확인 (보강된 쿼리 사용)
                decision = self._decide_by_wardrobe_v2(session, feedback, analysis)


        # 결정 기록
        action_entry = SessionEntry(
            entry_id=f"entry_{len(session.entries) + 1:03d}",
            entry_type="action",
            timestamp=datetime.now().isoformat(),
            content={
                "action": decision.action.value,
                "message": decision.message,
                "reasoning": decision.reasoning
            }
        )
        session.add_entry(action_entry)
        self.storage.save_session(session)

        return decision

    def _check_feedback_clarity(
        self,
        session: SessionLog,
        feedback: FeedbackInput
    ) -> Dict[str, Any]:
        """
        LLM으로 피드백 명확성 판단

        Returns:
            {
                "is_clear": bool,           # 명확한지 여부
                "requirements": [...],       # 추출된 요구사항
                "question": "...",           # 모호할 때 추가 질문
                "partial_requirements": [...] # 부분적으로 추출된 요구사항
            }
        """
        prompt = self._build_clarity_check_prompt(session, feedback)
        system_prompt = self._get_clarity_system_prompt()

        try:
            result = self.llm.generate_json(
                prompt=prompt,
                system_prompt=system_prompt
            )

            if result:
                return {
                    "is_clear": result.get("is_clear", False),
                    "requirements": result.get("requirements", []),
                    "question": result.get("question", ""),
                    "partial_requirements": result.get("partial_requirements", [])
                }
        except Exception as e:
            print(f"LLM 분석 실패: {e}")

        # 기본값: 명확하다고 가정
        return {
            "is_clear": True,
            "requirements": [feedback.feedback_text],
            "question": "",
            "partial_requirements": []
        }

    def _build_clarity_check_prompt(
        self,
        session: SessionLog,
        feedback: FeedbackInput
    ) -> str:
        """피드백 명확성 판단용 프롬프트"""
        prompt = f"""
사용자의 피드백이 명확한지 판단하세요.

## 원본 요청
{session.context.get("original_prompt", "코디 추천")}

## 사용자 피드백
- 범위: {', '.join([s.value for s in feedback.feedback_scopes]) if feedback.feedback_scopes else "전체"}
- 내용: "{feedback.feedback_text}"

## 판단 기준
- 명확함: 구체적인 색상, 스타일, 핏 등이 언급됨
  예: "상의가 너무 어두워요", "좀 더 캐주얼했으면 좋겠어요"
- 모호함: 무엇이 불만인지 알 수 없음
  예: "별로에요", "마음에 안들어요", "다른거요"

## 응답 형식 (JSON)
{{
    "is_clear": true | false,
    "requirements": ["명확할 때: 추출된 요구사항 리스트"],
    "question": "모호할 때: 사용자에게 할 추가 질문",
    "partial_requirements": ["부분적으로 파악된 요구사항"]
}}
"""
        return prompt

    def _get_clarity_system_prompt(self) -> str:
        """피드백 명확성 판단용 시스템 프롬프트"""
        return """당신은 패션 피드백 분석 전문가입니다.

사용자의 피드백이 충분히 명확한지 판단하세요.
- 명확하면 is_clear=true, 요구사항을 requirements에 추출
- 모호하면 is_clear=false, 추가 질문을 question에 작성

추가 질문은 친근하고 자연스러운 톤으로 작성하세요.
항상 유효한 JSON 형식으로 응답하세요."""

    def _decide_by_wardrobe_v2(
        self,
        session: SessionLog,
        feedback: FeedbackInput,
        analysis: Dict[str, Any]
    ) -> ManagerDecision:
        """
        보강된 컨텍스트를 사용하여 REGENERATE/BUYING 결정
        """
        constraints = analysis.get("constraints", [])
        avoid_attributes = analysis.get("avoid_attributes", {})

        # QueryBuilder로 임베딩용 쿼리 정제
        # Note: 현재 피드백은 이미 session.entries에 추가되어 있음 (_handle_negative_feedback에서 Line 308-319)
        feedbacks = self._collect_session_feedbacks(session)

        combined_query = self.query_builder.build_combined_query(
            original_query={"initial_request": session.context.get("original_prompt", "")},
            feedbacks=feedbacks
        )
        refined_query_text = combined_query.combined_text
        structured_info = combined_query.refined_query  # QueryBuilder의 구조화된 정보 활용

        # 재생성 횟수 체크
        regenerate_count = self._get_regenerate_count(session)

        # 이미 재생성 1회 했으면 → BUYING
        if regenerate_count >= self.config.max_regenerate_count:
            return self._create_buying_decision_v2(
                session, feedback, refined_query_text, avoid_attributes, structured_info
            )

        # WardrobeChecker로 옷장 확인
        if self.config.enable_wardrobe_check:
            # FashionCLIP용 영어 요구사항 사용 (있으면), 없으면 한국어 사용
            search_requirements = [refined_query_text]
            if structured_info and hasattr(structured_info, 'requirements_en') and structured_info.requirements_en:
                search_requirements = structured_info.requirements_en

            # 정제된 쿼리로 옷장 검색
            check_result = self.wardrobe_checker.can_fulfill(
                requirements=search_requirements,
                user_id=session.user_id,
                context={
                    "feedback": refined_query_text,  # 정제된 쿼리 사용 (일관성)
                    "refined_query": structured_info.to_dict() if structured_info else {},  # 구조화된 정보 전달
                    "constraints": constraints,
                    "avoid_attributes": avoid_attributes
                }
            )

            if check_result.is_possible:
                return self._create_regenerate_decision_v2(
                    session, feedback, refined_query_text, regenerate_count, structured_info
                )
            else:
                return self._create_buying_decision_v2(
                    session, feedback, refined_query_text, avoid_attributes, structured_info
                )

        return self._create_regenerate_decision_v2(
            session, feedback, refined_query_text, regenerate_count, structured_info
        )

    def _collect_session_feedbacks(self, session: SessionLog) -> List[str]:
        """세션 내 모든 피드백 텍스트 수집"""
        feedbacks = []
        for entry in session.entries:
            if entry.entry_type == "feedback" and not entry.content.get("is_positive"):
                feedbacks.append(entry.content.get("feedback_text", ""))
        return feedbacks


    def _create_regenerate_decision_v2(
        self,
        session: SessionLog,
        feedback: FeedbackInput,
        refined_query: str,
        regenerate_count: int,
        structured_info: Optional[Any] = None
    ) -> ManagerDecision:
        """보강된 쿼리를 포함한 REGENERATE 결정"""
        regenerate_data = self._build_regenerate_data(session, feedback)
        # 보강된 쿼리 추가
        regenerate_data["refined_query"] = refined_query
        # 구조화된 정보 추가 (mood, time, location, requirements, constraints)
        if structured_info:
            regenerate_data["structured_query"] = structured_info.to_dict()

        return ManagerDecision(
            action=ActionType.REGENERATE,
            message="분석된 취향을 반영해서 새로운 코디를 찾아볼게요!",
            reasoning=f"Context refined: {refined_query}",
            extracted_requirements=[refined_query],
            payload={"regenerate_data": regenerate_data}
        )

    def _create_buying_decision_v2(
        self,
        session: SessionLog,
        feedback: FeedbackInput,
        refined_query: str,
        avoid_attributes: Dict[str, Any],
        structured_info: Optional[Any] = None
    ) -> ManagerDecision:
        """보강된 쿼리를 포함한 BUYING 결정"""
        primary_scope = feedback.feedback_scopes[0] if feedback.feedback_scopes else FeedbackScope.FULL

        # 구조화된 정보도 context로 전달
        context = {
            "avoid_attributes": avoid_attributes
        }
        if structured_info:
            context["structured_query"] = structured_info.to_dict()

        buying_result = self.buying_trigger.recommend(
            original_prompt=session.context.get("original_prompt", ""),
            feedback_text=refined_query,  # 보강된 쿼리 사용
            feedback_scope=primary_scope,
            current_outfit=feedback.current_outfit,
            limit=5,
            context=context  # 구조화된 정보 전달
        )

        return ManagerDecision(
            action=ActionType.BUYING,
            message="옷장에 딱 맞는 아이템이 없네요. 유저님의 취향을 반영한 상품들을 추천해 드릴게요.",
            reasoning=f"Buying suggested with refined query: {refined_query}",
            extracted_requirements=[refined_query],
            buying_recommendations=buying_result
        )


    def _create_regenerate_decision(
        self,
        session: SessionLog,
        feedback: FeedbackInput,
        requirements: List[str],
        regenerate_count: int
    ) -> ManagerDecision:
        """REGENERATE 결정 생성"""
        # 생성팀에 전달할 데이터 구성
        regenerate_data = self._build_regenerate_data(session, feedback)

        return ManagerDecision(
            action=ActionType.REGENERATE,
            message="피드백을 반영해서 새로운 코디를 준비할게요!",
            reasoning=f"Wardrobe has matching items. Regenerate attempt {regenerate_count + 1}/{self.config.max_regenerate_count}",
            extracted_requirements=requirements,
            target_categories=[s.value for s in feedback.feedback_scopes] if feedback.feedback_scopes else [],
            payload={"regenerate_data": regenerate_data}
        )

    def _create_buying_decision(
        self,
        session: SessionLog,
        feedback: FeedbackInput,
        requirements: List[str]
    ) -> ManagerDecision:
        """BUYING 결정 생성"""
        if not self.config.enable_buying_recommendation:
            # 구매 추천 비활성화 시 메시지만
            return ManagerDecision(
                action=ActionType.BUYING,
                message="옷장에 딱 맞는 아이템이 없네요.",
                reasoning="Wardrobe check failed, buying recommendation disabled",
                extracted_requirements=requirements
            )

        # BuyingTrigger로 상품 추천
        # 복수 scope 중 첫 번째 사용 (BUYING은 단일 scope 기반 필터링)
        primary_scope = feedback.feedback_scopes[0] if feedback.feedback_scopes else FeedbackScope.FULL

        buying_result = self.buying_trigger.recommend(
            original_prompt=session.context.get("original_prompt", ""),
            feedback_text=feedback.feedback_text,
            feedback_scope=primary_scope,
            current_outfit=feedback.current_outfit,
            limit=5
        )

        return ManagerDecision(
            action=ActionType.BUYING,
            message="옷장에 딱 맞는 아이템이 없네요. 이런 상품은 어떠세요?",
            reasoning="Wardrobe check failed, recommending products",
            extracted_requirements=requirements,
            buying_recommendations=buying_result
        )

    def _build_regenerate_data(
        self,
        session: SessionLog,
        feedback: FeedbackInput
    ) -> Dict[str, Any]:
        """
        생성팀에 전달할 재생성 데이터 구성

        Returns:
            {
                "feedback": [
                    {
                        "scopes": ["FULL"],
                        "text": "별로에요"
                    },
                    {
                        "scopes": ["TOP"],
                        "text": "상의가 너무 딱딱해요"
                    }
                ],
                "clarifications": [
                    {
                        "type": "ask_more",
                        "question": "어떤 부분이...?",
                        "answer": "상의가 너무 딱딱해요"
                    }
                ]
            }
        """
        # 세션에서 대화 히스토리 추출
        entries = session.entries

        # 모든 피드백 수집 (순서대로)
        feedback_list = []
        for entry in entries:
            if entry.entry_type == "feedback" and not entry.content.get("is_positive"):
                fb_scopes = entry.content.get("feedback_scopes", ["FULL"])
                fb_text = entry.content.get("feedback_text", "")
                if fb_text:
                    feedback_list.append({
                        "scopes": fb_scopes,
                        "text": fb_text
                    })

        # 현재 피드백도 추가 (중복 방지)
        current_fb = {
            "scopes": [s.value for s in (feedback.feedback_scopes or [FeedbackScope.FULL])],
            "text": feedback.feedback_text
        }
        if current_fb not in feedback_list:
            feedback_list.append(current_fb)

        data = {
            "feedback": feedback_list
        }

        # ASK_MORE 질문/답변 쌍만 clarifications로
        clarifications = []
        ask_more_exists = False

        # ASK_MORE 질문/답변 쌍 추출
        for i, entry in enumerate(entries):
            if entry.entry_type == "action" and entry.content.get("action") == "ASK_MORE":
                ask_more_exists = True
                question = entry.content.get("message", "")
                # 다음 피드백 엔트리에서 답변 찾기
                for next_entry in entries[i+1:]:
                    if next_entry.entry_type == "feedback":
                        answer = next_entry.content.get("feedback_text", "")
                        if question and answer:
                            clarifications.append({
                                "type": "ask_more",
                                "question": question,
                                "answer": answer
                            })
                        break

        if clarifications:
            data["clarifications"] = clarifications

        return data


    # ==================== 유틸리티 ====================

    def _get_regenerate_count(self, session: SessionLog) -> int:
        """세션에서 재생성 횟수 조회"""
        return sum(
            1 for e in session.entries
            if e.entry_type == "action" and e.content.get("action") == "REGENERATE"
        )

    def _create_buying_decision_with_history(
        self,
        session: SessionLog,
        feedback: FeedbackInput
    ) -> ManagerDecision:
        """
        히스토리 기반 BUYING 결정 - 모든 피드백 종합

        재생성 횟수 초과 시 이전 피드백들을 모아서 추천에 활용
        """
        # 세션에서 모든 피드백 텍스트 수집
        all_feedbacks = []
        for entry in session.entries:
            if entry.entry_type == "feedback" and not entry.content.get("is_positive"):
                fb_text = entry.content.get("feedback_text", "")
                if fb_text:
                    all_feedbacks.append(fb_text)

        # 현재 피드백도 추가 (중복 방지)
        if feedback.feedback_text and feedback.feedback_text not in all_feedbacks:
            all_feedbacks.append(feedback.feedback_text)

        # 종합 피드백으로 추천 요청
        combined_feedback = " / ".join(all_feedbacks)

        buying_result = None
        if self.config.enable_buying_recommendation:
            # 복수 scope 중 첫 번째 사용
            primary_scope = feedback.feedback_scopes[0] if feedback.feedback_scopes else FeedbackScope.FULL

            buying_result = self.buying_trigger.recommend(
                original_prompt=session.context.get("original_prompt", ""),
                feedback_text=combined_feedback,  # 종합 피드백 사용
                feedback_scope=primary_scope,
                current_outfit=feedback.current_outfit,
                limit=5
            )

        return ManagerDecision(
            action=ActionType.BUYING,
            message="여러 번 시도해봤지만 딱 맞는 코디를 찾기 어렵네요. 이런 상품은 어떠세요?",
            reasoning=f"Max regenerate ({self.config.max_regenerate_count}) exceeded. Using feedback history: {all_feedbacks}",
            extracted_requirements=all_feedbacks,
            buying_recommendations=buying_result
        )

    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """세션 요약 조회"""
        session = self.get_session(session_id)
        if not session:
            return {}

        return {
            "session_id": session.session_id,
            "user_id": session.user_id,
            "status": session.status.value,
            "entry_count": len(session.entries),
            "summary": session.summary,
            "started_at": session.started_at,
            "ended_at": session.ended_at,
        }
