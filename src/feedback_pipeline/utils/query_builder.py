"""
Query Builder - Solar Pro 3 기반 지능적 쿼리 결합

Original Query + Feedbacks → 통합 쿼리 생성
"""

import os
import json
import re
import ast
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from openai import OpenAI
from dotenv import load_dotenv
from .detail_category import DETAIL_CAT_RULES, normalize_detail_category

# .env 파일 로드
load_dotenv()


# ==================== 구조화된 쿼리 모델 (향후 사용 가능) ====================
@dataclass
class RefinedQuery:
    """
    정제된 오리지널 쿼리 (향후 사용)

    의류 조합 시스템에 전달할 구조화된 정보
    """
    mood: List[str] = field(default_factory=list)          # 분위기: ["캐주얼", "편안한", "밝은"]
    time: Optional[str] = None                              # 시점: "봄", "여름", "주말", "평일"
    location: Optional[str] = None                          # 장소: "사무실", "데이트", "카페"
    requirements: List[str] = field(default_factory=list)  # 요구사항: ["상의는 밝은 색", "바지는 편한 핏"]
    requirements_en: List[str] = field(default_factory=list)  # 영어 요구사항 (FashionCLIP용): ["bright colored top", "comfortable fit pants"]
    constraints: List[str] = field(default_factory=list)   # 제한사항: ["검은색 제외", "타이트한 옷 제외"]
    target_detail_cats: List[str] = field(default_factory=list)  # 타겟 세부카테고리: ["Knitwear", "Sweatshirt"]
    avoid_detail_cats: List[str] = field(default_factory=list)  # 제외 세부카테고리: ["Knitwear"]
    location_confidence: float = 0.0                       # 장소 해석 신뢰도(0~1)
    resolved_styles: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    style_candidates: List[Dict[str, Any]] = field(default_factory=list)
    prefer_brightness: Optional[str] = None                 # 색상 밝기 선호: "light" or "dark"
    original_text: str = ""                                 # 원본 텍스트

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mood": self.mood,
            "time": self.time,
            "location": self.location,
            "requirements": self.requirements,
            "requirements_en": self.requirements_en,
            "constraints": self.constraints,
            "target_detail_cats": self.target_detail_cats,
            "avoid_detail_cats": self.avoid_detail_cats,
            "location_confidence": self.location_confidence,
            "resolved_styles": self.resolved_styles,
            "style_candidates": self.style_candidates,
            "prefer_brightness": self.prefer_brightness,
            "original_text": self.original_text,
        }


@dataclass
class StructuredOriginalQuery:
    """
    오리지널 쿼리에서 추출된 구조화된 정보 (LLM 분석 결과)

    FashionCLIP 임베딩 검색용 영어 쿼리 생성에 사용됨
    """
    time_context: List[str] = field(default_factory=list)    # ["tomorrow", "weekend"]
    color: List[str] = field(default_factory=list)           # ["black", "white", "gray"]
    size_fit: List[str] = field(default_factory=list)        # ["moderately loose", "relaxed fit"]
    season: List[str] = field(default_factory=list)          # ["early spring", "summer"]
    location: List[str] = field(default_factory=list)        # ["Hongdae club", "cafe"]
    mood: List[str] = field(default_factory=list)            # ["party/energetic", "casual"]
    user_constraints: List[str] = field(default_factory=list)  # 회피사항
    user_requirements: List[str] = field(default_factory=list) # 명시적 요구사항

    def to_embedding_query(self) -> str:
        """
        구조화된 정보를 FashionCLIP용 영어 검색 쿼리로 변환

        Returns:
            영어 검색 쿼리 문자열 (예: "relaxed fit beige knit for spring casual cafe look")
        """
        parts = []

        # 핏 정보 (가장 중요 - 의류 특성)
        if self.size_fit:
            parts.extend(self.size_fit[:2])

        # 색상 정보
        if self.color:
            parts.extend(self.color[:3])

        # 시즌 정보
        if self.season:
            parts.extend(self.season)

        # 무드 정보
        if self.mood:
            parts.extend(self.mood)

        # 장소 정보 (스타일 힌트)
        if self.location:
            parts.extend([f"{loc} style" for loc in self.location])

        # 명시적 요구사항
        if self.user_requirements:
            parts.extend(self.user_requirements)

        return " ".join(parts) if parts else "casual outfit"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StructuredOriginalQuery":
        """
        LLM 분석 결과 딕셔너리에서 StructuredOriginalQuery 생성

        Args:
            data: LLM이 반환한 구조화된 딕셔너리

        Returns:
            StructuredOriginalQuery 인스턴스
        """
        def extract_values(field_data):
            if isinstance(field_data, dict):
                return field_data.get("value", [])
            elif isinstance(field_data, list):
                return field_data
            return []

        return cls(
            time_context=extract_values(data.get("time_context", [])),
            color=extract_values(data.get("color", [])),
            size_fit=extract_values(data.get("size_fit", [])),
            season=extract_values(data.get("season", [])),
            location=extract_values(data.get("location", [])),
            mood=extract_values(data.get("mood", [])),
            user_constraints=extract_values(data.get("user_constraints", [])),
            user_requirements=extract_values(data.get("user_requirements", []))
        )


@dataclass
class CombinedQuery:
    """결합된 쿼리 결과"""
    combined_text: str
    reasoning: str
    original_query: str
    feedback_summary: str
    refined_query: Optional[RefinedQuery] = None  # 구조화된 정보


class QueryBuilder:
    """
    Original Query + Feedbacks → 통합 쿼리 생성

    Solar Pro 3 API 사용
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("UPSTAGE_API_KEY")
        self.base_url = "https://api.upstage.ai/v1"
        self.model = "solar-pro3"
        self.reasoning_effort = os.getenv("UPSTAGE_REASONING_EFFORT", "high")
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def build_combined_query(
        self,
        original_query: Dict[str, Any],
        feedbacks: List[str],
        feedback_scopes: Optional[List[str]] = None
    ) -> CombinedQuery:
        """
        Original Query + Feedbacks → 통합 쿼리 생성

        Args:
            original_query: {"initial_request": "...", "occasion": "...", ...}
            feedbacks: ["상의가 너무 어두워요", ...]
            feedback_scopes: ["TOP", "BOTTOM"] (옵션)

        Returns:
            CombinedQuery
        """
        # Original Query 텍스트 추출
        original_text = self._format_original_query(original_query)

        # Feedbacks 요약
        feedback_summary = self._summarize_feedbacks(feedbacks, feedback_scopes)

        # 피드백이 없으면 원본 반환
        if not feedbacks or not feedback_summary:
            return CombinedQuery(
                combined_text=original_text,
                reasoning="No feedbacks to combine",
                original_query=original_text,
                feedback_summary=""
            )

        # Solar Pro 3로 지능적 결합
        combined_result = self._combine_with_llm(
            original_text=original_text,
            feedbacks=feedbacks
        )

        # 결합된 쿼리를 구조화된 정보로 정제
        combined_text = combined_result["combined_query"]
        refined = self.refine_original_query(combined_text)

        return CombinedQuery(
            combined_text=combined_text,
            reasoning=combined_result.get("reasoning", ""),
            original_query=original_text,
            feedback_summary=feedback_summary,
            refined_query=refined
        )

    # ==================== 구조화된 쿼리 추출 ====================

    def refine_original_query(self, original_query: str) -> RefinedQuery:
        """
        오리지널 쿼리 정제

        사용자의 초기 요청을 분석하여 구조화된 정보 추출
        - 분위기 (mood)
        - 시점 (time)
        - 장소 (location)
        - 요구사항 (requirements)
        - 제한사항 (constraints)

        Args:
            original_query: "캐주얼한 봄 데이트룩 추천해줘"

        Returns:
            RefinedQuery
        """
        if not original_query or not original_query.strip():
            return RefinedQuery(original_text=original_query)

        # Solar Pro 3로 구조화된 정보 추출
        prompt = self._build_refine_prompt(original_query)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                reasoning_effort=self.reasoning_effort,
                stream=False,
            )
            content = getattr(response.choices[0].message, "content", "")
            if not content:
                raise ValueError("refine_original_query: empty content response")
            parsed = self._parse_json_from_model_output(content, "refine_original_query")
            refined = RefinedQuery(
                mood=parsed.get("mood", []),
                time=parsed.get("time"),
                location=parsed.get("location"),
                requirements=parsed.get("requirements", []),
                requirements_en=parsed.get("requirements_en", []),
                constraints=parsed.get("constraints", []),
                target_detail_cats=parsed.get("target_detail_cats", []),
                avoid_detail_cats=parsed.get("avoid_detail_cats", []),
                location_confidence=self._coerce_float(parsed.get("location_confidence"), 0.0),
                resolved_styles=self._coerce_resolved_styles(parsed.get("resolved_styles", {})),
                style_candidates=self._coerce_style_candidates(parsed.get("style_candidates", [])),
                prefer_brightness=parsed.get("prefer_brightness"),
                original_text=original_query
            )
            return self._apply_detail_cat_negation_rules(original_query, refined)

        except Exception as e:
            # Fallback: 빈 구조 반환
            print(f"[QueryBuilder] 쿼리 정제 실패, Fallback 사용: {e}")
            fallback = RefinedQuery(original_text=original_query)
            return self._apply_detail_cat_negation_rules(original_query, fallback)

    @staticmethod
    def _coerce_float(value: Any, default: float = 0.0, min_value: float = 0.0, max_value: float = 1.0) -> float:
        try:
            num = float(value)
        except (TypeError, ValueError):
            return default
        return max(min_value, min(max_value, num))

    @staticmethod
    def _coerce_style_label(style_value: Any) -> Optional[str]:
        if not isinstance(style_value, str):
            return None
        value = style_value.strip()
        return value if value else None

    @classmethod
    def _coerce_style_record(cls, record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not isinstance(record, dict):
            return None
        style = cls._coerce_style_label(record.get("style"))
        if not style:
            return None
        return {
            "style": style,
            "location_score": record.get("location_score", 0.0),
            "mood_score": record.get("mood_score", 0.0),
            "final_score": record.get("final_score", 0.0),
            "source": record.get("source", []),
            "reason": record.get("reason", ""),
        }

    def _coerce_style_candidates(self, candidates: Any) -> List[Dict[str, Any]]:
        if not isinstance(candidates, list):
            return []
        result: List[Dict[str, Any]] = []
        for candidate in candidates:
            record = self._coerce_style_record(candidate)
            if record:
                result.append(record)
        return result

    @staticmethod
    def _coerce_resolved_styles(raw: Any) -> Dict[str, Dict[str, Any]]:
        if not isinstance(raw, dict):
            return {}
        result: Dict[str, Dict[str, Any]] = {}
        if isinstance(primary := raw.get("primary"), dict):
            if primary.get("style"):
                result["primary"] = {
                    "style": primary.get("style"),
                    "location_score": primary.get("location_score", 0.0),
                    "mood_score": primary.get("mood_score", 0.0),
                    "final_score": primary.get("final_score", 0.0),
                    "source": primary.get("source", []),
                    "reason": primary.get("reason", ""),
                }
        if isinstance(secondary := raw.get("secondary"), dict):
            if secondary.get("style"):
                result["secondary"] = {
                    "style": secondary.get("style"),
                    "location_score": secondary.get("location_score", 0.0),
                    "mood_score": secondary.get("mood_score", 0.0),
                    "final_score": secondary.get("final_score", 0.0),
                    "source": secondary.get("source", []),
                    "reason": secondary.get("reason", ""),
                }
        return result

    def _apply_detail_cat_negation_rules(self, text: str, refined: RefinedQuery) -> RefinedQuery:
        """
        LLM 출력 검증/정규화:
        - 의미 재해석은 하지 않고 canonical 변환 + 충돌 정리만 수행
        """
        _ = text

        llm_target = {
            normalize_detail_category(cat)
            for cat in (refined.target_detail_cats or [])
            if normalize_detail_category(cat) in DETAIL_CAT_RULES
        }
        llm_avoid = {
            normalize_detail_category(cat)
            for cat in (refined.avoid_detail_cats or [])
            if normalize_detail_category(cat) in DETAIL_CAT_RULES
        }
        final_avoid = llm_avoid
        final_target = llm_target - final_avoid

        refined.avoid_detail_cats = sorted(final_avoid)
        refined.target_detail_cats = sorted(final_target)
        return refined

    def _build_refine_prompt(self, original_query: str) -> str:
        """쿼리 정제용 프롬프트"""
        return f"""
당신은 패션 추천 시스템의 쿼리 분석 전문가입니다.

## 역할
사용자의 의류 추천 요청을 분석하여 구조화된 정보를 추출하세요.
출력은 JSON만 반환하며, 값이 없으면 빈 값으로 표기하세요.

## 입력
사용자 요청: "{original_query}"

## 제약
- 15개 기준 스타일: ["고프코어","레트로","로맨틱","리조트","미니멀","스트릿","스포티","시크","시티보이","아웃도어","오피스","워크웨어","캐주얼","클래식","프레피"]
- 스타일 후보는 반드시 위 15개에 한정
- location/time/mood/요청 의도를 최대한 반영
- location 정보가 강하면 style 후보에 반영, 강도가 약하면 mood만 반영
- 15개에 매칭이 안 되면 style_candidates로 유사도 기반 후보를 채우고, style 후보는 비워도 됨
- 사용자 요청의 의류 카테고리 표현을 반드시 추출해 문맥에 맞는 표준 세부카테고리로 해석하고 target_detail_cats와 avoid_detail_cats에 반영하라.

## 추출 항목
1. **mood**: 분위기 키워드 (배열)
   - 예: ["캐주얼", "편안한", "밝은", "모던한"]

2. **time**: 시간적 맥락 (문자열 또는 null)
   - 예: "봄", "여름", "주말", "평일 오후"

3. **location**: 장소/상황 (문자열 또는 null)
   - 예: "사무실", "데이트", "카페", "야외 활동"

4. **location_confidence**: location 해석 신뢰도 (0~1 실수)
   - 명확한 지명/상황은 0.7~1.0
   - 약한 단서 또는 없음은 0~0.4

5. **requirements**: 명시적 요구사항 - 한국어 (배열)
   - 예: ["상의는 밝은 색", "편한 핏", "정장 느낌"]

6. **requirements_en**: 명시적 요구사항 - 영어 번역 (배열)
   - FashionCLIP 임베딩 검색용, 간결하게 번역
   - 예: ["bright colored top", "comfortable fit"]

7. **constraints**: 제한사항, 회피 사항 (배열)
   - 예: ["검은색 제외", "타이트한 옷 제외", "화려한 패턴 제외"]

8. **target_detail_cats**: 요청에서 언급된 의류 세부 카테고리 (배열)
   - 목록: 상의(Tee,Shirt,Sweatshirt,Knitwear), 하의(Denim,Chino,Trousers,Easy_pants,Work_pants,Short), 아우터(Jacket_Blouson,Coat,Cardigan,Jumper_Parka,Padding,Leather,Vest)

9. **avoid_detail_cats**: 제외할 의류 세부 카테고리 (배열)
   - 목록 동일

10. **prefer_brightness**: 색상 밝기 선호 (문자열 또는 null)
    - "light": 밝은색 선호
    - "dark": 어두운색 선호
    - null: 밝기 언급 없음

11. **resolved_styles**
    - 스타일을 확정한 결과: primary / secondary
    - 각 값은 style, location_score, mood_score, final_score, source, reason 포함
    - 매칭이 약하면 빈 객체로 둬도 됨

12. **style_candidates**
    - 15개 스타일과 유사도가 높은 후보 최대 3개
    - location/mood 매핑이 애매한 경우에만 채워 사용

## 출력 형식
반드시 JSON으로만 반환:
{{
  "mood": ["키워드1", "키워드2"],
  "time": "시점" 또는 null,
  "location": "장소" 또는 null,
  "location_confidence": 0.0,
  "requirements": ["요구사항1"],
  "requirements_en": ["requirement1 in English"],
  "constraints": ["제한사항1"],
  "target_detail_cats": ["세부카테고리1"],
  "avoid_detail_cats": ["제외카테고리1"],
  "resolved_styles": {{
    "primary": {{
      "style": "캐주얼",
      "location_score": 0.0,
      "mood_score": 0.0,
      "final_score": 0.0,
      "source": ["location", "mood"],
      "reason": ""
    }},
    "secondary": {{
      "style": "",
      "location_score": 0.0,
      "mood_score": 0.0,
      "final_score": 0.0,
      "source": [],
      "reason": ""
    }}
  }},
  "style_candidates": [
    {{
      "style": "",
      "location_score": 0.0,
      "mood_score": 0.0,
      "final_score": 0.0,
      "source": [],
      "reason": ""
    }}
  ],
  "prefer_brightness": "light" 또는 "dark" 또는 null
}}

## 예시
입력: "캐주얼한 봄 데이트룩 추천해줘"
출력:
{{
  "mood": ["캐주얼", "데이트 느낌"],
  "time": "봄",
  "location": "데이트",
  "location_confidence": 0.84,
  "requirements": ["캐주얼한 봄 데이트룩"],
  "requirements_en": ["casual spring date outfit"],
  "constraints": [],
  "target_detail_cats": [],
  "avoid_detail_cats": [],
  "resolved_styles": {{
    "primary": {{
      "style": "로맨틱",
      "location_score": 0.75,
      "mood_score": 0.62,
      "final_score": 0.74,
      "source": ["location", "mood"],
      "reason": "데이트 맥락에서 로맨틱성 우세"
    }},
    "secondary": {{
      "style": "캐주얼",
      "location_score": 0.0,
      "mood_score": 0.46,
      "final_score": 0.42,
      "source": ["mood"],
      "reason": "전체 무드가 편안한 편"
    }}
  }},
  "style_candidates": [],
  "prefer_brightness": null
}}

입력: "사무실에서 입을 편한 옷 추천. 검은색은 싫어"
출력:
{{
  "mood": ["편안한"],
  "time": null,
  "location": "사무실",
  "location_confidence": 0.88,
  "requirements": ["편한 핏의 사무실 옷"],
  "requirements_en": ["comfortable office wear"],
  "constraints": ["검은색 제외"],
  "target_detail_cats": [],
  "avoid_detail_cats": [],
  "resolved_styles": {{
    "primary": {{
      "style": "오피스",
      "location_score": 0.9,
      "mood_score": 0.34,
      "final_score": 0.82,
      "source": ["location"],
      "reason": "사무실이라는 장소 신호가 오피스로 강하게 수렴"
    }},
    "secondary": {{
      "style": "",
      "location_score": 0.0,
      "mood_score": 0.0,
      "final_score": 0.0,
      "source": [],
      "reason": ""
    }}
  }},
  "style_candidates": [],
  "prefer_brightness": null
}}

입력: "밝은 니트로 바꿔줘"
출력:
{{
  "mood": ["밝은"],
  "time": null,
  "location": null,
  "location_confidence": 0.0,
  "requirements": ["밝은 색 니트로 변경"],
  "requirements_en": ["bright colored knit sweater"],
  "constraints": [],
  "target_detail_cats": ["Knitwear"],
  "avoid_detail_cats": [],
  "resolved_styles": {{
    "primary": {{
      "style": "",
      "location_score": 0.0,
      "mood_score": 0.0,
      "final_score": 0.0,
      "source": [],
      "reason": ""
    }},
    "secondary": {{
      "style": "",
      "location_score": 0.0,
      "mood_score": 0.0,
      "final_score": 0.0,
      "source": [],
      "reason": ""
    }}
  }},
  "style_candidates": [
    {{
      "style": "캐주얼",
      "location_score": 0.2,
      "mood_score": 0.48,
      "final_score": 0.32,
      "source": ["mood"],
      "reason": "무드 추출값의 직접 매칭이 약해 후보화"
    }}
  ],
  "prefer_brightness": "light"
}}

입력: "어두운 색으로 바꿔줘"
출력:
{{
  "mood": ["어두운"],
  "time": null,
  "location": null,
  "location_confidence": 0.0,
  "requirements": ["어두운 색으로 변경"],
  "requirements_en": ["dark colored clothing"],
  "constraints": [],
  "target_detail_cats": [],
  "avoid_detail_cats": [],
  "resolved_styles": {{
    "primary": {{
      "style": "",
      "location_score": 0.0,
      "mood_score": 0.0,
      "final_score": 0.0,
      "source": [],
      "reason": ""
    }},
    "secondary": {{
      "style": "",
      "location_score": 0.0,
      "mood_score": 0.0,
      "final_score": 0.0,
      "source": [],
      "reason": ""
    }}
  }},
  "style_candidates": [],
  "prefer_brightness": "dark"
}}

입력: "니트 말고 다른걸로 바꿔줘"
출력:
{{
  "mood": [],
  "time": null,
  "location": null,
  "location_confidence": 0.0,
  "requirements": ["상의 변경"],
  "requirements_en": ["different top item"],
  "constraints": ["니트 제외"],
  "target_detail_cats": [],
  "avoid_detail_cats": ["Knitwear"],
  "resolved_styles": {{
    "primary": {{
      "style": "",
      "location_score": 0.0,
      "mood_score": 0.0,
      "final_score": 0.0,
      "source": [],
      "reason": ""
    }},
    "secondary": {{
      "style": "",
      "location_score": 0.0,
      "mood_score": 0.0,
      "final_score": 0.0,
      "source": [],
      "reason": ""
    }}
  }},
  "style_candidates": [],
  "prefer_brightness": null
}}
"""

    def _format_original_query(self, original_query: Dict[str, Any]) -> str:
        """Original Query → 텍스트 변환"""
        # 우선순위: initial_request > 개별 필드 조합
        if "initial_request" in original_query and original_query["initial_request"]:
            return original_query["initial_request"]

        # 개별 필드 조합
        parts = []
        if "occasion" in original_query:
            parts.append(original_query["occasion"])
        if "weather" in original_query:
            parts.append(original_query["weather"])
        if "style_preference" in original_query:
            parts.append(f"{original_query['style_preference']} 스타일")

        return ", ".join(parts) if parts else "코디 추천"

    def _summarize_feedbacks(
        self,
        feedbacks: List[str],
        feedback_scopes: Optional[List[str]] = None
    ) -> str:
        """Feedbacks → 요약 텍스트"""
        if not feedbacks:
            return ""
        return ", ".join(feedbacks)

    def _combine_with_llm(
        self,
        original_text: str,
        feedbacks: List[str]
    ) -> Dict[str, str]:
        """
        Solar Pro 3 API로 지능적 결합

        Returns:
            {"combined_query": "...", "reasoning": "..."}
        """
        # 프롬프트 생성
        prompt = self._build_prompt(original_text, feedbacks)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                reasoning_effort=self.reasoning_effort,
                stream=False,
            )
            content = getattr(response.choices[0].message, "content", "")
            if not content:
                raise ValueError("combine_with_llm: empty content response")
            parsed = self._parse_json_from_model_output(content, "combine_with_llm")
            return parsed

        except Exception as e:
            # Fallback: 단순 결합
            print(f"[QueryBuilder] Solar Pro 3 API 실패, Fallback 사용: {e}")
            return {
                "combined_query": f"{original_text}. {' '.join(feedbacks)}",
                "reasoning": "API failure - simple concatenation"
            }

    def _parse_json_from_model_output(self, content: str, source: str) -> Dict[str, Any]:
        """
        LLM 응답에서 JSON 객체만 안전하게 추출해 파싱.
        - 코드블록 제거
        - JSON 균형 추출
        - 후행 쉼표 정리
        - 따옴표 누락 key 보정(기본 키 집합 한정)
        - 실패 시 예외를 그대로 전달
        """
        if not isinstance(content, str):
            raise ValueError(f"{source}: model response content is not str")

        raw = content
        content = self._strip_markdown_wrappers(raw)
        content = self._normalize_json_text(content)

        candidates = self._extract_json_candidates(content)
        if not candidates:
            candidates = self._extract_json_candidates(raw)
        if not candidates:
            outer = self._extract_outer_json_block(raw)
            if outer:
                candidates.append(outer)

        parse_trace = []
        for candidate in candidates:
            candidate = self._clean_json_candidate(candidate)
            if not candidate:
                continue
            parse_errors = []
            try:
                parsed = json.loads(candidate)
                coerced = self._coerce_json_payload(parsed)
                if coerced is not None:
                    return coerced
                parse_errors.append("json: parsed_non_dict")
            except Exception as e:
                parse_errors.append(f"json: {e}")

            try:
                parsed = ast.literal_eval(candidate)
                coerced = self._coerce_json_payload(parsed)
                if coerced is not None:
                    return coerced
                parse_errors.append("ast: parsed_non_dict")
            except Exception as e:
                parse_errors.append(f"ast: {e}")

            repaired = self._repair_json(candidate)
            try:
                parsed = json.loads(repaired)
                coerced = self._coerce_json_payload(parsed)
                if coerced is not None:
                    return coerced
                parse_errors.append("json_repair: parsed_non_dict")
            except Exception as e:
                parse_errors.append(f"json_repair: {e}")

            try:
                parsed = ast.literal_eval(repaired)
                coerced = self._coerce_json_payload(parsed)
                if coerced is not None:
                    return coerced
                parse_errors.append("ast_repair: parsed_non_dict")
            except Exception as e:
                parse_errors.append(f"ast_repair: {e}")
            parse_trace.append(f"candidate={candidate[:200]} | errors={parse_errors}")

        # 파싱 실패 시 소스별 fallback(비정상 응답/노이즈)로 복구
        if source == "refine_original_query":
            fallback = self._build_refine_fallback(raw)
            if fallback:
                return fallback
        if source == "combine_with_llm":
            fallback = self._build_combine_fallback(raw)
            if fallback:
                return fallback

        # 디버깅용: 에러 발생 전에 원본 텍스트의 일부를 표시
        raise ValueError(
            f"{source}: JSON parse failed. raw={raw[:400]}, candidate_count={len(candidates)}, candidate_errors={parse_trace}"
        )

    @staticmethod
    def _coerce_json_payload(payload: Any) -> Optional[Dict[str, Any]]:
        """
        파서 결과를 dict로 정규화
        - dict: 그대로 사용
        - list/tuple: 첫 번째 dict 사용
        - 그 외: None
        """
        if isinstance(payload, dict):
            return payload
        if isinstance(payload, (list, tuple)):
            for item in payload:
                if isinstance(item, dict):
                    return item
        return None

    @staticmethod
    def _extract_outer_json_block(content: str) -> Optional[str]:
        if not isinstance(content, str):
            return None
        start = content.find("{")
        if start < 0:
            return None
        return QueryBuilder._extract_balanced_json(content[start:])

    @staticmethod
    def _extract_json_field(content: str, field: str) -> Optional[str]:
        if not isinstance(content, str):
            return None

        # key 패턴: "field", 'field', field
        marker_pattern = re.compile(
            rf'(?<!\w)(?:["\']?{re.escape(field)}["\']?)\s*:'
        )
        m = marker_pattern.search(content)
        if not m:
            return None
        idx = m.start()
        if idx < 0:
            return None
        colon = m.end() - 1
        if colon < 0:
            return None

        rest = content[colon + 1 :].lstrip()
        if not rest:
            return None

        # 문자열 값
        if rest[0] in {'"', "'"}:
            quote = rest[0]
            escape = False
            i = 1
            buff: List[str] = []
            while i < len(rest):
                ch = rest[i]
                if escape:
                    buff.append(ch)
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == quote:
                    return "".join(buff).strip()
                else:
                    buff.append(ch)
                i += 1
            return None

        # 객체/배열 값
        if rest[0] == "{":
            segment = QueryBuilder._extract_balanced_json_segment(rest, "{", "}")
            if segment:
                return segment
        if rest[0] == "[":
            segment = QueryBuilder._extract_balanced_json_segment(rest, "[", "]")
            if segment:
                return segment

        # 단일 스칼라 값
        match = re.match(r"[^,\]\}]+", rest, re.DOTALL)
        if not match:
            return None
        return match.group(0).strip().strip('"').strip("'")

    def _build_combine_fallback(self, raw_content: str) -> Optional[Dict[str, str]]:
        combined_query = self._extract_json_field(raw_content, "combined_query")
        reasoning = self._extract_json_field(raw_content, "reasoning")

        text = raw_content.strip()
        if not combined_query:
            combined_query = self._first_non_empty_line(text)
        if not combined_query:
            return None

        return {
            "combined_query": combined_query.strip(),
            "reasoning": reasoning or "fallback_parsing",
        }

    def _build_refine_fallback(self, raw_content: str) -> Dict[str, Any]:
        text = self._clean_text_for_llm(raw_content)
        requirements: List[str] = []
        if text:
            requirements.append(text[:120].strip())

        return {
            "mood": [],
            "time": None,
            "location": None,
            "requirements": requirements,
            "requirements_en": [req for req in requirements if req],
            "constraints": [],
            "target_detail_cats": [],
            "avoid_detail_cats": [],
            "location_confidence": 0.0,
            "resolved_styles": {},
            "style_candidates": [],
            "prefer_brightness": None,
        }

    @staticmethod
    def _clean_text_for_llm(content: str) -> str:
        if not isinstance(content, str):
            return ""
        text = content.strip()
        text = text.replace("```", "").replace("\n", " ")
        # 코드블록/설명형 텍스트 제거
        text = re.sub(r"\[[^\]]*?\]", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _first_non_empty_line(content: str) -> str:
        for line in content.splitlines():
            s = line.strip()
            if s:
                return s
        return ""

    @staticmethod
    def _get_detail_cat_rules() -> Dict[str, List[str]]:
        return DETAIL_CAT_RULES

    @staticmethod
    def _strip_markdown_wrappers(content: str) -> str:
        if not isinstance(content, str):
            return ""

        text = content.strip()
        block_pattern = re.compile(r"```(?:json|JSON)?\s*?\n(.*?)(?:\n```|$)", re.DOTALL)
        matches = block_pattern.findall(text)
        if matches:
            for match in matches:
                candidate = match.strip()
                if candidate:
                    return candidate

        # 개행 없이 닫힌 블록이 오는 케이스: ```json ... ```
        block_pattern = re.compile(r"```(?:json|JSON)?\s*(.*?)(?:\s*```|$)", re.DOTALL)
        matches = block_pattern.findall(text)
        if matches:
            for match in matches:
                candidate = match.strip()
                if candidate:
                    return candidate

        # 시작 backtick이 붙은 응답: ```json { ... } 또는 ``` { ... }
        if text.startswith("```"):
            text = text[3:].lstrip()
            if text.lower().startswith("json"):
                text = text[4:].lstrip()
            text = text.lstrip()
            text = re.sub(r"\n*```\s*$", "", text).strip()
            return text

        # 닫는 fence만 누락된 케이스: 마지막에서부터 자르기
        text = text.rstrip()
        if text.startswith("```json"):
            text = text[7:].strip()
        if text.startswith("```"):
            text = text[3:].strip()
        text = re.sub(r"```$", "", text).strip()
        return text

    @staticmethod
    def _normalize_json_text(content: str) -> str:
        """
        JSON 파싱을 위한 입력 정규화.
        - 앞뒤 불필요 백틱/코드블록 토큰 제거
        - 잘못 붙은 따옴표 토큰 정리
        """
        if not isinstance(content, str):
            return ""

        text = content.strip()
        text = text.replace("\r\n", "\n")
        text = text.replace("`", "")
        return text

    @staticmethod
    def _extract_balanced_json(content: str) -> Optional[str]:
        start = content.find("{")
        if start < 0:
            return None
        depth = 0
        in_string = False
        quote_char = ""
        escape = False
        for i in range(start, len(content)):
            ch = content[i]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == quote_char:
                    in_string = False
                continue

            if ch in ("'", '"'):
                in_string = True
                quote_char = ch
                continue

            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return content[start:i + 1]
        return None

    @staticmethod
    def _extract_balanced_json_segment(content: str, start_char: str, end_char: str) -> Optional[str]:
        if not isinstance(content, str) or not start_char or not end_char:
            return None
        start = content.find(start_char)
        if start < 0:
            return None

        depth = 0
        in_string = False
        quote_char = ""
        escape = False
        for i in range(start, len(content)):
            ch = content[i]
            if in_string:
                if escape:
                    escape = False
                    continue
                if ch == "\\":
                    escape = True
                    continue
                if ch == quote_char:
                    in_string = False
                continue

            if ch in ("'", '"'):
                in_string = True
                quote_char = ch
                continue

            if ch == start_char:
                depth += 1
            elif ch == end_char:
                depth -= 1
                if depth == 0:
                    return content[start:i + 1]
        return None

    def _extract_json_candidates(self, content: str) -> List[str]:
        """JSON 파싱 후보 문자열 리스트 생성(중복 제거, 우선순위 유지)."""
        candidates: List[str] = []
        seen = set()

        if not isinstance(content, str):
            return []

        # 코드블록에서 추출
        block_pattern = re.compile(r"```(?:json|JSON)?\s*?\n(.*?)(?:\n```|$)", re.DOTALL)
        for block in block_pattern.findall(content):
            block = block.strip()
            if not block:
                continue
            block = self._clean_json_candidate(block)
            if block not in seen:
                candidates.append(block)
                seen.add(block)

        # 균형 JSON 블록 추출 (코드블록 안/밖 모두 대응)
        for block in self._extract_balanced_json_candidates(content):
            block = block.strip()
            if block and block not in seen:
                candidates.append(block)
                seen.add(block)

        # 최종 정리 문자열
        if content and content not in seen:
            candidates.append(content)
            seen.add(content)

        # 공백/개행 정리 후보
        normalized = content.strip()
        if normalized and normalized != content and normalized not in seen:
            candidates.append(normalized)
            seen.add(normalized)

        return [c for c in candidates if c.strip()]

    @staticmethod
    def _extract_balanced_json_candidates(content: str) -> List[str]:
        """문자열 내 여러 JSON 객체를 균형 기반으로 추출."""
        if not isinstance(content, str):
            return []

        blocks: List[str] = []
        idx = 0
        n = len(content)
        while idx < n:
            if content[idx] != "{":
                idx += 1
                continue
            block = QueryBuilder._extract_balanced_json(content[idx:])
            if not block:
                idx += 1
                continue
            if block not in blocks:
                blocks.append(block)
            idx += content[idx:].find(block) + len(block)
            if len(block) == 0:
                idx += 1

        return blocks

    @staticmethod
    def _clean_json_candidate(content: str) -> str:
        text = content.strip()
        # trailing comma 제거
        text = re.sub(r",\s*([}\]])", r"\1", text)
        # 라인 단위 주석 제거 (모델이 실수로 남기는 경우)
        cleaned_lines = []
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("//") or stripped.startswith("#"):
                continue
            cleaned_lines.append(line)
        text = "\n".join(cleaned_lines).strip()
        return text

    @staticmethod
    def _repair_json(content: str) -> str:
        text = content
        # key가 따옴표 없이 나오는 케이스 보정 (예: requirements: [...])
        text = re.sub(
            r'([,{]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*:)',
            r'\1"\2"\3',
            text
        )
        # single-quoted key 처리
        text = re.sub(
            r'([,{]\s*)\'([A-Za-z_][A-Za-z0-9_]*)\'(\s*:)',
            r'\1"\2"\3',
            text
        )
        # single-quoted 문자열 값 처리
        text = re.sub(
            r':\s*\'([^\'\\]*(?:\\.[^\'\\]*)*)\'',
            lambda m: f': "{m.group(1)}"',
            text
        )
        text = re.sub(r",\s*([}\]])", r"\1", text)
        return text

    def _build_prompt(self, original_text: str, feedbacks: List[str]) -> str:
        """Solar Pro 3 프롬프트 생성 (최적화된 버전)"""
        feedback_text = "\n".join([f"  - {fb}" for fb in feedbacks])

        return f"""당신은 패션 추천 시스템의 쿼리 최적화 전문가입니다.

<task>
사용자의 초기 요청과 피드백을 분석하여 최종 의도를 반영한 자연스러운 쿼리를 생성하세요.
</task>

<input>
  <initial_request>{original_text}</initial_request>
  <feedback_list>
{feedback_text}
  </feedback_list>
</input>

<rules>
  <priority_rules>
    1. 피드백이 초기 요청과 모순되면 **최신 피드백을 우선**
    2. 부정 표현("~하지 말아줘", "~은 싫어")은 긍정 표현으로 전환
    3. 의도가 완전히 바뀐 경우 피드백만으로 쿼리 재구성
  </priority_rules>

  <preservation_rules>
    1. 초기 요청의 occasion(상황), season(계절), gender는 **피드백에서 명시적으로 변경하지 않는 한 유지**
    2. style, formality는 피드백에 따라 조정 가능
  </preservation_rules>

  <expression_rules>
    1. 자연스러운 한국어 문장으로 표현 (단순 키워드 나열 금지)
    2. 불필요한 수식어("너무", "좀 더", "조금") 제거
    3. 최종 쿼리 길이: 15-50자 권장
  </expression_rules>
</rules>

<edge_cases>
  - 피드백이 비어있음 → 초기 요청 그대로 반환
  - 피드백이 모순됨 (예: "밝게" vs "어둡게") → 가장 최신 피드백 우선
  - 초기 요청이 불명확함 → 피드백에서 명확한 의도 추출
  - 의도가 완전히 바뀜 (예: "데이트룩" → "운동복") → 피드백 중심으로 재구성
</edge_cases>

<output_format>
반드시 아래 JSON 형식으로만 응답:
{{
  "combined_query": "최종 통합 쿼리 (15-50자)",
  "reasoning": "적용한 규칙과 판단 근거 (예: priority_rule_1, preservation_rule_1 적용)"
}}
</output_format>

<examples>
  <example>
    <input>
      <initial_request>캐주얼한 봄 데이트룩</initial_request>
      <feedback_list>
        - 상의가 너무 어두워요
        - 좀 더 밝은 색으로
      </feedback_list>
    </input>
    <output>
{{
  "combined_query": "밝은 색상의 캐주얼한 봄 데이트룩",
  "reasoning": "preservation_rule_1 (봄, 데이트 유지) + priority_rule_2 (부정 → 긍정) + expression_rule_2 (수식어 제거)"
}}
    </output>
  </example>

  <example>
    <input>
      <initial_request>정장 스타일 출근룩</initial_request>
      <feedback_list>
        - 너무 딱딱해요
        - 캐주얼하게 바꿔주세요
      </feedback_list>
    </input>
    <output>
{{
  "combined_query": "세미 캐주얼한 출근룩",
  "reasoning": "preservation_rule_1 (출근 상황 유지) + priority_rule_1 (피드백 우선: 정장 → 캐주얼) + expression_rule_1 (자연스러운 표현)"
}}
    </output>
  </example>

  <example>
    <input>
      <initial_request>여름 휴가룩</initial_request>
      <feedback_list></feedback_list>
    </input>
    <output>
{{
  "combined_query": "여름 휴가룩",
  "reasoning": "edge_case: 피드백 비어있음 → 초기 요청 그대로 반환"
}}
    </output>
  </example>

  <example>
    <input>
      <initial_request>가을 데이트 코디</initial_request>
      <feedback_list>
        - 헬스장 갈 때 입을 옷으로 바꿔주세요
      </feedback_list>
    </input>
    <output>
{{
  "combined_query": "가을 운동복 스타일",
  "reasoning": "edge_case: 의도 완전 변경 → priority_rule_3 (피드백 중심 재구성) + preservation_rule_1 (계절 유지)"
}}
    </output>
  </example>
</examples>

<quality_check>
  - JSON 형식 준수
  - combined_query 길이: 15-50자
  - 부정 표현 -> 긍정 전환
  - 불필요한 수식어 제거
  - reasoning에 적용 규칙 명시
</quality_check>

위 규칙과 예시를 참고하여 JSON만 반환하세요."""


# ==================== 테스트 코드 ====================
if __name__ == "__main__":
    builder = QueryBuilder()

    # Test Case 1: 색상 변경
    print("=" * 60)
    print("Test Case 1: 색상 변경 피드백")
    print("=" * 60)
    result1 = builder.build_combined_query(
        original_query={"initial_request": "캐주얼한 봄 데이트룩"},
        feedbacks=["상의가 너무 어두워요", "좀 더 밝은 색으로 바꿔주세요"]
    )
    print(f"Original: {result1.original_query}")
    print(f"Feedbacks: {result1.feedback_summary}")
    print(f"Combined: {result1.combined_text}")
    print(f"Reasoning: {result1.reasoning}\n")

    # Test Case 2: 스타일 변경
    print("=" * 60)
    print("Test Case 2: 스타일 변경 피드백")
    print("=" * 60)
    result2 = builder.build_combined_query(
        original_query={"initial_request": "정장 스타일"},
        feedbacks=["너무 딱딱해요", "좀 더 캐주얼하게"]
    )
    print(f"Original: {result2.original_query}")
    print(f"Feedbacks: {result2.feedback_summary}")
    print(f"Combined: {result2.combined_text}")
    print(f"Reasoning: {result2.reasoning}\n")

    # Test Case 3: 피드백 없음
    print("=" * 60)
    print("Test Case 3: 피드백 없음")
    print("=" * 60)
    result3 = builder.build_combined_query(
        original_query={"initial_request": "여름 휴가룩"},
        feedbacks=[]
    )
    print(f"Original: {result3.original_query}")
    print(f"Feedbacks: {result3.feedback_summary}")
    print(f"Combined: {result3.combined_text}")
    print(f"Reasoning: {result3.reasoning}\n")

    # Test Case 4: 개별 필드 조합
    print("=" * 60)
    print("Test Case 4: 개별 필드 조합")
    print("=" * 60)
    result4 = builder.build_combined_query(
        original_query={
            "occasion": "데이트",
            "weather": "봄",
            "style_preference": "캐주얼"
        },
        feedbacks=["바지를 청바지로 바꿔주세요"]
    )
    print(f"Original: {result4.original_query}")
    print(f"Feedbacks: {result4.feedback_summary}")
    print(f"Combined: {result4.combined_text}")
    print(f"Reasoning: {result4.reasoning}\n")

    # Test Case 5: 의도 완전 변경
    print("=" * 60)
    print("Test Case 5: 의도 완전 변경")
    print("=" * 60)
    result5 = builder.build_combined_query(
        original_query={"initial_request": "가을 데이트 코디"},
        feedbacks=["헬스장 갈 때 입을 옷으로 바꿔주세요"]
    )
    print(f"Original: {result5.original_query}")
    print(f"Feedbacks: {result5.feedback_summary}")
    print(f"Combined: {result5.combined_text}")
    print(f"Reasoning: {result5.reasoning}\n")

    print("=" * 60)
    print("All tests completed!")
    print("=" * 60)

    # ==================== 구조화된 쿼리 확인 ====================
    print("\n" + "=" * 60)
    print("구조화된 쿼리 정보 확인 (Test 1)")
    print("=" * 60)
    if result1.refined_query:
        print(f"Mood: {result1.refined_query.mood}")
        print(f"Time: {result1.refined_query.time}")
        print(f"Location: {result1.refined_query.location}")
        print(f"Requirements: {result1.refined_query.requirements}")
        print(f"Constraints: {result1.refined_query.constraints}")

    print("\n" + "=" * 60)
    print("추가 구조화 테스트")
    print("=" * 60)

    refined_test = builder.refine_original_query("사무실에서 입을 편한 옷 추천. 검은색은 싫어")
    print(f"\n[Test] 사무실 편한 옷 (제한사항)")
    print(f"Original: {refined_test.original_text}")
    print(f"Mood: {refined_test.mood}")
    print(f"Time: {refined_test.time}")
    print(f"Location: {refined_test.location}")
    print(f"Requirements: {refined_test.requirements}")
    print(f"Constraints: {refined_test.constraints}")
