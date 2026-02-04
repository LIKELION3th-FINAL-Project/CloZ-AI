"""
Query Builder - Solar Pro 3 기반 지능적 쿼리 결합

Original Query + Feedbacks → 통합 쿼리 생성
"""

import os
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import requests
from dotenv import load_dotenv

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
    original_text: str = ""                                 # 원본 텍스트

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mood": self.mood,
            "time": self.time,
            "location": self.location,
            "requirements": self.requirements,
            "requirements_en": self.requirements_en,
            "constraints": self.constraints,
            "original_text": self.original_text,
        }


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
        self.model = "solar-pro"

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
        # 1. Original Query 텍스트 추출
        original_text = self._format_original_query(original_query)

        # 2. Feedbacks 요약
        feedback_summary = self._summarize_feedbacks(feedbacks, feedback_scopes)

        # 3. 피드백이 없으면 원본 반환
        if not feedbacks or not feedback_summary:
            return CombinedQuery(
                combined_text=original_text,
                reasoning="No feedbacks to combine",
                original_query=original_text,
                feedback_summary=""
            )

        # 4. Solar Pro 3로 지능적 결합
        combined_result = self._combine_with_llm(
            original_text=original_text,
            feedbacks=feedbacks
        )

        # 5. 결합된 쿼리를 구조화된 정보로 정제
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

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 300
        }

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=10
            )
            response.raise_for_status()

            result = response.json()
            content = result["choices"][0]["message"]["content"]

            # JSON 추출 (코드 블록 및 추가 텍스트 제거)
            content = content.strip()

            # ```json ... ``` 코드 블록 제거
            if content.startswith("```"):
                lines = content.split("\n")
                # json 시작과 ``` 끝 사이의 내용만 추출
                start_idx = 1  # ```json 다음 줄부터
                end_idx = len(lines)
                for i in range(1, len(lines)):
                    if lines[i].strip().startswith("```"):
                        end_idx = i
                        break
                content = "\n".join(lines[start_idx:end_idx])

            # JSON 부분만 추출 (### 등 추가 설명 제거)
            json_lines = []
            in_json = False
            for line in content.split("\n"):
                if line.strip().startswith("{"):
                    in_json = True
                if in_json:
                    json_lines.append(line)
                if line.strip().startswith("}"):
                    break

            content = "\n".join(json_lines)

            # JSON 파싱
            parsed = json.loads(content)

            return RefinedQuery(
                mood=parsed.get("mood", []),
                time=parsed.get("time"),
                location=parsed.get("location"),
                requirements=parsed.get("requirements", []),
                requirements_en=parsed.get("requirements_en", []),
                constraints=parsed.get("constraints", []),
                original_text=original_query
            )

        except Exception as e:
            # Fallback: 빈 구조 반환
            print(f"[QueryBuilder] 쿼리 정제 실패, Fallback 사용: {e}")
            return RefinedQuery(original_text=original_query)

    def _build_refine_prompt(self, original_query: str) -> str:
        """쿼리 정제용 프롬프트"""
        return f"""
당신은 패션 추천 시스템의 쿼리 분석 전문가입니다.

## 역할
사용자의 의류 추천 요청을 분석하여 구조화된 정보를 추출하세요.

## 입력
사용자 요청: "{original_query}"

## 추출 항목
- **mood**: 분위기, 스타일 키워드 (배열)
   - 예: ["캐주얼", "편안한", "밝은", "모던한"]

- **time**: 시간적 맥락 (문자열 또는 null)
   - 예: "봄", "여름", "주말", "평일 오후"

- **location**: 장소, 상황 (문자열 또는 null)
   - 예: "사무실", "데이트", "카페", "야외 활동"

- **requirements**: 명시적 요구사항 - 한국어 (배열)
   - 예: ["상의는 밝은 색", "편한 핏", "정장 느낌"]

- **requirements_en**: 명시적 요구사항 - 영어 번역 (배열)
   - FashionCLIP 임베딩 검색용 영어 번역
   - 의류 검색에 적합한 간결한 영어 표현 사용
   - 예: ["bright colored top", "comfortable fit", "formal style"]

- **constraints**: 제한사항, 회피 사항 (배열)
   - 예: ["검은색 제외", "타이트한 옷 제외", "화려한 패턴 제외"]

## 출력 형식
반드시 JSON으로만 반환:
{{
  "mood": ["키워드1", "키워드2"],
  "time": "시점" 또는 null,
  "location": "장소" 또는 null,
  "requirements": ["요구사항1", "요구사항2"],
  "requirements_en": ["requirement1 in English", "requirement2 in English"],
  "constraints": ["제한사항1"]
}}

## 예시
입력: "캐주얼한 봄 데이트룩 추천해줘"
출력:
{{
  "mood": ["캐주얼", "데이트 느낌"],
  "time": "봄",
  "location": "데이트",
  "requirements": ["캐주얼한 봄 데이트룩"],
  "requirements_en": ["casual spring date outfit"],
  "constraints": []
}}

입력: "사무실에서 입을 편한 옷 추천. 검은색은 싫어"
출력:
{{
  "mood": ["편안한", "오피스"],
  "time": null,
  "location": "사무실",
  "requirements": ["편한 핏의 사무실 옷"],
  "requirements_en": ["comfortable office wear"],
  "constraints": ["검은색 제외"]
}}

입력: "밝은 색 상의로 바꿔줘"
출력:
{{
  "mood": ["밝은"],
  "time": null,
  "location": null,
  "requirements": ["밝은 색 상의"],
  "requirements_en": ["bright colored top"],
  "constraints": []
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

        # Solar Pro 3 API 호출
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 250
        }

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=10
            )
            response.raise_for_status()

            result = response.json()
            content = result["choices"][0]["message"]["content"]

            # JSON 추출 (코드 블록 및 추가 텍스트 제거)
            content = content.strip()

            # ```json ... ``` 코드 블록 제거
            if content.startswith("```"):
                lines = content.split("\n")
                # json 시작과 ``` 끝 사이의 내용만 추출
                start_idx = 1  # ```json 다음 줄부터
                end_idx = len(lines)
                for i in range(1, len(lines)):
                    if lines[i].strip().startswith("```"):
                        end_idx = i
                        break
                content = "\n".join(lines[start_idx:end_idx])

            # JSON 부분만 추출 (### 등 추가 설명 제거)
            json_lines = []
            in_json = False
            for line in content.split("\n"):
                if line.strip().startswith("{"):
                    in_json = True
                if in_json:
                    json_lines.append(line)
                if line.strip().startswith("}"):
                    break

            content = "\n".join(json_lines)

            # JSON 파싱
            parsed = json.loads(content)
            return parsed

        except Exception as e:
            # Fallback: 단순 결합
            print(f"[QueryBuilder] Solar Pro 3 API 실패, Fallback 사용: {e}")
            return {
                "combined_query": f"{original_text}. {' '.join(feedbacks)}",
                "reasoning": "API failure - simple concatenation"
            }

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
    - 피드백이 초기 요청과 모순되면 **최신 피드백을 우선**
    - 부정 표현("~하지 말아줘", "~은 싫어")은 긍정 표현으로 전환
    - 의도가 완전히 바뀐 경우 피드백만으로 쿼리 재구성
  </priority_rules>

  <preservation_rules>
    - 초기 요청의 occasion(상황), season(계절), gender는 **피드백에서 명시적으로 변경하지 않는 한 유지**
    - style, formality는 피드백에 따라 조정 가능
  </preservation_rules>

  <expression_rules>
    - 자연스러운 한국어 문장으로 표현 (단순 키워드 나열 금지)
    - 불필요한 수식어("너무", "좀 더", "조금") 제거
    - 최종 쿼리 길이: 15-50자 권장
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
  - 부정 표현 → 긍정 전환
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
