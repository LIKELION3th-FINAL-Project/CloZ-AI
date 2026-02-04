import json
from typing import List, Dict, Any, Optional
from ..llm import LLMFactory, BaseLLM
from ..models import SessionLog, FeedbackInput, FeedbackScope

class ContextAnalyzer:
    """
    세션 기반 지능형 컨텍스트 분석기
    
    사용자의 피드백 히스토리를 분석하여 충돌을 해결하고,
    정제된 요구사항과 제약 조건을 추출합니다.
    """
    
    def __init__(self, llm: Optional[BaseLLM] = None):
        """
        초기화
        
        Args:
            llm: 분석에 사용할 LLM 인스턴스 (기본 gpt-4o-mini)
        """
        self.llm = llm or LLMFactory.create_manager_agent_llm()

    def analyze_session_context(
        self, 
        session: SessionLog, 
        current_feedback: FeedbackInput
    ) -> Dict[str, Any]:
        """
        세션 전체 맥락을 분석하여 피드백의 명확성과 제약 조건 도출
        
        Returns:
            {
                "is_clear": bool,               # 현 피드백의 명확성
                "question": str,                # 모호할 경우 질문
                "constraints": List[str],        # 추출된 제약 조건 (예: ["검은색 제외"])
                "avoid_attributes": Dict,       # 메타데이터 필터링용 속성
                "reasoning": str                # 분석 근거
            }
        
        Note: 쿼리 정제는 QueryBuilder가 담당합니다.
        """
        prompt = self._build_analysis_prompt(session, current_feedback)
        system_prompt = self._get_system_prompt()
        
        try:
            result = self.llm.generate_json(
                prompt=prompt,
                system_prompt=system_prompt
            )
            
            if result:
                return {
                    "is_clear": result.get("is_clear", True),
                    "question": result.get("question", ""),
                    "constraints": result.get("constraints", []),
                    "avoid_attributes": result.get("avoid_attributes", {}),
                    "reasoning": result.get("reasoning", "")
                }
        except Exception as e:
            print(f"[ContextAnalyzer] 분석 실패: {e}")
            
        # Fallback
        return {
            "is_clear": True,
            "question": "",
            "constraints": [],
            "avoid_attributes": {},
            "reasoning": "Fallback due to error"
        }


    def _build_analysis_prompt(self, session: SessionLog, current_feedback: FeedbackInput) -> str:
        """분석용 프롬프트 구성"""
        # 세션 히스토리 추출
        history = []
        for entry in session.entries:
            if entry.entry_type == "feedback" and not entry.content.get("is_positive"):
                history.append(f"- 피드백: {entry.content.get('feedback_text')}")
            elif entry.entry_type == "action":
                history.append(f"  (시스템 조치: {entry.content.get('action')})")

        prompt = f"""
패션 코디 세션의 맥락을 분석하여 유저의 의도를 정교하게 파악하세요.

## 원본 요청
{session.context.get("original_prompt", "알 수 없음")}

## 이전 피드백 히스토리
{chr(10).join(history) if history else "없음"}

## 현재 피드백
- 범위: {', '.join([s.value for s in current_feedback.feedback_scopes]) if current_feedback.feedback_scopes else "전체"}
- 내용: "{current_feedback.feedback_text}"

## 작업 지침
1. **명확성 판단**: 현재 피드백만으로 다음 행동을 정할 수 있는지 판단하세요.
2. **모순 해결**: 이전 피드백과 현재 피드백이 충돌한다면(예: 밝게 -> 너무 밝음), 유저가 원하는 '중간 영역'을 정의하세요.
3. **제약 조건 추출**: 명시적으로 거부하는 속성(색상, 핏, 소재 등)을 추출하세요.

Note: 쿼리 정제(refined_query)는 QueryBuilder가 담당하므로, 여기서는 명확성과 제약조건만 추출합니다.

## 응답 형식 (JSON) 예시
{{
    "is_clear": true | false,
    "question": "모호할 경우 사용자에게 던질 스무스한 질문",
    "constraints": ["제한 사항 리스트"],
    "avoid_attributes": {{
        "color": ["black"],
        "fit": ["tight"]
    }},
    "reasoning": "피드백 간의 충돌 해결 및 분석 근거"
}}
"""
        return prompt

    def _get_system_prompt(self) -> str:
        """시스템 프롬프트"""
        return """당신은 패션 전문 인터프리터입니다. 
사용자의 파편화된 피드백들을 종합하여, 시스템이 이해할 수 있는 정교한 스타일 가이드를 도출합니다.
유저가 싫어하는 것은 확실히 제외하고, 원하는 방향성의 핵심 키워드를 포착하세요."""
