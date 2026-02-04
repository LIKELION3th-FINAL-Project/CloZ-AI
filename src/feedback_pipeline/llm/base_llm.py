"""
LLM 추상화 인터페이스

모든 LLM 구현체의 기본 클래스
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import json


@dataclass
class LLMConfig:
    """LLM 설정"""
    model_name: str                         # "gpt-4o-mini", "claude-3-sonnet", etc.
    api_key: Optional[str] = None           # API 키
    base_url: Optional[str] = None          # 로컬 LLM URL (예: http://localhost:11434)
    temperature: float = 0.7
    max_tokens: int = 2048
    timeout: int = 30                       # 초
    retry_count: int = 3
    extra_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    """LLM 응답"""
    content: str                            # 응답 텍스트
    raw_response: Optional[Dict] = None     # 원본 API 응답
    usage: Optional[Dict] = None            # 토큰 사용량
    model: str = ""                         # 사용된 모델명

    def to_json(self) -> Optional[Dict]:
        """
        응답을 JSON으로 파싱

        Returns:
            파싱된 딕셔너리 또는 None (파싱 실패 시)
        """
        try:
            # ```json ... ``` 블록 제거
            content = self.content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

            return json.loads(content)
        except json.JSONDecodeError:
            return None


class BaseLLM(ABC):
    """
    LLM 추상 클래스

    모든 LLM 구현체는 이 클래스를 상속
    """

    def __init__(self, config: LLMConfig):
        self.config = config
        self._client = None

    @abstractmethod
    def _init_client(self):
        """클라이언트 초기화 (lazy loading)"""
        pass

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """
        텍스트 생성

        Args:
            prompt: 사용자 프롬프트
            system_prompt: 시스템 프롬프트 (선택)
            **kwargs: 추가 파라미터 (temperature, max_tokens 등)

        Returns:
            LLMResponse
        """
        pass

    def generate_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        schema: Optional[Dict] = None,
        **kwargs
    ) -> Optional[Dict]:
        """
        JSON 응답 생성

        Args:
            prompt: 사용자 프롬프트
            system_prompt: 시스템 프롬프트
            schema: 기대하는 JSON 스키마 (프롬프트에 포함)
            **kwargs: 추가 파라미터

        Returns:
            파싱된 JSON 딕셔너리 또는 None
        """
        # JSON 출력 지시를 프롬프트에 추가
        if schema:
            prompt = f"{prompt}\n\n출력 형식 (JSON):\n```json\n{json.dumps(schema, ensure_ascii=False, indent=2)}\n```"

        if system_prompt:
            system_prompt = f"{system_prompt}\n\n반드시 JSON 형식으로만 응답하세요."
        else:
            system_prompt = "반드시 JSON 형식으로만 응답하세요."

        response = self.generate(prompt, system_prompt, **kwargs)
        return response.to_json()

    @abstractmethod
    def generate_with_images(
        self,
        prompt: str,
        image_urls: List[str],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """
        이미지와 함께 텍스트 생성 (VLM)

        Args:
            prompt: 사용자 프롬프트
            image_urls: 이미지 URL 리스트
            system_prompt: 시스템 프롬프트
            **kwargs: 추가 파라미터

        Returns:
            LLMResponse
        """
        pass

    def health_check(self) -> bool:
        """
        LLM 연결 상태 확인

        Returns:
            연결 성공 여부
        """
        try:
            response = self.generate("Hello", max_tokens=10)
            return bool(response.content)
        except Exception:
            return False
