"""
OpenAI LLM 구현체

gpt-4o-mini, gpt-4o 등 OpenAI 모델 지원
"""

import os
from typing import Dict, Any, Optional, List

from .base_llm import BaseLLM, LLMConfig, LLMResponse


class OpenAILLM(BaseLLM):
    """
    OpenAI API를 사용하는 LLM 구현체

    지원 모델:
    - gpt-4o-mini (Manager Agent 권장)
    - gpt-4o
    - gpt-4-turbo
    """

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self._client = None

    def _init_client(self):
        """OpenAI 클라이언트 초기화 (lazy loading)"""
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError(
                    "openai 패키지가 필요합니다. pip install openai 실행하세요."
                )

            api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "OpenAI API 키가 필요합니다. "
                    "config.api_key 또는 OPENAI_API_KEY 환경변수를 설정하세요."
                )

            self._client = OpenAI(
                api_key=api_key,
                base_url=self.config.base_url,
                timeout=self.config.timeout,
            )

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
            system_prompt: 시스템 프롬프트
            **kwargs: temperature, max_tokens 등

        Returns:
            LLMResponse
        """
        self._init_client()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # 파라미터 설정 (kwargs가 config보다 우선)
        temperature = kwargs.get("temperature", self.config.temperature)
        max_tokens = kwargs.get("max_tokens", self.config.max_tokens)

        # API 호출
        response = self._client.chat.completions.create(
            model=self.config.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **self.config.extra_params
        )

        # 응답 파싱
        content = response.choices[0].message.content or ""
        usage = None
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        return LLMResponse(
            content=content,
            raw_response=response.model_dump() if hasattr(response, 'model_dump') else None,
            usage=usage,
            model=response.model,
        )

    def generate_with_images(
        self,
        prompt: str,
        image_urls: List[str],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """
        이미지와 함께 텍스트 생성 (Vision)

        Args:
            prompt: 사용자 프롬프트
            image_urls: 이미지 URL 리스트 (S3 URL, 로컬 경로, base64 등)
            system_prompt: 시스템 프롬프트
            **kwargs: temperature, max_tokens 등

        Returns:
            LLMResponse
        """
        self._init_client()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # 이미지 + 텍스트 메시지 구성
        content = [{"type": "text", "text": prompt}]
        for url in image_urls:
            content.append({
                "type": "image_url",
                "image_url": {"url": url}
            })

        messages.append({"role": "user", "content": content})

        # 파라미터 설정
        temperature = kwargs.get("temperature", self.config.temperature)
        max_tokens = kwargs.get("max_tokens", self.config.max_tokens)

        # Vision 모델 사용 (gpt-4o 권장)
        model = kwargs.get("model", self.config.model_name)
        if "vision" not in model and "4o" not in model:
            # gpt-4o-mini도 vision 지원
            pass

        # API 호출
        response = self._client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        content = response.choices[0].message.content or ""
        usage = None
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        return LLMResponse(
            content=content,
            raw_response=response.model_dump() if hasattr(response, 'model_dump') else None,
            usage=usage,
            model=response.model,
        )

    def generate_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        schema: Optional[Dict] = None,
        **kwargs
    ) -> Optional[Dict]:
        """
        JSON 응답 생성 (response_format 사용)

        OpenAI의 JSON mode 활용
        """
        self._init_client()
        import json

        messages = []
        final_system = system_prompt or ""
        final_system += "\n\n반드시 유효한 JSON 형식으로만 응답하세요."

        if schema:
            final_system += f"\n\n출력 스키마:\n{json.dumps(schema, ensure_ascii=False, indent=2)}"

        messages.append({"role": "system", "content": final_system})
        messages.append({"role": "user", "content": prompt})

        temperature = kwargs.get("temperature", self.config.temperature)
        max_tokens = kwargs.get("max_tokens", self.config.max_tokens)

        try:
            response = self._client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content or ""
            return json.loads(content)

        except Exception:
            # JSON mode 실패 시 일반 생성 후 파싱
            response = self.generate(prompt, system_prompt, **kwargs)
            return response.to_json()
