"""
OpenRouter LLM 구현체

OpenRouter API 지원 (OpenAI 호환 형식)
"""

import os
import json
import requests
from typing import Dict, Any, Optional, List

from .base_llm import BaseLLM, LLMConfig, LLMResponse


class OpenRouterLLM(BaseLLM):
    """
    OpenRouter API를 사용하는 LLM 구현체

    OpenAI 호환 형식이지만 모델명 형식이 다름:
    - openai/gpt-4o-mini (Manager Agent 권장)
    - openai/GPT-5 nano (Analyst Agent 권장)
    - google/gemini-2.0-flash-exp
    """

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.base_url = "https://openrouter.ai/api/v1"

    def _init_client(self):
        """OpenRouter는 requests를 사용하므로 별도 클라이언트 초기화 불필요"""
        pass

    def _get_api_key(self) -> str:
        """API 키 가져오기"""
        api_key = self.config.api_key or os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenRouter API 키가 필요합니다. "
                "config.api_key 또는 OPENROUTER_API_KEY 환경변수를 설정하세요."
            )
        return api_key

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
        api_key = self._get_api_key()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # 파라미터 설정
        temperature = kwargs.get("temperature", self.config.temperature)
        max_tokens = kwargs.get("max_tokens", self.config.max_tokens)

        # API 호출
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.config.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=self.config.timeout,
        )

        response.raise_for_status()
        data = response.json()

        # 응답 파싱
        content = data["choices"][0]["message"]["content"] or ""
        usage = None
        if "usage" in data:
            usage = {
                "prompt_tokens": data["usage"].get("prompt_tokens", 0),
                "completion_tokens": data["usage"].get("completion_tokens", 0),
                "total_tokens": data["usage"].get("total_tokens", 0),
            }

        return LLMResponse(
            content=content,
            raw_response=data,
            usage=usage,
            model=data.get("model", self.config.model_name),
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
            image_urls: 이미지 URL 리스트
            system_prompt: 시스템 프롬프트
            **kwargs: temperature, max_tokens 등

        Returns:
            LLMResponse
        """
        api_key = self._get_api_key()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # 이미지 + 텍스트 메시지 구성 (OpenAI 형식)
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

        # API 호출
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.config.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=self.config.timeout,
        )

        response.raise_for_status()
        data = response.json()

        content = data["choices"][0]["message"]["content"] or ""
        usage = None
        if "usage" in data:
            usage = {
                "prompt_tokens": data["usage"].get("prompt_tokens", 0),
                "completion_tokens": data["usage"].get("completion_tokens", 0),
                "total_tokens": data["usage"].get("total_tokens", 0),
            }

        return LLMResponse(
            content=content,
            raw_response=data,
            usage=usage,
            model=data.get("model", self.config.model_name),
        )

    def generate_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        schema: Optional[Dict] = None,
        **kwargs
    ) -> Optional[Dict]:
        """
        JSON 응답 생성

        OpenRouter는 response_format 지원 (OpenAI 호환)
        """
        api_key = self._get_api_key()

        messages = []
        final_system = system_prompt or ""
        final_system += "\n\n반드시 유효한 JSON 형식으로만 응답하세요."

        if schema:
            final_system += f"\n\n출력 스키마:\n{json.dumps(schema, ensure_ascii=False, indent=2)}"

        messages.append({"role": "system", "content": final_system})
        messages.append({"role": "user", "content": prompt})

        temperature = kwargs.get("temperature", self.config.temperature)
        max_tokens = kwargs.get("max_tokens", self.config.max_tokens)

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.config.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=self.config.timeout,
            )

            response.raise_for_status()
            data = response.json()

            content = data["choices"][0]["message"]["content"] or ""
            return json.loads(content)

        except Exception:
            # JSON mode 실패 시 일반 생성 후 파싱
            response = self.generate(prompt, system_prompt, **kwargs)
            return response.to_json()
