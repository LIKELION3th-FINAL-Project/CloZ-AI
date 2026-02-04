"""
Gemini LLM 구현체

Google Gemini API 지원

NOTE: google-generativeai 패키지는 deprecated 되었습니다.
향후 google.genai 패키지로 마이그레이션 필요.
현재는 OpenRouter 또는 OpenAI 사용을 권장합니다.
"""

import os
import json
import time
import random
from typing import Dict, Any, Optional, List

from .base_llm import BaseLLM, LLMConfig, LLMResponse


class GeminiLLM(BaseLLM):
    """
    Google Gemini API를 사용하는 LLM 구현체

    지원 모델:
    - gemini-1.5-flash (Manager Agent 권장 - 빠르고 저렴)
    - gemini-1.5-pro (Analyst Agent 권장 - 강력한 분석)
    """

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self._client = None
        self._model = None

    def _init_client(self):
        """Gemini 클라이언트 초기화 (lazy loading)"""
        if self._client is None:
            try:
                import google.generativeai as genai
            except ImportError:
                raise ImportError(
                    "google-generativeai 패키지가 필요합니다. "
                    "pip install google-generativeai 실행하세요."
                )

            api_key = self.config.api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError(
                    "Gemini API 키가 필요합니다. "
                    "config.api_key 또는 GEMINI_API_KEY 환경변수를 설정하세요."
                )

            genai.configure(api_key=api_key)
            self._client = genai
            self._model = genai.GenerativeModel(self.config.model_name)

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

        # 시스템 프롬프트를 프롬프트 앞에 추가
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        # 파라미터 설정
        temperature = kwargs.get("temperature", self.config.temperature)
        max_tokens = kwargs.get("max_tokens", self.config.max_tokens)
        retry_count = kwargs.get("retry_count", self.config.retry_count)

        generation_config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }

        # API 호출 (재시도 로직 포함)
        response = None
        last_exception = None
        
        for attempt in range(retry_count + 1):
            try:
                response = self._model.generate_content(
                    full_prompt,
                    generation_config=generation_config
                )
                break # 성공 시 루프 탈출
            except Exception as e:
                last_exception = e
                if "429" in str(e) and attempt < retry_count:
                    # 지수 백오프 적용 (1s, 2s, 4s, 8s, 16s + jitter)
                    sleep_time = (2 ** attempt) + random.uniform(0, 1)
                    time.sleep(sleep_time)
                    continue
                else:
                    break

        # 응답 파싱
        if response and hasattr(response, 'text'):
            content = response.text
        elif last_exception:
            content = f"Error: {str(last_exception)}"
        else:
            content = ""

        # 사용량 정보 (Gemini는 usage_metadata 제공)
        usage = None
        if response and hasattr(response, 'usage_metadata') and response.usage_metadata:
            usage = {
                "prompt_tokens": response.usage_metadata.prompt_token_count,
                "completion_tokens": response.usage_metadata.candidates_token_count,
                "total_tokens": response.usage_metadata.total_token_count,
            }

        return LLMResponse(
            content=content,
            raw_response=None,  # Gemini response는 직렬화 안됨
            usage=usage,
            model=self.config.model_name,
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
            image_urls: 이미지 URL 리스트 또는 로컬 경로
            system_prompt: 시스템 프롬프트
            **kwargs: temperature, max_tokens 등

        Returns:
            LLMResponse
        """
        self._init_client()

        # 시스템 프롬프트 처리
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        # 파라미터 설정
        temperature = kwargs.get("temperature", self.config.temperature)
        max_tokens = kwargs.get("max_tokens", self.config.max_tokens)
        retry_count = kwargs.get("retry_count", self.config.retry_count)

        generation_config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }

        # 이미지 로드
        from PIL import Image
        import requests
        from io import BytesIO

        content_parts = [full_prompt]
        for url in image_urls:
            if url.startswith("data:image"):
                # base64 data URL 처리
                import base64
                from io import BytesIO
                header, encoded = url.split(",", 1)
                data = base64.b64decode(encoded)
                img = Image.open(BytesIO(data))
            elif url.startswith("http"):
                # URL에서 이미지 다운로드
                response = requests.get(url)
                img = Image.open(BytesIO(response.content))
            else:
                # 로컬 경로
                img = Image.open(url)
            content_parts.append(img)

        # API 호출 (재시도 로직 포함)
        response = None
        last_exception = None
        content = ""

        for attempt in range(retry_count + 1):
            try:
                response = self._model.generate_content(
                    content_parts,
                    generation_config=generation_config
                )
                content = response.text if response.text else ""
                break # 성공 시 루프 탈출
            except Exception as e:
                last_exception = e
                # 429 Resource exhausted 에러 발생 시 재시도
                if "429" in str(e) and attempt < retry_count:
                    # 지수 백오프 적용 (1s, 2s, 4s, 8s, 16s + jitter)
                    sleep_time = (2 ** attempt) + random.uniform(0, 1)
                    time.sleep(sleep_time)
                    continue
                else:
                    content = f"Error: {str(e)}"
                    if response is not None and hasattr(response, 'candidates') and response.candidates:
                        finish_reason = response.candidates[0].finish_reason
                        content = f"Classification failed (Safety/FinishReason: {finish_reason})"
                    break

        usage = None
        if response and hasattr(response, 'usage_metadata') and response.usage_metadata:
            usage = {
                "prompt_tokens": response.usage_metadata.prompt_token_count,
                "completion_tokens": response.usage_metadata.candidates_token_count,
                "total_tokens": response.usage_metadata.total_token_count,
            }

        return LLMResponse(
            content=content,
            raw_response=None,
            usage=usage,
            model=self.config.model_name,
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

        Gemini는 JSON mode가 없으므로 프롬프트로 지시하고 파싱
        """
        # JSON 출력 지시를 프롬프트에 추가
        if schema:
            prompt = f"{prompt}\n\n출력 형식 (JSON):\n```json\n{json.dumps(schema, ensure_ascii=False, indent=2)}\n```"

        if system_prompt:
            system_prompt = f"{system_prompt}\n\n반드시 유효한 JSON 형식으로만 응답하세요. 코드 블록(```)으로 감싸지 마세요."
        else:
            system_prompt = "반드시 유효한 JSON 형식으로만 응답하세요. 코드 블록(```)으로 감싸지 마세요."

        response = self.generate(prompt, system_prompt, **kwargs)

        # JSON 파싱 (코드 블록 제거)
        content = response.content.strip()

        # ```json ... ``` 제거
        if content.startswith("```json"):
            content = content[7:]
        elif content.startswith("```"):
            content = content[3:]

        if content.endswith("```"):
            content = content[:-3]

        content = content.strip()

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # 파싱 실패 시 None 반환
            return None
