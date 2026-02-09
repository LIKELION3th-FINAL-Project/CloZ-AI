"""
LLM Factory

다양한 LLM 프로바이더를 통합 인터페이스로 생성
"""

from enum import Enum
from typing import Optional

from .base_llm import BaseLLM, LLMConfig


class LLMProvider(Enum):
    """지원하는 LLM 프로바이더"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"         # Ollama 등
    GEMINI = "gemini"       # Google Gemini
    OPENROUTER = "openrouter"  # OpenRouter


class LLMFactory:
    """
    LLM 인스턴스 생성 팩토리

    사용 예시:
        llm = LLMFactory.create(
            LLMProvider.OPENAI,
            LLMConfig(model_name="gpt-4o-mini", api_key="...")
        )
    """

    @staticmethod
    def create(provider: LLMProvider, config: LLMConfig) -> BaseLLM:
        """
        LLM 인스턴스 생성

        Args:
            provider: LLM 프로바이더
            config: LLM 설정

        Returns:
            BaseLLM 구현체
        """
        if provider == LLMProvider.OPENAI:
            from .openai_llm import OpenAILLM
            return OpenAILLM(config)

        elif provider == LLMProvider.GEMINI:
            from .gemini_llm import GeminiLLM
            return GeminiLLM(config)

        elif provider == LLMProvider.OPENROUTER:
            from .openrouter_llm import OpenRouterLLM
            return OpenRouterLLM(config)

        elif provider == LLMProvider.ANTHROPIC:
            # TODO: Anthropic 구현 필요 시 추가
            raise NotImplementedError(
                "Anthropic LLM은 아직 구현되지 않았습니다. "
                "필요 시 feedback_pipeline/llm/anthropic_llm.py를 구현하세요."
            )

        elif provider == LLMProvider.LOCAL:
            # TODO: 로컬 LLM 구현 필요 시 추가
            raise NotImplementedError(
                "로컬 LLM은 아직 구현되지 않았습니다. "
                "필요 시 feedback_pipeline/llm/local_llm.py를 구현하세요."
            )

        else:
            raise ValueError(f"지원하지 않는 LLM 프로바이더: {provider}")

    @staticmethod
    def create_openai(
        model_name: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        **kwargs
    ) -> BaseLLM:
        """
        OpenAI LLM 빠른 생성

        Args:
            model_name: 모델명 (기본: gpt-4o-mini)
            api_key: API 키 (없으면 환경변수 사용)
            temperature: 온도
            **kwargs: 추가 설정

        Returns:
            OpenAILLM 인스턴스
        """
        config = LLMConfig(
            model_name=model_name,
            api_key=api_key,
            temperature=temperature,
            **kwargs
        )
        return LLMFactory.create(LLMProvider.OPENAI, config)

    @staticmethod
    def create_gemini(
        model_name: str = "gemini-2.0-flash",
        api_key: Optional[str] = None,
        temperature: float = 0,
        **kwargs
    ) -> BaseLLM:
        """
        Gemini LLM 빠른 생성

        Args:
            model_name: 모델명 (기본: gemini-2.0-flash)
            api_key: API 키 (없으면 환경변수 사용)
            temperature: 온도
            **kwargs: 추가 설정

        Returns:
            GeminiLLM 인스턴스
        """
        config = LLMConfig(
            model_name=model_name,
            api_key=api_key,
            temperature=temperature,
            **kwargs
        )
        return LLMFactory.create(LLMProvider.GEMINI, config)

    @staticmethod
    def create_manager_agent_llm(api_key: Optional[str] = None) -> BaseLLM:
        """
        Manager Agent용 LLM 생성

        주석 변경으로 프로바이더 전환 가능:
        - OpenRouter: openai/gpt-4o-mini (기본)
        - OpenAI: gpt-4o-mini
        - Gemini: gemini-2.5-flash
        """
        # === OpenRouter (기본, 주석 해제) ===
        config = LLMConfig(
            model_name="openai/gpt-4o-mini",
            api_key=api_key,
            temperature=0,
            max_tokens=1024,
        )
        return LLMFactory.create(LLMProvider.OPENROUTER, config)

        # === OpenAI (주석 처리) ===
        # config = LLMConfig(
        #     model_name="gpt-4o-mini",
        #     api_key=api_key,
        #     temperature=0,
        #     max_tokens=1024,
        #     extra_params={"seed": 42}
        # )
        # return LLMFactory.create(LLMProvider.OPENAI, config)

        # === Gemini (주석 처리) ===
        # config = LLMConfig(
        #     model_name="gemini-2.5-flash",
        #     api_key=api_key,
        #     temperature=0,
        #     max_tokens=1024,
        #     seed=42
        # )
        # return LLMFactory.create(LLMProvider.GEMINI, config)

    @staticmethod
    def create_analyst_agent_llm(api_key: Optional[str] = None) -> BaseLLM:
        """
        Analyst Agent용 LLM 생성

        주석 변경으로 프로바이더 전환 가능:
        - OpenRouter: openai/gpt-oss-120b:free (기본)
        - OpenAI: GPT-5 nano
        - Gemini: gemini-3-flash-preview
        """
        # === OpenRouter (기본, 주석 해제) ===
        config = LLMConfig(
            model_name="openai/gpt-oss-120b:free",
            api_key=api_key,
            temperature=0,
            max_tokens=2048,
        )
        return LLMFactory.create(LLMProvider.OPENROUTER, config)

        # === OpenAI (주석 처리) ===
        # config = LLMConfig(
        #     model_name="GPT-5 nano",
        #     api_key=api_key,
        #     temperature=0.5,
        #     max_tokens=2048,
        #     extra_params={"seed": 42}
        # )
        # return LLMFactory.create(LLMProvider.OPENAI, config)

        # === Gemini (주석 처리) ===
        # config = LLMConfig(
        #     model_name="gemini-3-flash-preview",
        #     api_key=api_key,
        #     temperature=0.5,
        #     max_tokens=2048,
        # )
        # return LLMFactory.create(LLMProvider.GEMINI, config)
