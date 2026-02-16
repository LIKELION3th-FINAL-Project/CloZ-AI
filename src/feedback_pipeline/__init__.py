"""
AI Fashion Coordinator - Feedback System

Manager Agent + Analyst Agent 기반 피드백 루프 및 개인화 시스템

사용 예시:
    from feedback_pipeline import (
        # Models
        FeedbackInput, FeedbackScope, OutfitSet, ItemInfo,
        UserProfile, UserBias,
        SessionLog, SessionEntry,

        # LLM
        LLMFactory, LLMProvider,

        # Storage
        JsonStorage,

        # Agents
        ManagerAgent, AnalystAgent,
    )
"""

# Models
from .models import (
    CategoryMain,
    ItemInfo,
    OutfitSet,
    FeedbackScope,
    FeedbackInput,
    ManagerDecision,
    ActionType,
    UserProfile,
    UserBias,
    ColorPreferences,
    FitPreferences,
    SessionLog,
    SessionEntry,
    SessionStatus,
)

# LLM
from .llm import (
    BaseLLM,
    LLMConfig,
    LLMResponse,
    OpenAILLM,
    LLMFactory,
    LLMProvider,
)

# Storage
from .storage import (
    JsonStorage,
    StorageConfig,
)

# Agents
from .agents import (
    ManagerAgent,
    AnalystAgent,
)

# Interfaces
from .interfaces import (
    WardrobeCheckerInterface,
    WardrobeCheckResult,
    GenerationModelInterface,
    GenerationResult,
    BuyingTriggerInterface,
    BuyingRecommendation,
)

__all__ = [
    # Models - Feedback
    "CategoryMain",
    "ItemInfo",
    "OutfitSet",
    "FeedbackScope",
    "FeedbackInput",
    "ManagerDecision",
    "ActionType",
    # Models - User Profile
    "UserProfile",
    "UserBias",
    "ColorPreferences",
    "FitPreferences",
    # Models - Session
    "SessionLog",
    "SessionEntry",
    "SessionStatus",
    # LLM
    "BaseLLM",
    "LLMConfig",
    "LLMResponse",
    "OpenAILLM",
    "LLMFactory",
    "LLMProvider",
    # Storage
    "JsonStorage",
    "StorageConfig",
    # Agents
    "ManagerAgent",
    "AnalystAgent",
    # Interfaces
    "WardrobeCheckerInterface",
    "WardrobeCheckResult",
    "GenerationModelInterface",
    "GenerationResult",
    "BuyingTriggerInterface",
    "BuyingRecommendation",
]
