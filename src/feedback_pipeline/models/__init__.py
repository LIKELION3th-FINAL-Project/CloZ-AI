"""
피드백 시스템 데이터 모델

백엔드 API 스키마와 호환되는 내부 데이터 구조 정의
"""

from .feedback import (
    CategoryMain,
    ItemInfo,
    OutfitSet,
    FeedbackScope,
    FeedbackInput,
    ManagerDecision,
    ActionType,
)

from .user_profile import (
    UserProfile,
    UserBias,
    ColorPreferences,
    FitPreferences,
    KeywordMention,
    MIN_MENTIONS_FOR_PROFILE,
    CONFIDENCE_DECAY_DAYS,
)

from .session import (
    SessionLog,
    SessionEntry,
    SessionStatus,
)

__all__ = [
    # feedback.py
    "CategoryMain",
    "ItemInfo",
    "OutfitSet",
    "FeedbackScope",
    "FeedbackInput",
    "ManagerDecision",
    "ActionType",
    # user_profile.py
    "UserProfile",
    "UserBias",
    "ColorPreferences",
    "FitPreferences",
    "KeywordMention",
    "MIN_MENTIONS_FOR_PROFILE",
    "CONFIDENCE_DECAY_DAYS",
    # session.py
    "SessionLog",
    "SessionEntry",
    "SessionStatus",
]
