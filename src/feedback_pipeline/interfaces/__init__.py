"""
Interfaces

생성 모델 파트와의 연동 인터페이스 정의
"""

from .wardrobe_checker import (
    WardrobeCheckerInterface,
    WardrobeCheckResult,
)
from .generation_model import (
    GenerationModelInterface,
    GenerationResult,
)
from .buying_trigger import (
    BuyingTriggerInterface,
    BuyingRecommendation,
)

__all__ = [
    # Wardrobe Checker
    "WardrobeCheckerInterface",
    "WardrobeCheckResult",
    # Generation Model
    "GenerationModelInterface",
    "GenerationResult",
    # Buying Trigger
    "BuyingTriggerInterface",
    "BuyingRecommendation",
]
