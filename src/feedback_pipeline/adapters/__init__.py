"""
어댑터 모듈

main_simulation과 feedback_pipeline 간의 데이터 변환
"""

from .main_adapter import (
    convert_outfit_to_outfitset,
    get_detail_cat_from_classification,
)

__all__ = [
    "convert_outfit_to_outfitset",
    "get_detail_cat_from_classification",
]
