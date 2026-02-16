"""
Storage Layer

사용자 프로필, 세션 로그, 피드백 데이터 저장
"""

from .json_storage import JsonStorage, StorageConfig

__all__ = [
    "JsonStorage",
    "StorageConfig",
]
