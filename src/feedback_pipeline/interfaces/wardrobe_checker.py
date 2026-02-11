"""
Wardrobe Checker Interface

옷장 내 아이템으로 피드백 요구사항 충족 가능 여부 판단

NOTE: 이 인터페이스는 생성 모델 파트에서 구현해야 함
      피드백 시스템에서는 이 인터페이스를 통해 호출만 함
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class WardrobeCheckResult:
    """
    옷장 체크 결과

    Attributes:
        is_possible: 옷장 내 아이템으로 충족 가능 여부
        missing_items: 부족한 아이템 정보 (category, description)
        matching_items: 매칭된 기존 아이템 ID 목록
        reason: 판단 근거 (디버깅/로깅용)
        confidence: 판단 신뢰도 (0.0 ~ 1.0)
    """
    is_possible: bool
    missing_items: List[Dict[str, Any]] = field(default_factory=list)
    matching_items: List[str] = field(default_factory=list)
    candidate_pool: Dict[str, List[str]] = field(default_factory=dict)
    reason: str = ""
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_possible": self.is_possible,
            "missing_items": self.missing_items,
            "matching_items": self.matching_items,
            "candidate_pool": self.candidate_pool,
            "reason": self.reason,
            "confidence": self.confidence,
        }


class WardrobeCheckerInterface(ABC):
    """
    옷장 체크 인터페이스 (Abstract)

    생성 모델 파트에서 이 인터페이스를 구현해야 함

    사용 예시:
        class MyWardrobeChecker(WardrobeCheckerInterface):
            def __init__(self, user_closet_db):
                self.db = user_closet_db

            def can_fulfill(self, requirements, user_id, context):
                # 사용자 옷장 DB 조회
                # 요구사항 매칭 로직
                return WardrobeCheckResult(...)

        # Manager Agent에서 사용
        checker = MyWardrobeChecker(db)
        result = checker.can_fulfill(
            requirements=["밝은 색상의 상의", "슬림핏 하의"],
            user_id="user_001"
        )

        if not result.is_possible:
            # BUYING 전환
            missing = result.missing_items
    """

    @abstractmethod
    def can_fulfill(
        self,
        requirements: List[str],
        user_id: str,
        context: Optional[Dict[str, Any]] = None,
        target_categories: Optional[List[str]] = None,
        target_detail_cats: Optional[List[str]] = None,
        avoid_detail_cats: Optional[List[str]] = None,
    ) -> WardrobeCheckResult:
        """
        옷장 내 아이템으로 요구사항 충족 가능 여부 판단

        Args:
            requirements: 요구 조건 리스트
                - ["밝은 색상의 상의", "와이드핏 하의"]
                - ["캐주얼한 아우터"]
            user_id: 사용자 ID (옷장 DB 조회용)
            context: 추가 컨텍스트
                - occasion: 상황 (출근, 데이트 등)
                - weather: 날씨
                - current_outfit: 현재 코디 (부분 교체 시)

        Returns:
            WardrobeCheckResult
        """
        pass

    def check_category(
        self,
        category: str,
        requirements: List[str],
        user_id: str
    ) -> WardrobeCheckResult:
        """
        특정 카테고리에 대해서만 체크 (부분 피드백용)

        Args:
            category: "상의", "바지", "아우터"
            requirements: 해당 카테고리 요구 조건
            user_id: 사용자 ID

        Returns:
            WardrobeCheckResult
        """
        # 기본 구현: can_fulfill 호출
        return self.can_fulfill(
            requirements=requirements,
            user_id=user_id,
            context={"target_category": category}
        )


class DummyWardrobeChecker(WardrobeCheckerInterface):
    """
    테스트/개발용 더미 구현

    항상 is_possible=True 반환 (실제 체크 없음)
    """

    def can_fulfill(
        self,
        requirements: List[str],
        user_id: str,
        context: Optional[Dict[str, Any]] = None,
        target_categories: Optional[List[str]] = None,
        target_detail_cats: Optional[List[str]] = None,
        avoid_detail_cats: Optional[List[str]] = None,
    ) -> WardrobeCheckResult:
        """더미: 항상 가능 반환"""
        return WardrobeCheckResult(
            is_possible=True,
            matching_items=[],
            reason="[Dummy] 실제 체크 없이 항상 True 반환",
            confidence=0.0,
        )


class AlwaysBuyingWardrobeChecker(WardrobeCheckerInterface):
    """
    테스트용: 항상 BUYING 유도

    항상 is_possible=False 반환
    """

    def can_fulfill(
        self,
        requirements: List[str],
        user_id: str,
        context: Optional[Dict[str, Any]] = None,
        target_categories: Optional[List[str]] = None,
        target_detail_cats: Optional[List[str]] = None,
        avoid_detail_cats: Optional[List[str]] = None,
    ) -> WardrobeCheckResult:
        """더미: 항상 불가능 반환"""
        return WardrobeCheckResult(
            is_possible=False,
            missing_items=[
                {"category": "unknown", "description": req}
                for req in requirements
            ],
            reason="[Test] 강제 BUYING 테스트",
            confidence=1.0,
        )
