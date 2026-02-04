"""
Buying Trigger Interface

옷장에 없는 아이템에 대한 구매 추천

NOTE: 이 인터페이스는 피드백 시스템에서 직접 구현
      상품 DB 연동이 필요함
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from ..models import OutfitSet, FeedbackScope


@dataclass
class ProductRecommendation:
    """
    추천 상품 정보

    Attributes:
        product_id: 상품 ID
        product_name: 상품명
        brand: 브랜드
        price: 가격
        category_main: 대분류
        category_sub: 소분류
        product_url: 상품 URL
        product_image_path: 상품 이미지 경로
        match_score: 매칭 점수 (0.0 ~ 1.0)
        match_reason: 추천 이유
    """
    product_id: int
    product_name: str
    brand: str
    price: int
    category_main: str
    category_sub: str
    product_url: Optional[str] = None
    product_image_path: Optional[str] = None
    match_score: float = 0.0
    match_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "product_id": self.product_id,
            "product_name": self.product_name,
            "brand": self.brand,
            "price": self.price,
            "category_main": self.category_main,
            "category_sub": self.category_sub,
            "product_url": self.product_url,
            "product_image_path": self.product_image_path,
            "match_score": self.match_score,
            "match_reason": self.match_reason,
        }


@dataclass
class BuyingRecommendation:
    """
    구매 추천 결과

    Attributes:
        success: 추천 성공 여부
        products: 추천 상품 리스트 (3-5개)
        reasoning: 추천 근거
        target_category: 대상 카테고리
        sort_options: 지원하는 정렬 옵션
        feedback_analysis: Analyst Agent의 피드백 분석 결과 (선택)
        profile_updates: UserProfile에 반영할 업데이트 내용 (선택)
            - style_keywords_to_add: 3회 달성하여 추가될 선호 키워드
            - avoid_keywords_to_add: 3회 달성하여 추가될 회피 키워드
            - mentions_updated: 언급 횟수 업데이트 정보
    """
    success: bool
    products: List[ProductRecommendation] = field(default_factory=list)
    reasoning: str = ""
    target_category: Optional[str] = None
    sort_options: List[str] = field(default_factory=lambda: ["price_asc", "price_desc", "match_score"])
    feedback_analysis: Optional[Dict[str, Any]] = None
    profile_updates: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "products": [p.to_dict() for p in self.products],
            "reasoning": self.reasoning,
            "target_category": self.target_category,
            "sort_options": self.sort_options,
            "feedback_analysis": self.feedback_analysis,
            "profile_updates": self.profile_updates,
        }

    def sort_by_price(self, ascending: bool = True) -> List[ProductRecommendation]:
        """가격순 정렬"""
        return sorted(
            self.products,
            key=lambda p: p.price,
            reverse=not ascending
        )

    def sort_by_match_score(self) -> List[ProductRecommendation]:
        """매칭 점수순 정렬"""
        return sorted(
            self.products,
            key=lambda p: p.match_score,
            reverse=True
        )


class BuyingTriggerInterface(ABC):
    """
    구매 추천 인터페이스 (Abstract)

    상품 DB와 연동하여 피드백 기반 상품 추천

    사용 예시:
        class MyBuyingTrigger(BuyingTriggerInterface):
            def __init__(self, product_db):
                self.db = product_db

            def recommend(self, original_prompt, feedback_text, ...):
                # LLM으로 요구사항 분석
                # 상품 DB 검색
                # 매칭 점수 계산
                return BuyingRecommendation(...)

        # Manager Agent에서 사용
        trigger = MyBuyingTrigger(product_db)
        result = trigger.recommend(
            original_prompt="오피스룩 추천",
            feedback_text="상의가 너무 어두워요",
            feedback_scope=FeedbackScope.TOP,
            current_outfit=outfit,
            limit=5
        )
    """

    @abstractmethod
    def recommend(
        self,
        original_prompt: str,
        feedback_text: str,
        feedback_scope: FeedbackScope,
        current_outfit: OutfitSet,
        limit: int = 5,
        context: Optional[Dict[str, Any]] = None
    ) -> BuyingRecommendation:
        """
        피드백 기반 상품 추천

        Args:
            original_prompt: 원본 코디 요청
            feedback_text: 사용자 피드백 (NO 이유)
            feedback_scope: 피드백 범위
                - FULL: 전체 코디
                - TOP: 상의만
                - BOTTOM: 하의만
                - OUTER: 아우터만
            current_outfit: 현재 코디 (참조용)
            limit: 추천 개수 (기본 5개)
            context: 추가 컨텍스트
                - user_profile: 사용자 프로필
                - price_range: 가격 범위 필터

        Returns:
            BuyingRecommendation
        """
        pass

    def get_target_category(self, feedback_scope: FeedbackScope) -> Optional[str]:
        """
        피드백 범위에서 대상 카테고리 추출

        Args:
            feedback_scope: 피드백 범위

        Returns:
            카테고리 문자열 또는 None (전체)
        """
        scope_to_category = {
            FeedbackScope.TOP: "상의",
            FeedbackScope.BOTTOM: "바지",
            FeedbackScope.OUTER: "아우터",
            FeedbackScope.FULL: None,
        }
        return scope_to_category.get(feedback_scope)


class DummyBuyingTrigger(BuyingTriggerInterface):
    """
    테스트/개발용 더미 구현

    고정된 샘플 상품 반환
    """

    def recommend(
        self,
        original_prompt: str,
        feedback_text: str,
        feedback_scope: FeedbackScope,
        current_outfit: OutfitSet,
        limit: int = 5,
        context: Optional[Dict[str, Any]] = None
    ) -> BuyingRecommendation:
        """더미: 샘플 상품 반환"""
        target_category = self.get_target_category(feedback_scope)

        # 카테고리별 샘플 상품
        sample_products = {
            "상의": [
                ProductRecommendation(
                    product_id=1001,
                    product_name="[무신사] 오버핏 코튼 셔츠",
                    brand="무탠다드",
                    price=39000,
                    category_main="상의",
                    category_sub="셔츠/블라우스",
                    product_url="https://musinsa.com/1001",
                    match_score=0.85,
                    match_reason="밝은 색상, 캐주얼 핏"
                ),
                ProductRecommendation(
                    product_id=1002,
                    product_name="[무신사] 라운드넥 니트",
                    brand="커버낫",
                    price=49000,
                    category_main="상의",
                    category_sub="니트/스웨터",
                    match_score=0.78,
                    match_reason="부드러운 소재"
                ),
            ],
            "바지": [
                ProductRecommendation(
                    product_id=2001,
                    product_name="[무신사] 와이드 데님 팬츠",
                    brand="리바이스",
                    price=89000,
                    category_main="바지",
                    category_sub="데님 팬츠",
                    product_url="https://musinsa.com/2001",
                    match_score=0.82,
                    match_reason="와이드 핏 데님"
                ),
            ],
            "아우터": [
                ProductRecommendation(
                    product_id=3001,
                    product_name="[무신사] 울 블렌드 코트",
                    brand="무탠다드",
                    price=159000,
                    category_main="아우터",
                    category_sub="코트",
                    product_url="https://musinsa.com/3001",
                    match_score=0.90,
                    match_reason="클래식한 디자인"
                ),
            ],
        }

        # 대상 카테고리의 상품 반환
        if target_category and target_category in sample_products:
            products = sample_products[target_category][:limit]
        else:
            # 전체 카테고리에서 랜덤하게
            products = []
            for cat_products in sample_products.values():
                products.extend(cat_products)
            products = products[:limit]

        return BuyingRecommendation(
            success=True,
            products=products,
            reasoning=f"[Dummy] 피드백 '{feedback_text}'에 기반한 추천",
            target_category=target_category,
        )
