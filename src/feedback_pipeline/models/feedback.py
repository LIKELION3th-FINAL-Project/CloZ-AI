"""
피드백 관련 데이터 모델

백엔드 API Response 스키마 (POST /agent/recommend):
{
  "message": "희라님에게 어울리는 착장 3개를 준비했어요.",
  "outfits": [
    {
      "outfit_id": 1,
      "image_url": "https://...",
      "products": [
        {
          "product_id": 101,
          "category_main": "바지",
          "category_sub": "데님 팬츠",
          "product_name": "와이드 데님 팬츠"
        }
      ]
    }
  ]
}

상품 DB 스키마 (GET /products):
{
  "id": "4071245",
  "category_main": "아우터",
  "category_sub": "트레이닝 재킷",
  "brand": "아디다스",
  "product_name": "...",
  "price": 66640,
  "product_url": "https://www.musinsa.com/products/4071245",
  "product_image_path": "images/아우터/트레이닝 재킷/4071245.jpg"
}

NOTE: category_main 값은 "상의", "바지", "아우터" 사용 (하의 X)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional


class CategoryMain(Enum):
    """대분류 카테고리 (백엔드 기준)"""
    TOP = "상의"
    BOTTOM = "바지"     # NOTE: "하의" -> "바지"
    OUTER = "아우터"


class FeedbackScope(Enum):
    """피드백 대상 범위"""
    FULL = "FULL"       # 전체 코디
    TOP = "TOP"         # 상의만
    BOTTOM = "BOTTOM"   # 바지만
    OUTER = "OUTER"     # 아우터만


class ActionType(Enum):
    """Manager Agent 판단 결과 액션"""
    APPROVED = "APPROVED"       # YES 피드백, 정답 저장
    REGENERATE = "REGENERATE"   # 재생성 요청
    BUYING = "BUYING"           # 상품 추천
    ASK_MORE = "ASK_MORE"       # 추가 질문 필요


@dataclass
class ItemInfo:
    """
    개별 상품 정보

    백엔드 sources:
    1. /agent/recommend response: product_id, category_main, category_sub, product_name
    2. /products response: + brand, price, product_url, product_image_path

    BUYING 시에만 brand, price 등 상세 정보 추가
    """
    product_id: int                         # 상품 ID (백엔드는 int)
    product_name: str
    category_main: str                      # "상의", "바지", "아우터"
    category_sub: str                       # 38개 세부 카테고리
    brand: Optional[str] = None             # BUYING 시 추가
    price: Optional[int] = None             # BUYING 시 추가
    product_url: Optional[str] = None       # 무신사 상품 URL
    product_image_path: Optional[str] = None  # 상품 이미지 경로

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        result = {
            "product_id": self.product_id,
            "product_name": self.product_name,
            "category_main": self.category_main,
            "category_sub": self.category_sub,
        }
        if self.brand:
            result["brand"] = self.brand
        if self.price is not None:
            result["price"] = self.price
        if self.product_url:
            result["product_url"] = self.product_url
        if self.product_image_path:
            result["product_image_path"] = self.product_image_path
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ItemInfo":
        """딕셔너리에서 생성 (백엔드 response 호환)"""
        # product_id는 int 또는 str로 올 수 있음
        product_id = data.get("product_id") or data.get("id")
        if isinstance(product_id, str):
            product_id = int(product_id)

        return cls(
            product_id=product_id,
            product_name=data["product_name"],
            category_main=data["category_main"],
            category_sub=data["category_sub"],
            brand=data.get("brand"),
            price=data.get("price"),
            product_url=data.get("product_url"),
            product_image_path=data.get("product_image_path"),
        )


@dataclass
class OutfitSet:
    """
    코디 세트

    백엔드 outfits 배열의 각 요소에 대응
    사용자에게는 image_url만 표시
    피드백 시스템에서는 products 정보 활용
    """
    outfit_id: int
    image_url: str                      # S3 URL
    products: List[ItemInfo]            # 상품 리스트
    generation_prompt: Optional[str] = None  # 생성 프롬프트 (세션에서 관리)

    def get_item_by_category(self, category_main: str) -> Optional[ItemInfo]:
        """카테고리로 아이템 찾기"""
        for item in self.products:
            if item.category_main == category_main:
                return item
        return None

    def get_top(self) -> Optional[ItemInfo]:
        """상의 아이템"""
        return self.get_item_by_category(CategoryMain.TOP.value)

    def get_bottom(self) -> Optional[ItemInfo]:
        """바지 아이템 (백엔드 기준 category_main="바지")"""
        return self.get_item_by_category(CategoryMain.BOTTOM.value)

    def get_outer(self) -> Optional[ItemInfo]:
        """아우터 아이템"""
        return self.get_item_by_category(CategoryMain.OUTER.value)

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "outfit_id": self.outfit_id,
            "image_url": self.image_url,
            "products": [item.to_dict() for item in self.products],
            "generation_prompt": self.generation_prompt,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OutfitSet":
        """딕셔너리에서 생성"""
        return cls(
            outfit_id=data["outfit_id"],
            image_url=data["image_url"],
            products=[ItemInfo.from_dict(p) for p in data["products"]],
            generation_prompt=data.get("generation_prompt"),
        )


@dataclass
class FeedbackInput:
    """
    피드백 입력 데이터

    백엔드 POST /api/feedback 요청 body에 대응

    feedback_scopes: 복수 선택 가능 (예: [TOP, BOTTOM])
    프론트엔드 UI에서 체크박스로 선택
    """
    session_id: str
    user_id: str
    is_positive: bool               # YES=True, NO=False
    current_outfit: OutfitSet       # 사용자가 선택한 코디
    feedback_text: str = ""         # NO일 때 이유
    feedback_scopes: List[FeedbackScope] = field(default_factory=lambda: [FeedbackScope.FULL])
    target_items: Optional[List[str]] = None  # ["top", "bottom"]

    def __post_init__(self):
        """검증"""
        if not self.is_positive and not self.feedback_text:
            raise ValueError("NO 피드백 시 feedback_text는 필수입니다")

        # 빈 리스트면 FULL로 설정
        if not self.feedback_scopes:
            self.feedback_scopes = [FeedbackScope.FULL]

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "is_positive": self.is_positive,
            "outfit_id": self.current_outfit.outfit_id,
            "feedback_text": self.feedback_text,
            "feedback_scopes": [s.value for s in self.feedback_scopes],
            "target_items": self.target_items,
        }


@dataclass
class ManagerDecision:
    """
    Manager Agent 판단 결과

    백엔드 Response로 변환됨
    """
    action: ActionType
    message: str = ""                                       # 사용자에게 보여줄 메시지
    reasoning: str = ""                                     # 판단 근거 (내부용)
    payload: Dict[str, Any] = field(default_factory=dict)
    extracted_requirements: List[str] = field(default_factory=list)  # LLM이 추출한 요구사항
    target_categories: List[str] = field(default_factory=list)       # 변경 대상 카테고리
    buying_recommendations: Optional[Any] = None            # BuyingRecommendation 객체

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        result = {
            "action": self.action.value,
            "message": self.message,
            "reasoning": self.reasoning,
            "payload": self.payload,
        }
        if self.extracted_requirements:
            result["extracted_requirements"] = self.extracted_requirements
        if self.target_categories:
            result["target_categories"] = self.target_categories
        if self.buying_recommendations:
            result["buying_recommendations"] = (
                self.buying_recommendations.to_dict()
                if hasattr(self.buying_recommendations, 'to_dict')
                else self.buying_recommendations
            )
        return result
