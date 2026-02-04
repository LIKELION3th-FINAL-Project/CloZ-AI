"""
Generation Model Interface

코디 생성 모델과의 연동 인터페이스

NOTE: 이 인터페이스는 생성 모델 파트에서 구현해야 함
      Manager Agent에서 REGENERATE 시 이 인터페이스 통해 호출
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from ..models import OutfitSet


@dataclass
class GenerationResult:
    """
    코디 생성 결과

    Attributes:
        success: 생성 성공 여부
        outfits: 생성된 코디 리스트
        message: 사용자에게 보여줄 메시지
        metadata: 추가 메타데이터 (생성 시간, 모델 버전 등)
    """
    success: bool
    outfits: List[OutfitSet] = field(default_factory=list)
    message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "outfits": [o.to_dict() for o in self.outfits],
            "message": self.message,
            "metadata": self.metadata,
        }


class GenerationModelInterface(ABC):
    """
    코디 생성 모델 인터페이스 (Abstract)

    생성 모델 파트에서 이 인터페이스를 구현해야 함

    사용 예시:
        class MyGenerationModel(GenerationModelInterface):
            def generate(self, prompt, user_id, constraints, context):
                # 코디 생성 로직
                return GenerationResult(...)

            def regenerate(self, original_result, feedback, constraints):
                # 재생성 로직
                return GenerationResult(...)

        # Manager Agent에서 사용
        model = MyGenerationModel()
        result = model.regenerate(
            original_result=current_outfits,
            feedback="상의가 너무 어두워요",
            constraints={"color": ["밝은", "화이트", "베이지"]}
        )
    """

    @abstractmethod
    def generate(
        self,
        prompt: str,
        user_id: str,
        constraints: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> GenerationResult:
        """
        새로운 코디 생성

        Args:
            prompt: 사용자 요청 프롬프트
                - "오피스룩 추천해줘"
                - "주말 데이트 코디"
            user_id: 사용자 ID
            constraints: 제약 조건
                - colors: 선호/회피 색상
                - fit: 핏 선호도
                - categories: 필요한 카테고리
            context: 추가 컨텍스트
                - user_profile: 사용자 프로필 요약
                - occasion: 상황
                - weather: 날씨

        Returns:
            GenerationResult
        """
        pass

    @abstractmethod
    def regenerate(
        self,
        original_result: GenerationResult,
        feedback: str,
        constraints: Optional[Dict[str, Any]] = None,
        target_categories: Optional[List[str]] = None
    ) -> GenerationResult:
        """
        피드백 기반 재생성

        Args:
            original_result: 이전 생성 결과
            feedback: 사용자 피드백 텍스트
            constraints: 추가 제약 조건 (피드백에서 추출)
            target_categories: 변경 대상 카테고리
                - None: 전체 재생성
                - ["상의"]: 상의만 교체
                - ["바지", "아우터"]: 하의+아우터 교체

        Returns:
            GenerationResult
        """
        pass


class DummyGenerationModel(GenerationModelInterface):
    """
    테스트/개발용 더미 구현

    고정된 샘플 코디 반환
    """

    def generate(
        self,
        prompt: str,
        user_id: str,
        constraints: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> GenerationResult:
        """더미: 샘플 코디 반환"""
        from ..models import ItemInfo

        sample_outfit = OutfitSet(
            outfit_id=1,
            image_url="https://example.com/sample_outfit.jpg",
            products=[
                ItemInfo(
                    product_id=101,
                    product_name="[Sample] 화이트 셔츠",
                    category_main="상의",
                    category_sub="셔츠/블라우스"
                ),
                ItemInfo(
                    product_id=201,
                    product_name="[Sample] 네이비 슬랙스",
                    category_main="바지",
                    category_sub="슬랙스"
                )
            ]
        )

        return GenerationResult(
            success=True,
            outfits=[sample_outfit],
            message=f"[Dummy] '{prompt}' 요청에 대한 샘플 코디입니다.",
            metadata={"generator": "DummyGenerationModel"}
        )

    def regenerate(
        self,
        original_result: GenerationResult,
        feedback: str,
        constraints: Optional[Dict[str, Any]] = None,
        target_categories: Optional[List[str]] = None
    ) -> GenerationResult:
        """더미: 약간 수정된 샘플 코디 반환"""
        from ..models import ItemInfo

        # 원본에서 약간 변형
        sample_outfit = OutfitSet(
            outfit_id=2,
            image_url="https://example.com/regenerated_outfit.jpg",
            products=[
                ItemInfo(
                    product_id=102,
                    product_name="[Regenerated] 베이지 니트",
                    category_main="상의",
                    category_sub="니트/스웨터"
                ),
                ItemInfo(
                    product_id=202,
                    product_name="[Regenerated] 블랙 진",
                    category_main="바지",
                    category_sub="데님 팬츠"
                )
            ]
        )

        return GenerationResult(
            success=True,
            outfits=[sample_outfit],
            message=f"[Dummy] 피드백 '{feedback}'을 반영한 재생성 결과입니다.",
            metadata={
                "generator": "DummyGenerationModel",
                "feedback_applied": feedback,
                "target_categories": target_categories,
            }
        )
