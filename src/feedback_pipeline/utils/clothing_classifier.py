import base64
from io import BytesIO
from typing import List, Dict, Optional
from PIL import Image
import rembg
import numpy as np
import os
from ..llm.factory import LLMFactory
from ..llm.base_llm import BaseLLM
from google.generativeai.types import HarmCategory, HarmBlockThreshold


class ClothingClassifier:
    """
    의류 이미지 분류 모듈

    Gemini Vision을 사용하여 의류 이미지의 세부 카테고리를 분류한다.
    배경 제거(누끼) 기능도 포함.

    카테고리 체계: Havati 상품 데이터 기준
    """

    # Havati 카테고리 (상품 데이터 기준)
    CATEGORIES = {
        "상의": [
            "Tee", "Shirt", "Sweatshirt", "Knitwear"
        ],
        "바지": [
            "Denim", "Chino", "Trousers", "Easy_pants", "Work_pants", "Short"
        ],
        "아우터": [
            "Jacket_Blouson", "Coat", "Cardigan", "Jumper_Parka",
            "Padding", "Leather", "Vest"
        ]
    }

    def __init__(self, llm: Optional[BaseLLM] = None):
        """
        Args:
            llm: 분류에 사용할 LLM 인스턴스. None이면 Gemini 2.0 Flash 사용.
        """
        self.llm = llm or LLMFactory.create_gemini(model_name="gemini-2.0-flash")

    def remove_background(self, image: Image.Image) -> Image.Image:
        """
        이미지에서 배경을 제거합니다 (누끼 작업).

        Args:
            image: 원본 PIL Image

        Returns:
            배경이 제거된 투명 배경(RGBA)의 PIL Image
        """
        img_array = np.array(image)
        output_array = rembg.remove(img_array)
        return Image.fromarray(output_array)

    def classify_item(self, image: Image.Image, broad_category: str, use_original_if_failed: bool = True) -> str:
        """
        이미지를 분석하여 지정된 대 카테고리 내의 세부 카테고리를 반환합니다.

        Args:
            image: 분석할 PIL Image (배경 제거된 상태 권장)
            broad_category: 대 카테고리 ("상의", "바지", "아우터")
            use_original_if_failed: 분류 실패 시 원래 값을 반환할지 여부

        Returns:
            분류된 세부 카테고리 명칭
        """
        # "하의" -> "바지" 변환
        target_category = broad_category
        if target_category == "하의":
            target_category = "바지"

        if target_category not in self.CATEGORIES:
            raise ValueError(f"지원하지 않는 대 카테고리입니다: {broad_category}")

        sub_categories = self.CATEGORIES[target_category]

        # 이미지를 base64 data URL로 변환
        buffered = BytesIO()
        image.convert("RGB").save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        data_url = f"data:image/jpeg;base64,{img_str}"

        prompt = (
            f"당신은 패션 전문가입니다. 첨부된 이미지는 '{target_category}' 카테고리의 의류 사진입니다.\n"
            f"이 의류를 아래의 세부 카테고리 목록 중 가장 적절한 하나로 분류해 주세요:\n\n"
            f"세부 카테고리 목록: {', '.join(sub_categories)}\n\n"
            f"결과는 반드시 목록에 있는 이름 중 하나만 정확하게 답변해 주세요. 다른 설명은 생략하세요."
        )

        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

        response = self.llm.generate_with_images(
            prompt=prompt,
            image_urls=[data_url],
            max_tokens=500,
            safety_settings=safety_settings,
            temperature=0
        )

        result = response.content.strip()

        # 응답에서 카테고리명 매칭 (정확 또는 부분 매칭)
        for sub in sub_categories:
            if sub in result or result in sub:
                return sub

        return result
