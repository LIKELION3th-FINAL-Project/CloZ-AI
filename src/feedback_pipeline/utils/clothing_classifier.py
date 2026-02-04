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
    
    배경 제거(누끼) 후 GPT-4o-mini (Vision)를 사용하여 세부 카테고리를 분류합니다.
    """
    
    # 사전에 정의된 대 카테고리별 세부 카테고리 목록
    CATEGORIES = {
        "상의": [
            "맨투맨/스웨트", "후드 티셔츠", "셔츠/블라우스", "긴소매 티셔츠", 
            "반소매 티쳐츠", "피케/카라 티셔츠", "니트/스웨트", "민소매 티셔츠"
        ],
        "바지": [
            "데님 팬츠", "트레이닝/조거 팬츠", "코튼 팬츠", "슈트 팬츠/슬랙스", "숏 팬츠"
        ],
        "아우터": [
            "경량 패딩/패딩 베스트", "숏패딩/헤비 아우터", "롱패딩/헤비 아우터", 
            "플리스/뽀글이", "무스탕/퍼", "레더/라이더스 재킷", "코트", 
            "카디건", "후드 집업", "재킷", "블루종/MA-1"
        ]
    }

    def __init__(self, llm: Optional[BaseLLM] = None):
        """
        ClothingClassifier 초기화
        
        Args:
            llm: 분류에 사용할 LLM 인스턴스. 없을 경우 기본 Gemini 3 Flash를 생성합니다.
        """
        # 한국어 주석: LLM 인스턴스가 제공되지 않으면 LLMFactory를 통해 기본 모델을 생성합니다.
        # 아래 주석을 전환하여 Gemini와 GPT 간에 선택할 수 있습니다.
        self.llm = llm or LLMFactory.create_gemini(model_name="gemini-2.0-flash")
        # self.llm = llm or LLMFactory.create_openai(model_name="gpt-4o-mini")
    def remove_background(self, image: Image.Image) -> Image.Image:
        """
        이미지에서 배경을 제거합니다 (누끼 작업).
        
        Args:
            image: 원본 PIL Image
            
        Returns:
            배경이 제거된 투명 배경(RGBA)의 PIL Image
        """
        # 한국어 주석: rembg 라이브러리를 사용하여 이미지의 배경을 분리합니다.
        # numpy 배열로 변환하여 처리한 후 다시 PIL Image로 돌려줍니다.
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
        # 한국어 주석: 입력받은 대 카테고리가 유효한지 확인합니다.
        # '하의'로 들어올 경우 '바지'로 변환해주는 예외 처리를 추가할 수 있습니다.
        target_category = broad_category
        if target_category == "하의":
            target_category = "바지"
            
        if target_category not in self.CATEGORIES:
            raise ValueError(f"지원하지 않는 대 카테고리입니다: {broad_category}")

        sub_categories = self.CATEGORIES[target_category]
        
        # 한국어 주석: 이미지를 base64로 인코딩하여 LLM에 전달할 수 있는 데이터 URL 형식으로 만듭니다.
        buffered = BytesIO()
        image.convert("RGB").save(buffered, format="JPEG") # Vision 모델 전송용으로 RGB JPEG 권장
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        data_url = f"data:image/jpeg;base64,{img_str}"

        # 한국어 주석: 프롬프트를 구성하여 세부 카테고리 중 하나를 선택하도록 요청합니다.
        prompt = (
            f"당신은 패션 전문가입니다. 첨부된 이미지는 '{target_category}' 카테고리의 의류 사진입니다.\n"
            f"이 의류를 아래의 세부 카테고리 목록 중 가장 적절한 하나로 분류해 주세요:\n\n"
            f"세부 카테고리 목록: {', '.join(sub_categories)}\n\n"
            f"결과는 반드시 목록에 있는 이름 중 하나만 정확하게 답변해 주세요. 다른 설명은 생략하세요."
        )
        # 안전 설정 정의 (모든 필터를 끄는 설정)
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
        
        # 한국어 주석: LLM 응답에서 세부 카테고리 목록과 일치하는 항목을 찾습니다.
        # 정확히 일치하는 항목이 없을 경우, 응답이 카테고리명에 포함되거나 그 반대의 경우도 확인합니다.
        for sub in sub_categories:
            if sub in result or result in sub:
                return sub
        
        return result
