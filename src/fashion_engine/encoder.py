import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from typing import Any
from .config import FashionConfig

class CLIPEncoder:
    """CLIP 모델을 이용한 이미지 및 텍스트 임베딩 추출 클래스"""
    def __init__(self, config: FashionConfig):
        self.config = config
        self.processor = CLIPProcessor.from_pretrained(config.model_name)
        self.model = CLIPModel.from_pretrained(config.model_name).to(config.device).eval()
        if config.use_fp16 and config.device == "cuda":
            self.model.half()

    @torch.no_grad()
    def _extract_features(self, outputs: Any) -> torch.Tensor:
        """모델 결과에서 공통적으로 특성 텐서를 추출하고 정규화함"""
        if hasattr(outputs, "image_embeds"):
            features = outputs.image_embeds
        elif hasattr(outputs, "text_embeds"):
            features = outputs.text_embeds
        elif hasattr(outputs, "pooler_output"):
            features = outputs.pooler_output
        else:
            features = outputs if isinstance(outputs, torch.Tensor) else outputs[0]

        if features.dim() == 3: features = features.mean(dim=1)
        if features.dim() == 2 and features.size(0) == 1: features = features[0]
        
        return F.normalize(features.float(), dim=-1).cpu()

    def encode_image(self, path: str) -> torch.Tensor:
        """이미지 파일을 읽어 임베딩 벡터 생성"""
        img = Image.open(path).convert("RGB")
        inputs = self.processor(images=img, return_tensors="pt").to(self.config.device)
        if self.config.use_fp16 and self.config.device == "cuda":
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)
        
        outputs = self.model.get_image_features(**inputs)
        return self._extract_features(outputs)

    def encode_text(self, text: str) -> torch.Tensor:
        """텍스트 쿼리를 임베딩 벡터 생성"""
        inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(self.config.device)
        outputs = self.model.get_text_features(**inputs)
        return self._extract_features(outputs)
