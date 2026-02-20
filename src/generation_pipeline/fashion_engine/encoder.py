import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from typing import Any
from pathlib import Path
from ..utils.load import load_config

class CLIPEncoder:
    """CLIP 모델을 이용한 이미지 및 텍스트 임베딩 추출 클래스"""
    def __init__(self):
        self.config_path = Path(__file__).resolve().parents[3] / "configs" / "generation_model.yaml"
        self.config_file = load_config(self.config_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = CLIPProcessor.from_pretrained(self.config_file["model_name"])
        self.model = CLIPModel.from_pretrained(self.config_file["model_name"]).to(self.device).eval()
        self.use_fp16 = self.config_file["use_fp16"]
        self.model.half() if self.use_fp16 and self.device == "cuda" else self.model.float()

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
        
        return F.normalize(features.float(), dim=-1)

    def encode_image(self, image_path: str) -> torch.Tensor:
        """이미지 파일을 읽어 임베딩 벡터 생성"""
        img = Image.open(image_path).convert("RGB")
        inputs = self.processor(images = img, return_tensors = "pt").to(self.device)
        if self.config_file["use_fp16"] and self.device == "cuda":
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)
        
        outputs = self.model.get_image_features(**inputs)
        # return outputs
        return self._extract_features(outputs)

    def encode_text(self, text: str) -> torch.Tensor:
        """텍스트 쿼리를 임베딩 벡터 생성"""
        inputs = self.processor(text = [text], return_tensors = "pt", padding = True).to(self.device)
        outputs = self.model.get_text_features(**inputs)
        # return outputs
        return self._extract_features(outputs)
