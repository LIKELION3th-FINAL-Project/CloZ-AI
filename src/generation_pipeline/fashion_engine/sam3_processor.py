import os
import cv2
import torch
import numpy as np
from PIL import Image
from typing import Tuple, Dict, Union, List
from loguru import logger

class SAM3Processor:
    """SAM3 기반 의류 이미지 누끼 및 Fashion CLIP 검증 모듈"""
    
    # SAM3 설정 기본값
    DEFAULT_SAM3_ENABLED = True
    DEFAULT_SAM3_POINTS_PER_BATCH = 16
    DEFAULT_SAM3_MIN_AREA_RATIO = 0.04
    DEFAULT_SAM3_SCORE_THRESHOLD = 0.30
    DEFAULT_SAM3_MORPH_KERNEL = 5
    
    # Fashion CLIP 검증 기본값
    DEFAULT_FCLIP_ENABLED = True
    DEFAULT_FCLIP_THRESHOLD = 0.55
    DEFAULT_FCLIP_ALPHA_MAX = 0.85
    DEFAULT_FCLIP_ALPHA_MIN = 0.05
    
    def __init__(self, config: Union[Dict, None] = None):
        self.config = config or {}
        
        # SAM3 설정
        self.sam3_enabled = self.config.get("sam3_enabled", self.DEFAULT_SAM3_ENABLED)
        self.sam3_points_per_batch = self.config.get("sam3_points_per_batch", self.DEFAULT_SAM3_POINTS_PER_BATCH)
        self.sam3_min_area_ratio = self.config.get("sam3_min_area_ratio", self.DEFAULT_SAM3_MIN_AREA_RATIO)
        self.sam3_score_threshold = self.config.get("sam3_score_threshold", self.DEFAULT_SAM3_SCORE_THRESHOLD)
        self.sam3_morph_kernel = self.config.get("sam3_morph_kernel", self.DEFAULT_SAM3_MORPH_KERNEL)
        
        # Fashion CLIP 검증 설정
        self.fclip_enabled = self.config.get("fclip_verification_enabled", self.DEFAULT_FCLIP_ENABLED)
        self.fclip_threshold = self.config.get("fclip_threshold", self.DEFAULT_FCLIP_THRESHOLD)
        self.fclip_alpha_max = self.config.get("fclip_alpha_max_ratio", self.DEFAULT_FCLIP_ALPHA_MAX)
        self.fclip_alpha_min = self.config.get("fclip_alpha_min_ratio", self.DEFAULT_FCLIP_ALPHA_MIN)
        
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._sam3_generator = None
        self._fclip_model = None
        self._fclip_processor = None
        
        if self.sam3_enabled:
            self._init_sam3()
            
        if self.fclip_enabled:
            self._init_fclip()

    def _init_sam3(self):
        try:
            from transformers import pipeline
            hf_token = os.getenv("HF_TOKEN")
            if hf_token:
                from huggingface_hub import login
                login(token=hf_token)
                logger.info("HuggingFace 로그인 완료")
            
            self._sam3_generator = pipeline(
                "mask-generation",
                model="facebook/sam3",
                device=0 if self._device == "cuda" else -1,
            )
            logger.info(f"SAM3 모델 로드 완료 (device={self._device})")
        except Exception as e:
            logger.warning(f"SAM3 모델 로드 실패, 원본 이미지 사용: {e}")
            self.sam3_enabled = False

    def _init_fclip(self):
        try:
            from transformers import CLIPModel, CLIPProcessor
            model_name = "patrickjohncyh/fashion-clip"
            self._fclip_model = CLIPModel.from_pretrained(model_name).to(self._device)
            self._fclip_processor = CLIPProcessor.from_pretrained(model_name)
            logger.info(f"Fashion CLIP 모델 로드 완료 (device={self._device})")
        except Exception as e:
            logger.warning(f"Fashion CLIP 모델 로드 실패, 검증 비활성화: {e}")
            self.fclip_enabled = False

    def _to_binary_mask(self, mask_obj) -> np.ndarray:
        if torch.is_tensor(mask_obj):
            arr = mask_obj.detach().cpu().numpy()
        elif isinstance(mask_obj, Image.Image):
            arr = np.asarray(mask_obj)
        else:
            arr = np.asarray(mask_obj)

        arr = np.squeeze(arr)
        if arr.ndim == 3:
            arr = arr[..., 0]

        if arr.dtype == np.bool_:
            return arr.astype(np.uint8)
        if np.issubdtype(arr.dtype, np.floating):
            return (arr > 0).astype(np.uint8)

        thr = 0 if int(arr.max()) <= 1 else 127
        return (arr > thr).astype(np.uint8)

    def _resize_mask_to_image(self, mask: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
        th, tw = target_hw
        if mask.shape[:2] == (th, tw):
            return mask.astype(np.uint8)
        r = cv2.resize(mask.astype(np.uint8), (tw, th), interpolation=cv2.INTER_NEAREST)
        return (r > 0).astype(np.uint8)

    def _largest_cc(self, mask: np.ndarray) -> np.ndarray:
        if mask.sum() == 0:
            return mask
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        if n_labels <= 1:
            return mask
        areas = stats[1:, cv2.CC_STAT_AREA]
        best = 1 + int(np.argmax(areas))
        return (labels == best).astype(np.uint8)

    def _refine_mask(self, mask: np.ndarray) -> np.ndarray:
        k = np.ones((self.sam3_morph_kernel, self.sam3_morph_kernel), np.uint8)
        m = (mask > 0).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=1)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=1)
        m = self._largest_cc(m)
        return m

    def cutout_garment_with_sam3(self, img: Image.Image, category: str = "tops") -> Image.Image:
        if not self._sam3_generator:
            arr = np.array(img.convert("RGB"))
            alpha = np.full(arr.shape[:2], 255, dtype=np.uint8)
            return Image.fromarray(np.dstack([arr, alpha]))
        
        img_rgb = img.convert("RGB")
        img_np = np.asarray(img_rgb)
        H, W = img_np.shape[:2]
        img_area = H * W

        out = self._sam3_generator(img_rgb, points_per_batch=self.sam3_points_per_batch)
        outputs = out if isinstance(out, list) else [out]

        cands = []
        for item in outputs:
            if isinstance(item, dict) and "masks" in item:
                masks = item.get("masks", [])
                scores = item.get("scores", [1.0] * len(masks))

                for m, s in zip(masks, scores):
                    bm = self._to_binary_mask(m)
                    bm = self._resize_mask_to_image(bm, (H, W))
                    bm = self._refine_mask(bm)
                    area = int(bm.sum())

                    if area > int(self.sam3_min_area_ratio * img_area) and float(s) >= self.sam3_score_threshold:
                        cands.append((bm, float(s), area))

        if len(cands) == 0:
            alpha = np.full((H, W), 255, dtype=np.uint8)
            rgba = np.dstack([img_np, alpha])
            return Image.fromarray(rgba)

        cands.sort(key=lambda x: x[1] * max(1, x[2]), reverse=True)
        best = cands[0][0].astype(np.uint8)

        alpha = (best * 255).astype(np.uint8)
        rgba = np.dstack([img_np, alpha])
        return Image.fromarray(rgba)

    def _rgba_to_white_bg(self, img_rgba: Image.Image) -> Image.Image:
        rgba = np.asarray(img_rgba.convert('RGBA'))
        alpha = rgba[..., 3:4].astype(np.float32) / 255.0
        rgb = rgba[..., :3].astype(np.float32)
        white = np.ones_like(rgb) * 255.0
        out = rgb * alpha + white * (1.0 - alpha)
        return Image.fromarray(out.clip(0, 255).astype(np.uint8))

    def _clip_score(self, img_rgba: Image.Image, category: str) -> float:
        if not self._fclip_model or not self._fclip_processor:
            return 0.0
        
        if "top" in category.lower():
            pos_texts = ["a shirt on white background", "a top garment product photo"]
            neg_texts = ["a photo of a floor", "a photo of furniture", "a photo of a person wearing clothes"]
        else:
            pos_texts = ["pants on white background", "a bottom garment product photo"]
            neg_texts = ["a photo of a floor", "a photo of furniture", "a photo of a person wearing clothes"]
        
        all_texts = pos_texts + neg_texts
        img = self._rgba_to_white_bg(img_rgba) if img_rgba.mode == 'RGBA' else img_rgba.convert("RGB")

        inputs = self._fclip_processor(
            text=all_texts,
            images=img,
            return_tensors="pt",
            padding=True
        ).to(self._device)

        with torch.no_grad():
            outputs = self._fclip_model(**inputs)

        probs = outputs.logits_per_image.softmax(dim=1)[0]
        n_pos = len(pos_texts)
        pos_prob = probs[:n_pos].sum().item()
        neg_prob = probs[n_pos:].sum().item()
        return round(pos_prob / (pos_prob + neg_prob + 1e-8), 4)

    def verify_cutout(self, img_rgba: Image.Image, category: str) -> Tuple[bool, float, str]:
        arr = np.array(img_rgba.convert("RGBA"))
        alpha = arr[..., 3]
        fg_ratio = (alpha > 128).sum() / alpha.size

        if fg_ratio > self.fclip_alpha_max:
            return False, 0.0, f"alpha_too_full({fg_ratio:.2f})"

        if fg_ratio < self.fclip_alpha_min:
            return False, 0.0, f"alpha_too_empty({fg_ratio:.2f})"

        if self.fclip_enabled:
            score = self._clip_score(img_rgba, category)
            if score >= self.fclip_threshold:
                return True, score, "clip_ok"
            else:
                return False, score, f"clip_fail({score})"
        
        return True, 1.0, "alpha_ok"

    def rgba_to_black_matted_rgb(self, rgba_img: Image.Image) -> Image.Image:
        arr = np.asarray(rgba_img.convert('RGBA')).astype(np.float32)
        rgb = arr[..., :3]
        a = arr[..., 3:4] / 255.0
        out = rgb * a
        return Image.fromarray(out.clip(0, 255).astype(np.uint8))

    def get_optimal_garment_image(self, garment_path: str, category: str) -> Tuple[Image.Image, Dict]:
        raw_img = Image.open(garment_path).convert("RGB")
        
        if not self.sam3_enabled:
            return raw_img, {"used_cutout": False, "reason": "sam3_disabled"}
        
        cutout_rgba = self.cutout_garment_with_sam3(raw_img, category)
        is_valid, score, reason = self.verify_cutout(cutout_rgba, category)
        
        if is_valid:
            logger.info(f"  [{category}] 누끼 사용 (score={score:.3f}, reason={reason})")
            result_rgb = self.rgba_to_black_matted_rgb(cutout_rgba)
            return result_rgb, {"used_cutout": True, "score": score, "reason": reason}
        else:
            logger.info(f"  [{category}] 원본 fallback (score={score:.3f}, reason={reason})")
            return raw_img, {"used_cutout": False, "score": score, "reason": reason}
