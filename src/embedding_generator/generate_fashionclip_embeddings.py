"""
FashionCLIP 임베딩 생성 스크립트 (Colab GPU 전용)

옷장 이미지를 fashion-clip으로 임베딩

사용법:
- Colab에서 이 파일 업로드
- 데이터 폴더 업로드 (closet/)
- 실행

출력:
- wardrobe_embeddings.json
- chroma/ (ChromaDB)
"""

# ============================================================
# Colab 환경 설정 (첫 실행 시 주석 해제)
# ============================================================
# !pip install -q transformers torch chromadb pillow tqdm

import subprocess
import sys

def install_packages():
    """필수 패키지 설치"""
    packages = ["transformers", "torch", "chromadb", "pillow", "tqdm"]
    for pkg in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

# 패키지 설치 (필요시)
try:
    import transformers
except ImportError:
    print("패키지 설치 중...")
    install_packages()

import os
import json
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import chromadb



# FashionCLIP 모델 로드 (A100 최적화)
class FashionCLIPEmbedder:
    """
    FashionCLIP 임베딩 생성기 (A100 80GB 최적화)

    - 이미지 → 512차원 벡터
    - 텍스트 → 512차원 벡터
    - FP16 사용으로 속도 2배, 메모리 절반
    - 대용량 배치 처리
    """

    def __init__(self, device: str = None, use_fp16: bool = True):
        """
        Args:
            device: "cuda", "mps", 또는 "cpu" (None이면 자동 감지)
            use_fp16: FP16 사용 여부 (CUDA에서만 지원)
        """
        # 디바이스 자동 감지 (CUDA > MPS > CPU)
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        # FP16은 CUDA에서만 사용 (MPS/CPU는 FP32)
        self.use_fp16 = use_fp16 and self.device == "cuda"

        print(f"[FashionCLIP] 디바이스: {self.device}")
        print(f"[FashionCLIP] FP16: {self.use_fp16}")

        # GPU 메모리 확인
        if self.device == "cuda":
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"[FashionCLIP] GPU 메모리: {gpu_mem:.1f}GB")
        elif self.device == "mps":
            print("[FashionCLIP] Apple Silicon MPS 사용")

        # 모델 로드
        self.model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip")
        self.processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")

        # FP16 변환 (A100에서 2배 빠름)
        if self.use_fp16:
            self.model = self.model.half()

        self.model.to(self.device)
        self.model.eval()

        # torch.compile (PyTorch 2.0+, 추가 속도 향상)
        try:
            if hasattr(torch, 'compile') and self.device == "cuda":
                self.model = torch.compile(self.model, mode="reduce-overhead")
                print("[FashionCLIP] torch.compile 적용됨")
        except Exception as e:
            print(f"[FashionCLIP] torch.compile 스킵: {e}")

        print("[FashionCLIP] 모델 로드 완료")

    def embed_image(self, image_path: str) -> List[float]:
        """단일 이미지 임베딩 (배치 처리 권장)"""
        try:
            image = Image.open(image_path).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            if self.use_fp16:
                inputs = {k: v.half() if v.dtype == torch.float32 else v for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.get_image_features(**inputs)

                # 텐서 또는 객체 처리
                if hasattr(outputs, 'pooler_output'):
                    features = outputs.pooler_output
                elif hasattr(outputs, 'last_hidden_state'):
                    features = outputs.last_hidden_state[:, 0, :]
                else:
                    features = outputs  # 이미 텐서

            # 정규화
            features = F.normalize(features, p=2, dim=-1)
            return features[0].float().cpu().numpy().tolist()
        except Exception as e:
            print(f"[오류] 이미지 임베딩 실패: {image_path} - {e}")
            return None

    def embed_text(self, text: str) -> List[float]:
        """텍스트 임베딩"""
        inputs = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.get_text_features(**inputs)

            if hasattr(outputs, 'pooler_output'):
                features = outputs.pooler_output
            elif hasattr(outputs, 'last_hidden_state'):
                features = outputs.last_hidden_state[:, 0, :]
            else:
                features = outputs

        features = F.normalize(features, p=2, dim=-1)
        return features[0].float().cpu().numpy().tolist()

    def embed_images_batch(
        self,
        image_paths: List[str],
        batch_size: int = 128  # A100 80GB: 128~256 권장
    ) -> List[List[float]]:
        """
        배치로 이미지 임베딩 (A100 최적화)

        Args:
            image_paths: 이미지 경로 리스트
            batch_size: 배치 크기 (A100 80GB: 128~256 권장)

        Returns:
            임베딩 리스트 (실패 시 None)
        """
        embeddings = []

        for i in tqdm(range(0, len(image_paths), batch_size), desc="배치 임베딩"):
            batch_paths = image_paths[i:i+batch_size]
            batch_images = []
            valid_indices = []

            # 이미지 로드
            for idx, path in enumerate(batch_paths):
                try:
                    img = Image.open(path).convert("RGB")
                    batch_images.append(img)
                    valid_indices.append(i + idx)
                except Exception as e:
                    embeddings.append(None)

            if not batch_images:
                continue

            # 배치 처리
            inputs = self.processor(images=batch_images, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            if self.use_fp16:
                inputs = {k: v.half() if v.dtype == torch.float32 else v for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.get_image_features(**inputs)

                if hasattr(outputs, 'pooler_output'):
                    features = outputs.pooler_output
                elif hasattr(outputs, 'last_hidden_state'):
                    features = outputs.last_hidden_state[:, 0, :]
                else:
                    features = outputs

            features = F.normalize(features, p=2, dim=-1)

            for feat in features:
                embeddings.append(feat.float().cpu().numpy().tolist())

        return embeddings




# 옷장 이미지 임베딩 생성
def generate_wardrobe_embeddings(
    closet_dir: str,
    output_dir: str,
    embedder: FashionCLIPEmbedder,
    checkpoint_interval: int = 50  # 중간 저장 간격
):
    """
    옷장 이미지 임베딩 생성 (중간 저장 지원)

    Args:
        closet_dir: 옷장 이미지 디렉토리
        output_dir: 출력 디렉토리
        checkpoint_interval: 중간 저장 간격 (기본 50개마다)
    """
    print("\n옷장 임베딩 생성 시작")

    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, "wardrobe_checkpoint.json")
    output_path = os.path.join(output_dir, "wardrobe_embeddings.json")

    # 체크포인트 로드
    results = []
    processed_ids = set()
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            results = json.load(f)
            processed_ids = {r["id"] for r in results}
        print(f"체크포인트 로드: {len(results)}개 이미 처리됨")

    # 이미지 파일 수집
    image_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    image_files = []

    for root, _, files in os.walk(closet_dir):
        for file in files:
            if Path(file).suffix.lower() in image_extensions:
                image_files.append(os.path.join(root, file))

    print(f"총 {len(image_files)}개 이미지 처리 (남은: {len(image_files) - len(processed_ids)}개)")

    # 임베딩 생성
    new_count = 0

    for image_path in tqdm(image_files, desc="옷장 임베딩"):
        relative_path = os.path.relpath(image_path, closet_dir)
        item_id = f"item:{relative_path}"

        # 이미 처리된 것 스킵
        if item_id in processed_ids:
            continue

        # broad_cat 추출
        parts = relative_path.split(os.sep)
        broad_cat = parts[0] if parts else "unknown"

        # 임베딩 생성
        embedding = embedder.embed_image(image_path)

        if embedding:
            results.append({
                "id": item_id,
                "embedding": embedding,
                "metadata": {
                    "entity_type": "item",
                    "item_key": relative_path,
                    "broad_cat": broad_cat,
                    "detail_cat": None,
                    "source": "wardrobe"
                }
            })
            new_count += 1

            # 중간 저장
            if new_count % checkpoint_interval == 0:
                with open(checkpoint_path, "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False)
                print(f"\n체크포인트 저장: {len(results)}개")

    # 최종 저장
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 체크포인트 삭제
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    print(f"옷장 임베딩 저장 완료: {output_path} ({len(results)}개)")

    return results
