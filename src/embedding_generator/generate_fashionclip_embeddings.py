"""
FashionCLIP 임베딩 생성 스크립트 (Colab GPU 전용)

무신사 상품 이미지 & 옷장 이미지를 fashion-clip으로 임베딩

사용법:
1. Colab에서 이 파일 업로드
2. 데이터 폴더 업로드 (musinsa_images/, closet/, musinsa_ranking_result.json)
3. 실행

출력:
- musinsa_embeddings.json
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



# ============================================================
# 2. FashionCLIP 모델 로드 (A100 최적화)
# ============================================================
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




# ============================================================
# 3. 무신사 상품 임베딩 생성
# ============================================================
def generate_musinsa_embeddings(
    json_path: str,
    images_base_dir: str,
    output_dir: str,
    embedder: FashionCLIPEmbedder,
    checkpoint_interval: int = 100  # 중간 저장 간격
):
    """
    무신사 상품 이미지 임베딩 생성 (중간 저장 지원)
    
    Args:
        json_path: musinsa_ranking_result.json 경로
        images_base_dir: 이미지 베이스 디렉토리
        output_dir: 출력 디렉토리
        checkpoint_interval: 중간 저장 간격 (기본 100개마다)
    """
    print("\n무신사 상품 임베딩 생성 시작")
    
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, "musinsa_checkpoint.json")
    output_path = os.path.join(output_dir, "musinsa_embeddings.json")
    
    # 체크포인트 로드 (이어서 실행)
    results = []
    processed_ids = set()
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            results = json.load(f)
            processed_ids = {r["id"] for r in results}
        print(f"체크포인트 로드: {len(results)}개 이미 처리됨")
    
    # JSON 로드
    with open(json_path, "r", encoding="utf-8") as f:
        products = json.load(f)
    
    print(f"총 {len(products)}개 상품 처리 (남은: {len(products) - len(processed_ids)}개)")
    
    # 임베딩 생성
    new_count = 0
    
    for product in tqdm(products, desc="무신사 임베딩"):
        product_id = f"musinsa:{product['id']}"
        
        # 이미 처리된 것 스킵
        if product_id in processed_ids:
            continue
        
        # 이미지 경로 변환
        original_path = product.get("product_image_path", "")
        if "\\" in original_path:
            parts = original_path.split("\\")
            relative_path = "/".join(parts[-3:])
        else:
            parts = original_path.split("/")
            relative_path = "/".join(parts[-3:])
        
        image_path = os.path.join(images_base_dir, relative_path)
        
        if not os.path.exists(image_path):
            continue
        
        # 임베딩 생성
        embedding = embedder.embed_image(image_path)
        
        if embedding:
            results.append({
                "id": product_id,
                "embedding": embedding,
                "metadata": {
                    "product_id": product["id"],
                    "category_main": product.get("category_main", ""),
                    "category_sub": product.get("category_sub", ""),
                    "brand": product.get("brand", ""),
                    "product_name": product.get("product_name", ""),
                    "price": int(product.get("price", 0)),
                    "rating": float(product.get("rating", 0)),
                    "reviews": int(product.get("reviews", 0)),
                    "favorite": int(product.get("favorite", 0)),
                    "product_url": product.get("product_url", ""),
                    "image_path": relative_path
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
    
    print(f"무신사 임베딩 저장 완료: {output_path} ({len(results)}개)")
    
    return results



# ============================================================
# 4. 옷장 이미지 임베딩 생성
# ============================================================
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




# ============================================================
# 5. ChromaDB에 저장
# ============================================================
def save_to_chromadb(
    embeddings: List[Dict],
    collection_name: str,
    chroma_path: str
):
    """
    임베딩을 ChromaDB에 저장
    
    Args:
        embeddings: 임베딩 리스트 [{id, embedding, metadata}, ...]
        collection_name: 컬렉션 이름 ("wardrobe" 또는 "musinsa")
        chroma_path: ChromaDB 저장 경로
    """
    print(f"\nChromaDB 저장: {collection_name}")
    
    # ChromaDB 클라이언트
    client = chromadb.PersistentClient(path=chroma_path)
    
    # 기존 컬렉션 삭제 후 재생성
    try:
        client.delete_collection(collection_name)
    except:
        pass
    
    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}  # 코사인 유사도
    )
    
    # 배치 삽입
    batch_size = 1000
    for i in tqdm(range(0, len(embeddings), batch_size), desc="ChromaDB 저장"):
        batch = embeddings[i:i+batch_size]
        
        collection.add(
            ids=[item["id"] for item in batch],
            embeddings=[item["embedding"] for item in batch],
            metadatas=[item["metadata"] for item in batch]
        )
    
    print(f"ChromaDB 저장 완료: {collection_name} ({len(embeddings)}개)")


# ============================================================
# 6. 메인 실행
# ============================================================
if __name__ == "__main__":
    """
    Colab 사용법:
    1. 데이터 폴더 업로드 (musinsa_images/, closet/, musinsa_ranking_result.json)
    2. 아래 경로를 Colab 환경에 맞게 수정
    3. 실행: python generate_fashionclip_embeddings.py
    """
    
    # ========== 경로 설정 ==========
    # Colab에 데이터 업로드 후 아래 경로 확인/수정
    # 예: 드라이브 마운트 시 /content/drive/MyDrive/... 로 변경
    
    MUSINSA_JSON = "/content/drive/Othercomputers/내 MacBook Pro/Downloads/whats-in-my-closet/data/musinsa_ranking_result.json"    # 무신사 상품 JSON
    MUSINSA_IMAGES = "/content/drive/Othercomputers/내 MacBook Pro/Downloads/whats-in-my-closet/data/musinsa_images"               # 무신사 이미지 폴더
    CLOSET_DIR = "/content/drive/Othercomputers/내 MacBook Pro/Downloads/whats-in-my-closet/closet"                           # 옷장 이미지 폴더
    
    OUTPUT_DIR = "/content/drive/Othercomputers/내 MacBook Pro/Downloads/whats-in-my-closet/data/embeddings_output"                # 임베딩 JSON 출력
    CHROMA_PATH = "/content/drive/Othercomputers/내 MacBook Pro/Downloads/whats-in-my-closet/data/chroma_db"                       # ChromaDB 저장 경로                   # ChromaDB 저장 경로
    # ================================

    
    print("\n" + "="*60)
    print("FashionCLIP 임베딩 생성 시작")
    print("="*60)
    
    # 임베더 초기화 (GPU 사용)
    embedder = FashionCLIPEmbedder()
    
    # 1. 무신사 임베딩 생성
    if os.path.exists(MUSINSA_JSON) and os.path.exists(MUSINSA_IMAGES):
        musinsa_embeddings = generate_musinsa_embeddings(
            json_path=MUSINSA_JSON,
            images_base_dir=MUSINSA_IMAGES,
            output_dir=OUTPUT_DIR,
            embedder=embedder
        )
        save_to_chromadb(musinsa_embeddings, "musinsa", CHROMA_PATH)
    else:
        print("무신사 데이터 없음 - 스킵")
        musinsa_embeddings = []
    
    # 2. 옷장 임베딩 생성
    if os.path.exists(CLOSET_DIR):
        wardrobe_embeddings = generate_wardrobe_embeddings(
            closet_dir=CLOSET_DIR,
            output_dir=OUTPUT_DIR,
            embedder=embedder
        )
        save_to_chromadb(wardrobe_embeddings, "wardrobe", CHROMA_PATH)
    else:
        print("옷장 데이터 없음 - 스킵")
        wardrobe_embeddings = []
    
    # 결과 요약
    print("\n" + "="*60)
    print("임베딩 생성 완료!")
    print("="*60)
    print(f"  무신사: {len(musinsa_embeddings)}개")
    print(f"  옷장:   {len(wardrobe_embeddings)}개")
    print(f"\n  출력:")
    print(f"    - {OUTPUT_DIR}/musinsa_embeddings.json")
    print(f"    - {OUTPUT_DIR}/wardrobe_embeddings.json")
    print(f"    - {CHROMA_PATH}/")
