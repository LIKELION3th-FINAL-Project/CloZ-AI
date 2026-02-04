from dataclasses import dataclass
import torch
import os

@dataclass
class FashionConfig:
    """시스템 환경 설정 및 로컬 경로 관리 클래스"""
    
    # 상위 경로 설정 (D:\final_pj\test)
    root_path: str = "D:/final_pj/test"
    
    # 1. 사용자 옷장 데이터 이미지 폴더 (yj_closet)
    base_dir: str = "D:/final_pj/test/yj_closet"
    
    # 2. SQLite DB 경로 (root_path 아래 생성)
    db_path: str = "D:/final_pj/test/fashion_items.db"
    
    # 3. 레퍼런스 스타일 임베딩 DB (chromadb-reference-embeddings)
    chromadb_ref_dir: str = "D:/final_pj/test/chromadb-reference-embeddings/chromadb"
    
    # 4. 사용자 옷장 임베딩 캐시 DB (chromadb_mycloset-embeddings)
    chromadb_war_dir: str = "D:/final_pj/test/chromadb_mycloset-embeddings/chromadb"
    
    # 5. 모델 정보 및 기타 경로
    model_name: str = "patrickjohncyh/fashion-clip"
    model_img_path: str = "D:/final_pj/test/model.webp"
    
    # 실행 환경 설정
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_fp16: bool = True