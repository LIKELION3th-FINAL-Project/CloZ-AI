from dataclasses import dataclass
import torch

@dataclass
class FashionConfig:
    """시스템 환경 설정 및 경로 관리 클래스"""
    base_dir: str = "/content/drive/MyDrive/LikeLion/최종 프로젝트 개인 임시/테스트용"
    db_path: str = "/content/fashion_items.db"
    chromadb_dir: str = "/content/drive/MyDrive/멋쟁이사자처럼/project_final/chromadb"
    model_name: str = "patrickjohncyh/fashion-clip"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_fp16: bool = True
