from dataclasses import dataclass
import torch

@dataclass
class FashionConfig:
    """시스템 환경 설정 및 경로 관리 클래스"""
    ## DB 위치 확인해서 수정 필요
    base_dir: str = "/content/drive/MyDrive/LikeLion/최종 프로젝트 개인 임시/테스트용"
    db_path: str = "/content/fashion_items.db"
    chromadb_dir: str = "/content/drive/MyDrive/멋쟁이사자처럼/project_final/chromadb"
    wardrobe_db_path: str = "/content/drive/MyDrive/멋쟁이사자처럼/project_final/chromadb_mycloset"
    model_name: str = "patrickjohncyh/fashion-clip"
    model_img_path: str = "/content/model.webp"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_fp16: bool = True



