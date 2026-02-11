"""
피드백 파이프라인 설정

모든 경로 및 설정값을 중앙 관리
환경변수로 오버라이드 가능
"""

import os
from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()


def get_project_root() -> Path:
    """
    프로젝트 루트 디렉토리 반환
    
    src/feedback_pipeline/config.py 기준으로 상위 2단계
    """
    return Path(__file__).parent.parent.parent


@dataclass
class PathConfig:
    """경로 설정"""
    # 프로젝트 루트
    PROJECT_ROOT: Path = get_project_root()
    
    # 데이터 경로
    DATA_DIR: Path = None
    CHROMA_PATH: Path = None
    
    # 메타데이터 경로
    MUSINSA_METADATA_PATH: Path = None
    VISUAL_METADATA_PATH: Path = None
    
    # 사용자 데이터 경로
    USERS_DIR: Path = None
    
    def __post_init__(self):
        """환경변수 또는 기본값으로 초기화"""
        self.DATA_DIR = Path(os.getenv("DATA_DIR", self.PROJECT_ROOT / "data"))
        self.CHROMA_PATH = Path(os.getenv("CHROMA_PATH", self.DATA_DIR / "chroma_db"))
        self.MUSINSA_METADATA_PATH = Path(os.getenv(
            "MUSINSA_METADATA_PATH", 
            self.DATA_DIR / "musinsa_ranking_result.json"
        ))
        self.VISUAL_METADATA_PATH = Path(os.getenv(
            "VISUAL_METADATA_PATH",
            self.DATA_DIR / "visual_metadata_checkpoint.json"
        ))
        self.USERS_DIR = Path(os.getenv("USERS_DIR", self.DATA_DIR / "users"))


@dataclass
class EmbeddingConfig:
    """임베딩 설정"""
    # FashionCLIP 모델
    MODEL_NAME: str = "patrickjohncyh/fashion-clip"

    # 유사도 임계값
    WARDROBE_THRESHOLD: float = 0.2   # 옷장 검색

    # 배치 크기
    BATCH_SIZE: int = 32              # CPU 기준, GPU면 128 권장

    # ChromaDB 컬렉션 이름
    WARDROBE_COLLECTION: str = "wardrobe"
    PRODUCTS_COLLECTION: str = "products"  # 하바티 상품


@dataclass
class LLMConfig:
    """LLM 설정"""
    # API 키 (환경변수에서 로드)
    OPENAI_API_KEY: str = None
    GOOGLE_API_KEY: str = None
    OPENROUTER_API_KEY: str = None
    
    # 기본 모델
    DEFAULT_PROVIDER: str = "gemini"
    DEFAULT_MODEL: str = "gemini-2.0-flash"
    
    # 분석용 모델 (고품질)
    ANALYSIS_MODEL: str = "gpt-4o"
    
    def __post_init__(self):
        """환경변수에서 API 키 로드"""
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        self.OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")


@dataclass  
class PipelineConfig:
    """파이프라인 설정"""
    # 재생성 최대 횟수
    MAX_REGENERATE_COUNT: int = 1
    
    # 기능 활성화
    ENABLE_WARDROBE_CHECK: bool = True
    ENABLE_BUYING_RECOMMENDATION: bool = True
    
    # 프로필 업데이트
    MIN_MENTIONS_FOR_PROFILE: int = 3    # 3회 이상 언급 시 프로필 저장


# 전역 설정 인스턴스
paths = PathConfig()
embedding = EmbeddingConfig()
llm = LLMConfig()
pipeline = PipelineConfig()


def get_config():
    """
    전체 설정 반환
    
    사용 예시:
        from feedback_pipeline.config import get_config
        config = get_config()
        print(config['paths'].CHROMA_PATH)
    """
    return {
        'paths': paths,
        'embedding': embedding,
        'llm': llm,
        'pipeline': pipeline
    }
