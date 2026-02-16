"""
CloZ-AI 통합 설정

생성 파이프라인(yaml)과 피드백 파이프라인(dataclass) 설정을 통합 관리.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any

from src.generation_pipeline.utils.load import load_config
from src.feedback_pipeline.config import (
    PathConfig, EmbeddingConfig, LLMConfig, PipelineConfig,
    get_project_root
)


@dataclass
class GenerationConfig:
    """생성 파이프라인 설정 (configs/generation_model.yaml 기반)"""

    # 추천 관련
    item_top_k: int = 3
    combination_top_k: int = 3
    num_of_show: int = 3

    # 추천 가중치
    text_sim_weight: float = 0.2
    style_sim_weight: float = 0.5
    color_sim_weight: float = 0.15
    season_sim_weight: float = 0.1
    mood_sim_weight: float = 0.05

    # 레거시 가중치 (구 형식)
    legacy_text_weight: float = 0.3
    legacy_style_weight: float = 0.7

    # 평가 가중치
    overall_weight: float = 0.7

    # 모델
    model_name: str = "patrickjohncyh/fashion-clip"
    use_fp16: bool = True

    # 경로 (yaml에서 로드)
    user_clothes_dir: str = ""
    chromadb_user_war_embedding_dir: str = ""
    chromadb_ref_embedding_dir: str = ""
    user_body_image: str = ""

    # 카테고리 매핑
    folder_map: Dict[str, str] = field(default_factory=dict)
    cat_to_db: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, yaml_path: str = None) -> "GenerationConfig":
        """yaml 파일에서 설정 로드"""
        if yaml_path is None:
            yaml_path = get_project_root() / "configs" / "generation_model.yaml"

        raw = load_config(str(yaml_path))

        return cls(
            item_top_k=raw.get("item_top_k", 3),
            combination_top_k=raw.get("combination_top_k", 3),
            num_of_show=raw.get("num_of_show", 3),
            model_name=raw.get("model_name", "patrickjohncyh/fashion-clip"),
            use_fp16=raw.get("use_fp16", True),
            user_clothes_dir=raw.get("user_clothes_dir", ""),
            chromadb_user_war_embedding_dir=raw.get("chromadb_user_war_embedding_dir", ""),
            chromadb_ref_embedding_dir=raw.get("chromadb_ref_embedding_dir", ""),
            user_body_image=raw.get("user_body_image", ""),
            folder_map=raw.get("folder_map", {}),
            cat_to_db=raw.get("cat_to_db", {}),
        )


@dataclass
class UnifiedConfig:
    """통합 설정"""
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)

    @classmethod
    def load(cls, generation_yaml_path: str = None) -> "UnifiedConfig":
        """전체 설정 로드"""
        return cls(
            generation=GenerationConfig.from_yaml(generation_yaml_path),
            paths=PathConfig(),
            embedding=EmbeddingConfig(),
            llm=LLMConfig(),
            pipeline=PipelineConfig(),
        )


# 전역 설정 인스턴스 (lazy init)
_unified_config = None


def get_unified_config(generation_yaml_path: str = None) -> UnifiedConfig:
    """통합 설정 싱글톤"""
    global _unified_config
    if _unified_config is None:
        _unified_config = UnifiedConfig.load(generation_yaml_path)
    return _unified_config
