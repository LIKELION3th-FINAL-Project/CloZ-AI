from PIL import Image
from typing import Optional, Dict, Tuple, Union
from loguru import logger
# from fashn_vton import TryOnPipeline
from pathlib import Path
import os
from .sam3_processor import SAM3Processor

class VTONManager:
    """가상 피팅(Virtual Try-On) 실행 및 이미지 생성 클래스"""
    
    # VTON 하이퍼파라미터 기본값 (fashn-vton 1.5)
    DEFAULT_GUIDANCE_SCALE = 1.5  # classifier-free guidance 강도
    DEFAULT_NUM_TIMESTEPS = 30    # diffusion 샘플링 스텝 수
    DEFAULT_SEED = 42             # 재현용 랜덤 시드
    
    def __init__(self, guidance_scale: Union[float, None] = None, num_timesteps: Union[int, None] = None, seed: Union[int, None] = None, config: Union[Dict, None] = None):
        self.pipeline = None
        self.vton_weights_dir = self._resolve_weights_dir()
        
        # 하이퍼파라미터 설정
        self.guidance_scale = guidance_scale if guidance_scale is not None else self.DEFAULT_GUIDANCE_SCALE
        self.num_timesteps = num_timesteps if num_timesteps is not None else self.DEFAULT_NUM_TIMESTEPS
        self.seed = seed if seed is not None else self.DEFAULT_SEED
        
        # SAM3 프로세서 초기화
        self.sam3 = SAM3Processor(config=config)
        
        try:
            self.pipeline = TryOnPipeline(weights_dir=self.vton_weights_dir)
            logger.info("Fashion-VTON 파이프라인 로드 완료 (가중치는 재사용됩니다)")
        except ImportError as e:
            raise ImportError("fashn_vton 모듈을 찾을 수 없습니다. fashn-vton 설치가 필요합니다.") from e
        except Exception as e:
            raise RuntimeError(f"Fashion-VTON 로드 실패: {e}") from e

    def _resolve_weights_dir(self) -> Path:
        env_path = os.getenv("FASHN_VTON_WEIGHTS_DIR")
        candidates = []
        if env_path:
            candidates.append(Path(env_path).expanduser())

        project_root = Path(__file__).resolve().parents[3]
        candidates.extend(
            [
                project_root / "weights",
                Path.cwd() / "weights",
                Path(__file__).parents[1] / "fashn_vton" / "weights",
            ]
        )

        for candidate in candidates:
            if candidate.exists() and candidate.is_dir():
                return candidate

        searched = ", ".join(str(p) for p in candidates)
        raise FileNotFoundError(
            f"VTON weights 디렉토리를 찾을 수 없습니다. 확인한 경로: {searched}. "
            "환경변수 FASHN_VTON_WEIGHTS_DIR로 명시할 수 있습니다."
        )

    def try_on(self, person_img_path: str, outfit: Tuple, output_prefix: str, idx: int = 0) -> Optional[Dict]:
        """하의, 상의 순서로 가상 피팅 적용"""
        if not self.pipeline: 
            logger.warning("VTON 파이프라인이 로드되지 않았습니다.")
            return None
        
        # outfit: (pant, shirt) 순서
        pants, shirt = outfit
        person_img = Image.open(person_img_path).convert("RGB")
        
        logger.info(f"\n조합 #{idx + 1}")
        logger.info(f" - shirt: {shirt['path']}")
        logger.info(f" - pant : {pants['path']}")
        # logger.info(f" - outer: {outers['path']}")  # [원래 코드: outer 사용]


# ==================== [원래 코드 - outer 사용] ====================
# def try_on(self, person_img_path: str, outfit: Tuple, output_prefix: str, idx: int = 0) -> Optional[Dict]:
#     """상의, 하의 순서로 가상 피팅 적용"""
#     # outfit: (pant, outer, shirt) 순서라고 가정
#     pants, outers, shirt = outfit
#     person_img = Image.open(person_img_path).convert("RGB")
#     
#     logger.info(f"\n조합 #{idx + 1}")
#     logger.info(f" - shirt: {shirt['path']}")
#     logger.info(f" - pant : {pants['path']}")
#     # logger.info(f" - outer: {outers['path']}")
        
        try:
            # 하의 이미지 준비 (SAM3 적용)
            pants_img, pants_meta = self.sam3.get_optimal_garment_image(pants['path'], "bottoms")
            
            # 1. 하의 적용 (원본 이미지에 하의 합성)
            res_bottoms = self.pipeline(
                person_image=person_img, 
                garment_image=pants_img, 
                category="bottoms",
                guidance_scale=self.guidance_scale,
                num_timesteps=self.num_timesteps,
                seed=self.seed
            )
            bottom_path = f"{output_prefix}_q{idx}_bottom.png"
            res_bottoms.images[0].save(bottom_path)
            
            # 상의 이미지 준비 (SAM3 적용)
            shirt_img, shirt_meta = self.sam3.get_optimal_garment_image(shirt['path'], "tops")
            
            # 2. 상의 적용 (하의 합성 결과에 상의 합성)
            res_final = self.pipeline(
                person_image=res_bottoms.images[0], 
                garment_image=shirt_img, 
                category="tops",
                guidance_scale=self.guidance_scale,
                num_timesteps=self.num_timesteps,
                seed=self.seed
            )
            final_path = f"{output_prefix}_q{idx}_bottom_top.png"
            res_final.images[0].save(final_path)
            
            logger.info(f"Saved: {bottom_path}, {final_path}")
            
            return {
                "bottom_path": bottom_path,
                "final_path": final_path,
                "pants_meta": pants_meta,
                "shirt_meta": shirt_meta
            }
        except Exception as e:
            logger.error(f"VTON 이미지 생성 실패: {e}")
            return None
