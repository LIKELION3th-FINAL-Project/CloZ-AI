from PIL import Image
from typing import Optional, Dict, Tuple
from loguru import logger
from fashn_vton import TryOnPipeline
from pathlib import Path
import os

class VTONManager:
    """가상 피팅(Virtual Try-On) 실행 및 이미지 생성 클래스"""
    def __init__(self):
        self.pipeline = None
        self.vton_weights_dir = self._resolve_weights_dir()
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
        """상의, 하위 순차적으로 가상 피팅 적용"""
        if not self.pipeline: 
            logger.warning("VTON 파이프라인이 로드되지 않았습니다.")
            return None
        
        # outfit: (pant, outer, shirt) 순서라고 가정
        pants, outers, shirt = outfit
        person_img = Image.open(person_img_path).convert("RGB")
        
        logger.info(f"\n조합 #{idx + 1}")
        logger.info(f" - top   : {shirt['path']}")
        logger.info(f" - bottom: {pants['path']}")
        if outers:
            logger.info(f" - outer: {outers['path']}")
        else:
            logger.info(" - outer: (excluded)")
        
        try:
            # 상의 적용
            res_top = self.pipeline(
                person_image=person_img, 
                garment_image=Image.open(shirt['path']).convert("RGB"), 
                category="tops"
            )
            top_path = f"{output_prefix}_q{idx}_top.png"
            res_top.images[0].save(top_path)
            
            # 하의 적용
            res_final = self.pipeline(
                person_image=res_top.images[0], 
                garment_image=Image.open(pants['path']).convert("RGB"), 
                category="bottoms"
            )
            final_path = f"{output_prefix}_q{idx}_top_bottom.png"
            res_final.images[0].save(final_path)
            
            logger.info(f"Saved: {top_path}, {final_path}")
            
            return {
                "top_path": top_path,
                "final_path": final_path
            }
        except Exception as e:
            logger.error(f"VTON 이미지 생성 실패: {e}")
            return None
