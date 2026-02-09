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
        self.vton_weights_dir = Path(__file__).parents[1] / "fashn_vton" / "weights"
        try:
            self.pipeline = TryOnPipeline(weights_dir = self.vton_weights_dir)
            logger.info("✅ Fashion-VTON 파이프라인 로드 완료! (가중치는 재사용됩니다)")
        except ImportError:
            logger.warning("fashn_vton 모듈을 찾을 수 없습니다.")
        except Exception as e:
            logger.warning(f"Fashion-VTON 로드 실패: {e}")

    def try_on(self, person_img_path: str, outfit: Tuple, output_prefix: str, idx: int = 0) -> Optional[Dict]:
        """상의, 하위 순차적으로 가상 피팅 적용"""
        if not self.pipeline: 
            logger.warning("VTON 파이프라인이 로드되지 않았습니다.")
            return None
        
        # outfit: (pant, outer, shirt) 순서라고 가정
        pants, outers, shirt = outfit
        person_img = Image.open(person_img_path).convert("RGB")
        
        logger.info(f"\n🎯 조합 #{idx + 1}")
        logger.info(f" - shirt: {shirt['path']}")
        logger.info(f" - pant : {pants['path']}")
        logger.info(f" - outer: {outers['path']}")
        
        try:
            # 1. 상의 적용
            res_top = self.pipeline(
                person_image=person_img, 
                garment_image=Image.open(shirt['path']).convert("RGB"), 
                category="tops"
            )
            top_path = f"{output_prefix}_q{idx}_top.png"
            res_top.images[0].save(top_path)
            
            # 2. 하의 적용
            res_final = self.pipeline(
                person_image=res_top.images[0], 
                garment_image=Image.open(pants['path']).convert("RGB"), 
                category="bottoms"
            )
            final_path = f"{output_prefix}_q{idx}_top_bottom.png"
            res_final.images[0].save(final_path)
            
            logger.info(f"✅ Saved: {top_path}, {final_path}")
            
            return {
                "top_path": top_path,
                "final_path": final_path
            }
        except Exception as e:
            logger.error(f"VTON 이미지 생성 실패: {e}")
            return None
