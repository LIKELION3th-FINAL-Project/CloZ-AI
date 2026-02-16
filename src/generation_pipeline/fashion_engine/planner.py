import itertools
import unicodedata
import torch
import torch.nn.functional as F
import numpy as np
import os
from typing import Dict, List, Tuple
from loguru import logger
from .encoder import CLIPEncoder

class OutfitPlanner:
    """아이템 조합 생성 및 세트 코디네이션 평가 클래스"""
    def __init__(self, encoder: CLIPEncoder):
        self.encoder = encoder

    def generate_combinations(
        self,
        recommendations: Dict[str, List[Dict]],
        top_n: int = 3,
        include_outer: bool = True,
    ) -> List[Tuple]:
        """카테고리별 상위 아이템들로 가능한 모든 조합 생성"""
        pants = recommendations.get("pant", [])[:top_n]
        outers = recommendations.get("outer", [])[:top_n] if include_outer else [None]
        shirts = recommendations.get("shirt", [])[:top_n]
        
        if include_outer and not (pants and outers and shirts):
            logger.warning("조합 생성에 필요한 카테고리가 부족합니다.")
            return []
        if not include_outer and not (pants and shirts):
            logger.warning("조합 생성에 필요한 카테고리(상의/하의)가 부족합니다.")
            return []
        
        combinations = list(itertools.product(pants, outers, shirts))
        logger.info(f"총 {len(combinations)}개의 조합이 생성되었습니다.")
        logger.info(f"- bottom: {len(pants)}개")
        logger.info(f"- outer: {len(outers)}개")
        logger.info(f"- top: {len(shirts)}개")
        
        return combinations

    def evaluate_outfits(self, combinations: List[Tuple], style_profiles: Dict[str, torch.Tensor], 
                        target_style: str, query_embedding: torch.Tensor, overall_weight: float = 0.7) -> List[Dict]:
        """조합된 세트의 전체 조화도 및 스타일 부합도 평가 (가중치 파라미터 추가)"""
        target_style = unicodedata.normalize("NFC", target_style)
        if target_style not in style_profiles:
            if not style_profiles:
                logger.warning("스타일 프로필이 비어 있습니다.")
                return []
            fallback_style = next(iter(style_profiles.keys()))
            logger.warning(
                f"타겟 스타일 '{target_style}'이 프로필에 없습니다. '{fallback_style}' 스타일로 대체합니다."
            )
            target_style = fallback_style
        
        # 스타일 레퍼런스 정규화
        style_embs = F.normalize(style_profiles[target_style].to(torch.float32), dim=-1)
        results = []
        
        logger.info(f"총 {len(combinations)}개 조합 평가 중...")
        
        for i, combo in enumerate(combinations):
            if (i + 1) % 5 == 0:
                logger.info(f"  진행: {i + 1}/{len(combinations)}")
            
            # 3장 평균 임베딩 (전체 조화)
            valid_items = [item for item in combo if item is not None]
            embs = torch.stack([item['embedding'] for item in valid_items]).to(torch.float32)
            combo_emb = F.normalize(embs.mean(dim=0), dim=-1)
            harmony_sim = torch.max(combo_emb @ style_embs.T).item()
            
            # 개별 아이템 vs 스타일 평균 (개별 품질)
            indiv_sims = []
            for item in valid_items:
                item_emb = F.normalize(item['embedding'].to(torch.float32), dim=-1)
                sim = torch.max(item_emb @ style_embs.T).item()
                indiv_sims.append(sim)
            indiv_avg = np.mean(indiv_sims)
            
            # 최종 점수 (overall_weight 파라미터 적용)
            final_score = overall_weight * harmony_sim + (1 - overall_weight) * indiv_avg
            
            results.append({
                "combination": combo,
                "combination_idx": i,
                "final_score": final_score,
                "harmony_score": harmony_sim,
                "individual_avg": indiv_avg,
                "pant": combo[0]['path'].split('/')[-1],
                "outer": combo[1]['path'].split('/')[-1] if combo[1] else None,
                "shirt": combo[2]['path'].split('/')[-1],
                "bottom": combo[0]['path'].split('/')[-1],
                "top": combo[2]['path'].split('/')[-1],
            })
            
        results.sort(key=lambda x: x['final_score'], reverse=True)
        
        # 상위 결과 출력
        logger.info(f"\n평가 완료! 상위 3개 조합:")
        for rank, result in enumerate(results[:3], 1):
            logger.info(f"\n{rank}위 (조합 #{result['combination_idx'] + 1}):")
            logger.info(f"  최종 점수: {result['final_score']:.4f}")
            logger.info(f"  전체 조화 유사도: {result['harmony_score']:.4f}")
            logger.info(f"  개별 평균 유사도: {result['individual_avg']:.4f}")
            logger.info(f"  - bottom: {result['pant']}")
            logger.info(f"  - outer: {result['outer']}")
            logger.info(f"  - top: {result['shirt']}")
        
        return results
