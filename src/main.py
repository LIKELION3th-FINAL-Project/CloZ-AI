from fashion_engine import (
    CLIPEncoder, FashionDBManager, 
    FashionRecommender, OutfitPlanner, VTONManager, Visualizer
)
from utils.load import load_config
from understand_model.understand_model import UnderstandModel
from pathlib import Path
import torch
import unicodedata

if __name__ == "__main__":
    config_path = Path(__file__).parent / "configs" / "generation_model.yaml"
    understand_model = UnderstandModel()
    encoder = CLIPEncoder()
    recommender = FashionRecommender(encoder)
    planner = OutfitPlanner(encoder)
    vton = VTONManager()
    test_user_prompt = "오늘 홍대 가서 친구들이랑 놀건데 어떻게 입을까?"
    template_result = understand_model.chat(test_user_prompt)
    
    recommender.load_user_wardrobe()
    recommender.load_styles()
    
    cat_map_for_planner = {"상의": "shirt", "하의": "pant", "아우터": "outer"}
    for idx, test in enumerate(test_cases):
        logger.info("\n" + "=" * 60)
        logger.info(f"4. 쿼리 처리: {test.get('original_query', 'New Format Query')}")
        logger.info("=" * 60)
        
        # 1. 추천 아이템 검색
        recs_raw = recommender.recommend_from_agent(test, top_k=3)
        recs = {cat_map_for_planner.get(k, k): v for k, v in recs_raw.items()}
        
        if not recs: continue
        Visualizer.show_recommendations(recs, top_k=3)
        
        # 2. 조합 생성
        logger.info(f"\n5. 조합 생성 및 평가")
        combos = planner.generate_combinations(recs, top_n=3)
        
        if combos:
            # Context 기반 쿼리 임베딩 생성 로직 삽입
            context_parts = []
            if test.get("color", {}).get("value"):
                context_parts.append(", ".join(test["color"]["value"]))
            if test.get("mood", {}).get("value"):
                context_parts.append(", ".join(test["mood"]["value"]))
            if test.get("location", {}).get("value"):
                context_parts.append(", ".join(test["location"]["value"]))

            # 텍스트 조합 또는 원본 쿼리 사용
            context_text = " ".join(context_parts) if context_parts else test.get("original_query", "fashion outfit")
            
            # Planner 평가용 임베딩 생성
            avg_q_emb = encoder.encode_text(context_text).to(torch.float32)
            avg_q_emb /= (avg_q_emb.norm() + 1e-8)

            # 스타일 명칭 추출
            target_style = test.get("style", {}).get("value", ["캐주얼"])[0]

            logger.info("------------------------------------------------")
            logger.info(target_style)
            logger.info("------------------------------------------------")

            # 3. 평가 호출 (생성한 avg_q_emb 전달)
            best_outfits = planner.evaluate_outfits(
                combos, 
                recommender.style_profiles, 
                target_style,
                query_embedding=avg_q_emb, # context 임베딩 반영
                overall_weight=0.7
            )
            
            if best_outfits:
                Visualizer.show_top_combinations(best_outfits, num_to_show=3)
                vton.try_on(config.model_img_path, best_outfits[0]['combination'], f"output_q{idx}", idx)
    
    logger.info("\n✅ 파이프라인 완료!")