from fashion_engine import (
    CLIPEncoder, FashionDBManager, 
    FashionRecommender, OutfitPlanner, VTONManager, Visualizer
)
from utils.load import load_config, load_json
from understand_model.understand_model import UnderstandModel, extract_json_format, make_json_file
from pathlib import Path
from loguru import logger
import torch
import unicodedata

if __name__ == "__main__":
    config_root_path = Path(__file__).parents[1] / "configs" / "generation_model.yaml"
    result_root_path = Path(__file__).parents[1] / "results" / "model_response.json"
    generation_config_file = load_config(config_root_path)
    item_top_k = generation_config_file["item_top_k"]
    combination_top_k = generation_config_file["combination_top_k"]
    num_of_show = generation_config_file["num_of_show"]
    cat_map_planner = generation_config_file["cat_to_db"]
    
    understand_model = UnderstandModel()
    encoder = CLIPEncoder()
    recommender = FashionRecommender(encoder)
    planner = OutfitPlanner(encoder)
    vton = VTONManager()
    
    test_user_prompt = "오늘 홍대 가서 친구들이랑 놀건데 어떻게 입을까?"
    model_result = understand_model.chat(test_user_prompt)
    model_result_json = extract_json_format(model_result) # 문자열 -> 딕셔너리 포맷만 추출
    # make_json_file(model_result_json) # 문자열(딕셔너리 포맷) -> 딕셔너리로 변환
    model_resp_json_file = load_json(result_root_path)
    
    recommender.load_user_wardrobe()
    recommender.load_styles()
    
    recs_raw = recommender.recommend_from_agent(model_result_json, top_k = item_top_k)
    
    if not recs_raw:
        Visualizer.show_
    
    
    # # user prompt 하나 당
    # for idx, test in enumerate(model_resp_json_file):
    #     logger.info("\n" + "=" * 60)
    #     logger.info(f"User Prompt: {test_user_prompt}")
    #     logger.info("=" * 60)
        
    #     # 1. 추천 아이템 검색
    #     recs_raw = recommender.recommend_from_agent(test, top_k = item_top_k)
    #     recs = {cat_map_planner.get(k, k): v for k, v in recs_raw.items()} # key 이름 재구성
        
    #     if not recs: 
    #         continue
    #     Visualizer.show_recommendations(recs, top_k = item_top_k)
        
    #     # 2. 조합 생성
    #     logger.info(f"조합 생성 및 평가")
    #     combos = planner.generate_combinations(recs, top_n = combination_top_k)
        
    #     if combos:
    #         # Context 기반 쿼리 임베딩 생성 로직 삽입
    #         context_parts = []
    #         if test.get("color", {}).get("value"):
    #             context_parts.append(", ".join(test["color"]["value"]))
    #         if test.get("mood", {}).get("value"):
    #             context_parts.append(", ".join(test["mood"]["value"]))
    #         if test.get("location", {}).get("value"):
    #             context_parts.append(", ".join(test["location"]["value"]))

    #         # 텍스트 조합 또는 원본 쿼리 사용
    #         context_text = " ".join(context_parts) if context_parts else test.get("original_query", "fashion outfit")
            
    #         # Planner 평가용 임베딩 생성
    #         avg_q_emb = encoder.encode_text(context_text).to(torch.float32)
    #         avg_q_emb /= (avg_q_emb.norm() + 1e-8)

    #         # 스타일 명칭 추출
    #         target_style = test.get("style", {}).get("value", ["캐주얼"])[0]

    #         logger.info("------------------------------------------------")
    #         logger.info(target_style)
    #         logger.info("------------------------------------------------")

    #         # 3. 평가 호출 (생성한 avg_q_emb 전달)
    #         best_outfits = planner.evaluate_outfits(
    #             combos, 
    #             recommender.style_profiles, 
    #             target_style,
    #             query_embedding=avg_q_emb, # context 임베딩 반영
    #             overall_weight=0.7
    #         )
            
    #         if best_outfits:
    #             Visualizer.show_top_combinations(best_outfits, num_to_show = num_of_show)
    #             vton.try_on(config_root_path["chromadb_ref_embedding_dir"], best_outfits[0]['combination'], f"output_q{idx}", idx)
    
    # logger.info("\n✅ 파이프라인 완료!")