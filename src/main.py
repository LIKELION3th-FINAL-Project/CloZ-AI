from fashion_engine import (
    CLIPEncoder, FashionDBManager, 
    FashionRecommender, OutfitPlanner, VTONManager, Visualizer
)
from utils.load import load_config
import torch
import unicodedata
"""
def normalize_korean(text):
    
    #유니코드 정규화 (NFD)를 통해 한글 자모 분리

    return ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    )
"""

def run_fashion_system():
    """전체 패션 추천 및 코디 파이프라인 시뮬레이션 코드 (Context 기반 평가 적용)"""
    print("=" * 60)
    print("1. Fashion System 초기화")
    print("=" * 60)
    
    config = load_config("../configs/generation_model.yaml")
    encoder = CLIPEncoder()
    recommender = FashionRecommender(encoder)
    planner = OutfitPlanner(encoder)
    vton = VTONManager()
    
    recommender.load_user_wardrobe()
    recommender.load_styles()
    
    # [수정] 새로운 필드 기반 테스트 케이스 (필요 시)
    # 테스트 케이스 (새로운 구조화된 query 형식)
    test_cases = [
        {
            "original_query": "오늘 홍대 가서 친구들이랑 놀건데 어떻게 입을까?",
            "time_context": {
                "value": ["today"],
                "confidence": 0.9,
                "evidence": ["오늘"]
            },
            "location": {
                "value": ["Hongdae", "street", "urban"],
                "confidence": 0.9,
                "evidence": ["홍대"]
            },
            "mood": {
                "value": ["casual", "hangout", "energetic", "fun"],
                "confidence": 0.85,
                "evidence": ["친구들이랑 놀건데"]
            },
            "style": {
                "value": ["스트릿", "캐주얼", "스포티"],
                "confidence": 0.9,
                "evidence": ["홍대 친구들과 놀기 - 활동적이고 편안한 스타일"]
            },
            "color": {
                "value": ["black", "gray", "navy", "white"],
                "confidence": 0.6,
                "evidence": ["스트릿 스타일에서 선호되는 색상"]
            },
            "size_fit": {
                "value": ["oversized", "baggy", "relaxed", "loose"],
                "confidence": 0.7,
                "evidence": ["스트릿 스타일의 특징적인 핏"]
            },
            "season": {
                "value": ["early spring"],
                "confidence": 0.8,
                "evidence": ["2026-02-04"]
            },
            "user_constraints": {
                "value": [],
                "confidence": 0.0,
                "evidence": []
            },
            "user_requirements": {
                "value": [],
                "confidence": 0.0,
                "evidence": []
            }
        },
        {
            "original_query": "소개팅 나가는데 깔끔하게 입고 싶어",
            "time_context": {
                "value": [],
                "confidence": 0.0,
                "evidence": []
            },
            "location": {
                "value": ["cafe", "restaurant", "indoor"],
                "confidence": 0.7,
                "evidence": ["소개팅 장소 추론"]
            },
            "mood": {
                "value": ["formal", "date", "neat", "sophisticated"],
                "confidence": 0.9,
                "evidence": ["소개팅", "깔끔하게"]
            },
            "style": {
                "value": ["미니멀", "클래식", "오피스"],
                "confidence": 0.85,
                "evidence": ["깔끔한 스타일 요구사항"]
            },
            "color": {
                "value": ["white", "beige", "gray", "navy"],
                "confidence": 0.8,
                "evidence": ["깔끔한 인상을 주는 중립 색상"]
            },
            "size_fit": {
                "value": ["neat fit", "slim fit", "tailored"],
                "confidence": 0.8,
                "evidence": ["깔끔한 핏 요구사항"]
            },
            "season": {
                "value": ["early spring"],
                "confidence": 0.8,
                "evidence": ["2026-02-04"]
            },
            "user_constraints": {
                "value": [],
                "confidence": 0.0,
                "evidence": []
            },
            "user_requirements": {
                "value": [],
                "confidence": 0.0,
                "evidence": []
            }
        },
        {
            "original_query": "날씨 좋은데 가볍게 산책할 때 입을 만한 거",
            "time_context": {
                "value": [],
                "confidence": 0.0,
                "evidence": []
            },
            "location": {
                "value": ["outdoor", "park", "street", "nature"],
                "confidence": 0.8,
                "evidence": ["산책"]
            },
            "mood": {
                "value": ["relaxed", "light", "comfortable", "fresh"],
                "confidence": 0.85,
                "evidence": ["가볍게", "날씨 좋은"]
            },
            "style": {
                "value": ["캐주얼", "아웃도어", "리조트"],
                "confidence": 0.8,
                "evidence": ["산책하기 좋은 편안한 스타일"]
            },
            "color": {
                "value": ["blue", "light blue", "white", "beige", "pastel"],
                "confidence": 0.7,
                "evidence": ["밝고 산뜻한 색상"]
            },
            "size_fit": {
                "value": ["comfortable", "light", "relaxed fit"],
                "confidence": 0.75,
                "evidence": ["가볍게 입을 수 있는 핏"]
            },
            "season": {
                "value": ["early spring", "mild weather"],
                "confidence": 0.8,
                "evidence": ["2026-02-04", "날씨 좋은"]
            },
            "user_constraints": {
                "value": [],
                "confidence": 0.0,
                "evidence": []
            },
            "user_requirements": {
                "value": [],
                "confidence": 0.0,
                "evidence": []
            }
        },
    ]
    
    cat_map_for_planner = {"상의": "shirt", "하의": "pant", "아우터": "outer"}
    
    for idx, test in enumerate(test_cases):
        print("\n" + "=" * 60)
        print(f"4. 쿼리 처리: {test.get('original_query', 'New Format Query')}")
        print("=" * 60)
        
        # 1. 추천 아이템 검색
        recs_raw = recommender.recommend_from_agent(test, top_k=3)
        recs = {cat_map_for_planner.get(k, k): v for k, v in recs_raw.items()}
        
        if not recs: continue
        Visualizer.show_recommendations(recs, top_k=3)
        
        # 2. 조합 생성
        print(f"\n5. 조합 생성 및 평가")
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

            print("------------------------------------------------")
            print(target_style)
            print("------------------------------------------------")

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
    
    print("\n✅ 파이프라인 완료!")

if __name__ == "__main__":
    run_fashion_system()
