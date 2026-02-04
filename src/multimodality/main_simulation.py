from fashion_engine import (
    FashionConfig, CLIPEncoder, FashionDBManager, 
    FashionRecommender, OutfitPlanner, VTONManager, Visualizer
)

def run_fashion_system():
    """전체 패션 추천 및 코디 파이프라인 시뮬레이션 코드"""
    print("=" * 60)
    print("1. Fashion System 초기화")
    print("=" * 60)
    
    config = FashionConfig()
    encoder = CLIPEncoder(config)
    db = FashionDBManager(config)
    recommender = FashionRecommender(config, encoder, db)
    planner = OutfitPlanner(encoder)
    vton = VTONManager()
    
    # 데이터 준비
    print("\n" + "=" * 60)
    print("2. 데이터베이스 및 아이템 로드")
    print("=" * 60)
    # db.initialize_db() # 최초 1회 실행 필요 (주석 해제하여 사용)
    recommender.load_user_items()
    
    print("\n" + "=" * 60)
    print("3. ChromaDB에서 레퍼런스 임베딩 로드")
    print("=" * 60)
    recommender.load_styles()
    
    # 테스트 케이스
    test_cases = [
        {
            "original_query": "오늘 홍대 가서 친구들이랑 놀건데 어떻게 입을까?",
            "analyzed_intent": {"style": "스트릿", "categories": ["top", "bottom", "outer"]},
            "expanded_keywords": ["oversized black hoodie", "wide cargo pants", "baggy street fit", "urban style"]
        },
        {
            "original_query": "소개팅 나가는데 깔끔하게 입고 싶어",
            "analyzed_intent": {"style": "미니멀", "categories": ["top", "bottom", "outer"]},
            "expanded_keywords": ["white linen shirt", "beige straight slacks", "clean minimalist style", "neat fit"]
        },
        {
            "original_query": "날씨 좋은데 가볍게 산책할 때 입을 만한 거",
            "analyzed_intent": {"style": "캐주얼", "categories": ["top", "bottom", "outer"]},
            "expanded_keywords": ["vibrant blue knit", "light blue denim", "fresh daily look", "bright casual"]
        },
    ]
    
    all_results = {}
    
    for idx, test in enumerate(test_cases):
        print("\n" + "=" * 60)
        print(f"4-{idx+1}. 쿼리 처리: {test['original_query']}")
        print("=" * 60)
        
        # 1. 추천 아이템 검색
        recs = recommender.recommend(test, top_k=3)
        
        # 결과 출력 (콘솔)
        for cat, items in recs.items():
            print(f"\n  [{cat}] Top 3 Results:")
            for j, item in enumerate(items, 1):
                f_score = item.get('score', 0.0)
                t_sim = item.get('text_sim', 0.0)
                s_sim = item.get('style_sim', 0.0)
                print(f"    {j}. {item['id']} -> Total: {f_score:.4f} (Text: {t_sim:.2f}, Style: {s_sim:.2f})")
        
        # 2. 추천 결과 시각화
        Visualizer.show_recommendations(recs, top_k=3)
        
        # 3. 조합 생성 및 평가
        print(f"\n5-{idx+1}. 조합 생성 및 평가")
        print("-" * 60)
        combos = planner.generate_combinations(recs, top_n=3)
        
        if combos:
            best_outfits = planner.evaluate_outfits(
                combos, 
                recommender.style_profiles, 
                test['analyzed_intent']['style'],
                overall_weight=0.7
            )
            
            if best_outfits:
                # 4. 조합 시각화
                print(f"\n6-{idx+1}. 조합 시각화")
                print("-" * 60)
                Visualizer.show_top_combinations(best_outfits, num_to_show=3)
                
                # 5. VTON 이미지 생성
                print(f"\n7-{idx+1}. Fashion-VTON 이미지 생성")
                print("-" * 60)
                vton_result = vton.try_on(
                    person_img_path="/content/model.webp",
                    outfit=best_outfits[0]['combination'],
                    output_prefix=f"/content/output",
                    idx=idx
                )
                
                if vton_result:
                    Visualizer.show_vton_result(vton_result, test['original_query'], idx)
                
                all_results[idx] = {
                    "query": test,
                    "recommendations": recs,
                    "combinations": best_outfits[:3],
                    "vton_result": vton_result
                }
    
    print("\n" + "=" * 60)
    print("✅ 전체 파이프라인 완료!")
    print("=" * 60)
    return all_results

if __name__ == "__main__":
    run_fashion_system()
