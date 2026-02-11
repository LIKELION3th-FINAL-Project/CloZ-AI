"""
Main Adapter - main_simulation 출력을 feedback_pipeline 입력으로 변환

main_simulation (fashion_engine) 출력 형식:
- best_outfits: List[Dict] - OutfitPlanner.evaluate_outfits() 결과
  - combination: Tuple[Dict, Dict, Dict] - (pant, outer, shirt)
  - combination_idx: int
  - final_score: float
  - pant, outer, shirt: str (파일명)

feedback_pipeline 입력 형식:
- OutfitSet: outfit_id, image_url, products: List[ItemInfo]
- ItemInfo: product_id, product_name, category_main, category_sub
"""

import json
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

from ..models import OutfitSet, ItemInfo


# 카테고리 매핑 (main → feedback_pipeline)
CAT_EN_TO_KR = {
    "shirt": "상의",
    "pant": "바지",
    "outer": "아우터",
    # 혹시 한글로 오는 경우 대비
    "상의": "상의",
    "하의": "바지",
    "바지": "바지",
    "아우터": "아우터",
}

# 폴더명 → 대분류 매핑
FOLDER_TO_CAT = {
    "tops": "상의",
    "bottoms": "바지",
    "outers": "아우터",
}


def get_detail_cat_from_classification(
    filename: str,
    classification_results: Dict[str, Any],
    folder_hint: Optional[str] = None
) -> str:
    """
    classification_results.json에서 세부카테고리 조회

    Args:
        filename: 파일명 (예: "tops_1.jpeg")
        classification_results: classification_results.json 데이터
        folder_hint: 폴더 힌트 (예: "tops", "bottoms")

    Returns:
        세부카테고리 문자열 (예: "니트/스웨트")
    """
    # classification_results 키 형식: "closet/tops/tops_1.jpeg"
    possible_keys = []

    if folder_hint:
        possible_keys.append(f"closet/{folder_hint}/{filename}")

    # 폴더 힌트 없으면 모든 가능한 경로 시도
    for folder in ["tops", "bottoms", "outers"]:
        possible_keys.append(f"closet/{folder}/{filename}")

    for key in possible_keys:
        if key in classification_results:
            result = classification_results[key]
            return result.get("detail_cat", result.get("broad_cat", "기타"))

    # 못 찾으면 기본값
    return "기타"


def convert_outfit_to_outfitset(
    best_outfit: Dict[str, Any],
    output_image_path: str,
    classification_results: Optional[Dict[str, Any]] = None
) -> OutfitSet:
    """
    main_simulation의 best_outfit을 OutfitSet으로 변환

    Args:
        best_outfit: OutfitPlanner.evaluate_outfits() 결과의 한 요소
            - combination: Tuple[Dict, Dict, Dict]
            - combination_idx: int
            - pant, outer, shirt: str (파일명)
        output_image_path: VTON 출력 이미지 경로
        classification_results: classification_results.json 데이터 (옵션)

    Returns:
        OutfitSet 객체
    """
    products = []

    # combination 순서: (pant, outer, shirt)
    cat_order = ["pant", "outer", "shirt"]
    folder_map = {"pant": "bottoms", "outer": "outers", "shirt": "tops"}

    combination = best_outfit.get("combination", ())

    for i, cat_en in enumerate(cat_order):
        if i >= len(combination):
            continue

        item = combination[i]
        if item is None:
            continue

        # 파일명 추출
        if isinstance(item, dict):
            path = item.get("path", "")
            filename = Path(path).name if path else best_outfit.get(cat_en, "unknown")
            item_category = item.get("category", "")
        else:
            filename = str(item)
            item_category = ""

        # 대분류 결정
        if item_category:
            category_main = CAT_EN_TO_KR.get(item_category, item_category)
        else:
            category_main = CAT_EN_TO_KR.get(cat_en, "상의")

        # 세부카테고리 조회
        category_sub = "기타"
        if classification_results:
            folder_hint = folder_map.get(cat_en)
            category_sub = get_detail_cat_from_classification(
                filename, classification_results, folder_hint
            )

        # product_id 생성 (파일명 해시)
        product_id = abs(hash(filename)) % 1000000

        products.append(ItemInfo(
            product_id=product_id,
            product_name=filename,
            category_main=category_main,
            category_sub=category_sub,
        ))

    return OutfitSet(
        outfit_id=best_outfit.get("combination_idx", 0),
        image_url=output_image_path,
        products=products,
    )


def load_classification_results(path: str = "classification_results.json") -> Dict[str, Any]:
    """classification_results.json 로드"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"[경고] {path} 파일을 찾을 수 없습니다.")
        return {}
    except json.JSONDecodeError as e:
        print(f"[경고] {path} JSON 파싱 실패: {e}")
        return {}


# 테스트용
if __name__ == "__main__":
    # Mock 데이터
    mock_best_outfit = {
        "combination": (
            {"id": "pant/pants_1.jpeg", "category": "하의", "path": "closet/bottoms/pants_1.jpeg"},
            {"id": "outer/outer_1.jpeg", "category": "아우터", "path": "closet/outers/outer_1.jpeg"},
            {"id": "shirt/tops_1.jpeg", "category": "상의", "path": "closet/tops/tops_1.jpeg"},
        ),
        "combination_idx": 0,
        "final_score": 0.85,
        "pant": "pants_1.jpeg",
        "outer": "outer_1.jpeg",
        "shirt": "tops_1.jpeg"
    }

    # 변환 테스트
    classification = load_classification_results()
    outfit_set = convert_outfit_to_outfitset(
        mock_best_outfit,
        "output_images/output_q0.jpg",
        classification
    )

    print(f"OutfitSet 생성 완료:")
    print(f"  outfit_id: {outfit_set.outfit_id}")
    print(f"  image_url: {outfit_set.image_url}")
    print(f"  products: {len(outfit_set.products)}개")
    for p in outfit_set.products:
        print(f"    - {p.category_main}/{p.category_sub}: {p.product_name}")
