"""
세부 카테고리 정규화 유틸

역할:
- 사용자/LLM 텍스트의 카테고리 표현을 canonical detail category로 정규화
- 대카테고리 표현(상의/tops 등)을 canonical main category로 정규화
- checker/query_builder에서 공통 사용
"""

from typing import Dict, List, Set


# Canonical detail categories (Havati 기준)
TOP_DETAILS: List[str] = ["Tee", "Shirt", "Sweatshirt", "Knitwear"]
BOTTOM_DETAILS: List[str] = ["Denim", "Chino", "Trousers", "Easy_pants", "Work_pants", "Short"]
OUTER_DETAILS: List[str] = ["Jacket_Blouson", "Coat", "Cardigan", "Jumper_Parka", "Padding", "Leather", "Vest"]


# QueryBuilder에서 참조: canonical -> alias list
DETAIL_CAT_RULES: Dict[str, List[str]] = {
    "Tee": ["tee", "t-shirt", "t shirt", "티셔츠", "티", "반팔", "short sleeve", "short-sleeve"],
    "Shirt": ["shirt", "shirts", "셔츠", "남방", "button down", "button-down"],
    "Sweatshirt": ["sweatshirt", "sweat shirt", "맨투맨", "스웻셔츠", "스웨트셔츠"],
    "Knitwear": ["knit", "knitwear", "니트", "스웨터", "vest knit", "니트웨어"],
    "Denim": ["denim", "jeans", "진", "청바지", "데님"],
    "Chino": ["chino", "치노", "치노팬츠"],
    "Trousers": ["trousers", "slacks", "슬랙스", "트라우저"],
    "Easy_pants": ["easy pants", "easy_pants", "easypants", "조거", "밴딩팬츠", "이지팬츠"],
    "Work_pants": ["work pants", "work_pants", "카고", "워크팬츠", "cargo"],
    "Short": ["short", "shorts", "반바지", "쇼츠"],
    "Jacket_Blouson": ["jacket", "blouson", "자켓", "블루종"],
    "Coat": ["coat", "코트"],
    "Cardigan": ["cardigan", "가디건"],
    "Jumper_Parka": ["jumper", "parka", "점퍼", "파카"],
    "Padding": ["padding", "puffer", "패딩"],
    "Leather": ["leather", "가죽", "레더"],
    "Vest": ["vest", "베스트", "조끼"],
}


def _normalize_token(value: str) -> str:
    return str(value or "").strip().lower().replace("-", " ").replace("_", " ")


def build_detail_alias_map() -> Dict[str, str]:
    alias_to_canonical: Dict[str, str] = {}
    for canonical, patterns in DETAIL_CAT_RULES.items():
        alias_to_canonical[_normalize_token(canonical)] = canonical
        alias_to_canonical[canonical.lower()] = canonical
        for pattern in patterns:
            token = _normalize_token(pattern)
            if token:
                alias_to_canonical[token] = canonical
    return alias_to_canonical


DETAIL_ALIAS_TO_CANONICAL: Dict[str, str] = build_detail_alias_map()


# main category alias (checker 공통)
MAIN_CATEGORY_ALIASES: Dict[str, List[str]] = {
    "TOPS": ["상의", "탑", "top", "tops", "shirt", "shirts", "tee", "티", "티셔츠"],
    "BOTTOMS": ["하의", "바지", "bottom", "bottoms", "pant", "pants", "trouser", "denim", "jeans"],
    "OUTER": ["아우터", "겉옷", "outer", "outers", "jacket", "coat", "cardigan", "jumper", "parka"],
}


def _build_main_alias_map() -> Dict[str, str]:
    result: Dict[str, str] = {}
    for canonical, aliases in MAIN_CATEGORY_ALIASES.items():
        result[_normalize_token(canonical)] = canonical
        for alias in aliases:
            result[_normalize_token(alias)] = canonical
    return result


MAIN_ALIAS_TO_CANONICAL: Dict[str, str] = _build_main_alias_map()


DETAIL_GROUPS: Dict[str, Set[str]] = {
    "상의": set(TOP_DETAILS),
    "TOPS": set(TOP_DETAILS),
    "바지": set(BOTTOM_DETAILS),
    "하의": set(BOTTOM_DETAILS),
    "BOTTOMS": set(BOTTOM_DETAILS),
    "아우터": set(OUTER_DETAILS),
    "OUTER": set(OUTER_DETAILS),
}


def normalize_main_category(value: str) -> str:
    token = _normalize_token(value)
    if not token:
        return ""
    return MAIN_ALIAS_TO_CANONICAL.get(token, str(value).strip())


def normalize_detail_category(value: str) -> str:
    token = _normalize_token(value)
    if not token:
        return ""

    # direct exact
    direct = DETAIL_ALIAS_TO_CANONICAL.get(token)
    if direct:
        return direct

    # contains-based fallback (긴 alias 우선)
    aliases = sorted(DETAIL_ALIAS_TO_CANONICAL.keys(), key=len, reverse=True)
    for alias in aliases:
        if alias and (alias in token or token in alias):
            return DETAIL_ALIAS_TO_CANONICAL[alias]

    return str(value).strip()

