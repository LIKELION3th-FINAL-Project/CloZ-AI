"""
색상 유틸리티 - 밝기 분류

visual_metadata_checkpoint.json의 색상을 light/dark로 분류
"""

# 밝은 색상 (light)
LIGHT_COLORS = {
    "white", "cream", "ivory", "beige", "camel",
    "pink", "yellow", "mint", "orange", "light gray"
}

# 어두운 색상 (dark)
DARK_COLORS = {
    "black", "navy", "charcoal", "burgundy", "brown",
    "olive", "khaki", "dark gray"
}

# 중간 색상 (neutral) - 밝기 선호에 따라 포함/제외
NEUTRAL_COLORS = {
    "gray", "blue", "green", "red", "purple"
}

def get_brightness(color: str) -> str:
    """
    색상의 밝기 분류 반환

    Args:
        color: 색상 이름 (영어, 소문자)

    Returns:
        "light", "dark", or "neutral"
    """
    color_lower = color.lower().strip()

    if color_lower in LIGHT_COLORS:
        return "light"
    elif color_lower in DARK_COLORS:
        return "dark"
    else:
        return "neutral"

def is_light_color(color: str) -> bool:
    """밝은 색상인지 확인"""
    return get_brightness(color) == "light"

def is_dark_color(color: str) -> bool:
    """어두운 색상인지 확인"""
    return get_brightness(color) == "dark"

def get_colors_by_brightness(brightness: str) -> set:
    """
    밝기 분류에 해당하는 색상 목록 반환

    Args:
        brightness: "light" or "dark"

    Returns:
        해당 밝기의 색상 set
    """
    if brightness == "light":
        return LIGHT_COLORS
    elif brightness == "dark":
        return DARK_COLORS
    else:
        return NEUTRAL_COLORS

# 한국어 → 영어 밝기 매핑
KOREAN_BRIGHTNESS_KEYWORDS = {
    # 밝은
    "밝은": "light",
    "화사한": "light",
    "연한": "light",
    "파스텔": "light",
    "라이트": "light",
    "밝은색": "light",
    "밝은 색": "light",

    # 어두운
    "어두운": "dark",
    "진한": "dark",
    "무거운": "dark",
    "다크": "dark",
    "어두운색": "dark",
    "어두운 색": "dark",
}

def extract_brightness_preference(text: str) -> str | None:
    """
    텍스트에서 밝기 선호도 추출

    Args:
        text: 피드백 텍스트 (한국어 또는 영어)

    Returns:
        "light", "dark", or None
    """
    text_lower = text.lower()

    # 한국어 키워드 체크
    for keyword, brightness in KOREAN_BRIGHTNESS_KEYWORDS.items():
        if keyword in text_lower:
            return brightness

    # 영어 키워드 체크
    if any(w in text_lower for w in ["bright", "light", "pastel", "soft"]):
        return "light"
    if any(w in text_lower for w in ["dark", "deep", "heavy"]):
        return "dark"

    return None
