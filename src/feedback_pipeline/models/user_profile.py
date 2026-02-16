"""
사용자 프로필 데이터 모델

Analyst Agent가 추출한 취향 정보를 저장
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime


# ============================================
# 신뢰도 기반 Profile 업데이트 설정
# ============================================
MIN_MENTIONS_FOR_PROFILE = 3        # 3회 이상 동일 키워드 언급 시 저장
CONFIDENCE_DECAY_DAYS = 30          # 30일 후 신뢰도 감소 시작


@dataclass
class KeywordMention:
    """
    키워드 언급 추적

    세션별로 키워드가 몇 번 언급되었는지 추적하여
    MIN_MENTIONS_FOR_PROFILE 이상일 때만 Profile에 저장
    """
    keyword: str
    mention_count: int = 1
    first_mentioned: str = field(default_factory=lambda: datetime.now().isoformat())
    last_mentioned: str = field(default_factory=lambda: datetime.now().isoformat())
    sessions: List[str] = field(default_factory=list)  # 언급된 세션 ID들

    def to_dict(self) -> Dict:
        return {
            "keyword": self.keyword,
            "mention_count": self.mention_count,
            "first_mentioned": self.first_mentioned,
            "last_mentioned": self.last_mentioned,
            "sessions": self.sessions,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "KeywordMention":
        return cls(
            keyword=data["keyword"],
            mention_count=data.get("mention_count", 1),
            first_mentioned=data.get("first_mentioned", datetime.now().isoformat()),
            last_mentioned=data.get("last_mentioned", datetime.now().isoformat()),
            sessions=data.get("sessions", []),
        )

    def should_add_to_profile(self) -> bool:
        """Profile에 추가할지 판단 (3회 이상 언급 시)"""
        return self.mention_count >= MIN_MENTIONS_FOR_PROFILE

    def calculate_confidence(self) -> float:
        """
        신뢰도 계산

        - 기본 신뢰도: 언급 횟수 기반 (0.3 ~ 0.9)
        - 시간 decay: 마지막 언급 후 30일마다 -0.1
        """
        # 기본 신뢰도: 3회 = 0.3, 4회 = 0.4, ..., 9회+ = 0.9
        base = min(0.9, 0.3 + (self.mention_count - 3) * 0.1)

        # 시간 decay 계산
        try:
            last_dt = datetime.fromisoformat(self.last_mentioned)
            days_since = (datetime.now() - last_dt).days
            decay = max(0, (days_since - CONFIDENCE_DECAY_DAYS) // 30) * 0.1
        except (ValueError, TypeError):
            decay = 0

        return max(0.1, base - decay)

    def update_mention(self, session_id: str):
        """새로운 언급 기록"""
        self.mention_count += 1
        self.last_mentioned = datetime.now().isoformat()
        if session_id not in self.sessions:
            self.sessions.append(session_id)


@dataclass
class ColorPreferences:
    """색상 선호도"""
    warm: List[str] = field(default_factory=list)   # 따뜻한 색 ["베이지", "브라운"]
    cool: List[str] = field(default_factory=list)   # 차가운 색 ["네이비", "그레이"]
    avoid: List[str] = field(default_factory=list)  # 회피 색 ["형광색", "핫핑크"]

    def to_dict(self) -> Dict[str, List[str]]:
        return {
            "warm": self.warm,
            "cool": self.cool,
            "avoid": self.avoid,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, List[str]]) -> "ColorPreferences":
        return cls(
            warm=data.get("warm", []),
            cool=data.get("cool", []),
            avoid=data.get("avoid", []),
        )


@dataclass
class FitPreferences:
    """핏 선호도"""
    top: Optional[str] = None      # "oversized", "regular", "slim"
    bottom: Optional[str] = None
    outer: Optional[str] = None
    overall: str = ""               # 전체 설명

    def to_dict(self) -> Dict[str, Optional[str]]:
        return {
            "top": self.top,
            "bottom": self.bottom,
            "outer": self.outer,
            "overall": self.overall,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Optional[str]]) -> "FitPreferences":
        return cls(
            top=data.get("top"),
            bottom=data.get("bottom"),
            outer=data.get("outer"),
            overall=data.get("overall", ""),
        )


@dataclass
class UserBias:
    """
    사용자 취향 (Bias)

    Analyst Agent가 세션 로그에서 추출
    """
    color_preferences: ColorPreferences = field(default_factory=ColorPreferences)
    fit_preferences: FitPreferences = field(default_factory=FitPreferences)
    style_keywords: List[str] = field(default_factory=list)     # max 20개
    avoid_keywords: List[str] = field(default_factory=list)
    subjective_mappings: Dict[str, str] = field(default_factory=dict)  # {"차가운 색": "네이비, 그레이"}
    confidence_scores: Dict[str, float] = field(default_factory=dict)  # {"color_warm": 0.85}
    raw_insights: List[str] = field(default_factory=list)       # max 10개

    # 신뢰도 기반 키워드 추적 (3회 이상 언급 시에만 style/avoid_keywords로 승격)
    style_keyword_mentions: List[KeywordMention] = field(default_factory=list)
    avoid_keyword_mentions: List[KeywordMention] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "color_preferences": self.color_preferences.to_dict(),
            "fit_preferences": self.fit_preferences.to_dict(),
            "style_keywords": self.style_keywords,
            "avoid_keywords": self.avoid_keywords,
            "subjective_mappings": self.subjective_mappings,
            "confidence_scores": self.confidence_scores,
            "raw_insights": self.raw_insights,
            "style_keyword_mentions": [m.to_dict() for m in self.style_keyword_mentions],
            "avoid_keyword_mentions": [m.to_dict() for m in self.avoid_keyword_mentions],
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "UserBias":
        return cls(
            color_preferences=ColorPreferences.from_dict(data.get("color_preferences", {})),
            fit_preferences=FitPreferences.from_dict(data.get("fit_preferences", {})),
            style_keywords=data.get("style_keywords", []),
            avoid_keywords=data.get("avoid_keywords", []),
            subjective_mappings=data.get("subjective_mappings", {}),
            confidence_scores=data.get("confidence_scores", {}),
            raw_insights=data.get("raw_insights", []),
            style_keyword_mentions=[
                KeywordMention.from_dict(m) for m in data.get("style_keyword_mentions", [])
            ],
            avoid_keyword_mentions=[
                KeywordMention.from_dict(m) for m in data.get("avoid_keyword_mentions", [])
            ],
        )

    def add_style_keyword_mention(self, keyword: str, session_id: str) -> bool:
        """
        스타일 키워드 언급 추가

        Returns:
            True if keyword was promoted to style_keywords (3회 달성)
        """
        # 기존 mention 찾기
        existing = next((m for m in self.style_keyword_mentions if m.keyword == keyword), None)

        if existing:
            existing.update_mention(session_id)
            # 3회 달성 시 style_keywords로 승격
            if existing.should_add_to_profile() and keyword not in self.style_keywords:
                self.style_keywords.append(keyword)
                self.confidence_scores[f"style_{keyword}"] = existing.calculate_confidence()
                return True
        else:
            # 새로운 mention 생성
            new_mention = KeywordMention(
                keyword=keyword,
                mention_count=1,
                sessions=[session_id]
            )
            self.style_keyword_mentions.append(new_mention)

        return False

    def add_avoid_keyword_mention(self, keyword: str, session_id: str) -> bool:
        """
        회피 키워드 언급 추가

        Returns:
            True if keyword was promoted to avoid_keywords (3회 달성)
        """
        # 기존 mention 찾기
        existing = next((m for m in self.avoid_keyword_mentions if m.keyword == keyword), None)

        if existing:
            existing.update_mention(session_id)
            # 3회 달성 시 avoid_keywords로 승격
            if existing.should_add_to_profile() and keyword not in self.avoid_keywords:
                self.avoid_keywords.append(keyword)
                self.confidence_scores[f"avoid_{keyword}"] = existing.calculate_confidence()
                return True
        else:
            # 새로운 mention 생성
            new_mention = KeywordMention(
                keyword=keyword,
                mention_count=1,
                sessions=[session_id]
            )
            self.avoid_keyword_mentions.append(new_mention)

        return False

    def get_keyword_mention_count(self, keyword: str, is_avoid: bool = False) -> int:
        """특정 키워드의 언급 횟수 조회"""
        mentions = self.avoid_keyword_mentions if is_avoid else self.style_keyword_mentions
        existing = next((m for m in mentions if m.keyword == keyword), None)
        return existing.mention_count if existing else 0


@dataclass
class UserProfile:
    """
    사용자 프로필

    data/users/{user_id}/profile.json에 저장
    """
    user_id: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())
    session_count: int = 0
    biases: UserBias = field(default_factory=UserBias)
    context_summary: str = ""  # LLM Context용 압축 요약 (max 2000자)

    def to_dict(self) -> Dict:
        return {
            "user_id": self.user_id,
            "created_at": self.created_at,
            "last_updated": self.last_updated,
            "session_count": self.session_count,
            "biases": self.biases.to_dict(),
            "context_summary": self.context_summary,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "UserProfile":
        return cls(
            user_id=data["user_id"],
            created_at=data.get("created_at", datetime.now().isoformat()),
            last_updated=data.get("last_updated", datetime.now().isoformat()),
            session_count=data.get("session_count", 0),
            biases=UserBias.from_dict(data.get("biases", {})),
            context_summary=data.get("context_summary", ""),
        )

    def update_timestamp(self):
        """타임스탬프 업데이트"""
        self.last_updated = datetime.now().isoformat()
        self.session_count += 1
