"""
세션 로그 데이터 모델

피드백 대화 로그를 저장하여 Analyst Agent 분석에 활용
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional
from datetime import datetime


class SessionStatus(Enum):
    """세션 상태"""
    ACTIVE = "active"               # 세션 진행 중
    COMPLETED = "completed"         # YES로 승인하여 정상 종료
    ABANDONED = "abandoned"         # 중도 이탈 (타임아웃)
    BUYING_REDIRECT = "buying_redirect"  # 구매 추천으로 전환되어 종료


@dataclass
class SessionEntry:
    """
    세션 로그 엔트리

    outfit_generation, feedback, clarification 등
    """
    entry_id: str
    entry_type: str     # "outfit_generation", "feedback", "clarification", "action"
    timestamp: str
    content: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "entry_type": self.entry_type,
            "timestamp": self.timestamp,
            "content": self.content,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionEntry":
        return cls(
            entry_id=data["entry_id"],
            entry_type=data["entry_type"],
            timestamp=data["timestamp"],
            content=data["content"],
            metadata=data.get("metadata", {}),
        )


@dataclass
class SessionLog:
    """
    세션 로그

    data/users/{user_id}/sessions/{session_id}.json에 저장
    """
    session_id: str
    user_id: str
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    ended_at: Optional[str] = None
    status: SessionStatus = SessionStatus.ACTIVE
    context: Dict[str, Any] = field(default_factory=dict)  # occasion, weather, initial_request
    entries: List[SessionEntry] = field(default_factory=list)
    summary: Dict[str, int] = field(default_factory=lambda: {
        "total_generations": 0,
        "approved_count": 0,
        "rejected_count": 0,
        "buying_recommendations": 0,
    })

    def add_entry(self, entry: SessionEntry):
        """엔트리 추가"""
        self.entries.append(entry)

        # summary 업데이트
        if entry.entry_type == "outfit_generation":
            self.summary["total_generations"] += 1
        elif entry.entry_type == "feedback":
            if entry.content.get("is_positive"):
                self.summary["approved_count"] += 1
            else:
                self.summary["rejected_count"] += 1
        elif entry.entry_type == "action" and entry.content.get("action") == "BUYING":
            self.summary["buying_recommendations"] += 1

    def close(self, status: SessionStatus):
        """세션 종료"""
        self.ended_at = datetime.now().isoformat()
        self.status = status

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "status": self.status.value,
            "context": self.context,
            "entries": [e.to_dict() for e in self.entries],
            "summary": self.summary,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionLog":
        return cls(
            session_id=data["session_id"],
            user_id=data["user_id"],
            started_at=data.get("started_at", datetime.now().isoformat()),
            ended_at=data.get("ended_at"),
            status=SessionStatus(data.get("status", "active")),
            context=data.get("context", {}),
            entries=[SessionEntry.from_dict(e) for e in data.get("entries", [])],
            summary=data.get("summary", {
                "total_generations": 0,
                "approved_count": 0,
                "rejected_count": 0,
                "buying_recommendations": 0,
            }),
        )
