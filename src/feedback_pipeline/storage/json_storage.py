"""
JSON File Storage

사용자 프로필, 세션 로그, 승인된 코디 데이터를 JSON 파일로 저장/로드

디렉토리 구조:
data/
├── users/{user_id}/
│   ├── profile.json
│   └── sessions/{session_id}.json
└── feedback_datasets/{user_id}/
    └── approved_outfits.json
"""

import os
import json
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any
from datetime import datetime
from pathlib import Path

from ..models import (
    UserProfile,
    SessionLog,
    SessionEntry,
    OutfitSet,
    FeedbackInput,
)


@dataclass
class StorageConfig:
    """Storage 설정"""
    base_dir: str = "./data"            # 기본 데이터 디렉토리

    def __post_init__(self):
        # 환경변수에서 경로 오버라이드
        env_dir = os.getenv("DATA_DIR")
        if env_dir:
            self.base_dir = env_dir


class JsonStorage:
    """
    JSON 파일 기반 Storage

    사용 예시:
        storage = JsonStorage()

        # 프로필 저장/로드
        storage.save_user_profile(profile)
        profile = storage.load_user_profile("user_123")

        # 세션 저장/로드
        storage.save_session(session_log)
        session = storage.load_session("user_123", "session_456")

        # 승인된 코디 저장
        storage.save_approved_outfit("user_123", outfit)
    """

    def __init__(self, config: Optional[StorageConfig] = None):
        self.config = config or StorageConfig()
        self._ensure_directories()

    def _ensure_directories(self):
        """기본 디렉토리 생성"""
        Path(self.config.base_dir).mkdir(parents=True, exist_ok=True)
        Path(f"{self.config.base_dir}/users").mkdir(exist_ok=True)
        Path(f"{self.config.base_dir}/feedback_datasets").mkdir(exist_ok=True)

    # ==================== 경로 헬퍼 ====================

    def _user_dir(self, user_id: str) -> Path:
        """사용자 디렉토리 경로"""
        path = Path(f"{self.config.base_dir}/users/{user_id}")
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _sessions_dir(self, user_id: str) -> Path:
        """세션 디렉토리 경로"""
        path = self._user_dir(user_id) / "sessions"
        path.mkdir(exist_ok=True)
        return path

    def _profile_path(self, user_id: str) -> Path:
        """프로필 파일 경로"""
        return self._user_dir(user_id) / "profile.json"

    def _session_path(self, user_id: str, session_id: str) -> Path:
        """세션 파일 경로"""
        return self._sessions_dir(user_id) / f"{session_id}.json"

    def _approved_outfits_path(self, user_id: str) -> Path:
        """승인된 코디 파일 경로"""
        path = Path(f"{self.config.base_dir}/feedback_datasets/{user_id}")
        path.mkdir(parents=True, exist_ok=True)
        return path / "approved_outfits.json"

    # ==================== User Profile ====================

    def save_user_profile(self, profile: UserProfile) -> bool:
        """
        사용자 프로필 저장

        Args:
            profile: UserProfile 객체

        Returns:
            저장 성공 여부
        """
        try:
            path = self._profile_path(profile.user_id)
            data = profile.to_dict()
            data["updated_at"] = datetime.now().isoformat()

            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            return True
        except Exception as e:
            print(f"프로필 저장 실패: {e}")
            return False

    def load_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """
        사용자 프로필 로드

        Args:
            user_id: 사용자 ID

        Returns:
            UserProfile 또는 None (없을 경우)
        """
        try:
            path = self._profile_path(user_id)
            if not path.exists():
                return None

            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            return UserProfile.from_dict(data)
        except Exception as e:
            print(f"프로필 로드 실패: {e}")
            return None

    def get_or_create_profile(self, user_id: str) -> UserProfile:
        """
        프로필 로드 또는 새로 생성

        Args:
            user_id: 사용자 ID

        Returns:
            UserProfile (기존 또는 새로 생성된)
        """
        profile = self.load_user_profile(user_id)
        if profile is None:
            profile = UserProfile(user_id=user_id)
            self.save_user_profile(profile)
        return profile

    # ==================== Session Log ====================

    def save_session(self, session: SessionLog) -> bool:
        """
        세션 로그 저장

        Args:
            session: SessionLog 객체

        Returns:
            저장 성공 여부
        """
        try:
            path = self._session_path(session.user_id, session.session_id)
            data = session.to_dict()

            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            return True
        except Exception as e:
            print(f"세션 저장 실패: {e}")
            return False

    def load_session(self, user_id: str, session_id: str) -> Optional[SessionLog]:
        """
        세션 로그 로드

        Args:
            user_id: 사용자 ID
            session_id: 세션 ID

        Returns:
            SessionLog 또는 None
        """
        try:
            path = self._session_path(user_id, session_id)
            if not path.exists():
                return None

            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            return SessionLog.from_dict(data)
        except Exception as e:
            print(f"세션 로드 실패: {e}")
            return None

    def list_sessions(self, user_id: str) -> List[str]:
        """
        사용자의 모든 세션 ID 목록

        Args:
            user_id: 사용자 ID

        Returns:
            세션 ID 리스트
        """
        try:
            sessions_dir = self._sessions_dir(user_id)
            return [
                f.stem for f in sessions_dir.glob("*.json")
            ]
        except Exception:
            return []

    def get_recent_sessions(
        self,
        user_id: str,
        limit: int = 10
    ) -> List[SessionLog]:
        """
        최근 세션 목록 로드

        Args:
            user_id: 사용자 ID
            limit: 최대 개수

        Returns:
            SessionLog 리스트 (최신순)
        """
        sessions = []
        session_ids = self.list_sessions(user_id)

        for sid in session_ids:
            session = self.load_session(user_id, sid)
            if session:
                sessions.append(session)

        # 시작 시간 기준 정렬 (최신순)
        sessions.sort(key=lambda s: s.started_at, reverse=True)
        return sessions[:limit]

    # ==================== Approved Outfits (정답 데이터셋) ====================

    def save_approved_outfit(
        self,
        user_id: str,
        outfit: OutfitSet,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        승인된 코디를 정답 데이터셋에 추가

        Args:
            user_id: 사용자 ID
            outfit: 승인된 OutfitSet
            context: 추가 컨텍스트 (원본 프롬프트, 세션 정보 등)

        Returns:
            저장 성공 여부
        """
        try:
            path = self._approved_outfits_path(user_id)

            # 기존 데이터 로드
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            else:
                data = {"user_id": user_id, "outfits": []}

            # 새 항목 추가
            entry = {
                "outfit": outfit.to_dict(),
                "approved_at": datetime.now().isoformat(),
                "context": context or {},
            }
            data["outfits"].append(entry)
            data["updated_at"] = datetime.now().isoformat()

            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            return True
        except Exception as e:
            print(f"승인 코디 저장 실패: {e}")
            return False

    def load_approved_outfits(self, user_id: str) -> List[Dict[str, Any]]:
        """
        승인된 코디 목록 로드

        Args:
            user_id: 사용자 ID

        Returns:
            승인된 코디 리스트
        """
        try:
            path = self._approved_outfits_path(user_id)
            if not path.exists():
                return []

            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            return data.get("outfits", [])
        except Exception:
            return []

    def count_approved_outfits(self, user_id: str) -> int:
        """승인된 코디 개수"""
        return len(self.load_approved_outfits(user_id))

    # ==================== 유틸리티 ====================

    def delete_user_data(self, user_id: str) -> bool:
        """
        사용자 데이터 전체 삭제 (주의!)

        Args:
            user_id: 사용자 ID

        Returns:
            삭제 성공 여부
        """
        import shutil
        try:
            user_dir = self._user_dir(user_id)
            if user_dir.exists():
                shutil.rmtree(user_dir)

            approved_path = self._approved_outfits_path(user_id)
            if approved_path.exists():
                approved_path.unlink()

            return True
        except Exception as e:
            print(f"사용자 데이터 삭제 실패: {e}")
            return False

    def export_user_data(self, user_id: str) -> Dict[str, Any]:
        """
        사용자 데이터 전체 내보내기 (GDPR 등 대응)

        Args:
            user_id: 사용자 ID

        Returns:
            전체 사용자 데이터 딕셔너리
        """
        profile = self.load_user_profile(user_id)
        sessions = self.get_recent_sessions(user_id, limit=100)
        approved = self.load_approved_outfits(user_id)

        return {
            "user_id": user_id,
            "exported_at": datetime.now().isoformat(),
            "profile": profile.to_dict() if profile else None,
            "sessions": [s.to_dict() for s in sessions],
            "approved_outfits": approved,
        }
