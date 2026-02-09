"""
WardrobeBuilder - 옷장 구축 통합 모듈

사용자 옷장 이미지 입력 → 세부 카테고리 분류 → 임베딩 생성 → ChromaDB 저장

사용법:
    # 신규 사용자: 폴더 전체 처리
    builder = WardrobeBuilder(user_id="user_001")
    result = builder.build_from_directory("closet/")

    # 기존 사용자: 새 옷 추가
    success, item = builder.add_item("new_jacket.jpg", broad_cat="outers")
"""

import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any

from PIL import Image
import chromadb

# 프로젝트 루트를 path에 추가
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ..utils.clothing_classifier import ClothingClassifier


# broad_cat 매핑
BROAD_CAT_MAPPING = {
    "상의": "tops",
    "바지": "bottoms",
    "하의": "bottoms",
    "아우터": "outers",
}

BROAD_CAT_TO_KOREAN = {
    "tops": "상의",
    "bottoms": "바지",
    "outers": "아우터",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass
class WardrobeItem:
    """옷장 아이템 데이터 모델"""
    item_id: str           # "user_id:broad_cat/filename"
    user_id: str
    image_path: str
    broad_cat: str         # "tops", "bottoms", "outers"
    detail_cat: str        # "후드 티셔츠", "데님 팬츠" 등
    embedding: List[float] # 512차원
    created_at: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BuildResult:
    """옷장 구축 결과"""
    success: bool
    total_items: int
    added_items: int
    skipped_items: int
    failed_items: List[Dict[str, str]] = field(default_factory=list)
    collection_name: str = ""


class WardrobeBuilder:
    """
    옷장 구축 통합 모듈

    역할:
    1. 이미지 업로드 → 분류 → 임베딩 → ChromaDB 저장 (전체 파이프라인)
    2. 기존 컬렉션에 새 아이템 upsert (중복 방지)
    3. 사용자별 옷장 관리
    """

    def __init__(
        self,
        user_id: str,
        chroma_path: str = None,
        collection_name: str = "wardrobe",
        device: str = None,
        use_background_removal: bool = False  # 기본 False (속도)
    ):
        """
        초기화

        Args:
            user_id: 사용자 ID
            chroma_path: ChromaDB 경로 (기본: data/chroma_wardrobe)
            collection_name: 컬렉션 이름 (기본: "wardrobe")
            device: FashionCLIP 디바이스 ("cuda", "mps", "cpu")
            use_background_removal: 배경 제거 사용 여부
        """
        self.user_id = user_id
        self.chroma_path = chroma_path or str(PROJECT_ROOT / "data" / "chroma_wardrobe")
        self.collection_name = collection_name
        self.device = device
        self.use_background_removal = use_background_removal

        # 지연 로딩을 위해 None으로 초기화
        self._classifier = None
        self._embedder = None
        self._collection = None
        self._chroma_client = None

        print(f"[WardrobeBuilder] 초기화: user_id={user_id}")

    @property
    def classifier(self) -> ClothingClassifier:
        """ClothingClassifier 지연 로딩"""
        if self._classifier is None:
            print("[WardrobeBuilder] ClothingClassifier 로딩...")
            self._classifier = ClothingClassifier()
        return self._classifier

    @property
    def embedder(self):
        """FashionCLIPEmbedder 지연 로딩"""
        if self._embedder is None:
            print("[WardrobeBuilder] FashionCLIPEmbedder 로딩...")
            sys.path.append(str(PROJECT_ROOT / "embedding_generator"))
            from generate_fashionclip_embeddings import FashionCLIPEmbedder
            self._embedder = FashionCLIPEmbedder(device=self.device)
        return self._embedder

    @property
    def collection(self):
        """ChromaDB 컬렉션 지연 로딩"""
        if self._collection is None:
            self._collection = self._get_or_create_collection()
        return self._collection

    def _get_or_create_collection(self):
        """ChromaDB 컬렉션 가져오기 또는 생성"""
        os.makedirs(self.chroma_path, exist_ok=True)

        self._chroma_client = chromadb.PersistentClient(path=self.chroma_path)

        collection = self._chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        print(f"[WardrobeBuilder] 컬렉션 로드: {self.collection_name} ({collection.count()}개 아이템)")
        return collection

    def build_from_directory(
        self,
        closet_dir: str,
        force_rebuild: bool = False,
        limit_per_category: int = None
    ) -> BuildResult:
        """
        디렉토리에서 전체 옷장 구축

        Args:
            closet_dir: 옷장 이미지 디렉토리 (tops/, bottoms/, outers/ 하위 구조)
            force_rebuild: True이면 기존 사용자 데이터 삭제 후 재생성
            limit_per_category: 카테고리당 처리할 최대 개수 (테스트용)

        Returns:
            BuildResult
        """
        closet_path = Path(closet_dir)
        if not closet_path.exists():
            return BuildResult(
                success=False,
                total_items=0,
                added_items=0,
                skipped_items=0,
                failed_items=[{"path": closet_dir, "reason": "디렉토리 없음"}],
                collection_name=self.collection_name
            )

        # force_rebuild 시 기존 데이터 삭제
        if force_rebuild:
            self._delete_user_items()

        # 이미지 파일 수집
        items_to_process = []
        for broad_cat in ["tops", "bottoms", "outers"]:
            cat_dir = closet_path / broad_cat
            if not cat_dir.exists():
                continue

            count = 0
            for file in sorted(cat_dir.iterdir()):
                if file.suffix.lower() in IMAGE_EXTENSIONS:
                    items_to_process.append((str(file), broad_cat))
                    count += 1
                    if limit_per_category and count >= limit_per_category:
                        break

        print(f"[WardrobeBuilder] 처리할 이미지: {len(items_to_process)}개")

        # 배치 처리
        return self.add_items(items_to_process)

    def add_item(
        self,
        image_path: str,
        broad_cat: str,
        detail_cat: str = None
    ) -> Tuple[bool, Optional[WardrobeItem]]:
        """
        단일 아이템 추가 (upsert)

        Args:
            image_path: 이미지 파일 경로
            broad_cat: 대분류 ("tops", "bottoms", "outers")
            detail_cat: 세부분류 (None이면 자동 분류)

        Returns:
            (성공 여부, WardrobeItem 또는 None)
        """
        # broad_cat 정규화
        broad_cat = BROAD_CAT_MAPPING.get(broad_cat, broad_cat)
        if broad_cat not in BROAD_CAT_TO_KOREAN:
            print(f"[ERROR] 지원하지 않는 카테고리: {broad_cat}")
            return (False, None)

        # 파이프라인 실행
        item = self._process_item(image_path, broad_cat, detail_cat)
        if item is None:
            return (False, None)

        # ChromaDB 저장
        success = self._save_to_chromadb(item)
        return (success, item if success else None)

    def add_items(
        self,
        items: List[Tuple[str, str, Optional[str]]]
    ) -> BuildResult:
        """
        여러 아이템 배치 추가

        Args:
            items: [(image_path, broad_cat, detail_cat), ...]
                   또는 [(image_path, broad_cat), ...]
        """
        total = len(items)
        added = 0
        skipped = 0
        failed = []

        for i, item_tuple in enumerate(items):
            # 튜플 언패킹 (2개 또는 3개 요소)
            if len(item_tuple) == 2:
                image_path, broad_cat = item_tuple
                detail_cat = None
            else:
                image_path, broad_cat, detail_cat = item_tuple

            print(f"[{i+1}/{total}] 처리 중: {Path(image_path).name}")

            success, item = self.add_item(image_path, broad_cat, detail_cat)

            if success:
                added += 1
                print(f"  → 추가됨: {item.detail_cat}")
            else:
                failed.append({"path": image_path, "reason": "처리 실패"})

        return BuildResult(
            success=len(failed) == 0,
            total_items=total,
            added_items=added,
            skipped_items=skipped,
            failed_items=failed,
            collection_name=self.collection_name
        )

    def _process_item(
        self,
        image_path: str,
        broad_cat: str,
        detail_cat: str = None
    ) -> Optional[WardrobeItem]:
        """
        단일 아이템 처리 파이프라인

        1. 이미지 로드
        2. (옵션) 배경 제거
        3. 세부 카테고리 분류
        4. 임베딩 생성
        5. WardrobeItem 반환
        """
        try:
            # 1. 이미지 로드
            if not os.path.exists(image_path):
                print(f"[ERROR] 파일 없음: {image_path}")
                return None

            image = Image.open(image_path).convert("RGB")

            # 2. 배경 제거 (선택적)
            if self.use_background_removal:
                image = self.classifier.remove_background(image)

            # 3. 세부 카테고리 분류
            if detail_cat is None:
                korean_cat = BROAD_CAT_TO_KOREAN[broad_cat]
                detail_cat = self.classifier.classify_item(image, korean_cat)

            # 4. 임베딩 생성
            embedding = self.embedder.embed_image(image_path)
            if embedding is None:
                print(f"[ERROR] 임베딩 생성 실패: {image_path}")
                return None

            # 5. WardrobeItem 생성
            item_id = self._generate_item_id(image_path, broad_cat)

            return WardrobeItem(
                item_id=item_id,
                user_id=self.user_id,
                image_path=image_path,
                broad_cat=broad_cat,
                detail_cat=detail_cat,
                embedding=embedding,
                created_at=datetime.now().isoformat()
            )

        except Exception as e:
            print(f"[ERROR] 아이템 처리 실패: {image_path} - {e}")
            return None

    def _generate_item_id(self, image_path: str, broad_cat: str) -> str:
        """아이템 고유 ID 생성: user_id:broad_cat/filename"""
        filename = Path(image_path).name
        return f"{self.user_id}:{broad_cat}/{filename}"

    def _save_to_chromadb(self, item: WardrobeItem) -> bool:
        """ChromaDB에 저장 (upsert)"""
        try:
            metadata = {
                "entity_type": "item",
                "user_id": item.user_id,
                "item_key": f"{item.broad_cat}/{Path(item.image_path).name}",
                "broad_cat": item.broad_cat,
                "detail_cat": item.detail_cat,
                "source": "wardrobe",
                "created_at": item.created_at,
            }

            self.collection.upsert(
                ids=[item.item_id],
                embeddings=[item.embedding],
                metadatas=[metadata]
            )

            return True
        except Exception as e:
            print(f"[ERROR] ChromaDB 저장 실패: {item.item_id} - {e}")
            return False

    def _delete_user_items(self):
        """현재 사용자의 모든 아이템 삭제"""
        try:
            # user_id로 필터링하여 삭제
            self.collection.delete(
                where={"user_id": self.user_id}
            )
            print(f"[WardrobeBuilder] 사용자 {self.user_id}의 기존 데이터 삭제됨")
        except Exception as e:
            print(f"[WARN] 기존 데이터 삭제 실패: {e}")

    def get_statistics(self) -> Dict[str, Any]:
        """사용자 옷장 통계 조회"""
        try:
            # 사용자 아이템 조회
            results = self.collection.get(
                where={"user_id": self.user_id},
                include=["metadatas"]
            )

            total = len(results["ids"])

            # 카테고리별 집계
            by_broad_cat = {}
            by_detail_cat = {}

            for meta in results["metadatas"]:
                broad = meta.get("broad_cat", "unknown")
                detail = meta.get("detail_cat", "unknown")

                by_broad_cat[broad] = by_broad_cat.get(broad, 0) + 1
                by_detail_cat[detail] = by_detail_cat.get(detail, 0) + 1

            return {
                "user_id": self.user_id,
                "total_items": total,
                "by_broad_cat": by_broad_cat,
                "by_detail_cat": by_detail_cat
            }
        except Exception as e:
            print(f"[ERROR] 통계 조회 실패: {e}")
            return {"user_id": self.user_id, "total_items": 0}

    def get_item(self, item_id: str) -> Optional[WardrobeItem]:
        """ID로 아이템 조회"""
        try:
            results = self.collection.get(
                ids=[item_id],
                include=["embeddings", "metadatas"]
            )

            if not results["ids"]:
                return None

            meta = results["metadatas"][0]
            return WardrobeItem(
                item_id=item_id,
                user_id=meta.get("user_id", ""),
                image_path=meta.get("item_key", ""),
                broad_cat=meta.get("broad_cat", ""),
                detail_cat=meta.get("detail_cat", ""),
                embedding=results["embeddings"][0],
                created_at=meta.get("created_at", "")
            )
        except Exception as e:
            print(f"[ERROR] 아이템 조회 실패: {item_id} - {e}")
            return None

    def delete_item(self, item_id: str) -> bool:
        """아이템 삭제"""
        try:
            self.collection.delete(ids=[item_id])
            return True
        except Exception as e:
            print(f"[ERROR] 아이템 삭제 실패: {item_id} - {e}")
            return False


# 테스트용 메인
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="WardrobeBuilder 테스트")
    parser.add_argument("--user", type=str, required=True, help="사용자 ID")
    parser.add_argument("--closet", type=str, required=True, help="옷장 디렉토리")
    parser.add_argument("--limit", type=int, default=None, help="카테고리당 최대 개수")

    args = parser.parse_args()

    builder = WardrobeBuilder(user_id=args.user)
    result = builder.build_from_directory(args.closet, limit_per_category=args.limit)

    print("\n=== 결과 ===")
    print(f"성공: {result.success}")
    print(f"총 아이템: {result.total_items}")
    print(f"추가됨: {result.added_items}")
    print(f"실패: {len(result.failed_items)}")

    print("\n=== 통계 ===")
    stats = builder.get_statistics()
    print(f"총 아이템: {stats['total_items']}")
    print(f"카테고리별: {stats['by_broad_cat']}")
