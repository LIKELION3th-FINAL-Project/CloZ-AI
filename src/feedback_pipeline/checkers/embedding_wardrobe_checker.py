"""
임베딩 기반 옷장 체크 구현체 (FashionCLIP + ChromaDB)
"""
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# FashionCLIP embedder import
sys.path.append(str(Path(__file__).parent.parent.parent / "scripts"))
from generate_fashionclip_embeddings import FashionCLIPEmbedder

from ..interfaces.wardrobe_checker import WardrobeCheckerInterface, WardrobeCheckResult

try:
    import chromadb
    import numpy as np
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False


class EmbeddingWardrobeChecker(WardrobeCheckerInterface):
    """
    임베딩 기반 옷장 체크 구현체

    FashionCLIP 임베딩으로 사용자 옷장에서 피드백 요구사항과 유사한 아이템 검색
    이전 추천과 다른 새로운 재생성 후보 반환
    """

    def __init__(
        self,
        threshold: float = 0.2,  # FashionCLIP 텍스트-이미지 유사도는 보통 0.2~0.35
        chroma_path: str = "data/chroma_db",
        collection_name: str = "wardrobe",
        device: str = None
    ):
        """
        초기화

        Args:
            threshold: 유사도 임계값 (FashionCLIP 기준 0.2~0.3 권장)
            chroma_path: ChromaDB 경로
            collection_name: 옷장 컬렉션 이름
            device: FashionCLIP 디바이스 ("cuda", "mps", or "cpu")
        """
        self.threshold = threshold

        if not CHROMADB_AVAILABLE:
            print("[경고] chromadb가 설치되지 않음. Dummy 모드로 동작")
            self.chroma_client = None
            self.wardrobe_collection = None
            self.embedder = None
            return

        # ChromaDB 클라이언트 초기화
        self.chroma_client = chromadb.PersistentClient(path=chroma_path)
        try:
            self.wardrobe_collection = self.chroma_client.get_collection(collection_name)
        except Exception as e:
            print(f"[경고] 컬렉션 '{collection_name}' 없음: {e}")
            self.wardrobe_collection = None
            self.embedder = None
            return

        # FashionCLIP 모델 로드
        try:
            self.embedder = FashionCLIPEmbedder(device=device, use_fp16=False)  # CPU에서는 FP16 비활성화
        except Exception as e:
            print(f"[경고] FashionCLIP 로드 실패: {e}")
            self.embedder = None

    def can_fulfill(
        self,
        requirements: List[str],
        user_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> WardrobeCheckResult:
        """
        옷장에서 요구사항을 충족하는 재생성 후보 찾기

        Args:
            requirements: QueryBuilder에서 정제된 쿼리 리스트
            user_id: 사용자 ID
            context: 추가 컨텍스트 (previous_items 등)

        Returns:
            WardrobeCheckResult(is_possible, matching_items, confidence)
        """
        # Fallback: ChromaDB 또는 embedder 없으면 더미 동작
        if not self.wardrobe_collection or not self.embedder:
            return WardrobeCheckResult(
                is_possible=True,
                matching_items=[],
                reason="[Dummy] ChromaDB 또는 FashionCLIP 없음",
                confidence=0.5
            )

        # 1. 쿼리 텍스트 결합
        query_text = " ".join(requirements) if requirements else "casual outfit"

        # 2. FashionCLIP으로 쿼리 임베딩 생성
        try:
            query_embedding = self.embedder.embed_text(query_text)
        except Exception as e:
            return WardrobeCheckResult(
                is_possible=False,
                matching_items=[],
                reason=f"임베딩 생성 실패: {e}",
                confidence=0.0
            )

        # 3. ChromaDB 유사도 검색
        try:
            results = self.wardrobe_collection.query(
                query_embeddings=[query_embedding],
                n_results=20,  # 충분한 후보 가져오기
                # where={"user_id": user_id}  # 메타데이터에 user_id 있을 경우
            )
        except Exception as e:
            return WardrobeCheckResult(
                is_possible=False,
                matching_items=[],
                reason=f"ChromaDB 검색 실패: {e}",
                confidence=0.0
            )

        # 4. Threshold 필터링 (유사도 >= threshold)
        candidates = []
        all_similarities = []  # 디버깅용

        if results['ids'] and results['ids'][0] and results['distances'] and results['distances'][0]:
            for i, distance in enumerate(results['distances'][0]):
                # ChromaDB cosine distance: 1 - cosine_similarity
                # 따라서 similarity = 1 - distance
                similarity = 1 - distance

                all_similarities.append(similarity)

                if similarity >= self.threshold:
                    candidates.append({
                        'id': results['ids'][0][i],
                        'similarity': similarity,
                        'metadata': results['metadatas'][0][i] if results['metadatas'] and results['metadatas'][0] else {}
                    })

            # 디버깅: 유사도 범위 출력
            if all_similarities:
                print(f"  [DEBUG] 유사도 범위: {min(all_similarities):.3f} ~ {max(all_similarities):.3f}")

        # 5. 이전 추천 아이템 필터링 (중복 제거)
        previous_items = []
        if context and 'previous_items' in context:
            previous_items = context['previous_items']

        if previous_items and candidates:
            candidates = self._filter_previous_items(candidates, previous_items)

        # 6. 결과 반환
        is_possible = len(candidates) > 0
        confidence = sum(c['similarity'] for c in candidates) / len(candidates) if candidates else 0.0

        # matching_items는 아이템 ID 문자열 리스트로 반환
        matching_items_ids = [c['id'] for c in candidates]

        reason = f"검색 완료: {len(candidates)}개 후보 (threshold={self.threshold})"
        if previous_items:
            reason += f", 이전 {len(previous_items)}개 제외"

        return WardrobeCheckResult(
            is_possible=is_possible,
            matching_items=matching_items_ids,
            reason=reason,
            confidence=confidence
        )

    def _filter_previous_items(
        self,
        candidates: List[Dict],
        previous_items: List[str]
    ) -> List[Dict]:
        """
        이전에 추천된 아이템과 너무 유사한 것 제거

        Args:
            candidates: 현재 후보 리스트
            previous_items: 이전 추천 아이템 ID 리스트

        Returns:
            필터링된 후보 리스트
        """
        if not previous_items or not self.wardrobe_collection:
            return candidates

        try:
            # 이전 아이템 임베딩 가져오기
            previous_embeddings_result = self.wardrobe_collection.get(
                ids=previous_items,
                include=['embeddings']
            )

            if not previous_embeddings_result['embeddings']:
                return candidates

            previous_embeddings = previous_embeddings_result['embeddings']

            # 현재 후보와 이전 아이템 간 유사도 계산
            filtered = []
            for candidate in candidates:
                # 후보 아이템 임베딩 가져오기
                candidate_result = self.wardrobe_collection.get(
                    ids=[candidate['id']],
                    include=['embeddings']
                )

                if not candidate_result['embeddings']:
                    continue

                candidate_embedding = candidate_result['embeddings'][0]

                # 이전 아이템과의 최대 유사도 계산
                max_similarity_to_previous = 0
                for prev_emb in previous_embeddings:
                    sim = self._cosine_similarity(candidate_embedding, prev_emb)
                    max_similarity_to_previous = max(max_similarity_to_previous, sim)

                # Threshold 이하만 유지 (90% 이상 유사하면 제외)
                if max_similarity_to_previous < 0.9:
                    filtered.append(candidate)

            return filtered

        except Exception as e:
            print(f"[경고] 이전 아이템 필터링 실패: {e}")
            return candidates

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """코사인 유사도 계산"""
        if not CHROMADB_AVAILABLE:
            return 0.0

        v1 = np.array(vec1)
        v2 = np.array(vec2)
        return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
