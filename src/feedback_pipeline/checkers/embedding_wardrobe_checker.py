"""
임베딩 기반 옷장 체크 구현체 (FashionCLIP + ChromaDB)
"""
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# FashionCLIP embedder import
sys.path.append(str(Path(__file__).parent.parent.parent / "embedding_generator"))
from generate_fashionclip_embeddings import FashionCLIPEmbedder

from ..interfaces.wardrobe_checker import WardrobeCheckerInterface, WardrobeCheckResult
from ..utils.detail_category import (
    DETAIL_ALIAS_TO_CANONICAL,
    MAIN_CATEGORY_ALIASES,
    normalize_detail_category,
    normalize_main_category,
)

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
    DETAIL_FIELD = "detail_cat"

    def __init__(
        self,
        threshold: float = 0.2,  # FashionCLIP 텍스트-이미지 유사도는 보통 0.2~0.35
        chroma_path: str = "data/chroma_wardrobe",
        collection_name: str = "wardrobe",
        device: str = None,
        embedder: Optional[FashionCLIPEmbedder] = None
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

        # FashionCLIP 모델 로드 (주입된 것이 없으면 생성)
        if embedder:
            self.embedder = embedder
        else:
            try:
                self.embedder = FashionCLIPEmbedder(device=device, use_fp16=False)  # CPU에서는 FP16 비활성화
            except Exception as e:
                print(f"[경고] FashionCLIP 로드 실패: {e}")
                self.embedder = None

        self._main_cat_alias_to_havati = MAIN_CATEGORY_ALIASES
        self._detail_alias_to_canonical = DETAIL_ALIAS_TO_CANONICAL

    @staticmethod
    def _meta_values(meta: Optional[Dict[str, Any]], *keys: str) -> List[str]:
        if not meta:
            return []
        values: List[str] = []
        for key in keys:
            raw = meta.get(key)
            if raw is None:
                continue
            if isinstance(raw, (list, tuple, set)):
                values.extend([str(v).strip() for v in raw if str(v).strip()])
            else:
                candidate = str(raw).strip()
                if candidate:
                    values.append(candidate)
        deduped = []
        for value in values:
            if value not in deduped:
                deduped.append(value)
        return deduped

    @staticmethod
    def _normalize_main_category(value: str) -> str:
        return normalize_main_category(value)

    def _normalize_detail_category(self, value: str) -> str:
        return normalize_detail_category(value)

    @staticmethod
    def _main_filter_condition(main_categories: List[str]) -> Optional[Dict[str, Any]]:
        if not main_categories:
            return None
        if len(main_categories) == 1:
            cat = main_categories[0]
            return {
                "$or": [
                    {"broad_cat": cat},
                    {"category_main": cat},
                    {"broad_cat": cat.lower()},
                    {"category_main": cat.lower()},
                ]
            }
        return {
            "$or": [
                {
                    "$or": [
                        {"broad_cat": cat},
                        {"category_main": cat},
                        {"broad_cat": cat.lower()},
                        {"category_main": cat.lower()},
                    ]
                }
                for cat in main_categories
            ]
        }

    @staticmethod
    def _detail_filter_condition(detail_cats: List[str]) -> Optional[Dict[str, Any]]:
        if not detail_cats:
            return None
        if len(detail_cats) == 1:
            detail = detail_cats[0]
            return {EmbeddingWardrobeChecker.DETAIL_FIELD: detail}
        return {
            "$or": [
                {EmbeddingWardrobeChecker.DETAIL_FIELD: detail}
                for detail in detail_cats
            ]
        }

    def can_fulfill(
        self,
        requirements: List[str],
        user_id: str,
        context: Optional[Dict[str, Any]] = None,
        target_categories: Optional[List[str]] = None,
        target_detail_cats: Optional[List[str]] = None,
        avoid_detail_cats: Optional[List[str]] = None,
    ) -> WardrobeCheckResult:
        """
        옷장에서 요구사항을 충족하는 재생성 후보 찾기

        Args:
            requirements: QueryBuilder에서 정제된 쿼리 리스트
            user_id: 사용자 ID
            context: 추가 컨텍스트 (previous_items 등)
            target_categories: 필터링할 대카테고리 리스트 (예: ['tops', 'bottoms'])
            target_detail_cats: 필터링할 세부카테고리 리스트 (예: ['Knitwear', 'Sweatshirt'])
        """
        # Fallback: ChromaDB 또는 embedder 없으면 더미 동작
        if not self.wardrobe_collection or not self.embedder:
            return WardrobeCheckResult(
                is_possible=True,
                matching_items=[],
                reason="[Dummy] ChromaDB 또는 FashionCLIP 없음",
                confidence=0.5
            )

        # 쿼리 텍스트 결합
        query_text = " ".join(requirements) if requirements else "casual outfit"

        # FashionCLIP으로 쿼리 임베딩 생성
        try:
            query_embedding = self.embedder.embed_text(query_text)
        except Exception as e:
            return WardrobeCheckResult(
                is_possible=False,
                matching_items=[],
                reason=f"임베딩 생성 실패: {e}",
                confidence=0.0
            )

        # ChromaDB 유사도 검색 + 4. 소수 아이템 bypass/threshold 필터링
        candidates = []
        all_similarities = []
        bypass_details = []
        seen_ids = set()
        total_results = 0
        avoid_detail_lower = {str(x).strip().lower() for x in (avoid_detail_cats or []) if str(x).strip()}
        normalized_target_detail_cats = []
        if target_detail_cats:
            for raw in target_detail_cats:
                normalized = self._normalize_detail_category(raw)
                if normalized:
                    normalized_target_detail_cats.append(normalized)
            normalized_target_detail_cats = list(dict.fromkeys(normalized_target_detail_cats))
        print(
            "  [WardrobeChecker] query="
            f"'{query_text}', target_categories={target_categories or []}, "
            f"target_detail_cats={normalized_target_detail_cats}, avoid_detail_cats={sorted(avoid_detail_lower)}"
        )

        def _make_where_filter(detail_cat: Optional[str] = None):
            filters = []
            if user_id:
                filters.append({"user_id": user_id})
            if target_categories:
                normalized_main = [
                    self._normalize_main_category(cat)
                    for cat in target_categories
                ]
                normalized_main = [cat for cat in normalized_main if cat]
                normalized_main = normalized_main or list(dict.fromkeys(target_categories))
                main_filter = EmbeddingWardrobeChecker._main_filter_condition(normalized_main)
                if main_filter:
                    filters.append(main_filter)
            if detail_cat:
                normalized_detail = self._normalize_detail_category(detail_cat)
                if normalized_detail:
                    detail_filter = self._detail_filter_condition([normalized_detail])
                    if detail_filter:
                        filters.append(detail_filter)
            elif normalized_target_detail_cats and len(normalized_target_detail_cats) == 1:
                normalized_detail = normalized_target_detail_cats[0]
                if normalized_detail:
                    detail_filter = self._detail_filter_condition([normalized_detail])
                    if detail_filter:
                        filters.append(detail_filter)

            if len(filters) == 1:
                return filters[0]
            if len(filters) > 1:
                return {"$and": filters}
            return None

        def _collect_from_results(results: Dict[str, Any], bypass_threshold: bool):
            local_candidates = []
            local_sims = []
            count = len(results['ids'][0]) if results['ids'] and results['ids'][0] else 0
            if count == 0:
                return local_candidates, local_sims, count

            for i, distance in enumerate(results['distances'][0]):
                similarity = 1 - distance
                local_sims.append(similarity)
                if bypass_threshold or similarity >= self.threshold:
                    cid = results['ids'][0][i]
                    if cid in seen_ids:
                        continue
                    meta = results['metadatas'][0][i] if results['metadatas'] and results['metadatas'][0] else {}
                    detail_value = str(meta.get(self.DETAIL_FIELD, "") or "").strip()
                    normalized_details = [self._normalize_detail_category(detail_value)] if detail_value else []
                    detail_cat = str(normalized_details[0]).strip().lower() if normalized_details else ""
                    if avoid_detail_lower and detail_cat in avoid_detail_lower:
                        continue
                    seen_ids.add(cid)
                    local_candidates.append({
                        'id': cid,
                        'similarity': similarity,
                        'metadata': meta
                    })
            return local_candidates, local_sims, count

        try:
            # detail_cat 복수 지정 시 카테고리별로 분리 검색/분리 bypass 적용
            if normalized_target_detail_cats and len(normalized_target_detail_cats) > 1:
                for detail_cat in normalized_target_detail_cats:
                    where_filter = _make_where_filter(detail_cat=detail_cat)
                    results = self.wardrobe_collection.query(
                        query_embeddings=[query_embedding],
                        n_results=20,
                        where=where_filter,
                    )
                    detail_total = len(results['ids'][0]) if results['ids'] and results['ids'][0] else 0
                    total_results += detail_total
                    detail_bypass = detail_total <= 3
                    print(
                        f"  [WardrobeChecker] detail='{detail_cat}' raw_hits={detail_total}, "
                        f"bypass={detail_bypass}, where={where_filter}"
                    )
                    if detail_bypass and detail_total > 0:
                        bypass_details.append(f"{detail_cat}:{detail_total}")
                        print(
                            f"  [INFO] 세부카테고리 '{detail_cat}' 결과 {detail_total}개 (<=3) "
                            "→ 임베딩 비교 생략, 전부 반환"
                        )
                    part_candidates, part_sims, _ = _collect_from_results(results, detail_bypass)
                    print(
                        f"  [WardrobeChecker] detail='{detail_cat}' kept_after_threshold={len(part_candidates)}"
                    )
                    candidates.extend(part_candidates)
                    all_similarities.extend(part_sims)
            else:
                where_filter = _make_where_filter()
                results = self.wardrobe_collection.query(
                    query_embeddings=[query_embedding],
                    n_results=20,
                    where=where_filter,
                )
                total_results = len(results['ids'][0]) if results['ids'] and results['ids'][0] else 0
                bypass_threshold = bool(normalized_target_detail_cats) and total_results <= 3
                print(
                    f"  [WardrobeChecker] raw_hits={total_results}, bypass={bypass_threshold}, where={where_filter}"
                )
                if bypass_threshold and total_results > 0:
                    bypass_details.append(f"all:{total_results}")
                    print(f"  [INFO] 세부카테고리 결과 {total_results}개 (<=3) → 임베딩 비교 생략, 전부 반환")
                part_candidates, part_sims, _ = _collect_from_results(results, bypass_threshold)
                print(f"  [WardrobeChecker] kept_after_threshold={len(part_candidates)}")
                candidates.extend(part_candidates)
                all_similarities.extend(part_sims)

            if all_similarities:
                print(f"  [DEBUG] 유사도 범위: {min(all_similarities):.3f} ~ {max(all_similarities):.3f}")
        except Exception as e:
            return WardrobeCheckResult(
                is_possible=False,
                matching_items=[],
                reason=f"ChromaDB 검색 실패: {e}",
                confidence=0.0
            )

        # 이전 추천 아이템 필터링 (중복 제거)
        previous_items = []
        if context and 'previous_items' in context:
            previous_items = context['previous_items']

        if previous_items and candidates:
            before_prev_filter = len(candidates)
            candidates = self._filter_previous_items(candidates, previous_items)
            print(
                f"  [WardrobeChecker] previous_item_filter removed={before_prev_filter - len(candidates)}, "
                f"remaining={len(candidates)}"
            )

        # 부정 카테고리 기반 요청에서 후보가 과소(<=1)하면
        # 동일 파트 내 비제외 카테고리로 fallback 확장한다.
        if avoid_detail_lower and target_categories and len(target_categories) == 1 and len(candidates) <= 1:
            try:
                fallback_where = _make_where_filter(detail_cat=None)
                fallback_results = self.wardrobe_collection.query(
                    query_embeddings=[query_embedding],
                    n_results=50,
                    where=fallback_where,
                )
                fallback_raw = len(fallback_results['ids'][0]) if fallback_results['ids'] and fallback_results['ids'][0] else 0
                print(
                    f"  [WardrobeChecker] fallback_triggered raw_hits={fallback_raw}, where={fallback_where}"
                )
                fallback_candidates, _, _ = _collect_from_results(fallback_results, bypass_threshold=True)
                print(
                    f"  [WardrobeChecker] fallback_kept_before_prev_filter={len(fallback_candidates)}"
                )
                if previous_items and fallback_candidates:
                    fallback_before_prev = len(fallback_candidates)
                    fallback_candidates = self._filter_previous_items(fallback_candidates, previous_items)
                    print(
                        f"  [WardrobeChecker] fallback_prev_filter removed="
                        f"{fallback_before_prev - len(fallback_candidates)}, remaining={len(fallback_candidates)}"
                    )
                # 기존 후보 유지 + 중복 제거 후 확장
                merged = {c["id"]: c for c in candidates}
                for c in fallback_candidates:
                    merged.setdefault(c["id"], c)
                candidates = list(merged.values())
                print(f"  [WardrobeChecker] fallback_merged_total={len(candidates)}")
            except Exception:
                pass

        # 결과 반환
        is_possible = len(candidates) > 0
        confidence = sum(c['similarity'] for c in candidates) / len(candidates) if candidates else 0.0

        # matching_items는 아이템 ID 문자열 리스트로 반환
        matching_items_ids = [c['id'] for c in candidates]
        candidate_pool = self._build_candidate_pool(candidates)
        pool_counts = {k: len(v) for k, v in candidate_pool.items()}
        print(f"  [WardrobeChecker] final_candidates={len(candidates)}, pool_counts={pool_counts}")
        for scope in ("TOP", "BOTTOM", "OUTER"):
            scoped = candidate_pool.get(scope, [])
            if scoped:
                print(f"  [WardrobeChecker] pool_preview[{scope}]={scoped[:3]}")

        if bypass_details:
            reason = (
                f"검색 완료: {len(candidates)}개 후보 "
                f"(detail별 bypass 적용: {', '.join(bypass_details)})"
            )
        else:
            reason = f"검색 완료: {len(candidates)}개 후보 (threshold={self.threshold})"
        if previous_items:
            reason += f", 이전 {len(previous_items)}개 제외"

        return WardrobeCheckResult(
            is_possible=is_possible,
            matching_items=matching_items_ids,
            candidate_pool=candidate_pool,
            reason=reason,
            confidence=confidence
        )

    def _build_candidate_pool(self, candidates: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        pool_with_scores: Dict[str, List[Dict[str, Any]]] = {}
        for candidate in candidates:
            cid = str(candidate.get("id", ""))
            filename = cid.split("/")[-1] if "/" in cid else cid.split(":")[-1]
            meta = candidate.get("metadata", {}) or {}
            broad_candidates = self._meta_values(meta, "broad_cat", "category_main")
            broad = str(broad_candidates[0]).lower() if broad_candidates else ""

            cat = None
            if broad in {"shirt", "top", "tops", "상의"}:
                cat = "TOP"
            elif broad in {"pant", "pants", "bottom", "bottoms", "바지", "하의"}:
                cat = "BOTTOM"
            elif broad in {"outer", "outers", "아우터"}:
                cat = "OUTER"
            elif "shirt/" in cid or "tops_" in cid:
                cat = "TOP"
            elif "pant/" in cid or "bottoms_" in cid:
                cat = "BOTTOM"
            elif "outer/" in cid or "outers_" in cid:
                cat = "OUTER"

            if cat and filename:
                pool_with_scores.setdefault(cat, []).append({
                    "filename": filename,
                    "similarity": float(candidate.get("similarity", 0.0) or 0.0),
                })

        # 생성팀 전달 정책: 파트별 후보는 최대 3개로 제한
        pool: Dict[str, List[str]] = {}
        for cat, items in pool_with_scores.items():
            items.sort(key=lambda x: x["similarity"], reverse=True)
            seen = set()
            picked = []
            for item in items:
                filename = item["filename"]
                if filename in seen:
                    continue
                seen.add(filename)
                picked.append(filename)
                if len(picked) >= 3:
                    break
            if picked:
                pool[cat] = picked

        return pool

    def _filter_previous_items(
        self,
        candidates: List[Dict],
        previous_items: List[str]
    ) -> List[Dict]:
        """
        이전 추천과 동일한 아이템(ID/파일명)만 제거

        유사도 기반 제거는 같은 카테고리 후보(예: 다른 니트/다른 데님)까지
        과도하게 제외할 수 있으므로 exact match만 제외한다.
        """
        if not previous_items:
            return candidates

        previous_names = {self._normalize_item_name(x) for x in previous_items if x}
        if not previous_names:
            return candidates

        filtered = []
        for candidate in candidates:
            candidate_name = self._normalize_item_name(candidate.get("id", ""))
            if candidate_name in previous_names:
                continue
            filtered.append(candidate)

        return filtered

    def _normalize_item_name(self, raw: str) -> str:
        text = str(raw or "").strip()
        if not text:
            return text
        if ":" in text:
            text = text.split(":")[-1]
        if "/" in text:
            text = text.split("/")[-1]
        return text.lower()

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """코사인 유사도 계산"""
        if not CHROMADB_AVAILABLE:
            return 0.0

        v1 = np.array(vec1)
        v2 = np.array(vec2)
        return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
