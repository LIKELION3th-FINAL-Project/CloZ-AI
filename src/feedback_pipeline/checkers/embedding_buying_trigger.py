"""
임베딩 기반 상품 추천 (FashionCLIP + ChromaDB)

Havati 상품 전용
"""
import sys
import json
from pathlib import Path
from typing import Dict, List, Optional, Any

# FashionCLIP embedder import
sys.path.append(str(Path(__file__).parent.parent.parent / "embedding_generator"))
from generate_fashionclip_embeddings import FashionCLIPEmbedder

from ..interfaces.buying_trigger import BuyingTriggerInterface, BuyingRecommendation, ProductRecommendation
from ..models import FeedbackScope, OutfitSet
from ..utils.color_utils import get_brightness, get_colors_by_brightness, LIGHT_COLORS, DARK_COLORS
from ..utils.detail_category import (
    DETAIL_ALIAS_TO_CANONICAL,
    DETAIL_GROUPS,
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


class EmbeddingBuyingTrigger(BuyingTriggerInterface):
    """
    임베딩 기반 상품 추천 구현체

    FashionCLIP 임베딩으로 Havati 상품에서 피드백 요구사항과 유사한 상품 검색

    가중치:
        similarity * 0.55 + color_bonus(0.30) + style_bonus(max 0.15) + brightness_bonus(max 0.15)
    """
    DETAIL_FIELD = "category_sub"

    _DEFAULT_DETAIL_ALIAS_TO_CANONICAL = DETAIL_ALIAS_TO_CANONICAL

    def __init__(
        self,
        threshold: float = 0.15,
        chroma_path: str = "data/chroma_db_products",
        collection_name: str = "products",
        metadata_path: str = "data/product_embedding_checkpoint.json",
        device: str = None,
        embedder: Optional[FashionCLIPEmbedder] = None
    ):
        """
        초기화

        Args:
            threshold: 유사도 임계값 (FashionCLIP 기준 0.15~0.25 권장)
            chroma_path: ChromaDB 경로
            collection_name: Havati 상품 컬렉션 이름
            metadata_path: Havati 상품 메타데이터 JSON 경로 (color, style_tags 포함)
            device: FashionCLIP 디바이스 ("cuda", "mps", or "cpu")
        """
        self.threshold = threshold
        self.metadata_path = metadata_path
        self._metadata_cache: Dict[str, Dict] = {}

        if not CHROMADB_AVAILABLE:
            print("[경고] chromadb가 설치되지 않음. Dummy 모드로 동작")
            self.chroma_client = None
            self.collection = None
            self.embedder = None
            return

        # ChromaDB 클라이언트 초기화
        self.chroma_client = chromadb.PersistentClient(path=chroma_path)
        try:
            self.collection = self.chroma_client.get_collection(collection_name)
        except Exception as e:
            print(f"[경고] 컬렉션 '{collection_name}' 없음: {e}")
            self.collection = None
            self.embedder = None
            return

        # FashionCLIP 모델 로드 (주입된 것이 없으면 생성)
        if embedder:
            self.embedder = embedder
        else:
            try:
                self.embedder = FashionCLIPEmbedder(device=device, use_fp16=False)
            except Exception as e:
                print(f"[경고] FashionCLIP 로드 실패: {e}")
                self.embedder = None

        # 메타데이터 로드
        self._load_metadata()
        self._detail_alias_to_canonical = dict(self._DEFAULT_DETAIL_ALIAS_TO_CANONICAL)
        self._main_cat_alias_to_havati = MAIN_CATEGORY_ALIASES
        self._detail_groups = DETAIL_GROUPS

    def _load_metadata(self):
        """Havati 상품 메타데이터 로드 (color, style_tags 포함)"""
        try:
            metadata_file = Path(self.metadata_path)
            if not metadata_file.exists():
                project_root = Path(__file__).parent.parent.parent
                metadata_file = project_root / self.metadata_path

            if metadata_file.exists():
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    products = json.load(f)
                    for product_id, data in products.items():
                        self._metadata_cache[product_id] = data
                print(f"[INFO] Havati 메타데이터 로드 완료: {len(self._metadata_cache)}개 상품")
            else:
                print(f"[경고] 메타데이터 파일 없음: {self.metadata_path}")
        except Exception as e:
            print(f"[경고] 메타데이터 로드 실패: {e}")

    def recommend(
        self,
        original_prompt: str,
        feedback_text: str,
        feedback_scope: FeedbackScope,
        current_outfit: Optional[OutfitSet] = None,
        limit: int = 5,
        context: Optional[Dict[str, Any]] = None
    ) -> BuyingRecommendation:
        """
        피드백 기반 상품 추천

        Args:
            original_prompt: 원본 코디 요청
            feedback_text: 피드백 텍스트 (정제된 쿼리)
            feedback_scope: 피드백 범위
            current_outfit: 현재 코디
            limit: 추천 개수
            context: 추가 컨텍스트
                - structured_query: QueryBuilder의 구조화된 정보 (requirements_en 포함)
                - avoid_attributes: 회피할 속성
        """
        target_category = self.get_target_category(feedback_scope)

        # Fallback: ChromaDB 또는 embedder 없으면 더미 동작
        if not self.collection or not self.embedder:
            return BuyingRecommendation(
                success=False,
                products=[],
                reasoning="[Dummy] ChromaDB 또는 FashionCLIP 없음",
                target_category=target_category
            )

        # 검색 쿼리 결정 (영어 우선)
        search_query = feedback_text
        avoid_colors = []
        prefer_colors = []
        prefer_styles = []
        prefer_brightness = None  # "light" or "dark"
        target_detail_cats = []  # 세부카테고리 필터
        avoid_detail_cats = []   # 세부카테고리 제외

        if context:
            # 영어 요구사항 사용
            if 'structured_query' in context:
                structured = context['structured_query']
                if isinstance(structured, dict):
                    if structured.get('requirements_en'):
                        search_query = " ".join(structured['requirements_en'])
                    # 세부카테고리 추출
                    if structured.get('target_detail_cats'):
                        target_detail_cats = structured['target_detail_cats']
                    if structured.get('avoid_detail_cats'):
                        avoid_detail_cats = structured['avoid_detail_cats']
                    # 선호 색상 추출
                    prefer_colors = structured.get('prefer_colors', [])
                    # 제한사항에서 회피 색상 추출
                    for constraint in structured.get('constraints', []):
                        constraint_lower = constraint.lower()
                        if '검은색' in constraint or 'black' in constraint_lower:
                            avoid_colors.append('black')
                        if '흰색' in constraint or 'white' in constraint_lower:
                            avoid_colors.append('white')
                    # mood에서 스타일 추출
                    prefer_styles = structured.get('mood', [])
                    # 밝기 선호도 추출
                    prefer_brightness = structured.get('prefer_brightness')

            # 복수 피드백 범위가 전달되면 단일 category_main 필터를 강제하지 않는다.
            scopes = context.get("feedback_scopes")
            if isinstance(scopes, list) and len(scopes) > 1:
                target_category = None

            # avoid_attributes에서도 추출
            if 'avoid_attributes' in context:
                avoid_attrs = context['avoid_attributes']
                if isinstance(avoid_attrs, dict):
                    avoid_colors.extend(avoid_attrs.get('colors', []))

        target_detail_cats = self._sanitize_detail_cats_for_target_category(
            target_detail_cats, target_category
        )
        avoid_detail_cats = self._sanitize_detail_cats_for_target_category(
            avoid_detail_cats, target_category
        )

        # 필터 조건 로그
        print(
            "  [BuyingTrigger] detail_cats="
            f"{target_detail_cats}, avoid_detail_cats={avoid_detail_cats}, "
            f"colors={prefer_colors}, brightness={prefer_brightness}"
        )
        print(
            f"  [BuyingTrigger] search_query='{search_query}', "
            f"target_category={target_category}, limit={limit}"
        )

        # FashionCLIP으로 쿼리 임베딩 생성
        try:
            query_embedding = self.embedder.embed_text(search_query)
        except Exception as e:
            return BuyingRecommendation(
                success=False,
                products=[],
                reasoning=f"임베딩 생성 실패: {e}",
                target_category=target_category
            )

        # ChromaDB 유사도 검색
        try:
            where_filter = self._build_where_filter(
                target_category=target_category,
                target_detail_cats=target_detail_cats if len(target_detail_cats) <= 1 else None,
                single_detail_cat=None,
            )

            print(f"  [BuyingTrigger] ChromaDB filter: {where_filter}")

            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=limit * 5,
                where=where_filter
            )
            raw_hits = len(results["ids"][0]) if results.get("ids") and results["ids"][0] else 0
            print(f"  [BuyingTrigger] raw_hits={raw_hits}, threshold={self.threshold}")
        except Exception as e:
            return BuyingRecommendation(
                success=False,
                products=[],
                reasoning=f"ChromaDB 검색 실패: {e}",
                target_category=target_category
            )

        # Threshold 필터링 및 상품 정보 구성
        candidates = self._extract_candidates_from_query_results(
            results=results,
            avoid_colors=avoid_colors,
            avoid_detail_cats=avoid_detail_cats,
            prefer_colors=prefer_colors,
            prefer_styles=prefer_styles,
            prefer_brightness=prefer_brightness,
        )
        print(f"  [BuyingTrigger] kept_after_threshold={len(candidates)}")
        if candidates:
            preview = [
                (
                    c.get("product_id"),
                    c.get("category_sub"),
                    round(float(c.get("similarity", 0.0)), 3),
                    round(float(c.get("final_score", 0.0)), 3),
                )
                for c in candidates[:5]
            ]
            print(f"  [BuyingTrigger] candidate_preview={preview}")

        fallback_used = False
        if not candidates:
            fallback_where = self._build_where_filter(
                target_category=target_category,
                target_detail_cats=None,
                single_detail_cat=None,
            )
            try:
                fallback_results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=max(limit * 10, 30),
                    where=fallback_where,
                )
                fallback_raw_hits = len(fallback_results["ids"][0]) if fallback_results.get("ids") and fallback_results["ids"][0] else 0
                print(
                    f"  [BuyingTrigger] fallback_filter={fallback_where}, "
                    f"fallback_raw_hits={fallback_raw_hits}"
                )
                candidates = self._extract_candidates_from_query_results(
                    results=fallback_results,
                    avoid_colors=avoid_colors,
                    avoid_detail_cats=avoid_detail_cats,
                    prefer_colors=prefer_colors,
                    prefer_styles=prefer_styles,
                    prefer_brightness=prefer_brightness,
                    min_similarity=None,
                )
                print(f"  [BuyingTrigger] fallback_kept={len(candidates)}")
                fallback_used = bool(candidates)
            except Exception:
                pass

        # 최종 점수 기준 정렬 및 상위 N개 선택
        candidates.sort(key=lambda x: x['final_score'], reverse=True)

        selected_by_cat = None
        if len(target_detail_cats) > 1:
            selected_by_cat = {cat: [] for cat in target_detail_cats}
            selected_ids = set()

            def _match_cat(candidate_cat: str, wanted_cat: str) -> bool:
                return (candidate_cat or "").strip().lower() == (wanted_cat or "").strip().lower()

            for detail_cat in target_detail_cats:
                cat_candidates = [c for c in candidates if _match_cat(c.get("category_sub", ""), detail_cat)]
                for c in cat_candidates:
                    if len(selected_by_cat[detail_cat]) >= 3:
                        break
                    pid = c.get("product_id")
                    if pid in selected_ids:
                        continue
                    selected_by_cat[detail_cat].append(c)
                    selected_ids.add(pid)
                print(
                    f"  [BuyingTrigger] selected_by_cat[{detail_cat}] "
                    f"seed_count={len(selected_by_cat[detail_cat])}"
                )

            for detail_cat in target_detail_cats:
                while len(selected_by_cat[detail_cat]) < 3:
                    detail_where = self._build_where_filter(
                        target_category=target_category,
                        target_detail_cats=None,
                        single_detail_cat=detail_cat,
                    )
                    extra_results = self.collection.query(
                        query_embeddings=[query_embedding],
                        n_results=max(limit * 8, 30),
                        where=detail_where,
                    )
                    extras = self._extract_candidates_from_query_results(
                        results=extra_results,
                        avoid_colors=avoid_colors,
                        avoid_detail_cats=avoid_detail_cats,
                        prefer_colors=prefer_colors,
                        prefer_styles=prefer_styles,
                        prefer_brightness=prefer_brightness,
                    )
                    added = False
                    for c in extras:
                        if len(selected_by_cat[detail_cat]) >= 3:
                            break
                        pid = c.get("product_id")
                        if pid in selected_ids:
                            continue
                        selected_by_cat[detail_cat].append(c)
                        selected_ids.add(pid)
                        added = True
                    if added:
                        print(
                            f"  [BuyingTrigger] selected_by_cat[{detail_cat}] "
                            f"expanded_count={len(selected_by_cat[detail_cat])}"
                        )
                    if not added:
                        break

            ordered = []
            for detail_cat in target_detail_cats:
                ordered.extend(selected_by_cat[detail_cat][:3])
            if ordered:
                candidates = ordered
            by_cat_counts = {cat: len(items) for cat, items in selected_by_cat.items()}
            print(f"  [BuyingTrigger] per_detail_final_counts={by_cat_counts}")

        effective_limit = max(limit, len(target_detail_cats) * 3) if len(target_detail_cats) > 1 else limit

        products = []
        for candidate in candidates[:effective_limit]:
            product = self._create_product_recommendation(candidate)
            if product:
                products.append(product)
        product_preview = [
            (p.product_id, p.category_sub, round(float(p.match_score), 3))
            for p in products[:5]
        ]
        print(
            f"  [BuyingTrigger] final_products={len(products)} "
            f"(effective_limit={effective_limit}) preview={product_preview}"
        )

        grouped_products = None
        if selected_by_cat:
            grouped_products = {}
            by_id = {str(p.product_id): p for p in products}
            for detail_cat, cat_candidates in selected_by_cat.items():
                grouped_products[detail_cat] = []
                for c in cat_candidates[:3]:
                    pid = str(c.get("product_id", ""))
                    p = by_id.get(pid)
                    if p:
                        grouped_products[detail_cat].append(p)

        # 결과 반환
        success = len(products) > 0
        reasoning = f"'{search_query}' 검색 완료: {len(products)}개 상품 추천 (threshold={self.threshold})"
        if fallback_used:
            reasoning += " [fallback:no-threshold]"

        return BuyingRecommendation(
            success=success,
            products=products,
            reasoning=reasoning,
            target_category=target_category
            ,
            grouped_products=grouped_products
        )

    def _sanitize_detail_cats_for_target_category(
        self,
        detail_cats: List[str],
        target_category: Optional[str],
    ) -> List[str]:
        if not detail_cats:
            return []
        normalized = []
        for value in detail_cats:
            normalized_value = self._normalize_detail_category(value)
            if normalized_value:
                normalized.append(normalized_value)
        normalized = list(dict.fromkeys(normalized))

        if not target_category:
            return normalized

        allowed = self._detail_groups.get(target_category)
        if not allowed:
            return normalized

        return [cat for cat in normalized if cat in allowed]

    @staticmethod
    def _meta_values(meta: Optional[Dict[str, Any]], *keys: str) -> List[str]:
        """
        Chroma 또는 메타 캐시에서 다양한 키로 metadata 값을 읽는다.
        값이 list/tuple이면 flatten.
        """
        if not meta:
            return []
        result: List[str] = []
        for key in keys:
            raw = meta.get(key)
            if raw is None:
                continue
            if isinstance(raw, (list, tuple, set)):
                for item in raw:
                    if item is not None:
                        result.append(str(item))
            else:
                result.append(str(raw))
        # 중복/빈값 제거
        cleaned = []
        for value in result:
            candidate = value.strip()
            if not candidate:
                continue
            if candidate not in cleaned:
                cleaned.append(candidate)
        return cleaned

    @staticmethod
    def _normalize_main_category(value: str) -> str:
        return normalize_main_category(value)

    @classmethod
    def _normalize_detail_category(cls, value: str) -> str:
        return normalize_detail_category(value)

    @staticmethod
    def _detail_filter_condition(detail_cats: List[str]) -> Optional[Dict[str, Any]]:
        if not detail_cats:
            return None
        if len(detail_cats) == 1:
            detail = detail_cats[0]
            return {EmbeddingBuyingTrigger.DETAIL_FIELD: detail}
        return {
            "$or": [
                {EmbeddingBuyingTrigger.DETAIL_FIELD: cat}
                for cat in detail_cats
            ]
        }

    @staticmethod
    def _main_filter_condition(main_categories: List[str]) -> Optional[Dict[str, Any]]:
        if not main_categories:
            return None
        if len(main_categories) == 1:
            cat = main_categories[0]
            return {
                "$or": [
                    {"category_main": cat},
                    {"broad_cat": cat},
                    {"category_main": cat.lower()},
                    {"broad_cat": cat.lower()},
                ]
            }
        return {
            "$or": [
                {
                    "$or": [
                        {"category_main": cat},
                        {"broad_cat": cat},
                        {"category_main": cat.lower()},
                        {"broad_cat": cat.lower()},
                    ]
                }
                for cat in main_categories
            ]
        }

    def _build_where_filter(
        self,
        target_category: Optional[str],
        target_detail_cats: Optional[List[str]],
        single_detail_cat: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        filters = []
        if target_category:
            normalized_main = self._normalize_main_category(target_category)
            main_cats = [normalized_main]
            if normalized_main.upper() != "TOPS" and normalized_main == target_category:
                for aliases in self._main_cat_alias_to_havati.values():
                    if target_category in aliases:
                        main_cats.extend([a for a in aliases])
                        break
            if normalized_main and normalized_main.upper() == "TOPS":
                main_cats = ["TOPS", "tops", "top", "shirts"]
            elif normalized_main and normalized_main.upper() == "BOTTOMS":
                main_cats = ["BOTTOMS", "bottom", "bottoms", "pants", "pant"]
            elif normalized_main and normalized_main.upper() == "OUTER":
                main_cats = ["OUTER", "outer", "outers"]

            if main_cats:
                filters.append(self._main_filter_condition(main_cats))

        detail_cats = None
        if single_detail_cat:
            detail_cats = [self._normalize_detail_category(single_detail_cat)]
        elif target_detail_cats:
            detail_cats = [self._normalize_detail_category(cat) for cat in target_detail_cats]
            detail_cats = [cat for cat in detail_cats if cat]
        if detail_cats:
            filters.append(self._detail_filter_condition(detail_cats))

        if len(filters) == 1:
            return filters[0]
        if len(filters) > 1:
            return {"$and": filters}
        return None

    def _extract_candidates_from_query_results(
        self,
        results: Dict[str, Any],
        avoid_colors: List[str],
        avoid_detail_cats: List[str],
        prefer_colors: List[str],
        prefer_styles: List[str],
        prefer_brightness: Optional[str],
        min_similarity: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        candidates = []
        all_similarities = []
        if results['ids'] and results['ids'][0] and results['distances'] and results['distances'][0]:
            for i, distance in enumerate(results['distances'][0]):
                similarity = 1 - distance
                all_similarities.append(similarity)
                threshold = self.threshold if min_similarity is None else min_similarity
                if threshold is not None and similarity < threshold:
                    continue

                product_id = results['ids'][0][i]
                meta = self._metadata_cache.get(product_id, {})
                product_color = meta.get('color', '')
                product_styles = meta.get('style_tags', [])
                detail_value = str(meta.get(self.DETAIL_FIELD, "") or "").strip()
                normalized_details = [self._normalize_detail_category(detail_value)] if detail_value else []
                avoid_detail_lower = {d.lower() for d in avoid_detail_cats}
                product_sub = normalized_details[0] if normalized_details else ""

                if product_color and product_color.lower() in [c.lower() for c in avoid_colors]:
                    continue
                if product_sub and product_sub.lower() in avoid_detail_lower:
                    continue

                brightness_bonus = 0.0
                if prefer_brightness and product_color:
                    product_brightness = get_brightness(product_color)
                    if prefer_brightness == "light":
                        if product_brightness == "dark":
                            continue
                        if product_brightness == "light":
                            brightness_bonus = 0.15
                    elif prefer_brightness == "dark":
                        if product_brightness == "light":
                            continue
                        if product_brightness == "dark":
                            brightness_bonus = 0.15

                color_bonus = 0.0
                if prefer_colors and product_color:
                    if product_color.lower() in [c.lower() for c in prefer_colors]:
                        color_bonus = 0.30

                style_bonus = 0.0
                if prefer_styles and product_styles:
                    meaningful_styles = [s for s in product_styles if s != "캐주얼"]
                    matching_styles = set(s.lower() for s in prefer_styles) & set(s.lower() for s in meaningful_styles)
                    if matching_styles:
                        style_bonus = min(len(matching_styles) * 0.10, 0.15)

                if color_bonus > 0:
                    brightness_bonus = 0.0

                final_score = similarity * 0.55 + color_bonus + style_bonus + brightness_bonus

                candidates.append({
                    'product_id': product_id,
                    'similarity': similarity,
                    'color_bonus': color_bonus,
                    'style_bonus': style_bonus,
                    'brightness_bonus': brightness_bonus,
                    'final_score': final_score,
                    'color': product_color,
                    'styles': product_styles,
                    'category_sub': product_sub,
                    'category_main_candidates': self._meta_values(meta, "category_main", "broad_cat"),
                })

            if all_similarities:
                print(f"  [BuyingTrigger] 유사도 범위: {min(all_similarities):.3f} ~ {max(all_similarities):.3f}")
        return candidates

    def _create_product_recommendation(
        self,
        candidate: Dict[str, Any]
    ) -> Optional[ProductRecommendation]:
        """
        상품 추천 객체 생성

        Args:
            candidate: recommend()에서 계산된 후보 정보
                - product_id, similarity, color_bonus, style_bonus, brightness_bonus,
                  final_score, color, styles

        Returns:
            ProductRecommendation 또는 None
        """
        product_id = candidate['product_id']
        metadata = self._metadata_cache.get(product_id)

        if not metadata:
            return None

        try:
            # 추천 이유 생성
            reason_parts = [f"유사도: {candidate['similarity']:.3f}"]
            if candidate['color']:
                reason_parts.append(f"색상: {candidate['color']}")
            if candidate['styles']:
                reason_parts.append(f"스타일: {', '.join(candidate['styles'][:2])}")
            if candidate['color_bonus'] > 0:
                reason_parts.append(f"색상 매칭: +{candidate['color_bonus']:.2f}")
            if candidate['style_bonus'] > 0:
                reason_parts.append(f"스타일 매칭: +{candidate['style_bonus']:.2f}")
            if candidate['brightness_bonus'] > 0:
                reason_parts.append(f"밝기 매칭: +{candidate['brightness_bonus']:.2f}")

            return ProductRecommendation(
                product_id=metadata.get('id', ''),
                product_name=metadata.get('product_name', ''),
                brand=metadata.get('brand', ''),
                price=int(metadata.get('price', 0)),
                category_main=metadata.get('category_main', ''),
                category_sub=metadata.get('category_sub', ''),
                product_url=metadata.get('product_url'),
                product_image_path=metadata.get('product_image_path'),
                match_score=candidate['final_score'],
                match_reason=" | ".join(reason_parts)
            )
        except Exception as e:
            print(f"[경고] 상품 정보 생성 실패: {e}")
            return None
