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

        # 1. 검색 쿼리 결정 (영어 우선)
        search_query = feedback_text
        avoid_colors = []
        prefer_colors = []
        prefer_styles = []
        prefer_brightness = None  # "light" or "dark"
        target_detail_cats = []  # 세부카테고리 필터

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

            # avoid_attributes에서도 추출
            if 'avoid_attributes' in context:
                avoid_attrs = context['avoid_attributes']
                if isinstance(avoid_attrs, dict):
                    avoid_colors.extend(avoid_attrs.get('colors', []))

        # 필터 조건 로그
        print(f"  [BuyingTrigger] detail_cats={target_detail_cats}, colors={prefer_colors}, brightness={prefer_brightness}")

        # 2. FashionCLIP으로 쿼리 임베딩 생성
        try:
            query_embedding = self.embedder.embed_text(search_query)
        except Exception as e:
            return BuyingRecommendation(
                success=False,
                products=[],
                reasoning=f"임베딩 생성 실패: {e}",
                target_category=target_category
            )

        # 3. ChromaDB 유사도 검색
        try:
            # 카테고리 필터링 구성
            # Havati 대분류: TOPS, BOTTOMS, OUTER
            CATEGORY_TO_HAVATI = {
                "상의": "TOPS",
                "바지": "BOTTOMS",
                "아우터": "OUTER",
            }

            where_filter = None
            filters = []

            # 대분류 필터 (한글 → Havati 영문)
            if target_category:
                havati_cat = CATEGORY_TO_HAVATI.get(target_category, target_category)
                filters.append({"category_main": havati_cat})

            # 세부분류 필터 (category_sub) - Havati 카테고리 그대로 사용
            if target_detail_cats:
                if len(target_detail_cats) == 1:
                    filters.append({"category_sub": target_detail_cats[0]})
                elif len(target_detail_cats) > 1:
                    filters.append({"$or": [{"category_sub": cat} for cat in target_detail_cats]})

            # 필터 결합
            if len(filters) == 1:
                where_filter = filters[0]
            elif len(filters) > 1:
                where_filter = {"$and": filters}

            print(f"  [BuyingTrigger] ChromaDB filter: {where_filter}")

            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=limit * 5,
                where=where_filter
            )
        except Exception as e:
            return BuyingRecommendation(
                success=False,
                products=[],
                reasoning=f"ChromaDB 검색 실패: {e}",
                target_category=target_category
            )

        # 4. Threshold 필터링 및 상품 정보 구성
        candidates = []
        all_similarities = []

        if results['ids'] and results['ids'][0] and results['distances'] and results['distances'][0]:
            for i, distance in enumerate(results['distances'][0]):
                # ChromaDB cosine distance: 1 - cosine_similarity
                similarity = 1 - distance
                all_similarities.append(similarity)

                if similarity >= self.threshold:
                    product_id = results['ids'][0][i]

                    # Havati 메타데이터에서 color, style_tags 직접 조회
                    meta = self._metadata_cache.get(product_id, {})
                    product_color = meta.get('color', '')
                    product_styles = meta.get('style_tags', [])

                    # 회피 색상 필터링
                    if product_color and product_color.lower() in [c.lower() for c in avoid_colors]:
                        continue

                    # 밝기 선호도 필터링
                    brightness_bonus = 0.0
                    if prefer_brightness and product_color:
                        product_brightness = get_brightness(product_color)
                        if prefer_brightness == "light":
                            if product_brightness == "dark":
                                continue
                            elif product_brightness == "light":
                                brightness_bonus = 0.15
                        elif prefer_brightness == "dark":
                            if product_brightness == "light":
                                continue
                            elif product_brightness == "dark":
                                brightness_bonus = 0.15

                    # 색상 직접 매칭 보너스 (최우선, 0.30)
                    color_bonus = 0.0
                    if prefer_colors and product_color:
                        if product_color.lower() in [c.lower() for c in prefer_colors]:
                            color_bonus = 0.30

                    # 스타일 매칭 보너스 (캐주얼 제외, 최대 0.15)
                    style_bonus = 0.0
                    if prefer_styles and product_styles:
                        meaningful_styles = [s for s in product_styles if s != "캐주얼"]
                        matching_styles = set(s.lower() for s in prefer_styles) & set(s.lower() for s in meaningful_styles)
                        if matching_styles:
                            style_bonus = min(len(matching_styles) * 0.10, 0.15)

                    # 밝기 보너스는 색상 매칭 없을 때만 적용
                    if color_bonus > 0:
                        brightness_bonus = 0.0

                    # 최종 점수: 임베딩(0.55) + 색상(0.30) + 스타일(0.15) + 밝기(0.15)
                    final_score = similarity * 0.55 + color_bonus + style_bonus + brightness_bonus

                    candidates.append({
                        'product_id': product_id,
                        'similarity': similarity,
                        'color_bonus': color_bonus,
                        'style_bonus': style_bonus,
                        'brightness_bonus': brightness_bonus,
                        'final_score': final_score,
                        'color': product_color,
                        'styles': product_styles
                    })

            # 디버깅: 유사도 범위 출력
            if all_similarities:
                print(f"  [BuyingTrigger] 유사도 범위: {min(all_similarities):.3f} ~ {max(all_similarities):.3f}")

        # 5. 최종 점수 기준 정렬 및 상위 N개 선택
        candidates.sort(key=lambda x: x['final_score'], reverse=True)

        products = []
        for candidate in candidates[:limit]:
            product = self._create_product_recommendation(candidate)
            if product:
                products.append(product)

        # 5. 결과 반환
        success = len(products) > 0
        reasoning = f"'{search_query}' 검색 완료: {len(products)}개 상품 추천 (threshold={self.threshold})"

        return BuyingRecommendation(
            success=success,
            products=products,
            reasoning=reasoning,
            target_category=target_category
        )

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
