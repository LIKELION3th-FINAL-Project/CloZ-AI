"""
임베딩 기반 상품 추천 (FashionCLIP + ChromaDB)
"""
import sys
import json
from pathlib import Path
from typing import Dict, List, Optional, Any

# FashionCLIP embedder import
sys.path.append(str(Path(__file__).parent.parent.parent / "scripts"))
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

    FashionCLIP 임베딩으로 무신사 상품에서 피드백 요구사항과 유사한 상품 검색
    """

    def __init__(
        self,
        threshold: float = 0.15,  # FashionCLIP 텍스트-이미지 유사도는 보통 0.2~0.35
        chroma_path: str = "data/chroma_db",
        collection_name: str = "musinsa",
        metadata_path: str = "data/musinsa_ranking_result.json",
        visual_metadata_path: str = "data/visual_metadata_checkpoint.json",
        device: str = None,
        embedder: Optional[FashionCLIPEmbedder] = None
    ):
        """
        초기화

        Args:
            threshold: 유사도 임계값 (FashionCLIP 기준 0.15~0.25 권장)
            chroma_path: ChromaDB 경로
            collection_name: 무신사 컬렉션 이름
            metadata_path: 무신사 상품 메타데이터 JSON 경로
            visual_metadata_path: 시각적 메타데이터 JSON 경로 (color, style_tags)
            device: FashionCLIP 디바이스 ("cuda", "mps", or "cpu")
        """
        self.threshold = threshold
        self.metadata_path = metadata_path
        self.visual_metadata_path = visual_metadata_path
        self._metadata_cache: Dict[str, Dict] = {}  # 메타데이터 캐시
        self._visual_metadata_cache: Dict[str, Dict] = {}  # 시각적 메타데이터 캐시

        if not CHROMADB_AVAILABLE:
            print("[경고] chromadb가 설치되지 않음. Dummy 모드로 동작")
            self.chroma_client = None
            self.musinsa_collection = None
            self.embedder = None
            return

        # ChromaDB 클라이언트 초기화
        self.chroma_client = chromadb.PersistentClient(path=chroma_path)
        try:
            self.musinsa_collection = self.chroma_client.get_collection(collection_name)
        except Exception as e:
            print(f"[경고] 컬렉션 '{collection_name}' 없음: {e}")
            self.musinsa_collection = None
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
        self._load_visual_metadata()

    def _load_metadata(self):
        """무신사 상품 메타데이터 로드"""
        try:
            metadata_file = Path(self.metadata_path)
            if not metadata_file.exists():
                # 프로젝트 루트 기준으로 재시도
                project_root = Path(__file__).parent.parent.parent
                metadata_file = project_root / self.metadata_path

            if metadata_file.exists():
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    products = json.load(f)
                    for product in products:
                        product_id = product.get('id', '')
                        self._metadata_cache[f"musinsa:{product_id}"] = product
                print(f"[INFO] 무신사 메타데이터 로드 완료: {len(self._metadata_cache)}개 상품")
            else:
                print(f"[경고] 메타데이터 파일 없음: {self.metadata_path}")
        except Exception as e:
            print(f"[경고] 메타데이터 로드 실패: {e}")

    def _load_visual_metadata(self):
        """시각적 메타데이터 로드 (color, style_tags)"""
        try:
            visual_file = Path(self.visual_metadata_path)
            if not visual_file.exists():
                project_root = Path(__file__).parent.parent.parent
                visual_file = project_root / self.visual_metadata_path

            if visual_file.exists():
                with open(visual_file, 'r', encoding='utf-8') as f:
                    visual_data = json.load(f)
                    for product_id, data in visual_data.items():
                        if data.get('success'):
                            self._visual_metadata_cache[product_id] = {
                                'color': data.get('color', ''),
                                'style_tags': data.get('style_tags', [])
                            }
                print(f"[INFO] 시각적 메타데이터 로드 완료: {len(self._visual_metadata_cache)}개 상품")
            else:
                print(f"[경고] 시각적 메타데이터 파일 없음: {self.visual_metadata_path}")
        except Exception as e:
            print(f"[경고] 시각적 메타데이터 로드 실패: {e}")

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
        if not self.musinsa_collection or not self.embedder:
            return BuyingRecommendation(
                success=False,
                products=[],
                reasoning="[Dummy] ChromaDB 또는 FashionCLIP 없음",
                target_category=target_category
            )

        # 1. 검색 쿼리 결정 (영어 우선)
        search_query = feedback_text
        avoid_colors = []
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
                    # 제한사항에서 색상 추출
                    for constraint in structured.get('constraints', []):
                        constraint_lower = constraint.lower()
                        if '검은색' in constraint or 'black' in constraint_lower:
                            avoid_colors.append('black')
                        if '흰색' in constraint or 'white' in constraint_lower:
                            avoid_colors.append('white')
                        # 다른 색상도 추가 가능
                    # mood에서 스타일 추출
                    prefer_styles = structured.get('mood', [])
                    # 밝기 선호도 추출
                    prefer_brightness = structured.get('prefer_brightness')

            # avoid_attributes에서도 추출
            if 'avoid_attributes' in context:
                avoid_attrs = context['avoid_attributes']
                if isinstance(avoid_attrs, dict):
                    avoid_colors.extend(avoid_attrs.get('colors', []))
                    # 스타일도 회피 가능

        # [DEBUG] 필터 조건 확인
        print(f"  [DEBUG] target_detail_cats: {target_detail_cats}")
        print(f"  [DEBUG] prefer_brightness: {prefer_brightness}")
        print(f"  [DEBUG] avoid_colors: {avoid_colors}")
        print(f"  [DEBUG] prefer_styles: {prefer_styles}")

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
            where_filter = None
            filters = []

            # 대분류 필터 (category_main)
            if target_category:
                filters.append({"category_main": target_category})

            # 세부분류 필터 (category_sub) - 무신사는 category_sub 사용
            if target_detail_cats:
                # 옷장 카테고리명 → 무신사 카테고리명 매핑
                to_musinsa_cat = {
                    "니트/스웨트": "니트/스웨터",
                    "니트/스웨터": "니트/스웨터",
                    "맨투맨": "맨투맨/스웨트셔츠",
                    "스웨트셔츠": "맨투맨/스웨트셔츠",
                    "후드": "후드 티셔츠",
                    "후드티": "후드 티셔츠",
                }
                # 매핑된 카테고리로 변환
                mapped_cats = []
                for cat in target_detail_cats:
                    if cat in to_musinsa_cat:
                        mapped_cats.append(to_musinsa_cat[cat])
                    else:
                        mapped_cats.append(cat)

                if len(mapped_cats) == 1:
                    filters.append({"category_sub": mapped_cats[0]})
                elif len(mapped_cats) > 1:
                    filters.append({"$or": [{"category_sub": cat} for cat in mapped_cats]})

            # 필터 결합
            if len(filters) == 1:
                where_filter = filters[0]
            elif len(filters) > 1:
                where_filter = {"$and": filters}

            print(f"  [DEBUG] ChromaDB where_filter: {where_filter}")

            results = self.musinsa_collection.query(
                query_embeddings=[query_embedding],
                n_results=limit * 5,  # 필터링 후 limit개 남도록 여유있게
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
                    raw_id = product_id.replace("musinsa:", "")

                    # 시각적 메타데이터로 필터링 및 점수 보정
                    visual_meta = self._visual_metadata_cache.get(raw_id, {})
                    product_color = visual_meta.get('color', '')
                    product_styles = visual_meta.get('style_tags', [])

                    # 회피 색상 필터링
                    if product_color and product_color.lower() in [c.lower() for c in avoid_colors]:
                        continue  # 제외

                    # 밝기 선호도 필터링
                    brightness_bonus = 0.0
                    if prefer_brightness and product_color:
                        product_brightness = get_brightness(product_color)
                        if prefer_brightness == "light":
                            if product_brightness == "dark":
                                continue  # 어두운 색상 제외
                            elif product_brightness == "light":
                                brightness_bonus = 0.15  # 밝은 색상 보너스
                        elif prefer_brightness == "dark":
                            if product_brightness == "light":
                                continue  # 밝은 색상 제외
                            elif product_brightness == "dark":
                                brightness_bonus = 0.15  # 어두운 색상 보너스

                    # 점수 계산: 임베딩 유사도 (0.6) + 스타일 매칭 보너스 (0.25) + 밝기 보너스 (0.15)
                    style_bonus = 0.0
                    if prefer_styles and product_styles:
                        # 선호 스타일과 겹치는 개수에 따라 보너스
                        matching_styles = set(s.lower() for s in prefer_styles) & set(s.lower() for s in product_styles)
                        if matching_styles:
                            style_bonus = min(len(matching_styles) * 0.1, 0.25)  # 최대 0.25

                    final_score = similarity * 0.6 + style_bonus + brightness_bonus

                    candidates.append({
                        'product_id': product_id,
                        'similarity': similarity,
                        'style_bonus': style_bonus,
                        'final_score': final_score,
                        'color': product_color,
                        'styles': product_styles
                    })

            # 디버깅: 유사도 범위 출력
            if all_similarities:
                print(f"  [DEBUG] 무신사 검색 유사도 범위: {min(all_similarities):.3f} ~ {max(all_similarities):.3f}")

        # 5. 최종 점수 기준 정렬 및 상위 N개 선택
        candidates.sort(key=lambda x: x['final_score'], reverse=True)

        products = []
        for candidate in candidates[:limit]:
            product = self._create_product_recommendation(
                product_id=candidate['product_id'],
                similarity=candidate['similarity'],
                search_query=search_query,
                style_bonus=candidate['style_bonus'],
                color=candidate['color'],
                styles=candidate['styles']
            )
            if product:
                products.append(product)

        if not products:
            return BuyingRecommendation(
                success=False,
                products=[],
                reasoning=f"'{search_query}' 검색 결과 없음 (threshold: {self.threshold})",
                target_category=target_category
            )

        return BuyingRecommendation(
            success=True,
            products=products,
            reasoning=f"FashionCLIP 임베딩 유사도 검색 결과 ({len(products)}개 상품)",
            target_category=target_category
        )

    def _create_product_recommendation(
        self,
        product_id: str,
        similarity: float,
        search_query: str,
        style_bonus: float = 0.0,
        color: str = "",
        styles: List[str] = None
    ) -> Optional[ProductRecommendation]:
        """
        메타데이터를 조회하여 ProductRecommendation 생성
        """
        metadata = self._metadata_cache.get(product_id, {})

        if not metadata:
            # ChromaDB에서 직접 조회 시도
            try:
                result = self.musinsa_collection.get(
                    ids=[product_id],
                    include=["metadatas"]
                )
                if result['metadatas'] and result['metadatas'][0]:
                    metadata = result['metadatas'][0]
            except:
                pass

        if not metadata:
            return None

        # match_reason 구성
        reason_parts = []
        reason_parts.append(f"임베딩 유사도: {similarity:.3f}")
        if color:
            reason_parts.append(f"색상: {color}")
        if styles:
            reason_parts.append(f"스타일: {', '.join(styles[:3])}")
        if style_bonus > 0:
            reason_parts.append(f"스타일 보너스: +{style_bonus:.2f}")

        return ProductRecommendation(
            product_id=metadata.get('id', product_id.replace("musinsa:", "")),
            product_name=metadata.get('product_name', 'Unknown'),
            brand=metadata.get('brand', 'Unknown'),
            price=metadata.get('price', 0),
            category_main=metadata.get('category_main', '상의'),
            category_sub=metadata.get('category_sub', ''),
            image_url=metadata.get('image_url', ''),
            product_url=metadata.get('product_url', ''),
            match_score=similarity * 0.7 + style_bonus,  # 최종 점수
            match_reason=" | ".join(reason_parts)
        )

    def get_target_category(self, feedback_scope: FeedbackScope) -> str:
        """피드백 범위에 따른 타겟 카테고리 결정"""
        scope_to_category = {
            FeedbackScope.TOP: "상의",
            FeedbackScope.BOTTOM: "바지",
            FeedbackScope.OUTER: "아우터",
            FeedbackScope.FULL: None,  # 전체
        }
        return scope_to_category.get(feedback_scope)

    def check_availability(self, product_id: str) -> bool:
        """상품 재고/판매 여부 확인"""
        if not self.musinsa_collection:
            return True

        try:
            result = self.musinsa_collection.get(ids=[product_id])
            return bool(result['ids'])
        except:
            return False


# 테스트용
if __name__ == "__main__":
    trigger = EmbeddingBuyingTrigger()

    # 테스트: 밝은 니트 검색
    result = trigger.recommend(
        original_prompt="캐주얼한 데이트룩",
        feedback_text="밝은 니트로 바꿔줘",
        feedback_scope=FeedbackScope.TOP,
        context={
            "structured_query": {
                "requirements_en": ["bright colored knit sweater"],
                "target_detail_cats": ["니트/스웨트"],
                "prefer_brightness": "light",
                "mood": ["캐주얼", "밝은"]
            }
        }
    )

    print(f"\n추천 결과: {result.success}")
    print(f"추천 상품 수: {len(result.products)}")
    for p in result.products:
        print(f"  - {p.brand}: {p.product_name[:30]}... (점수: {p.match_score:.3f})")
        print(f"    이유: {p.match_reason}")
