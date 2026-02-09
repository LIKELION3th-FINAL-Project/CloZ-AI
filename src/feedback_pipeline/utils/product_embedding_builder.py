"""
ProductEmbeddingBuilder - 상품 임베딩 구축 통합 모듈

havati_products 상품 데이터를 임베딩 + 시각적 메타데이터(color, style_tags)와 함께 ChromaDB에 저장

사용법:
    builder = ProductEmbeddingBuilder("havati_products/products.json")
    result = builder.build_all(limit=10)  # 샘플 테스트
    result = builder.build_all()          # 전체 빌드
"""

import os
import re
import sys
import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any

from PIL import Image
import chromadb

PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# 스타일 옵션 (기존 extract_visual_metadata.py와 동일)
STYLE_OPTIONS = [
    "고프코어", "레트로", "로맨틱", "리조트", "미니멀",
    "스트릿", "스포티", "시크", "시티보이", "아웃도어",
    "오피스", "워크웨어", "캐주얼", "클래식", "프레피"
]

# 색상 옵션 (정규화된 색상)
COLOR_OPTIONS = [
    "white", "black", "navy", "beige", "gray", "blue", "brown", "green", "red",
    "pink", "cream", "camel", "khaki", "charcoal", "burgundy", "olive", "purple",
    "yellow", "orange", "mint", "ivory"
]

# 상품명에서 추출한 색상 → 정규화된 색상 매핑
COLOR_MAPPING = {
    # 기본 색상
    "BLACK": "black",
    "WHITE": "white",
    "NAVY": "navy",
    "BEIGE": "beige",
    "GRAY": "gray",
    "GREY": "gray",
    "BLUE": "blue",
    "BROWN": "brown",
    "GREEN": "green",
    "RED": "red",
    "PINK": "pink",
    "CREAM": "cream",
    "CAMEL": "camel",
    "KHAKI": "khaki",
    "CHARCOAL": "charcoal",
    "BURGUNDY": "burgundy",
    "OLIVE": "olive",
    "PURPLE": "purple",
    "YELLOW": "yellow",
    "ORANGE": "orange",
    "MINT": "mint",
    "IVORY": "ivory",

    # 변형 색상
    "INDIGO": "navy",
    "DENIM": "blue",
    "DARK NAVY": "navy",
    "LIGHT BLUE": "blue",
    "DARK BLUE": "navy",
    "LIGHT GRAY": "gray",
    "DARK GRAY": "charcoal",
    "LIGHT GREY": "gray",
    "DARK GREY": "charcoal",
    "OFF WHITE": "ivory",
    "OFF-WHITE": "ivory",
    "ECRU": "cream",
    "NATURAL": "beige",
    "TAN": "camel",
    "SAND": "beige",
    "NUDE": "beige",
    "TAUPE": "beige",
    "OATMEAL": "cream",
    "HEATHER GRAY": "gray",
    "HEATHER GREY": "gray",
    "MELANGE": "gray",
    "WINE": "burgundy",
    "MAROON": "burgundy",
    "FOREST": "green",
    "SAGE": "green",
    "MOSS": "olive",
    "RUST": "orange",
    "CORAL": "pink",
    "SALMON": "pink",
    "LAVENDER": "purple",
    "MUSTARD": "yellow",
    "GOLD": "yellow",
    "SILVER": "gray",
    "BRICK": "red",
    "CHOCOLATE": "brown",
    "COFFEE": "brown",
    "MOCHA": "brown",
    "ESPRESSO": "brown",
    "COGNAC": "brown",
    "TOBACCO": "brown",
    "WASHED": "blue",  # WASHED는 보통 청바지 색
    "FADED": "blue",
    "VINTAGE": "blue",
    "USED": "blue",
    "RAW": "navy",
    "RINSE": "navy",
    "ONE WASH": "navy",
}

# Rate limit 설정
MAX_RETRIES = 5
INITIAL_BACKOFF = 1.0
MAX_BACKOFF = 60.0


@dataclass
class ProductItem:
    """상품 아이템 데이터 모델 (musinsa_ranking_result.json 형식 호환)"""
    id: str                      # "havati:001_BRAND_NAME"
    product_name: str            # 상품명
    brand: str
    price: int
    product_url: str             # 상품 URL
    product_image_path: str      # 이미지 경로
    category_main: str           # "TOPS", "BOTTOMS", "OUTER"
    category_sub: str            # "Jacket_Blouson"
    color: str                   # "navy", "black" 등
    style_tags: List[str]        # ["캐주얼", "미니멀"]
    embedding: List[float]       # 512차원

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class BuildResult:
    """빌드 결과"""
    success: bool
    total_items: int
    added_items: int
    skipped_items: int
    failed_items: List[Dict[str, str]] = field(default_factory=list)
    collection_name: str = ""


class ProductEmbeddingBuilder:
    """
    상품 임베딩 구축 통합 모듈

    역할:
    1. products.json 로드
    2. 색상 추출 (이름 파싱 + LLM fallback)
    3. 스타일 태그 추출 (Gemini)
    4. FashionCLIP 임베딩 생성
    5. ChromaDB 저장 (upsert)
    """

    def __init__(
        self,
        products_json_path: str,
        chroma_path: str = None,
        collection_name: str = "products",
        device: str = None,
        gemini_api_key: str = None
    ):
        """
        초기화

        Args:
            products_json_path: havati_products/products.json 경로
            chroma_path: ChromaDB 경로 (기본: data/chroma_db)
            collection_name: 컬렉션 이름 (기본: "products")
            device: FashionCLIP 디바이스 ("cuda", "mps", "cpu")
            gemini_api_key: Gemini API 키 (환경변수 fallback)
        """
        self.products_json_path = products_json_path
        self.chroma_path = chroma_path or str(PROJECT_ROOT / "data" / "chroma_db")
        self.collection_name = collection_name
        self.device = device
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

        # 체크포인트 경로
        self.checkpoint_path = str(PROJECT_ROOT / "data" / "product_embedding_checkpoint.json")

        # 지연 로딩
        self._embedder = None
        self._gemini_model = None
        self._collection = None
        self._chroma_client = None

        print(f"[ProductEmbeddingBuilder] 초기화")
        print(f"  - products: {products_json_path}")
        print(f"  - chroma: {self.chroma_path}/{collection_name}")

    @property
    def embedder(self):
        """FashionCLIPEmbedder 지연 로딩"""
        if self._embedder is None:
            print("[ProductEmbeddingBuilder] FashionCLIPEmbedder 로딩...")
            sys.path.append(str(PROJECT_ROOT / "embedding_generator"))
            from generate_fashionclip_embeddings import FashionCLIPEmbedder
            self._embedder = FashionCLIPEmbedder(device=self.device)
        return self._embedder

    @property
    def gemini_model(self):
        """Gemini 모델 지연 로딩"""
        if self._gemini_model is None:
            if not self.gemini_api_key:
                raise ValueError("GEMINI_API_KEY 또는 GOOGLE_API_KEY 환경변수가 필요합니다")

            print("[ProductEmbeddingBuilder] Gemini 모델 로딩...")
            import google.generativeai as genai
            genai.configure(api_key=self.gemini_api_key)
            self._gemini_model = genai.GenerativeModel('gemini-2.0-flash')
        return self._gemini_model

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

        print(f"[ProductEmbeddingBuilder] 컬렉션 로드: {self.collection_name} ({collection.count()}개 아이템)")
        return collection

    def build_all(
        self,
        limit: int = None,
        checkpoint_interval: int = 50
    ) -> BuildResult:
        """
        전체 상품 빌드

        Args:
            limit: 처리할 최대 개수 (테스트용)
            checkpoint_interval: 체크포인트 저장 간격

        Returns:
            BuildResult
        """
        # products.json 로드
        print(f"\n상품 데이터 로드: {self.products_json_path}")
        with open(self.products_json_path, 'r', encoding='utf-8') as f:
            products = json.load(f)

        total_products = len(products)
        print(f"총 {total_products}개 상품 로드됨")

        # 체크포인트 로드
        checkpoint = self._load_checkpoint()
        processed_ids = set(checkpoint.keys())

        if checkpoint:
            print(f"체크포인트 발견: {len(checkpoint)}개 이미 처리됨")

        # limit 적용
        if limit:
            products = products[:limit]
            print(f"limit 적용: {len(products)}개만 처리")

        print("\n" + "=" * 60)
        print("상품 임베딩 생성 시작")
        print("=" * 60)

        added = 0
        skipped = 0
        failed = []

        for idx, product in enumerate(products, start=1):
            # 상품 ID 생성
            image_file = product.get("image_file", "")
            product_id = self._generate_product_id(image_file)

            # 이미 처리됨
            if product_id in processed_ids:
                skipped += 1
                if idx % 50 == 0:
                    print(f"[{idx}/{len(products)}] 스킵 (이미 처리됨)")
                continue

            print(f"[{idx}/{len(products)}] 처리 중: {product.get('name', '')[:50]}...")

            # 상품 처리
            success, item = self._process_product(product, product_id)

            if success and item:
                # ChromaDB 저장
                save_success = self._save_to_chromadb(item)
                if save_success:
                    # 체크포인트에는 embedding 제외 (용량 절약)
                    item_dict = item.to_dict()
                    item_dict.pop('embedding', None)
                    checkpoint[product_id] = item_dict
                    added += 1
                    print(f"  → color={item.color}, styles={item.style_tags}")
                else:
                    failed.append({"id": product_id, "reason": "ChromaDB 저장 실패"})
            else:
                failed.append({"id": product_id, "reason": "처리 실패"})

            # 체크포인트 저장
            if idx % checkpoint_interval == 0:
                self._save_checkpoint(checkpoint)
                print(f"\n체크포인트 저장: {len(checkpoint)}개")

        # 최종 저장
        self._save_checkpoint(checkpoint)

        print("\n" + "=" * 60)
        print("빌드 완료!")
        print("=" * 60)
        print(f"  총 상품: {len(products)}")
        print(f"  추가됨: {added}")
        print(f"  스킵됨: {skipped}")
        print(f"  실패: {len(failed)}")

        return BuildResult(
            success=len(failed) == 0,
            total_items=len(products),
            added_items=added,
            skipped_items=skipped,
            failed_items=failed,
            collection_name=self.collection_name
        )

    def _process_product(
        self,
        product: Dict,
        product_id: str
    ) -> Tuple[bool, Optional[ProductItem]]:
        """
        단일 상품 처리

        1. 색상 추출
        2. 스타일 태그 추출
        3. 임베딩 생성
        """
        try:
            product_name = product.get("name", "")
            brand = product.get("brand", "")
            price_str = product.get("price", "0")
            price = int(price_str.replace(",", "")) if price_str else 0
            product_url = product.get("url", "")
            image_file = product.get("image_file", "")
            category_main = product.get("main_category", "")
            category_sub = product.get("sub_category", "")

            # 이미지 경로 확인
            image_path = str(PROJECT_ROOT / image_file)
            if not os.path.exists(image_path):
                print(f"  [ERROR] 이미지 없음: {image_path}")
                return (False, None)

            # 1. 색상 추출
            color = self._extract_color_from_name(product_name)
            if not color:
                color = self._extract_color_via_llm(image_path)

            # 2. 스타일 태그 추출
            style_tags = self._extract_style_tags(image_path)

            # 3. 임베딩 생성
            embedding = self.embedder.embed_image(image_path)
            if embedding is None:
                print(f"  [ERROR] 임베딩 생성 실패")
                return (False, None)

            # ProductItem 생성 (musinsa 형식 호환)
            item = ProductItem(
                id=product_id,
                product_name=product_name,
                brand=brand,
                price=price,
                product_url=product_url,
                product_image_path=image_file,
                category_main=category_main,
                category_sub=category_sub,
                color=color,
                style_tags=style_tags,
                embedding=embedding
            )

            return (True, item)

        except Exception as e:
            print(f"  [ERROR] 처리 실패: {e}")
            return (False, None)

    def _extract_color_from_name(self, name: str) -> Optional[str]:
        """
        상품명에서 색상 추출

        예: "[BRAND] JACKET (INDIGO)" → "navy"
        """
        # 괄호 안 색상 추출: (COLOR) 패턴
        match = re.search(r'\(([A-Z\s\-\/]+)\)\s*$', name.upper())
        if match:
            raw_color = match.group(1).strip()

            # 정규화 매핑에서 찾기
            if raw_color in COLOR_MAPPING:
                return COLOR_MAPPING[raw_color]

            # 부분 매칭 시도
            for key, value in COLOR_MAPPING.items():
                if key in raw_color or raw_color in key:
                    return value

        return None

    def _extract_color_via_llm(self, image_path: str) -> str:
        """
        LLM으로 이미지에서 색상 추출 (fallback)
        """
        retry_count = 0
        backoff_time = INITIAL_BACKOFF

        while retry_count < MAX_RETRIES:
            try:
                image = Image.open(image_path)
                prompt = f"""이 의류 이미지의 주요 색상을 아래 목록에서 하나만 선택해주세요.

색상 목록: {', '.join(COLOR_OPTIONS)}

반드시 목록에 있는 색상 중 하나만 답변해주세요. 다른 설명 없이 색상명만 답변하세요."""

                response = self.gemini_model.generate_content([prompt, image])
                result = response.text.strip().lower()

                # 유효한 색상인지 확인
                if result in COLOR_OPTIONS:
                    return result

                # 부분 매칭
                for color in COLOR_OPTIONS:
                    if color in result:
                        return color

                return "unknown"

            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                    retry_count += 1
                    if retry_count >= MAX_RETRIES:
                        return "unknown"

                    wait_time = min(backoff_time, MAX_BACKOFF)
                    print(f"  Rate limit. {wait_time:.1f}초 대기 ({retry_count}/{MAX_RETRIES})...")
                    time.sleep(wait_time)
                    backoff_time *= 2
                else:
                    print(f"  [WARN] 색상 LLM 추출 실패: {error_msg}")
                    return "unknown"

        return "unknown"

    def _extract_style_tags(self, image_path: str) -> List[str]:
        """
        Gemini로 스타일 태그 추출 (1-3개)
        """
        retry_count = 0
        backoff_time = INITIAL_BACKOFF

        while retry_count < MAX_RETRIES:
            try:
                image = Image.open(image_path)
                prompt = f"""이 의류 이미지를 분석해서 어울리는 스타일 태그를 1-3개 선택해주세요.

스타일 목록: {', '.join(STYLE_OPTIONS)}

반드시 JSON 형식으로만 응답하세요:
{{"style_tags": ["캐주얼", "미니멀"]}}"""

                response = self.gemini_model.generate_content([prompt, image])
                result = self._parse_style_response(response.text)
                return result

            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                    retry_count += 1
                    if retry_count >= MAX_RETRIES:
                        return ["캐주얼"]

                    wait_time = min(backoff_time, MAX_BACKOFF)
                    print(f"  Rate limit. {wait_time:.1f}초 대기 ({retry_count}/{MAX_RETRIES})...")
                    time.sleep(wait_time)
                    backoff_time *= 2
                else:
                    print(f"  [WARN] 스타일 추출 실패: {error_msg}")
                    return ["캐주얼"]

        return ["캐주얼"]

    def _parse_style_response(self, response_text: str) -> List[str]:
        """스타일 응답 파싱"""
        try:
            text = response_text.strip()

            # JSON 블록 추출
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()

            data = json.loads(text)
            style_tags = data.get("style_tags", [])

            # 유효한 스타일만 필터링
            valid_styles = [s for s in style_tags if s in STYLE_OPTIONS]

            if not valid_styles:
                return ["캐주얼"]

            return valid_styles[:3]  # 최대 3개

        except json.JSONDecodeError:
            return ["캐주얼"]
        except Exception:
            return ["캐주얼"]

    def _generate_product_id(self, image_file: str) -> str:
        """상품 ID 생성: havati:filename"""
        filename = Path(image_file).stem
        return f"havati:{filename}"

    def _save_to_chromadb(self, item: ProductItem) -> bool:
        """ChromaDB에 저장 (upsert) - musinsa 형식 호환"""
        try:
            metadata = {
                "entity_type": "product",
                "id": item.id,
                "product_name": item.product_name,
                "brand": item.brand,
                "price": item.price,
                "product_url": item.product_url,
                "product_image_path": item.product_image_path,
                "category_main": item.category_main,
                "category_sub": item.category_sub,
                "color": item.color,
                "style_tags": ",".join(item.style_tags),  # list → 콤마 구분 문자열
                "source": "havati",
            }

            self.collection.upsert(
                ids=[item.id],
                embeddings=[item.embedding],
                metadatas=[metadata]
            )

            return True
        except Exception as e:
            print(f"  [ERROR] ChromaDB 저장 실패: {e}")
            return False

    def _load_checkpoint(self) -> Dict:
        """체크포인트 로드"""
        if os.path.exists(self.checkpoint_path):
            with open(self.checkpoint_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def _save_checkpoint(self, data: Dict):
        """체크포인트 저장"""
        os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
        with open(self.checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def get_statistics(self) -> Dict[str, Any]:
        """컬렉션 통계 조회"""
        try:
            count = self.collection.count()

            # 샘플 조회
            if count > 0:
                results = self.collection.get(
                    limit=min(100, count),
                    include=["metadatas"]
                )

                # 카테고리별 집계
                by_category = {}
                by_color = {}

                for meta in results["metadatas"]:
                    cat = meta.get("category_main", "unknown")
                    color = meta.get("color", "unknown")

                    by_category[cat] = by_category.get(cat, 0) + 1
                    by_color[color] = by_color.get(color, 0) + 1

                return {
                    "collection_name": self.collection_name,
                    "total_items": count,
                    "by_category": by_category,
                    "by_color": by_color
                }

            return {
                "collection_name": self.collection_name,
                "total_items": 0
            }
        except Exception as e:
            print(f"[ERROR] 통계 조회 실패: {e}")
            return {"total_items": 0}


# 테스트용 메인
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ProductEmbeddingBuilder 테스트")
    parser.add_argument("--products", type=str, required=True, help="products.json 경로")
    parser.add_argument("--limit", type=int, default=None, help="처리할 최대 개수")

    args = parser.parse_args()

    builder = ProductEmbeddingBuilder(products_json_path=args.products)
    result = builder.build_all(limit=args.limit)

    print("\n=== 결과 ===")
    print(f"성공: {result.success}")
    print(f"총 아이템: {result.total_items}")
    print(f"추가됨: {result.added_items}")
    print(f"스킵됨: {result.skipped_items}")
    print(f"실패: {len(result.failed_items)}")

    print("\n=== 통계 ===")
    stats = builder.get_statistics()
    print(f"총 아이템: {stats.get('total_items', 0)}")
    print(f"카테고리별: {stats.get('by_category', {})}")
    print(f"색상별: {stats.get('by_color', {})}")
