"""
VLM 기반 시각적 메타데이터 추출 스크립트

Gemini 2.0 Flash를 사용하여 무신사 상품 이미지에서 color, style_tags 추출
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

sys.path.insert(0, str(Path(__file__).parent.parent))

import google.generativeai as genai
from PIL import Image


STYLE_OPTIONS = [
    "고프코어", "레트로", "로맨틱", "리조트", "미니멀",
    "스트릿", "스포티", "시크", "시티보이", "아웃도어",
    "오피스", "워크웨어", "캐주얼", "클래식", "프레피"
]

COLOR_OPTIONS = [
    "white", "black", "navy", "beige", "gray", "blue", "brown", "green", "red",
    "pink", "cream", "camel", "khaki", "charcoal", "burgundy", "olive", "purple",
    "yellow", "orange", "mint", "ivory"
]

MAX_RETRIES = 5
INITIAL_BACKOFF = 1.0
MAX_BACKOFF = 60.0


@dataclass
class VisualMetadata:
    """추출된 시각적 메타데이터"""
    product_id: str
    color: str
    style_tags: List[str]
    success: bool = True
    error_message: Optional[str] = None

    def to_dict(self) -> Dict:
        return asdict(self)


class GeminiVisualMetadataExtractor:
    """Gemini 2.0 Flash 기반 메타데이터 추출"""

    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash')

    def extract(self, image_path: str, product_id: str) -> VisualMetadata:
        """단일 이미지에서 메타데이터 추출"""

        retry_count = 0
        backoff_time = INITIAL_BACKOFF

        while retry_count < MAX_RETRIES:
            try:
                image = Image.open(image_path)
                prompt = self._build_prompt()
                response = self.model.generate_content([prompt, image])
                result = self._parse_response(response.text, product_id)
                return result

            except Exception as e:
                error_msg = str(e)

                if "429" in error_msg or "Resource exhausted" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                    retry_count += 1

                    if retry_count >= MAX_RETRIES:
                        print(f"Rate limit 최대 재시도 초과: {product_id}")
                        return VisualMetadata(
                            product_id=product_id,
                            color="unknown",
                            style_tags=[],
                            success=False,
                            error_message=f"Rate limit after {MAX_RETRIES} retries"
                        )

                    wait_time = min(backoff_time, MAX_BACKOFF)
                    print(f"Rate limit 감지. {wait_time:.1f}초 대기 후 재시도 ({retry_count}/{MAX_RETRIES})...")
                    time.sleep(wait_time)
                    backoff_time *= 2

                else:
                    print(f"추출 실패 ({product_id}): {error_msg}")
                    return VisualMetadata(
                        product_id=product_id,
                        color="unknown",
                        style_tags=[],
                        success=False,
                        error_message=error_msg
                    )

        return VisualMetadata(
            product_id=product_id,
            color="unknown",
            style_tags=[],
            success=False,
            error_message="Max retries exceeded"
        )

    def _build_prompt(self) -> str:
        """VLM 프롬프트 생성"""
        return f"""이 의류 이미지를 분석해서 다음 정보를 추출해주세요:

주요 색상 (택1):
{', '.join(COLOR_OPTIONS)}

어울리는 스타일 (평균 1-3개):
{', '.join(STYLE_OPTIONS)}

반드시 JSON 형식으로만 응답:
{{"color": "navy", "style_tags": ["캐주얼", "미니멀"]}}
"""

    def _parse_response(self, response_text: str, product_id: str) -> VisualMetadata:
        """응답 파싱"""
        try:
            response_text = response_text.strip()

            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()

            data = json.loads(response_text)

            color = data.get("color", "unknown")
            style_tags = data.get("style_tags", [])

            if color not in COLOR_OPTIONS:
                color_lower = color.lower()
                matched = False
                for valid_color in COLOR_OPTIONS:
                    if valid_color in color_lower or color_lower in valid_color:
                        color = valid_color
                        matched = True
                        break
                if not matched:
                    color = "unknown"

            valid_styles = [s for s in style_tags if s in STYLE_OPTIONS]
            if not valid_styles:
                valid_styles = ["캐주얼"]

            return VisualMetadata(
                product_id=product_id,
                color=color,
                style_tags=valid_styles,
                success=True
            )

        except json.JSONDecodeError as e:
            print(f"JSON 파싱 실패 ({product_id}): {response_text[:100]}")
            return VisualMetadata(
                product_id=product_id,
                color="unknown",
                style_tags=["캐주얼"],
                success=False,
                error_message=f"JSON parse error: {str(e)}"
            )
        except Exception as e:
            print(f"응답 처리 실패 ({product_id}): {str(e)}")
            return VisualMetadata(
                product_id=product_id,
                color="unknown",
                style_tags=["캐주얼"],
                success=False,
                error_message=str(e)
            )


# 배치 처리

class BatchProcessor:
    """배치 처리 관리"""

    def __init__(
        self,
        extractor: GeminiVisualMetadataExtractor,
        musinsa_json_path: str,
        output_path: str,
        checkpoint_path: str
    ):
        self.extractor = extractor
        self.musinsa_json_path = musinsa_json_path
        self.output_path = output_path
        self.checkpoint_path = checkpoint_path

        self.total_count = 0
        self.success_count = 0
        self.failed_count = 0
        self.skipped_count = 0

    def run(self, start_index: int = 0, limit: Optional[int] = None):
        """배치 실행"""

        print(f"\n무신사 데이터 로드: {self.musinsa_json_path}")
        with open(self.musinsa_json_path, 'r', encoding='utf-8') as f:
            products = json.load(f)

        total_products = len(products)
        print(f"총 {total_products}개 상품 로드됨")

        existing_results = self._load_checkpoint()
        processed_ids = set(existing_results.keys())

        if existing_results:
            print(f"체크포인트 발견: {len(existing_results)}개 이미 처리됨")

        end_index = min(start_index + limit, total_products) if limit else total_products
        products_to_process = products[start_index:end_index]

        print(f"\n처리 시작: {start_index} ~ {end_index} ({len(products_to_process)}개)")
        print("=" * 60)

        for idx, product in enumerate(products_to_process, start=start_index + 1):
            product_id = product['id']

            if product_id in processed_ids:
                self.skipped_count += 1
                if idx % 10 == 0:
                    print(f"[{idx}/{end_index}] 스킵 (이미 처리됨): {product_id}")
                continue

            image_path = self._convert_image_path(product['product_image_path'])

            if not os.path.exists(image_path):
                print(f"[{idx}/{end_index}] 이미지 없음: {image_path}")
                existing_results[product_id] = VisualMetadata(
                    product_id=product_id,
                    color="unknown",
                    style_tags=[],
                    success=False,
                    error_message="Image file not found"
                ).to_dict()
                self.failed_count += 1
                continue

            print(f"[{idx}/{end_index}] 처리 중: {product_id} - {product['product_name'][:40]}...")

            result = self.extractor.extract(image_path, product_id)

            existing_results[product_id] = result.to_dict()

            if result.success:
                self.success_count += 1
                print(f"  성공: color={result.color}, styles={result.style_tags}")
            else:
                self.failed_count += 1
                print(f"  실패: {result.error_message}")

            self.total_count += 1

            if idx % 10 == 0:
                self._save_checkpoint(existing_results)
                self._print_progress(idx, end_index)

        print("\n" + "=" * 60)
        print("최종 결과 저장 중...")
        self._save_checkpoint(existing_results)

        self._print_final_stats(total_products)

    def _load_checkpoint(self) -> Dict:
        """체크포인트 로드"""
        if os.path.exists(self.checkpoint_path):
            with open(self.checkpoint_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def _save_checkpoint(self, results: Dict):
        """체크포인트 저장"""
        with open(self.checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    def _convert_image_path(self, windows_path: str) -> str:
        """Windows 경로 변환"""
        parts = windows_path.replace('\\', '/').split('/')

        if 'musinsa_images' in parts:
            idx = parts.index('musinsa_images')
            relative_path = '/'.join(parts[idx:])

            project_root = Path(__file__).parent.parent
            return str(project_root / 'data' / relative_path)

        return windows_path

    def _print_progress(self, current: int, total: int):
        """진행상황 출력"""
        percent = (current / total) * 100
        print(f"\n진행률: {current}/{total} ({percent:.1f}%)")
        print(f"   성공: {self.success_count}, 실패: {self.failed_count}, 스킵: {self.skipped_count}")

    def _print_final_stats(self, total_products: int):
        """최종 통계 출력"""
        print("\n" + "=" * 60)
        print("처리 완료!")
        print("=" * 60)
        print(f"총 상품 수: {total_products}")
        print(f"처리 완료: {self.total_count}")
        print(f"  성공: {self.success_count}")
        print(f"  실패: {self.failed_count}")
        print(f"  스킵: {self.skipped_count}")

        if self.total_count > 0:
            success_rate = (self.success_count / self.total_count) * 100
            print(f"\n성공률: {success_rate:.1f}%")

        print(f"\n결과 저장 위치: {self.checkpoint_path}")


def main():
    """메인 실행 함수"""

    from dotenv import load_dotenv
    project_root = Path(__file__).parent.parent
    load_dotenv(project_root / '.env')

    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        print("오류: GOOGLE_API_KEY 환경 변수가 설정되지 않았습니다.")
        print("\n.env 파일을 확인해주세요:")
        print("  GOOGLE_API_KEY=your-api-key-here")
        sys.exit(1)

    musinsa_json_path = project_root / 'data' / 'musinsa_ranking_result.json'
    output_path = project_root / 'data' / 'visual_metadata.json'
    checkpoint_path = project_root / 'data' / 'visual_metadata_checkpoint.json'

    print("Gemini 2.0 Flash 초기화 중...")
    extractor = GeminiVisualMetadataExtractor(api_key)

    processor = BatchProcessor(
        extractor=extractor,
        musinsa_json_path=str(musinsa_json_path),
        output_path=str(output_path),
        checkpoint_path=str(checkpoint_path)
    )

    processor.run(start_index=0, limit=None)


if __name__ == '__main__':
    main()
