# CloZ-AI 통합 진행 상황

## 전체 목표
팀 리포(CloZ-AI main)의 생성 파이프라인 + 내 피드백 파이프라인 → 하나의 실행 가능한 서비스

## 실행 흐름
```
[사전] 옷장 등록 (별도 모듈) — 이미지 → 분류 → 임베딩 → ChromaDB
[pipeline.py]:
  1. 프롬프트 입력 (자연어)
  2. UnderstandModel이 JSON 파싱
  3. FashionRecommender가 옷장 임베딩과 비교 (카테고리별 3개)
  4. OutfitPlanner가 최적 조합 선택
  5. VTONManager가 가상 피팅 이미지 생성
  6. 사용자 판단 → 추천-피드백 루프 (REGENERATE/BUYING/ASK_MORE/APPROVED)
```

---

## Step 1: 팀 코드 복사 + 디렉토리 생성
- [ ] `src/generation_pipeline/` 디렉토리 구조 생성
- [ ] `fashion_engine/*` 복사
- [ ] `understand_model/*` 복사
- [ ] `utils/load.py` 복사
- [ ] `configs/` 디렉토리 복사 (generation_model.yaml, llm_base_understand.yaml, json_template.json)
- [ ] `__init__.py` 파일들 생성

## Step 2: 복사한 사본의 import 경로 수정
- [ ] `fashion_engine/encoder.py` — `from utils.load` → `from ..utils.load`, parents[2] → parents[3]
- [ ] `fashion_engine/recommender.py` — 동일 import 수정 + parents 수정
- [ ] `fashion_engine/db_manager.py` — import 수정
- [ ] `understand_model/understand_model.py` — import 수정 + parents 수정
- [ ] import 테스트: `python -c "from src.generation_pipeline import CLIPEncoder"`

## Step 3: 통합 설정 (src/config.py)
- [ ] UnifiedConfig 클래스 생성
- [ ] ChromaDB 경로 통일
- [ ] 컬렉션명 설정 관리

## Step 4: RealGenerationModel 구현
- [ ] `src/feedback_pipeline/interfaces/real_generation_model.py` 생성
- [ ] GenerationModelInterface 구현 (generate, regenerate)
- [ ] main_adapter.py의 convert_outfit_to_outfitset() 재사용

## Step 5: 통합 진입점 (src/pipeline.py)
- [ ] CloZPipeline 클래스 생성
- [ ] run() — 전체 파이프라인 1회 실행
- [ ] process_feedback() — 피드백 처리 + REGENERATE 자동 재생성
- [ ] interactive_session() — 대화형 CLI
- [ ] `python -m src.pipeline`으로 실행 가능

## Step 6: requirements.txt 병합
- [ ] 팀 의존성 추가 (loguru, pydantic 등)
- [ ] .env.example 업데이트

## Step 7: 검증
- [ ] import 테스트
- [ ] 기존 feedback_pipeline 동작 확인
- [ ] 통합 pipeline 전체 시나리오 테스트
- [ ] pytest tests/

---

## 주의사항
- 팀 main 코드 원본은 절대 수정 안 함 (복사한 사본만 수정)
- recommender.py:194 globals() 버그 → 팀에 전달 (우리 코드에서 우회)
- 카테고리 매핑: 팀 "하의" ↔ 내 "바지" → main_adapter.py가 처리
- CLIPEncoder(온라인) vs FashionCLIPEmbedder(오프라인) 둘 다 유지
- fashn_vton import 실패 시 graceful fallback

---

## 참조 경로
- 팀 리포 클론: `/Users/mj/Downloads/CloZ-AI-main/`
- 내 리포: `/Users/mj/Downloads/whats-in-my-closet/`
- 플랜 파일: `/Users/mj/.claude/plans/quiet-coalescing-shore.md`
