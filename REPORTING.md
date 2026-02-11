# Reporting Log

개발자가 바로 파악할 수 있게 `기존 -> 현재 / 이유` 형식으로 누적 기록합니다.

## 2026-02-11

### 1) ChromaDB 컬렉션명 정합성
- 파일: `src/generation_pipeline/fashion_engine/recommender.py`
- 기존 -> 현재: `load_user_wardrobe(collection_name="mycloset-embedding")` -> `load_user_wardrobe(collection_name="wardrobe")`
- 이유: 실제 `chroma_wardrobe` DB의 컬렉션명이 `wardrobe`라서, 기존 값은 빈 컬렉션을 만들어 추천이 0건으로 떨어짐.

### 2) Chroma upsert 포맷 오류 수정
- 파일: `src/generation_pipeline/fashion_engine/recommender.py`
- 기존 -> 현재: `metadatas={"item_key": ..., "broad_cat": ...}` -> `metadatas=[{"item_key": ..., "broad_cat": ...}]`
- 이유: ChromaDB `upsert`는 `ids/embeddings/metadatas` 길이 정합이 필요한 리스트 포맷을 기대함. 기존 포맷은 런타임 실패/누락 유발 가능.

### 3) 옷장 경로 자동 보정 추가
- 파일: `src/generation_pipeline/fashion_engine/recommender.py`
- 기존 -> 현재: 설정 경로만 사용 -> 설정 경로 실패 시 `closet_mj` 후보 경로 자동 탐색(`project_root/cwd` 기준)
- 이유: Colab/로컬에서 경로가 `data/closet_mj` vs `closet_mj`로 자주 달라 초기화 실패가 반복됨.

### 4) 옷장 로딩 실패 가시성 강화
- 파일: `src/generation_pipeline/fashion_engine/recommender.py`
- 기존 -> 현재: 예외 다수 `except: pass` -> 실패 원인별 로그(`user_clothes_dir`, Chroma 조회/업서트, 이미지 임베딩 실패, 0개 로드) 출력
- 이유: 기존엔 실패해도 조용히 넘어가서 원인 파악이 어려웠음.

### 5) VTON 필수화
- 파일: `src/generation_pipeline/fashion_engine/vton.py`
- 기존 -> 현재: 초기화 실패 시 warning 후 진행 -> `ImportError/RuntimeError` 즉시 발생
- 이유: 요구사항상 VTON 없는 실행은 허용되지 않음.

### 6) VTON 가중치 경로 탐색 보강
- 파일: `src/generation_pipeline/fashion_engine/vton.py`
- 기존 -> 현재: `src/generation_pipeline/fashn_vton/weights` 단일 경로 -> `FASHN_VTON_WEIGHTS_DIR`, `<project_root>/weights`, `<cwd>/weights`, 기존 경로 순 탐색
- 이유: Colab에서 실제 다운로드 위치가 프로젝트 루트 `weights/`인 경우가 많아 초기화 실패 발생.

### 7) 파이프라인의 VTON 실패 처리 변경
- 파일: `src/pipeline.py`
- 기존 -> 현재: `_init_vton()` 실패 시 `None` 반환 후 계속 진행 -> `RuntimeError` 발생으로 즉시 실패
- 이유: VTON 필수 정책 반영.

### 8) 생성 단계 VTON 전제조건 명시
- 파일: `src/feedback_pipeline/interfaces/real_generation_model.py`
- 기존 -> 현재: `self.vton`/`user_body_image` 없어도 일부 경로 진행 -> 조건 미충족 시 즉시 `GenerationResult(success=False, ...)`
- 이유: 실패를 늦게 발견하지 않도록 초기에 명확히 중단.

### 9) CUDA/CPU 디바이스 불일치 수정
- 파일: `src/generation_pipeline/fashion_engine/recommender.py`
- 기존 -> 현재: `q_emb`(CUDA)와 `item_emb`(CPU) 혼합 계산 -> 추천 계산 텐서를 CPU로 통일
- 이유: `Expected all tensors to be on the same device (cuda:0 and cpu)` 런타임 오류 해결.

### 10) Colab 경로 설정 정리
- 파일: `configs/generation_model.yaml`
- 기존 -> 현재: `user_clothes_dir`가 `.../data/closet_mj`, 이미지 경로는 상대경로 -> `.../closet_mj` 및 `tops/bottoms/outers` 절대경로로 통일
- 이유: 실제 폴더 구조와 설정 불일치 해소.

### 11) 의존성 추가
- 파일: `requirements.txt`
- 기존 -> 현재: `rembg` 없음 -> `rembg>=2.0.0` 추가
- 이유: `fashn-vton` 실행 중 `ModuleNotFoundError: rembg` 발생.

### 12) 스타일 키 미일치 보정 (casual vs 캐주얼)
- 파일: `src/feedback_pipeline/interfaces/real_generation_model.py`
- 기존 -> 현재: 기본 스타일/영문 스타일값이 `casual` 그대로 사용 -> `casual -> 캐주얼` 등 alias 매핑 후 사용, 기본값도 `캐주얼`로 변경
- 이유: 스타일 프로필 키가 한글(예: `캐주얼`) 중심이라 `casual` 입력 시 플래너에서 타겟 스타일 미존재로 실패.

### 13) 플래너 스타일 폴백 추가
- 파일: `src/generation_pipeline/fashion_engine/planner.py`
- 기존 -> 현재: 타겟 스타일 미존재 시 즉시 `[]` 반환 -> 첫 번째 사용 가능한 스타일로 자동 대체 후 평가 진행
- 이유: 스타일 키 불일치/입력 편차가 있어도 조합 평가가 완전히 중단되지 않도록 하기 위함.

### 14) 15개 스타일 전체 영문/변형 매핑 추가
- 파일: `src/feedback_pipeline/interfaces/real_generation_model.py`
- 기존 -> 현재: 일부 스타일(`casual` 등)만 제한 매핑 -> 레퍼런스 15개 스타일 전체(고프코어/레트로/로맨틱/리조트/미니멀/스트릿/스포티/시크/시티보이/아웃도어/오피스/워크웨어/캐주얼/클래식/프레피)로 영문/표기 변형 매핑 확장
- 이유: LLM 출력이 영어/하이픈/언더스코어/띄어쓰기 변형으로 들어와도 style profile 키(한글)와 안정적으로 매칭되도록 보장.

### 15) BUYING 출력 크래시 수정 (`recommendations` -> `products`)
- 파일: `src/pipeline.py`
- 기존 -> 현재: `decision.buying_recommendations.recommendations` 접근 -> `decision.buying_recommendations.products` 접근 + dataclass(`to_dict`) 처리
- 이유: `BuyingRecommendation` 스키마는 `recommendations` 필드가 없고 `products`만 있어 `AttributeError`로 인터랙티브 세션이 중단됨.

### 16) 샘플 CLI BUYING 출력도 동일 수정
- 파일: `src/feedback_pipeline/main.py`
- 기존 -> 현재: `decision.buying_recommendations.recommendations` 접근 -> `products` 접근 + dataclass(`to_dict`) 처리
- 이유: 동일한 필드명 불일치 버그가 테스트/데모 CLI에도 존재.

### 17) REGENERATE 대상 범위 전달 누락 수정 (v2 경로)
- 파일: `src/feedback_pipeline/agents/manager_agent.py`
- 기존 -> 현재: `_create_regenerate_decision_v2()`에서 `target_categories` 미설정 -> `feedback_scopes` 기반 `target_categories` 설정
- 이유: 재생성 시 어떤 카테고리를 바꿔야 하는지 하위 생성 모델이 몰라 동일 코디 재선정 가능성이 높았음.

### 18) 재생성 시 기존 코디 재선정 방지(범위 기반 제외) 추가
- 파일: `src/feedback_pipeline/interfaces/real_generation_model.py`
- 기존 -> 현재: `exclude_items` 전달만 하고 실제 추천 필터링에 미사용 -> `exclude_map` 생성 후 카테고리별 추천 결과에서 기존 아이템 제거
- 이유: `FULL/TOP/BOTTOM/OUTER` 피드백 범위에 맞춰 기존 착장을 후보에서 배제해야 재생성 결과가 실질적으로 변경됨.

### 19) 재생성 후보셋 전달 스키마 추가
- 파일: `src/feedback_pipeline/interfaces/wardrobe_checker.py`
- 기존 -> 현재: `WardrobeCheckResult`에 `matching_items`만 존재 -> `candidate_pool`(TOP/BOTTOM/OUTER별 후보 ID 목록) 필드 추가
- 이유: 재생성에서 단순 가능/불가가 아니라 실제 후보 집합을 생성 모듈로 전달하기 위함.

### 20) 옷장체커에서 카테고리별 후보풀 생성
- 파일: `src/feedback_pipeline/checkers/embedding_wardrobe_checker.py`
- 기존 -> 현재: 검색 결과를 단일 리스트(`matching_items`)로만 반환 -> broad_cat/id 패턴 기반으로 `candidate_pool` 구성 후 반환
- 이유: 재생성에서 카테고리 단위 후보 제한(예: 니트/데님 기반)을 적용하려면 카테고리별 후보가 필요.

### 21) Manager -> Regenerate payload로 후보풀 전달
- 파일: `src/feedback_pipeline/agents/manager_agent.py`
- 기존 -> 현재: `_create_regenerate_decision_v2()`가 `structured_query`만 전달 -> `check_result.candidate_pool`도 `regenerate_data`에 포함
- 이유: 워드로브 체크 결과를 실제 재생성 입력으로 연결해 "체커 후보 내 최적화" 흐름을 만들기 위함.

### 22) Buying 추천 다중 세부카테고리 3개씩 우선 선택
- 파일: `src/feedback_pipeline/checkers/embedding_buying_trigger.py`
- 기존 -> 현재: 다중 detail_cat도 OR 검색 후 전체 상위 N개 -> 각 detail_cat별 상위 3개씩 우선 선택 후 반환(중복 제거)
- 이유: `Knitwear + Denim`처럼 복수 요청 시 한 카테고리로 쏠리지 않게 하기 위함.

### 23) 생성 모듈 재생성 경로 보강 (후보셋/아우터 제외)
- 파일: `src/pipeline.py`
- 기존 -> 현재: REGENERATE에서 `structured_query`만 전달 -> `candidate_pool`도 constraints로 전달
- 이유: Manager가 만든 후보풀을 generation 쪽에서 실제 사용하도록 연결.

### 24) 재생성에서 후보풀 기반 필터 + 아우터 제외 신호 처리
- 파일: `src/feedback_pipeline/interfaces/real_generation_model.py`
- 기존 -> 현재: 재생성 시 recommender 전체 검색 재사용 -> `candidate_pool` 필터 적용 + 피드백 텍스트에서 `아우터 제외` 감지 시 `include_outer=False`
- 이유: 재생성을 "전체 재검색"이 아닌 "전달된 후보 내부 재조합"으로 수렴시키기 위함.

### 25) 조합기에서 아우터 optional 지원
- 파일: `src/generation_pipeline/fashion_engine/planner.py`
- 기존 -> 현재: `(pant, outer, shirt)` 3피스 필수 -> `include_outer=False`일 때 `(pant, None, shirt)` 조합 생성 가능
- 이유: "아우터 빼줘" 피드백을 조합 단계에서 반영하기 위함.

### 26) VTON/Adapter의 outer=None 호환 처리
- 파일: `src/generation_pipeline/fashion_engine/vton.py`, `src/feedback_pipeline/adapters/main_adapter.py`
- 기존 -> 현재: outer 딕셔너리 전제 접근 -> outer가 `None`인 경우 로그/변환에서 안전하게 skip
- 이유: 아우터 제외 조합이 들어와도 파이프라인이 깨지지 않도록 하기 위함.

### 27) candidate_pool 키를 파일명 기준으로 통일
- 파일: `src/feedback_pipeline/checkers/embedding_wardrobe_checker.py`
- 기존 -> 현재: `candidate_pool`에 `mj:tops/tops_15.jpeg` 같은 DB ID 저장 -> `tops_15.jpeg` 같은 파일명 기준 저장
- 이유: 재생성 추천 아이템 ID(`shirt/...`)와 옷장 DB ID(`mj:tops/...`) 스키마가 달라 매칭이 깨지던 문제 해소.

### 28) 재생성 필터 fail-closed 적용
- 파일: `src/feedback_pipeline/interfaces/real_generation_model.py`
- 기존 -> 현재: 후보/제외 필터 매칭 실패 시 원본 후보로 롤백(`kept if kept else items`) -> 롤백 제거, 매칭 실패 시 빈 후보 유지
- 이유: 후보 제한 실패가 곧바로 전체 후보 재검색으로 풀려 재생성 제약이 무력화되던 문제 방지.

### 29) 아우터 제외 신호 인식 강화
- 파일: `src/feedback_pipeline/interfaces/real_generation_model.py`
- 기존 -> 현재: 고정 문자열 일부만 감지(`아우터 제외`, `아우터 빼`) -> `아우터/겉옷/outer` + `제외/빼/없애/remove/without` 조합 감지
- 이유: 표현 변형(예: "아우터는 제외해주고")에서도 `include_outer=False`를 일관되게 적용하기 위함.

### 30) 내부 키 네이밍 TOP/BOTTOM/OUTER 통일
- 파일: `src/feedback_pipeline/interfaces/real_generation_model.py`
- 기존 -> 현재: `상의/하의/아우터`와 `shirt/pant/outer` 혼용 -> 내부 필터 로직은 `TOP/BOTTOM/OUTER`로 통일, planner 경계에서만 변환
- 이유: 재생성/후보필터/제외필터에서 키 변환 누락으로 발생하는 불일치 제거.

### 31) BUYING 복수 카테고리 3개씩 보강 (분리 쿼리 포함)
- 파일: `src/feedback_pipeline/checkers/embedding_buying_trigger.py`
- 기존 -> 현재: 다중 detail_cat에서 OR 검색 후 전체 상위 정렬 -> detail_cat별로 최대 3개를 우선 채우고 부족 시 카테고리별 추가 쿼리로 보완
- 이유: `Knitwear + Denim` 요청 시 한쪽 카테고리로 쏠리지 않고 각 파트 추천을 분리 보장하기 위함.

### 32) BUYING 결과를 카테고리별 분리 출력
- 파일: `src/feedback_pipeline/interfaces/buying_trigger.py`, `src/pipeline.py`, `src/feedback_pipeline/main.py`
- 기존 -> 현재: 단일 리스트 출력 -> `grouped_products` 필드 추가 후 `[Knitwear]`, `[Denim]`처럼 그룹별 3개씩 표시
- 이유: 복수 파트 추천을 섞지 않고 사용자 가독성을 높이기 위함.

### 33) WardrobeCheckResult 타입 정합성 수정
- 파일: `src/feedback_pipeline/interfaces/wardrobe_checker.py`
- 기존 -> 현재: `matching_items: List[int]` -> `matching_items: List[str]`
- 이유: 실제 저장/전달되는 아이템 식별자가 문자열 ID이므로 타입 선언 정합성 보완.

## Error Reporting (Runtime)

### E-01) `ModuleNotFoundError: No module named 'fashn_vton'`
- 기존 -> 현재: 모듈 미설치 상태에서 import 실패 -> `fashn-vton` 설치 후 import 가능 상태
- 이유: VTON 핵심 의존성 누락.

### E-02) `ModuleNotFoundError: No module named 'rembg'`
- 기존 -> 현재: `fashn-vton` 종속 패키지 누락 -> `requirements.txt`에 `rembg` 추가 및 설치 가이드 반영
- 이유: VTON 전처리 의존성 누락.

### E-03) `ModuleNotFoundError: No module named 'src'` (Colab)
- 기존 -> 현재: 프로젝트 루트가 `sys.path`에 미포함 -> Colab에서 루트 경로 추가 후 import 가능
- 이유: Colab 실행 경로와 패키지 경로 불일치.

### E-04) `옷장에서 매칭되는 아이템을 찾을 수 없습니다`
- 기존 -> 현재: 컬렉션명/경로/업서트 포맷 문제로 item_db가 비거나 후보가 누락 -> 컬렉션명 `wardrobe`, 경로 자동보정, upsert 포맷 및 로그 보강
- 이유: 옷장 로딩/캐시 정합성 오류.

### E-05) `user_clothes_dir 경로가 존재하지 않습니다`
- 기존 -> 현재: `.../data/closet_mj` 하드코딩 경로 사용 -> 실제 Colab 경로 및 자동보정 fallback 적용
- 이유: 한글 정규화 포함 경로 불일치.

### E-06) `Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu`
- 기존 -> 현재: 추천 스코어 계산 시 GPU/CPU 혼합 텐서 연산 -> 계산 텐서 디바이스 통일
- 이유: 임베딩 텐서 디바이스 불일치.

### E-07) `타겟 스타일 'casual'이 프로필에 없습니다`
- 기존 -> 현재: 영문 스타일 라벨이 한글 style profile 키와 미매칭 -> 15개 스타일 alias 매핑 + 플래너 fallback 적용
- 이유: 스타일 라벨 체계(영문 입력 vs 한글 DB) 불일치.

### E-08) `AttributeError: 'BuyingRecommendation' object has no attribute 'recommendations'`
- 기존 -> 현재: BUYING 출력에서 존재하지 않는 필드 접근 -> `products` 필드 접근으로 수정
- 이유: 데이터 모델(`BuyingRecommendation`)과 출력 코드 필드명 불일치.

### E-09) REGENERATE 후 동일/유사 코디 반복
- 기존 -> 현재: 재생성 시 추천 후보 제한 없이 전체 재검색 -> 범위 기반 제외 + candidate_pool 전달/적용 + 아우터 제외 신호 처리
- 이유: 재생성 경로가 초기 생성 경로와 거의 동일하게 동작하여 변경 강도가 약했음.

## 운영 규칙
- 앞으로 수정 건은 동일하게 `기존 -> 현재 / 이유` 형식으로 즉시 추가합니다.

### 34) 이전 아이템 제거 로직을 유사도 기반 -> ID 기반으로 변경
- 파일: `src/feedback_pipeline/checkers/embedding_wardrobe_checker.py`
- 기존 -> 현재: `_filter_previous_items()`에서 임베딩 코사인 유사도(0.9 기준)로 후보 제거 -> 후보 ID/파일명과 이전 착장 파일명을 정규화해 exact match만 제거
- 이유: "다른 니트/다른 데님"처럼 같은 카테고리의 정상 대안까지 유사도 필터로 함께 제거되는 문제를 방지.

### 35) candidate_pool 필터를 완전 fail-closed로 보강
- 파일: `src/feedback_pipeline/interfaces/real_generation_model.py`
- 기존 -> 현재: `candidate_pool` 키가 있어도 빈 목록이면 `if not allowed` 분기로 전체 후보 유지 -> `cat not in normalized_pool`일 때만 유지하고, 키가 존재하면 빈 목록도 그대로 적용(=빈 후보)
- 이유: 매칭 실패/빈 후보 상황에서 전체 후보로 풀리는 우회 경로를 제거해 재생성 제약을 강제.

### 36) 생성 로그 키 표기 통일(top/bottom/outer)
- 파일: `src/generation_pipeline/fashion_engine/planner.py`, `src/generation_pipeline/fashion_engine/vton.py`
- 기존 -> 현재: 로그에 `pant/shirt/outer` 혼용 -> 로그 표기를 `top/bottom/outer`로 통일 (내부 호환을 위해 결과 키는 기존 `pant/shirt` 유지 + `top/bottom` 별칭 추가)
- 이유: 운영 로그 가독성과 디버깅 일관성을 높이기 위함.

### 37) `target_detail_cats` 복수 요청 시 카테고리별 bypass 적용
- 파일: `src/feedback_pipeline/checkers/embedding_wardrobe_checker.py`
- 기존 -> 현재: `Knitwear+Denim`을 OR 단일 쿼리로 합쳐 `total_results<=3`만 bypass 판단 -> `detail_cat`별 분리 쿼리로 각각 `<=3` bypass를 독립 적용
- 이유: 카테고리별 재고가 적은 경우(예: Knitwear 2, Denim 2)에도 합산 4개 때문에 bypass가 꺼져 한쪽 후보가 0개 되는 문제를 방지.

### 38) 재생성 추천 조회 top_k 확장 (candidate_pool 교집합 손실 방지)
- 파일: `src/feedback_pipeline/interfaces/real_generation_model.py`
- 기존 -> 현재: 재생성에서도 `recommend_from_agent(..., top_k=item_top_k=3)` 고정 -> `candidate_pool`가 있을 때 `top_k=max(item_top_k, 20)`로 확장 후 후보풀 교집합 적용
- 이유: 상위 3개에 원하는 카테고리 아이템(예: tops_15)이 포함되지 않으면 `candidate_pool` 교집합 단계에서 카테고리 후보가 0개가 되어 재생성이 실패하는 문제를 방지.

### 39) 재생성 실패 원인 메시지 출력 보강
- 파일: `src/pipeline.py`
- 기존 -> 현재: 인터랙티브 세션에서 실패 시 `재생성에 실패했습니다.`만 출력 -> `GenerationResult.message`를 함께 출력
- 이유: `TOP/BOTTOM 후보 부족` 같은 실제 실패 원인을 즉시 확인해 디버깅 시간을 줄이기 위함.

### 40) `candidate_pool`에서 빈 카테고리 키 제거
- 파일: `src/feedback_pipeline/checkers/embedding_wardrobe_checker.py`
- 기존 -> 현재: `candidate_pool`를 `{TOP:[], BOTTOM:[], OUTER:[]}`로 항상 채워 반환 -> 실제 후보가 있는 카테고리 키만 반환
- 이유: TOP만 재생성해야 하는 요청에서도 빈 `BOTTOM/OUTER`가 fail-closed에 걸려 하의/아우터 후보를 0개로 만들어 재생성이 실패하던 문제를 방지.

### 41) 재생성 실패 메시지 전달 누락 수정
- 파일: `src/pipeline.py`
- 기존 -> 현재: 재생성 실패 시 `new_generation_result`를 result에 저장하지 않아 UI에서 `(원인 미상)` 출력 -> 성공/실패와 무관하게 `new_generation_result` 저장
- 이유: 실패 원인 메시지(예: 후보 부족)를 사용자/개발자가 즉시 확인할 수 있도록 하기 위함.

### 42) 파트 선택 재생성 시 비선택 파트 고정(keep_map) 적용
- 파일: `src/feedback_pipeline/interfaces/real_generation_model.py`
- 기존 -> 현재: 선택한 파트만 제한하고 나머지 파트는 재추천되어 변경 가능 -> `keep_map` 도입으로 비선택 파트(TOP/BOTTOM/OUTER)는 기존 아이템 파일명으로 강제 고정
- 이유: 사용자 의도(예: TOP만 변경, 하의/아우터 유지)를 재생성 결과에 정확히 반영하기 위함.

### 43) 선택 파트 외 candidate_pool 적용 차단
- 파일: `src/feedback_pipeline/interfaces/real_generation_model.py`
- 기존 -> 현재: candidate_pool에 TOP/BOTTOM/OUTER가 섞여 들어오면 비선택 파트까지 제한 가능 -> `target_categories` 기준으로 변경 대상 파트에만 candidate_pool 적용
- 이유: 예를 들어 TOP만 변경 요청인데 OUTER/BOTTOM 후보가 의도치 않게 축소되는 부작용 방지.

### 44) 비선택 파트 고정 실패 시 기존 아이템 강제 복원
- 파일: `src/feedback_pipeline/interfaces/real_generation_model.py`
- 기존 -> 현재: keep_map 필터 후 카테고리 후보가 0개면 재생성 실패 -> `item_db`에서 기존 파일명 매칭으로 해당 파트 1개를 강제 복원
- 이유: 파트 고정 정책("선택 파트만 변경")을 실패 없이 보장하기 위함.

### 45) TOP/BOTTOM 단일 파트 재생성 시 OUTER 오판 제외 버그 수정
- 파일: `src/feedback_pipeline/interfaces/real_generation_model.py`
- 기존 -> 현재: `_should_include_outer()`가 detail_cat(top/bottom 계열)만 보고 `include_outer=False`로 내려 아우터를 의도치 않게 제거 -> `target_categories`에 OUTER가 없고 FULL도 아니면 OUTER 유지(`True`)를 우선 적용
- 이유: "TOP만 변경" 요청에서 비선택 파트(아우터)가 사라지는 잘못된 동작을 방지.

### 46) 재생성 의사결정 케이스 매트릭스 시뮬레이션 수행
- 대상: TOP 단일 변경, TOP+BOTTOM 복수 변경, 아우터 명시 제외
- 기존 -> 현재: 수정 후 케이스별 기대 불변식(비선택 파트 고정, 명시 제외만 아우터 제거) 검증 없음 -> 3개 시나리오를 규칙 기반 시뮬레이션으로 점검
- 이유: 단일 버그 수정 후 연쇄 회귀를 사전에 잡기 위한 안전장치.

### 47) 부정문 의도(말고/제외/빼고) 규칙 보정 추가
- 파일: `src/feedback_pipeline/utils/query_builder.py`
- 기존 -> 현재: LLM이 `니트 말고`를 `target_detail_cats=['Knitwear']`로 오해해도 그대로 사용 -> 후처리 규칙으로 부정문 주변 키워드를 `avoid_detail_cats`로 강제 이동, 충돌 시 avoid 우선
- 이유: 자연어 부정 의도 역전으로 인해 재생성/구매 추천이 정반대로 동작하던 문제를 방지.

### 48) 옷장 체크에 제외 세부카테고리 필터 반영
- 파일: `src/feedback_pipeline/checkers/embedding_wardrobe_checker.py`, `src/feedback_pipeline/agents/manager_agent.py`, `src/feedback_pipeline/interfaces/wardrobe_checker.py`
- 기존 -> 현재: `target_detail_cats` include 필터만 존재 -> `avoid_detail_cats`를 추가해 후보 수집 단계에서 제외 적용, Manager에서 scope 기반 target_categories와 함께 전달
- 이유: "니트 말고" 같은 요청에서 니트 후보가 재생성 후보풀에 남지 않도록 보장.

### 49) BUYING 추천에도 제외 세부카테고리 반영
- 파일: `src/feedback_pipeline/checkers/embedding_buying_trigger.py`
- 기존 -> 현재: 구매 추천은 `target_detail_cats`만 반영 -> `avoid_detail_cats`를 후보 필터에 반영해 제외 카테고리 상품을 제거
- 이유: 재생성 실패 후 BUYING으로 전환되어도 사용자의 회피 의도(예: 니트 제외)를 일관되게 유지.

### 50) 최신 피드백 원문 기반 부정 카테고리 보강(Manager 안전장치)
- 파일: `src/feedback_pipeline/agents/manager_agent.py`
- 기존 -> 현재: QueryBuilder/LLM 결합 결과만 신뢰 -> 최신 피드백 원문(`feedback.feedback_text`)에서 `말고/제외/빼고` 패턴을 직접 추출해 `avoid_detail_cats`를 강제 병합
- 이유: LLM 결합 단계에서 부정 의도가 희석되어도 의도 역전(예: "니트 말고" -> 니트 추천)되지 않도록 이중 보호.

### 51) 부정 요청에서 후보 과소 시 동일 파트 fallback 확장
- 파일: `src/feedback_pipeline/checkers/embedding_wardrobe_checker.py`
- 기존 -> 현재: `avoid_detail_cats` 적용 후 후보가 1개 이하로 줄어들면 그대로 단일 조합 -> 같은 파트(target_categories 기준)에서 비제외 카테고리 후보를 추가 조회(`n_results=50`, threshold bypass)하여 후보풀 확장
- 이유: "니트 말고" 같은 요청에서 하나만 남는 단일 조합 현상을 줄이고, 실제로 '다른 카테고리' 대안을 찾도록 유도.

### 52) 추천 점수에서 season/mood 유사도가 실제 반영되지 않던 버그 수정
- 파일: `src/generation_pipeline/fashion_engine/recommender.py`
- 기존 -> 현재: season/mood 유사도를 `globals()`에 기록해 로컬 점수 변수(`season_sim`, `mood_sim`)에 반영되지 않음 -> 로컬 변수에 직접 반영
- 이유: 프롬프트에 계절/무드 정보가 들어와도 최종 점수가 거의 변하지 않아 동일 코디가 반복되던 문제를 해결.

### 53) 초기 생성 입력 스키마 정규화 추가(understand -> recommender)
- 파일: `src/feedback_pipeline/interfaces/real_generation_model.py`
- 기존 -> 현재: `understand_model` 응답이 `style` 없이 들어오면 recommender가 사실상 기본값 경로로 수렴 -> `_normalize_initial_agent_json()`으로 필드 구조 보정 + mood/prompt 기반 style 보강 후 추천
- 이유: "여름 코디", "운동하기 편한 옷" 같은 요청이 초기 생성 점수에 실제 반영되도록 하기 위함.

### 54) 피드백 범위 입력 파싱 개선 (쉼표 지원)
- 파일: `src/pipeline.py`
- 기존 -> 현재: 공백 기준 split만 사용해 `3, 4` 입력 시 `4`만 파싱되거나 실패 -> 정규식으로 `1~4` 숫자 토큰을 추출해 `3,4`/`3, 4`/`3 4` 모두 지원
- 이유: 복수 범위 입력이 잘못 파싱되어 의도와 다른 액션(단일 OUTER 처리 등)으로 흐르던 문제 방지.

### 55) 복수 범위 BUYING에서 단일 카테고리 강제 제거
- 파일: `src/feedback_pipeline/agents/manager_agent.py`
- 기존 -> 현재: BUYING 시 항상 첫 번째 scope를 primary로 사용 -> scope가 2개 이상이면 `FULL`로 전달하여 단일 category_main 필터를 강제하지 않음
- 이유: `BOTTOM+OUTER` 같은 복수 피드백에서 `OUTER + Short` 같은 모순 필터가 생겨 추천이 비는 문제를 방지.

### 56) BUYING detail_cat/category_main 정합성 필터 추가
- 파일: `src/feedback_pipeline/checkers/embedding_buying_trigger.py`
- 기존 -> 현재: target_category와 맞지 않는 detail_cat(예: OUTER + Short)도 그대로 where에 반영 -> 카테고리별 허용 detail_cat만 유지하도록 sanitize
- 이유: 상충 필터로 검색 결과 0개가 되는 문제를 차단하고 추천 안정성 향상.

### 57) BUYING 결과 0건 시 no-threshold fallback 추가
- 파일: `src/feedback_pipeline/checkers/embedding_buying_trigger.py`
- 기존 -> 현재: threshold/필터 조합으로 후보가 0개면 그대로 빈 추천 반환 -> 동일 카테고리 조건으로 재조회 후 `min_similarity=None`(threshold 비적용)로 fallback 추천 수행
- 이유: `[결정]: BUYING` 이후 추천 목록이 비어 사용자 경험이 끊기는 문제를 방지.

### 58) BUYING 출력부 빈 grouped 처리 보강
- 파일: `src/pipeline.py`, `src/feedback_pipeline/main.py`
- 기존 -> 현재: `grouped_products` 키가 존재하지만 각 그룹이 비어 있으면 아무 항목도 출력되지 않음 -> grouped 출력 여부(`printed`)를 추적해 미출력 시 일반 `products`로 폴백, 그래도 없으면 안내 문구 출력
- 이유: 추천 데이터가 비어 있거나 그룹 구성에 실패해도 CLI에서 상태를 명확히 보여 디버깅 가능성을 높이기 위함.

### 59) 코드 이모티콘 로그 제거
- 파일: `src/generation_pipeline/fashion_engine/vton.py`
- 기존 -> 현재: 로그 문자열에 `✅`, `🎯` 이모티콘 포함 -> 동일 의미의 일반 텍스트 로그로 변경
- 이유: 운영 로그를 텍스트 기반으로 통일하고 가독성/파싱 안정성을 높이기 위함.

### 60) 번호형 주석 접두어 제거 (`# 1.`, `# 2.` ...)
- 파일: `src/**/*.py`
- 기존 -> 현재: 주석 라인에 번호 접두어가 포함된 형태(`# 1. ...`) 다수 존재 -> 주석 문장은 유지하고 번호 접두어만 일괄 제거
- 이유: 주석 스타일을 단일화하고 수정 시 번호 재정렬 부담을 줄이기 위함.

### 61) 재생성 경로 리팩토링: 스코프/아이템 매칭 규칙 단일화
- 파일: `src/feedback_pipeline/interfaces/real_generation_model.py`
- 기존 -> 현재: `TOP/BOTTOM/OUTER` 스코프 판별, 카테고리 매핑, 파일명/ID 매칭 로직이 여러 함수에 중복 구현되어 수정 시 회귀 위험이 큼 -> 공통 유틸(`_map_product_scope`, `_extract_change_scopes`, `_is_full_scope_requested`, `_item_name_tokens`, `_filter_items_by_name_set`)로 통합하고 `exclude/keep/candidate_pool` 필터가 동일 규칙을 사용하도록 정리
- 이유: CTO 관점에서 서비스 운영 시 가장 높은 리스크는 "같은 규칙이 여러 곳에서 다르게 동작"하는 구조이며, 이번 통합으로 재생성 로직의 규칙 일관성과 유지보수성을 높이기 위함.

### 62) ManagerAgent 스코프 처리 공통화
- 파일: `src/feedback_pipeline/agents/manager_agent.py`
- 기존 -> 현재: `feedback_scopes` 해석(`FULL` 기본값, 문자열 변환, 옷장 카테고리 매핑)이 메서드별로 중복 구현 -> `_normalized_feedback_scopes`, `_feedback_scope_values`, `_to_wardrobe_target_categories` 헬퍼로 일원화
- 이유: REGENERATE/BUYING/옷장체크 경계에서 scope 해석이 달라지는 회귀를 막고, 정책 변경 시 한 지점에서 관리 가능하게 하기 위함.

### 63) `requirements`에 FASHN VTON 직접 포함
- 파일: `requirements.txt`
- 기존 -> 현재: Colab에서 `git clone` + `pip install -e`를 별도 셀로 수동 실행해야 했음 -> `git+https://github.com/fashn-AI/fashn-vton-1.5.git`, `fashn-human-parser`, `huggingface_hub`를 requirements에 포함
- 이유: `pip install -r requirements.txt` 한 번으로 VTON 패키지 의존성까지 바로 설치되게 해 초기 셋업 단계를 단축하기 위함.

### 64) VTON 가중치 다운로드 스크립트 내장화
- 파일: `scripts/setup_vton_weights.py`
- 기존 -> 현재: 외부 repo의 `download_weights.py`를 직접 실행해야 했음 -> 내부 스크립트로 `model.safetensors`, `dwpose` ONNX 파일 다운로드 + human parser warmup을 수행
- 이유: 외부 repo clone 의존을 제거하고, 동일한 가중치 준비 절차를 프로젝트 내부에서 재현 가능하게 만들기 위함.

### 65) Colab 원커맨드 부트스트랩 추가
- 파일: `scripts/bootstrap_colab.py`
- 기존 -> 현재: 설치/가중치 준비를 여러 셀로 수동 분리 실행 -> 부트스트랩 스크립트 1회 실행으로 `pip install -r requirements.txt` + 가중치 다운로드 수행
- 이유: Colab 런타임 초기화 시 사람이 순서 실수하는 리스크를 줄이고 재현성을 높이기 위함.

### 66) 생성 설정 하드코딩 경로 제거 (상대경로화)
- 파일: `configs/generation_model.yaml`
- 기존 -> 현재: Colab 절대 경로(`/content/drive/...`)로 고정 -> 프로젝트 기준 상대 경로(`data/...`, `closet_mj/...`)로 변경
- 이유: 로컬/Colab/서버 환경에서 동일 설정 파일을 공유할 수 있도록 환경 종속 하드코딩을 제거하기 위함.

### 67) 설정 로더 경로 자동 해석 + 환경변수 오버라이드 지원
- 파일: `src/generation_pipeline/utils/load.py`
- 기존 -> 현재: YAML 값을 문자열 그대로 사용 -> 프로젝트 루트 기준 상대경로를 절대경로로 자동 해석하고, `CHROMADB_REF_EMBEDDING_DIR` 같은 환경변수로 오버라이드 가능하도록 개선
- 이유: 배포 환경별 경로 차이를 코드 수정 없이 환경변수로 제어하게 만들어 운영 안정성과 이식성을 높이기 위함.

### 68) Colab 전용 부트스트랩 스크립트 제거 + README 실행 가이드 분리
- 파일: `scripts/bootstrap_colab.py`(삭제), `README.md`(신규)
- 기존 -> 현재: Colab용 원커맨드 스크립트로 실행 경로를 안내 -> 스크립트 의존을 제거하고 `README.md`에 `python 실행`/`ipynb(Colab)` 절차를 분리 문서화
- 이유: 사용자가 요청한 운영 방식(스크립트 추가보다 문서 중심)으로 단순화하고, 환경별 실행 방법을 명시적으로 관리하기 위함.

### 69) README에 weights 비포함 정책 명시
- 파일: `README.md`
- 기존 -> 현재: 실행 순서 중심 안내만 존재 -> `weights/`와 데이터는 레포에 포함하지 않고 런타임에서 다운로드해 사용한다는 정책을 상단에 명시
- 이유: 협업 시 대용량 파일 커밋을 방지하고, 환경마다 동일한 준비 절차를 따르도록 기준을 고정하기 위함.
