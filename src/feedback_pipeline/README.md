# CloZ-AI Recommendation-Feedback Pipeline

이 패키지는 사용자의 취향을 학습하고 실시간 피드백을 통해 최적의 코디를 제공하는 지능형 피드백 파이프라인입니다.

## 주요 기능
- **ManagerAgent**: 실시간 YES/NO 피드백 처리 및 의사결정 (REGENERATE vs BUYING)
- **AnalystAgent**: 세션 종료 후 사용자 취향 분석 및 프로필 업데이트
- **ContextAnalyzer**: LLM 기반 지능형 피드백 의도 분석
- **QueryBuilder**: Solar Pro 3를 사용한 정교한 검색 쿼리 결합
- **Embedding Checkers**: FashionCLIP 기반 옷장 매칭 및 상품 추천

## 실행 방법

### 1. 환경 설정
`.env.example` 파일을 복사하여 `.env` 파일을 생성하고 필수 API Key를 설정합니다.
- `OPENAI_API_KEY`: Manager/Analyst Agent 분석용
- `UPSTAGE_API_KEY`: QueryBuilder용 (Solar Pro 3)
- `GOOGLE_API_KEY`: Embedding Generator용 (선택)

### 2. 패키지 설치
```bash
pip install -r src/feedback_pipeline/requirements.txt
```

### 3. 상호작용형 베타 테스트 실행
실제 사용자처럼 쿼리를 입력하고 피드백을 주고받으며 테스트하려면 아래 명령어를 실행하세요.
```bash
# 프로젝트 루트 디렉토리에서 실행
python -m src.feedback_pipeline.main
```
- **기능**: 오리지널 쿼리 입력, YES/NO 피드백, 마음에 안 드는 부위(상의/하의 등) 복수 선택, 상세 이유 입력 및 AI 의사결정 확인.

## 디렉토리 구조
- `agents/`: Manager 및 Analyst 에이전트
- `checkers/`: 옷장 및 상품 추천 로직
- `models/`: 데이터 모델 정의
- `utils/`: 쿼리 빌더 및 컨텍스트 분석기
- `interfaces/`: 확장성을 위한 인터페이스 정의
- `storage/`: 세션 및 프로필 저장소

## 연동 가이드
타 모듈에서 이 파이프라인을 호출할 때는 `src.feedback_pipeline.agents.ManagerAgent`를 초기화하여 `process_feedback` 메서드를 사용하십시오.
데이터 규격은 `src.feedback_pipeline.models`에 정의된 `FeedbackInput` 및 `OutfitSet`을 따릅니다.
