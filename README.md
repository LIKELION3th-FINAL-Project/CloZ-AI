# 프로젝트 개요
사용자가 자연어로 문장을 입력받았을 때, AI는 이를 기반으로 사용자의 옷장 및 외부 상품을 기반으로 착장을 추천해줍니다.

(그림 (자연어 → 이미지 나오는거))

### 문제 정의

- 패션 추천은 단순히 유사 이미지 검색으로 해결되지 않습니다. → 사용자의 자연어와 참조 이미지를 통해 최적의 패션을 추천해줍니다.
- 사용자의 상황, 무드, 장소 등은 구조화되지 않습니다. → 패션 도메인에 맞도록 스타일, 색상, 장소 등을 구조화하여 정보를 활용합니다.
- 기존의 추천 시스템은 단일 유사도 기반으로 동작합니다. → 텍스트 유사도, 스타일 유사도, 색상 유사도 등 멀티 팩터 랭킹 시스템을 사용합니다.
- 기존의 추천 시스템은 불확실성이 고려되지 않습니다. → confidence값을 부여하여 불확실성에 대해 처리합니다.

<br>

---

# 아키텍쳐 및 주요 개발 내용

(그림)

단일 모델이 아닌 모듈형 파이프라인을 구축
- UnderstandModel: 사용자의 자연어에서 의도 추출
- Recommender: 아이템 단위 랭킹
- Planner: 조합을 최적화
- VTON: 시각적 검증

<br>

---
# Python 실행 (로컬/서버)
로컬(Python)과 Colab(ipynb)에서 동일하게 실행할 수 있는 최소 실행 가이드입니다.
- `weights/`, 이미지 데이터(closet, products)는 레포에 포함하지 않습니다.
- 실행 환경에서 가중치를 내려받아 사용해야 합니다.

### Python 실행

1. 의존성 설치
```bash
pip install -r requirements.txt
```

2. VTON 가중치 다운로드
```bash
python scripts/setup_vton_weights.py
export FASHN_VTON_WEIGHTS_DIR="/absolute/path/to/weights" # (선택) 가중치 경로를 직접 지정하는 경우
```

3. 모듈 설치
```bash
pip install -e . # 프로젝트 루트 디렉터리에서 실행
```

5. 실행 예시
```python
from src.pipeline import CloZPipeline

pipeline = CloZPipeline()
pipeline.interactive_session(user_id="your-id")
```



### Colab(ipynb) 실행

1. 의존성 설치
```python
!pip install -r requirements.txt
```

2. VTON 가중치 다운로드
```python
!python scripts/setup_vton_weights.py --weights-dir "/content/drive/MyDrive/~~/weights"
os.environ["FASHN_VTON_WEIGHTS_DIR"] = "/content/drive/MyDrive/~~/weights" # (선택) 가중치 경로를 직접 지정하는 경우
```

3. (Colab) 
```python
import sys
sys.path.insert(0, "/content/drive/MyDrive/~~")

from src.pipeline import CloZPipeline
pipeline = CloZPipeline()
pipeline.interactive_session(user_id="your-id")
```


