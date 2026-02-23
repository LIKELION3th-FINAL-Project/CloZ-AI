# whats-in-my-closet

로컬(Python)과 Colab(ipynb)에서 동일하게 실행할 수 있는 최소 실행 가이드입니다.

## 중요

- `weights/`, 이미지 데이터(closet, products)는 레포에 포함하지 않습니다.
- 실행 환경에서 가중치를 내려받아 사용해야 합니다.

## Python 실행 (로컬/서버)

1. 의존성 설치

```bash
pip install -r requirements.txt
```

2. VTON 가중치 다운로드

```bash
python scripts/setup_vton_weights.py
```

3. (선택) 가중치 경로를 직접 지정하는 경우

```bash
export FASHN_VTON_WEIGHTS_DIR="/absolute/path/to/weights"
```

4. 실행 예시

```python
from src.pipeline import CloZPipeline

pipeline = CloZPipeline()
pipeline.interactive_session(user_id="your-id")
```

## ipynb / Colab 실행

### A. 런타임 준비

```python
!pip install -r requirements.txt
```

### B. VTON 가중치 다운로드

```python
!python scripts/setup_vton_weights.py --weights-dir "/content/drive/MyDrive/~~/weights"
```

### C. 가중치 경로 지정 (권장)

```python
import os
os.environ["FASHN_VTON_WEIGHTS_DIR"] = "/content/drive/MyDrive/~~/weights"
```

### D. 프로젝트 경로 등록 후 실행

```python
import sys
sys.path.insert(0, "/content/drive/MyDrive/~~")

from src.pipeline import CloZPipeline
pipeline = CloZPipeline()
pipeline.interactive_session(user_id="your-id")
```

## 설정/경로 정책

- `configs/generation_model.yaml`은 상대경로 기준입니다.
- 실제 절대경로는 `src/generation_pipeline/utils/load.py`에서 자동 해석됩니다.
- 필요 시 환경변수로 오버라이드 가능합니다.
  - 예: `USER_CLOTHES_DIR`, `USER_BODY_IMAGE`, `CHROMADB_REF_EMBEDDING_DIR`, `CHROMADB_USER_WAR_EMBEDDING_DIR`
