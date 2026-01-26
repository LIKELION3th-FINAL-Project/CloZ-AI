SHELL := /bin/bash
PY := python
RUFF := ruff
PYTEST := pytest

# ---- project conventions ----
APP := app
SRC := src
CFG_DIR := configs
OUT_DIR := outputs

# 기본 config 경로(필요 시 CLI로 override)
TRAIN_CFG ?= $(CFG_DIR)/train.yaml
INFER_CFG ?= $(CFG_DIR)/infer.yaml

# ML 실험 공통 변수
SEED ?= 42
DEVICE ?= cuda
RUN_NAME ?= dev

.PHONY: help check lint format format-check test \
        train infer eval \
        smoke data-dir outputs-dir clean deep-clean

help:
	@echo "== Quality =="
	@echo "  make check            - lint + format-check + test"
	@echo "  make format           - apply formatter"
	@echo "== ML =="
	@echo "  make train            - run training"
	@echo "  make infer            - run inference"
	@echo "  make eval             - run evaluation"
	@echo "== Ops =="
	@echo "  make clean            - remove caches"
	@echo "  make deep-clean       - remove caches + outputs"

# ---- quality ----
lint:
	$(RUFF) check .

format:
	$(RUFF) format .

format-check:
	$(RUFF) format --check .

test:
	$(PYTEST)

check: lint format-check test

# ---- dirs ----
outputs-dir:
	@mkdir -p $(OUT_DIR)

data-dir:
	@mkdir -p data

# ---- ML entrypoints (scripts/ 기준) ----
train: outputs-dir
	$(PY) scripts/train.py --config $(TRAIN_CFG) --seed $(SEED) --device $(DEVICE) --run-name $(RUN_NAME)

infer: outputs-dir
	$(PY) scripts/infer.py --config $(INFER_CFG) --device $(DEVICE) --run-name $(RUN_NAME)

eval: outputs-dir
	$(PY) scripts/eval.py --config $(EVAL_CFG) --run-name $(RUN_NAME)

# ---- sanity ----
smoke:
	$(PY) -c "import $(APP); print('import ok')"

# ---- clean ----
clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__ **/*.pyc

deep-clean: clean
	rm -rf $(OUT_DIR)
