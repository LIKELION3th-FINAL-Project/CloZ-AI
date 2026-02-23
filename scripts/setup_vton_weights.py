#!/usr/bin/env python3
"""
Download FASHN VTON weights without cloning external repos.

Usage:
    python scripts/setup_vton_weights.py
    python scripts/setup_vton_weights.py --weights-dir /path/to/weights
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from huggingface_hub import hf_hub_download


def _download_tryon_model(weights_dir: Path) -> None:
    hf_hub_download(
        repo_id="fashn-ai/fashn-vton-1.5",
        filename="model.safetensors",
        local_dir=str(weights_dir),
    )


def _download_dwpose_models(weights_dir: Path) -> None:
    dwpose_dir = weights_dir / "dwpose"
    dwpose_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("yolox_l.onnx", "dw-ll_ucoco_384.onnx"):
        hf_hub_download(
            repo_id="fashn-ai/DWPose",
            filename=filename,
            local_dir=str(dwpose_dir),
        )


def _warmup_human_parser(skip: bool) -> None:
    if skip:
        return
    from fashn_human_parser import FashnHumanParser

    _ = FashnHumanParser(device="cpu")


def _verify_files(weights_dir: Path) -> Iterable[Path]:
    required = (
        weights_dir / "model.safetensors",
        weights_dir / "dwpose" / "yolox_l.onnx",
        weights_dir / "dwpose" / "dw-ll_ucoco_384.onnx",
    )
    missing = [path for path in required if not path.exists()]
    return missing


def main() -> int:
    parser = argparse.ArgumentParser(description="Setup FASHN VTON weights")
    parser.add_argument(
        "--weights-dir",
        type=str,
        default=None,
        help="Target weights directory. Default: <project_root>/weights",
    )
    parser.add_argument(
        "--skip-human-parser",
        action="store_true",
        help="Skip fashn-human-parser warmup download",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    weights_dir = Path(args.weights_dir).expanduser() if args.weights_dir else (project_root / "weights")
    weights_dir.mkdir(parents=True, exist_ok=True)

    print(f"[setup_vton_weights] target={weights_dir}")
    _download_tryon_model(weights_dir)
    _download_dwpose_models(weights_dir)
    _warmup_human_parser(args.skip_human_parser)

    missing = list(_verify_files(weights_dir))
    if missing:
        print("[setup_vton_weights] missing files:")
        for path in missing:
            print(f"  - {path}")
        return 1

    print("[setup_vton_weights] done")
    print(f"[setup_vton_weights] export FASHN_VTON_WEIGHTS_DIR={weights_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
