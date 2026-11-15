#!/usr/bin/env python3
"""Evaluate the six E1.1 probes with the frozen E0.3 harness (ROADMAP E1.1).

Each probe is scored on full dev and eval through
:func:`spoof_detection.evaluate.main`, reading the front-end-specific PNG root
and tagging ``metrics.json`` with ``--frontend`` so provenance records the true
representation.  The metric contract (split, attack subset, threshold, sample
count) comes from the harness unchanged.

Usage::

    uv run python scripts/evaluate_e1_1.py
    uv run python scripts/evaluate_e1_1.py --frontends mel128 --splits dev
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from spoof_detection import evaluate  # noqa: E402
from spoof_detection.frontends import FRONTEND_ORDER  # noqa: E402

DEFAULT_CHECKPOINT_ROOT = "Models/Iteration1/e_1_1"
DEFAULT_IMAGE_ROOT = "data/derived/e1_1_frontend"
DEFAULT_OUTPUT_ROOT = "reports/phase1/e1_1_frontend/runs"


def _resolve(value: str | Path, repo_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (repo_root / path).resolve()


def _checkpoint_name(frontend: str) -> str:
    return f"resnet18_frozen_{frontend}.pth"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--frontends",
        nargs="+",
        choices=FRONTEND_ORDER,
        default=list(FRONTEND_ORDER),
        help="front-ends to evaluate (default: all six)",
    )
    parser.add_argument(
        "--checkpoint-root",
        default=DEFAULT_CHECKPOINT_ROOT,
        help=f"probe checkpoint root (default: {DEFAULT_CHECKPOINT_ROOT})",
    )
    parser.add_argument(
        "--image-root",
        default=DEFAULT_IMAGE_ROOT,
        help=f"front-end image root (default: {DEFAULT_IMAGE_ROOT})",
    )
    parser.add_argument(
        "--output-root",
        default=DEFAULT_OUTPUT_ROOT,
        help=f"evaluation output root (default: {DEFAULT_OUTPUT_ROOT})",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("dev", "eval"),
        default=["dev", "eval"],
        help="ASVspoof splits to score (default: dev eval)",
    )
    parser.add_argument(
        "--source",
        choices=("image", "audio"),
        default="image",
        help="read exported PNGs or recompute from audio (default: image)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="inference batch size (default: 64)"
    )
    parser.add_argument(
        "--num-workers", type=int, default=4, help="DataLoader workers (default: 4)"
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="inference device (default: auto)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="smoke only: deterministic class-stratified N clips per split",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = REPO_ROOT
    checkpoint_root = _resolve(args.checkpoint_root, repo_root)
    image_root = _resolve(args.image_root, repo_root)
    output_root = _resolve(args.output_root, repo_root)

    missing = []
    for frontend in args.frontends:
        checkpoint = checkpoint_root / frontend / _checkpoint_name(frontend)
        if not checkpoint.is_file():
            missing.append(str(checkpoint))
    if missing:
        for path in missing:
            print(f"error: checkpoint not found: {path}", file=sys.stderr)
        return 2

    for frontend in args.frontends:
        checkpoint = checkpoint_root / frontend / _checkpoint_name(frontend)
        frontend_output = output_root / frontend
        argv = [
            "--checkpoint",
            str(checkpoint),
            "--asvspoof",
            *args.splits,
            "--source",
            args.source,
            "--data-root",
            str(repo_root / "data" / "LA"),
            "--image-root",
            str(image_root / frontend),
            "--output-root",
            str(frontend_output),
            "--frontend",
            frontend,
            "--experiment-id",
            "E1.1",
            "--title",
            f"E1.1 front-end probe: {frontend}",
            "--batch-size",
            str(args.batch_size),
            "--num-workers",
            str(args.num_workers),
            "--device",
            args.device,
        ]
        if args.limit is not None:
            argv.extend(["--limit", str(args.limit)])

        print(f"\n[{frontend}] evaluating {checkpoint.name}")
        code = evaluate.main(argv)
        if code != 0:
            print(f"error: evaluation failed for {frontend}", file=sys.stderr)
            return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
