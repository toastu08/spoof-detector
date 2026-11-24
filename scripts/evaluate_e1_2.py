#!/usr/bin/env python3
"""Evaluate the E1.2 rendering-condition probes with the frozen harness.

Each condition/seed is scored on full dev and eval through
:func:`spoof_detection.evaluate.main`, reading the condition-specific PNG root
and tagging ``metrics.json`` with ``--frontend`` so provenance records the true
rendering condition.  The metric contract (split, attack subset, threshold,
sample count) comes from the harness unchanged.

Usage::

    uv run python scripts/evaluate_e1_2.py
    uv run python scripts/evaluate_e1_2.py --conditions magma_flip --splits dev
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from spoof_detection import evaluate  # noqa: E402
from spoof_detection.rendering import RENDER_CONDITION_ORDER  # noqa: E402

DEFAULT_CHECKPOINT_ROOT = "Models/Iteration1/e_1_2"
DEFAULT_IMAGE_ROOT = "data/derived/e1_2_rendering"
DEFAULT_OUTPUT_ROOT = "reports/phase1/e1_2_rendering/runs"


def _resolve(value: str | Path, repo_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (repo_root / path).resolve()


def checkpoint_name(condition: str, seed: int) -> str:
    """Return the checkpoint filename for one condition and seed."""
    return f"resnet18_frozen_{condition}_seed{seed}.pth"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=RENDER_CONDITION_ORDER,
        default=list(RENDER_CONDITION_ORDER),
        help="rendering conditions to evaluate (default: all eight)",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42],
        help="training seeds to evaluate (default: 42)",
    )
    parser.add_argument(
        "--checkpoint-root",
        default=DEFAULT_CHECKPOINT_ROOT,
        help=f"probe checkpoint root (default: {DEFAULT_CHECKPOINT_ROOT})",
    )
    parser.add_argument(
        "--image-root",
        default=DEFAULT_IMAGE_ROOT,
        help=f"condition image root (default: {DEFAULT_IMAGE_ROOT})",
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
        "--batch-size",
        type=int,
        default=64,
        help="inference batch size (default: 64)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader workers (default: 4)",
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

    missing: list[str] = []
    for condition in args.conditions:
        for seed in args.seeds:
            checkpoint = (
                checkpoint_root
                / condition
                / f"seed{seed}"
                / checkpoint_name(condition, seed)
            )
            if not checkpoint.is_file():
                missing.append(str(checkpoint))
    if missing:
        for path in missing:
            print(f"error: checkpoint not found: {path}", file=sys.stderr)
        return 2

    for condition in args.conditions:
        for seed in args.seeds:
            checkpoint = (
                checkpoint_root
                / condition
                / f"seed{seed}"
                / checkpoint_name(condition, seed)
            )
            seed_output = output_root / condition / f"seed{seed}"
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
                str(image_root / condition),
                "--output-root",
                str(seed_output),
                "--frontend",
                condition,
                "--experiment-id",
                "E1.2",
                "--title",
                f"E1.2 rendering condition: {condition} (seed {seed})",
                "--batch-size",
                str(args.batch_size),
                "--num-workers",
                str(args.num_workers),
                "--device",
                args.device,
            ]
            if args.limit is not None:
                argv.extend(["--limit", str(args.limit)])

            print(
                f"\n[{condition} seed={seed}] evaluating {checkpoint.name}"
            )
            code = evaluate.main(argv)
            if code != 0:
                print(
                    f"error: evaluation failed for {condition} seed {seed}",
                    file=sys.stderr,
                )
                return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
