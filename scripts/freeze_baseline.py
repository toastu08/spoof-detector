#!/usr/bin/env python3
"""Audit and freeze the E0.3 two-checkpoint baseline.

Run the canonical evaluation first, then a second full run into a separate root,
then execute this script.  It verifies:

1. the training manifest hash and class balance,
2. strict checkpoint loading and parameter counts,
3. the dev/eval metric contract,
4. run-to-run EER reproducibility within a pre-registered tolerance,
5. that the classifier head is trained rather than randomly initialised.

Usage::

    uv run python scripts/freeze_baseline.py \
        --baseline-root reports/baseline \
        --rerun-root /tmp/opencode/e0.3-rerun \
        --manifest manifests/train_seed42.csv \
        --output reports/baseline/freeze_report.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from spoof_detection.baseline import (  # noqa: E402
    DEFAULT_BASELINE_ROOT,
    DEFAULT_CHECKPOINTS,
    DEFAULT_MANIFEST,
    DEFAULT_RERUN_ROOT,
    EXPERIMENT_ID,
    HEAD_CONTROL_LIMIT,
    HEAD_CONTROL_MAX_RANDOM_AUC,
    HEAD_CONTROL_MIN_GAP,
    HEAD_CONTROL_MIN_TRAINED_AUC,
    MAX_EER_DELTA,
    build_freeze_report,
    write_freeze_report,
)


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify and freeze the E0.3 two-checkpoint baseline.",
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "checkpoint to audit; repeat for multiple checkpoints "
            "(default: the two first-iteration checkpoints)"
        ),
    )
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=Path(DEFAULT_BASELINE_ROOT),
        help="canonical baseline output root (default: reports/baseline)",
    )
    parser.add_argument(
        "--rerun-root",
        type=Path,
        default=Path(DEFAULT_RERUN_ROOT),
        help="second-run output root (default: /tmp/opencode/e0.3-rerun)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(DEFAULT_MANIFEST),
        help="training manifest (default: manifests/train_seed42.csv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(DEFAULT_BASELINE_ROOT) / "freeze_report.json",
        help="audit JSON output path",
    )
    parser.add_argument(
        "--max-eer-delta",
        type=float,
        default=MAX_EER_DELTA,
        help=(
            "maximum absolute EER difference between runs as a fraction "
            "(default: 0.01, i.e. 1.0 percentage point)"
        ),
    )
    parser.add_argument(
        "--min-trained-auc",
        type=float,
        default=HEAD_CONTROL_MIN_TRAINED_AUC,
        help="minimum trained-head AUC on the control subset",
    )
    parser.add_argument(
        "--max-random-auc",
        type=float,
        default=HEAD_CONTROL_MAX_RANDOM_AUC,
        help="maximum reinitialized-head AUC on the control subset",
    )
    parser.add_argument(
        "--min-head-gap",
        type=float,
        default=HEAD_CONTROL_MIN_GAP,
        help="minimum trained-minus-random AUC gap",
    )
    parser.add_argument(
        "--experiment-id",
        default=EXPERIMENT_ID,
        help="expected experiment identifier in metrics.json",
    )
    parser.add_argument(
        "--head-control-limit",
        type=int,
        default=HEAD_CONTROL_LIMIT,
        help="number of dev clips used for the head control",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="device for the head control",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="head-control inference batch size",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="head-control DataLoader workers",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoints = (
        [_resolve(path) for path in args.checkpoint]
        if args.checkpoint
        else list(DEFAULT_CHECKPOINTS)
    )
    report = build_freeze_report(
        checkpoints=checkpoints,
        baseline_root=_resolve(args.baseline_root),
        rerun_root=_resolve(args.rerun_root),
        manifest_path=_resolve(args.manifest),
        max_eer_delta=args.max_eer_delta,
        min_trained_auc=args.min_trained_auc,
        max_random_auc=args.max_random_auc,
        min_gap=args.min_head_gap,
        experiment_id=args.experiment_id,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        head_control_limit=args.head_control_limit,
    )
    output_path = write_freeze_report(_resolve(args.output), report)

    print(f"Wrote {output_path}")
    manifest = report["training_manifest"]
    print(
        f"manifest: {manifest['rows']} rows "
        f"({manifest['counts']['bonafide']} bonafide / "
        f"{manifest['counts']['spoof']} spoof) "
        f"{'PASS' if manifest['passed'] else 'FAIL'}"
    )
    for checkpoint in report["checkpoints"]:
        print(f"\n{checkpoint['stem']}:")
        audit = checkpoint["audit"]
        print(
            f"  checkpoint: {audit['total_parameters']} params "
            f"({audit['head_parameters']} head), "
            f"missing={len(audit['missing_keys'])}, "
            f"unexpected={len(audit['unexpected_keys'])}"
        )
        print(
            f"  head: SHA={audit['head_state_sha256'][:12]}…, "
            f"bias={audit['head_last_bias']:.6f}, "
            f"BN steps={audit['batch_norm_steps']}"
        )
        comparison = checkpoint["comparison"]
        if comparison is None:
            print("  rerun comparison: MISSING")
        else:
            for split, values in comparison["splits"].items():
                eer_delta = values["eer_delta"]
                auc_delta = values["auc_delta"]
                print(
                    f"  {split}: EER {values['eer_reference']:.6f} vs "
                    f"{values['eer_rerun']:.6f} "
                    f"(Δ={eer_delta:.6f}), "
                    f"AUC Δ={auc_delta:.6f} "
                    f"{'PASS' if values['passed'] else 'FAIL'}"
                )
        head = checkpoint["head_control"]
        print(
            f"  head control: trained AUC={head['trained_auc']:.4f}, "
            f"random AUC={head['random_head_auc']:.4f}, "
            f"gap={head['auc_gap']:.4f} "
            f"{'PASS' if head['passed'] else 'FAIL'}"
        )
        if not checkpoint["passed"]:
            for failure in checkpoint["failures"]:
                print(f"  FAIL: {failure}")

    print(
        f"\nbaseline freeze: {'PASS' if report['passed'] else 'FAIL'}"
    )
    if not report["passed"]:
        for failure in report["failures"]:
            print(f"  {failure}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
