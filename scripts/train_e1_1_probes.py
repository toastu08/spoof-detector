#!/usr/bin/env python3
"""Train the six E1.1 head-only probes (ROADMAP E1.1).

For every front-end this trains a fresh classifier head on a frozen ImageNet
ResNet18 backbone, using the baseline-faithful recipe recorded in
``2_model.ipynb`` (unweighted BCE, Adam 1e-3, batch 32, <=20 epochs, patience 5
on full-dev loss).  The only deliberate differences are a fixed seed and a
deterministic head re-initialisation shared by all front-ends.

Expected inputs, produced by ``scripts/export_e1_1_frontends.py``::

    data/derived/e1_1_frontend/<frontend>/{train,dev,eval}/
    manifests/e1_1_frontend/<frontend>_{train,dev,eval}.csv

Checkpoints are written under ``Models/Iteration1/e_1_1/<frontend>/``.

Usage::

    uv run python scripts/train_e1_1_probes.py
    uv run python scripts/train_e1_1_probes.py --frontends mel128
    uv run python scripts/train_e1_1_probes.py --frontends mel128 --limit 200
"""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from spoof_detection.frontends import FRONTEND_ORDER  # noqa: E402
from spoof_detection.train_probe import (  # noqa: E402
    ProbeTrainingConfig,
    train_probe,
)

DEFAULT_IMAGE_ROOT = "data/derived/e1_1_frontend"
DEFAULT_MANIFEST_ROOT = "manifests/e1_1_frontend"
DEFAULT_CHECKPOINT_ROOT = "Models/Iteration1/e_1_1"


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
        help="front-ends to train (default: all six)",
    )
    parser.add_argument(
        "--image-root",
        default=DEFAULT_IMAGE_ROOT,
        help=f"exported image root (default: {DEFAULT_IMAGE_ROOT})",
    )
    parser.add_argument(
        "--manifest-root",
        default=DEFAULT_MANIFEST_ROOT,
        help=f"per-frontend manifest root (default: {DEFAULT_MANIFEST_ROOT})",
    )
    parser.add_argument(
        "--checkpoint-root",
        default=DEFAULT_CHECKPOINT_ROOT,
        help=f"checkpoint output root (default: {DEFAULT_CHECKPOINT_ROOT})",
    )
    parser.add_argument("--seed", type=int, default=42, help="seed (default: 42)")
    parser.add_argument(
        "--batch-size", type=int, default=32, help="batch size (default: 32)"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=1e-3, help="Adam lr (default: 1e-3)"
    )
    parser.add_argument(
        "--max-epochs", type=int, default=20, help="max epochs (default: 20)"
    )
    parser.add_argument(
        "--patience", type=int, default=5, help="early-stopping patience (default: 5)"
    )
    parser.add_argument(
        "--num-workers", type=int, default=4, help="DataLoader workers (default: 4)"
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="training device (default: auto)",
    )
    parser.add_argument(
        "--backbone-weights",
        choices=("torchvision", "checkpoint"),
        default="torchvision",
        help=(
            "torchvision reproduces the notebook initialisation; checkpoint "
            "loads the adapted backbone from the E0.3 checkpoint (default: torchvision)"
        ),
    )
    parser.add_argument(
        "--freeze-bn",
        action="store_true",
        help="keep batch-norm in eval mode instead of the baseline-faithful train mode",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="smoke only: class-stratified N rows per split (writes *_partial checkpoints)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite existing checkpoints for the selected front-ends",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.limit is not None and args.limit < 2:
        print("error: --limit must be at least 2", file=sys.stderr)
        return 2

    repo_root = REPO_ROOT
    image_root = _resolve(args.image_root, repo_root)
    manifest_root = _resolve(args.manifest_root, repo_root)
    checkpoint_root = _resolve(args.checkpoint_root, repo_root)
    command = shlex.join([sys.executable, *sys.argv])

    results: list[dict[str, object]] = []
    for frontend in args.frontends:
        train_manifest = manifest_root / f"{frontend}_train.csv"
        dev_manifest = manifest_root / f"{frontend}_dev.csv"
        for path in (train_manifest, dev_manifest):
            if not path.is_file():
                print(
                    f"error: missing manifest {path}; run scripts/export_e1_1_frontends.py",
                    file=sys.stderr,
                )
                return 2
        for split in ("train", "dev"):
            split_root = image_root / frontend / split
            if not split_root.is_dir():
                print(
                    f"error: missing image split {split_root}; "
                    "run scripts/export_e1_1_frontends.py",
                    file=sys.stderr,
                )
                return 2

        output_dir = checkpoint_root / frontend
        checkpoint_out = output_dir / _checkpoint_name(frontend)
        if args.limit is not None:
            checkpoint_out = output_dir / (
                checkpoint_out.stem + f"_partial{args.limit}.pth"
            )

        print(f"\n[{frontend}] training probe -> {checkpoint_out}")
        config = ProbeTrainingConfig(
            frontend=frontend,
            train_manifest=train_manifest,
            dev_manifest=dev_manifest,
            checkpoint_out=checkpoint_out,
            seed=args.seed,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_epochs=args.max_epochs,
            patience=args.patience,
            num_workers=args.num_workers,
            device=args.device,
            freeze_bn=args.freeze_bn,
            backbone_weights=args.backbone_weights,
            limit=args.limit,
            force=args.force,
            command=command,
        )
        try:
            result = train_probe(config)
        except FileExistsError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        results.append(
            {
                "frontend": frontend,
                "checkpoint": str(result.checkpoint),
                "best_epoch": result.best_epoch,
                "epochs_run": result.epochs_run,
                "best_dev_loss": result.best_dev_loss,
                "best_dev_eer": result.best_dev_eer,
                "best_dev_auc": result.best_dev_auc,
            }
        )
        print(
            f"[{frontend}] best epoch {result.best_epoch} "
            f"(dev_loss={result.best_dev_loss:.4f}, "
            f"dev_eer={result.best_dev_eer})"
        )

    print("\nsummary")
    for row in results:
        eer = row["best_dev_eer"]
        eer_text = f"{eer:.4f}" if isinstance(eer, float) else "n/a"
        print(
            f"  {row['frontend']:<12} epoch={row['best_epoch']:<2} "
            f"dev_loss={row['best_dev_loss']:.4f} dev_eer={eer_text}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
