#!/usr/bin/env python3
"""Train the E1.2 rendering-condition probes (ROADMAP E1.2).

Each condition retrains a fresh classifier head on the same frozen ImageNet
ResNet18 backbone and the same 10,080-clip seed-42 training manifest.  The only
variable is how the 128-band mel matrix was rendered.

Expected inputs, produced by ``scripts/export_e1_2_rendering.py``::

    data/derived/e1_2_rendering/<condition>/{train,dev,eval}/
    manifests/e1_2_rendering/<condition>_{train,dev}.csv

Checkpoints are written under::

    Models/Iteration1/e_1_2/<condition>/seed<seed>/

so multi-seed replication can be added later without overwriting a run's
``config.json`` or ``train_log.csv``.

Usage::

    uv run python scripts/train_e1_2_probes.py
    uv run python scripts/train_e1_2_probes.py --conditions magma_flip
    uv run python scripts/train_e1_2_probes.py --conditions magma_flip --limit 200
"""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from spoof_detection.rendering import (  # noqa: E402
    RENDER_CONDITION_ORDER,
    rendering_payload,
)
from spoof_detection.train_probe import (  # noqa: E402
    ProbeTrainingConfig,
    train_probe,
)

DEFAULT_IMAGE_ROOT = "data/derived/e1_2_rendering"
DEFAULT_MANIFEST_ROOT = "manifests/e1_2_rendering"
DEFAULT_CHECKPOINT_ROOT = "Models/Iteration1/e_1_2"


def _resolve(value: str | Path, repo_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (repo_root / path).resolve()


def checkpoint_name(condition: str, seed: int) -> str:
    """Return the checkpoint filename for one condition and seed."""
    return f"resnet18_frozen_{condition}_seed{seed}.pth"


def seed_dir(
    checkpoint_root: Path,
    condition: str,
    seed: int,
) -> Path:
    """Return the per-condition, per-seed output directory."""
    return Path(checkpoint_root) / condition / f"seed{seed}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=RENDER_CONDITION_ORDER,
        default=list(RENDER_CONDITION_ORDER),
        help="rendering conditions to train (default: all eight)",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42],
        help="training seeds (default: 42)",
    )
    parser.add_argument(
        "--image-root",
        default=DEFAULT_IMAGE_ROOT,
        help=f"exported image root (default: {DEFAULT_IMAGE_ROOT})",
    )
    parser.add_argument(
        "--manifest-root",
        default=DEFAULT_MANIFEST_ROOT,
        help=f"per-condition manifest root (default: {DEFAULT_MANIFEST_ROOT})",
    )
    parser.add_argument(
        "--checkpoint-root",
        default=DEFAULT_CHECKPOINT_ROOT,
        help=f"checkpoint output root (default: {DEFAULT_CHECKPOINT_ROOT})",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="batch size (default: 32)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Adam lr (default: 1e-3)",
    )
    parser.add_argument(
        "--max-epochs", type=int, default=20, help="max epochs (default: 20)"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="early-stopping patience (default: 5)",
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
        help="overwrite existing checkpoints for the selected conditions and seeds",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.limit is not None and args.limit < 2:
        print("error: --limit must be at least 2", file=sys.stderr)
        return 2
    if len(set(args.seeds)) != len(args.seeds):
        print("error: --seeds contains duplicates", file=sys.stderr)
        return 2

    repo_root = REPO_ROOT
    image_root = _resolve(args.image_root, repo_root)
    manifest_root = _resolve(args.manifest_root, repo_root)
    checkpoint_root = _resolve(args.checkpoint_root, repo_root)
    command = shlex.join([sys.executable, *sys.argv])

    results: list[dict[str, object]] = []
    for condition in args.conditions:
        train_manifest = manifest_root / f"{condition}_train.csv"
        dev_manifest = manifest_root / f"{condition}_dev.csv"
        for path in (train_manifest, dev_manifest):
            if not path.is_file():
                print(
                    f"error: missing manifest {path}; "
                    "run scripts/export_e1_2_rendering.py",
                    file=sys.stderr,
                )
                return 2
        for split in ("train", "dev"):
            split_root = image_root / condition / split
            if not split_root.is_dir():
                print(
                    f"error: missing image split {split_root}; "
                    "run scripts/export_e1_2_rendering.py",
                    file=sys.stderr,
                )
                return 2

        payload = rendering_payload(condition)
        for seed in args.seeds:
            output_dir = seed_dir(checkpoint_root, condition, seed)
            checkpoint_out = output_dir / checkpoint_name(condition, seed)
            if args.limit is not None:
                checkpoint_out = output_dir / (
                    checkpoint_out.stem + f"_partial{args.limit}.pth"
                )

            print(
                f"\n[{condition} seed={seed}] training probe -> {checkpoint_out}"
            )
            config = ProbeTrainingConfig(
                frontend=condition,
                train_manifest=train_manifest,
                dev_manifest=dev_manifest,
                checkpoint_out=checkpoint_out,
                seed=seed,
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
                experiment_id="E1.2",
                provenance_payload=payload,
            )
            try:
                result = train_probe(config)
            except FileExistsError as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 2
            results.append(
                {
                    "condition": condition,
                    "seed": seed,
                    "checkpoint": str(result.checkpoint),
                    "best_epoch": result.best_epoch,
                    "epochs_run": result.epochs_run,
                    "best_dev_loss": result.best_dev_loss,
                    "best_dev_eer": result.best_dev_eer,
                    "best_dev_auc": result.best_dev_auc,
                }
            )
            print(
                f"[{condition} seed={seed}] best epoch {result.best_epoch} "
                f"(dev_loss={result.best_dev_loss:.4f}, "
                f"dev_eer={result.best_dev_eer})"
            )

    print("\nsummary")
    for row in results:
        eer = row["best_dev_eer"]
        eer_text = f"{eer:.4f}" if isinstance(eer, float) else "n/a"
        print(
            f"  {row['condition']:<14} seed={row['seed']:<3} "
            f"epoch={row['best_epoch']:<2} dev_loss={row['best_dev_loss']:.4f} "
            f"dev_eer={eer_text}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
