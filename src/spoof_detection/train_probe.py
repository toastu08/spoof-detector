"""Seeded head-only probe training for the E1.1 and E1.2 ablations.

The classifier head is retrained for every image condition while the ResNet18
backbone is frozen, matching the first-iteration recipe recorded in
``2_model.ipynb``:

* ``requires_grad=False`` on the backbone, but the model is in ``train()`` mode,
  so batch-norm running statistics still update;
* unweighted ``BCEWithLogitsLoss``, Adam ``lr=1e-3``, batch size 32;
* at most 20 epochs, early stopping with patience 5 on full-dev loss.

The differences from the notebook are deliberate and documented in the E1.1
report: a fixed seed (the notebook recorded none), a deterministic head
re-initialisation shared by all conditions, and a recorded command/config.

All reusable logic lives here; ``scripts/train_e1_1_probes.py`` and
``scripts/train_e1_2_probes.py`` are thin CLIs.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18

from . import preprocess
from .evaluate import environment_info, git_info, resolve_device
from .frontends import frontend_payload
from .metrics import compute_auc, compute_eer
from .model import AudioSpoofDetector, load_model_from_checkpoint, sha256_file

EXPERIMENT_ID = "E1.1"
DEFAULT_REFERENCE_CHECKPOINT = (
    Path(__file__).resolve().parents[2]
    / "Models"
    / "Iteration1"
    / "best_audio_spoof_model_unweighted.pth"
)

BACKBONE_AFFINE_EXCLUDED_SUFFIXES = (
    ".running_mean",
    ".running_var",
    ".num_batches_tracked",
)


@dataclass(frozen=True)
class ProbeTrainingConfig:
    """Everything needed to train and record one probe."""

    frontend: str
    train_manifest: Path
    dev_manifest: Path
    checkpoint_out: Path
    reference_checkpoint: Path = DEFAULT_REFERENCE_CHECKPOINT
    seed: int = 42
    batch_size: int = 32
    learning_rate: float = 1e-3
    max_epochs: int = 20
    patience: int = 5
    num_workers: int = 4
    device: str = "auto"
    freeze_bn: bool = False
    backbone_weights: str = "torchvision"
    verify_backbone: bool = True
    limit: int | None = None
    force: bool = False
    command: str = ""
    experiment_id: str = EXPERIMENT_ID
    provenance_payload: dict[str, object] | None = None


@dataclass(frozen=True)
class ProbeTrainingResult:
    """Summary returned after a successful run."""

    checkpoint: Path
    checkpoint_sha256: str
    best_epoch: int
    epochs_run: int
    best_dev_loss: float
    best_dev_eer: float | None
    best_dev_auc: float | None
    log_path: Path
    config_path: Path


def _jsonable(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        numeric = float(value)
        return None if math.isnan(numeric) or math.isinf(numeric) else numeric
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and torch, and request deterministic cuDNN."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _worker_init(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def _freeze_bn(backbone: nn.Module) -> None:
    for module in backbone.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()


def build_probe_model(
    seed: int,
    reference_checkpoint: Path = DEFAULT_REFERENCE_CHECKPOINT,
    device: str | torch.device = "cpu",
    freeze_bn: bool = False,
    backbone_weights: str = "torchvision",
) -> AudioSpoofDetector:
    """Build a frozen-backbone model with a deterministic fresh head.

    ``backbone_weights="torchvision"`` reproduces the notebook, which started
    from ``resnet18(weights=ResNet18_Weights.DEFAULT)``.  The checkpoint option
    exists for tests and offline runs; it loads the backbone (including adapted
    batch-norm statistics) from ``reference_checkpoint``.

    The head is always re-initialised after the backbone is loaded, using
    ``torch.manual_seed(seed)`` so every front-end starts from the same head.
    """
    model = AudioSpoofDetector()

    if backbone_weights == "torchvision":
        reference = resnet18(weights=ResNet18_Weights.DEFAULT)
        reference_state = {
            key: value
            for key, value in reference.state_dict().items()
            if not key.startswith("fc.")
        }
        missing, unexpected = model.backbone.load_state_dict(
            reference_state, strict=False
        )
        expected_missing = {
            "fc.0.weight",
            "fc.0.bias",
            "fc.3.weight",
            "fc.3.bias",
        }
        if set(missing) != expected_missing or unexpected:
            raise RuntimeError(
                "unexpected backbone initialisation diff: "
                f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
            )
        del reference
    elif backbone_weights == "checkpoint":
        reference_model, _report = load_model_from_checkpoint(
            reference_checkpoint, device="cpu"
        )
        model.backbone.load_state_dict(reference_model.backbone.state_dict())
        del reference_model
    else:
        raise ValueError(
            "backbone_weights must be 'torchvision' or 'checkpoint', "
            f"got {backbone_weights!r}"
        )

    # Deterministic head re-initialisation, identical for every front-end.
    torch.manual_seed(seed)
    for layer in (model.backbone.fc[0], model.backbone.fc[3]):
        layer.reset_parameters()  # type: ignore[attr-defined]

    for parameter in model.backbone.parameters():
        parameter.requires_grad = False
    for parameter in model.backbone.fc.parameters():
        parameter.requires_grad = True

    model.to(device)
    model.train()
    if freeze_bn:
        _freeze_bn(model.backbone)
    return model


def verify_backbone_parity(
    model: AudioSpoofDetector,
    reference_checkpoint: Path = DEFAULT_REFERENCE_CHECKPOINT,
) -> dict[str, object]:
    """Assert the frozen conv/BN-affine weights equal the E0.3 checkpoint."""
    reference_model, report = load_model_from_checkpoint(
        reference_checkpoint, device="cpu"
    )
    reference_state = reference_model.backbone.state_dict()
    model_state = model.backbone.state_dict()

    keys = [
        key
        for key in reference_state
        if not key.startswith("fc.")
        and not key.endswith(BACKBONE_AFFINE_EXCLUDED_SUFFIXES)
    ]
    mismatched = [
        key
        for key in keys
        if not torch.equal(
            model_state[key].detach().cpu(), reference_state[key].detach().cpu()
        )
    ]
    if mismatched:
        raise RuntimeError(
            "backbone does not match the E0.3 checkpoint for "
            f"{len(mismatched)} tensors, e.g. {mismatched[:5]}"
        )
    return {
        "checked_keys": len(keys),
        "mismatched_keys": mismatched,
        "reference_checkpoint": str(reference_checkpoint),
        "reference_sha256": report.sha256,
        "note": (
            "Conv/BN affine weights and biases only; batch-norm running "
            "statistics are excluded because the baseline updated them."
        ),
    }


def _resolve(value: str | Path, base_dir: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base_dir / path).resolve()


def _limit_frame(frame: pd.DataFrame, limit: int | None) -> pd.DataFrame:
    if limit is None or limit >= len(frame):
        return frame.reset_index(drop=True)
    per_class = max(1, int(limit) // 2)
    parts = [
        frame[frame["label"] == label].head(per_class) for label in (0, 1)
    ]
    selected = pd.concat(parts)
    if len(selected) < limit:
        remaining = frame.drop(selected.index).head(int(limit) - len(selected))
        selected = pd.concat([selected, remaining])
    return selected.sort_index().reset_index(drop=True)


def load_probe_frame(
    manifest_path: Path,
    base_dir: Path,
    limit: int | None = None,
) -> pd.DataFrame:
    """Load a probe manifest with ``image_path`` and binary ``label`` columns."""
    manifest_path = Path(manifest_path)
    frame = pd.read_csv(manifest_path)
    path_column = next(
        (
            column
            for column in ("image_path", "img_path")
            if column in frame.columns
        ),
        None,
    )
    if path_column is None:
        raise ValueError(
            f"{manifest_path} has no image_path or img_path column"
        )
    if "label" not in frame.columns:
        raise ValueError(f"{manifest_path} is missing the label column")

    frame = frame.copy()
    frame["image_path"] = [
        str(_resolve(value, base_dir)) for value in frame[path_column]
    ]
    missing = [
        path for path in frame["image_path"] if not Path(path).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} images are missing, e.g. {missing[:5]}"
        )
    frame["label"] = frame["label"].astype(int)
    labels = set(frame["label"].unique())
    if not labels.issubset({0, 1}):
        raise ValueError(f"{manifest_path} labels must be 0 or 1, got {labels}")
    return _limit_frame(frame, limit)


class ProbeImageDataset(Dataset):
    """Minimal image dataset that returns ``(tensor, float_label)``."""

    def __init__(self, frame: pd.DataFrame, transform: object) -> None:
        self.frame = frame.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.frame.iloc[index]
        with Image.open(row["image_path"]) as image:
            tensor = self.transform(image.convert("RGB"))  # type: ignore[operator]
        label = torch.tensor(float(row["label"]), dtype=torch.float32)
        return tensor, label


def _evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    total = 0
    scores: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    with torch.inference_mode():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(images).reshape(-1)
            loss = criterion(logits, targets)
            total_loss += float(loss.item()) * images.size(0)
            total += images.size(0)
            scores.append(torch.sigmoid(logits).float().cpu().numpy())
            labels.append(targets.cpu().numpy())
    if total == 0:
        raise RuntimeError("empty evaluation loader")
    return (
        total_loss / total,
        np.concatenate(scores),
        np.concatenate(labels).astype(int),
    )


def _eer_auc(
    scores: np.ndarray,
    labels: np.ndarray,
) -> tuple[float | None, float | None]:
    spoof = scores[labels == 1]
    bonafide = scores[labels == 0]
    if spoof.size == 0 or bonafide.size == 0:
        return None, None
    return (
        float(compute_eer(spoof, bonafide).eer),
        float(compute_auc(spoof, bonafide)),
    )


def save_checkpoint(
    model: AudioSpoofDetector,
    checkpoint_path: Path,
) -> dict[str, object]:
    """Save ``network.``-prefixed weights and verify a strict reload."""
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        key.replace("backbone.", "network.", 1): value.detach().cpu()
        for key, value in model.state_dict().items()
    }
    temporary = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(checkpoint_path)

    _reloaded, report = load_model_from_checkpoint(
        checkpoint_path, device="cpu"
    )
    if report.missing_keys or report.unexpected_keys:
        raise RuntimeError(
            "saved checkpoint does not strict-load: "
            f"missing={report.missing_keys}, "
            f"unexpected={report.unexpected_keys}"
        )
    return {
        "path": str(checkpoint_path),
        "sha256": report.sha256,
        "state_dict_keys": report.state_dict_keys,
        "remapped_keys": report.remapped_keys,
        "missing_keys": list(report.missing_keys),
        "unexpected_keys": list(report.unexpected_keys),
        "total_parameters": report.total_parameters,
        "backbone_parameters": report.backbone_parameters,
        "head_parameters": report.head_parameters,
        "head_state_sha256": report.head_state_sha256,
        "head_last_bias": report.head_last_bias,
        "batch_norm_steps": list(report.batch_norm_steps),
    }


def train_probe(config: ProbeTrainingConfig) -> ProbeTrainingResult:
    """Train one front-end probe and write its checkpoint, log and config."""
    device = resolve_device(config.device)
    if config.checkpoint_out.exists() and not config.force:
        raise FileExistsError(
            f"{config.checkpoint_out} already exists; pass --force to overwrite"
        )
    if not Path(config.train_manifest).is_file():
        raise FileNotFoundError(
            f"training manifest not found: {config.train_manifest}"
        )
    if not Path(config.dev_manifest).is_file():
        raise FileNotFoundError(f"dev manifest not found: {config.dev_manifest}")

    repo_root = Path(__file__).resolve().parents[2]
    set_seed(config.seed)
    model = build_probe_model(
        seed=config.seed,
        reference_checkpoint=config.reference_checkpoint,
        device=device,
        freeze_bn=config.freeze_bn,
        backbone_weights=config.backbone_weights,
    )
    parity = (
        verify_backbone_parity(model, config.reference_checkpoint)
        if config.verify_backbone
        else {"skipped": True}
    )

    train_frame = load_probe_frame(
        config.train_manifest, repo_root, limit=config.limit
    )
    dev_frame = load_probe_frame(
        config.dev_manifest, repo_root, limit=config.limit
    )
    for name, frame in (("train", train_frame), ("dev", dev_frame)):
        if frame["label"].nunique() < 2:
            raise ValueError(f"{name} split must contain both classes")

    transform = preprocess.build_inference_transform()
    loader_kwargs: dict[str, object] = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": device.type == "cuda",
    }
    if config.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["worker_init_fn"] = _worker_init
    generator = torch.Generator()
    generator.manual_seed(config.seed)

    train_loader = DataLoader(
        ProbeImageDataset(train_frame, transform),
        shuffle=True,
        generator=generator,
        **loader_kwargs,
    )
    dev_loader = DataLoader(
        ProbeImageDataset(dev_frame, transform),
        shuffle=False,
        **loader_kwargs,
    )

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.learning_rate,
    )

    best_loss = math.inf
    best_epoch = 0
    best_eer: float | None = None
    best_auc: float | None = None
    patience_counter = 0
    rows: list[dict[str, object]] = []

    for epoch in range(1, config.max_epochs + 1):
        started = time.perf_counter()
        model.train()
        if config.freeze_bn:
            _freeze_bn(model.backbone)

        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images).reshape(-1)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            train_loss += float(loss.item()) * images.size(0)
            predictions = (torch.sigmoid(logits) >= 0.5).float()
            train_correct += int((predictions == targets).sum().item())
            train_total += images.size(0)

        train_loss /= max(1, train_total)
        train_accuracy = train_correct / max(1, train_total)

        dev_loss, dev_scores, dev_labels = _evaluate_loader(
            model, dev_loader, criterion, device
        )
        dev_accuracy = float(
            ((dev_scores >= 0.5).astype(int) == dev_labels).mean()
        )
        dev_eer, dev_auc = _eer_auc(dev_scores, dev_labels)

        improved = dev_loss < best_loss
        if improved:
            best_loss = dev_loss
            best_epoch = epoch
            best_eer = dev_eer
            best_auc = dev_auc
            patience_counter = 0
            checkpoint_report = save_checkpoint(model, config.checkpoint_out)
        else:
            patience_counter += 1

        elapsed = time.perf_counter() - started
        rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "dev_loss": dev_loss,
                "dev_accuracy": dev_accuracy,
                "dev_eer": dev_eer,
                "dev_auc": dev_auc,
                "improved": improved,
                "best_epoch": best_epoch,
                "seconds": elapsed,
            }
        )
        eer_text = f"{dev_eer:.4f}" if dev_eer is not None else "n/a"
        print(
            f"[{config.frontend}] epoch {epoch:02d}/{config.max_epochs} "
            f"train_loss={train_loss:.4f} dev_loss={dev_loss:.4f} "
            f"dev_eer={eer_text} {'*' if improved else ''}"
        )
        if patience_counter >= config.patience:
            print(f"[{config.frontend}] early stopping at epoch {epoch}")
            break

    if best_epoch == 0:
        raise RuntimeError("training did not produce a checkpoint")

    output_dir = Path(config.checkpoint_out).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train_log.csv"
    pd.DataFrame(rows).to_csv(log_path, index=False)

    config_path = output_dir / "config.json"
    payload = {
        "experiment": config.experiment_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": config.command,
        "frontend": config.frontend,
        "frontend_payload": (
            config.provenance_payload
            if config.provenance_payload is not None
            else frontend_payload(config.frontend)
        ),
        "training": {
            "seed": config.seed,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "max_epochs": config.max_epochs,
            "patience": config.patience,
            "loss": "BCEWithLogitsLoss (unweighted)",
            "optimizer": "Adam",
            "selection": "best full-dev BCE loss",
            "freeze_bn": config.freeze_bn,
            "backbone_weights": config.backbone_weights,
            "limit": config.limit,
        },
        "data": {
            "train_manifest": str(config.train_manifest),
            "train_manifest_sha256": sha256_file(config.train_manifest),
            "train_rows": int(len(train_frame)),
            "dev_manifest": str(config.dev_manifest),
            "dev_manifest_sha256": sha256_file(config.dev_manifest),
            "dev_rows": int(len(dev_frame)),
        },
        "model": {
            "architecture": (
                "ResNet18 -> Linear(512,128) -> ReLU -> Dropout(0.5) -> Linear(128,1)"
            ),
            "frozen": "backbone parameters requires_grad=False",
            "batch_norm": "train mode (baseline-faithful)" if not config.freeze_bn else "eval mode",
            "head_init": f"torch.manual_seed({config.seed}) + reset_parameters()",
            "backbone_parity": parity,
        },
        "result": {
            "checkpoint": checkpoint_report,
            "best_epoch": best_epoch,
            "epochs_run": len(rows),
            "best_dev_loss": best_loss,
            "best_dev_eer": best_eer,
            "best_dev_auc": best_auc,
            "train_log": str(log_path),
        },
        "environment": environment_info(device),
        "git": git_info(),
    }
    config_path.write_text(
        json.dumps(_jsonable(payload), indent=2) + "\n",
        encoding="utf-8",
    )

    return ProbeTrainingResult(
        checkpoint=Path(config.checkpoint_out),
        checkpoint_sha256=str(checkpoint_report["sha256"]),
        best_epoch=best_epoch,
        epochs_run=len(rows),
        best_dev_loss=best_loss,
        best_dev_eer=best_eer,
        best_dev_auc=best_auc,
        log_path=log_path,
        config_path=config_path,
    )
