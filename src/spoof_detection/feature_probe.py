"""Feature-level control probes for the E1.4 shortcut ablation.

The image conditions in :mod:`spoof_detection.shortcuts` keep the frozen-backbone
+ fresh-head methodology.  The duration and metadata shortcuts are *vectors*, not
images, so they get a tiny logistic-regression classifier instead.  Three probes
are defined:

* ``duration_only`` - clip duration in seconds before the 4-second fix;
* ``metadata_system_id`` - whether the clip carries a spoof-system ID
  (``system_id != "-"``), the tautological metadata ceiling;
* ``metadata_speaker_id`` - one-hot speaker identity, which is expected to carry
  no label signal (train/dev/eval speakers are disjoint).

All three share the same split/seed discipline and the same threshold-free
EER/AUC metric contract as the image conditions.  The classifier is a single
linear layer fitted on the train split only; dev is used for early stopping and
eval is the clean estimate.  Features are standardised with train statistics.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .dataset import repo_relative, sha256_file
from .evaluate import (
    describe_attack_subset,
    environment_info,
    git_info,
    resolve_device,
)
from .metrics import compute_auc, compute_eer, summarize_split
from .train_probe import set_seed

FEATURE_PROBE_ORDER = (
    "duration_only",
    "metadata_system_id",
    "metadata_speaker_id",
)


@dataclass(frozen=True)
class FeatureProbeResult:
    """Summary returned after training one feature probe."""

    probe_dir: Path
    best_epoch: int
    best_dev_loss: float
    best_dev_eer: float | None
    best_dev_auc: float | None


def feature_probe_payload(name: str) -> dict[str, object]:
    """Return the JSON-serialisable provenance block for a feature probe."""
    descriptions = {
        "duration_only": "clip duration in seconds before the 4-second fix",
        "metadata_system_id": (
            "binary feature: system_id != '-' (spoof-system presence)"
        ),
        "metadata_speaker_id": "one-hot speaker identity",
    }
    return {
        "condition": name,
        "kind": "feature_probe",
        "model": "logistic regression (nn.Linear(d, 1) + BCE)",
        "description": descriptions[name],
        "feature": name,
        "standardization": "z-score using train statistics",
        "selection": "best full-dev BCE loss",
    }


def _clip_duration(path: str | Path) -> float:
    return float(librosa.get_duration(path=str(path)))


def feature_matrix(
    name: str,
    records: pd.DataFrame,
    speaker_categories: list[str] | None = None,
) -> tuple[np.ndarray, list[str] | None]:
    """Return the feature matrix and the speaker categories used (if any).

    ``records`` must contain ``source_path``, ``system_id`` and ``speaker_id``.
    ``speaker_categories`` fixes the one-hot column order; when ``None`` it is
    derived from the records (used at training time).
    """
    if name == "duration_only":
        values = np.array(
            [[_clip_duration(path) for path in records["source_path"]]],
            dtype=np.float32,
        ).T
        return values, None

    if name == "metadata_system_id":
        values = np.array(
            [
                [1.0 if str(system_id) != "-" else 0.0]
                for system_id in records["system_id"]
            ],
            dtype=np.float32,
        )
        return values, None

    if name == "metadata_speaker_id":
        if speaker_categories is None:
            speaker_categories = sorted(
                {str(speaker) for speaker in records["speaker_id"]}
            )
        index = {category: i for i, category in enumerate(speaker_categories)}
        values = np.zeros((len(records), len(speaker_categories)), dtype=np.float32)
        for row, speaker in enumerate(map(str, records["speaker_id"])):
            if speaker in index:
                values[row, index[speaker]] = 1.0
        return values, speaker_categories

    raise ValueError(f"unknown feature probe {name!r}")


def _standardize(
    train_x: np.ndarray,
    dev_x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Standardise features with train statistics; return mean/std too."""
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std = np.where(std == 0.0, 1.0, std)
    return (train_x - mean) / std, (dev_x - mean) / std, mean, std


def _train_logistic(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_dev: np.ndarray,
    y_dev: np.ndarray,
    seed: int,
    learning_rate: float,
    max_epochs: int,
    patience: int,
    device: torch.device,
) -> tuple[nn.Module, int, float]:
    """Fit a single linear layer with Adam and early stopping on dev BCE."""
    set_seed(seed)
    model = nn.Linear(x_train.shape[1], 1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    xt = torch.tensor(x_train, dtype=torch.float32, device=device)
    yt = torch.tensor(y_train, dtype=torch.float32, device=device)
    xd = torch.tensor(x_dev, dtype=torch.float32, device=device)
    yd = torch.tensor(y_dev, dtype=torch.float32, device=device)

    best_loss = math.inf
    best_epoch = 0
    best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    patience_counter = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(xt).reshape(-1), yt)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.inference_mode():
            dev_loss = float(criterion(model(xd).reshape(-1), yd).item())

        if dev_loss < best_loss:
            best_loss = dev_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().clone()
                for key, value in model.state_dict().items()
            }
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            break

    model.load_state_dict(best_state)
    model.eval()
    return model, best_epoch, best_loss


def _state_from_model(
    model: nn.Module,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    speaker_categories: list[str] | None,
) -> dict[str, object]:
    state_dict = model.state_dict()
    return {
        "weights": state_dict["weight"].detach().cpu().numpy(),
        "bias": float(state_dict["bias"].detach().cpu().numpy()[0]),
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "speaker_categories": speaker_categories,
    }


def _predict(state: dict[str, object], x: np.ndarray) -> np.ndarray:
    weights = np.asarray(state["weights"], dtype=np.float32).reshape(-1)
    bias = float(state["bias"])
    logits = x @ weights + bias
    return 1.0 / (1.0 + np.exp(-logits))


def train_feature_probe(
    name: str,
    train_records: pd.DataFrame,
    dev_records: pd.DataFrame,
    output_dir: Path,
    seed: int = 42,
    learning_rate: float = 1e-2,
    max_epochs: int = 2000,
    patience: int = 200,
    device: str = "auto",
    force: bool = False,
    command: str = "",
) -> FeatureProbeResult:
    """Train one feature probe and write ``feature_probe.pth`` + ``config.json``."""
    if name not in FEATURE_PROBE_ORDER:
        raise ValueError(f"unknown feature probe {name!r}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "feature_probe.pth"
    if checkpoint_path.exists() and not force:
        raise FileExistsError(
            f"{checkpoint_path} already exists; pass --force to overwrite"
        )

    resolved = resolve_device(device)
    x_train, speaker_categories = feature_matrix(name, train_records)
    x_dev, _speaker_categories = feature_matrix(
        name, dev_records, speaker_categories=speaker_categories
    )
    y_train = train_records["label"].astype(int).to_numpy()
    y_dev = dev_records["label"].astype(int).to_numpy()

    x_train_s, x_dev_s, mean, std = _standardize(x_train, x_dev)
    model, best_epoch, best_loss = _train_logistic(
        x_train_s,
        y_train,
        x_dev_s,
        y_dev,
        seed=seed,
        learning_rate=learning_rate,
        max_epochs=max_epochs,
        patience=patience,
        device=resolved,
    )
    state = _state_from_model(model, mean, std, speaker_categories)

    torch.save(state, checkpoint_path)

    dev_scores = _predict(state, x_dev_s)
    dev_eer, dev_auc = None, None
    spoof = dev_scores[y_dev == 1]
    bonafide = dev_scores[y_dev == 0]
    if spoof.size and bonafide.size:
        dev_eer = float(compute_eer(spoof, bonafide).eer)
        dev_auc = float(compute_auc(spoof, bonafide))

    config_path = output_dir / "config.json"
    config = {
        "experiment": "E1.4",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": command,
        "condition": name,
        "feature_payload": feature_probe_payload(name),
        "training": {
            "seed": seed,
            "learning_rate": learning_rate,
            "max_epochs": max_epochs,
            "patience": patience,
            "loss": "BCEWithLogitsLoss (unweighted)",
            "optimizer": "Adam",
            "selection": "best full-dev BCE loss",
            "standardization": "z-score using train statistics",
        },
        "data": {
            "train_rows": int(len(train_records)),
            "dev_rows": int(len(dev_records)),
            "feature_dim": int(x_train.shape[1]),
            "speaker_categories": speaker_categories,
        },
        "result": {
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "best_epoch": best_epoch,
            "best_dev_loss": best_loss,
            "best_dev_eer": dev_eer,
            "best_dev_auc": dev_auc,
        },
        "environment": environment_info(resolved),
        "git": git_info(),
    }
    config_path.write_text(
        json.dumps(_jsonable(config), indent=2) + "\n", encoding="utf-8"
    )

    return FeatureProbeResult(
        probe_dir=output_dir,
        best_epoch=best_epoch,
        best_dev_loss=best_loss,
        best_dev_eer=dev_eer,
        best_dev_auc=dev_auc,
    )


def evaluate_feature_probe(
    name: str,
    probe_dir: Path,
    dev_records: pd.DataFrame,
    eval_records: pd.DataFrame,
    seed: int,
    metrics_path: Path,
    device: str = "auto",
    command: str = "",
) -> tuple[dict[str, object], Path]:
    """Score dev/eval with a trained feature probe and write ``metrics.json``."""
    probe_dir = Path(probe_dir)
    metrics_path = Path(metrics_path)
    checkpoint_path = probe_dir / "feature_probe.pth"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"feature probe checkpoint not found: {checkpoint_path}"
        )

    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    speaker_categories = state.get("speaker_categories")
    mean = np.asarray(state["feature_mean"], dtype=np.float32)
    std = np.asarray(state["feature_std"], dtype=np.float32)
    std = np.where(std == 0.0, 1.0, std)

    def score(records: pd.DataFrame) -> np.ndarray:
        x, _cats = feature_matrix(name, records, speaker_categories=speaker_categories)
        x = (x - mean) / std
        return _predict(state, x)

    split_frames: dict[str, pd.DataFrame] = {}
    for split, records in (("dev", dev_records), ("eval", eval_records)):
        frame = records[
            ["speaker_id", "system_id", "label"]
        ].copy()
        frame["score"] = score(records)
        split_frames[split] = frame

    dev_frame = split_frames["dev"]
    dev_summary = summarize_split(dev_frame)
    dev_threshold = dev_summary.get("eer_threshold_descriptive")

    metrics: dict[str, object] = {
        "experiment": "E1.4",
        "title": f"E1.4 feature probe: {name} (seed {seed})",
        "frontend": name,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": command,
        "checkpoint": {
            "path": repo_relative(checkpoint_path, Path(__file__).resolve().parents[2]),
            "sha256": sha256_file(checkpoint_path),
            "note": "logistic-regression feature probe; no backbone parameters",
        },
        "preprocessing": feature_probe_payload(name),
        "thresholds": {
            "fixed_0.5": 0.5,
            "dev_eer": dev_threshold,
            "selected_on": "dev" if dev_threshold is not None else None,
        },
        "splits": {
            "dev": {
                "provenance": {"split": "dev", "source": "feature"},
                "attack_subset": describe_attack_subset(dev_frame),
                "metrics": dev_summary,
            },
            "eval": {
                "provenance": {"split": "eval", "source": "feature"},
                "attack_subset": describe_attack_subset(split_frames["eval"]),
                "metrics": summarize_split(split_frames["eval"]),
            },
        },
    }

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(_jsonable(metrics), indent=2) + "\n", encoding="utf-8"
    )
    return metrics, metrics_path


def _jsonable(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        numeric = float(value)
        return None if math.isnan(numeric) or math.isinf(numeric) else numeric
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.ndarray,)):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value
