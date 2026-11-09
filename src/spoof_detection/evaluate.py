"""Batch evaluation CLI for the frozen ResNet18 audio spoof detector.

Typical usage::

    uv run python -m spoof_detection.evaluate \
        --checkpoint Models/Itteration1/best_audio_spoof_model_unweighted.pth \
        --asvspoof dev eval \
        --source image \
        --output-root reports/baseline

The harness scores clips in source order, writes per-clip scores, and produces
``metrics.json`` plus a Markdown summary.  It never trains or tunes on eval:
the only data-selected operating threshold is the EER threshold computed on the
dev split, which is then transferred to eval.
"""

from __future__ import annotations

import os

# Rendering audio through matplotlib must never require a display, and worker
# processes inherit this setting.
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import json
import math
import platform
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import librosa
import matplotlib
import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from . import preprocess
from .dataset import repo_relative, sha256_file
from .metrics import compute_eer, summarize_split
from .model import (
    AudioSpoofDetector,
    load_model_from_checkpoint,
)
from .protocol import (
    SPLIT_SPECS,
    audio_dir_for,
    load_cm_protocol,
    protocol_path_for,
    validate_cm_protocol,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "LA"
DEFAULT_IMAGE_ROOT = REPO_ROOT / "data" / "dataset_img"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "reports" / "baseline"

FIXED_THRESHOLD = 0.5
SAMPLE_SELECTION = (
    "deterministic head-stratified: first N/2 records from each class, "
    "sorted by source row"
)


@dataclass(frozen=True)
class ClipRecord:
    """One evaluation clip and its provenance."""

    row_index: int
    clip_id: str
    protocol_row: int | None
    speaker_id: str
    system_id: str
    label: int
    source_path: Path


@dataclass(frozen=True)
class ThresholdReference:
    """A dev-selected threshold loaded from a previous metrics file."""

    threshold: float
    selected_on: str
    source_path: str
    checkpoint_sha256: str


@dataclass(frozen=True)
class EvaluationConfig:
    """All inputs for one or more checkpoint evaluations."""

    checkpoints: tuple[Path, ...]
    asvspoof_splits: tuple[str, ...] | None
    manifest_path: Path | None
    manifest_base_dir: Path | None
    split_name: str
    source: str
    data_root: Path
    image_root: Path
    output_root: Path
    output_dir: Path | None
    batch_size: int
    num_workers: int
    device: str
    limit: int | None
    threshold_from: Path | None
    show_progress: bool
    command: str


def records_to_frame(records: Sequence[ClipRecord]) -> pd.DataFrame:
    """Convert records to the canonical dataframe used by metrics and CSVs."""
    rows = [
        {
            "row_index": record.row_index,
            "clip_id": record.clip_id,
            "protocol_row": (
                record.protocol_row if record.protocol_row is not None else pd.NA
            ),
            "speaker_id": record.speaker_id,
            "system_id": record.system_id,
            "label": record.label,
            "source_path": str(record.source_path),
        }
        for record in records
    ]
    frame = pd.DataFrame(rows)
    frame["protocol_row"] = pd.array(frame["protocol_row"], dtype="Int64")
    return frame


def _resolve_path(value: str | Path, base_dir: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _clean_optional_identifier(value: object, fallback: str) -> str:
    if value is None or value is pd.NA:
        return fallback
    try:
        if pd.isna(value):
            return fallback
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return text if text and text.lower() not in {"nan", "<na>"} else fallback


def _map_label(value: object) -> int:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"0", "bonafide", "bona-fide", "real"}:
            return 0
        if normalized in {"1", "spoof", "spoofed", "fake"}:
            return 1
    try:
        label = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        label = -1
    if label in (0, 1):
        return label
    raise ValueError(f"unsupported label value: {value!r}")


def describe_attack_subset(frame: pd.DataFrame) -> str:
    """Return the metric-contract attack subset for an evaluated frame."""
    system_ids = frame["system_id"].astype(str).tolist()
    attacks = sorted(
        {system_id for system_id in system_ids if system_id not in {"-", "", "nan"}}
    )
    if not attacks:
        return "none (bonafide only)" if not (frame["label"] == 1).any() else "unknown"
    expected_dev = [f"A{index:02d}" for index in range(1, 7)]
    expected_eval = [f"A{index:02d}" for index in range(7, 20)]
    if attacks == expected_dev:
        return "A01-A06"
    if attacks == expected_eval:
        return "A07-A19"
    return ", ".join(attacks)


def limit_records(
    records: Sequence[ClipRecord],
    limit: int,
) -> list[ClipRecord]:
    """Select a deterministic class-stratified subset for a smoke test."""
    if limit >= len(records):
        return list(records)

    per_class = max(1, limit // 2)
    bonafide = [record for record in records if record.label == 0][:per_class]
    spoof = [record for record in records if record.label == 1][:per_class]
    selected = bonafide + spoof
    selected_indices = {record.row_index for record in selected}

    if len(selected) < limit:
        for record in records:
            if record.row_index not in selected_indices:
                selected.append(record)
                selected_indices.add(record.row_index)
                if len(selected) >= limit:
                    break

    return sorted(selected, key=lambda record: record.row_index)


def load_asvspoof_records(
    split: str,
    source: str,
    data_root: Path = DEFAULT_DATA_ROOT,
    image_root: Path = DEFAULT_IMAGE_ROOT,
    repo_root: Path = REPO_ROOT,
) -> tuple[list[ClipRecord], dict[str, object]]:
    """Load one ASVspoof split from its CM protocol and exported images."""
    if split not in SPLIT_SPECS:
        raise ValueError(f"unknown ASVspoof split: {split}")

    protocol = load_cm_protocol(
        split,
        data_root=data_root,
        repo_root=repo_root,
    )
    problems = validate_cm_protocol(protocol, split)
    if problems:
        raise ValueError(
            f"{split} protocol validation failed: " + "; ".join(problems)
        )

    protocol_path = protocol_path_for(split, data_root)
    provenance: dict[str, object] = {
        "dataset": "ASVspoof 2019 LA",
        "split": split,
        "source": source,
        "data_root": repo_relative(data_root, repo_root),
        "protocol_path": repo_relative(protocol_path, repo_root),
        "protocol_sha256": sha256_file(protocol_path),
    }
    if split == "train":
        provenance["note"] = (
            "ASVspoof train uses the full 25,380-image export; use "
            "--manifest manifests/train_seed42.csv for the 10,080-clip "
            "first-iteration training subset."
        )

    if source == "image":
        image_root = Path(image_root)
        metadata_path = image_root / split / f"metadata_{split}.csv"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"image metadata not found: {metadata_path}")

        metadata = pd.read_csv(metadata_path)
        required = {"speaker_id", "system_id", "label", "img_path"}
        missing_columns = required - set(metadata.columns)
        if missing_columns:
            raise ValueError(
                f"{metadata_path} is missing columns: {sorted(missing_columns)}"
            )
        if len(metadata) != len(protocol):
            raise ValueError(
                f"{metadata_path} has {len(metadata)} rows, protocol has "
                f"{len(protocol)}"
            )

        metadata = metadata.copy()
        metadata["protocol_row"] = range(len(metadata))
        alignment_checks = {
            "label": protocol["label"].to_numpy(),
            "speaker_id": protocol["speaker_id"].astype(str).to_numpy(),
            "system_id": protocol["system_id"].astype(str).to_numpy(),
        }
        for column, expected in alignment_checks.items():
            observed = metadata[column]
            if column != "label":
                observed = observed.astype(str)
            if not np.array_equal(observed.to_numpy(), expected):
                raise ValueError(
                    f"{metadata_path} {column} does not align row-wise with "
                    f"{protocol_path}"
                )

        path_problems: list[str] = []
        source_paths: list[Path] = []
        for row in metadata.itertuples(index=False):
            protocol_row = int(row.protocol_row)
            label = int(row.label)
            folder = "bonafide" if label == 0 else "spoofed"
            raw_path = str(row.img_path).replace("\\", "/")
            expected_suffix = f"sample_{protocol_row}.png"
            if not raw_path.endswith(expected_suffix):
                path_problems.append(
                    f"row {protocol_row}: path does not end with {expected_suffix}"
                )
            if f"/{folder}/" not in raw_path:
                path_problems.append(
                    f"row {protocol_row}: path is not in the {folder}/ folder"
                )
            source_paths.append(
                image_root / split / folder / f"sample_{protocol_row}.png"
            )
        if path_problems:
            raise ValueError(
                "image metadata paths do not match protocol rows: "
                + "; ".join(path_problems[:5])
            )

        missing_files = [path for path in source_paths if not path.is_file()]
        if missing_files:
            raise FileNotFoundError(
                f"{len(missing_files)} image files are missing, e.g. "
                f"{[str(path) for path in missing_files[:5]]}"
            )

        records = [
            ClipRecord(
                row_index=index,
                clip_id=str(row.audio_file_name),
                protocol_row=int(row.protocol_row),
                speaker_id=str(row.speaker_id),
                system_id=str(row.system_id),
                label=int(row.label),
                source_path=source_path,
            )
            for index, (row, source_path) in enumerate(
                zip(protocol.itertuples(index=False), source_paths)
            )
        ]
        provenance.update(
            {
                "image_root": repo_relative(image_root, repo_root),
                "metadata_path": repo_relative(metadata_path, repo_root),
                "metadata_sha256": sha256_file(metadata_path),
            }
        )
        return records, provenance

    if source != "audio":
        raise ValueError(f"unknown source: {source}")

    audio_dir = audio_dir_for(split, data_root)
    source_paths = [
        _resolve_path(row.source_flac, repo_root)
        for row in protocol.itertuples(index=False)
    ]
    missing_files = [path for path in source_paths if not path.is_file()]
    if missing_files:
        raise FileNotFoundError(
            f"{len(missing_files)} audio files are missing, e.g. "
            f"{[str(path) for path in missing_files[:5]]}"
        )

    records = [
        ClipRecord(
            row_index=index,
            clip_id=str(row.audio_file_name),
            protocol_row=int(row.protocol_row),
            speaker_id=str(row.speaker_id),
            system_id=str(row.system_id),
            label=int(row.label),
            source_path=source_path,
        )
        for index, (row, source_path) in enumerate(
            zip(protocol.itertuples(index=False), source_paths)
        )
    ]
    provenance.update(
        {
            "audio_dir": repo_relative(audio_dir, repo_root),
        }
    )
    return records, provenance


def load_manifest_records(
    manifest_path: Path,
    source: str,
    base_dir: Path,
    split_name: str,
    repo_root: Path = REPO_ROOT,
) -> tuple[list[ClipRecord], dict[str, object]]:
    """Load a generic labelled CSV manifest for an external corpus."""
    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")

    frame = pd.read_csv(manifest_path)
    if "label" not in frame.columns:
        raise ValueError(f"{manifest_path} is missing the required 'label' column")

    if source == "image":
        candidates = ("image_path", "img_path")
    elif source == "audio":
        candidates = ("source_flac", "audio_path", "file_path", "path")
    else:
        raise ValueError(f"unknown source: {source}")

    path_column = next(
        (column for column in candidates if column in frame.columns),
        None,
    )
    if path_column is None:
        raise ValueError(
            f"{manifest_path} has no {source} path column; expected one of "
            f"{list(candidates)}"
        )

    clip_ids = (
        frame["clip_id"].astype(str)
        if "clip_id" in frame.columns
        else (
            frame["audio_file_name"].astype(str)
            if "audio_file_name" in frame.columns
            else pd.Series(
                [f"clip_{index:06d}" for index in range(len(frame))]
            )
        )
    )
    speaker_ids = (
        frame["speaker_id"].map(
            lambda value: _clean_optional_identifier(value, "unknown")
        )
        if "speaker_id" in frame.columns
        else pd.Series(["unknown"] * len(frame))
    )
    system_column = (
        "system_id"
        if "system_id" in frame.columns
        else "attack_id" if "attack_id" in frame.columns else None
    )
    system_ids = (
        frame[system_column].map(
            lambda value: _clean_optional_identifier(value, "-")
        )
        if system_column is not None
        else pd.Series(["-"] * len(frame))
    )
    protocol_rows = (
        pd.to_numeric(frame["protocol_row"], errors="coerce")
        if "protocol_row" in frame.columns
        else pd.Series([pd.NA] * len(frame), dtype="Int64")
    )

    source_paths: list[Path] = []
    missing_paths: list[int] = []
    for index, value in enumerate(frame[path_column].tolist()):
        if value is None or value is pd.NA or pd.isna(value):
            raise ValueError(
                f"{manifest_path} row {index} has an empty {path_column} value"
            )
        path = _resolve_path(value, base_dir)
        source_paths.append(path)
        if not path.is_file():
            missing_paths.append(index)
    if missing_paths:
        raise FileNotFoundError(
            f"{len(missing_paths)} manifest files are missing, e.g. "
            f"{[str(source_paths[index]) for index in missing_paths[:5]]}"
        )

    records = [
        ClipRecord(
            row_index=index,
            clip_id=str(clip_ids.iloc[index]),
            protocol_row=(
                None if pd.isna(protocol_rows.iloc[index]) else int(protocol_rows.iloc[index])
            ),
            speaker_id=str(speaker_ids.iloc[index]),
            system_id=str(system_ids.iloc[index]),
            label=_map_label(frame["label"].iloc[index]),
            source_path=source_paths[index],
        )
        for index in range(len(frame))
    ]

    provenance: dict[str, object] = {
        "dataset": "external manifest",
        "split": split_name,
        "source": source,
        "manifest_path": repo_relative(manifest_path, repo_root),
        "manifest_sha256": sha256_file(manifest_path),
        "manifest_base_dir": repo_relative(base_dir, repo_root),
        "manifest_rows": len(frame),
    }
    return records, provenance


class ClipDataset(Dataset):
    """Dataset that returns ``(index, image_tensor, label)``."""

    def __init__(
        self,
        records: Sequence[ClipRecord],
        source: str,
        transform: object,
    ) -> None:
        self.records = list(records)
        self.source = source
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[int, torch.Tensor, int]:
        record = self.records[index]
        if self.source == "image":
            with Image.open(record.source_path) as image:
                image = image.convert("RGB")
                tensor = self.transform(image)  # type: ignore[operator]
        else:
            image = preprocess.load_mel_image(record.source_path)
            tensor = self.transform(image)  # type: ignore[operator]
        return index, tensor, record.label


def resolve_device(requested: str) -> torch.device:
    """Resolve ``auto``/``cpu``/``cuda`` to a concrete torch device."""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is not available")
    return torch.device(requested)


def environment_info(device: torch.device) -> dict[str, object]:
    """Record the software and hardware environment for this run."""
    if device.type == "cuda":
        device_name = torch.cuda.get_device_name(device)
    else:
        device_name = platform.processor() or platform.machine()
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "librosa": librosa.__version__,
        "matplotlib": matplotlib.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cudnn_version": (
            torch.backends.cudnn.version()
            if torch.cuda.is_available()
            else None
        ),
        "device": str(device),
        "device_name": device_name,
    }


def git_info(repo_root: Path = REPO_ROOT) -> dict[str, object]:
    """Record the current git commit and working-tree state."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return {
            "commit": commit,
            "dirty": bool(status),
            "status_porcelain": status.splitlines(),
        }
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        return {"commit": None, "dirty": None, "error": str(exc)}


def score_records(
    model: AudioSpoofDetector,
    records: Sequence[ClipRecord],
    source: str,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    show_progress: bool = True,
    description: str = "",
) -> pd.DataFrame:
    """Score every record and return the canonical frame with logits/scores."""
    transform = preprocess.build_inference_transform()
    dataset = ClipDataset(records, source, transform)
    loader_kwargs: dict[str, object] = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
    loader = DataLoader(dataset, **loader_kwargs)

    logits = np.empty(len(records), dtype=np.float64)
    scores = np.empty(len(records), dtype=np.float64)
    model.eval()
    with torch.inference_mode():
        for indices, batch, _labels in tqdm(
            loader,
            desc=description,
            disable=not show_progress,
            unit="batch",
        ):
            batch = batch.to(device, non_blocking=True)
            logit = model(batch).reshape(-1)
            batch_indices = indices.numpy()
            logits[batch_indices] = logit.detach().float().cpu().numpy()
            scores[batch_indices] = (
                torch.sigmoid(logit).detach().float().cpu().numpy()
            )

    if not np.isfinite(scores).all():
        raise RuntimeError("model produced non-finite scores")

    frame = records_to_frame(records)
    frame["logit"] = logits
    frame["score"] = scores
    return frame


def load_threshold_from_metrics(path: Path) -> ThresholdReference:
    """Load a dev-selected EER threshold from a previous metrics.json."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"threshold metrics file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("partial") is True:
        raise ValueError(
            f"{path} is a partial run; refusing to use its threshold"
        )
    thresholds = payload.get("thresholds") or {}
    threshold = thresholds.get("dev_eer")
    selected_on = thresholds.get("selected_on")
    checkpoint = payload.get("checkpoint") or {}
    checkpoint_sha256 = checkpoint.get("sha256")
    if threshold is None:
        raise ValueError(f"{path} has no dev_eer threshold")
    if selected_on != "dev":
        raise ValueError(
            f"{path} threshold was not selected on dev (selected_on={selected_on!r})"
        )
    if not checkpoint_sha256:
        raise ValueError(f"{path} does not record a checkpoint SHA-256")
    return ThresholdReference(
        threshold=float(threshold),
        selected_on=str(selected_on),
        source_path=str(path),
        checkpoint_sha256=str(checkpoint_sha256),
    )


def _output_dir_for(
    config: EvaluationConfig,
    checkpoint_path: Path,
) -> Path:
    if config.output_dir is not None:
        if len(config.checkpoints) != 1:
            raise ValueError("--output-dir requires exactly one --checkpoint")
        return config.output_dir
    name = checkpoint_path.stem
    if config.limit is not None:
        name = f"{name}_partial{config.limit}"
    return config.output_root / name


def _to_jsonable(value: object) -> object:
    """Convert numpy/pandas scalar types and non-finite floats for JSON."""
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        numeric = float(value)
        return None if math.isnan(numeric) or math.isinf(numeric) else numeric
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if value is pd.NA:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _fmt(value: object, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def _pct(value: object) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * float(value):.2f}%"


def _fmt_threshold(value: object) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.6f}"


def render_summary(metrics: dict[str, object]) -> str:
    """Render a readable Markdown summary from the metrics payload."""
    checkpoint = metrics["checkpoint"]  # type: ignore[index]
    splits = metrics["splits"]  # type: ignore[index]
    lines: list[str] = [
        f"# Baseline evaluation: `{Path(str(checkpoint['path'])).name}`",
        "",
        f"Generated: `{metrics['generated_at_utc']}`",
        "",
        "## Metric contract",
        "",
        "| Split | Attack subset | Operating thresholds | Samples | Balance |",
        "|---|---|---|---:|---|",
    ]

    for split, payload in splits.items():  # type: ignore[union-attr]
        summary = payload["metrics"]
        counts = summary["sample_counts"]
        operating = ", ".join(
            f"{name}={_fmt_threshold(values['threshold'])}"
            for name, values in summary["threshold_metrics"].items()
        )
        lines.append(
            f"| `{split}` | {payload['attack_subset']} | {operating} | "
            f"{counts['total']} | {counts['bonafide']} / {counts['spoof']} |"
        )

    lines.extend(
        [
            "",
            "Threshold-free metrics: EER and AUC are computed over all spoof "
            "clips against all bonafide clips in the split. They are `n/a` "
            "when a split has only one class.",
            "",
            "## Aggregate metrics",
            "",
            "| Split | EER (threshold-free) | AUC (threshold-free) | Brier | ECE (10 bins) |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for split, payload in splits.items():  # type: ignore[union-attr]
        summary = payload["metrics"]
        lines.append(
            f"| `{split}` | {_pct(summary['eer'])} | "
            f"{_fmt(summary['auc'])} | "
            f"{_fmt(summary['calibration']['brier'])} | "
            f"{_fmt(summary['calibration']['ece'])} |"
        )

    lines.extend(
        [
            "",
            "## Threshold metrics",
            "",
            "| Split | Threshold | Value | Accuracy | Bonafide recall | "
            "Spoof recall | FPR | FNR |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for split, payload in splits.items():  # type: ignore[union-attr]
        summary = payload["metrics"]
        for name, values in summary["threshold_metrics"].items():
            lines.append(
                f"| `{split}` | `{name}` | {_fmt_threshold(values['threshold'])} | "
                f"{_pct(values['accuracy'])} | "
                f"{_pct(values['bonafide_recall'])} | "
                f"{_pct(values['spoof_recall'])} | "
                f"{_pct(values['false_positive_rate'])} | "
                f"{_pct(values['false_negative_rate'])} |"
            )

    lines.extend(
        [
            "",
            "## Per-attack EER",
            "",
            "Per-attack EER uses that attack's spoof clips as positives and all "
            "bonafide clips in the split as negatives.",
            "",
            "| Split | Attack | n_spoof | n_bonafide | EER | AUC |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for split, payload in splits.items():  # type: ignore[union-attr]
        summary = payload["metrics"]
        if not summary["per_attack"]:
            lines.append(f"| `{split}` | none | 0 | 0 | n/a | n/a |")
            continue
        for attack_id, values in summary["per_attack"].items():
            lines.append(
                f"| `{split}` | {attack_id} | {values['n_spoof']} | "
                f"{values['n_bonafide']} | {_pct(values['eer'])} | "
                f"{_fmt(values['auc'])} |"
            )

    lines.extend(
        [
            "",
            "## Per-speaker metrics",
            "",
            "Per-speaker EER uses same-speaker bonafide clips as negatives. "
            "Recalls are at the fixed 0.5 threshold.",
            "",
            "| Split | Speaker | n_spoof | n_bonafide | Spoof recall@0.5 | "
            "Bonafide recall@0.5 | EER |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for split, payload in splits.items():  # type: ignore[union-attr]
        summary = payload["metrics"]
        for speaker_id, values in summary["per_speaker"].items():
            lines.append(
                f"| `{split}` | {speaker_id} | {values['n_spoof']} | "
                f"{values['n_bonafide']} | "
                f"{_pct(values['spoof_recall_at_0.5'])} | "
                f"{_pct(values['bonafide_recall_at_0.5'])} | "
                f"{_pct(values['eer'])} |"
            )

    lines.extend(
        [
            "",
            "## Checkpoint",
            "",
            f"- Path: `{checkpoint['path']}`",
            f"- SHA-256: `{checkpoint['sha256']}`",
            f"- Parameters: {checkpoint['total_parameters']} total "
            f"({checkpoint['backbone_parameters']} backbone, "
            f"{checkpoint['head_parameters']} head)",
            f"- State-dict keys: {checkpoint['state_dict_keys']} "
            f"({checkpoint['remapped_keys']} remapped)",
            f"- Missing keys: {checkpoint['missing_keys'] or 'none'}",
            f"- Unexpected keys: {checkpoint['unexpected_keys'] or 'none'}",
            "",
            "## Provenance",
            "",
        ]
    )
    for split, payload in splits.items():  # type: ignore[union-attr]
        provenance = payload["provenance"]
        lines.append(f"- **{split}**: {provenance}")
    lines.extend(
        [
            "",
            "## Environment and command",
            "",
            f"- Git commit: `{metrics['git'].get('commit')}` "
            f"(dirty: {metrics['git'].get('dirty')})",
            f"- Device: `{metrics['environment']['device']}` "
            f"({metrics['environment']['device_name']})",
            f"- Python / torch / torchvision: "
            f"{metrics['environment']['python']} / "
            f"{metrics['environment']['torch']} / "
            f"{metrics['environment']['torchvision']}",
            f"- Batch size / workers: {metrics['inference']['batch_size']} / "
            f"{metrics['inference']['num_workers']}",
            "",
            "```bash",
            str(metrics["command"]),
            "```",
            "",
            "## Caveats",
            "",
            "- Scores are uncalibrated; Brier and ECE are diagnostics, not "
            "evidence of calibrated probabilities.",
            "- EER uses the ASVspoof DET-curve convention (spoof is positive). "
            "Per-attack EER reuses all bonafide clips as negatives; per-speaker "
            "EER uses same-speaker bonafides.",
            "- Only the dev EER threshold is selected on data. The `dev_eer` "
            "threshold in an eval report is transferred from dev and is never "
            "selected on eval. Other EER thresholds are descriptive points on "
            "the DET curve, not operating thresholds.",
            "- Minimum t-DCF and confidence intervals are not reported in E0.2.",
        ]
    )
    if metrics.get("partial"):
        lines.append(
            f"- **Partial run:** only {metrics['limit']} clips per split were "
            f"scored ({metrics['sample_selection']}). Do not treat these as "
            "baseline results."
        )
    lines.append("")
    return "\n".join(lines)


def write_outputs(
    output_dir: Path,
    metrics: dict[str, object],
    scores_by_split: dict[str, pd.DataFrame],
) -> list[Path]:
    """Write metrics.json, per-clip scores and summary.md."""
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(_to_jsonable(metrics), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    written = [metrics_path]
    for split, frame in scores_by_split.items():
        score_path = output_dir / f"scores_{split}.csv"
        frame.to_csv(
            score_path,
            index=False,
            float_format="%.8f",
            na_rep="",
        )
        written.append(score_path)

    summary_path = output_dir / "summary.md"
    summary_path.write_text(render_summary(metrics), encoding="utf-8")
    written.append(summary_path)
    return written


def run_evaluation(config: EvaluationConfig) -> list[Path]:
    """Run one or more checkpoint evaluations and return output directories."""
    if not config.checkpoints:
        raise ValueError("at least one checkpoint is required")
    stems = [path.stem for path in config.checkpoints]
    if len(set(stems)) != len(stems):
        raise ValueError(
            "checkpoint filenames must be unique when --output-dir is not used"
        )
    if config.output_dir is not None and len(config.checkpoints) != 1:
        raise ValueError("--output-dir requires exactly one --checkpoint")

    device = resolve_device(config.device)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if config.manifest_path is not None:
        base_dir = config.manifest_base_dir or REPO_ROOT
        records, provenance = load_manifest_records(
            config.manifest_path,
            source=config.source,
            base_dir=base_dir,
            split_name=config.split_name,
        )
        split_records = {config.split_name: records}
        split_provenance = {config.split_name: provenance}
        asvspoof_splits: list[str] = []
    else:
        if not config.asvspoof_splits:
            raise ValueError("ASVspoof splits or a manifest are required")
        split_records = {}
        split_provenance = {}
        for split in config.asvspoof_splits:
            records, provenance = load_asvspoof_records(
                split,
                source=config.source,
                data_root=config.data_root,
                image_root=config.image_root,
            )
            split_records[split] = records
            split_provenance[split] = provenance
        asvspoof_splits = list(config.asvspoof_splits)

    if config.limit is not None:
        split_records = {
            split: limit_records(records, config.limit)
            for split, records in split_records.items()
        }

    threshold_reference = (
        load_threshold_from_metrics(config.threshold_from)
        if config.threshold_from is not None
        else None
    )
    if threshold_reference is not None and asvspoof_splits:
        if "dev" in asvspoof_splits:
            raise ValueError(
                "--threshold-from cannot be combined with an in-run dev split"
            )
        if any(split != "eval" for split in asvspoof_splits):
            raise ValueError(
                "--threshold-from is only valid for eval-only ASVspoof runs "
                "or manifest evaluations"
            )

    output_dirs: list[Path] = []
    for checkpoint_path in config.checkpoints:
        model, checkpoint_report = load_model_from_checkpoint(
            checkpoint_path,
            device=device,
        )
        if threshold_reference is not None:
            if threshold_reference.checkpoint_sha256 != checkpoint_report.sha256:
                raise ValueError(
                    "threshold file was produced by a different checkpoint: "
                    f"{threshold_reference.checkpoint_sha256} != "
                    f"{checkpoint_report.sha256}"
                )

        scores_by_split: dict[str, pd.DataFrame] = {}
        for split, records in split_records.items():
            scores_by_split[split] = score_records(
                model,
                records,
                source=config.source,
                device=device,
                batch_size=config.batch_size,
                num_workers=config.num_workers,
                show_progress=config.show_progress,
                description=f"{checkpoint_path.stem}:{split}",
            )

        dev_threshold: float | None = None
        threshold_source = "none"
        if "dev" in asvspoof_splits:
            dev_frame = scores_by_split["dev"]
            dev_threshold = compute_eer(
                dev_frame.loc[dev_frame["label"] == 1, "score"].to_numpy(),
                dev_frame.loc[dev_frame["label"] == 0, "score"].to_numpy(),
            ).threshold
            threshold_source = "in_run_dev"
        elif threshold_reference is not None:
            dev_threshold = threshold_reference.threshold
            threshold_source = f"file:{threshold_reference.source_path}"

        split_payloads: dict[str, object] = {}
        for split, frame in scores_by_split.items():
            extra_thresholds = (
                {"dev_eer": dev_threshold}
                if dev_threshold is not None
                else None
            )
            summary = summarize_split(
                frame,
                extra_thresholds=extra_thresholds,
            )
            split_payloads[split] = {
                "provenance": split_provenance[split],
                "attack_subset": describe_attack_subset(frame),
                "metrics": summary,
            }

        output_dir = _output_dir_for(config, checkpoint_path)
        metrics: dict[str, object] = {
            "experiment": "E0.2",
            "title": "Baseline evaluation harness",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "command": config.command,
            "checkpoint": {
                "path": repo_relative(checkpoint_path, REPO_ROOT),
                "sha256": checkpoint_report.sha256,
                "state_dict_keys": checkpoint_report.state_dict_keys,
                "remapped_keys": checkpoint_report.remapped_keys,
                "missing_keys": list(checkpoint_report.missing_keys),
                "unexpected_keys": list(checkpoint_report.unexpected_keys),
                "total_parameters": checkpoint_report.total_parameters,
                "backbone_parameters": checkpoint_report.backbone_parameters,
                "head_parameters": checkpoint_report.head_parameters,
            },
            "inference": {
                "source": config.source,
                "batch_size": config.batch_size,
                "num_workers": config.num_workers,
                "device": str(device),
                "cudnn_benchmark": False,
                "cudnn_deterministic": True,
            },
            "preprocessing": {
                "version": preprocess.PREPROCESSING_VERSION,
                "sample_rate": preprocess.SAMPLE_RATE,
                "target_seconds": preprocess.TARGET_SECONDS,
                "target_length": preprocess.TARGET_LENGTH,
                "n_mels": preprocess.N_MELS,
                "colormap": preprocess.COLORMAP,
                "power_to_db_ref": "np.max",
                "resize": [
                    preprocess.INFERENCE_SIZE,
                    preprocess.INFERENCE_SIZE,
                ],
                "imagenet_mean": list(preprocess.IMAGENET_MEAN),
                "imagenet_std": list(preprocess.IMAGENET_STD),
            },
            "thresholds": {
                "fixed_0.5": FIXED_THRESHOLD,
                "dev_eer": dev_threshold,
                "selected_on": "dev" if dev_threshold is not None else None,
                "source": threshold_source,
                "policy": (
                    "Only the dev EER threshold is selected on data. EER "
                    "thresholds reported for other splits are descriptive DET "
                    "points, not operating thresholds."
                ),
            },
            "partial": config.limit is not None,
            "limit": config.limit,
            "sample_selection": (
                SAMPLE_SELECTION if config.limit is not None else "all records"
            ),
            "environment": environment_info(device),
            "git": git_info(),
            "splits": split_payloads,
        }

        write_outputs(output_dir, metrics, scores_by_split)
        output_dirs.append(output_dir)

        print(f"[{checkpoint_path.name}] wrote {output_dir}")
        for split, payload in split_payloads.items():
            summary = payload["metrics"]
            eer_text = (
                f"{summary['eer']:.4f}"
                if summary["eer"] is not None
                else "n/a"
            )
            auc_text = (
                f"{summary['auc']:.4f}"
                if summary["auc"] is not None
                else "n/a"
            )
            print(
                f"  {split}: EER={eer_text} "
                f"AUC={auc_text} "
                f"n={summary['sample_counts']['total']} "
                f"(bonafide={summary['sample_counts']['bonafide']}, "
                f"spoof={summary['sample_counts']['spoof']})"
            )

    return output_dirs


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or a positive integer")
    return parsed


def _resolve_cli_path(value: Path, base: Path = REPO_ROOT) -> Path:
    if value.is_absolute():
        return value
    return (base / value).resolve()


def _safe_split_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned or "external"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Batch-evaluate a frozen audio spoof detector checkpoint on "
            "ASVspoof 2019 LA or a labelled CSV manifest."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  uv run python -m spoof_detection.evaluate \\\n"
            "    --checkpoint Models/Itteration1/"
            "best_audio_spoof_model_unweighted.pth \\\n"
            "    --asvspoof dev eval --source image \\\n"
            "    --output-root reports/baseline\n\n"
            "  uv run python -m spoof_detection.evaluate \\\n"
            "    --checkpoint Models/Itteration1/"
            "best_audio_spoof_model_unweighted.pth \\\n"
            "    --manifest manifests/train_seed42.csv --source image \\\n"
            "    --split-name train_seed42 --output-dir /tmp/opencode/e0.2\n"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        type=Path,
        metavar="PATH",
        help="checkpoint to evaluate; repeat for multiple checkpoints",
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--asvspoof",
        nargs="+",
        choices=sorted(SPLIT_SPECS),
        metavar="SPLIT",
        help="ASVspoof 2019 LA split(s) to evaluate: train, dev, eval",
    )
    source_group.add_argument(
        "--manifest",
        type=Path,
        metavar="CSV",
        help="labelled CSV manifest for an external corpus",
    )
    parser.add_argument(
        "--split-name",
        default="external",
        help="name used for a manifest split in reports (default: external)",
    )
    parser.add_argument(
        "--source",
        choices=("image", "audio"),
        default="image",
        help="read exported PNGs or recompute from source audio (default: image)",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/LA"),
        help="ASVspoof data root (default: data/LA)",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=Path("data/dataset_img"),
        help="exported image root (default: data/dataset_img)",
    )
    parser.add_argument(
        "--manifest-base-dir",
        type=Path,
        default=None,
        help="base directory for relative manifest paths (default: repo root)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("reports/baseline"),
        help="output root; each checkpoint gets a subdirectory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="explicit output directory; valid with exactly one checkpoint",
    )
    parser.add_argument(
        "--batch-size",
        type=_positive_int,
        default=64,
        help="inference batch size (default: 64)",
    )
    parser.add_argument(
        "--num-workers",
        type=_non_negative_int,
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
        type=_positive_int,
        default=None,
        help=(
            "smoke-test only: score a deterministic class-stratified subset "
            "of N clips per split; partial runs get a _partialN output folder"
        ),
    )
    parser.add_argument(
        "--threshold-from",
        type=Path,
        default=None,
        metavar="METRICS_JSON",
        help=(
            "use the dev EER threshold from a previous metrics.json; only "
            "valid for eval-only ASVspoof runs or manifest evaluations"
        ),
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="disable the tqdm progress bar",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    argv_list = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(argv_list)

    if args.limit is not None and args.limit < 2:
        parser.error("--limit must be at least 2 to include both classes")

    if args.manifest is not None and args.threshold_from is not None:
        if args.asvspoof is not None:
            parser.error("--threshold-from cannot be combined with --asvspoof")

    if args.asvspoof is not None:
        splits = list(dict.fromkeys(args.asvspoof))
        if args.threshold_from is not None:
            if "dev" in splits:
                parser.error(
                    "--threshold-from cannot be combined with an in-run dev split"
                )
            if any(split != "eval" for split in splits):
                parser.error(
                    "--threshold-from is only valid with --asvspoof eval"
                )
    else:
        splits = None

    if args.output_dir is not None and len(args.checkpoint) != 1:
        parser.error("--output-dir requires exactly one --checkpoint")

    if args.output_dir is None:
        stems = [path.stem for path in args.checkpoint]
        if len(set(stems)) != len(stems):
            parser.error(
                "checkpoint filenames must be unique when --output-dir is not used"
            )

    checkpoints = tuple(
        _resolve_cli_path(path) for path in args.checkpoint
    )
    manifest_base_dir = (
        _resolve_cli_path(args.manifest_base_dir)
        if args.manifest_base_dir is not None
        else None
    )
    config = EvaluationConfig(
        checkpoints=checkpoints,
        asvspoof_splits=tuple(splits) if splits is not None else None,
        manifest_path=(
            _resolve_cli_path(args.manifest)
            if args.manifest is not None
            else None
        ),
        manifest_base_dir=manifest_base_dir,
        split_name=_safe_split_name(args.split_name),
        source=args.source,
        data_root=_resolve_cli_path(args.data_root),
        image_root=_resolve_cli_path(args.image_root),
        output_root=_resolve_cli_path(args.output_root),
        output_dir=(
            _resolve_cli_path(args.output_dir)
            if args.output_dir is not None
            else None
        ),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        limit=args.limit,
        threshold_from=(
            _resolve_cli_path(args.threshold_from)
            if args.threshold_from is not None
            else None
        ),
        show_progress=not args.no_progress,
        command=shlex.join(
            [sys.executable, "-m", "spoof_detection.evaluate", *argv_list]
        ),
    )

    try:
        run_evaluation(config)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
