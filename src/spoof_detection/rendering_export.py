"""Deterministic image export for the E1.2 rendering ablation.

The export reuses the E1.1 split layout so the frozen evaluation harness can read
the new images unchanged::

    <condition>/<split>/bonafide|spoofed/sample_<index>.png
    <condition>/<split>/metadata_<split>.csv

The difference from :mod:`spoof_detection.frontend_export` is that every
condition shares the same 128-band mel representation, so one worker decodes the
audio and computes the mel **once** per clip, then writes every selected
rendering condition.  That keeps the eight-condition export close to the cost of
a single E1.1 front-end.

Rendering is delegated to :mod:`spoof_detection.rendering`; this module only
handles layout, parallel execution, manifests and verification.
"""

from __future__ import annotations

import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from .dataset import repo_relative, sha256_file
from .frontend_export import (
    EXPORT_MANIFEST_COLUMNS,
    SplitExport,
    folder_for_label,
)
from .preprocess import generate_mel_spectrogram
from .rendering import get_rendering_spec, save_rendering_png


def _render_clip(
    task: tuple[str, tuple[tuple[str, str], ...]],
) -> list[tuple[str, str, str]]:
    """Worker: decode one clip, compute the mel once, render every condition."""
    source, destinations = task
    spec_db = generate_mel_spectrogram(source)
    rendered: list[tuple[str, str, str]] = []
    for condition, destination in destinations:
        save_rendering_png(condition, spec_db, destination)
        rendered.append((condition, destination, sha256_file(destination)))
    return rendered


def _split_dir(output_root: Path, condition: str, split: str) -> Path:
    return output_root / condition / split


def export_rendering_split(
    records: pd.DataFrame,
    conditions: tuple[str, ...] | list[str],
    split: str,
    output_root: Path,
    repo_root: Path,
    workers: int = 8,
    show_progress: bool = True,
) -> dict[str, SplitExport]:
    """Render every condition for one split and return one export per condition."""
    if records.empty:
        raise ValueError("records must not be empty")
    conditions = tuple(conditions)
    if not conditions:
        raise ValueError("conditions must not be empty")
    if len(set(conditions)) != len(conditions):
        raise ValueError(f"duplicate conditions: {list(conditions)}")
    for condition in conditions:
        get_rendering_spec(condition)

    if "source_path" not in records.columns:
        raise ValueError("records must include a resolved source_path column")

    output_root = Path(output_root)
    split_dirs: dict[str, Path] = {}
    for condition in conditions:
        split_dir = _split_dir(output_root, condition, split)
        (split_dir / "bonafide").mkdir(parents=True, exist_ok=True)
        (split_dir / "spoofed").mkdir(parents=True, exist_ok=True)
        split_dirs[condition] = split_dir

    tasks: list[tuple[str, tuple[tuple[str, str], ...]]] = []
    for row in records.itertuples(index=False):
        folder = folder_for_label(int(row.label))
        destinations = tuple(
            (
                condition,
                str(
                    split_dirs[condition]
                    / folder
                    / f"sample_{int(row.file_index)}.png"
                ),
            )
            for condition in conditions
        )
        tasks.append((str(row.source_path), destinations))

    chunksize = max(1, len(tasks) // (max(1, workers) * 4))
    rendered: dict[str, list[tuple[str, str]]] = {
        condition: [] for condition in conditions
    }
    with ProcessPoolExecutor(max_workers=max(1, workers)) as pool:
        iterator = pool.map(_render_clip, tasks, chunksize=chunksize)
        for task_output in tqdm(
            iterator,
            total=len(tasks),
            desc=f"e1.2:{split}",
            unit="image",
            disable=not show_progress,
        ):
            for condition, destination, digest in task_output:
                rendered[condition].append((destination, digest))

    expected = len(tasks)
    for condition, outputs in rendered.items():
        if len(outputs) != expected:
            raise RuntimeError(
                f"{condition}: rendered {len(outputs)} images, expected {expected}"
            )

    exports: dict[str, SplitExport] = {}
    for condition in conditions:
        split_dir = split_dirs[condition]
        frame = records.copy().reset_index(drop=True)
        frame["image_path"] = [
            repo_relative(Path(destination), repo_root)
            for destination, _digest in rendered[condition]
        ]
        frame["sha256_png"] = [
            digest for _destination, digest in rendered[condition]
        ]
        missing = [
            destination
            for destination, _digest in rendered[condition]
            if not Path(destination).is_file()
        ]
        if missing:
            raise RuntimeError(
                f"{condition}: {len(missing)} rendered images are missing, "
                f"e.g. {missing[:3]}"
            )
        metadata_path = split_dir / (
            "metadata_train_seed42.csv"
            if split == "train"
            else f"metadata_{split}.csv"
        )
        exports[condition] = SplitExport(
            split=split,
            frame=frame[list(EXPORT_MANIFEST_COLUMNS)],
            split_dir=split_dir,
            metadata_path=metadata_path,
        )
    return exports


def _rgb_array(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def verify_rendering_determinism(
    condition: str,
    records: pd.DataFrame,
    sample: int = 1,
) -> dict[str, object]:
    """Re-render sampled clips twice for one condition and compare PNG hashes."""
    if sample <= 0:
        return {"checked": 0, "matched": 0, "skipped": True}
    if "source_path" not in records.columns:
        return {
            "checked": 0,
            "matched": 0,
            "skipped": True,
            "reason": "source_path column is required to re-render",
        }

    rows = records.head(int(sample))
    details: list[dict[str, object]] = []
    matched = 0
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_root = Path(temp_dir)
        for index, row in enumerate(rows.itertuples(index=False)):
            first = temp_root / f"{index}_a.png"
            second = temp_root / f"{index}_b.png"
            spec_db = generate_mel_spectrogram(row.source_path)
            save_rendering_png(condition, spec_db, first)
            save_rendering_png(condition, spec_db, second)
            same = sha256_file(first) == sha256_file(second)
            matched += int(same)
            details.append(
                {
                    "file_index": int(row.file_index),
                    "identical": same,
                }
            )
    return {
        "checked": len(rows),
        "matched": matched,
        "details": details,
    }


def verify_flip_pair_parity(
    flipped: SplitExport,
    unflipped: SplitExport,
    repo_root: Path,
    sample: int = 10,
) -> dict[str, object]:
    """Check that the unflipped PNG is the exact vertical flip of the flipped one."""
    if sample <= 0:
        return {"checked": 0, "matched": 0, "skipped": True}
    if flipped.split != unflipped.split:
        raise ValueError(
            f"split mismatch: {flipped.split!r} vs {unflipped.split!r}"
        )

    left = flipped.frame.head(int(sample))
    right = unflipped.frame.head(int(sample))
    if not np.array_equal(
        left["file_index"].to_numpy(), right["file_index"].to_numpy()
    ):
        raise ValueError("flip-pair frames are not aligned")

    matched = 0
    mismatches: list[str] = []
    for left_row, right_row in zip(
        left.itertuples(index=False), right.itertuples(index=False)
    ):
        flipped_rgb = _rgb_array(Path(repo_root) / left_row.image_path)
        unflipped_rgb = _rgb_array(Path(repo_root) / right_row.image_path)
        if np.array_equal(np.flipud(flipped_rgb), unflipped_rgb):
            matched += 1
        else:
            mismatches.append(str(left_row.image_path))
    return {
        "checked": len(left),
        "matched": matched,
        "mismatched": len(mismatches),
        "mismatch_examples": mismatches[:5],
    }


def verify_rgb_pair_parity(
    rgba_export: SplitExport,
    rgb_export: SplitExport,
    repo_root: Path,
    sample: int = 10,
) -> dict[str, object]:
    """Check that RGBA and RGB encodings decode to identical RGB pixels."""
    if sample <= 0:
        return {"checked": 0, "matched": 0, "skipped": True}
    if rgba_export.split != rgb_export.split:
        raise ValueError(
            f"split mismatch: {rgba_export.split!r} vs {rgb_export.split!r}"
        )

    left = rgba_export.frame.head(int(sample))
    right = rgb_export.frame.head(int(sample))
    if not np.array_equal(
        left["file_index"].to_numpy(), right["file_index"].to_numpy()
    ):
        raise ValueError("RGBA/RGB frames are not aligned")

    matched = 0
    mismatches: list[str] = []
    for left_row, right_row in zip(
        left.itertuples(index=False), right.itertuples(index=False)
    ):
        rgba_rgb = _rgb_array(Path(repo_root) / left_row.image_path)
        rgb_rgb = _rgb_array(Path(repo_root) / right_row.image_path)
        if np.array_equal(rgba_rgb, rgb_rgb):
            matched += 1
        else:
            mismatches.append(str(left_row.image_path))
    return {
        "checked": len(left),
        "matched": matched,
        "mismatched": len(mismatches),
        "mismatch_examples": mismatches[:5],
    }
