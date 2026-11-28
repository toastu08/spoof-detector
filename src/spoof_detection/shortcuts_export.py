"""Deterministic image export for the E1.4 shortcut ablation.

The export reuses the E1.1/E1.2 split layout so the frozen evaluation harness can
read the new images unchanged::

    <condition>/<split>/bonafide|spoofed/sample_<index>.png
    <condition>/<split>/metadata_<split>.csv

Every condition shares the same 128-band mel representation.  One worker decodes
the audio once, computes the mel **and** the speech mask once, then writes every
selected condition.  The mask (silence/noise shortcut) is the only variable; the
``full`` condition is the unchanged control.

Rendering is delegated to :mod:`spoof_detection.shortcuts`; this module only
handles layout, parallel execution, manifests and verification.
"""

from __future__ import annotations

import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from .dataset import repo_relative, sha256_file
from .frontend_export import (
    EXPORT_MANIFEST_COLUMNS,
    SplitExport,
    folder_for_label,
)
from .preprocess import load_fixed_audio
from .shortcuts import (
    _mel_db,
    get_shortcut_spec,
    save_shortcut_png,
    speech_mask,
)

# Export manifests carry two extra diagnostic columns on top of the shared
# EXPORT_MANIFEST_COLUMNS so degenerate clips (all-speech / all-non-speech) are
# visible in the record.
SHORTCUT_MANIFEST_COLUMNS = tuple(EXPORT_MANIFEST_COLUMNS) + (
    "n_speech_frames",
    "n_nonspeech_frames",
)


def _render_clip(
    task: tuple[str, tuple[tuple[str, str], ...]],
) -> tuple[int, int, list[tuple[str, str, str]]]:
    """Worker: decode one clip, compute the mel and mask once, render all conditions."""
    source, destinations = task
    y_fixed, sr = load_fixed_audio(source)
    spec_db = _mel_db(y_fixed, sr)
    mask = speech_mask(y_fixed, sr)
    n_speech = int(mask.sum())
    n_nonspeech = int((~mask).sum())
    rendered: list[tuple[str, str, str]] = []
    for condition, destination in destinations:
        save_shortcut_png(condition, spec_db, mask, destination)
        rendered.append((condition, destination, sha256_file(destination)))
    return n_speech, n_nonspeech, rendered


def _split_dir(output_root: Path, condition: str, split: str) -> Path:
    return output_root / condition / split


def export_shortcut_split(
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
        get_shortcut_spec(condition)

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
    speech_counts: list[int] = []
    nonspeech_counts: list[int] = []
    with ProcessPoolExecutor(max_workers=max(1, workers)) as pool:
        iterator = pool.map(_render_clip, tasks, chunksize=chunksize)
        for n_speech, n_nonspeech, task_output in iterator:
            speech_counts.append(n_speech)
            nonspeech_counts.append(n_nonspeech)
            for condition, destination, digest in task_output:
                rendered[condition].append((destination, digest))

    expected = len(tasks)
    if len(speech_counts) != expected:
        raise RuntimeError(
            f"{split}: rendered {len(speech_counts)} clips, expected {expected}"
        )
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
        frame["n_speech_frames"] = speech_counts
        frame["n_nonspeech_frames"] = nonspeech_counts
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
            frame=frame[list(SHORTCUT_MANIFEST_COLUMNS)],
            split_dir=split_dir,
            metadata_path=metadata_path,
        )
    return exports


def verify_shortcut_determinism(
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
            y_fixed, sr = load_fixed_audio(row.source_path)
            spec_db = _mel_db(y_fixed, sr)
            mask = speech_mask(y_fixed, sr)
            save_shortcut_png(condition, spec_db, mask, first)
            save_shortcut_png(condition, spec_db, mask, second)
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
