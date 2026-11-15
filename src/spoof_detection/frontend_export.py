"""Deterministic image export for the E1.1 front-end ablation.

The export mirrors the ``data/dataset_img`` layout so the existing evaluation
harness can read the new images unchanged:

    <frontend>/<split>/bonafide|spoofed/sample_<index>.png
    <frontend>/<split>/metadata_<split>.csv

Train uses the seed-42 manifest and ``sample_<sample_index>.png`` naming.  Dev
and eval use the CM protocol and ``sample_<protocol_row>.png`` naming, which is
what :mod:`spoof_detection.evaluate` expects.

Rendering is delegated to :mod:`spoof_detection.frontends`; this module only
handles layout, parallel execution, manifests and verification.
"""

from __future__ import annotations

import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .dataset import repo_relative, sha256_file
from .frontends import save_frontend_png
from .protocol import load_cm_protocol

EXPECTED_SPLIT_COUNTS: dict[str, dict[str, int]] = {
    "train": {"total": 10_080, "bonafide": 2_580, "spoof": 7_500},
    "dev": {"total": 24_844, "bonafide": 2_548, "spoof": 22_296},
    "eval": {"total": 71_237, "bonafide": 7_355, "spoof": 63_882},
}

TRAIN_MANIFEST_COLUMNS = (
    "sample_index",
    "protocol_row",
    "speaker_id",
    "system_id",
    "label",
    "audio_file_name",
    "source_flac",
)
EXPORT_MANIFEST_COLUMNS = (
    "file_index",
    "protocol_row",
    "speaker_id",
    "system_id",
    "label",
    "audio_file_name",
    "source_flac",
    "image_path",
    "sha256_png",
)


@dataclass(frozen=True)
class SplitExport:
    """One rendered split plus its on-disk metadata."""

    split: str
    frame: pd.DataFrame
    split_dir: Path
    metadata_path: Path

    @property
    def counts(self) -> dict[str, int]:
        labels = self.frame["label"].astype(int)
        return {
            "total": int(len(self.frame)),
            "bonafide": int((labels == 0).sum()),
            "spoof": int((labels == 1).sum()),
        }


def folder_for_label(label: int) -> str:
    """Return the export folder for a binary label."""
    return "bonafide" if int(label) == 0 else "spoofed"


def prepare_output_dir(
    out_dir: Path,
    derived_root: Path,
    force: bool,
) -> None:
    """Create ``out_dir``, refusing to clobber anything outside ``derived_root``."""
    resolved = Path(out_dir).resolve()
    derived = Path(derived_root).resolve()
    if not resolved.is_relative_to(derived) or resolved == derived:
        raise ValueError(
            f"refusing to write outside {derived_root}; got {out_dir}"
        )

    if out_dir.exists() and any(out_dir.iterdir()):
        if not force:
            raise FileExistsError(
                f"{out_dir} is not empty; pass --force to re-export it"
            )
        shutil.rmtree(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)


def build_split_records(
    split: str,
    repo_root: Path,
    train_manifest: Path,
    data_root: Path | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    """Return the records to render for one split.

    Train membership comes only from the seed-42 manifest (never the full
    25,380-image export).  Dev and eval come from the CM protocol in source
    order.  ``file_index`` is the suffix used in ``sample_<index>.png``.
    """
    repo_root = Path(repo_root)
    if split == "train":
        frame = pd.read_csv(train_manifest)
        missing = set(TRAIN_MANIFEST_COLUMNS) - set(frame.columns)
        if missing:
            raise ValueError(
                f"{train_manifest} is missing columns: {sorted(missing)}"
            )
        frame = frame[list(TRAIN_MANIFEST_COLUMNS)].copy()
        frame["file_index"] = frame["sample_index"].astype(int)
    elif split in {"dev", "eval"}:
        frame = load_cm_protocol(
            split,
            data_root=data_root,
            repo_root=repo_root,
        )
        frame = frame[
            [
                "protocol_row",
                "speaker_id",
                "system_id",
                "label",
                "audio_file_name",
                "source_flac",
            ]
        ].copy()
        frame["file_index"] = frame["protocol_row"].astype(int)
    else:
        raise ValueError(f"unknown split {split!r}; expected train, dev or eval")

    if limit is not None:
        frame = frame.head(int(limit)).copy()

    frame["source_path"] = [
        str((repo_root / value).resolve())
        if not Path(value).is_absolute()
        else str(Path(value))
        for value in frame["source_flac"]
    ]
    return frame.reset_index(drop=True)


def _render_one(task: tuple[str, str, str]) -> tuple[str, str]:
    """Worker: render one clip and return ``(destination, sha256)``."""
    frontend, source, destination = task
    save_frontend_png(frontend, source, destination)
    return destination, sha256_file(destination)


def export_split(
    records: pd.DataFrame,
    frontend: str,
    split: str,
    split_dir: Path,
    repo_root: Path,
    workers: int = 8,
    show_progress: bool = True,
) -> SplitExport:
    """Render every record in one split and return the export frame."""
    if records.empty:
        raise ValueError("records must not be empty")

    split_dir = Path(split_dir)
    (split_dir / "bonafide").mkdir(parents=True, exist_ok=True)
    (split_dir / "spoofed").mkdir(parents=True, exist_ok=True)

    tasks: list[tuple[str, str, str]] = []
    destinations: list[Path] = []
    for row in records.itertuples(index=False):
        folder = folder_for_label(int(row.label))
        destination = split_dir / folder / f"sample_{int(row.file_index)}.png"
        tasks.append((frontend, str(row.source_path), str(destination)))
        destinations.append(destination)

    chunksize = max(1, len(tasks) // (max(1, workers) * 4))
    outputs: list[tuple[str, str]] = []
    with ProcessPoolExecutor(max_workers=max(1, workers)) as pool:
        iterator = pool.map(_render_one, tasks, chunksize=chunksize)
        for output in tqdm(
            iterator,
            total=len(tasks),
            desc=f"{frontend}:{split}",
            unit="image",
            disable=not show_progress,
        ):
            outputs.append(output)

    if len(outputs) != len(tasks):
        raise RuntimeError(
            f"rendered {len(outputs)} images, expected {len(tasks)}"
        )

    frame = records.copy().reset_index(drop=True)
    frame["image_path"] = [
        repo_relative(Path(destination), repo_root)
        for destination, _digest in outputs
    ]
    frame["sha256_png"] = [digest for _destination, digest in outputs]

    missing = [
        path for path in destinations if not Path(path).is_file()
    ]
    if missing:
        raise RuntimeError(
            f"{len(missing)} rendered images are missing, e.g. {missing[:3]}"
        )

    metadata_path = split_dir / (
        "metadata_train_seed42.csv" if split == "train" else f"metadata_{split}.csv"
    )
    return SplitExport(
        split=split,
        frame=frame[list(EXPORT_MANIFEST_COLUMNS)],
        split_dir=split_dir,
        metadata_path=metadata_path,
    )


def metadata_frame(export: SplitExport) -> pd.DataFrame:
    """Return the four-column metadata CSV used by the image export."""
    return pd.DataFrame(
        {
            "speaker_id": export.frame["speaker_id"],
            "system_id": export.frame["system_id"],
            "label": export.frame["label"],
            "img_path": [
                f"./{path}" for path in export.frame["image_path"]
            ],
        }
    )


def write_metadata(export: SplitExport) -> Path:
    """Write the metadata CSV next to the images."""
    export.metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_frame(export).to_csv(export.metadata_path, index=False)
    return export.metadata_path


def write_manifest(export: SplitExport, manifest_path: Path) -> Path:
    """Write the machine-readable split manifest (used by training)."""
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    export.frame.to_csv(manifest_path, index=False)
    return manifest_path


def verify_mel128_parity(
    export: SplitExport,
    repo_root: Path,
    sample: int = 25,
) -> dict[str, object]:
    """Compare sampled exported images with the reference mel128 export.

    The reference is ``data/dataset_img/<split>/<folder>/sample_<protocol_row>.png``.
    Train images are named by ``sample_index`` but the reference uses
    ``protocol_row``, so the comparison uses the recorded protocol row.
    """
    import numpy as np
    from PIL import Image

    if sample <= 0:
        return {"checked": 0, "matched": 0, "mismatched": 0, "skipped": True}

    reference_root = Path(repo_root) / "data" / "dataset_img" / export.split
    rows = export.frame.head(int(sample))
    checked = 0
    matched = 0
    missing = 0
    mismatches: list[str] = []
    for row in rows.itertuples(index=False):
        folder = folder_for_label(int(row.label))
        generated = Path(repo_root) / row.image_path
        reference = (
            reference_root / folder / f"sample_{int(row.protocol_row)}.png"
        )
        if not reference.is_file() or not generated.is_file():
            missing += 1
            continue
        checked += 1
        with Image.open(generated) as image:
            generated_rgb = np.asarray(image.convert("RGB"))
        with Image.open(reference) as image:
            reference_rgb = np.asarray(image.convert("RGB"))
        if np.array_equal(generated_rgb, reference_rgb):
            matched += 1
        else:
            mismatches.append(str(row.image_path))

    return {
        "checked": checked,
        "matched": matched,
        "mismatched": len(mismatches),
        "missing_reference": missing,
        "mismatch_examples": mismatches[:5],
        "reference_root": repo_relative(reference_root, repo_root),
    }


def verify_determinism(
    export: SplitExport,
    frontend: str,
    sample: int = 1,
    records: pd.DataFrame | None = None,
) -> dict[str, object]:
    """Re-render sampled rows twice and compare PNG hashes.

    ``records`` must contain ``source_path`` and ``file_index``; it defaults to
    ``export.frame`` but the export frame omits ``source_path``, so callers that
    only have the written manifest can pass the pre-export records instead.
    """
    if sample <= 0:
        return {"checked": 0, "matched": 0, "skipped": True}

    source = records if records is not None else export.frame
    if source is None or source.empty:
        return {"checked": 0, "matched": 0, "skipped": True}
    if "source_path" not in source.columns:
        return {
            "checked": 0,
            "matched": 0,
            "skipped": True,
            "reason": "source_path column is required to re-render",
        }

    rows = source.head(int(sample))
    matched = 0
    details: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_root = Path(temp_dir)
        for index, row in enumerate(rows.itertuples(index=False)):
            first = temp_root / f"{index}_a.png"
            second = temp_root / f"{index}_b.png"
            save_frontend_png(frontend, row.source_path, first)
            save_frontend_png(frontend, row.source_path, second)
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
