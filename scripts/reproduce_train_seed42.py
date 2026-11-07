#!/usr/bin/env python3
"""Reconstruct the first-iteration 10,080-clip training set (ROADMAP E0.1).

The script reproduces, in order:

1. the ASVspoof 2019 LA CM training protocol import,
2. the seed-42 spoof undersampling (22,800 -> 7,500) and seed-42 shuffle,
3. the 4-second, 128-band ``magma`` mel-spectrogram export,
4. the provenance manifest, hashes and pixel-parity verification.

It never writes into ``data/LA``, ``data/dataset_img`` or any checkpoint.  New
images go to ``data/derived/train_seed42/`` by default.

Usage::

    uv run python scripts/reproduce_train_seed42.py
    uv run python scripts/reproduce_train_seed42.py --force
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.image as mpimg
import numpy as np
import pandas as pd
from tqdm import tqdm

from spoof_detection.preprocess import (
    COLORMAP,
    N_MELS,
    SAMPLE_RATE,
    TARGET_SECONDS,
    generate_mel_spectrogram,
    save_mel_spectrogram_png,
)
from spoof_detection.protocol import (
    load_train_protocol,
    validate_train_protocol,
)
from spoof_detection.sampling import (
    DEFAULT_N_SPOOF,
    DEFAULT_SEED,
    build_seed42_subset,
    subset_summary,
    validate_seed42_subset,
)

DEFAULT_OUT_DIR = "data/derived/train_seed42"
DEFAULT_MANIFEST = "manifests/train_seed42.csv"
DEFAULT_REPORT_DIR = "reports/phase0/E0.1"
DEFAULT_REFERENCE_DIR = "data/dataset_img/train"

MANIFEST_COLUMNS = [
    "sample_index",
    "protocol_row",
    "speaker_id",
    "system_id",
    "label",
    "audio_file_name",
    "source_flac",
    "image_path",
    "sha256_flac",
    "sha256_png",
]


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Return the hex SHA-256 of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(repo_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip()


def git_state(repo_root: Path) -> dict[str, object]:
    """Return the current commit and whether the working tree is dirty."""
    commit = _git(repo_root, "rev-parse", "HEAD") or "unknown"
    status = _git(repo_root, "status", "--porcelain")
    return {"commit": commit, "dirty": bool(status)}


def resolve_path(repo_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def repo_relative(path: Path, repo_root: Path) -> str:
    return Path(path).resolve().relative_to(repo_root.resolve()).as_posix()


def prepare_output_dir(out_dir: Path, derived_root: Path, force: bool) -> None:
    """Create ``out_dir``, refusing to clobber anything outside ``data/derived``."""
    resolved = out_dir.resolve()
    if not resolved.is_relative_to(derived_root.resolve()) or resolved == derived_root.resolve():
        raise SystemExit(
            f"refusing to write outside {derived_root}; got --out-dir {out_dir}"
        )

    if out_dir.exists() and any(out_dir.iterdir()):
        if not force:
            raise SystemExit(
                f"{out_dir} is not empty; pass --force to regenerate it"
            )
        shutil.rmtree(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)


def export_images(subset: pd.DataFrame, out_dir: Path, repo_root: Path) -> pd.DataFrame:
    """Render every clip and return the manifest rows."""
    missing = [
        row.source_flac
        for row in subset.itertuples(index=False)
        if not (repo_root / row.source_flac).exists()
    ]
    if missing:
        raise SystemExit(
            f"{len(missing)} source FLAC files are missing, e.g. {missing[:3]}"
        )

    records: list[dict[str, object]] = []
    for row in tqdm(
        subset.itertuples(index=False),
        total=len(subset),
        desc="rendering mel spectrograms",
        unit="clip",
    ):
        source = repo_root / row.source_flac
        folder = "bonafide" if int(row.label) == 0 else "spoofed"
        destination = out_dir / folder / f"sample_{int(row.sample_index)}.png"

        spec_db = generate_mel_spectrogram(source)
        save_mel_spectrogram_png(spec_db, destination)

        records.append(
            {
                "sample_index": int(row.sample_index),
                "protocol_row": int(row.protocol_row),
                "speaker_id": row.speaker_id,
                "system_id": row.system_id,
                "label": int(row.label),
                "audio_file_name": row.audio_file_name,
                "source_flac": row.source_flac,
                "image_path": repo_relative(destination, repo_root),
                "sha256_flac": sha256_file(source),
                "sha256_png": sha256_file(destination),
            }
        )

    return pd.DataFrame.from_records(records, columns=MANIFEST_COLUMNS)


def check_pixel_parity(
    subset: pd.DataFrame,
    out_dir: Path,
    reference_dir: Path,
    repo_root: Path,
) -> pd.DataFrame:
    """Compare every generated image against the full-split reference export."""
    rows: list[dict[str, object]] = []
    for row in tqdm(
        subset.itertuples(index=False),
        total=len(subset),
        desc="verifying pixel parity",
        unit="image",
    ):
        folder = "bonafide" if int(row.label) == 0 else "spoofed"
        generated = out_dir / folder / f"sample_{int(row.sample_index)}.png"
        reference = (
            reference_dir / folder / f"sample_{int(row.protocol_row)}.png"
        )

        if not reference.exists():
            match = False
            detail = "reference missing"
        else:
            generated_pixels = mpimg.imread(generated)
            reference_pixels = mpimg.imread(reference)
            match = (
                generated_pixels.shape == reference_pixels.shape
                and np.array_equal(generated_pixels, reference_pixels)
            )
            detail = "" if match else "pixel mismatch"

        rows.append(
            {
                "sample_index": int(row.sample_index),
                "protocol_row": int(row.protocol_row),
                "image_path": repo_relative(generated, repo_root),
                "reference_path": repo_relative(reference, repo_root),
                "match": match,
                "detail": detail,
            }
        )

    return pd.DataFrame.from_records(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=None, help="repository root (auto-detected)")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="image output directory")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST, help="manifest CSV path")
    parser.add_argument("--report-dir", default=DEFAULT_REPORT_DIR, help="report directory")
    parser.add_argument(
        "--reference-dir",
        default=DEFAULT_REFERENCE_DIR,
        help="full-split export used for pixel-parity verification",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="regenerate a non-empty output directory (inside data/derived only)",
    )
    parser.add_argument(
        "--skip-parity",
        action="store_true",
        help="skip the 10,080-image pixel-parity check",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = (
        Path(args.repo_root).resolve() if args.repo_root else REPO_ROOT
    )
    out_dir = resolve_path(repo_root, args.out_dir)
    manifest_path = resolve_path(repo_root, args.manifest)
    report_dir = resolve_path(repo_root, args.report_dir)
    reference_dir = resolve_path(repo_root, args.reference_dir)
    derived_root = repo_root / "data" / "derived"

    protocol_path = (
        repo_root
        / "data"
        / "LA"
        / "ASVspoof2019_LA_cm_protocols"
        / "ASVspoof2019.LA.cm.train.trn.txt"
    )
    audio_dir = repo_root / "data" / "LA" / "ASVspoof2019_LA_train" / "flac"

    print(f"repository : {repo_root}")
    print(f"protocol   : {repo_relative(protocol_path, repo_root)}")

    protocol = load_train_protocol(
        protocol_path=protocol_path,
        audio_dir=audio_dir,
        repo_root=repo_root,
    )
    problems = validate_train_protocol(protocol)
    if problems:
        print("protocol validation failed:")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    subset = build_seed42_subset(protocol)
    problems = validate_seed42_subset(subset)
    if problems:
        print("subset validation failed (not writing anything):")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    summary = subset_summary(subset)
    print("subset valid:")
    print(f"  rows={summary['rows']} counts={summary['label_counts']}")
    print(f"  speakers={summary['speakers']} attacks={summary['attack_counts']}")

    prepare_output_dir(out_dir, derived_root, args.force)
    manifest = export_images(subset, out_dir, repo_root)

    if len(manifest) != summary["rows"]:
        print(
            f"export produced {len(manifest)} rows, expected {summary['rows']}"
        )
        return 1

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False)

    metadata_path = out_dir / "metadata_train_seed42.csv"
    metadata = manifest[["speaker_id", "system_id", "label", "image_path"]].rename(
        columns={"image_path": "img_path"}
    )
    metadata["img_path"] = metadata["img_path"].map(lambda value: f"./{value}")
    metadata.to_csv(metadata_path, index=False)

    report_dir.mkdir(parents=True, exist_ok=True)

    if args.skip_parity:
        parity_summary: dict[str, object] = {"skipped": True}
    else:
        parity = check_pixel_parity(subset, out_dir, reference_dir, repo_root)
        parity_path = report_dir / "pixel_parity.csv"
        parity.to_csv(parity_path, index=False)
        matched = int(parity["match"].sum())
        parity_summary = {
            "skipped": False,
            "checked": len(parity),
            "matched": matched,
            "mismatched": len(parity) - matched,
            "reference_dir": repo_relative(reference_dir, repo_root),
            "report": repo_relative(parity_path, repo_root),
        }

    meta = {
        "experiment": "E0.1",
        "title": "Reconstruct the 10,080-clip first-iteration training subset",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": git_state(repo_root),
        "sources": {
            "protocol": {
                "path": repo_relative(protocol_path, repo_root),
                "sha256": sha256_file(protocol_path),
            },
            "audio_dir": repo_relative(audio_dir, repo_root),
            "reference_dir": repo_relative(reference_dir, repo_root),
        },
        "recipe": {
            "bonafide": "all rows with label=bonafide, protocol order",
            "spoof": (
                f"DataFrame.sample(n={DEFAULT_N_SPOOF}, "
                f"random_state={DEFAULT_SEED}) on the spoof pool"
            ),
            "shuffle": (
                "pd.concat([bonafide, spoof_sample]).sample("
                f"frac=1, random_state={DEFAULT_SEED}).reset_index(drop=True)"
            ),
        },
        "params": {
            "seed": DEFAULT_SEED,
            "n_spoof": DEFAULT_N_SPOOF,
            "sample_rate": SAMPLE_RATE,
            "target_seconds": TARGET_SECONDS,
            "n_mels": N_MELS,
            "cmap": COLORMAP,
            "render": "plt.imsave(cmap='magma', origin='lower')",
            "power_to_db_ref": "np.max",
        },
        "counts": summary,
        "outputs": {
            "images": repo_relative(out_dir, repo_root),
            "manifest": repo_relative(manifest_path, repo_root),
            "manifest_sha256": sha256_file(manifest_path),
            "metadata": repo_relative(metadata_path, repo_root),
        },
        "parity": parity_summary,
        "environment": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "limitations": [
            "The original 10,080 PNGs no longer exist; pixel parity is verified "
            "against the later full-split export, not the lost originals.",
            "sha256_png is byte-level and may change with matplotlib/zlib "
            "versions; the pixel comparison is the authoritative check.",
        ],
    }

    try:
        import librosa

        meta["environment"]["librosa"] = librosa.__version__
    except ImportError:  # pragma: no cover - librosa is a hard dependency
        pass

    try:
        import matplotlib as mpl

        meta["environment"]["matplotlib"] = mpl.__version__
    except ImportError:  # pragma: no cover
        pass

    meta_path = manifest_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    print(f"images    : {repo_relative(out_dir, repo_root)}")
    print(f"manifest  : {repo_relative(manifest_path, repo_root)}")
    print(f"metadata  : {repo_relative(metadata_path, repo_root)}")
    print(f"meta      : {repo_relative(meta_path, repo_root)}")
    if parity_summary.get("skipped"):
        print("parity    : skipped")
    else:
        print(
            "parity    : "
            f"{parity_summary['matched']}/{parity_summary['checked']} pixel-identical"
        )
        if parity_summary["mismatched"]:
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
