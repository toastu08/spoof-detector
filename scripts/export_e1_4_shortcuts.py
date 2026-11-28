#!/usr/bin/env python3
"""Export the E1.4 shortcut image sets (ROADMAP E1.4).

Every condition shares the 128-band mel representation; only whether speech,
non-speech, or neither is masked changes.  Images go to

    data/derived/e1_4_shortcuts/<condition>/{train,dev,eval}/

and per-condition manifests to ``manifests/e1_4_shortcuts/``.  The script never
writes into ``data/LA``, ``data/dataset_img``, or any earlier experiment's
export.

Usage::

    uv run python scripts/export_e1_4_shortcuts.py
    uv run python scripts/export_e1_4_shortcuts.py --conditions full --splits dev
    uv run python scripts/export_e1_4_shortcuts.py --limit 100 --force
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

os.environ.setdefault("MPLBACKEND", "Agg")

from spoof_detection.dataset import repo_relative, sha256_file  # noqa: E402
from spoof_detection.frontend_export import (  # noqa: E402
    EXPECTED_SPLIT_COUNTS,
    build_split_records,
    prepare_output_dir,
    verify_mel128_parity,
    write_manifest,
    write_metadata,
)
from spoof_detection.shortcuts import (  # noqa: E402
    SHORTCUT_ORDER,
    shortcut_payload,
)
from spoof_detection.shortcuts_export import (  # noqa: E402
    export_shortcut_split,
    verify_shortcut_determinism,
)

DEFAULT_OUTPUT_ROOT = "data/derived/e1_4_shortcuts"
DEFAULT_MANIFEST_ROOT = "manifests/e1_4_shortcuts"
DEFAULT_REPORT_PATH = "reports/phase1/e1_4_shortcuts/export_manifest.json"
DEFAULT_TRAIN_MANIFEST = "manifests/train_seed42.csv"
DEFAULT_SPLITS = ("train", "dev", "eval")


def _jsonable(value: object) -> object:
    """Convert numpy/pandas scalars and paths for JSON output."""
    import numpy as np
    import pandas as pd

    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if value is pd.NA:
        return None
    return value


def _git_state(repo_root: Path) -> dict[str, object]:
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        status = subprocess.check_output(
            ["git", "-C", str(repo_root), "status", "--porcelain"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return {"commit": commit, "dirty": bool(status)}
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        return {"commit": None, "dirty": None, "error": str(exc)}


def _environment() -> dict[str, object]:
    import librosa
    import matplotlib
    import numpy as np
    import pandas as pd

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "librosa": librosa.__version__,
        "matplotlib": matplotlib.__version__,
    }


def _resolve(value: str | Path, repo_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (repo_root / path).resolve()


def _add_limit_suffix(path: Path, limit: int | None) -> Path:
    if limit is None:
        return path
    return path.with_name(f"{path.stem}_partial{limit}{path.suffix}")


def _check_parity(report: dict[str, object]) -> bool:
    if report.get("skipped"):
        return True
    if report.get("missing_reference"):
        return False
    checked = int(report.get("checked") or 0)
    matched = int(report.get("matched") or 0)
    return checked > 0 and matched == checked


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=SHORTCUT_ORDER,
        default=list(SHORTCUT_ORDER),
        help="shortcut conditions to export (default: all three)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=DEFAULT_SPLITS,
        default=list(DEFAULT_SPLITS),
        help="splits to export (default: train dev eval)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(12, (os.cpu_count() or 4) - 2)),
        help="parallel rendering workers (default: CPU count - 2, max 12)",
    )
    parser.add_argument(
        "--output-root",
        default=DEFAULT_OUTPUT_ROOT,
        help=f"derived image root (default: {DEFAULT_OUTPUT_ROOT})",
    )
    parser.add_argument(
        "--manifest-root",
        default=DEFAULT_MANIFEST_ROOT,
        help=f"per-condition manifest root (default: {DEFAULT_MANIFEST_ROOT})",
    )
    parser.add_argument(
        "--report-path",
        default=DEFAULT_REPORT_PATH,
        help=f"export report path (default: {DEFAULT_REPORT_PATH})",
    )
    parser.add_argument(
        "--train-manifest",
        default=DEFAULT_TRAIN_MANIFEST,
        help=f"seed-42 training manifest (default: {DEFAULT_TRAIN_MANIFEST})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="smoke only: render the first N rows per split into *_partialN roots",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-export non-empty split directories inside data/derived",
    )
    parser.add_argument(
        "--verify-parity",
        type=int,
        default=25,
        help="images per parity check (0 disables; default: 25)",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="disable the tqdm progress bar",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = REPO_ROOT
    derived_root = repo_root / "data" / "derived"
    output_root = _add_limit_suffix(
        _resolve(args.output_root, repo_root), args.limit
    )
    manifest_root = _add_limit_suffix(
        _resolve(args.manifest_root, repo_root), args.limit
    )
    report_path = _add_limit_suffix(
        _resolve(args.report_path, repo_root), args.limit
    )
    train_manifest = _resolve(args.train_manifest, repo_root)

    if args.limit is not None and args.limit < 1:
        print("error: --limit must be at least 1", file=sys.stderr)
        return 2
    if args.verify_parity < 0:
        print("error: --verify-parity must be >= 0", file=sys.stderr)
        return 2
    if not train_manifest.is_file():
        print(
            f"error: training manifest not found: {train_manifest}",
            file=sys.stderr,
        )
        return 2

    started = time.perf_counter()
    command = shlex.join([sys.executable, *sys.argv])
    payload: dict[str, object] = {
        "experiment": "E1.4",
        "title": "Is the model using spoofing cues or shortcuts?",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        ),
        "command": command,
        "git": _git_state(repo_root),
        "environment": _environment(),
        "sources": {
            "train_manifest": repo_relative(train_manifest, repo_root),
            "train_manifest_sha256": sha256_file(train_manifest),
            "data_root": repo_relative(repo_root / "data" / "LA", repo_root),
        },
        "output_root": repo_relative(output_root, repo_root),
        "manifest_root": repo_relative(manifest_root, repo_root),
        "limit": args.limit,
        "conditions": {
            condition: shortcut_payload(condition)
            for condition in args.conditions
        },
        "splits": {},
    }

    for split in args.splits:
        print(f"\n[e1.4:{split}] preparing {len(args.conditions)} conditions")
        for condition in args.conditions:
            split_dir = output_root / condition / split
            prepare_output_dir(split_dir, derived_root, args.force)

        records = build_split_records(
            split,
            repo_root=repo_root,
            train_manifest=train_manifest,
            data_root=repo_root / "data" / "LA",
            limit=args.limit,
        )
        if args.limit is None:
            expected = EXPECTED_SPLIT_COUNTS[split]
            labels = records["label"].astype(int)
            observed = {
                "total": int(len(records)),
                "bonafide": int((labels == 0).sum()),
                "spoof": int((labels == 1).sum()),
            }
            if observed != expected:
                print(
                    f"error: {split} counts {observed} != expected {expected}",
                    file=sys.stderr,
                )
                return 1

        exports = export_shortcut_split(
            records,
            conditions=tuple(args.conditions),
            split=split,
            output_root=output_root,
            repo_root=repo_root,
            workers=args.workers,
            show_progress=not args.no_progress,
        )

        split_payload: dict[str, object] = {
            "counts": exports[args.conditions[0]].counts,
            "conditions": {},
            "control_parity": {"skipped": True},
        }
        for condition in args.conditions:
            export = exports[condition]
            metadata_path = write_metadata(export)
            manifest_path = write_manifest(
                export, manifest_root / f"{condition}_{split}.csv"
            )
            determinism = verify_shortcut_determinism(
                condition, records, sample=1
            )
            split_payload["conditions"][condition] = {  # type: ignore[index]
                "counts": export.counts,
                "images_dir": repo_relative(export.split_dir, repo_root),
                "metadata": repo_relative(metadata_path, repo_root),
                "metadata_sha256": sha256_file(metadata_path),
                "manifest": repo_relative(manifest_path, repo_root),
                "manifest_sha256": sha256_file(manifest_path),
                "sample_sha256": {
                    str(int(row.file_index)): row.sha256_png
                    for row in export.frame.head(5).itertuples(index=False)
                },
                "determinism": determinism,
            }
            print(
                f"  {condition}: {export.counts['total']} images "
                f"({export.counts['bonafide']} / {export.counts['spoof']})"
            )
            if determinism.get("checked") and (
                determinism["matched"] != determinism["checked"]
            ):
                print(
                    f"error: determinism failed for {condition}",
                    file=sys.stderr,
                )
                return 1

        if "full" in exports:
            control_parity = verify_mel128_parity(
                exports["full"],
                repo_root=repo_root,
                sample=args.verify_parity,
            )
            split_payload["control_parity"] = control_parity
            print(
                "  control parity: "
                f"{control_parity.get('matched')}/{control_parity.get('checked')} "
                "pixel-identical to data/dataset_img"
            )
            if not _check_parity(control_parity):
                print(
                    f"error: control parity failed: {control_parity}",
                    file=sys.stderr,
                )
                return 1

        payload["splits"][split] = split_payload  # type: ignore[index]

    payload["elapsed_seconds"] = round(time.perf_counter() - started, 2)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(_jsonable(payload), indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"\nwrote {repo_relative(output_root, repo_root)}")
    print(f"wrote {repo_relative(report_path, repo_root)}")
    print(f"elapsed {payload['elapsed_seconds']} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
