#!/usr/bin/env python3
"""Run the E0.5 data and leakage audit.

The audit is read-only.  It verifies protocol alignment, speaker and filename
isolation, content-hash duplicates, training-manifest containment, enrollment
exclusions, and baseline threshold lineage, then writes a Markdown report and a
machine-readable JSON companion.

Examples::

    uv run python scripts/audit_data.py
    uv run python scripts/audit_data.py --skip-hashes
    uv run python scripts/audit_data.py --output-md /tmp/opencode/audit.md
"""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from spoof_detection.audit import (  # noqa: E402
    AuditConfig,
    run_audit,
    write_audit,
)
from spoof_detection.dataset import DEFAULT_DATA_ROOT, repo_relative  # noqa: E402


def resolve_repo_path(value: str | Path) -> Path:
    """Resolve a CLI path relative to the repository root."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def print_summary(report: dict[str, object]) -> None:
    """Print one line per check and the final conclusion."""
    print("E0.5 data and leakage audit")
    print()
    print(f"{'id':<6} {'status':<6} check")
    for check in report.get("checks") or []:
        print(
            f"{check.get('id'):<6} {str(check.get('status')).upper():<6} "
            f"{check.get('title')}"
        )
        if check.get("status") == "fail":
            print(f"       {check.get('detail')}")
    conclusion = report.get("conclusion") or {}
    print()
    print(f"conclusion: {str(conclusion.get('status')).upper()}")
    if conclusion.get("failures"):
        print(f"failed checks: {', '.join(conclusion['failures'])}")
    if conclusion.get("skipped"):
        print(f"skipped checks: {', '.join(conclusion['skipped'])}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Audit the ASVspoof 2019 LA data and baseline artifacts for "
            "train/dev/eval leakage (E0.5)."
        )
    )
    parser.add_argument(
        "--data-root",
        default=str(DEFAULT_DATA_ROOT),
        help="extracted ASVspoof 2019 LA root (default: data/LA)",
    )
    parser.add_argument(
        "--image-root",
        default=str(REPO_ROOT / "data" / "dataset_img"),
        help="spectrogram export root (default: data/dataset_img)",
    )
    parser.add_argument(
        "--derived-root",
        default=str(REPO_ROOT / "data" / "derived" / "train_seed42"),
        help="reconstructed training subset (default: data/derived/train_seed42)",
    )
    parser.add_argument(
        "--manifest",
        default=str(REPO_ROOT / "manifests" / "train_seed42.csv"),
        help="training manifest (default: manifests/train_seed42.csv)",
    )
    parser.add_argument(
        "--baseline-root",
        default=str(REPO_ROOT / "reports" / "baseline"),
        help="frozen baseline root (default: reports/baseline)",
    )
    parser.add_argument(
        "--freeze-report",
        default=str(REPO_ROOT / "reports" / "baseline" / "freeze_report.json"),
        help="E0.3 freeze report (default: reports/baseline/freeze_report.json)",
    )
    parser.add_argument(
        "--output-md",
        default=str(REPO_ROOT / "reports" / "audit" / "data_audit.md"),
        help="Markdown output (default: reports/audit/data_audit.md)",
    )
    parser.add_argument(
        "--output-json",
        default=str(REPO_ROOT / "reports" / "audit" / "data_audit.json"),
        help="JSON output (default: reports/audit/data_audit.json)",
    )
    parser.add_argument(
        "--skip-hashes",
        action="store_true",
        help=(
            "skip the SHA-256 pass over every audio and image file; structural "
            "checks still run"
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="hashing threads (default: 8)",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="print the summary without writing the report files",
    )
    args = parser.parse_args(argv)

    command = "uv run python " + shlex.join(
        [repo_relative(Path(__file__)), *sys.argv[1:]]
    )
    config = AuditConfig(
        data_root=resolve_repo_path(args.data_root),
        image_root=resolve_repo_path(args.image_root),
        derived_root=resolve_repo_path(args.derived_root),
        manifest_path=resolve_repo_path(args.manifest),
        baseline_root=resolve_repo_path(args.baseline_root),
        freeze_report_path=resolve_repo_path(args.freeze_report),
        hash_files=not args.skip_hashes,
        workers=args.workers,
        command=command,
    )

    try:
        report = run_audit(config)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print_summary(report)
    if not args.no_write:
        markdown_path, json_path = write_audit(
            report,
            resolve_repo_path(args.output_md),
            resolve_repo_path(args.output_json),
        )
        print()
        print(f"markdown: {repo_relative(markdown_path)}")
        print(f"json:     {repo_relative(json_path)}")

    return 0 if report["conclusion"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
