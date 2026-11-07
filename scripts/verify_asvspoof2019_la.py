#!/usr/bin/env python3
"""Verify an ASVspoof 2019 LA installation.

This script does not download or extract anything.  It checks that an existing
``data/LA`` tree (and optionally the archive it came from) matches the layout,
counts, and protocol hashes documented in ``README.md``.  On success it writes
``manifests/asvspoof2019_la.meta.json``.

Examples::

    uv run python scripts/verify_asvspoof2019_la.py
    uv run python scripts/verify_asvspoof2019_la.py --archive data/downloads/LA.zip
    uv run python scripts/verify_asvspoof2019_la.py \
        --archive data/archive.zip --accept-any-archive
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from spoof_detection.dataset import (  # noqa: E402
    DEFAULT_DATA_ROOT,
    DEFAULT_MANIFEST_PATH,
    OFFICIAL_ARCHIVE_MD5,
    OFFICIAL_ARCHIVE_SIZE,
    OFFICIAL_DOI,
    ArchiveReport,
    DatasetReport,
    build_manifest,
    repo_relative,
    validate_archive,
    validate_dataset,
    write_manifest,
)

SPLIT_ORDER = ("train", "dev", "eval")


def resolve_repo_path(value: str | Path) -> Path:
    """Resolve a CLI path relative to the repository root."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def _print_split_table(report: DatasetReport | ArchiveReport) -> None:
    print(
        f"{'split':<8} {'protocol':>10} {'bonafide':>10} {'spoof':>10} "
        f"{'flac':>10} {'enroll':>8}  status"
    )
    for name in SPLIT_ORDER:
        split = report.splits.get(name)
        if split is None:
            continue
        status = "OK" if split.valid else "FAIL"
        print(
            f"{name:<8} {split.protocol_rows:>10,} {split.bonafide:>10,} "
            f"{split.spoof:>10,} {split.flac_files:>10,} "
            f"{split.enrollment_files:>8,}  {status}"
        )


def _print_problems(problems: list[str]) -> None:
    for problem in problems:
        print(f"  - {problem}")


def print_dataset_report(report: DatasetReport) -> None:
    print("ASVspoof 2019 LA dataset verification")
    print(f"data root: {repo_relative(report.root)}")
    print()
    _print_split_table(report)
    print()
    print(f"total files: {report.total_files:,}")
    print(f"total bytes: {report.total_bytes:,}")
    print(f"{'README.LA.txt':<16} sha256: {report.readme_sha256 or 'missing'}")
    print(f"{'LICENSE.txt':<16} sha256: {report.license_sha256 or 'missing'}")
    if report.problems:
        print()
        print("dataset problems:")
        _print_problems(report.problems)
        print("dataset validation: FAIL")
    else:
        print("dataset validation: PASS")


def print_archive_report(report: ArchiveReport) -> None:
    print("ASVspoof 2019 LA archive verification")
    print(f"archive: {repo_relative(report.path)}")
    print(f"size: {report.size_bytes:,} bytes")
    print(f"md5: {report.md5}")
    print(f"sha256: {report.sha256}")
    print(f"source: {report.source}")
    print(f"official LA.zip: {'yes' if report.is_official else 'no'}")
    print(f"official size/md5: {OFFICIAL_ARCHIVE_SIZE:,} / {OFFICIAL_ARCHIVE_MD5}")
    print()
    _print_split_table(report)
    print()
    print(f"archive files: {report.total_files:,}")
    print(f"archive uncompressed bytes: {report.total_bytes:,}")
    if report.warnings:
        print()
        print("archive warnings:")
        for warning in report.warnings:
            print(f"  ! {warning}")
    if report.problems:
        print()
        print("archive problems:")
        _print_problems(report.problems)
        print("archive validation: FAIL")
    else:
        print("archive validation: PASS")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Verify an existing ASVspoof 2019 LA installation against the "
            "project's expected layout and checksums."
        )
    )
    parser.add_argument(
        "--data-root",
        default=str(DEFAULT_DATA_ROOT),
        help="extracted dataset root (default: data/LA)",
    )
    parser.add_argument(
        "--archive",
        default=None,
        help="optional LA.zip-style archive to verify",
    )
    parser.add_argument(
        "--accept-any-archive",
        action="store_true",
        help=(
            "accept an archive whose size/MD5 differs from the official LA.zip; "
            "its extracted contents are still validated"
        ),
    )
    parser.add_argument(
        "--report",
        default=str(DEFAULT_MANIFEST_PATH),
        help="provenance manifest path (default: manifests/asvspoof2019_la.meta.json)",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="do not write the provenance manifest",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="only print the final result and any problems",
    )
    args = parser.parse_args(argv)

    data_root = resolve_repo_path(args.data_root)
    archive_path = resolve_repo_path(args.archive) if args.archive else None
    report_path = resolve_repo_path(args.report)

    if not data_root.exists() and archive_path is None:
        print(f"error: data root not found: {data_root}", file=sys.stderr)
        print(
            "Download ASVspoof 2019 LA and extract it to data/LA; "
            "see README.md for the exact commands.",
            file=sys.stderr,
        )
        return 2

    archive_report: ArchiveReport | None = None
    if archive_path is not None:
        if not archive_path.is_file():
            print(f"error: archive not found: {archive_path}", file=sys.stderr)
            return 2
        archive_report = validate_archive(
            archive_path,
            allow_unofficial=args.accept_any_archive,
        )
        if not args.quiet:
            print_archive_report(archive_report)
            print()
        if not archive_report.valid:
            if args.quiet:
                print("archive validation: FAIL", file=sys.stderr)
                _print_problems(archive_report.problems)
            return 1

    dataset_report: DatasetReport | None = None
    if data_root.exists():
        dataset_report = validate_dataset(data_root)
        if not args.quiet:
            print_dataset_report(dataset_report)
            print()
        if not dataset_report.valid:
            if args.quiet:
                print("dataset validation: FAIL", file=sys.stderr)
                _print_problems(dataset_report.problems)
            return 1
    else:
        if args.quiet:
            print("PASS")
        else:
            print(f"data root not found: {data_root}")
            print(
                "Archive verified. Extract it to data/LA and rerun without "
                "--archive to validate the dataset."
            )
        return 0

    if dataset_report is not None and not args.no_report:
        manifest = build_manifest(
            dataset_report,
            archive_report,
            command="uv run python "
            + " ".join(
                [repo_relative(Path(sys.argv[0])), *sys.argv[1:]]
            ),
        )
        write_manifest(report_path, manifest)
        if not args.quiet:
            print(f"manifest: {repo_relative(report_path)}")

    if args.quiet:
        print("PASS")
    else:
        print(f"DOI: {OFFICIAL_DOI}")
        print("result: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
