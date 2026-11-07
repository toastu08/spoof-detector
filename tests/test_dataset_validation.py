"""Fast checks for the ASVspoof 2019 LA dataset validator.

The tests use a miniature synthetic dataset so they do not require the 7.6 GB
ASVspoof download.
"""

from __future__ import annotations

import hashlib
import sys
import tempfile
import unittest
import zipfile
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from spoof_detection.dataset import (  # noqa: E402
    DatasetSpec,
    SplitSpec,
    hash_file,
    parse_cm_protocol,
    sha256_file,
    validate_archive,
    validate_dataset,
)

PROTOCOL_TEXT = (
    "LA_0001 LA_D_0000001 - - bonafide\n"
    "LA_0001 LA_D_0000002 - A01 spoof\n"
    "LA_0001 LA_D_0000003 - A01 spoof\n"
)
README_TEXT = "readme\n"
LICENSE_TEXT = "license\n"
FLAC_NAMES = (
    "LA_D_0000001",
    "LA_D_0000002",
    "LA_D_0000003",
    "LA_D_A0000001",
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _build_mini_tree(root: Path) -> DatasetSpec:
    """Create a tiny valid dataset and return its matching spec."""
    _write(root / "README.LA.txt", README_TEXT)
    _write(
        root / "ASVspoof2019_LA_cm_protocols"
        / "ASVspoof2019.LA.cm.dev.trl.txt",
        PROTOCOL_TEXT,
    )
    (root / "ASVspoof2019_LA_asv_protocols").mkdir(parents=True, exist_ok=True)
    (root / "ASVspoof2019_LA_asv_scores").mkdir(parents=True, exist_ok=True)
    _write(root / "ASVspoof2019_LA_dev" / "LICENSE.txt", LICENSE_TEXT)
    for name in FLAC_NAMES:
        _write(root / "ASVspoof2019_LA_dev" / "flac" / f"{name}.flac", name)

    files = [path for path in root.rglob("*") if path.is_file()]
    total_files = len(files)
    total_bytes = sum(path.stat().st_size for path in files)
    protocol_sha256 = sha256_file(
        root / "ASVspoof2019_LA_cm_protocols"
        / "ASVspoof2019.LA.cm.dev.trl.txt"
    )

    return DatasetSpec(
        name="mini ASVspoof 2019 LA",
        splits=(
            SplitSpec(
                name="dev",
                dirname="ASVspoof2019_LA_dev",
                protocol_name="ASVspoof2019.LA.cm.dev.trl.txt",
                protocol_rows=3,
                bonafide=1,
                spoof=2,
                flac_files=4,
                enrollment_files=1,
                protocol_sha256=protocol_sha256,
                enrollment_prefix="LA_D_A",
            ),
        ),
        cm_protocols_dirname="ASVspoof2019_LA_cm_protocols",
        asv_protocols_dirname="ASVspoof2019_LA_asv_protocols",
        asv_scores_dirname="ASVspoof2019_LA_asv_scores",
        asv_protocols_count=0,
        asv_scores_count=0,
        readme_name="README.LA.txt",
        readme_sha256=hashlib.sha256(README_TEXT.encode()).hexdigest(),
        license_name="LICENSE.txt",
        license_sha256=hashlib.sha256(LICENSE_TEXT.encode()).hexdigest(),
        total_files=total_files,
        total_bytes=total_bytes,
        archive_size=0,
        archive_md5="0" * 32,
    )


def _build_mini_zip(path: Path, root: Path) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for source in sorted(root.rglob("*")):
            if not source.is_file():
                continue
            arcname = "LA/" + source.relative_to(root).as_posix()
            archive.write(source, arcname=arcname)


class ProtocolParsingTest(unittest.TestCase):
    def test_parses_counts_and_names(self) -> None:
        summary = parse_cm_protocol(PROTOCOL_TEXT)
        self.assertEqual(summary.rows, 3)
        self.assertEqual(summary.bonafide, 1)
        self.assertEqual(summary.spoof, 2)
        self.assertEqual(
            summary.names,
            frozenset(
                {"LA_D_0000001", "LA_D_0000002", "LA_D_0000003"}
            ),
        )

    def test_rejects_wrong_column_count(self) -> None:
        with self.assertRaises(ValueError):
            parse_cm_protocol("LA_0001 LA_D_0000001 - - bonafide extra\n")

    def test_rejects_unknown_label(self) -> None:
        with self.assertRaises(ValueError):
            parse_cm_protocol("LA_0001 LA_D_0000001 - - fake\n")

    def test_rejects_duplicate_audio_names(self) -> None:
        duplicate = (
            "LA_0001 LA_D_0000001 - - bonafide\n"
            "LA_0001 LA_D_0000001 - A01 spoof\n"
        )
        with self.assertRaises(ValueError):
            parse_cm_protocol(duplicate)


class DatasetValidationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name) / "LA"
        self.root.mkdir()
        self.spec = _build_mini_tree(self.root)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_accepts_valid_tree(self) -> None:
        report = validate_dataset(self.root, self.spec)
        self.assertEqual(report.problems, [])
        self.assertTrue(report.valid)
        self.assertEqual(report.total_files, self.spec.total_files)
        self.assertEqual(report.total_bytes, self.spec.total_bytes)
        self.assertTrue(report.splits["dev"].valid)

    def test_rejects_missing_protocol_flac(self) -> None:
        (
            self.root
            / "ASVspoof2019_LA_dev"
            / "flac"
            / "LA_D_0000002.flac"
        ).unlink()
        report = validate_dataset(self.root, self.spec)
        self.assertFalse(report.valid)
        self.assertTrue(
            any("missing" in problem for problem in report.problems),
            report.problems,
        )
        self.assertFalse(report.splits["dev"].valid)

    def test_rejects_unexpected_flac(self) -> None:
        _write(
            self.root / "ASVspoof2019_LA_dev" / "flac" / "LA_D_9999999.flac",
            "extra",
        )
        report = validate_dataset(self.root, self.spec)
        self.assertFalse(report.valid)
        self.assertTrue(
            any("unexpected FLAC" in problem for problem in report.problems),
            report.problems,
        )

    def test_rejects_bad_protocol_hash(self) -> None:
        bad_spec = replace(
            self.spec,
            splits=(
                replace(self.spec.splits[0], protocol_sha256="0" * 64),
            ),
        )
        report = validate_dataset(self.root, bad_spec)
        self.assertFalse(report.valid)
        self.assertTrue(
            any("SHA-256 mismatch" in problem for problem in report.problems),
            report.problems,
        )

    def test_rejects_wrong_total_file_count(self) -> None:
        bad_spec = replace(
            self.spec,
            total_files=(self.spec.total_files or 0) + 1,
        )
        report = validate_dataset(self.root, bad_spec)
        self.assertFalse(report.valid)
        self.assertTrue(
            any("files" in problem for problem in report.problems),
            report.problems,
        )

    def test_rejects_missing_readme(self) -> None:
        (self.root / "README.LA.txt").unlink()
        report = validate_dataset(self.root, self.spec)
        self.assertFalse(report.valid)
        self.assertTrue(
            any("README.LA.txt" in problem for problem in report.problems),
            report.problems,
        )


class ArchiveValidationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        base = Path(self.temp_dir.name)
        self.root = base / "LA"
        self.root.mkdir()
        self.spec = _build_mini_tree(self.root)
        self.archive = base / "mini.zip"
        _build_mini_zip(self.archive, self.root)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_accepts_unofficial_archive_when_allowed(self) -> None:
        report = validate_archive(
            self.archive,
            self.spec,
            allow_unofficial=True,
        )
        self.assertEqual(report.problems, [])
        self.assertTrue(report.valid)
        self.assertFalse(report.is_official)
        self.assertTrue(report.warnings)
        self.assertEqual(report.total_files, self.spec.total_files)
        self.assertEqual(report.total_bytes, self.spec.total_bytes)

    def test_rejects_unofficial_archive_by_default(self) -> None:
        report = validate_archive(
            self.archive,
            self.spec,
            allow_unofficial=False,
        )
        self.assertFalse(report.valid)
        self.assertTrue(
            any("official LA.zip" in problem for problem in report.problems),
            report.problems,
        )

    def test_rejects_unsafe_member(self) -> None:
        unsafe = Path(self.temp_dir.name) / "unsafe.zip"
        with zipfile.ZipFile(unsafe, "w") as archive:
            archive.writestr("LA/../evil.txt", "evil")
        report = validate_archive(unsafe, self.spec, allow_unofficial=True)
        self.assertFalse(report.valid)
        self.assertTrue(
            any("unsafe path" in problem for problem in report.problems),
            report.problems,
        )

    def test_rejects_member_outside_la_root(self) -> None:
        outside = Path(self.temp_dir.name) / "outside.zip"
        with zipfile.ZipFile(outside, "w") as archive:
            archive.writestr("evil.txt", "evil")
        report = validate_archive(outside, self.spec, allow_unofficial=True)
        self.assertFalse(report.valid)
        self.assertTrue(
            any("outside the expected LA/ root" in problem for problem in report.problems),
            report.problems,
        )

    def test_rejects_missing_protocol(self) -> None:
        missing = Path(self.temp_dir.name) / "missing.zip"
        with zipfile.ZipFile(missing, "w") as archive:
            for source in sorted(self.root.rglob("*")):
                if not source.is_file():
                    continue
                if source.name == "ASVspoof2019.LA.cm.dev.trl.txt":
                    continue
                arcname = "LA/" + source.relative_to(self.root).as_posix()
                archive.write(source, arcname=arcname)
        report = validate_archive(missing, self.spec, allow_unofficial=True)
        self.assertFalse(report.valid)
        self.assertTrue(
            any("missing protocol" in problem for problem in report.problems),
            report.problems,
        )


class HashHelperTest(unittest.TestCase):
    def test_hash_file_matches_hashlib(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sample.bin"
            payload = b"spoof detection"
            path.write_bytes(payload)
            digests = hash_file(path, ("md5", "sha256"))
            self.assertEqual(digests["md5"], hashlib.md5(payload).hexdigest())
            self.assertEqual(
                digests["sha256"],
                hashlib.sha256(payload).hexdigest(),
            )


if __name__ == "__main__":
    unittest.main()
