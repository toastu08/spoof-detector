"""Tests for the E0.5 data and leakage audit."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from spoof_detection.audit import (  # noqa: E402
    AuditConfig,
    HashCache,
    check_baseline_artifacts,
    check_baseline_score_alignment,
    check_baseline_threshold_lineage,
    check_duplicate_hashes,
    check_enrollment_exclusions,
    check_manifest_files,
    check_manifest_integrity,
    check_manifest_isolation,
    check_split_alignment,
    check_split_isolation,
    load_manifest_audit,
    render_markdown,
    run_audit,
    write_audit,
)
from spoof_detection.metrics import compute_eer  # noqa: E402
from spoof_detection.protocol import SPLIT_SPECS  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_bytes(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def protocol_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "protocol_row": range(len(rows)),
            "speaker_id": [str(row["speaker_id"]) for row in rows],
            "audio_file_name": [str(row["audio_file_name"]) for row in rows],
            "system_id": [str(row["system_id"]) for row in rows],
            "label": [int(row["label"]) for row in rows],
            "source_flac": [str(row["source_flac"]) for row in rows],
        }
    )


def write_protocol_files(root: Path, split: str, rows: list[dict[str, object]]) -> Path:
    spec = SPLIT_SPECS[split]
    protocol_path = root / "ASVspoof2019_LA_cm_protocols" / spec.protocol_filename
    protocol_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"{row['speaker_id']} {row['audio_file_name']} - {row['system_id']} "
        f"{'bonafide' if int(row['label']) == 0 else 'spoof'}"
        for row in rows
    ]
    protocol_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    flac_dir = root / spec.split_dirname / spec.flac_dirname
    flac_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        write_bytes(flac_dir / f"{row['audio_file_name']}.flac", b"audio")
    return protocol_path


def write_metadata(
    image_root: Path,
    split: str,
    rows: list[dict[str, object]],
) -> Path:
    records = []
    for index, row in enumerate(rows):
        label = int(row["label"])
        folder = "bonafide" if label == 0 else "spoofed"
        path = image_root / split / folder / f"sample_{index}.png"
        write_bytes(path, f"image-{split}-{index}".encode())
        records.append(
            {
                "speaker_id": row["speaker_id"],
                "system_id": row["system_id"],
                "label": label,
                "img_path": str(path),
            }
        )
    metadata_path = image_root / split / f"metadata_{split}.csv"
    pd.DataFrame(records).to_csv(metadata_path, index=False)
    return metadata_path


def make_manifest(
    tmp: Path,
    protocol: pd.DataFrame,
    image_root: Path,
) -> tuple[Path, pd.DataFrame]:
    derived_root = image_root.parent / "derived"
    records = []
    for index, row in protocol.iterrows():
        label = int(row["label"])
        folder = "bonafide" if label == 0 else "spoofed"
        image_path = write_bytes(
            derived_root / folder / f"sample_{index}.png",
            f"derived-{index}".encode(),
        )
        audio_path = Path(str(row["source_flac"]))
        if not audio_path.is_file():
            write_bytes(audio_path, f"audio-{index}".encode())
        records.append(
            {
                "sample_index": index,
                "protocol_row": int(row["protocol_row"]),
                "speaker_id": row["speaker_id"],
                "system_id": row["system_id"],
                "label": label,
                "audio_file_name": row["audio_file_name"],
                "source_flac": str(audio_path),
                "image_path": str(image_path),
                "sha256_flac": sha256(audio_path),
                "sha256_png": sha256(image_path),
            }
        )
    frame = pd.DataFrame(records)
    manifest_path = tmp / "manifest.csv"
    frame.to_csv(manifest_path, index=False)
    return manifest_path, frame


class SplitAlignmentTest(unittest.TestCase):
    def _fixture(self, tmp: Path) -> tuple[Path, Path, pd.DataFrame]:
        data_root = tmp / "LA"
        image_root = tmp / "images"
        rows = [
            {
                "speaker_id": "LA_1",
                "audio_file_name": "LA_T_1",
                "system_id": "-",
                "label": 0,
            },
            {
                "speaker_id": "LA_2",
                "audio_file_name": "LA_T_2",
                "system_id": "A01",
                "label": 1,
            },
        ]
        write_metadata(image_root, "train", rows)
        protocol = protocol_frame(
            [
                {**row, "source_flac": str(data_root / f"{row['audio_file_name']}.flac")}
                for row in rows
            ]
        )
        return data_root, image_root, protocol

    def test_alignment_passes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            data_root, image_root, protocol = self._fixture(tmp)
            result = check_split_alignment(
                "train",
                data_root,
                image_root,
                protocol=protocol,
                strict_counts=False,
            )
            self.assertEqual(result.status, "pass", result.detail)
            self.assertEqual(result.evidence["image_files"], 2)

    def test_detects_label_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            data_root, image_root, protocol = self._fixture(tmp)
            metadata_path = image_root / "train" / "metadata_train.csv"
            metadata = pd.read_csv(metadata_path)
            metadata.loc[0, "label"] = 1
            metadata.to_csv(metadata_path, index=False)
            result = check_split_alignment(
                "train",
                data_root,
                image_root,
                protocol=protocol,
                strict_counts=False,
            )
            self.assertEqual(result.status, "fail")
            self.assertIn("label", result.detail)

    def test_detects_missing_image(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            data_root, image_root, protocol = self._fixture(tmp)
            (image_root / "train" / "bonafide" / "sample_0.png").unlink()
            result = check_split_alignment(
                "train",
                data_root,
                image_root,
                protocol=protocol,
                strict_counts=False,
            )
            self.assertEqual(result.status, "fail")
            self.assertIn("missing", result.detail)

    def test_detects_orphan_image(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            data_root, image_root, protocol = self._fixture(tmp)
            write_bytes(image_root / "train" / "bonafide" / "sample_9.png", b"x")
            result = check_split_alignment(
                "train",
                data_root,
                image_root,
                protocol=protocol,
                strict_counts=False,
            )
            self.assertEqual(result.status, "fail")
            self.assertIn("not referenced", result.detail)

    def test_detects_filename_index_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            data_root, image_root, protocol = self._fixture(tmp)
            metadata_path = image_root / "train" / "metadata_train.csv"
            metadata = pd.read_csv(metadata_path)
            metadata.loc[1, "img_path"] = metadata.loc[0, "img_path"]
            metadata.to_csv(metadata_path, index=False)
            result = check_split_alignment(
                "train",
                data_root,
                image_root,
                protocol=protocol,
                strict_counts=False,
            )
            self.assertEqual(result.status, "fail")


class SplitIsolationTest(unittest.TestCase):
    def test_disjoint_splits_pass(self) -> None:
        protocols = {
            "train": protocol_frame(
                [
                    {"speaker_id": "T1", "audio_file_name": "LA_T_1", "system_id": "A01", "label": 1, "source_flac": "x"},
                ]
            ),
            "dev": protocol_frame(
                [
                    {"speaker_id": "D1", "audio_file_name": "LA_D_1", "system_id": "A01", "label": 1, "source_flac": "x"},
                ]
            ),
            "eval": protocol_frame(
                [
                    {"speaker_id": "E1", "audio_file_name": "LA_E_1", "system_id": "A07", "label": 1, "source_flac": "x"},
                ]
            ),
        }
        results = check_split_isolation(protocols, expected_attacks={})
        self.assertTrue(all(result.passed for result in results), results)

    def test_detects_speaker_overlap(self) -> None:
        protocols = {
            "train": protocol_frame(
                [{"speaker_id": "X", "audio_file_name": "LA_T_1", "system_id": "A01", "label": 1, "source_flac": "x"}]
            ),
            "dev": protocol_frame(
                [{"speaker_id": "X", "audio_file_name": "LA_D_1", "system_id": "A01", "label": 1, "source_flac": "x"}]
            ),
            "eval": protocol_frame(
                [{"speaker_id": "E", "audio_file_name": "LA_E_1", "system_id": "A07", "label": 1, "source_flac": "x"}]
            ),
        }
        isolation, _ = check_split_isolation(protocols)
        self.assertEqual(isolation.status, "fail")
        self.assertIn("speaker overlap", isolation.detail)

    def test_detects_filename_overlap(self) -> None:
        protocols = {
            "train": protocol_frame(
                [{"speaker_id": "T", "audio_file_name": "LA_T_1", "system_id": "A01", "label": 1, "source_flac": "x"}]
            ),
            "dev": protocol_frame(
                [{"speaker_id": "D", "audio_file_name": "LA_T_1", "system_id": "A01", "label": 1, "source_flac": "x"}]
            ),
            "eval": protocol_frame(
                [{"speaker_id": "E", "audio_file_name": "LA_E_1", "system_id": "A07", "label": 1, "source_flac": "x"}]
            ),
        }
        isolation, _ = check_split_isolation(protocols)
        self.assertEqual(isolation.status, "fail")
        self.assertIn("filename overlap", isolation.detail)


class DuplicateHashTest(unittest.TestCase):
    def test_cross_split_duplicate_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            payload = b"same audio"
            train_file = write_bytes(tmp / "train" / "a.flac", payload)
            dev_file = write_bytes(tmp / "dev" / "b.flac", payload)
            hashes = HashCache(workers=1)
            results = check_duplicate_hashes(
                {"train": [train_file], "dev": [dev_file], "eval": []},
                {"train": [], "dev": [], "eval": []},
                hashes,
            )
            audio = results[0]
            self.assertEqual(audio.status, "fail")
            self.assertIn("more than one split", audio.detail)

    def test_explained_within_split_duplicate_passes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            payload = b"same train image"
            first = write_bytes(tmp / "dataset_img" / "train" / "a.png", payload)
            second = write_bytes(tmp / "derived" / "b.png", payload)
            hashes = HashCache(workers=1)
            results = check_duplicate_hashes(
                {"train": [], "dev": [], "eval": []},
                {"train": [first, second], "dev": [], "eval": []},
                hashes,
                derived_root=tmp / "derived",
            )
            image = results[1]
            self.assertEqual(image.status, "pass")
            self.assertEqual(image.evidence["cross_split_groups"], 0)
            self.assertEqual(image.evidence["within_split_groups"], 1)
            self.assertEqual(image.evidence["within_split_explained_groups"], 1)
            self.assertEqual(image.evidence["within_split_unexplained_groups"], 0)

    def test_unexplained_within_split_duplicate_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            payload = b"same export image"
            first = write_bytes(tmp / "dataset_img" / "train" / "a.png", payload)
            second = write_bytes(tmp / "dataset_img" / "train" / "b.png", payload)
            hashes = HashCache(workers=1)
            results = check_duplicate_hashes(
                {"train": [], "dev": [], "eval": []},
                {"train": [first, second], "dev": [], "eval": []},
                hashes,
                derived_root=tmp / "derived",
            )
            image = results[1]
            self.assertEqual(image.status, "fail")
            self.assertEqual(image.evidence["within_split_unexplained_groups"], 1)


class ManifestAuditTest(unittest.TestCase):
    def _fixture(self, tmp: Path):
        data_root = tmp / "LA"
        image_root = tmp / "images"
        rows = [
            {"speaker_id": "LA_1", "audio_file_name": "LA_T_1", "system_id": "-", "label": 0},
            {"speaker_id": "LA_2", "audio_file_name": "LA_T_2", "system_id": "A01", "label": 1},
            {"speaker_id": "LA_3", "audio_file_name": "LA_T_3", "system_id": "-", "label": 0},
        ]
        for index, row in enumerate(rows):
            write_bytes(
                data_root / f"{row['audio_file_name']}.flac",
                f"audio-{index}".encode(),
            )
        protocol = protocol_frame(
            [
                {**row, "source_flac": str(data_root / f"{row['audio_file_name']}.flac")}
                for row in rows
            ]
        )
        manifest_path, frame = make_manifest(tmp, protocol, image_root)
        return protocol, manifest_path, frame

    def test_manifest_integrity_passes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            protocol, manifest_path, _ = self._fixture(tmp)
            audit = load_manifest_audit(manifest_path, protocol, tmp / "derived")
            result = check_manifest_integrity(
                audit,
                expected_sha256=audit.sha256,
                expected_rows=3,
                expected_counts={0: 2, 1: 1},
            )
            self.assertEqual(result.status, "pass", result.detail)

    def test_manifest_integrity_detects_protocol_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            protocol, manifest_path, frame = self._fixture(tmp)
            frame.loc[1, "speaker_id"] = "LA_WRONG"
            frame.to_csv(manifest_path, index=False)
            audit = load_manifest_audit(manifest_path, protocol, tmp / "derived")
            result = check_manifest_integrity(
                audit,
                expected_sha256=audit.sha256,
                expected_rows=3,
                expected_counts={0: 2, 1: 1},
            )
            self.assertEqual(result.status, "fail")
            self.assertIn("speaker_id", result.detail)

    def test_manifest_files_pass_and_detect_missing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            protocol, manifest_path, _ = self._fixture(tmp)
            audit = load_manifest_audit(manifest_path, protocol, tmp / "derived")
            result = check_manifest_files(
                audit,
                HashCache(workers=1),
                expected_class_counts={"bonafide": 2, "spoofed": 1},
            )
            self.assertEqual(result.status, "pass", result.detail)
            (tmp / "derived" / "spoofed" / "sample_1.png").unlink()
            result = check_manifest_files(
                audit,
                HashCache(workers=1),
                expected_class_counts={"bonafide": 2, "spoofed": 1},
            )
            self.assertEqual(result.status, "fail")
            self.assertIn("derived images are missing", result.detail)

    def test_manifest_detects_dev_clip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            protocol, manifest_path, _ = self._fixture(tmp)
            audit = load_manifest_audit(manifest_path, protocol, tmp / "derived")
            dev = protocol_frame(
                [
                    {
                        "speaker_id": "D1",
                        "audio_file_name": "LA_T_1",
                        "system_id": "A01",
                        "label": 1,
                        "source_flac": "x",
                    }
                ]
            )
            eval_frame = protocol_frame(
                [
                    {
                        "speaker_id": "E1",
                        "audio_file_name": "LA_E_1",
                        "system_id": "A07",
                        "label": 1,
                        "source_flac": "x",
                    }
                ]
            )
            result = check_manifest_isolation(
                audit,
                {"dev": dev, "eval": eval_frame},
                None,
                None,
            )
            self.assertEqual(result.status, "fail")
            self.assertIn("also appear in dev", result.detail)


class EnrollmentTest(unittest.TestCase):
    def test_enrollment_exclusions_pass(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            dev_rows = [
                {"speaker_id": "D1", "audio_file_name": "LA_D_1", "system_id": "A01", "label": 1}
            ]
            eval_rows = [
                {"speaker_id": "E1", "audio_file_name": "LA_E_1", "system_id": "A07", "label": 1}
            ]
            write_protocol_files(tmp, "dev", dev_rows)
            write_protocol_files(tmp, "eval", eval_rows)
            write_bytes(tmp / SPLIT_SPECS["dev"].split_dirname / "flac" / "LA_D_A1.flac", b"enroll")
            write_bytes(tmp / SPLIT_SPECS["eval"].split_dirname / "flac" / "LA_E_A1.flac", b"enroll")
            protocols = {
                "train": protocol_frame([]),
                "dev": protocol_frame([{**dev_rows[0], "source_flac": "x"}]),
                "eval": protocol_frame([{**eval_rows[0], "source_flac": "x"}]),
            }
            result = check_enrollment_exclusions(
                tmp,
                protocols,
                expected_counts={"train": 0, "dev": 1, "eval": 1},
            )
            self.assertEqual(result.status, "pass", result.detail)


class BaselineLineageTest(unittest.TestCase):
    def _fixture(self, tmp: Path):
        baseline_root = tmp / "baseline"
        image_root = tmp / "images"
        checkpoint_dir = baseline_root / "unweighted"
        checkpoint_dir.mkdir(parents=True)
        rows = {
            "dev": [
                {"speaker_id": "D1", "audio_file_name": "LA_D_1", "system_id": "-", "label": 0, "score": 0.1},
                {"speaker_id": "D1", "audio_file_name": "LA_D_2", "system_id": "A01", "label": 1, "score": 0.9},
            ],
            "eval": [
                {"speaker_id": "E1", "audio_file_name": "LA_E_1", "system_id": "-", "label": 0, "score": 0.2},
                {"speaker_id": "E1", "audio_file_name": "LA_E_2", "system_id": "A07", "label": 1, "score": 0.8},
            ],
        }
        protocols = {}
        for split, split_rows in rows.items():
            folder_labels = ["bonafide", "spoofed"]
            score_rows = []
            for index, row in enumerate(split_rows):
                folder = folder_labels[int(row["label"])]
                source = write_bytes(
                    image_root / split / folder / f"sample_{index}.png",
                    f"{split}-{index}".encode(),
                )
                score_rows.append(
                    {
                        "row_index": index,
                        "clip_id": row["audio_file_name"],
                        "protocol_row": index,
                        "speaker_id": row["speaker_id"],
                        "system_id": row["system_id"],
                        "label": row["label"],
                        "source_path": str(source),
                        "logit": row["score"],
                        "score": row["score"],
                    }
                )
            pd.DataFrame(score_rows).to_csv(
                checkpoint_dir / f"scores_{split}.csv", index=False
            )
            protocols[split] = protocol_frame(
                [
                    {
                        **row,
                        "source_flac": str(image_root / split / f"{row['audio_file_name']}.flac"),
                    }
                    for row in split_rows
                ]
            )

        threshold = compute_eer([0.9], [0.1]).threshold
        metrics = {
            "checkpoint": {
                "path": "Models/Iteration1/unweighted.pth",
                "sha256": "checkpoint-sha",
                "total_parameters": 123,
                "missing_keys": [],
                "unexpected_keys": [],
            },
            "thresholds": {
                "dev_eer": threshold,
                "selected_on": "dev",
                "source": "in_run_dev",
                "policy": "dev only",
            },
            "splits": {
                "dev": {
                    "attack_subset": "A01-A06",
                    "metrics": {
                        "sample_counts": {"total": 2, "bonafide": 1, "spoof": 1},
                        "threshold_metrics": {
                            "fixed_0.5": {"threshold": 0.5},
                            "dev_eer": {"threshold": threshold},
                        },
                    },
                },
                "eval": {
                    "attack_subset": "A07-A19",
                    "metrics": {
                        "sample_counts": {"total": 2, "bonafide": 1, "spoof": 1},
                        "threshold_metrics": {
                            "fixed_0.5": {"threshold": 0.5},
                            "dev_eer": {"threshold": threshold},
                        },
                    },
                },
            },
        }
        (checkpoint_dir / "metrics.json").write_text(
            json.dumps(metrics), encoding="utf-8"
        )
        freeze_report = tmp / "freeze_report.json"
        freeze_report.write_text(
            json.dumps(
                {
                    "training_manifest": {"sha256": "manifest-sha", "passed": True},
                    "checkpoints": [
                        {
                            "stem": "unweighted",
                            "audit": {
                                "sha256": "checkpoint-sha",
                                "total_parameters": 123,
                            },
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        return baseline_root, image_root, protocols, metrics, freeze_report, threshold

    def test_threshold_lineage_passes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            baseline_root, *_ , threshold = self._fixture(tmp)
            result = check_baseline_threshold_lineage(baseline_root)
            self.assertEqual(result.status, "pass", result.detail)
            self.assertAlmostEqual(
                result.evidence["unweighted"]["recomputed_dev_threshold"],
                threshold,
            )

    def test_threshold_lineage_rejects_eval_selection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            baseline_root, _image_root, _protocols, metrics, _freeze, _ = self._fixture(tmp)
            metrics["thresholds"]["selected_on"] = "eval"
            (baseline_root / "unweighted" / "metrics.json").write_text(
                json.dumps(metrics), encoding="utf-8"
            )
            result = check_baseline_threshold_lineage(baseline_root)
            self.assertEqual(result.status, "fail")
            self.assertIn("selected_on", result.detail)

    def test_score_alignment_detects_row_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            baseline_root, image_root, protocols, *_ = self._fixture(tmp)
            scores_path = baseline_root / "unweighted" / "scores_dev.csv"
            scores = pd.read_csv(scores_path).iloc[:1]
            scores.to_csv(scores_path, index=False)
            result = check_baseline_score_alignment(
                baseline_root, protocols, image_root
            )
            self.assertEqual(result.status, "fail")
            self.assertIn("scores for", result.detail)

    def test_artifact_identity_detects_checkpoint_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            baseline_root, _image_root, _protocols, metrics, freeze_report, _ = self._fixture(tmp)
            metrics["checkpoint"]["sha256"] = "different"
            (baseline_root / "unweighted" / "metrics.json").write_text(
                json.dumps(metrics), encoding="utf-8"
            )
            result = check_baseline_artifacts(
                baseline_root, freeze_report, "manifest-sha"
            )
            self.assertEqual(result.status, "fail")
            self.assertIn("checkpoint hash", result.detail)


class ReportTest(unittest.TestCase):
    def test_markdown_and_json_written(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            report = {
                "experiment": "E0.5",
                "title": "Data and leakage audit",
                "generated_at_utc": "2026-11-14T00:00:00+00:00",
                "command": "uv run python scripts/audit_data.py",
                "git": {"commit": "abc", "dirty": False},
                "inputs": {
                    "training_manifest": {
                        "path": "manifests/train_seed42.csv",
                        "sha256": "manifest-sha",
                    },
                    "protocols": {},
                    "metadata": {},
                },
                "inventory": {},
                "checks": [
                    {
                        "id": "A.1",
                        "title": "layout",
                        "status": "pass",
                        "detail": "ok",
                        "evidence": {},
                    }
                ],
                "conclusion": {
                    "status": "pass",
                    "passed": True,
                    "failures": [],
                    "skipped": [],
                    "policy": "dev is a model-selection split",
                },
            }
            markdown_path, json_path = write_audit(
                report, tmp / "audit.md", tmp / "audit.json"
            )
            self.assertIn("E0.5", markdown_path.read_text(encoding="utf-8"))
            self.assertEqual(
                json.loads(json_path.read_text(encoding="utf-8"))["conclusion"][
                    "status"
                ],
                "pass",
            )
            self.assertIn("PASS", render_markdown(report))


class RealDataAuditTest(unittest.TestCase):
    """Run the structural audit against the real repository artifacts."""

    def test_real_data_structural_audit(self) -> None:
        required = [
            REPO_ROOT / "data" / "LA" / "ASVspoof2019_LA_cm_protocols",
            REPO_ROOT / "data" / "dataset_img" / "dev" / "metadata_dev.csv",
            REPO_ROOT / "data" / "derived" / "train_seed42",
            REPO_ROOT / "reports" / "baseline" / "freeze_report.json",
        ]
        if not all(path.exists() for path in required):
            self.skipTest("real ASVspoof data or baseline artifacts are unavailable")
        report = run_audit(AuditConfig(hash_files=False))
        self.assertTrue(
            report["conclusion"]["passed"],
            report["conclusion"]["failures"],
        )
        self.assertEqual(report["conclusion"]["skipped"], ["B.3", "B.4"])


if __name__ == "__main__":
    unittest.main()
