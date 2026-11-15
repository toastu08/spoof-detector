"""Tests for the E1.1 export layout, manifest glue and summarizer."""

from __future__ import annotations

import importlib.util
import os

os.environ.setdefault("MPLBACKEND", "Agg")

import sys
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from spoof_detection.frontend_export import (  # noqa: E402
    SplitExport,
    build_split_records,
    folder_for_label,
    metadata_frame,
    prepare_output_dir,
    verify_determinism,
)
from spoof_detection.preprocess import SAMPLE_RATE  # noqa: E402

TRAIN_MANIFEST = REPO_ROOT / "manifests" / "train_seed42.csv"


def _tone(seconds: float = 4.0, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    t = np.arange(int(seconds * sample_rate), dtype=np.float32) / sample_rate
    return (0.1 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)


def _write_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    pcm = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    pcm = np.round(pcm * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


def _load_script(name: str):
    path = REPO_ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ExportLayoutTest(unittest.TestCase):
    def test_folder_for_label(self) -> None:
        self.assertEqual(folder_for_label(0), "bonafide")
        self.assertEqual(folder_for_label(1), "spoofed")

    def test_prepare_output_dir_refuses_outside_derived(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            derived = root / "data" / "derived"
            derived.mkdir(parents=True)
            with self.assertRaises(ValueError):
                prepare_output_dir(root / "elsewhere", derived, force=False)

    def test_prepare_output_dir_requires_force_when_non_empty(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            derived = root / "derived"
            target = derived / "mel64" / "dev"
            target.mkdir(parents=True)
            (target / "existing.png").write_bytes(b"x")
            with self.assertRaises(FileExistsError):
                prepare_output_dir(target, derived, force=False)
            prepare_output_dir(target, derived, force=True)
            self.assertTrue(target.is_dir())
            self.assertFalse((target / "existing.png").exists())

    @unittest.skipUnless(TRAIN_MANIFEST.is_file(), "seed-42 manifest missing")
    def test_train_records_use_sample_index_and_exist(self) -> None:
        frame = build_split_records(
            "train",
            repo_root=REPO_ROOT,
            train_manifest=TRAIN_MANIFEST,
            data_root=REPO_ROOT / "data" / "LA",
            limit=5,
        )
        self.assertEqual(len(frame), 5)
        np.testing.assert_array_equal(
            frame["file_index"].to_numpy(), frame["sample_index"].to_numpy()
        )
        self.assertTrue(frame["source_path"].map(Path.is_file).all())

    @unittest.skipUnless(TRAIN_MANIFEST.is_file(), "seed-42 manifest missing")
    def test_dev_records_use_protocol_row(self) -> None:
        frame = build_split_records(
            "dev",
            repo_root=REPO_ROOT,
            train_manifest=TRAIN_MANIFEST,
            data_root=REPO_ROOT / "data" / "LA",
            limit=5,
        )
        self.assertEqual(len(frame), 5)
        np.testing.assert_array_equal(
            frame["file_index"].to_numpy(), frame["protocol_row"].to_numpy()
        )

    def test_metadata_frame_columns(self) -> None:
        frame = pd.DataFrame(
            {
                "file_index": [0, 1],
                "protocol_row": [10, 11],
                "speaker_id": ["s1", "s2"],
                "system_id": ["-", "A01"],
                "label": [0, 1],
                "audio_file_name": ["a", "b"],
                "source_flac": ["x", "y"],
                "image_path": ["data/a.png", "data/b.png"],
                "sha256_png": ["0", "1"],
            }
        )
        export = SplitExport(
            split="dev",
            frame=frame,
            split_dir=Path("data"),
            metadata_path=Path("data/metadata_dev.csv"),
        )
        metadata = metadata_frame(export)
        self.assertEqual(
            list(metadata.columns),
            ["speaker_id", "system_id", "label", "img_path"],
        )
        self.assertEqual(metadata["img_path"].tolist()[0], "./data/a.png")


class DeterminismTest(unittest.TestCase):
    def test_render_twice_matches(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            clip = root / "tone.wav"
            _write_wav(clip, _tone(), SAMPLE_RATE)
            records = pd.DataFrame(
                {
                    "file_index": [0, 1],
                    "protocol_row": [0, 1],
                    "speaker_id": ["s", "s"],
                    "system_id": ["-", "A01"],
                    "label": [0, 1],
                    "audio_file_name": ["a", "b"],
                    "source_flac": [str(clip), str(clip)],
                    "image_path": ["a.png", "b.png"],
                    "sha256_png": ["0", "1"],
                    "source_path": [str(clip), str(clip)],
                }
            )
            # The written export frame deliberately omits source_path; the script
            # passes the pre-export records for the re-render check.
            export_frame = records.drop(columns=["source_path"])
            export = SplitExport(
                split="dev",
                frame=export_frame,
                split_dir=root,
                metadata_path=root / "metadata.csv",
            )
            report = verify_determinism(
                export, frontend="mel64", sample=2, records=records
            )
            self.assertEqual(report["checked"], 2)
            self.assertEqual(report["matched"], 2)

    def test_render_without_source_path_is_skipped(self) -> None:
        export = SplitExport(
            split="dev",
            frame=pd.DataFrame({"file_index": [0], "image_path": ["a.png"]}),
            split_dir=Path("."),
            metadata_path=Path("metadata.csv"),
        )
        report = verify_determinism(export, frontend="mel64", sample=1)
        # No source_path column -> nothing to re-render.
        self.assertEqual(report["checked"], 0)


class SummarizerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = _load_script("summarize_e1_1.py")

    def _metrics(self) -> dict[str, object]:
        return {
            "frontend": "mel64",
            "checkpoint": {"sha256": "abc"},
            "thresholds": {"dev_eer": 0.5},
            "splits": {
                "dev": {
                    "attack_subset": "A01-A06",
                    "metrics": {
                        "sample_counts": {
                            "total": 10,
                            "bonafide": 4,
                            "spoof": 6,
                        },
                        "eer": 0.1,
                        "auc": 0.95,
                        "threshold_metrics": {
                            "fixed_0.5": {
                                "false_positive_rate": 0.25,
                                "spoof_recall": 0.5,
                            }
                        },
                        "per_attack": {
                            "A01": {
                                "n_spoof": 3,
                                "n_bonafide": 4,
                                "eer": 0.2,
                                "auc": 0.9,
                                "eer_threshold_descriptive": 0.4,
                            }
                        },
                    },
                }
            },
        }

    def test_summary_rows_keep_metric_contract(self) -> None:
        rows = self.module._summary_rows(
            "mel64", self._metrics(), Path("metrics.json")
        )
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["frontend"], "mel64")
        self.assertEqual(row["split"], "dev")
        self.assertEqual(row["attack_subset"], "A01-A06")
        self.assertEqual(row["samples_total"], 10)
        self.assertEqual(row["samples_bonafide"], 4)
        self.assertEqual(row["samples_spoof"], 6)
        self.assertAlmostEqual(row["threshold_free_eer"], 0.1)
        self.assertAlmostEqual(row["fpr_at_0.5"], 0.25)
        self.assertAlmostEqual(row["dev_eer_threshold"], 0.5)

    def test_per_attack_rows(self) -> None:
        rows = self.module._per_attack_rows("mel64", self._metrics())
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["attack_id"], "A01")
        self.assertAlmostEqual(rows[0]["eer"], 0.2)

    def test_markdown_mentions_threshold_free(self) -> None:
        summary = pd.DataFrame(
            self.module._summary_rows(
                "mel64", self._metrics(), Path("metrics.json")
            )
        )
        per_attack = pd.DataFrame(
            self.module._per_attack_rows("mel64", self._metrics())
        )
        markdown = self.module._results_markdown(
            summary,
            per_attack,
            [{"frontend": "mel64", "metrics_path": "metrics.json", "present": True}],
        )
        self.assertIn("threshold-free", markdown)
        self.assertIn("mel64", markdown)
        self.assertIn("A01", markdown)


if __name__ == "__main__":
    unittest.main()
