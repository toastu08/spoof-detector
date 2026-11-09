"""Tests for the evaluation CLI, data loaders, and report plumbing."""

from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from spoof_detection.evaluate import (  # noqa: E402
    describe_attack_subset,
    limit_records,
    load_asvspoof_records,
    load_manifest_records,
    load_threshold_from_metrics,
    main,
    records_to_frame,
)
from spoof_detection.model import AudioSpoofDetector  # noqa: E402
from spoof_detection.preprocess import load_mel_image  # noqa: E402
from spoof_detection.protocol import (  # noqa: E402
    load_cm_protocol,
    validate_cm_protocol,
)

DEV_METADATA = REPO_ROOT / "data" / "dataset_img" / "dev" / "metadata_dev.csv"
DEV_PROTOCOL = (
    REPO_ROOT
    / "data"
    / "LA"
    / "ASVspoof2019_LA_cm_protocols"
    / "ASVspoof2019.LA.cm.dev.trl.txt"
)
EVAL_PROTOCOL = (
    REPO_ROOT
    / "data"
    / "LA"
    / "ASVspoof2019_LA_cm_protocols"
    / "ASVspoof2019.LA.cm.eval.trl.txt"
)


class EvaluationCLITest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.manual_seed(0)
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp_dir.name)
        cls.image_dir = cls.root / "images"
        cls.image_dir.mkdir()

        rng = np.random.default_rng(0)
        rows = []
        for index in range(8):
            label = 0 if index < 4 else 1
            folder = "bonafide" if label == 0 else "spoofed"
            path = cls.image_dir / f"{folder}_{index}.png"
            pixels = rng.integers(0, 256, size=(128, 126, 4), dtype=np.uint8)
            Image.fromarray(pixels, mode="RGBA").save(path)
            rows.append(
                {
                    "clip_id": f"clip_{index}",
                    "image_path": str(path),
                    "label": label,
                    "speaker_id": f"speaker_{index % 2}",
                    "system_id": "-" if label == 0 else f"A{index - 3:02d}",
                    "protocol_row": index,
                }
            )

        cls.manifest = cls.root / "manifest.csv"
        pd.DataFrame(rows).to_csv(cls.manifest, index=False)
        cls.rows = rows

        cls.checkpoint = cls.root / "synthetic.pth"
        model = AudioSpoofDetector()
        torch.save(
            {
                key.replace("backbone.", "network.", 1): value
                for key, value in model.state_dict().items()
            },
            cls.checkpoint,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp_dir.cleanup()

    def _run(self, *extra: str) -> tuple[int, Path]:
        output_dir = self.root / "output"
        argv = [
            "--checkpoint",
            str(self.checkpoint),
            "--manifest",
            str(self.manifest),
            "--source",
            "image",
            "--output-dir",
            str(output_dir),
            "--device",
            "cpu",
            "--batch-size",
            "4",
            "--num-workers",
            "0",
            "--no-progress",
            *extra,
        ]
        with contextlib.redirect_stdout(io.StringIO()):
            code = main(argv)
        return code, output_dir

    def test_end_to_end_is_deterministic(self) -> None:
        code, output_dir = self._run()
        self.assertEqual(code, 0)
        metrics_path = output_dir / "metrics.json"
        scores_path = output_dir / "scores_external.csv"
        self.assertTrue(metrics_path.is_file())
        self.assertTrue(scores_path.is_file())
        self.assertTrue((output_dir / "summary.md").is_file())

        first_scores = scores_path.read_bytes()
        first_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        self.assertEqual(first_metrics["checkpoint"]["missing_keys"], [])
        self.assertEqual(first_metrics["checkpoint"]["unexpected_keys"], [])
        self.assertEqual(first_metrics["splits"]["external"]["metrics"]["sample_counts"]["total"], 8)
        self.assertIsNone(first_metrics["thresholds"]["dev_eer"])
        self.assertFalse(first_metrics["partial"])

        code, output_dir = self._run()
        self.assertEqual(code, 0)
        second_scores = scores_path.read_bytes()
        second_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        self.assertEqual(first_scores, second_scores)
        self.assertEqual(
            first_metrics["splits"]["external"]["metrics"],
            second_metrics["splits"]["external"]["metrics"],
        )

    def test_partial_run_uses_separate_directory(self) -> None:
        output_root = self.root / "partial-root"
        argv = [
            "--checkpoint",
            str(self.checkpoint),
            "--manifest",
            str(self.manifest),
            "--source",
            "image",
            "--output-root",
            str(output_root),
            "--device",
            "cpu",
            "--batch-size",
            "4",
            "--num-workers",
            "0",
            "--no-progress",
            "--limit",
            "4",
        ]
        with contextlib.redirect_stdout(io.StringIO()):
            code = main(argv)
        self.assertEqual(code, 0)
        output_dir = output_root / "synthetic_partial4"
        self.assertTrue(output_dir.is_dir())
        metrics = json.loads(
            (output_dir / "metrics.json").read_text(encoding="utf-8")
        )
        self.assertTrue(metrics["partial"])
        self.assertEqual(metrics["limit"], 4)
        scores = pd.read_csv(output_dir / "scores_external.csv")
        self.assertEqual(len(scores), 4)
        self.assertEqual(set(scores["label"]), {0, 1})

    def test_single_class_manifest_reports_unknown_eer(self) -> None:
        manifest = self.root / "spoof-only.csv"
        pd.DataFrame(
            [row for row in self.rows if row["label"] == 1]
        ).to_csv(manifest, index=False)
        output_dir = self.root / "single-class"
        argv = [
            "--checkpoint",
            str(self.checkpoint),
            "--manifest",
            str(manifest),
            "--source",
            "image",
            "--split-name",
            "spoof_only",
            "--output-dir",
            str(output_dir),
            "--device",
            "cpu",
            "--batch-size",
            "4",
            "--num-workers",
            "0",
            "--no-progress",
        ]
        with contextlib.redirect_stdout(io.StringIO()):
            code = main(argv)
        self.assertEqual(code, 0)
        metrics = json.loads(
            (output_dir / "metrics.json").read_text(encoding="utf-8")
        )
        summary = metrics["splits"]["spoof_only"]["metrics"]
        self.assertIsNone(summary["eer"])
        self.assertIsNone(summary["auc"])
        self.assertEqual(
            summary["sample_counts"],
            {"total": 4, "bonafide": 0, "spoof": 4},
        )
        self.assertIsNotNone(
            summary["threshold_metrics"]["fixed_0.5"]["spoof_recall"]
        )
        self.assertIsNone(
            summary["threshold_metrics"]["fixed_0.5"]["bonafide_recall"]
        )

    def test_manifest_loader_maps_string_labels(self) -> None:
        manifest = self.root / "string-labels.csv"
        pd.DataFrame(
            [
                {
                    "clip_id": "one",
                    "image_path": self.rows[0]["image_path"],
                    "label": "bonafide",
                    "speaker_id": "s1",
                    "system_id": "-",
                },
                {
                    "clip_id": "two",
                    "image_path": self.rows[4]["image_path"],
                    "label": "spoof",
                    "speaker_id": "s1",
                    "system_id": "A01",
                },
            ]
        ).to_csv(manifest, index=False)
        records, provenance = load_manifest_records(
            manifest,
            source="image",
            base_dir=self.root,
            split_name="external",
        )
        self.assertEqual([record.label for record in records], [0, 1])
        self.assertEqual([record.clip_id for record in records], ["one", "two"])
        self.assertEqual(provenance["manifest_rows"], 2)
        self.assertEqual(len(provenance["manifest_sha256"]), 64)

    def test_manifest_requires_label_column(self) -> None:
        manifest = self.root / "no-label.csv"
        pd.DataFrame(
            [{"image_path": self.rows[0]["image_path"]}]
        ).to_csv(manifest, index=False)
        with self.assertRaises(ValueError):
            load_manifest_records(
                manifest,
                source="image",
                base_dir=self.root,
                split_name="external",
            )

    def test_manifest_rejects_missing_file(self) -> None:
        manifest = self.root / "missing-file.csv"
        pd.DataFrame(
            [
                {
                    "image_path": str(self.root / "does-not-exist.png"),
                    "label": 0,
                }
            ]
        ).to_csv(manifest, index=False)
        with self.assertRaises(FileNotFoundError):
            load_manifest_records(
                manifest,
                source="image",
                base_dir=self.root,
                split_name="external",
            )

    def test_limit_records_is_deterministic_and_stratified(self) -> None:
        records, _ = load_manifest_records(
            self.manifest,
            source="image",
            base_dir=self.root,
            split_name="external",
        )
        selected = limit_records(records, 4)
        self.assertEqual(len(selected), 4)
        self.assertEqual({record.label for record in selected}, {0, 1})
        self.assertEqual(
            [record.row_index for record in selected],
            sorted(record.row_index for record in selected),
        )
        self.assertEqual(selected, limit_records(records, 4))

    def test_records_to_frame_columns(self) -> None:
        records, _ = load_manifest_records(
            self.manifest,
            source="image",
            base_dir=self.root,
            split_name="external",
        )
        frame = records_to_frame(records)
        self.assertEqual(
            list(frame.columns),
            [
                "row_index",
                "clip_id",
                "protocol_row",
                "speaker_id",
                "system_id",
                "label",
                "source_path",
            ],
        )
        self.assertEqual(len(frame), 8)

    def test_describe_attack_subset(self) -> None:
        dev = pd.DataFrame(
            {
                "label": [0, 1],
                "system_id": ["-", "A01"],
            }
        )
        self.assertEqual(describe_attack_subset(dev), "A01")
        full_dev = pd.DataFrame(
            {
                "label": [1] * 6,
                "system_id": [f"A{index:02d}" for index in range(1, 7)],
            }
        )
        self.assertEqual(describe_attack_subset(full_dev), "A01-A06")
        full_eval = pd.DataFrame(
            {
                "label": [1] * 13,
                "system_id": [f"A{index:02d}" for index in range(7, 20)],
            }
        )
        self.assertEqual(describe_attack_subset(full_eval), "A07-A19")
        unknown = pd.DataFrame({"label": [1], "system_id": ["-"]})
        self.assertEqual(describe_attack_subset(unknown), "unknown")


class ThresholdReferenceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _write(self, payload: dict[str, object]) -> Path:
        path = self.root / "metrics.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def _payload(self) -> dict[str, object]:
        return {
            "partial": False,
            "thresholds": {"dev_eer": 0.42, "selected_on": "dev"},
            "checkpoint": {"sha256": "abc"},
        }

    def test_loads_dev_threshold(self) -> None:
        reference = load_threshold_from_metrics(self._write(self._payload()))
        self.assertAlmostEqual(reference.threshold, 0.42)
        self.assertEqual(reference.selected_on, "dev")
        self.assertEqual(reference.checkpoint_sha256, "abc")

    def test_rejects_partial_run(self) -> None:
        payload = self._payload()
        payload["partial"] = True
        with self.assertRaises(ValueError):
            load_threshold_from_metrics(self._write(payload))

    def test_rejects_non_dev_threshold(self) -> None:
        payload = self._payload()
        payload["thresholds"] = {"dev_eer": 0.42, "selected_on": "eval"}
        with self.assertRaises(ValueError):
            load_threshold_from_metrics(self._write(payload))

    def test_rejects_missing_threshold(self) -> None:
        payload = self._payload()
        payload["thresholds"] = {"selected_on": "dev"}
        with self.assertRaises(ValueError):
            load_threshold_from_metrics(self._write(payload))

    def test_rejects_missing_checkpoint_hash(self) -> None:
        payload = self._payload()
        payload["checkpoint"] = {}
        with self.assertRaises(ValueError):
            load_threshold_from_metrics(self._write(payload))


class ASVspoofProtocolTest(unittest.TestCase):
    def test_dev_and_eval_protocol_counts(self) -> None:
        if not DEV_PROTOCOL.is_file() or not EVAL_PROTOCOL.is_file():
            self.skipTest("ASVspoof CM protocols are not available")
        for split, expected_rows, expected_counts in (
            ("dev", 24_844, {0: 2_548, 1: 22_296}),
            ("eval", 71_237, {0: 7_355, 1: 63_882}),
        ):
            protocol = load_cm_protocol(split)
            self.assertEqual(validate_cm_protocol(protocol, split), [])
            self.assertEqual(len(protocol), expected_rows)
            self.assertEqual(
                protocol["label"].value_counts().to_dict(),
                expected_counts,
            )

    def test_dev_metadata_alignment(self) -> None:
        if not DEV_METADATA.is_file():
            self.skipTest("dev image export is not available")
        records, provenance = load_asvspoof_records(
            "dev",
            source="image",
        )
        self.assertEqual(len(records), 24_844)
        self.assertEqual(
            sum(record.label == 0 for record in records),
            2_548,
        )
        self.assertEqual(
            sum(record.label == 1 for record in records),
            22_296,
        )
        self.assertEqual(records[0].protocol_row, 0)
        self.assertTrue(records[0].source_path.is_file())
        self.assertEqual(len(provenance["protocol_sha256"]), 64)
        self.assertEqual(len(provenance["metadata_sha256"]), 64)

    def test_audio_render_matches_stored_png(self) -> None:
        if not DEV_PROTOCOL.is_file() or not DEV_METADATA.is_file():
            self.skipTest("dev data is not available")
        protocol = load_cm_protocol("dev")
        # Protocol rows 0..2547 are bonafide; row 2548 is the first spoof.
        for protocol_row in (0, 2_548):
            row = protocol.iloc[protocol_row]
            folder = "bonafide" if int(row["label"]) == 0 else "spoofed"
            stored_path = (
                REPO_ROOT
                / "data"
                / "dataset_img"
                / "dev"
                / folder
                / f"sample_{protocol_row}.png"
            )
            if not stored_path.is_file():
                self.skipTest("dev image export is not available")
            audio_path = REPO_ROOT / row["source_flac"]
            rendered = np.asarray(load_mel_image(audio_path).convert("RGB"))
            stored = np.asarray(Image.open(stored_path).convert("RGB"))
            self.assertEqual(rendered.shape, stored.shape)
            self.assertTrue(
                np.array_equal(rendered, stored),
                f"rendered audio does not match {stored_path}",
            )


if __name__ == "__main__":
    unittest.main()
