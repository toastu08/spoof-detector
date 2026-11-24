"""Tests for the E1.2 rendering registry, exporter and summarizer."""

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
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from spoof_detection import evaluate  # noqa: E402
from spoof_detection.frontend_export import metadata_frame, write_metadata  # noqa: E402
from spoof_detection.preprocess import (  # noqa: E402
    PREPROCESSING_VERSION,
    SAMPLE_RATE,
    generate_mel_spectrogram,
    render_mel_rgb,
)
from spoof_detection.rendering import (  # noqa: E402
    RENDER_CONDITION_ORDER,
    RENDER_CONDITIONS,
    RAW_DB_RANGE,
    get_rendering_spec,
    render_rendering_rgb,
    rendering_payload,
    save_rendering_png,
)
from spoof_detection.rendering_export import (  # noqa: E402
    export_rendering_split,
    verify_flip_pair_parity,
    verify_rgb_pair_parity,
    verify_rendering_determinism,
)


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


def _rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def _load_script(name: str):
    path = REPO_ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class RenderingRegistryTest(unittest.TestCase):
    def test_order_covers_all_conditions(self) -> None:
        self.assertEqual(
            set(RENDER_CONDITION_ORDER), set(RENDER_CONDITIONS)
        )
        self.assertEqual(len(RENDER_CONDITION_ORDER), 8)

    def test_control_matches_existing_contract(self) -> None:
        spec = get_rendering_spec("magma_flip")
        self.assertEqual(spec.version, PREPROCESSING_VERSION)
        self.assertEqual(spec.colormap, "magma")
        self.assertTrue(spec.flip)
        self.assertEqual(spec.encoding, "rgba")
        self.assertEqual(spec.normalization, "per_clip_minmax")

    def test_rawdb_is_fixed_range_grayscale(self) -> None:
        spec = get_rendering_spec("rawdb_flip")
        self.assertEqual(spec.colormap, "gray")
        self.assertEqual(spec.normalization, "fixed_db_range")
        self.assertEqual(spec.db_range, RAW_DB_RANGE)

    def test_payload_contract(self) -> None:
        for name in RENDER_CONDITION_ORDER:
            with self.subTest(condition=name):
                payload = rendering_payload(name)
                self.assertEqual(payload["condition"], name)
                self.assertEqual(payload["version"], RENDER_CONDITIONS[name].version)
                self.assertEqual(payload["representation"], "mel128")
                self.assertEqual(payload["params"]["n_mels"], 128)
                self.assertIn(payload["encoding"], {"RGBA", "RGB"})

    def test_unknown_condition_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            get_rendering_spec("not_a_condition")
        with self.assertRaises(ValueError):
            rendering_payload("not_a_condition")


class RenderingPixelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp_dir.name)
        cls.clip = cls.root / "tone.wav"
        _write_wav(cls.clip, _tone(), SAMPLE_RATE)
        cls.spec_db = generate_mel_spectrogram(cls.clip)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp_dir.cleanup()

    def _save(self, condition: str, name: str) -> Path:
        output = self.root / f"{name}.png"
        save_rendering_png(condition, self.spec_db, output)
        return output

    def test_control_matches_existing_renderer(self) -> None:
        output = self._save("magma_flip", "control")
        np.testing.assert_array_equal(
            _rgb(output), render_mel_rgb(self.spec_db)
        )

    def test_noflip_is_exact_vertical_flip(self) -> None:
        for flipped_name, unflipped_name in (
            ("magma_flip", "magma_noflip"),
            ("viridis_flip", "viridis_noflip"),
            ("gray_flip", "gray_noflip"),
        ):
            with self.subTest(pair=flipped_name):
                flipped = self._save(flipped_name, flipped_name)
                unflipped = self._save(unflipped_name, unflipped_name)
                np.testing.assert_array_equal(
                    np.flipud(_rgb(flipped)), _rgb(unflipped)
                )

    def test_gray_channels_are_equal(self) -> None:
        rgb = _rgb(self._save("gray_flip", "gray"))
        np.testing.assert_array_equal(rgb[:, :, 0], rgb[:, :, 1])
        np.testing.assert_array_equal(rgb[:, :, 1], rgb[:, :, 2])

    def test_rgb_encoding_matches_rgba(self) -> None:
        rgba = _rgb(self._save("magma_flip", "rgba"))
        rgb = _rgb(self._save("magma_flip_rgb", "rgb"))
        np.testing.assert_array_equal(rgba, rgb)

    def test_rawdb_uses_fixed_range_without_autoscale(self) -> None:
        spec = np.array([[-80.0], [-40.0]], dtype=np.float64)
        output = self.root / "rawdb.png"
        save_rendering_png("rawdb_flip", spec, output)
        rgb = _rgb(output)
        # flip=True: spec row 1 (-40 dB -> 0.5) is the top row.
        np.testing.assert_allclose(rgb[0, 0], [127, 127, 127], atol=2)
        np.testing.assert_array_equal(rgb[1, 0], [0, 0, 0])

    def test_render_is_deterministic(self) -> None:
        first = self.root / "determinism_a.png"
        second = self.root / "determinism_b.png"
        save_rendering_png("viridis_noflip", self.spec_db, first)
        save_rendering_png("viridis_noflip", self.spec_db, second)
        self.assertEqual(first.read_bytes(), second.read_bytes())

    def test_png_dimensions(self) -> None:
        output = self._save("gray_noflip", "dimensions")
        with Image.open(output) as image:
            width, height = image.size
        self.assertEqual((height, width), (128, 126))

    def test_render_helper_matches_saved_png(self) -> None:
        for name in ("magma_flip", "viridis_noflip", "rawdb_flip"):
            with self.subTest(condition=name):
                output = self._save(name, f"helper_{name}")
                np.testing.assert_allclose(
                    _rgb(output),
                    render_rendering_rgb(name, self.spec_db),
                    atol=1,
                )


class EvaluateDispatchTest(unittest.TestCase):
    def test_all_image_conditions_include_both_experiments(self) -> None:
        self.assertIn("mel128", evaluate.ALL_IMAGE_CONDITIONS)
        self.assertIn("magma_flip", evaluate.ALL_IMAGE_CONDITIONS)

    def test_payload_dispatch(self) -> None:
        frontend_payload = evaluate._payload_for_condition("mel128")
        rendering = evaluate._payload_for_condition("magma_flip")
        self.assertEqual(frontend_payload["frontend"], "mel128")
        self.assertEqual(rendering["condition"], "magma_flip")


class RenderingExportTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.repo_root = Path(self.temp_dir.name)
        self.clip = self.repo_root / "tone.wav"
        _write_wav(self.clip, _tone(), SAMPLE_RATE)
        self.records = pd.DataFrame(
            {
                "file_index": [0, 1],
                "protocol_row": [0, 1],
                "speaker_id": ["s1", "s2"],
                "system_id": ["-", "A01"],
                "label": [0, 1],
                "audio_file_name": ["a", "b"],
                "source_flac": [str(self.clip), str(self.clip)],
                "source_path": [str(self.clip), str(self.clip)],
            }
        )
        self.output_root = (
            self.repo_root / "data" / "derived" / "e1_2_rendering"
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_matrix_export_layout(self) -> None:
        conditions = ("magma_flip", "gray_noflip")
        exports = export_rendering_split(
            self.records,
            conditions=conditions,
            split="dev",
            output_root=self.output_root,
            repo_root=self.repo_root,
            workers=1,
            show_progress=False,
        )
        self.assertEqual(set(exports), set(conditions))
        for condition in conditions:
            export = exports[condition]
            self.assertEqual(export.counts["total"], 2)
            self.assertEqual(export.counts["bonafide"], 1)
            self.assertEqual(export.counts["spoof"], 1)
            for path in (
                export.split_dir
                / "bonafide"
                / "sample_0.png",
                export.split_dir / "spoofed" / "sample_1.png",
            ):
                self.assertTrue(path.is_file())
            self.assertEqual(
                list(export.frame.columns),
                [
                    "file_index",
                    "protocol_row",
                    "speaker_id",
                    "system_id",
                    "label",
                    "audio_file_name",
                    "source_flac",
                    "image_path",
                    "sha256_png",
                ],
            )

        metadata = metadata_frame(exports["magma_flip"])
        self.assertEqual(
            list(metadata.columns),
            ["speaker_id", "system_id", "label", "img_path"],
        )
        metadata_path = write_metadata(exports["magma_flip"])
        self.assertTrue(metadata_path.is_file())

    def test_parity_and_determinism_checks(self) -> None:
        conditions = ("magma_flip", "magma_noflip", "magma_flip_rgb")
        exports = export_rendering_split(
            self.records,
            conditions=conditions,
            split="dev",
            output_root=self.output_root,
            repo_root=self.repo_root,
            workers=1,
            show_progress=False,
        )
        flip_report = verify_flip_pair_parity(
            exports["magma_flip"],
            exports["magma_noflip"],
            repo_root=self.repo_root,
            sample=2,
        )
        self.assertEqual(flip_report["checked"], 2)
        self.assertEqual(flip_report["matched"], 2)

        rgb_report = verify_rgb_pair_parity(
            exports["magma_flip"],
            exports["magma_flip_rgb"],
            repo_root=self.repo_root,
            sample=2,
        )
        self.assertEqual(rgb_report["checked"], 2)
        self.assertEqual(rgb_report["matched"], 2)

        determinism = verify_rendering_determinism(
            "gray_noflip", self.records, sample=2
        )
        self.assertEqual(determinism["checked"], 2)
        self.assertEqual(determinism["matched"], 2)


class SummarizerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = _load_script("summarize_e1_2.py")

    def _metrics(self, eer: float = 0.1, auc: float = 0.95) -> dict[str, object]:
        return {
            "thresholds": {"dev_eer": 0.5},
            "checkpoint": {"sha256": "abc"},
            "splits": {
                "dev": {
                    "attack_subset": "A01-A06",
                    "metrics": {
                        "sample_counts": {
                            "total": 10,
                            "bonafide": 4,
                            "spoof": 6,
                        },
                        "eer": eer,
                        "auc": auc,
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
            "magma_flip", 42, self._metrics(), Path("metrics.json")
        )
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["condition"], "magma_flip")
        self.assertEqual(row["seed"], 42)
        self.assertEqual(row["split"], "dev")
        self.assertEqual(row["attack_subset"], "A01-A06")
        self.assertEqual(row["samples_total"], 10)
        self.assertEqual(row["samples_bonafide"], 4)
        self.assertEqual(row["samples_spoof"], 6)
        self.assertAlmostEqual(row["threshold_free_eer"], 0.1)
        self.assertAlmostEqual(row["fpr_at_0.5"], 0.25)
        self.assertAlmostEqual(row["dev_eer_threshold"], 0.5)
        self.assertEqual(row["colormap"], "magma")

    def test_per_attack_rows(self) -> None:
        rows = self.module._per_attack_rows(
            "magma_flip", 42, self._metrics()
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["attack_id"], "A01")
        self.assertAlmostEqual(rows[0]["eer"], 0.2)

    def test_aggregate_by_condition_includes_delta(self) -> None:
        summary = pd.DataFrame(
            self.module._summary_rows(
                "magma_flip", 42, self._metrics(0.1), Path("a.json")
            )
            + self.module._summary_rows(
                "gray_flip", 42, self._metrics(0.12), Path("b.json")
            ),
            columns=self.module.SUMMARY_COLUMNS,
        )
        aggregate = self.module._aggregate_by_condition(summary)
        self.assertEqual(len(aggregate), 2)
        gray = aggregate[aggregate["condition"] == "gray_flip"].iloc[0]
        self.assertEqual(gray["n_seeds"], 1)
        self.assertTrue(np.isnan(gray["threshold_free_eer_std"]))
        self.assertAlmostEqual(gray["delta_eer_vs_control_pts"], 2.0)

    def test_per_attack_aggregate(self) -> None:
        per_attack = pd.DataFrame(
            self.module._per_attack_rows("magma_flip", 42, self._metrics()),
            columns=self.module.PER_ATTACK_COLUMNS,
        )
        aggregate = self.module._aggregate_per_attack(per_attack)
        self.assertEqual(len(aggregate), 1)
        self.assertEqual(aggregate.iloc[0]["attack_id"], "A01")

    def test_colormap_statistics(self) -> None:
        statistics = self.module._colormap_statistics(
            ("magma", "viridis", "gray")
        )
        self.assertEqual(len(statistics), 3)
        gray = statistics[statistics["colormap"] == "gray"].iloc[0]
        magma = statistics[statistics["colormap"] == "magma"].iloc[0]
        self.assertAlmostEqual(gray["mean_saturation"], 0.0)
        self.assertGreater(magma["mean_saturation"], 0.0)

    def test_markdown_mentions_threshold_free(self) -> None:
        summary = pd.DataFrame(
            self.module._summary_rows(
                "magma_flip", 42, self._metrics(), Path("metrics.json")
            ),
            columns=self.module.SUMMARY_COLUMNS,
        )
        by_condition = self.module._aggregate_by_condition(summary)
        per_attack = pd.DataFrame(
            self.module._per_attack_rows("magma_flip", 42, self._metrics()),
            columns=self.module.PER_ATTACK_COLUMNS,
        )
        per_attack_by_condition = self.module._aggregate_per_attack(per_attack)
        statistics = self.module._colormap_statistics(("magma",))
        markdown = self.module._results_markdown(
            summary,
            by_condition,
            per_attack_by_condition,
            statistics,
            [
                {
                    "condition": "magma_flip",
                    "seed": 42,
                    "metrics_path": "metrics.json",
                    "present": True,
                }
            ],
            ["magma_flip"],
            [42],
        )
        self.assertIn("threshold-free", markdown)
        self.assertIn("magma_flip", markdown)
        self.assertIn("ΔEER", markdown)


if __name__ == "__main__":
    unittest.main()
