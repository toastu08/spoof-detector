"""Tests for the E1.4 shortcut registry, masking, exporter and feature probes."""

from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import sys
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from spoof_detection import evaluate  # noqa: E402
from spoof_detection.feature_probe import (  # noqa: E402
    FEATURE_PROBE_ORDER,
    evaluate_feature_probe,
    feature_matrix,
    feature_probe_payload,
    train_feature_probe,
)
from spoof_detection.frontend_export import metadata_frame, write_metadata  # noqa: E402
from spoof_detection.preprocess import (  # noqa: E402
    PREPROCESSING_VERSION,
    SAMPLE_RATE,
    TARGET_LENGTH,
    generate_mel_spectrogram,
    load_fixed_audio,
    normalize_mel_db,
    render_mel_rgb,
)
from spoof_detection.shortcuts import (  # noqa: E402
    SHORTCUT_ORDER,
    SHORTCUTS,
    get_shortcut_spec,
    masked_db,
    render_shortcut_rgb,
    save_shortcut_png,
    shortcut_payload,
    speech_mask,
)
from spoof_detection.shortcuts_export import (  # noqa: E402
    SHORTCUT_MANIFEST_COLUMNS,
    export_shortcut_split,
    verify_shortcut_determinism,
)


def _tone(seconds: float = 4.0, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    t = np.arange(int(seconds * sample_rate), dtype=np.float32) / sample_rate
    return (0.1 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)


def _tone_then_silence(sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """1 s of 440 Hz tone followed by 3 s of silence (4 s total)."""
    tone = _tone(1.0, sample_rate)
    silence = np.zeros(3 * sample_rate, dtype=np.float32)
    return np.concatenate([tone, silence]).astype(np.float32)


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


class ShortcutRegistryTest(unittest.TestCase):
    def test_order_covers_all_conditions(self) -> None:
        self.assertEqual(set(SHORTCUT_ORDER), set(SHORTCUTS))
        self.assertEqual(len(SHORTCUT_ORDER), 3)

    def test_full_matches_existing_contract(self) -> None:
        spec = get_shortcut_spec("full")
        self.assertEqual(spec.version, PREPROCESSING_VERSION)
        self.assertEqual(spec.keep, "all")

    def test_payload_contract(self) -> None:
        for name in SHORTCUT_ORDER:
            with self.subTest(condition=name):
                payload = shortcut_payload(name)
                self.assertEqual(payload["condition"], name)
                self.assertEqual(payload["version"], SHORTCUTS[name].version)
                self.assertEqual(payload["representation"], "mel128")
                self.assertEqual(payload["params"]["n_mels"], 128)
                self.assertEqual(payload["params"]["mask_keep"], SHORTCUTS[name].keep)

    def test_unknown_condition_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            get_shortcut_spec("not_a_condition")
        with self.assertRaises(ValueError):
            shortcut_payload("not_a_condition")


class SpeechMaskTest(unittest.TestCase):
    def test_mask_length_is_time_frames(self) -> None:
        y_fixed = _tone_then_silence()
        mask = speech_mask(y_fixed, SAMPLE_RATE)
        self.assertEqual(mask.shape, (126,))
        self.assertEqual(mask.dtype, np.bool_)

    def test_tone_then_silence_boundary(self) -> None:
        y_fixed = _tone_then_silence()
        mask = speech_mask(y_fixed, SAMPLE_RATE)
        # The first ~1 s is tone (speech); the last ~3 s is silence.
        self.assertTrue(mask[:20].all())
        self.assertFalse(mask[-40:].any())

    def test_constant_tone_is_all_speech(self) -> None:
        y_fixed = _tone()
        mask = speech_mask(y_fixed, SAMPLE_RATE)
        self.assertTrue(mask.all())


class MaskedDbTest(unittest.TestCase):
    def setUp(self) -> None:
        self.spec_db = np.linspace(-80.0, 0.0, 128 * 10).reshape(128, 10)
        self.mask = np.array([True] * 4 + [False] * 6)

    def test_full_is_unchanged(self) -> None:
        np.testing.assert_array_equal(
            masked_db("full", self.spec_db, self.mask), self.spec_db
        )

    def test_speech_only_masks_nonspeech(self) -> None:
        result = masked_db("speech_only", self.spec_db, self.mask)
        np.testing.assert_array_equal(result[:, ~self.mask], np.zeros((128, 6)))
        normalized = normalize_mel_db(self.spec_db)
        np.testing.assert_allclose(result[:, self.mask], normalized[:, self.mask])

    def test_nonspeech_only_masks_speech(self) -> None:
        result = masked_db("nonspeech_only", self.spec_db, self.mask)
        np.testing.assert_array_equal(result[:, self.mask], np.zeros((128, 4)))
        normalized = normalize_mel_db(self.spec_db)
        np.testing.assert_allclose(result[:, ~self.mask], normalized[:, ~self.mask])


class ShortcutPixelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp_dir.name)
        cls.clip = cls.root / "tone.wav"
        _write_wav(cls.clip, _tone(), SAMPLE_RATE)
        cls.y_fixed, cls.sr = load_fixed_audio(cls.clip)
        cls.spec_db = generate_mel_spectrogram(cls.clip)
        cls.mask = speech_mask(cls.y_fixed, cls.sr)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp_dir.cleanup()

    def test_full_matches_existing_renderer(self) -> None:
        output = self.root / "full.png"
        save_shortcut_png("full", self.spec_db, self.mask, output)
        np.testing.assert_array_equal(_rgb(output), render_mel_rgb(self.spec_db))

    def test_png_dimensions(self) -> None:
        output = self.root / "dim.png"
        save_shortcut_png("speech_only", self.spec_db, self.mask, output)
        with Image.open(output) as image:
            width, height = image.size
        self.assertEqual((height, width), (128, 126))

    def test_render_is_deterministic(self) -> None:
        first = self.root / "det_a.png"
        second = self.root / "det_b.png"
        save_shortcut_png("nonspeech_only", self.spec_db, self.mask, first)
        save_shortcut_png("nonspeech_only", self.spec_db, self.mask, second)
        self.assertEqual(first.read_bytes(), second.read_bytes())

    def test_render_helper_matches_saved_png(self) -> None:
        for name in ("full", "speech_only", "nonspeech_only"):
            with self.subTest(condition=name):
                output = self.root / f"helper_{name}.png"
                save_shortcut_png(name, self.spec_db, self.mask, output)
                np.testing.assert_allclose(
                    _rgb(output),
                    render_shortcut_rgb(name, self.spec_db, self.mask),
                    atol=1,
                )


class EvaluateDispatchTest(unittest.TestCase):
    def test_all_image_conditions_include_shortcuts(self) -> None:
        for name in SHORTCUT_ORDER:
            self.assertIn(name, evaluate.ALL_IMAGE_CONDITIONS)

    def test_payload_dispatch(self) -> None:
        payload = evaluate._payload_for_condition("full")
        self.assertEqual(payload["condition"], "full")
        self.assertIn("mask_keep", payload["params"])


class ShortcutExportTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.repo_root = Path(self.temp_dir.name)
        self.clip = self.repo_root / "tone.wav"
        _write_wav(self.clip, _tone_then_silence(), SAMPLE_RATE)
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
            self.repo_root / "data" / "derived" / "e1_4_shortcuts"
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_export_layout_and_columns(self) -> None:
        conditions = ("full", "speech_only", "nonspeech_only")
        exports = export_shortcut_split(
            self.records,
            conditions=conditions,
            split="dev",
            output_root=self.output_root,
            repo_root=self.repo_root,
            workers=1,
            show_progress=False,
        )
        self.assertEqual(set(exports), set(conditions))
        export = exports["full"]
        self.assertEqual(export.counts["total"], 2)
        self.assertEqual(export.counts["bonafide"], 1)
        self.assertEqual(export.counts["spoof"], 1)
        self.assertEqual(list(export.frame.columns), list(SHORTCUT_MANIFEST_COLUMNS))
        self.assertEqual(int(export.frame["n_speech_frames"].sum()), 2 * export.frame["n_speech_frames"].iloc[0])
        # n_speech + n_nonspeech == 126 for every clip.
        self.assertTrue(
            (export.frame["n_speech_frames"] + export.frame["n_nonspeech_frames"] == 126).all()
        )
        for path in (
            export.split_dir / "bonafide" / "sample_0.png",
            export.split_dir / "spoofed" / "sample_1.png",
        ):
            self.assertTrue(path.is_file())

        metadata = metadata_frame(exports["full"])
        self.assertEqual(
            list(metadata.columns), ["speaker_id", "system_id", "label", "img_path"]
        )
        metadata_path = write_metadata(exports["full"])
        self.assertTrue(metadata_path.is_file())

    def test_determinism_check(self) -> None:
        determinism = verify_shortcut_determinism(
            "nonspeech_only", self.records, sample=2
        )
        self.assertEqual(determinism["checked"], 2)
        self.assertEqual(determinism["matched"], 2)


class FeatureProbeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp_dir.name)
        cls.clip = cls.root / "tone.wav"
        _write_wav(cls.clip, _tone(), SAMPLE_RATE)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp_dir.cleanup()

    def _records(self, n: int = 4) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "speaker_id": [f"s{i}" for i in range(n)],
                "system_id": ["-", "A01"] * (n // 2),
                "label": [0, 1] * (n // 2),
                "source_path": [str(self.clip)] * n,
            }
        )

    def test_order(self) -> None:
        self.assertEqual(
            FEATURE_PROBE_ORDER,
            ("duration_only", "metadata_system_id", "metadata_speaker_id"),
        )

    def test_duration_feature_shape(self) -> None:
        records = self._records(4)
        x, cats = feature_matrix("duration_only", records)
        self.assertEqual(x.shape, (4, 1))
        self.assertIsNone(cats)
        self.assertTrue(np.all(np.isfinite(x)))

    def test_system_id_feature(self) -> None:
        records = self._records(4)
        x, cats = feature_matrix("metadata_system_id", records)
        self.assertIsNone(cats)
        np.testing.assert_array_equal(x.reshape(-1), [0.0, 1.0, 0.0, 1.0])

    def test_speaker_id_feature_one_hot(self) -> None:
        records = self._records(4)
        x, cats = feature_matrix("metadata_speaker_id", records)
        self.assertEqual(cats, ["s0", "s1", "s2", "s3"])
        self.assertEqual(x.shape, (4, 4))
        np.testing.assert_array_equal(x, np.eye(4))

    def test_payload(self) -> None:
        payload = feature_probe_payload("duration_only")
        self.assertEqual(payload["condition"], "duration_only")
        self.assertEqual(payload["kind"], "feature_probe")

    def test_end_to_end_system_id_probe(self) -> None:
        train = self._records(4)
        dev = self._records(4)
        eval_records = self._records(4)
        out_dir = self.root / "probe"
        result = train_feature_probe(
            "metadata_system_id",
            train,
            dev,
            output_dir=out_dir,
            seed=42,
            device="cpu",
        )
        self.assertIsNotNone(result.best_dev_auc)
        self.assertGreater(result.best_dev_auc, 0.9)

        metrics_path = self.root / "metrics.json"
        metrics, written = evaluate_feature_probe(
            "metadata_system_id",
            out_dir,
            dev,
            eval_records,
            42,
            metrics_path=metrics_path,
            device="cpu",
        )
        self.assertTrue(written.is_file())
        eval_auc = metrics["splits"]["eval"]["metrics"]["auc"]
        self.assertGreater(eval_auc, 0.9)


class SummarizerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import importlib.util

        path = REPO_ROOT / "scripts" / "summarize_e1_4.py"
        spec = importlib.util.spec_from_file_location("summarize_e1_4", path)
        assert spec.loader is not None
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def _metrics(self, eer: float = 0.1, auc: float = 0.95) -> dict[str, object]:
        return {
            "thresholds": {"dev_eer": 0.5},
            "checkpoint": {"sha256": "abc"},
            "splits": {
                "dev": {
                    "attack_subset": "A01-A06",
                    "metrics": {
                        "sample_counts": {"total": 10, "bonafide": 4, "spoof": 6},
                        "eer": eer,
                        "auc": auc,
                        "threshold_metrics": {
                            "fixed_0.5": {
                                "false_positive_rate": 0.2,
                                "spoof_recall": 0.8,
                            }
                        },
                        "per_attack": {},
                    },
                }
            },
        }

    def test_image_rows_keep_metric_contract(self) -> None:
        rows = self.module._image_rows("full", 42, self._metrics(), Path("m.json"))
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["condition"], "full")
        self.assertEqual(row["keep"], "all")
        self.assertEqual(row["split"], "dev")
        self.assertEqual(row["attack_subset"], "A01-A06")
        self.assertEqual(row["samples_total"], 10)
        self.assertAlmostEqual(row["threshold_free_eer"], 0.1)
        self.assertAlmostEqual(row["fpr_at_0.5"], 0.2)

    def test_feature_rows(self) -> None:
        rows = self.module._feature_rows(
            "duration_only", 42, self._metrics(), Path("m.json")
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["probe"], "duration_only")
        self.assertAlmostEqual(rows[0]["threshold_free_auc"], 0.95)

    def test_markdown_mentions_threshold_free(self) -> None:
        summary = pd.DataFrame(
            self.module._image_rows("full", 42, self._metrics(), Path("m.json")),
            columns=self.module.SUMMARY_COLUMNS,
        )
        features = pd.DataFrame(
            self.module._feature_rows(
                "duration_only", 42, self._metrics(), Path("m.json")
            ),
            columns=self.module.FEATURE_COLUMNS,
        )
        per_attack = pd.DataFrame(columns=self.module.PER_ATTACK_COLUMNS)
        markdown = self.module._results_markdown(summary, features, per_attack)
        self.assertIn("threshold-free", markdown)
        self.assertIn("full", markdown)
        self.assertIn("duration_only", markdown)


if __name__ == "__main__":
    unittest.main()