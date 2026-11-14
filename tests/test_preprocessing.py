"""Contract tests for the audio-to-image preprocessing pipeline.

These tests lock the verified chain in AGENTS.md section 4.2.  They use
synthetic audio for deterministic unit checks and a few real dev clips for
pixel parity against the stored export.
"""

from __future__ import annotations

import os

# Rendering must never require a display.
os.environ.setdefault("MPLBACKEND", "Agg")

import sys
import tempfile
import unittest
import wave
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from spoof_detection.preprocess import (  # noqa: E402
    COLORMAP,
    IMAGENET_MEAN,
    IMAGENET_STD,
    INFERENCE_SIZE,
    N_MELS,
    PREPROCESSING_VERSION,
    SAMPLE_RATE,
    TARGET_LENGTH,
    TARGET_SECONDS,
    build_inference_transform,
    colorize_mel,
    generate_mel_spectrogram,
    load_fixed_audio,
    load_mel_image,
    normalize_mel_db,
    render_mel_rgb,
)
from spoof_detection.protocol import load_cm_protocol  # noqa: E402

DEV_PROTOCOL = (
    REPO_ROOT
    / "data"
    / "LA"
    / "ASVspoof2019_LA_cm_protocols"
    / "ASVspoof2019.LA.cm.dev.trl.txt"
)
DEV_IMAGE_ROOT = REPO_ROOT / "data" / "dataset_img" / "dev"


def _tone(seconds: float, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
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


class AudioContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp_dir.name)
        cls.short = cls.root / "short.wav"
        cls.long = cls.root / "long.wav"
        cls.eight_khz = cls.root / "eight_khz.wav"
        cls.silence = cls.root / "silence.wav"
        _write_wav(cls.short, _tone(1.0), SAMPLE_RATE)
        _write_wav(cls.long, _tone(5.0), SAMPLE_RATE)
        _write_wav(cls.eight_khz, _tone(1.0, sample_rate=8_000), 8_000)
        _write_wav(cls.silence, np.zeros(SAMPLE_RATE, dtype=np.float32), SAMPLE_RATE)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp_dir.cleanup()

    def test_short_audio_is_zero_padded(self) -> None:
        loaded, loaded_sr = librosa.load(str(self.short), sr=SAMPLE_RATE)
        fixed, sr = load_fixed_audio(self.short)
        self.assertEqual(sr, SAMPLE_RATE)
        self.assertEqual(loaded_sr, SAMPLE_RATE)
        self.assertEqual(fixed.shape, (TARGET_LENGTH,))
        np.testing.assert_array_equal(fixed[: len(loaded)], loaded)
        self.assertTrue(np.all(fixed[len(loaded) :] == 0.0))

    def test_long_audio_is_truncated(self) -> None:
        loaded, _ = librosa.load(str(self.long), sr=SAMPLE_RATE)
        fixed, _ = load_fixed_audio(self.long)
        self.assertGreater(len(loaded), TARGET_LENGTH)
        self.assertEqual(fixed.shape, (TARGET_LENGTH,))
        np.testing.assert_array_equal(fixed, loaded[:TARGET_LENGTH])

    def test_eight_khz_audio_is_resampled_and_padded(self) -> None:
        fixed, sr = load_fixed_audio(self.eight_khz)
        self.assertEqual(sr, SAMPLE_RATE)
        self.assertEqual(fixed.shape, (TARGET_LENGTH,))
        self.assertGreater(float(np.abs(fixed).sum()), 0.0)
        self.assertTrue(np.all(fixed[SAMPLE_RATE:] == 0.0))

    def test_mel_shape_and_db_range(self) -> None:
        spec = generate_mel_spectrogram(self.short)
        self.assertEqual(spec.shape, (N_MELS, 126))
        self.assertTrue(np.isfinite(spec).all())
        self.assertAlmostEqual(float(spec.max()), 0.0, places=6)
        self.assertGreaterEqual(float(spec.min()), -80.0001)

    def test_mel_n_mels_is_honoured(self) -> None:
        spec = generate_mel_spectrogram(self.short, n_mels=64)
        self.assertEqual(spec.shape, (64, 126))

    def test_mel_matches_direct_librosa_computation(self) -> None:
        y_fixed, sr = load_fixed_audio(self.short)
        mel = librosa.feature.melspectrogram(
            y=y_fixed,
            sr=sr,
            n_mels=N_MELS,
        )
        expected = librosa.power_to_db(mel, ref=np.max)
        actual = generate_mel_spectrogram(self.short)
        np.testing.assert_array_equal(actual, expected)

    def test_mel_is_deterministic(self) -> None:
        first = generate_mel_spectrogram(self.short)
        second = generate_mel_spectrogram(self.short)
        np.testing.assert_array_equal(first, second)

    def test_silence_is_finite(self) -> None:
        spec = generate_mel_spectrogram(self.silence)
        self.assertTrue(np.isfinite(spec).all())
        self.assertEqual(float(spec.min()), float(spec.max()))


class NormalizationTest(unittest.TestCase):
    def test_min_max_maps_to_unit_interval(self) -> None:
        spec = np.array([[-80.0, -40.0], [0.0, -20.0]])
        normalized = normalize_mel_db(spec)
        self.assertAlmostEqual(float(normalized.min()), 0.0)
        self.assertAlmostEqual(float(normalized.max()), 1.0)
        self.assertAlmostEqual(float(normalized[1, 0]), 1.0)
        self.assertAlmostEqual(float(normalized[0, 0]), 0.0)
        self.assertTrue(np.isfinite(normalized).all())

    def test_constant_spectrogram_maps_to_zeros(self) -> None:
        for value in (0.0, -12.5):
            spec = np.full((4, 5), value)
            normalized = normalize_mel_db(spec)
            np.testing.assert_array_equal(normalized, np.zeros_like(spec))

    def test_normalization_preserves_shape(self) -> None:
        spec = np.arange(12, dtype=np.float32).reshape(3, 4)
        self.assertEqual(normalize_mel_db(spec).shape, spec.shape)


class RenderingTest(unittest.TestCase):
    def test_colormap_and_vertical_flip(self) -> None:
        normalized = np.array([[0.0], [1.0]])
        rgb = colorize_mel(normalized)
        low = (np.asarray(plt.get_cmap("magma")(0.0))[:3] * 255).astype(np.uint8)
        high = (np.asarray(plt.get_cmap("magma")(1.0))[:3] * 255).astype(np.uint8)
        np.testing.assert_array_equal(rgb[0, 0], high)
        np.testing.assert_array_equal(rgb[1, 0], low)

    def test_colormap_parameter_is_honoured(self) -> None:
        normalized = np.array([[0.0, 0.5, 1.0]])
        magma = colorize_mel(normalized, cmap="magma")
        viridis = colorize_mel(normalized, cmap="viridis")
        self.assertFalse(np.array_equal(magma, viridis))
        expected = (
            np.asarray(plt.get_cmap("viridis")(normalized))[0, :, :3] * 255
        ).astype(np.uint8)
        np.testing.assert_array_equal(viridis[0], expected)

    def test_render_mel_rgb_combines_normalization_and_colormap(self) -> None:
        rng = np.random.default_rng(0)
        spec = rng.uniform(-80.0, 0.0, size=(16, 20))
        expected = colorize_mel(normalize_mel_db(spec))
        np.testing.assert_array_equal(render_mel_rgb(spec), expected)

    def test_render_mel_rgb_shape_and_dtype(self) -> None:
        spec = np.zeros((N_MELS, 126))
        rgb = render_mel_rgb(spec)
        self.assertEqual(rgb.shape, (N_MELS, 126, 3))
        self.assertEqual(rgb.dtype, np.uint8)
        self.assertGreaterEqual(int(rgb.min()), 0)
        self.assertLessEqual(int(rgb.max()), 255)


class InferenceTransformTest(unittest.TestCase):
    def test_resize_and_imagenet_normalization(self) -> None:
        rgb = (10, 20, 30)
        image = Image.new("RGB", (126, 128), rgb)
        tensor = build_inference_transform()(image)
        self.assertEqual(tuple(tensor.shape), (3, INFERENCE_SIZE, INFERENCE_SIZE))
        self.assertEqual(tensor.dtype, torch.float32)
        for channel, (value, mean, std) in enumerate(
            zip(rgb, IMAGENET_MEAN, IMAGENET_STD)
        ):
            expected = (value / 255.0 - mean) / std
            self.assertTrue(
                torch.allclose(
                    tensor[channel],
                    torch.full((INFERENCE_SIZE, INFERENCE_SIZE), expected),
                )
            )

    def test_contract_constants(self) -> None:
        self.assertEqual(SAMPLE_RATE, 16_000)
        self.assertEqual(TARGET_SECONDS, 4)
        self.assertEqual(TARGET_LENGTH, 64_000)
        self.assertEqual(N_MELS, 128)
        self.assertEqual(COLORMAP, "magma")
        self.assertEqual(INFERENCE_SIZE, 224)
        self.assertEqual(PREPROCESSING_VERSION, "mel128-magma-4s-v1")


class ExportParityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp_dir.name)
        cls.silence = cls.root / "silence.wav"
        _write_wav(cls.silence, np.zeros(SAMPLE_RATE, dtype=np.float32), SAMPLE_RATE)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp_dir.cleanup()

    def test_silence_render_matches_export_path(self) -> None:
        spec_db = generate_mel_spectrogram(self.silence)
        app_rgb = render_mel_rgb(spec_db)
        export_rgb = np.asarray(load_mel_image(self.silence).convert("RGB"))
        np.testing.assert_array_equal(app_rgb, export_rgb)

    def test_real_clip_render_matches_export_and_stored_png(self) -> None:
        if not DEV_PROTOCOL.is_file():
            self.skipTest("ASVspoof dev protocol is not available")
        protocol = load_cm_protocol("dev")
        for protocol_row in (0, 1_000, 2_548, 20_000):
            row = protocol.iloc[protocol_row]
            audio_path = REPO_ROOT / row["source_flac"]
            if not audio_path.is_file():
                self.skipTest("ASVspoof dev audio is not available")

            spec_db = generate_mel_spectrogram(audio_path)
            app_rgb = render_mel_rgb(spec_db)
            export_rgb = np.asarray(
                load_mel_image(audio_path).convert("RGB")
            )
            np.testing.assert_array_equal(
                app_rgb,
                export_rgb,
                err_msg=f"app rendering differs from export for row {protocol_row}",
            )

            folder = "bonafide" if int(row["label"]) == 0 else "spoofed"
            stored_path = (
                DEV_IMAGE_ROOT / folder / f"sample_{protocol_row}.png"
            )
            if stored_path.is_file():
                stored_rgb = np.asarray(Image.open(stored_path).convert("RGB"))
                np.testing.assert_array_equal(
                    app_rgb,
                    stored_rgb,
                    err_msg=f"app rendering differs from {stored_path}",
                )


if __name__ == "__main__":
    unittest.main()
