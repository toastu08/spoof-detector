"""Tests for the E1.1 time-frequency front-end registry.

These lock the representation shapes and the mel128 parity contract.  ``mel128``
must stay pixel-identical to the existing ``data/dataset_img`` export, so it
reuses :mod:`spoof_detection.preprocess` verbatim.
"""

from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import sys
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from spoof_detection.frontends import (  # noqa: E402
    FRONTEND_ORDER,
    FRONTENDS,
    compute_frontend,
    frontend_payload,
    get_frontend_spec,
    random_projection_matrix,
    random_projection_sha256,
    render_frontend_rgb,
    save_frontend_png,
)
from spoof_detection.preprocess import (  # noqa: E402
    PREPROCESSING_VERSION,
    SAMPLE_RATE,
    generate_mel_spectrogram,
)
from spoof_detection.protocol import load_cm_protocol  # noqa: E402

EXPECTED_SHAPES = {
    "mel64": (64, 126),
    "mel128": (128, 126),
    "mel256": (256, 126),
    "linear_stft": (1025, 126),
    "mfcc40": (40, 126),
    "randproj128": (128, 126),
}

DEV_PROTOCOL = (
    REPO_ROOT
    / "data"
    / "LA"
    / "ASVspoof2019_LA_cm_protocols"
    / "ASVspoof2019.LA.cm.dev.trl.txt"
)
DEV_IMAGE_ROOT = REPO_ROOT / "data" / "dataset_img" / "dev"


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


class FrontendRegistryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp_dir.name)
        cls.clip = cls.root / "tone.wav"
        _write_wav(cls.clip, _tone(), SAMPLE_RATE)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp_dir.cleanup()

    def test_order_covers_all_registered_frontends(self) -> None:
        self.assertEqual(set(FRONTEND_ORDER), set(FRONTENDS))
        self.assertEqual(len(FRONTEND_ORDER), 6)

    def test_compute_frontend_shapes(self) -> None:
        for name in FRONTEND_ORDER:
            with self.subTest(frontend=name):
                representation = compute_frontend(name, self.clip)
                self.assertEqual(representation.shape, EXPECTED_SHAPES[name])
                self.assertEqual(
                    representation.shape, FRONTENDS[name].expected_shape
                )
                self.assertTrue(np.isfinite(representation).all())

    def test_mel128_matches_preprocess_exactly(self) -> None:
        expected = generate_mel_spectrogram(self.clip)
        actual = compute_frontend("mel128", self.clip)
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(
            FRONTENDS["mel128"].version, PREPROCESSING_VERSION
        )

    def test_unknown_frontend_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            get_frontend_spec("not_a_frontend")
        with self.assertRaises(ValueError):
            compute_frontend("not_a_frontend", self.clip)

    def test_render_rgb_matches_export_orientation(self) -> None:
        rgb = render_frontend_rgb("mel64", self.clip)
        self.assertEqual(rgb.shape, (64, 126, 3))
        self.assertEqual(rgb.dtype, np.uint8)

    def test_save_png_dimensions(self) -> None:
        output = self.root / "mel64.png"
        save_frontend_png("mel64", self.clip, output)
        with Image.open(output) as image:
            width, height = image.size
        self.assertEqual((height, width), (64, 126))

    def test_save_png_is_deterministic(self) -> None:
        first = self.root / "det_a.png"
        second = self.root / "det_b.png"
        save_frontend_png("linear_stft", self.clip, first)
        save_frontend_png("linear_stft", self.clip, second)
        self.assertEqual(first.read_bytes(), second.read_bytes())

    def test_random_projection_is_fixed_and_seeded(self) -> None:
        first = random_projection_matrix()
        second = random_projection_matrix()
        np.testing.assert_array_equal(first, second)
        other = random_projection_matrix(seed=1)
        self.assertFalse(np.array_equal(first, other))
        self.assertEqual(
            random_projection_sha256(), random_projection_sha256()
        )
        self.assertNotEqual(
            random_projection_sha256(), random_projection_sha256(seed=1)
        )

    def test_frontend_payload_contract(self) -> None:
        for name in FRONTEND_ORDER:
            with self.subTest(frontend=name):
                payload = frontend_payload(name)
                self.assertEqual(payload["frontend"], name)
                self.assertEqual(payload["version"], FRONTENDS[name].version)
                self.assertIn("params", payload)
        rand = frontend_payload("randproj128")
        self.assertEqual(
            rand["random_projection_sha256"], random_projection_sha256()
        )


@unittest.skipUnless(
    DEV_PROTOCOL.is_file() and DEV_IMAGE_ROOT.is_dir(),
    "ASVspoof dev audio and the stored mel128 export are required",
)
class Mel128ParityTest(unittest.TestCase):
    def test_two_dev_clips_are_pixel_identical(self) -> None:
        protocol = load_cm_protocol(
            "dev", data_root=REPO_ROOT / "data" / "LA", repo_root=REPO_ROOT
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            for row in protocol.head(2).itertuples(index=False):
                label = int(row.label)
                folder = "bonafide" if label == 0 else "spoofed"
                source = REPO_ROOT / row.source_flac
                reference = (
                    DEV_IMAGE_ROOT / folder / f"sample_{int(row.protocol_row)}.png"
                )
                if not source.is_file() or not reference.is_file():
                    self.skipTest("sample dev clip is unavailable")
                generated = temp_root / "generated.png"
                save_frontend_png("mel128", source, generated)
                with Image.open(generated) as image:
                    generated_rgb = np.asarray(image.convert("RGB"))
                with Image.open(reference) as image:
                    reference_rgb = np.asarray(image.convert("RGB"))
                np.testing.assert_array_equal(generated_rgb, reference_rgb)


if __name__ == "__main__":
    unittest.main()
