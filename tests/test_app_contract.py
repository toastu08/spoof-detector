"""Contract tests for the FastAPI inference endpoint.

The endpoint is called as a coroutine with a real ``UploadFile`` and a stub
model.  This keeps the tests fast and deterministic while still exercising the
app's upload validation, preprocessing integration, response contract, and
temporary-file cleanup.
"""

from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import asyncio
import base64
import io
import sys
import tempfile
import unittest
import wave
from pathlib import Path

import torch
from fastapi import HTTPException
from PIL import Image
from starlette.datastructures import UploadFile

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

ROOT_CHECKPOINT = REPO_ROOT / "best_audio_spoof_model_unweighted.pth"

# Skip the module when the local checkpoint is absent.  If the checkpoint exists
# but fails the strict load, importing app raises and the test run fails loudly.
if ROOT_CHECKPOINT.is_file():
    import app as app_module
else:  # pragma: no cover - exercised only when the checkpoint is absent
    app_module = None


def _wav_bytes(seconds: float = 1.0) -> bytes:
    sample_rate = 16_000
    count = int(seconds * sample_rate)
    t = torch.arange(count, dtype=torch.float32) / sample_rate
    samples = (0.1 * torch.sin(2.0 * torch.pi * 440.0 * t)).numpy()
    pcm = (samples * 32767.0).round().astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())
    return buffer.getvalue()


class _StubModel:
    """Return a fixed logit independent of the input tensor."""

    def __init__(self, logit: float) -> None:
        self.logit = float(logit)
        self.training = True

    def eval(self) -> "_StubModel":
        self.training = False
        return self

    def to(self, device: torch.device) -> "_StubModel":
        return self

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.full((tensor.shape[0], 1), self.logit, dtype=torch.float32)


@unittest.skipIf(
    app_module is None,
    "root checkpoint is not available; app contract tests require a loadable model",
)
class AppContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.original_tempdir = tempfile.tempdir
        tempfile.tempdir = self.temp_dir.name

        self.original_model = app_module.model
        self.original_device = app_module.device
        self.original_process = app_module.process_audio_to_tensor
        app_module.device = torch.device("cpu")
        app_module.model = _StubModel(0.2)

    def tearDown(self) -> None:
        app_module.model = self.original_model
        app_module.device = self.original_device
        app_module.process_audio_to_tensor = self.original_process
        tempfile.tempdir = self.original_tempdir
        self.temp_dir.cleanup()

    def _upload(self, filename: str, data: bytes = b"not audio") -> UploadFile:
        return UploadFile(file=io.BytesIO(data), filename=filename)

    def _call(self, upload: UploadFile):
        return asyncio.run(app_module.predict_audio(upload))

    def _assert_tempdir_empty(self) -> None:
        self.assertEqual(os.listdir(self.temp_dir.name), [])

    def test_rejects_unsupported_extension(self) -> None:
        with self.assertRaises(HTTPException) as context:
            self._call(self._upload("clip.txt"))
        self.assertEqual(context.exception.status_code, 400)
        self.assertEqual(context.exception.detail, "Invalid file type.")
        self._assert_tempdir_empty()

    def test_rejects_all_unsupported_extensions_before_tempfile(self) -> None:
        for filename in ("clip.txt", "clip.mp4", "clip", "clip.flac.txt"):
            with self.subTest(filename=filename):
                with self.assertRaises(HTTPException) as context:
                    self._call(self._upload(filename))
                self.assertEqual(context.exception.status_code, 400)
                self._assert_tempdir_empty()

    def test_supported_extensions_reach_processing(self) -> None:
        def fail(_path: str) -> None:
            raise RuntimeError("processing failed")

        app_module.process_audio_to_tensor = fail
        for filename in ("clip.flac", "clip.wav", "clip.mp3"):
            with self.subTest(filename=filename):
                with self.assertRaises(HTTPException) as context:
                    self._call(self._upload(filename))
                self.assertEqual(context.exception.status_code, 500)
                self.assertEqual(context.exception.detail, "processing failed")
                self._assert_tempdir_empty()

    def test_processing_failure_cleans_temporary_file(self) -> None:
        def fail(_path: str) -> None:
            raise RuntimeError("boom")

        app_module.process_audio_to_tensor = fail
        with self.assertRaises(HTTPException) as context:
            self._call(self._upload("clip.wav"))
        self.assertEqual(context.exception.status_code, 500)
        self.assertEqual(context.exception.detail, "boom")
        self._assert_tempdir_empty()

    def test_success_response_contract_and_cleanup(self) -> None:
        response = self._call(
            self._upload("clip.wav", _wav_bytes())
        )

        probability = torch.sigmoid(torch.tensor(0.2)).item()
        expected_confidence = round(max(probability, 1.0 - probability) * 100, 2)

        self.assertEqual(
            set(response),
            {"prediction", "confidence", "raw_probability", "spectrogram"},
        )
        self.assertEqual(response["prediction"], "AI Generated (Spoof)")
        self.assertEqual(response["confidence"], f"{expected_confidence}%")
        self.assertEqual(
            response["raw_probability"],
            round(probability, 4),
        )

        png = base64.b64decode(response["spectrogram"])
        with Image.open(io.BytesIO(png)) as image:
            self.assertEqual(image.size, (126, 128))
            self.assertEqual(image.mode, "RGB")

        self._assert_tempdir_empty()

    def test_bonafide_response_contract(self) -> None:
        app_module.model = _StubModel(-0.2)
        response = self._call(
            self._upload("clip.wav", _wav_bytes())
        )

        probability = torch.sigmoid(torch.tensor(-0.2)).item()
        expected_confidence = round(max(probability, 1.0 - probability) * 100, 2)

        self.assertEqual(response["prediction"], "Real Human (Bonafide)")
        self.assertEqual(response["confidence"], f"{expected_confidence}%")
        self.assertEqual(response["raw_probability"], round(probability, 4))
        self._assert_tempdir_empty()

    def test_threshold_boundary_is_spoof(self) -> None:
        app_module.model = _StubModel(0.0)
        response = self._call(
            self._upload("clip.wav", _wav_bytes())
        )
        self.assertEqual(response["prediction"], "AI Generated (Spoof)")
        self.assertEqual(response["confidence"], "50.0%")
        self.assertEqual(response["raw_probability"], 0.5)

    def test_app_loaded_a_strict_checkpoint(self) -> None:
        report = app_module.model_load_report
        self.assertEqual(report.missing_keys, ())
        self.assertEqual(report.unexpected_keys, ())
        self.assertEqual(report.state_dict_keys, 124)
        self.assertEqual(report.total_parameters, 11_242_305)
        self.assertEqual(report.head_parameters, 65_793)


if __name__ == "__main__":
    unittest.main()
