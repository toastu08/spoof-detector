"""Tests for the shared model definition and strict checkpoint loading."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from spoof_detection.model import (  # noqa: E402
    HEAD_PARAMETERS,
    TOTAL_PARAMETERS,
    AudioSpoofDetector,
    CheckpointError,
    load_model_from_checkpoint,
    remap_state_dict_keys,
)


class CheckpointLoadingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.manual_seed(0)
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp_dir.name)
        cls.model = AudioSpoofDetector()
        cls.prefixed_state_dict = {
            key.replace("backbone.", "network.", 1): value
            for key, value in cls.model.state_dict().items()
        }
        cls.checkpoint_path = cls.root / "synthetic.pth"
        torch.save(cls.prefixed_state_dict, cls.checkpoint_path)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp_dir.cleanup()

    def test_remaps_network_prefix(self) -> None:
        remapped, count = remap_state_dict_keys(self.prefixed_state_dict)
        self.assertEqual(count, len(self.prefixed_state_dict))
        self.assertTrue(all(key.startswith("backbone.") for key in remapped))
        self.assertNotIn("network.conv1.weight", remapped)
        self.assertIn("backbone.conv1.weight", remapped)

    def test_strict_roundtrip(self) -> None:
        model, report = load_model_from_checkpoint(self.checkpoint_path)
        self.assertEqual(report.missing_keys, ())
        self.assertEqual(report.unexpected_keys, ())
        self.assertEqual(report.remapped_keys, report.state_dict_keys)
        self.assertEqual(report.total_parameters, TOTAL_PARAMETERS)
        self.assertEqual(report.head_parameters, HEAD_PARAMETERS)
        self.assertEqual(
            report.backbone_parameters,
            TOTAL_PARAMETERS - HEAD_PARAMETERS,
        )
        self.assertFalse(model.training)
        self.assertTrue(
            all(not parameter.requires_grad for parameter in model.parameters())
        )

    def test_unprefixed_state_dict_is_accepted(self) -> None:
        path = self.root / "unprefixed.pth"
        torch.save(self.model.state_dict(), path)
        model, report = load_model_from_checkpoint(path)
        self.assertEqual(report.missing_keys, ())
        self.assertEqual(report.unexpected_keys, ())
        self.assertEqual(report.remapped_keys, 0)
        self.assertIsInstance(model, AudioSpoofDetector)

    def test_missing_key_raises(self) -> None:
        state_dict = dict(self.prefixed_state_dict)
        del state_dict["network.fc.3.weight"]
        path = self.root / "missing.pth"
        torch.save(state_dict, path)
        with self.assertRaises(CheckpointError) as context:
            load_model_from_checkpoint(path)
        self.assertIn("missing keys", str(context.exception))
        self.assertIn("fc.3.weight", str(context.exception))

    def test_unexpected_key_raises(self) -> None:
        state_dict = dict(self.prefixed_state_dict)
        state_dict["network.unexpected"] = torch.zeros(1)
        path = self.root / "unexpected.pth"
        torch.save(state_dict, path)
        with self.assertRaises(CheckpointError) as context:
            load_model_from_checkpoint(path)
        self.assertIn("unexpected keys", str(context.exception))
        self.assertIn("unexpected", str(context.exception))

    def test_missing_file_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_model_from_checkpoint(self.root / "does-not-exist.pth")


if __name__ == "__main__":
    unittest.main()
