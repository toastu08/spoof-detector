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
    sha256_file,
)

UNWEIGHTED_CHECKPOINT = (
    REPO_ROOT
    / "Models"
    / "Itteration1"
    / "best_audio_spoof_model_unweighted.pth"
)
WEIGHTED_CHECKPOINT = (
    REPO_ROOT
    / "Models"
    / "Itteration1"
    / "best_audio_spoof_model_weighted.pth"
)
ROOT_UNWEIGHTED_CHECKPOINT = (
    REPO_ROOT / "best_audio_spoof_model_unweighted.pth"
)

# Frozen by E0.3; see reports/baseline/freeze_report.json.
EXPECTED_HEAD_SHA256 = {
    "unweighted": "de07fcb49e82ce28add233e10753f8885bfb1a11e0177cb8b89bcb6fc11685d4",
    "weighted": "48c383336ddd257763adbbf00c94bd0a9047f4597375b7dfa3f683cf9422382a",
}
EXPECTED_HEAD_BIAS = {
    "unweighted": -0.234170,
    "weighted": -0.500105,
}
EXPECTED_BATCH_NORM_STEPS = {
    "unweighted": (4_095,),
    "weighted": (6_930,),
}


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
        self.assertEqual(len(report.head_state_sha256), 64)
        self.assertIsInstance(report.head_last_bias, float)
        self.assertEqual(report.batch_norm_steps, (0,))
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

    def test_only_exact_network_prefix_is_remapped(self) -> None:
        tensor = torch.zeros(1)
        remapped, count = remap_state_dict_keys(
            {
                "network.conv1.weight": tensor,
                "network_conv1.weight": tensor,
            }
        )
        self.assertEqual(count, 1)
        self.assertIn("backbone.conv1.weight", remapped)
        self.assertIn("network_conv1.weight", remapped)

    def test_broken_prefix_raises_instead_of_loading_silently(self) -> None:
        state_dict = dict(self.prefixed_state_dict)
        state_dict["network_conv1.weight"] = state_dict.pop(
            "network.conv1.weight"
        )
        path = self.root / "broken-prefix.pth"
        torch.save(state_dict, path)
        with self.assertRaises(CheckpointError) as context:
            load_model_from_checkpoint(path)
        self.assertIn("missing keys", str(context.exception))
        self.assertIn("unexpected keys", str(context.exception))


class RealCheckpointContractTest(unittest.TestCase):
    """Contract tests against the four frozen first-iteration artifacts."""

    def _load(self, path: Path):
        if not path.is_file():
            self.skipTest(f"checkpoint not available: {path}")
        return load_model_from_checkpoint(path, device="cpu")

    def _assert_common_contract(self, report) -> None:
        self.assertEqual(report.state_dict_keys, 124)
        self.assertEqual(report.remapped_keys, 124)
        self.assertEqual(report.missing_keys, ())
        self.assertEqual(report.unexpected_keys, ())
        self.assertEqual(report.total_parameters, TOTAL_PARAMETERS)
        self.assertEqual(report.head_parameters, HEAD_PARAMETERS)
        self.assertEqual(
            report.backbone_parameters,
            TOTAL_PARAMETERS - HEAD_PARAMETERS,
        )
        self.assertEqual(len(report.head_state_sha256), 64)
        self.assertEqual(len(report.sha256), 64)

    def _assert_named_checkpoint(self, name: str, path: Path) -> None:
        model, report = self._load(path)
        self._assert_common_contract(report)
        self.assertEqual(report.head_state_sha256, EXPECTED_HEAD_SHA256[name])
        self.assertAlmostEqual(
            report.head_last_bias,
            EXPECTED_HEAD_BIAS[name],
            delta=1e-5,
        )
        self.assertEqual(
            report.batch_norm_steps,
            EXPECTED_BATCH_NORM_STEPS[name],
        )
        self.assertFalse(model.training)
        self.assertTrue(
            all(not parameter.requires_grad for parameter in model.parameters())
        )

    def test_unweighted_checkpoint(self) -> None:
        self._assert_named_checkpoint("unweighted", UNWEIGHTED_CHECKPOINT)

    def test_weighted_checkpoint(self) -> None:
        self._assert_named_checkpoint("weighted", WEIGHTED_CHECKPOINT)

    def test_root_checkpoint_strict_loads(self) -> None:
        self._assert_named_checkpoint("unweighted", ROOT_UNWEIGHTED_CHECKPOINT)

    def test_root_checkpoint_is_byte_identical_to_models_copy(self) -> None:
        if not (
            ROOT_UNWEIGHTED_CHECKPOINT.is_file()
            and UNWEIGHTED_CHECKPOINT.is_file()
        ):
            self.skipTest("root or Models unweighted checkpoint is not available")
        self.assertEqual(
            sha256_file(ROOT_UNWEIGHTED_CHECKPOINT),
            sha256_file(UNWEIGHTED_CHECKPOINT),
        )


if __name__ == "__main__":
    unittest.main()
