"""Tests for the E0.3 baseline freeze and audit helpers."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from spoof_detection.baseline import (  # noqa: E402
    audit_checkpoint,
    audit_training_manifest,
    compare_metric_runs,
    evaluate_head_control,
    validate_baseline_contract,
)
from spoof_detection.model import AudioSpoofDetector  # noqa: E402


def _metrics_payload(
    dev_eer: float = 0.06,
    eval_eer: float = 0.11,
    dev_counts: dict[str, int] | None = None,
) -> dict[str, object]:
    return {
        "experiment": "E0.3",
        "partial": False,
        "checkpoint": {
            "sha256": "abc123",
            "missing_keys": [],
            "unexpected_keys": [],
            "total_parameters": 11_242_305,
            "head_parameters": 65_793,
        },
        "thresholds": {"dev_eer": 0.84, "selected_on": "dev"},
        "splits": {
            "dev": {
                "attack_subset": "A01-A06",
                "metrics": {
                    "eer": dev_eer,
                    "auc": 0.986,
                    "sample_counts": dev_counts
                    or {"total": 24_844, "bonafide": 2_548, "spoof": 22_296},
                },
            },
            "eval": {
                "attack_subset": "A07-A19",
                "metrics": {
                    "eer": eval_eer,
                    "auc": 0.953,
                    "sample_counts": {
                        "total": 71_237,
                        "bonafide": 7_355,
                        "spoof": 63_882,
                    },
                },
            },
        },
    }


class CompareMetricRunsTest(unittest.TestCase):
    def test_identical_runs_pass(self) -> None:
        result = compare_metric_runs(_metrics_payload(), _metrics_payload())
        self.assertTrue(result["passed"])
        self.assertEqual(result["splits"]["dev"]["eer_delta"], 0.0)
        self.assertEqual(result["splits"]["eval"]["auc_delta"], 0.0)

    def test_delta_within_one_point_passes(self) -> None:
        result = compare_metric_runs(
            _metrics_payload(),
            _metrics_payload(dev_eer=0.065, eval_eer=0.115),
        )
        self.assertTrue(result["passed"])

    def test_delta_above_one_point_fails(self) -> None:
        result = compare_metric_runs(
            _metrics_payload(),
            _metrics_payload(dev_eer=0.075, eval_eer=0.11),
        )
        self.assertFalse(result["passed"])
        self.assertIn("dev", result["failures"][0])

    def test_missing_split_fails(self) -> None:
        candidate = _metrics_payload()
        del candidate["splits"]["eval"]  # type: ignore[index]
        result = compare_metric_runs(_metrics_payload(), candidate)
        self.assertFalse(result["passed"])
        self.assertTrue(
            any("eval" in failure for failure in result["failures"])
        )

    def test_count_mismatch_fails(self) -> None:
        candidate = _metrics_payload(
            dev_counts={"total": 24_843, "bonafide": 2_548, "spoof": 22_295}
        )
        result = compare_metric_runs(_metrics_payload(), candidate)
        self.assertFalse(result["passed"])
        self.assertIn("sample counts differ", result["failures"][0])


class BaselineContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.audit = {
            "sha256": "abc123",
            "parameter_counts_match": True,
        }

    def test_valid_contract_passes(self) -> None:
        failures = validate_baseline_contract(
            _metrics_payload(),
            self.audit,
        )
        self.assertEqual(failures, [])

    def test_wrong_experiment_fails(self) -> None:
        payload = _metrics_payload()
        payload["experiment"] = "E0.2"
        failures = validate_baseline_contract(payload, self.audit)
        self.assertTrue(any("experiment" in failure for failure in failures))

    def test_wrong_attack_subset_fails(self) -> None:
        payload = _metrics_payload()
        payload["splits"]["dev"]["attack_subset"] = "A01-A05"  # type: ignore[index]
        failures = validate_baseline_contract(payload, self.audit)
        self.assertTrue(
            any("attack subset" in failure for failure in failures)
        )

    def test_threshold_must_be_selected_on_dev(self) -> None:
        payload = _metrics_payload()
        payload["thresholds"]["selected_on"] = "eval"  # type: ignore[index]
        failures = validate_baseline_contract(payload, self.audit)
        self.assertTrue(any("selected on dev" in f for f in failures))


class ManifestAuditTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.path = self.root / "manifest.csv"
        pd.DataFrame(
            {
                "label": [0, 0, 1, 1],
                "speaker_id": ["s1", "s2", "s1", "s2"],
            }
        ).to_csv(self.path, index=False)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_matching_manifest_passes(self) -> None:
        from spoof_detection.dataset import sha256_file

        report = audit_training_manifest(
            self.path,
            expected_sha256=sha256_file(self.path),
            expected_rows=4,
            expected_counts={"bonafide": 2, "spoof": 2},
        )
        self.assertTrue(report["passed"])
        self.assertEqual(report["counts"], {"bonafide": 2, "spoof": 2})

    def test_wrong_hash_fails(self) -> None:
        report = audit_training_manifest(
            self.path,
            expected_sha256="0" * 64,
            expected_rows=4,
            expected_counts={"bonafide": 2, "spoof": 2},
        )
        self.assertFalse(report["passed"])
        self.assertTrue(any("SHA-256" in f for f in report["failures"]))


class CheckpointAuditTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.manual_seed(0)
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp_dir.name)
        model = AudioSpoofDetector()
        prefixed = {
            key.replace("backbone.", "network.", 1): value
            for key, value in model.state_dict().items()
        }
        cls.checkpoint = cls.root / "synthetic.pth"
        torch.save(prefixed, cls.checkpoint)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp_dir.cleanup()

    def test_audit_records_identity_and_counts(self) -> None:
        audit = audit_checkpoint(self.checkpoint)
        self.assertEqual(len(audit["sha256"]), 64)
        self.assertEqual(len(audit["head_state_sha256"]), 64)
        self.assertEqual(audit["state_dict_keys"], 124)
        self.assertEqual(audit["remapped_keys"], 124)
        self.assertEqual(audit["total_parameters"], 11_242_305)
        self.assertEqual(audit["head_parameters"], 65_793)
        self.assertTrue(audit["parameter_counts_match"])
        self.assertEqual(audit["batch_norm_steps"], [0])
        self.assertEqual(audit["inferred_epochs_at_batch32"], [0.0])
        self.assertTrue(audit["head_last_bias_within_fresh_init_bound"])


class HeadControlCriteriaTest(unittest.TestCase):
    def test_trained_head_beats_random_head(self) -> None:
        result = evaluate_head_control(0.98, 0.50)
        self.assertTrue(result["passed"])

    def test_low_trained_auc_fails(self) -> None:
        result = evaluate_head_control(0.60, 0.50)
        self.assertFalse(result["passed"])

    def test_high_random_auc_fails(self) -> None:
        result = evaluate_head_control(0.98, 0.80)
        self.assertFalse(result["passed"])


if __name__ == "__main__":
    unittest.main()
