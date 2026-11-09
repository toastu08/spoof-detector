"""Unit tests for the evaluation metrics."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from spoof_detection.metrics import (  # noqa: E402
    compute_auc,
    compute_calibration,
    compute_det_curve,
    compute_eer,
    compute_threshold_metrics,
    summarize_split,
)


class EERTest(unittest.TestCase):
    def test_perfect_separation_has_zero_eer(self) -> None:
        result = compute_eer([0.9, 0.8], [0.1, 0.2])
        self.assertAlmostEqual(result.eer, 0.0)
        # The DET minimum lands on the largest bonafide score, 0.2.
        self.assertAlmostEqual(result.threshold, 0.2)
        self.assertLess(result.threshold, 0.8)

    def test_inverted_scores_have_eer_one(self) -> None:
        result = compute_eer([0.1, 0.2], [0.9, 0.8])
        self.assertAlmostEqual(result.eer, 1.0)

    def test_overlapping_scores_have_eer_half(self) -> None:
        result = compute_eer([0.6, 0.4], [0.3, 0.7])
        self.assertAlmostEqual(result.eer, 0.5)
        # The DET minimum lands on the score 0.4, where FRR == FAR == 0.5.
        self.assertAlmostEqual(result.threshold, 0.4)

    def test_completely_tied_scores_fall_back_to_chance(self) -> None:
        result = compute_eer([0.5, 0.5], [0.5, 0.5])
        self.assertAlmostEqual(result.eer, 0.5)
        self.assertAlmostEqual(result.threshold, 0.5)

    def test_det_curve_endpoints(self) -> None:
        frr, far, thresholds = compute_det_curve([0.9, 0.8], [0.1, 0.2])
        self.assertEqual(frr[0], 0.0)
        self.assertEqual(far[0], 1.0)
        self.assertEqual(frr[-1], 1.0)
        self.assertEqual(far[-1], 0.0)
        self.assertEqual(len(frr), len(far))
        self.assertEqual(len(frr), len(thresholds))

    def test_rejects_empty_input(self) -> None:
        with self.assertRaises(ValueError):
            compute_eer([], [0.5])

    def test_rejects_non_finite_input(self) -> None:
        with self.assertRaises(ValueError):
            compute_eer([0.5, float("nan")], [0.5])


class AUCTest(unittest.TestCase):
    def test_perfect_ranking(self) -> None:
        self.assertAlmostEqual(compute_auc([0.9, 0.8], [0.1, 0.2]), 1.0)

    def test_inverted_ranking(self) -> None:
        self.assertAlmostEqual(compute_auc([0.1, 0.2], [0.9, 0.8]), 0.0)

    def test_chance_ranking(self) -> None:
        self.assertAlmostEqual(compute_auc([0.6, 0.4], [0.3, 0.7]), 0.5)

    def test_all_ties_is_chance(self) -> None:
        self.assertAlmostEqual(compute_auc([0.5, 0.5], [0.5, 0.5]), 0.5)

    def test_ties_receive_average_rank(self) -> None:
        # Spoof 0.5 ties with one bonafide 0.5 and beats the other bonafide.
        # Pairwise AUC = (1 + 0.5) / 2 = 0.75.
        self.assertAlmostEqual(compute_auc([0.5], [0.5, 0.1]), 0.75)


class ThresholdMetricsTest(unittest.TestCase):
    def test_confusion_and_rates(self) -> None:
        metrics = compute_threshold_metrics(
            labels=[0, 0, 1, 1],
            scores=[0.2, 0.8, 0.4, 0.6],
            threshold=0.5,
        )
        self.assertAlmostEqual(metrics["accuracy"], 0.5)
        self.assertAlmostEqual(metrics["balanced_accuracy"], 0.5)
        self.assertAlmostEqual(metrics["bonafide_recall"], 0.5)
        self.assertAlmostEqual(metrics["spoof_recall"], 0.5)
        self.assertAlmostEqual(metrics["false_positive_rate"], 0.5)
        self.assertAlmostEqual(metrics["false_negative_rate"], 0.5)
        self.assertAlmostEqual(metrics["precision"], 0.5)
        self.assertEqual(
            metrics["confusion"],
            {"tp": 1, "tn": 1, "fp": 1, "fn": 1},
        )

    def test_threshold_boundary_is_inclusive(self) -> None:
        metrics = compute_threshold_metrics([0, 1], [0.5, 0.5], 0.5)
        self.assertEqual(metrics["confusion"], {"tp": 1, "tn": 0, "fp": 1, "fn": 0})

    def test_rejects_bad_labels(self) -> None:
        with self.assertRaises(ValueError):
            compute_threshold_metrics([0, 2], [0.1, 0.9], 0.5)


class CalibrationTest(unittest.TestCase):
    def test_perfect_calibration(self) -> None:
        metrics = compute_calibration([0, 1], [0.0, 1.0])
        self.assertAlmostEqual(metrics["brier"], 0.0)
        self.assertAlmostEqual(metrics["ece"], 0.0)

    def test_constant_half_scores(self) -> None:
        metrics = compute_calibration([0, 0, 1, 1], [0.5, 0.5, 0.5, 0.5])
        self.assertAlmostEqual(metrics["brier"], 0.25)
        self.assertAlmostEqual(metrics["ece"], 0.0)

    def test_rejects_out_of_range_scores(self) -> None:
        with self.assertRaises(ValueError):
            compute_calibration([0, 1], [-0.1, 1.1])


class SummarizeSplitTest(unittest.TestCase):
    def setUp(self) -> None:
        self.records = pd.DataFrame(
            {
                "label": [0, 0, 1, 1, 1],
                "score": [0.1, 0.2, 0.8, 0.9, 0.7],
                "system_id": ["-", "-", "A01", "A01", "A02"],
                "speaker_id": ["s1", "s1", "s1", "s1", "s2"],
            }
        )

    def test_counts_attacks_and_speakers(self) -> None:
        summary = summarize_split(
            self.records,
            extra_thresholds={"dev_eer": 0.5},
        )
        self.assertEqual(
            summary["sample_counts"],
            {"total": 5, "bonafide": 2, "spoof": 3},
        )
        self.assertEqual(summary["attack_ids"], ["A01", "A02"])
        self.assertEqual(summary["speaker_count"], 2)
        self.assertAlmostEqual(summary["eer"], 0.0)
        self.assertAlmostEqual(summary["auc"], 1.0)
        self.assertEqual(
            set(summary["threshold_metrics"]),
            {"fixed_0.5", "dev_eer"},
        )
        self.assertEqual(summary["per_attack"]["A01"]["n_spoof"], 2)
        self.assertEqual(summary["per_attack"]["A01"]["n_bonafide"], 2)
        self.assertAlmostEqual(summary["per_attack"]["A01"]["eer"], 0.0)
        self.assertEqual(summary["per_speaker"]["s2"]["n_spoof"], 1)
        self.assertEqual(summary["per_speaker"]["s2"]["n_bonafide"], 0)
        self.assertIsNone(summary["per_speaker"]["s2"]["eer"])

    def test_single_class_split_reports_unknown_ranking_metrics(self) -> None:
        only_bonafide = self.records[self.records["label"] == 0]
        summary = summarize_split(only_bonafide)
        self.assertIsNone(summary["eer"])
        self.assertIsNone(summary["auc"])
        self.assertIsNone(summary["eer_threshold_descriptive"])
        self.assertEqual(summary["sample_counts"]["spoof"], 0)
        self.assertIsNone(
            summary["threshold_metrics"]["fixed_0.5"]["spoof_recall"]
        )
        self.assertAlmostEqual(
            summary["threshold_metrics"]["fixed_0.5"]["bonafide_recall"],
            1.0,
        )

    def test_rejects_missing_columns(self) -> None:
        with self.assertRaises(ValueError):
            summarize_split(self.records.drop(columns=["system_id"]))

    def test_rejects_out_of_range_scores(self) -> None:
        bad = self.records.copy()
        bad.loc[0, "score"] = 1.5
        with self.assertRaises(ValueError):
            summarize_split(bad)

    def test_numpy_arrays_are_accepted(self) -> None:
        result = compute_eer(
            np.array([0.9, 0.8]),
            np.array([0.1, 0.2]),
        )
        self.assertAlmostEqual(result.eer, 0.0)


if __name__ == "__main__":
    unittest.main()
