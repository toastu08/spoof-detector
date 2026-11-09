"""Evaluation metrics for the frozen ResNet18 baseline.

The primary metric is the equal error rate (EER) computed with the ASVspoof
2019 DET-curve convention: the positive class is spoof, and a higher score
means "more spoof".  AUC is the rank-based probability that a spoof clip scores
above a bonafide clip.

All functions operate on plain NumPy arrays or a dataframe with the canonical
columns ``label``, ``score``, ``system_id`` and ``speaker_id``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Collection, Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EERResult:
    """Equal error rate and the descriptive threshold where it occurs."""

    eer: float
    threshold: float


def _as_float_array(
    values: Collection[float] | np.ndarray,
    name: str,
    *,
    allow_empty: bool = False,
) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size == 0 and not allow_empty:
        raise ValueError(f"{name} must not be empty")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def compute_det_curve(
    target_scores: Collection[float] | np.ndarray,
    nontarget_scores: Collection[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(frr, far, thresholds)`` using the ASVspoof DET convention.

    Higher scores are treated as target (spoof) scores.  The threshold array is
    ascending and is included for descriptive reporting only.
    """
    target = _as_float_array(target_scores, "target_scores")
    nontarget = _as_float_array(nontarget_scores, "nontarget_scores")

    n_scores = target.size + nontarget.size
    all_scores = np.concatenate((target, nontarget))
    labels = np.concatenate(
        (np.ones(target.size, dtype=float), np.zeros(nontarget.size, dtype=float))
    )

    indices = np.argsort(all_scores, kind="mergesort")
    labels = labels[indices]

    target_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = nontarget.size - (
        np.arange(1, n_scores + 1) - target_trial_sums
    )

    frr = np.concatenate((np.atleast_1d(0.0), target_trial_sums / target.size))
    far = np.concatenate(
        (np.atleast_1d(1.0), nontarget_trial_sums / nontarget.size)
    )
    thresholds = np.concatenate(
        (
            np.atleast_1d(all_scores[indices[0]] - 0.001),
            all_scores[indices],
        )
    )
    return frr, far, thresholds


def compute_eer(
    spoof_scores: Collection[float] | np.ndarray,
    bonafide_scores: Collection[float] | np.ndarray,
) -> EERResult:
    """Compute the equal error rate and its descriptive threshold."""
    spoof = _as_float_array(spoof_scores, "spoof_scores")
    bonafide = _as_float_array(bonafide_scores, "bonafide_scores")

    all_scores = np.concatenate((spoof, bonafide))
    if np.unique(all_scores).size == 1:
        # The official DET curve is undefined for a completely tied score set;
        # a detector with no ranking information has EER 0.5.
        return EERResult(eer=0.5, threshold=float(all_scores[0]))

    frr, far, thresholds = compute_det_curve(spoof, bonafide)
    min_index = int(np.argmin(np.abs(frr - far)))
    eer = float(np.mean((frr[min_index], far[min_index])))
    return EERResult(eer=eer, threshold=float(thresholds[min_index]))


def compute_auc(
    spoof_scores: Collection[float] | np.ndarray,
    bonafide_scores: Collection[float] | np.ndarray,
) -> float:
    """Return the rank-based AUC with spoof as the positive class."""
    spoof = _as_float_array(spoof_scores, "spoof_scores")
    bonafide = _as_float_array(bonafide_scores, "bonafide_scores")

    all_scores = np.concatenate((spoof, bonafide))
    order = np.argsort(all_scores, kind="mergesort")
    sorted_scores = all_scores[order]

    ranks = np.empty(all_scores.size, dtype=float)
    start = 0
    while start < sorted_scores.size:
        end = start
        while (
            end + 1 < sorted_scores.size
            and sorted_scores[end + 1] == sorted_scores[start]
        ):
            end += 1
        # 1-based average rank for the tied block.
        average_rank = (start + end + 2) / 2.0
        ranks[order[start : end + 1]] = average_rank
        start = end + 1

    rank_sum = float(ranks[: spoof.size].sum())
    auc = (
        rank_sum - spoof.size * (spoof.size + 1) / 2.0
    ) / (spoof.size * bonafide.size)
    return float(auc)


def compute_threshold_metrics(
    labels: Collection[int] | np.ndarray,
    scores: Collection[float] | np.ndarray,
    threshold: float,
) -> dict[str, object]:
    """Return accuracy, recalls, rates and confusion counts at ``threshold``."""
    y = np.asarray(labels, dtype=int).reshape(-1)
    s = _as_float_array(scores, "scores")
    if y.size != s.size:
        raise ValueError("labels and scores must have the same length")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("labels must be 0 (bonafide) or 1 (spoof)")

    predicted = (s >= threshold).astype(int)
    true_positive = int(np.sum((predicted == 1) & (y == 1)))
    true_negative = int(np.sum((predicted == 0) & (y == 0)))
    false_positive = int(np.sum((predicted == 1) & (y == 0)))
    false_negative = int(np.sum((predicted == 0) & (y == 1)))

    n_total = int(y.size)
    n_positive = true_positive + false_negative
    n_negative = true_negative + false_positive

    bonafide_recall = (
        true_negative / n_negative if n_negative else None
    )
    spoof_recall = true_positive / n_positive if n_positive else None
    balanced_accuracy = (
        (bonafide_recall + spoof_recall) / 2.0
        if bonafide_recall is not None and spoof_recall is not None
        else None
    )

    return {
        "threshold": float(threshold),
        "accuracy": (true_positive + true_negative) / n_total,
        "balanced_accuracy": balanced_accuracy,
        "bonafide_recall": bonafide_recall,
        "spoof_recall": spoof_recall,
        "false_positive_rate": (
            false_positive / n_negative if n_negative else None
        ),
        "false_negative_rate": (
            false_negative / n_positive if n_positive else None
        ),
        "precision": (
            true_positive / (true_positive + false_positive)
            if (true_positive + false_positive)
            else None
        ),
        "confusion": {
            "tp": true_positive,
            "tn": true_negative,
            "fp": false_positive,
            "fn": false_negative,
        },
        "n": n_total,
    }


def compute_calibration(
    labels: Collection[int] | np.ndarray,
    scores: Collection[float] | np.ndarray,
    n_bins: int = 10,
) -> dict[str, object]:
    """Return Brier score and equal-width expected calibration error."""
    y = np.asarray(labels, dtype=float).reshape(-1)
    s = _as_float_array(scores, "scores")
    if y.size != s.size:
        raise ValueError("labels and scores must have the same length")
    if n_bins < 1:
        raise ValueError("n_bins must be positive")
    if np.any((s < 0.0) | (s > 1.0)):
        raise ValueError("scores must lie in [0, 1] for calibration")

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(s, bin_edges[1:-1], right=False)
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    ece = 0.0
    bins: list[dict[str, object]] = []
    for bin_index in range(n_bins):
        mask = bin_indices == bin_index
        count = int(mask.sum())
        if count == 0:
            bins.append(
                {
                    "bin": bin_index,
                    "lower": float(bin_edges[bin_index]),
                    "upper": float(bin_edges[bin_index + 1]),
                    "count": 0,
                    "mean_score": None,
                    "spoof_rate": None,
                }
            )
            continue
        mean_score = float(s[mask].mean())
        spoof_rate = float(y[mask].mean())
        ece += (count / y.size) * abs(spoof_rate - mean_score)
        bins.append(
            {
                "bin": bin_index,
                "lower": float(bin_edges[bin_index]),
                "upper": float(bin_edges[bin_index + 1]),
                "count": count,
                "mean_score": mean_score,
                "spoof_rate": spoof_rate,
            }
        )

    return {
        "brier": float(np.mean((s - y) ** 2)),
        "ece": float(ece),
        "n_bins": n_bins,
        "bins": bins,
    }


def _eer_auc_summary(
    positive_scores: Collection[float] | np.ndarray,
    negative_scores: Collection[float] | np.ndarray,
) -> dict[str, object]:
    positive = _as_float_array(
        positive_scores, "positive_scores", allow_empty=True
    )
    negative = _as_float_array(
        negative_scores, "negative_scores", allow_empty=True
    )
    result: dict[str, object] = {
        "n_spoof": int(positive.size),
        "n_bonafide": int(negative.size),
        "eer": None,
        "auc": None,
        "eer_threshold_descriptive": None,
    }
    if positive.size == 0 or negative.size == 0:
        result["note"] = "EER/AUC require both spoof and bonafide samples"
        return result

    eer = compute_eer(positive, negative)
    result["eer"] = eer.eer
    result["auc"] = compute_auc(positive, negative)
    result["eer_threshold_descriptive"] = eer.threshold
    return result


def _clean_identifier(series: pd.Series, fallback: str) -> pd.Series:
    cleaned = series.astype("object").where(series.notna(), fallback)
    return cleaned.astype(str).replace({"": fallback, "nan": fallback})


def summarize_split(
    records: pd.DataFrame,
    extra_thresholds: Mapping[str, float] | None = None,
) -> dict[str, object]:
    """Compute aggregate, per-attack and per-speaker metrics for one split.

    ``records`` must contain ``label``, ``score``, ``system_id`` and
    ``speaker_id``.  Per-attack EER uses that attack's spoof clips against all
    bonafide clips in the split.  Per-speaker EER uses same-speaker bonafide
    clips as the negative class.
    """
    required = {"label", "score", "system_id", "speaker_id"}
    missing = required - set(records.columns)
    if missing:
        raise ValueError(f"records is missing columns: {sorted(missing)}")
    if records.empty:
        raise ValueError("records must not be empty")

    records = records.copy()
    records["system_id"] = _clean_identifier(records["system_id"], "-")
    records["speaker_id"] = _clean_identifier(records["speaker_id"], "unknown")

    y = records["label"].to_numpy(dtype=int)
    scores = _as_float_array(records["score"], "score")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("labels must be 0 (bonafide) or 1 (spoof)")
    if np.any((scores < 0.0) | (scores > 1.0)):
        raise ValueError("scores must lie in [0, 1]")

    bonafide_scores = scores[y == 0]
    spoof_scores = scores[y == 1]

    # EER and AUC require both classes.  A single-class external corpus (for
    # example an all-spoof smoke set) still gets threshold metrics and score
    # summaries, with the ranking metrics explicitly reported as None.
    eer = (
        compute_eer(spoof_scores, bonafide_scores)
        if bonafide_scores.size and spoof_scores.size
        else None
    )
    threshold_metrics: dict[str, object] = {
        "fixed_0.5": compute_threshold_metrics(y, scores, 0.5)
    }
    for name, threshold in (extra_thresholds or {}).items():
        threshold_metrics[name] = compute_threshold_metrics(
            y, scores, float(threshold)
        )

    attack_ids = sorted(
        {
            system_id
            for system_id in records["system_id"].tolist()
            if system_id != "-"
        }
    )
    per_attack: dict[str, object] = {}
    bonafide_frame = records[records["label"] == 0]
    for attack_id in attack_ids:
        positive = records[
            (records["label"] == 1) & (records["system_id"] == attack_id)
        ]
        per_attack[attack_id] = _eer_auc_summary(
            positive["score"], bonafide_frame["score"]
        )

    per_speaker: dict[str, object] = {}
    speaker_ids = sorted(records["speaker_id"].unique().tolist())
    for speaker_id in speaker_ids:
        positive = records[
            (records["speaker_id"] == speaker_id) & (records["label"] == 1)
        ]
        negative = records[
            (records["speaker_id"] == speaker_id) & (records["label"] == 0)
        ]
        entry = _eer_auc_summary(positive["score"], negative["score"])
        entry["spoof_recall_at_0.5"] = (
            float((positive["score"].to_numpy(dtype=float) >= 0.5).mean())
            if len(positive)
            else None
        )
        entry["bonafide_recall_at_0.5"] = (
            float((negative["score"].to_numpy(dtype=float) < 0.5).mean())
            if len(negative)
            else None
        )
        per_speaker[speaker_id] = entry

    return {
        "sample_counts": {
            "total": int(y.size),
            "bonafide": int(bonafide_scores.size),
            "spoof": int(spoof_scores.size),
        },
        "attack_ids": attack_ids,
        "speaker_count": int(len(speaker_ids)),
        "score_summary": {
            "min": float(scores.min()),
            "max": float(scores.max()),
            "mean": float(scores.mean()),
        },
        "eer": eer.eer if eer is not None else None,
        "eer_threshold_descriptive": eer.threshold if eer is not None else None,
        "auc": (
            compute_auc(spoof_scores, bonafide_scores)
            if eer is not None
            else None
        ),
        "calibration": compute_calibration(y, scores),
        "threshold_metrics": threshold_metrics,
        "per_attack": per_attack,
        "per_speaker": per_speaker,
    }
