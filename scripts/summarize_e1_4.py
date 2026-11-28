#!/usr/bin/env python3
"""Aggregate the E1.4 shortcut probes into one report.

Reads the harness output written by ``scripts/evaluate_e1_4.py``::

    <runs-root>/<condition>/seed<seed>/<checkpoint-stem>/metrics.json   (image)
    <runs-root>/<probe>/seed<seed>/metrics.json                         (feature)

and writes::

    <output-root>/summary.csv                  one row per image condition/seed/split
    <output-root>/summary_by_condition.csv     image conditions, with delta vs `full`
    <output-root>/per_attack_eer.csv           image conditions per attack
    <output-root>/feature_probes.csv           one row per feature probe/seed/split
    <output-root>/RESULTS.md                   human-readable comparison

Every row keeps the G4 metric contract: split, attack subset, threshold, and
sample count.  EER/AUC are threshold-free; the fixed-0.5 columns are descriptive.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from spoof_detection.feature_probe import FEATURE_PROBE_ORDER  # noqa: E402
from spoof_detection.shortcuts import SHORTCUT_ORDER, get_shortcut_spec  # noqa: E402

DEFAULT_RUNS_ROOT = "reports/phase1/e1_4_shortcuts/runs"
DEFAULT_OUTPUT_ROOT = "reports/phase1/e1_4_shortcuts"
CONTROL_CONDITION = "full"

SUMMARY_COLUMNS = [
    "condition",
    "seed",
    "split",
    "attack_subset",
    "threshold_free_eer",
    "threshold_free_auc",
    "samples_total",
    "samples_bonafide",
    "samples_spoof",
    "dev_eer_threshold",
    "fpr_at_0.5",
    "spoof_recall_at_0.5",
    "checkpoint_sha256",
    "keep",
    "metrics_path",
]

FEATURE_COLUMNS = [
    "probe",
    "seed",
    "split",
    "attack_subset",
    "threshold_free_eer",
    "threshold_free_auc",
    "samples_total",
    "samples_bonafide",
    "samples_spoof",
    "fpr_at_0.5",
    "spoof_recall_at_0.5",
    "checkpoint_sha256",
    "metrics_path",
]

PER_ATTACK_COLUMNS = [
    "condition",
    "seed",
    "split",
    "attack_id",
    "n_spoof",
    "n_bonafide",
    "eer",
    "auc",
]


def checkpoint_stem(condition: str, seed: int, limit: int | None = None) -> str:
    name = f"resnet18_frozen_{condition}_seed{seed}"
    if limit is not None:
        name = f"{name}_partial{limit}"
    return name


def image_metrics_path(
    runs_root: Path,
    condition: str,
    seed: int,
    limit: int | None = None,
) -> Path:
    return (
        Path(runs_root)
        / condition
        / f"seed{seed}"
        / checkpoint_stem(condition, seed, limit)
        / "metrics.json"
    )


def feature_metrics_path(runs_root: Path, probe: str, seed: int) -> Path:
    return Path(runs_root) / probe / f"seed{seed}" / "metrics.json"


def _float(value: object) -> float | None:
    try:
        numeric = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _load_metrics(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _image_rows(
    condition: str,
    seed: int,
    metrics: dict[str, object],
    path: Path,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    thresholds = metrics.get("thresholds") or {}
    checkpoint = metrics.get("checkpoint") or {}
    keep = get_shortcut_spec(condition).keep
    for split, payload in (metrics.get("splits") or {}).items():
        summary = (payload or {}).get("metrics") or {}
        counts = summary.get("sample_counts") or {}
        fixed = (summary.get("threshold_metrics") or {}).get("fixed_0.5") or {}
        rows.append(
            {
                "condition": condition,
                "seed": seed,
                "split": split,
                "attack_subset": (payload or {}).get("attack_subset"),
                "threshold_free_eer": _float(summary.get("eer")),
                "threshold_free_auc": _float(summary.get("auc")),
                "samples_total": counts.get("total"),
                "samples_bonafide": counts.get("bonafide"),
                "samples_spoof": counts.get("spoof"),
                "dev_eer_threshold": _float(thresholds.get("dev_eer")),
                "fpr_at_0.5": _float(fixed.get("false_positive_rate")),
                "spoof_recall_at_0.5": _float(fixed.get("spoof_recall")),
                "checkpoint_sha256": checkpoint.get("sha256"),
                "keep": keep,
                "metrics_path": str(path),
            }
        )
    return rows


def _per_attack_rows(
    condition: str,
    seed: int,
    metrics: dict[str, object],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for split, payload in (metrics.get("splits") or {}).items():
        summary = (payload or {}).get("metrics") or {}
        for attack_id, entry in (summary.get("per_attack") or {}).items():
            entry = entry or {}
            rows.append(
                {
                    "condition": condition,
                    "seed": seed,
                    "split": split,
                    "attack_id": attack_id,
                    "n_spoof": entry.get("n_spoof"),
                    "n_bonafide": entry.get("n_bonafide"),
                    "eer": _float(entry.get("eer")),
                    "auc": _float(entry.get("auc")),
                }
            )
    return rows


def _feature_rows(
    probe: str,
    seed: int,
    metrics: dict[str, object],
    path: Path,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    checkpoint = metrics.get("checkpoint") or {}
    for split, payload in (metrics.get("splits") or {}).items():
        summary = (payload or {}).get("metrics") or {}
        counts = summary.get("sample_counts") or {}
        fixed = (summary.get("threshold_metrics") or {}).get("fixed_0.5") or {}
        rows.append(
            {
                "probe": probe,
                "seed": seed,
                "split": split,
                "attack_subset": (payload or {}).get("attack_subset"),
                "threshold_free_eer": _float(summary.get("eer")),
                "threshold_free_auc": _float(summary.get("auc")),
                "samples_total": counts.get("total"),
                "samples_bonafide": counts.get("bonafide"),
                "samples_spoof": counts.get("spoof"),
                "fpr_at_0.5": _float(fixed.get("false_positive_rate")),
                "spoof_recall_at_0.5": _float(fixed.get("spoof_recall")),
                "checkpoint_sha256": checkpoint.get("sha256"),
                "metrics_path": str(path),
            }
        )
    return rows


def _format(value: object, digits: int = 4) -> str:
    numeric = _float(value)
    return "n/a" if numeric is None else f"{numeric:.{digits}f}"


def _format_pts(value: object) -> str:
    numeric = _float(value)
    return "n/a" if numeric is None else f"{numeric:+.2f}"


def _image_table(summary: pd.DataFrame, split: str) -> list[str]:
    subset = summary[summary["split"] == split].sort_values("condition")
    if subset.empty:
        return ["_No runs found._", ""]
    lines = [
        "| Condition | Keep | n | EER | AUC | FPR@0.5 | Recall@0.5 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in subset.iterrows():
        lines.append(
            f"| `{row['condition']}` | {row['keep']} | {row['samples_total']} | "
            f"{_format(row['threshold_free_eer'])} | {_format(row['threshold_free_auc'])} | "
            f"{_format(row['fpr_at_0.5'])} | {_format(row['spoof_recall_at_0.5'])} |"
        )
    lines.append("")
    return lines


def _feature_table(features: pd.DataFrame, split: str) -> list[str]:
    subset = features[features["split"] == split].sort_values("probe")
    if subset.empty:
        return ["_No feature-probe rows found._", ""]
    lines = [
        "| Probe | n | EER | AUC | FPR@0.5 | Recall@0.5 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in subset.iterrows():
        lines.append(
            f"| `{row['probe']}` | {row['samples_total']} | "
            f"{_format(row['threshold_free_eer'])} | {_format(row['threshold_free_auc'])} | "
            f"{_format(row['fpr_at_0.5'])} | {_format(row['spoof_recall_at_0.5'])} |"
        )
    lines.append("")
    return lines


def _per_attack_table(per_attack: pd.DataFrame, split: str) -> list[str]:
    subset = per_attack[per_attack["split"] == split]
    if subset.empty:
        return ["_No per-attack rows found._", ""]
    attacks = sorted(subset["attack_id"].unique())
    conditions = sorted(subset["condition"].unique())
    lines = ["| Condition | " + " | ".join(attacks) + " |"]
    lines.append("|---|" + "---|" * len(attacks))
    for condition in conditions:
        values = []
        for attack in attacks:
            match = subset[
                (subset["condition"] == condition) & (subset["attack_id"] == attack)
            ]
            values.append(
                _format(match["eer"].iloc[0]) if not match.empty else "n/a"
            )
        lines.append(f"| `{condition}` | " + " | ".join(values) + " |")
    lines.append("")
    return lines


def _results_markdown(
    summary: pd.DataFrame,
    features: pd.DataFrame,
    per_attack: pd.DataFrame,
) -> str:
    lines = ["# E1.4 shortcut ablation — results\n"]
    lines.append(
        "Generated by `scripts/summarize_e1_4.py` from the frozen E0.3 harness.\n"
    )
    lines.append(
        "**Metric contract.** EER/AUC are *threshold-free*. The `FPR@0.5` / "
        "`Recall@0.5` columns are a descriptive fixed operating point, not a "
        "tuned threshold.\n"
    )
    lines.append("## Eval image conditions (clean estimate, A07–A19)\n")
    lines.extend(_image_table(summary, "eval"))
    lines.append("## Dev image conditions (model-selection split, A01–A06)\n")
    lines.extend(_image_table(summary, "dev"))
    lines.append("## Eval per-attack EER (image conditions)\n")
    lines.extend(_per_attack_table(per_attack, "eval"))
    lines.append("## Feature probes — eval (clean estimate)\n")
    lines.extend(_feature_table(features, "eval"))
    lines.append("## Feature probes — dev (model-selection split)\n")
    lines.extend(_feature_table(features, "dev"))
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-root",
        default=DEFAULT_RUNS_ROOT,
        help=f"evaluation output root (default: {DEFAULT_RUNS_ROOT})",
    )
    parser.add_argument(
        "--output-root",
        default=DEFAULT_OUTPUT_ROOT,
        help=f"report output root (default: {DEFAULT_OUTPUT_ROOT})",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=SHORTCUT_ORDER,
        default=list(SHORTCUT_ORDER),
        help="image conditions to summarize (default: all three)",
    )
    parser.add_argument(
        "--feature-probes",
        nargs="+",
        choices=FEATURE_PROBE_ORDER,
        default=list(FEATURE_PROBE_ORDER),
        help="feature probes to summarize (default: all three)",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42],
        help="seeds to summarize (default: 42)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="smoke only: match the harness *_partialN output directory",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = REPO_ROOT
    runs_root = Path(args.runs_root)
    if not runs_root.is_absolute():
        runs_root = (repo_root / runs_root).resolve()
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = (repo_root / output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    image_rows: list[dict[str, object]] = []
    attack_rows: list[dict[str, object]] = []
    for condition in args.conditions:
        for seed in args.seeds:
            path = image_metrics_path(runs_root, condition, seed, args.limit)
            if not path.is_file():
                print(f"warning: missing {path}", file=sys.stderr)
                continue
            metrics = _load_metrics(path)
            image_rows.extend(_image_rows(condition, seed, metrics, path))
            attack_rows.extend(_per_attack_rows(condition, seed, metrics))

    feature_rows: list[dict[str, object]] = []
    for probe in args.feature_probes:
        for seed in args.seeds:
            path = feature_metrics_path(runs_root, probe, seed)
            if not path.is_file():
                print(f"warning: missing {path}", file=sys.stderr)
                continue
            metrics = _load_metrics(path)
            feature_rows.extend(_feature_rows(probe, seed, metrics, path))

    summary = pd.DataFrame(image_rows, columns=SUMMARY_COLUMNS)
    per_attack = pd.DataFrame(attack_rows, columns=PER_ATTACK_COLUMNS)
    features = pd.DataFrame(feature_rows, columns=FEATURE_COLUMNS)

    summary_path = output_root / "summary.csv"
    per_attack_path = output_root / "per_attack_eer.csv"
    features_path = output_root / "feature_probes.csv"
    results_path = output_root / "RESULTS.md"

    summary.to_csv(summary_path, index=False)
    per_attack.to_csv(per_attack_path, index=False)
    features.to_csv(features_path, index=False)
    results_path.write_text(
        _results_markdown(summary, features, per_attack),
        encoding="utf-8",
    )

    print(f"wrote {summary_path}")
    print(f"wrote {per_attack_path}")
    print(f"wrote {features_path}")
    print(f"wrote {results_path}")
    if not summary.empty:
        print("\n" + summary.to_string(index=False))
    if not features.empty:
        print("\n" + features.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
