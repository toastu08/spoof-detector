#!/usr/bin/env python3
"""Aggregate the E1.2 rendering-condition probes into one report.

Reads the harness output written by ``scripts/evaluate_e1_2.py``::

    <runs-root>/<condition>/seed<seed>/<checkpoint-stem>/metrics.json

and writes::

    <output-root>/summary.csv                  one row per condition/seed/split
    <output-root>/summary_by_condition.csv     mean/std/range across seeds
    <output-root>/per_attack_eer.csv           one row per condition/seed/split/attack
    <output-root>/per_attack_by_condition.csv  per-attack aggregation
    <output-root>/colormap_statistics.csv      RGB-locus spread per colormap
    <output-root>/RESULTS.md                   human-readable comparison

Every row keeps the G4 metric contract: split, attack subset, threshold, and
sample count.  EER/AUC are threshold-free; the fixed-0.5 columns are a
descriptive operating point, not a tuned threshold.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from spoof_detection.rendering import (  # noqa: E402
    RENDER_CONDITION_ORDER,
    RENDER_CONDITIONS,
    get_rendering_spec,
)

DEFAULT_RUNS_ROOT = "reports/phase1/e1_2_rendering/runs"
DEFAULT_OUTPUT_ROOT = "reports/phase1/e1_2_rendering"
CONTROL_CONDITION = "magma_flip"

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
    "metrics_path",
    "colormap",
    "orientation",
    "flip",
    "encoding",
    "normalization",
    "db_range",
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
    "eer_threshold_descriptive",
]


def checkpoint_stem(condition: str, seed: int, limit: int | None = None) -> str:
    """Return the harness output directory name for one condition and seed."""
    name = f"resnet18_frozen_{condition}_seed{seed}"
    if limit is not None:
        name = f"{name}_partial{limit}"
    return name


def metrics_path(
    runs_root: Path,
    condition: str,
    seed: int,
    limit: int | None = None,
) -> Path:
    """Return the metrics.json path for one condition/seed run."""
    return (
        Path(runs_root)
        / condition
        / f"seed{seed}"
        / checkpoint_stem(condition, seed, limit)
        / "metrics.json"
    )


def _float(value: object) -> float | None:
    try:
        numeric = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _load_metrics(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _spec_fields(condition: str) -> dict[str, object]:
    try:
        spec = get_rendering_spec(condition)
    except ValueError:
        return {
            "colormap": None,
            "orientation": None,
            "flip": None,
            "encoding": None,
            "normalization": None,
            "db_range": None,
        }
    return {
        "colormap": spec.colormap,
        "orientation": "lower" if spec.flip else "upper",
        "flip": spec.flip,
        "encoding": spec.encoding.upper(),
        "normalization": spec.normalization,
        "db_range": (
            list(spec.db_range) if spec.db_range is not None else None
        ),
    }


def _summary_rows(
    condition: str,
    seed: int,
    metrics: dict[str, object],
    run_metrics_path: Path,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    thresholds = metrics.get("thresholds") or {}
    dev_threshold = thresholds.get("dev_eer")
    checkpoint = metrics.get("checkpoint") or {}
    spec = _spec_fields(condition)
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
                "dev_eer_threshold": _float(dev_threshold),
                "fpr_at_0.5": _float(fixed.get("false_positive_rate")),
                "spoof_recall_at_0.5": _float(fixed.get("spoof_recall")),
                "checkpoint_sha256": checkpoint.get("sha256"),
                "metrics_path": str(run_metrics_path),
                **spec,
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
                    "eer_threshold_descriptive": _float(
                        entry.get("eer_threshold_descriptive")
                    ),
                }
            )
    return rows


def _aggregate_by_condition(summary: pd.DataFrame) -> pd.DataFrame:
    """Return per-condition mean/std/range across seeds, plus deltas vs control."""
    if summary.empty:
        return pd.DataFrame(
            columns=[
                "condition",
                "split",
                "n_seeds",
                "threshold_free_eer_mean",
                "threshold_free_eer_std",
                "threshold_free_eer_min",
                "threshold_free_eer_max",
                "threshold_free_auc_mean",
                "threshold_free_auc_std",
                "threshold_free_auc_min",
                "threshold_free_auc_max",
                "dev_eer_threshold_mean",
                "fpr_at_0.5_mean",
                "spoof_recall_at_0.5_mean",
                "delta_eer_vs_control_pts",
                "delta_auc_vs_control",
                "colormap",
                "orientation",
                "flip",
                "encoding",
                "normalization",
                "db_range",
            ]
        )

    rows: list[dict[str, object]] = []
    summary = summary.copy()
    for column in (
        "threshold_free_eer",
        "threshold_free_auc",
        "dev_eer_threshold",
        "fpr_at_0.5",
        "spoof_recall_at_0.5",
    ):
        summary[column] = pd.to_numeric(summary[column], errors="coerce")
    for (condition, split), frame in summary.groupby(
        ["condition", "split"], sort=True
    ):
        spec = _spec_fields(str(condition))
        rows.append(
            {
                "condition": condition,
                "split": split,
                "n_seeds": int(frame["seed"].nunique()),
                "threshold_free_eer_mean": frame["threshold_free_eer"].mean(),
                "threshold_free_eer_std": frame["threshold_free_eer"].std(),
                "threshold_free_eer_min": frame["threshold_free_eer"].min(),
                "threshold_free_eer_max": frame["threshold_free_eer"].max(),
                "threshold_free_auc_mean": frame["threshold_free_auc"].mean(),
                "threshold_free_auc_std": frame["threshold_free_auc"].std(),
                "threshold_free_auc_min": frame["threshold_free_auc"].min(),
                "threshold_free_auc_max": frame["threshold_free_auc"].max(),
                "dev_eer_threshold_mean": frame["dev_eer_threshold"].mean(),
                "fpr_at_0.5_mean": frame["fpr_at_0.5"].mean(),
                "spoof_recall_at_0.5_mean": frame[
                    "spoof_recall_at_0.5"
                ].mean(),
                **spec,
            }
        )
    result = pd.DataFrame(rows)

    control = result[result["condition"] == CONTROL_CONDITION].set_index(
        "split"
    )
    deltas: list[float | None] = []
    auc_deltas: list[float | None] = []
    for row in result.itertuples(index=False):
        if row.condition == CONTROL_CONDITION or row.split not in control.index:
            deltas.append(None)
            auc_deltas.append(None)
            continue
        base = control.loc[row.split]
        deltas.append(
            None
            if pd.isna(row.threshold_free_eer_mean)
            or pd.isna(base["threshold_free_eer_mean"])
            else float(
                (row.threshold_free_eer_mean - base["threshold_free_eer_mean"])
                * 100.0
            )
        )
        auc_deltas.append(
            None
            if pd.isna(row.threshold_free_auc_mean)
            or pd.isna(base["threshold_free_auc_mean"])
            else float(
                row.threshold_free_auc_mean - base["threshold_free_auc_mean"]
            )
        )
    result["delta_eer_vs_control_pts"] = deltas
    result["delta_auc_vs_control"] = auc_deltas
    return result


def _aggregate_per_attack(per_attack: pd.DataFrame) -> pd.DataFrame:
    if per_attack.empty:
        return pd.DataFrame(
            columns=[
                "condition",
                "split",
                "attack_id",
                "n_seeds",
                "eer_mean",
                "eer_std",
                "eer_min",
                "eer_max",
                "auc_mean",
            ]
        )

    rows: list[dict[str, object]] = []
    per_attack = per_attack.copy()
    for column in ("eer", "auc"):
        per_attack[column] = pd.to_numeric(
            per_attack[column], errors="coerce"
        )
    for (condition, split, attack_id), frame in per_attack.groupby(
        ["condition", "split", "attack_id"], sort=True
    ):
        rows.append(
            {
                "condition": condition,
                "split": split,
                "attack_id": attack_id,
                "n_seeds": int(frame["seed"].nunique()),
                "eer_mean": frame["eer"].mean(),
                "eer_std": frame["eer"].std(),
                "eer_min": frame["eer"].min(),
                "eer_max": frame["eer"].max(),
                "auc_mean": frame["auc"].mean(),
            }
        )
    return pd.DataFrame(rows)


def _colormap_statistics(
    colormaps: tuple[str, ...] | list[str] = ("magma", "viridis", "gray"),
) -> pd.DataFrame:
    """Describe the RGB locus of each colormap, to ground the RGB-richness claim."""
    import matplotlib.pyplot as plt

    values = np.linspace(0.0, 1.0, 256)
    rows: list[dict[str, object]] = []
    for name in sorted(set(colormaps)):
        rgb = np.asarray(plt.get_cmap(name)(values))[:, :3]
        luminance = rgb @ np.array([0.2126, 0.7152, 0.0722])
        rows.append(
            {
                "colormap": name,
                "r_range": float(rgb[:, 0].max() - rgb[:, 0].min()),
                "g_range": float(rgb[:, 1].max() - rgb[:, 1].min()),
                "b_range": float(rgb[:, 2].max() - rgb[:, 2].min()),
                "mean_channel_std": float(rgb.std(axis=0).mean()),
                "mean_saturation": float(
                    (rgb.max(axis=1) - rgb.min(axis=1)).mean()
                ),
                "luminance_min": float(luminance.min()),
                "luminance_max": float(luminance.max()),
                "luminance_range": float(
                    luminance.max() - luminance.min()
                ),
            }
        )
    return pd.DataFrame(rows)


def _format_number(value: object, digits: int = 4) -> str:
    numeric = _float(value)
    return "n/a" if numeric is None else f"{numeric:.{digits}f}"


def _format_points(value: object) -> str:
    numeric = _float(value)
    return "n/a" if numeric is None else f"{numeric:+.2f}"


def _summary_table(
    frame: pd.DataFrame,
    split: str,
    digits: int = 4,
) -> list[str]:
    subset = frame[frame["split"] == split].sort_values("condition")
    if subset.empty:
        return ["_No runs found._", ""]

    lines = [
        "| Condition | Colormap | Orientation | dB scaling | Encoding | n | "
        "EER | EER sd | AUC | ΔEER (pts) | FPR@0.5 | Recall@0.5 |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in subset.iterrows():
        lines.append(
            f"| `{row['condition']}` | {row['colormap']} | "
            f"{row['orientation']} | {row['normalization']} | "
            f"{row['encoding']} | {row['n_seeds']} | "
            f"{_format_number(row['threshold_free_eer_mean'], digits)} | "
            f"{_format_number(row['threshold_free_eer_std'], digits)} | "
            f"{_format_number(row['threshold_free_auc_mean'], 4)} | "
            f"{_format_points(row['delta_eer_vs_control_pts'])} | "
            f"{_format_number(row['fpr_at_0.5_mean'], digits)} | "
            f"{_format_number(row['spoof_recall_at_0.5_mean'], digits)} |"
        )
    lines.append("")
    return lines


def _per_attack_table(
    per_attack_by_condition: pd.DataFrame,
    split: str,
) -> list[str]:
    subset = per_attack_by_condition[
        per_attack_by_condition["split"] == split
    ]
    if subset.empty:
        return ["_No per-attack rows found._", ""]

    attacks = sorted(subset["attack_id"].unique())
    conditions = sorted(subset["condition"].unique())
    lines = ["| Condition | " + " | ".join(attacks) + " |"]
    lines.append("|---|" + "---|" * len(attacks))
    for condition in conditions:
        values: list[str] = []
        for attack in attacks:
            match = subset[
                (subset["condition"] == condition)
                & (subset["attack_id"] == attack)
            ]
            values.append(
                _format_number(match["eer_mean"].iloc[0])
                if not match.empty
                else "n/a"
            )
        lines.append(f"| `{condition}` | " + " | ".join(values) + " |")
    lines.append("")
    return lines


def _colormap_table(statistics: pd.DataFrame) -> list[str]:
    if statistics.empty:
        return ["_No colormap statistics._", ""]
    lines = [
        "| Colormap | R range | G range | B range | Mean channel sd | "
        "Mean saturation | Luminance range |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in statistics.itertuples(index=False):
        lines.append(
            f"| {row.colormap} | {row.r_range:.3f} | {row.g_range:.3f} | "
            f"{row.b_range:.3f} | {row.mean_channel_std:.3f} | "
            f"{row.mean_saturation:.3f} | {row.luminance_range:.3f} |"
        )
    lines.append("")
    return lines


def _results_markdown(
    summary: pd.DataFrame,
    by_condition: pd.DataFrame,
    per_attack_by_condition: pd.DataFrame,
    colormap_statistics: pd.DataFrame,
    run_rows: list[dict[str, object]],
    conditions: list[str],
    seeds: list[int],
) -> str:
    lines: list[str] = []
    lines.append("# E1.2 rendering ablation — results\n")
    lines.append(
        "Generated by `scripts/summarize_e1_2.py` from the frozen E0.3 harness.\n"
    )
    lines.append(
        "**Metric contract.** EER/AUC are *threshold-free*. The "
        "`FPR@0.5` / `Recall@0.5` columns are a descriptive fixed operating "
        "point, not a tuned threshold. The dev-selected EER threshold is "
        "recorded in `summary.csv` for provenance but is not the primary "
        "operating point.\n"
    )
    lines.append(
        f"Conditions: {', '.join(f'`{name}`' for name in conditions)}. "
        f"Seeds: {', '.join(str(seed) for seed in seeds)}.\n"
    )

    missing = run_rows and all(not row["present"] for row in run_rows)
    if missing:
        lines.append(
            "> No `metrics.json` files were found; run "
            "`make e1-2-evaluate` first.\n"
        )

    lines.append("## Eval summary by condition\n")
    lines.extend(_summary_table(by_condition, "eval"))
    lines.append("## Dev summary by condition (model-selection split)\n")
    lines.extend(_summary_table(by_condition, "dev"))
    lines.append("## Eval per-attack EER (mean across seeds)\n")
    lines.extend(_per_attack_table(per_attack_by_condition, "eval"))
    lines.append("## Colormap RGB locus\n")
    lines.extend(_colormap_table(colormap_statistics))

    lines.append("## Run inventory\n")
    lines.append("| Condition | Seed | metrics.json | present |")
    lines.append("|---|---:|---|:---:|")
    for row in run_rows:
        lines.append(
            f"| `{row['condition']}` | {row['seed']} | "
            f"`{row['metrics_path']}` | "
            f"{'yes' if row['present'] else 'no'} |"
        )
    lines.append("")
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
        choices=RENDER_CONDITION_ORDER,
        default=list(RENDER_CONDITION_ORDER),
        help="conditions to summarize (default: all eight)",
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

    summary_rows: list[dict[str, object]] = []
    attack_rows: list[dict[str, object]] = []
    run_rows: list[dict[str, object]] = []
    for condition in args.conditions:
        for seed in args.seeds:
            path = metrics_path(runs_root, condition, seed, args.limit)
            present = path.is_file()
            run_rows.append(
                {
                    "condition": condition,
                    "seed": seed,
                    "metrics_path": str(path),
                    "present": present,
                }
            )
            if not present:
                print(f"warning: missing {path}", file=sys.stderr)
                continue
            metrics = _load_metrics(path)
            summary_rows.extend(
                _summary_rows(condition, seed, metrics, path)
            )
            attack_rows.extend(
                _per_attack_rows(condition, seed, metrics)
            )

    summary = pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS)
    per_attack = pd.DataFrame(attack_rows, columns=PER_ATTACK_COLUMNS)
    by_condition = _aggregate_by_condition(summary)
    per_attack_by_condition = _aggregate_per_attack(per_attack)

    colormaps = tuple(
        sorted(
            {
                spec.colormap
                for spec in RENDER_CONDITIONS.values()
                if spec.name in set(args.conditions)
            }
        )
    )
    colormap_statistics = _colormap_statistics(colormaps or ("magma",))

    summary_path = output_root / "summary.csv"
    by_condition_path = output_root / "summary_by_condition.csv"
    per_attack_path = output_root / "per_attack_eer.csv"
    per_attack_by_condition_path = (
        output_root / "per_attack_by_condition.csv"
    )
    colormap_path = output_root / "colormap_statistics.csv"
    results_path = output_root / "RESULTS.md"

    summary.to_csv(summary_path, index=False)
    by_condition.to_csv(by_condition_path, index=False)
    per_attack.to_csv(per_attack_path, index=False)
    per_attack_by_condition.to_csv(per_attack_by_condition_path, index=False)
    colormap_statistics.to_csv(colormap_path, index=False)
    results_path.write_text(
        _results_markdown(
            summary,
            by_condition,
            per_attack_by_condition,
            colormap_statistics,
            run_rows,
            list(args.conditions),
            list(args.seeds),
        ),
        encoding="utf-8",
    )

    print(f"wrote {summary_path}")
    print(f"wrote {by_condition_path}")
    print(f"wrote {per_attack_path}")
    print(f"wrote {per_attack_by_condition_path}")
    print(f"wrote {colormap_path}")
    print(f"wrote {results_path}")
    if not by_condition.empty:
        print("\n" + by_condition.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
