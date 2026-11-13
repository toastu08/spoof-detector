"""Audit and freeze the E0.3 two-checkpoint baseline.

The E0.3 gate requires more than a pair of metric files: it must show that the
checkpoints loaded exactly, that the classifier head is not an untrained random
head, that the evaluation contract was met, and that a second full run reproduces
the first within a pre-registered tolerance.

Only the dev split is used to select the operating threshold.  Eval is never
used for threshold selection.  The notebook training loop used dev for early
stopping, so the reports label dev as a model-selection split rather than a
clean holdout.
"""

from __future__ import annotations

import copy
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd
import torch

from .dataset import repo_relative, sha256_file
from .evaluate import (
    environment_info,
    git_info,
    limit_records,
    load_asvspoof_records,
    resolve_device,
    score_records,
)
from .metrics import compute_auc, compute_eer
from .model import (
    HEAD_PARAMETERS,
    TOTAL_PARAMETERS,
    CheckpointLoadReport,
    load_model_from_checkpoint,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

EXPERIMENT_ID = "E0.3"
DEFAULT_CHECKPOINTS = (
    REPO_ROOT
    / "Models"
    / "Itteration1"
    / "best_audio_spoof_model_unweighted.pth",
    REPO_ROOT
    / "Models"
    / "Itteration1"
    / "best_audio_spoof_model_weighted.pth",
)
DEFAULT_BASELINE_ROOT = REPO_ROOT / "reports" / "baseline"
DEFAULT_RERUN_ROOT = Path("/tmp/opencode/e0.3-rerun")
DEFAULT_MANIFEST = REPO_ROOT / "manifests" / "train_seed42.csv"

TRAIN_SEED42_SHA256 = (
    "22f98abdf3392e04f7df3632809f81d033d5687773946cd103f5066846ed195e"
)
TRAIN_SEED42_ROWS = 10_080
TRAIN_SEED42_COUNTS = {"bonafide": 2_580, "spoof": 7_500}

# The notebook trained with batch_size=32 and the manifest has 10,080 rows, so
# 315 batches/epoch.  The counters are recorded for audit, not as proof of the
# exact run: the weighted checkpoint's counter exceeds the visible 20-epoch
# loop, which means the checkpoint came from a re-executed or unrecorded run.
BATCH_SIZE_INFERRED = 32
STEPS_PER_EPOCH = TRAIN_SEED42_ROWS // BATCH_SIZE_INFERRED
FRESH_INIT_BIAS_BOUND = 1.0 / math.sqrt(128.0)

EXPECTED_SPLITS: dict[str, dict[str, object]] = {
    "dev": {
        "attack_subset": "A01-A06",
        "total": 24_844,
        "bonafide": 2_548,
        "spoof": 22_296,
    },
    "eval": {
        "attack_subset": "A07-A19",
        "total": 71_237,
        "bonafide": 7_355,
        "spoof": 63_882,
    },
}

HEAD_CONTROL_LIMIT = 1_000
HEAD_CONTROL_SEED = 0
HEAD_CONTROL_MIN_TRAINED_AUC = 0.90
HEAD_CONTROL_MAX_RANDOM_AUC = 0.70
HEAD_CONTROL_MIN_GAP = 0.20
MAX_EER_DELTA = 0.01


def _resolve(path: str | Path, repo_root: Path = REPO_ROOT) -> Path:
    path = Path(path)
    return path if path.is_absolute() else (repo_root / path).resolve()


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _number(value: object) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(numeric) or math.isinf(numeric) else numeric


def audit_checkpoint(
    checkpoint_path: str | Path,
    repo_root: Path = REPO_ROOT,
) -> dict[str, object]:
    """Load one checkpoint strictly and return its audit record."""
    checkpoint_path = _resolve(checkpoint_path, repo_root)
    _model, report = load_model_from_checkpoint(checkpoint_path, device="cpu")
    steps = list(report.batch_norm_steps)
    inferred_epochs = [
        round(step / STEPS_PER_EPOCH, 3) for step in steps
    ]
    return {
        "path": repo_relative(checkpoint_path, repo_root),
        "sha256": report.sha256,
        "state_dict_keys": report.state_dict_keys,
        "remapped_keys": report.remapped_keys,
        "missing_keys": list(report.missing_keys),
        "unexpected_keys": list(report.unexpected_keys),
        "total_parameters": report.total_parameters,
        "backbone_parameters": report.backbone_parameters,
        "head_parameters": report.head_parameters,
        "head_state_sha256": report.head_state_sha256,
        "head_last_bias": report.head_last_bias,
        "head_last_bias_within_fresh_init_bound": (
            abs(report.head_last_bias) <= FRESH_INIT_BIAS_BOUND
        ),
        "fresh_init_bias_bound": FRESH_INIT_BIAS_BOUND,
        "batch_norm_steps": steps,
        "inferred_epochs_at_batch32": inferred_epochs,
        "parameter_counts_match": (
            report.total_parameters == TOTAL_PARAMETERS
            and report.head_parameters == HEAD_PARAMETERS
        ),
    }


def audit_training_manifest(
    manifest_path: str | Path,
    repo_root: Path = REPO_ROOT,
    expected_sha256: str = TRAIN_SEED42_SHA256,
    expected_rows: int = TRAIN_SEED42_ROWS,
    expected_counts: Mapping[str, int] = TRAIN_SEED42_COUNTS,
) -> dict[str, object]:
    """Verify the seed-42 training manifest hash, row count, and balance."""
    manifest_path = _resolve(manifest_path, repo_root)
    failures: list[str] = []
    if not manifest_path.is_file():
        return {
            "path": repo_relative(manifest_path, repo_root),
            "passed": False,
            "failures": [f"manifest not found: {manifest_path}"],
        }

    frame = pd.read_csv(manifest_path)
    if "label" not in frame.columns:
        failures.append("manifest has no label column")
        labels = pd.Series(dtype=int)
    else:
        labels = pd.to_numeric(frame["label"], errors="coerce")
        if labels.isna().any():
            failures.append("manifest has non-numeric labels")

    bonafide = int((labels == 0).sum()) if len(labels) else 0
    spoof = int((labels == 1).sum()) if len(labels) else 0
    unknown = int(len(labels) - bonafide - spoof) if len(labels) else 0
    counts = {"bonafide": bonafide, "spoof": spoof}
    actual_sha256 = sha256_file(manifest_path)

    if len(frame) != expected_rows:
        failures.append(
            f"manifest has {len(frame)} rows, expected {expected_rows}"
        )
    if counts != dict(expected_counts):
        failures.append(
            f"manifest counts are {counts}, expected {dict(expected_counts)}"
        )
    if unknown:
        failures.append(f"manifest has {unknown} labels outside 0/1")
    if actual_sha256 != expected_sha256:
        failures.append(
            f"manifest SHA-256 is {actual_sha256}, expected {expected_sha256}"
        )

    return {
        "path": repo_relative(manifest_path, repo_root),
        "sha256": actual_sha256,
        "expected_sha256": expected_sha256,
        "rows": len(frame),
        "expected_rows": expected_rows,
        "counts": counts,
        "expected_counts": dict(expected_counts),
        "passed": not failures,
        "failures": failures,
    }


def validate_baseline_contract(
    metrics: Mapping[str, object],
    checkpoint_audit: Mapping[str, object],
    experiment_id: str = EXPERIMENT_ID,
) -> list[str]:
    """Return a list of violations of the frozen E0.3 metric contract."""
    failures: list[str] = []

    if metrics.get("experiment") != experiment_id:
        failures.append(
            f"experiment is {metrics.get('experiment')!r}, expected "
            f"{experiment_id!r}"
        )
    if metrics.get("partial") is not False:
        failures.append("metrics.json is not a full run")

    checkpoint = metrics.get("checkpoint") or {}
    if checkpoint.get("sha256") != checkpoint_audit.get("sha256"):
        failures.append("checkpoint SHA-256 does not match the loaded checkpoint")
    if checkpoint.get("missing_keys") != []:
        failures.append("checkpoint load has missing keys")
    if checkpoint.get("unexpected_keys") != []:
        failures.append("checkpoint load has unexpected keys")
    if checkpoint.get("total_parameters") != TOTAL_PARAMETERS:
        failures.append("total parameter count does not match the model")
    if checkpoint.get("head_parameters") != HEAD_PARAMETERS:
        failures.append("head parameter count does not match the model")

    splits = metrics.get("splits") or {}
    if set(splits) != set(EXPECTED_SPLITS):
        failures.append(
            f"split set is {sorted(splits)}, expected {sorted(EXPECTED_SPLITS)}"
        )

    for split, expected in EXPECTED_SPLITS.items():
        payload = splits.get(split) or {}
        summary = payload.get("metrics") or {}
        counts = summary.get("sample_counts") or {}
        if payload.get("attack_subset") != expected["attack_subset"]:
            failures.append(
                f"{split}: attack subset is {payload.get('attack_subset')!r}, "
                f"expected {expected['attack_subset']!r}"
            )
        for key in ("total", "bonafide", "spoof"):
            if counts.get(key) != expected[key]:
                failures.append(
                    f"{split}: {key} count is {counts.get(key)}, "
                    f"expected {expected[key]}"
                )
        if summary.get("eer") is None:
            failures.append(f"{split}: EER is missing")
        if summary.get("auc") is None:
            failures.append(f"{split}: AUC is missing")

    thresholds = metrics.get("thresholds") or {}
    if thresholds.get("dev_eer") is None:
        failures.append("dev EER threshold is missing")
    if thresholds.get("selected_on") != "dev":
        failures.append("operating threshold was not selected on dev")

    return failures


def compare_metric_runs(
    reference: Mapping[str, object],
    candidate: Mapping[str, object],
    max_eer_delta: float = MAX_EER_DELTA,
) -> dict[str, object]:
    """Compare two metrics.json payloads on EER, AUC, and contract fields."""
    failures: list[str] = []
    split_reports: dict[str, object] = {}
    reference_splits = reference.get("splits") or {}
    candidate_splits = candidate.get("splits") or {}

    for split in sorted(set(reference_splits) | set(candidate_splits)):
        reference_payload = reference_splits.get(split) or {}
        candidate_payload = candidate_splits.get(split) or {}
        reference_metrics = reference_payload.get("metrics") or {}
        candidate_metrics = candidate_payload.get("metrics") or {}

        reference_eer = _number(reference_metrics.get("eer"))
        candidate_eer = _number(candidate_metrics.get("eer"))
        reference_auc = _number(reference_metrics.get("auc"))
        candidate_auc = _number(candidate_metrics.get("auc"))

        eer_delta = (
            abs(reference_eer - candidate_eer)
            if reference_eer is not None and candidate_eer is not None
            else None
        )
        auc_delta = (
            abs(reference_auc - candidate_auc)
            if reference_auc is not None and candidate_auc is not None
            else None
        )
        counts_match = (
            reference_metrics.get("sample_counts")
            == candidate_metrics.get("sample_counts")
        )
        attack_match = (
            reference_payload.get("attack_subset")
            == candidate_payload.get("attack_subset")
        )
        passed = (
            eer_delta is not None
            and eer_delta <= max_eer_delta
            and counts_match
            and attack_match
        )
        if not passed:
            reasons = []
            if eer_delta is None:
                reasons.append("EER is missing in one or both runs")
            elif eer_delta > max_eer_delta:
                reasons.append(
                    f"EER delta {eer_delta:.6f} exceeds {max_eer_delta:.6f}"
                )
            if not counts_match:
                reasons.append("sample counts differ")
            if not attack_match:
                reasons.append("attack subset differs")
            failures.append(f"{split}: " + "; ".join(reasons))

        split_reports[split] = {
            "eer_reference": reference_eer,
            "eer_rerun": candidate_eer,
            "eer_delta": eer_delta,
            "auc_reference": reference_auc,
            "auc_rerun": candidate_auc,
            "auc_delta": auc_delta,
            "sample_counts_match": counts_match,
            "attack_subset_match": attack_match,
            "passed": passed,
        }

    reference_threshold = _number(
        (reference.get("thresholds") or {}).get("dev_eer")
    )
    candidate_threshold = _number(
        (candidate.get("thresholds") or {}).get("dev_eer")
    )
    threshold_delta = (
        abs(reference_threshold - candidate_threshold)
        if reference_threshold is not None and candidate_threshold is not None
        else None
    )

    return {
        "max_eer_delta": max_eer_delta,
        "dev_eer_threshold_reference": reference_threshold,
        "dev_eer_threshold_rerun": candidate_threshold,
        "dev_eer_threshold_delta": threshold_delta,
        "splits": split_reports,
        "passed": not failures,
        "failures": failures,
    }


def evaluate_head_control(
    trained_auc: float,
    random_auc: float,
    min_trained_auc: float = HEAD_CONTROL_MIN_TRAINED_AUC,
    max_random_auc: float = HEAD_CONTROL_MAX_RANDOM_AUC,
    min_gap: float = HEAD_CONTROL_MIN_GAP,
) -> dict[str, object]:
    """Apply the pre-registered trained-vs-random-head criteria."""
    gap = trained_auc - random_auc
    failures: list[str] = []
    if trained_auc < min_trained_auc:
        failures.append(
            f"trained-head AUC {trained_auc:.4f} < {min_trained_auc:.4f}"
        )
    if random_auc > max_random_auc:
        failures.append(
            f"random-head AUC {random_auc:.4f} > {max_random_auc:.4f}"
        )
    if gap < min_gap:
        failures.append(f"AUC gap {gap:.4f} < {min_gap:.4f}")
    return {
        "trained_auc": trained_auc,
        "random_head_auc": random_auc,
        "auc_gap": gap,
        "min_trained_auc": min_trained_auc,
        "max_random_head_auc": max_random_auc,
        "min_gap": min_gap,
        "passed": not failures,
        "failures": failures,
    }


def run_head_control(
    checkpoint_path: str | Path,
    records: Sequence[object],
    device: torch.device,
    batch_size: int = 64,
    num_workers: int = 4,
    seed: int = HEAD_CONTROL_SEED,
    min_trained_auc: float = HEAD_CONTROL_MIN_TRAINED_AUC,
    max_random_auc: float = HEAD_CONTROL_MAX_RANDOM_AUC,
    min_gap: float = HEAD_CONTROL_MIN_GAP,
) -> dict[str, object]:
    """Score a deterministic subset with the trained and reinitialized heads."""
    checkpoint_path = Path(checkpoint_path)
    model, _report = load_model_from_checkpoint(checkpoint_path, device=device)
    trained_frame = score_records(
        model,
        records,
        source="image",
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        show_progress=False,
        description=f"head-control:{checkpoint_path.stem}:trained",
    )
    trained_auc = compute_auc(
        trained_frame.loc[trained_frame["label"] == 1, "score"].to_numpy(),
        trained_frame.loc[trained_frame["label"] == 0, "score"].to_numpy(),
    )
    trained_eer = compute_eer(
        trained_frame.loc[trained_frame["label"] == 1, "score"].to_numpy(),
        trained_frame.loc[trained_frame["label"] == 0, "score"].to_numpy(),
    ).eer

    random_model = copy.deepcopy(model)
    torch.manual_seed(seed)
    for module in random_model.backbone.fc.modules():
        reset = getattr(module, "reset_parameters", None)
        if callable(reset):
            reset()
    random_model.eval()
    random_model.requires_grad_(False)
    random_frame = score_records(
        random_model,
        records,
        source="image",
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        show_progress=False,
        description=f"head-control:{checkpoint_path.stem}:random",
    )
    random_auc = compute_auc(
        random_frame.loc[random_frame["label"] == 1, "score"].to_numpy(),
        random_frame.loc[random_frame["label"] == 0, "score"].to_numpy(),
    )
    random_eer = compute_eer(
        random_frame.loc[random_frame["label"] == 1, "score"].to_numpy(),
        random_frame.loc[random_frame["label"] == 0, "score"].to_numpy(),
    ).eer

    result = evaluate_head_control(
        trained_auc,
        random_auc,
        min_trained_auc=min_trained_auc,
        max_random_auc=max_random_auc,
        min_gap=min_gap,
    )
    result.update(
        {
            "n": len(records),
            "n_bonafide": sum(
                1 for record in records if getattr(record, "label", None) == 0
            ),
            "n_spoof": sum(
                1 for record in records if getattr(record, "label", None) == 1
            ),
            "selection": (
                "deterministic head-stratified: first 500 records from each "
                "class, sorted by source row"
            ),
            "seed": seed,
            "trained_eer": trained_eer,
            "random_head_eer": random_eer,
        }
    )
    return result


def build_freeze_report(
    checkpoints: Sequence[str | Path] = DEFAULT_CHECKPOINTS,
    baseline_root: str | Path = DEFAULT_BASELINE_ROOT,
    rerun_root: str | Path = DEFAULT_RERUN_ROOT,
    manifest_path: str | Path = DEFAULT_MANIFEST,
    max_eer_delta: float = MAX_EER_DELTA,
    min_trained_auc: float = HEAD_CONTROL_MIN_TRAINED_AUC,
    max_random_auc: float = HEAD_CONTROL_MAX_RANDOM_AUC,
    min_gap: float = HEAD_CONTROL_MIN_GAP,
    experiment_id: str = EXPERIMENT_ID,
    device: str = "auto",
    batch_size: int = 64,
    num_workers: int = 4,
    head_control_limit: int = HEAD_CONTROL_LIMIT,
    repo_root: Path = REPO_ROOT,
) -> dict[str, object]:
    """Build the machine-readable E0.3 freeze report."""
    baseline_root = _resolve(baseline_root, repo_root)
    rerun_root = _resolve(rerun_root, repo_root)
    manifest_path = _resolve(manifest_path, repo_root)
    resolved_device = resolve_device(device)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    failures: list[str] = []
    manifest_report = audit_training_manifest(
        manifest_path,
        repo_root=repo_root,
    )
    failures.extend(
        f"manifest: {failure}" for failure in manifest_report["failures"]
    )

    dev_records, dev_provenance = load_asvspoof_records(
        "dev",
        source="image",
        repo_root=repo_root,
    )
    control_records = limit_records(dev_records, head_control_limit)

    checkpoint_reports: list[dict[str, object]] = []
    for checkpoint_path in checkpoints:
        checkpoint_path = _resolve(checkpoint_path, repo_root)
        stem = checkpoint_path.stem
        checkpoint_failures: list[str] = []
        audit = audit_checkpoint(checkpoint_path, repo_root=repo_root)
        if not audit["parameter_counts_match"]:
            checkpoint_failures.append("parameter counts do not match the model")

        baseline_metrics_path = baseline_root / stem / "metrics.json"
        rerun_metrics_path = rerun_root / stem / "metrics.json"
        baseline_metrics: dict[str, object] | None = None
        rerun_metrics: dict[str, object] | None = None
        if not baseline_metrics_path.is_file():
            checkpoint_failures.append(
                f"baseline metrics not found: {baseline_metrics_path}"
            )
        else:
            baseline_metrics = _load_json(baseline_metrics_path)
        if not rerun_metrics_path.is_file():
            checkpoint_failures.append(
                f"rerun metrics not found: {rerun_metrics_path}"
            )
        else:
            rerun_metrics = _load_json(rerun_metrics_path)

        contract_failures: list[str] = []
        comparison: dict[str, object] | None = None
        if baseline_metrics is not None:
            contract_failures = validate_baseline_contract(
                baseline_metrics,
                audit,
                experiment_id=experiment_id,
            )
            checkpoint_failures.extend(contract_failures)
        if baseline_metrics is not None and rerun_metrics is not None:
            comparison = compare_metric_runs(
                baseline_metrics,
                rerun_metrics,
                max_eer_delta=max_eer_delta,
            )
            checkpoint_failures.extend(comparison["failures"])

        head_control = run_head_control(
            checkpoint_path,
            control_records,
            device=resolved_device,
            batch_size=batch_size,
            num_workers=num_workers,
            min_trained_auc=min_trained_auc,
            max_random_auc=max_random_auc,
            min_gap=min_gap,
        )
        checkpoint_failures.extend(
            f"head control: {failure}" for failure in head_control["failures"]
        )

        checkpoint_report = {
            "path": audit["path"],
            "stem": stem,
            "audit": audit,
            "baseline_metrics_path": repo_relative(
                baseline_metrics_path, repo_root
            ),
            "rerun_metrics_path": str(rerun_metrics_path),
            "contract_failures": contract_failures,
            "comparison": comparison,
            "head_control": head_control,
            "passed": not checkpoint_failures,
            "failures": checkpoint_failures,
        }
        checkpoint_reports.append(checkpoint_report)
        failures.extend(
            f"{stem}: {failure}" for failure in checkpoint_failures
        )

    return {
        "experiment": experiment_id,
        "title": "Frozen two-checkpoint baseline audit",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_root": repo_relative(baseline_root, repo_root),
        "rerun_root": str(rerun_root),
        "training_manifest": manifest_report,
        "dev_control_subset": {
            "split": "dev",
            "attack_subset": "A01-A06",
            "provenance": dev_provenance,
            "n": len(control_records),
            "n_bonafide": sum(
                1 for record in control_records if record.label == 0
            ),
            "n_spoof": sum(
                1 for record in control_records if record.label == 1
            ),
            "selection": (
                f"deterministic head-stratified: first {head_control_limit // 2} "
                "records from each class, sorted by source row"
            ),
        },
        "thresholds": {
            "max_eer_delta": max_eer_delta,
            "max_eer_delta_units": "absolute EER fraction (0.01 = 1 point)",
            "min_trained_auc": min_trained_auc,
            "max_random_head_auc": max_random_auc,
            "min_head_gap": min_gap,
            "selected_on": "dev",
        },
        "environment": environment_info(resolved_device),
        "git": git_info(repo_root),
        "checkpoints": checkpoint_reports,
        "passed": not failures,
        "failures": failures,
    }


def write_freeze_report(
    path: str | Path,
    report: Mapping[str, object],
) -> Path:
    """Write the audit report as pretty-printed JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


__all__ = [
    "DEFAULT_BASELINE_ROOT",
    "DEFAULT_CHECKPOINTS",
    "DEFAULT_MANIFEST",
    "DEFAULT_RERUN_ROOT",
    "EXPERIMENT_ID",
    "EXPECTED_SPLITS",
    "FRESH_INIT_BIAS_BOUND",
    "HEAD_CONTROL_LIMIT",
    "STEPS_PER_EPOCH",
    "TRAIN_SEED42_COUNTS",
    "TRAIN_SEED42_ROWS",
    "TRAIN_SEED42_SHA256",
    "audit_checkpoint",
    "audit_training_manifest",
    "build_freeze_report",
    "compare_metric_runs",
    "evaluate_head_control",
    "run_head_control",
    "validate_baseline_contract",
    "write_freeze_report",
]
