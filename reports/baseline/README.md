# Frozen two-checkpoint baseline (E0.3)

**Status:** frozen 2026-11-13
**Git commit at run time:** `67a81104592090857ffd8b85de87edbc4ade9e07`
(clean working tree, as recorded in every `metrics.json`)
**Preprocessing version:** `mel128-magma-4s-v1`
**Training manifest for both checkpoints:** `manifests/train_seed42.csv`
(SHA-256 `22f98abdf3392e04f7df3632809f81d033d5687773946cd103f5066846ed195e`,
10,080 rows: 2,580 bonafide / 7,500 spoof)
**Raw dataset:** ASVspoof 2019 LA, verified by
`manifests/asvspoof2019_la.meta.json`
**Full experiment report:** [`reports/phase0/E0.3/README.md`](../phase0/E0.3/README.md)

This directory is the frozen E0.3 baseline. A future change is compared against
these numbers only if it uses the same preprocessing contract, the same manifest,
and the same dev/eval split definitions.

## Metric contract

Every result below is on ASVspoof 2019 LA with image input (stored PNGs):

| Split | Attack subset | Threshold(s) | Samples | Bonafide / spoof |
|---|---|---|---:|---:|
| dev | A01–A06 | threshold-free, fixed `0.5`, dev-selected EER threshold | 24,844 | 2,548 / 22,296 |
| eval | A07–A19 | threshold-free, fixed `0.5`, dev-selected EER threshold transferred unchanged | 71,237 | 7,355 / 63,882 |

The dev EER threshold is selected **per checkpoint on dev only**. Eval is never
used for threshold selection. Every eval EER threshold stored in `metrics.json`
is a descriptive DET point, not an operating threshold.

**Dev is a model-selection split, not a clean holdout.** The original notebook
used full dev for early stopping, so the reported dev numbers are optimistic.
Eval is the clean performance estimate (different speakers and unseen attacks),
although both splits come from the same corpus.

## Checkpoints

| Checkpoint | SHA-256 | Parameters | Head parameters | Head state SHA-256 (prefix) | Final head bias | Batch-norm steps |
|---|---|---:|---:|---|---:|---:|
| `best_audio_spoof_model_unweighted.pth` | `abf8d9c9169d363bf9e9076b61689056a78d814c95de9634aea9acf96afc54d8` | 11,242,305 | 65,793 | `de07fcb49e82…` | −0.234170 | 4,095 |
| `best_audio_spoof_model_weighted.pth` | `214e2c813c4f48602ddf124590daf20a008f18385a5339029736a6c16837eb93` | 11,242,305 | 65,793 | `48c383336ddd…` | −0.500105 | 6,930 |

Both checkpoints load strictly: 124 state-dict keys, all 124 remapped from
`network.*` to `backbone.*`, with **zero missing and zero unexpected keys**. The
11,242,305 parameters are 11,176,512 backbone + 65,793 head; the extra 9,620
batch-norm buffer elements bring the tensor total to 11,251,925.

## Frozen baseline table

Threshold-free EER/AUC, all spoof clips against all bonafide clips in the split:

| Checkpoint | Split | Attack subset | Threshold | Samples (bonafide/spoof) | EER | AUC |
|---|---|---|---|---:|---:|---:|
| unweighted | dev | A01–A06 | threshold-free | 24,844 (2,548 / 22,296) | **6.16%** | 0.9860 |
| unweighted | eval | A07–A19 | threshold-free | 71,237 (7,355 / 63,882) | **11.50%** | 0.9527 |
| weighted | dev | A01–A06 | threshold-free | 24,844 (2,548 / 22,296) | **5.42%** | 0.9881 |
| weighted | eval | A07–A19 | threshold-free | 71,237 (7,355 / 63,882) | **11.13%** | 0.9558 |

Operating-threshold behaviour:

| Checkpoint | Split | Attack subset | Threshold | Samples (bonafide/spoof) | Accuracy | Bonafide recall | Spoof recall | FPR | FNR |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| unweighted | dev | A01–A06 | `0.5` | 24,844 (2,548 / 22,296) | 96.51% | 80.18% | 98.38% | 19.82% | 1.62% |
| unweighted | dev | A01–A06 | `0.839741` (dev EER) | 24,844 (2,548 / 22,296) | 93.85% | 93.84% | 93.86% | 6.16% | 6.14% |
| unweighted | eval | A07–A19 | `0.5` | 71,237 (7,355 / 63,882) | 90.21% | 83.45% | 90.99% | 16.55% | 9.01% |
| unweighted | eval | A07–A19 | `0.839741` (transferred) | 71,237 (7,355 / 63,882) | 83.84% | 94.98% | 82.55% | 5.02% | 17.45% |
| weighted | dev | A01–A06 | `0.5` | 24,844 (2,548 / 22,296) | 96.38% | 87.99% | 97.34% | 12.01% | 2.66% |
| weighted | dev | A01–A06 | `0.752020` (dev EER) | 24,844 (2,548 / 22,296) | 94.59% | 94.58% | 94.59% | 5.42% | 5.41% |
| weighted | eval | A07–A19 | `0.5` | 71,237 (7,355 / 63,882) | 88.05% | 90.58% | 87.76% | 9.42% | 12.24% |
| weighted | eval | A07–A19 | `0.752020` (transferred) | 71,237 (7,355 / 63,882) | 83.91% | 95.32% | 82.60% | 4.68% | 17.40% |

Accuracy is misleading under the ~1:2.9 (dev) and ~1:8.7 (eval) class imbalance.
Per-class recall and EER are the meaningful numbers.

**Reading the table:** the weighted checkpoint improves eval EER by 0.37
percentage points (11.50% → 11.13%) and cuts the fixed-`0.5` false-positive rate
from 16.55% to 9.42%, at the cost of spoof recall (90.99% → 87.76%). The gain is
small relative to the change in operating point; neither checkpoint is calibrated.

## Reproducibility

A second full run into `/tmp/opencode/e0.3-rerun` reproduced the canonical run:

| Checkpoint | Split | EER (first / second) | Δ EER | Δ AUC | Score CSV |
|---|---|---:|---:|---:|---|
| unweighted | dev | 0.061554 / 0.061554 | 0.000000 | 0.000000 | byte-identical (`1fbd18f82cf0…`) |
| unweighted | eval | 0.115024 / 0.115024 | 0.000000 | 0.000000 | byte-identical (`209a85c37b56…`) |
| weighted | dev | 0.054170 / 0.054170 | 0.000000 | 0.000000 | byte-identical (`5243b8e99886…`) |
| weighted | eval | 0.111349 / 0.111349 | 0.000000 | 0.000000 | byte-identical (`7494cfb155e0…`) |

The pre-registered tolerance was ±1.0 absolute EER point (`0.01` fraction). Both
checkpoints reproduce at exactly zero difference; `metrics.json` and `summary.md`
differ only in the generation timestamp.

## Checkpoint-load and head-control audit

`reports/baseline/freeze_report.json` records the full audit. The metric runs
were recorded at code commit `67a8110` with a clean tree. The committed audit was
regenerated at report commit `04e8c84`, also with a clean tree, over the same
immutable metric and rerun files; its `git.dirty` field is `false`.

- Training manifest hash, row count, and class balance match the frozen values.
- Both checkpoints load with zero missing/unexpected keys and the expected
  parameter counts.
- Head control on a deterministic 1,000-clip dev subset
  (split dev, A01–A06, 500 bonafide / 500 spoof, threshold-free, seed 0):

| Checkpoint | Trained-head AUC | Trained-head EER | Reinitialised-head AUC | Reinitialised-head EER | AUC gap | Gate |
|---|---:|---:|---:|---:|---:|---|
| unweighted | 0.9994 | 1.40% | 0.5689 | 45.00% | 0.4304 | pass |
| weighted | 0.9994 | 1.40% | 0.5590 | 46.20% | 0.4404 | pass |

The pre-registered criteria were: trained AUC ≥ 0.90, reinitialised AUC ≤ 0.70,
and gap ≥ 0.20. Both checkpoints pass, so the classifier head is trained rather
than randomly initialised.

## Commands

```bash
# Canonical run (both checkpoints, full dev+eval)
make evaluate

# Second full run for the reproducibility check
make evaluate-rerun

# Manifest/checkpoint/contract/reproducibility/head-control audit
uv run python scripts/freeze_baseline.py
```

`make freeze-baseline` runs all three steps. The canonical run writes
`reports/baseline/<checkpoint-stem>/`; the rerun writes to
`/tmp/opencode/e0.3-rerun/` (override with `RERUN_ROOT=...`).

## Artifacts

- `reports/baseline/best_audio_spoof_model_unweighted/` — `metrics.json`,
  `scores_dev.csv`, `scores_eval.csv`, `summary.md` (refreshed in place from the
  E0.2 run; score CSVs are byte-identical to E0.2).
- `reports/baseline/best_audio_spoof_model_weighted/` — same set, first full run.
- `reports/baseline/freeze_report.json` — machine-readable audit.
- `src/spoof_detection/baseline.py` — audit helpers.
- `scripts/freeze_baseline.py` — audit CLI.
- `tests/test_baseline.py` — contract, comparison, manifest, and head-control
  tests.

## Limitations

- Scores are uncalibrated; ECE and Brier are diagnostics only (AGENTS.md G5).
- Min t-DCF, confidence intervals, and significance tests are not implemented.
- Per-attack EER reuses all bonafide clips of the split as negatives.
- Dev was used for early stopping by the original training loop, so it is a
  model-selection split; eval is the clean estimate.
- Training provenance is incomplete: the unweighted notebook training cell and
  output are missing, and the weighted checkpoint's batch-norm counter (6,930
  steps ≈ 22 epochs at 315 steps/epoch) conflicts with the visible notebook run.
  The exact training schedules are not reproducible from the committed notebook
  and are not claimed as such.
