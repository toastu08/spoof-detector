# E0.3 — Freeze the two-checkpoint baseline

**Status:** complete (2026-11-13)
**Git commit at run time:** `67a81104592090857ffd8b85de87edbc4ade9e07`
(clean working tree; recorded in every `metrics.json`)
**Preprocessing version:** `mel128-magma-4s-v1`
**Training manifest:** `manifests/train_seed42.csv`
(SHA-256 `22f98abdf3392e04f7df3632809f81d033d5687773946cd103f5066846ed195e`)
**Frozen baseline:** [`reports/baseline/README.md`](../../baseline/README.md)
**Audit output:** [`reports/baseline/freeze_report.json`](../../baseline/freeze_report.json)
**Audit regeneration commit:** `04e8c8485b9744bffdb3b0e52bc99ffbdc7ba9d2`
(clean tree; regenerated over the same immutable metric and rerun files)

## Metric contract

| Split | Attack subset | Threshold(s) | Samples | Bonafide / spoof |
|---|---|---|---:|---:|
| dev | A01–A06 | threshold-free, fixed `0.5`, dev-selected EER threshold | 24,844 | 2,548 / 22,296 |
| eval | A07–A19 | threshold-free, fixed `0.5`, dev-selected EER threshold transferred unchanged | 71,237 | 7,355 / 63,882 |

The dev EER threshold is selected separately for each checkpoint on dev only.
Eval is never used for threshold selection. All eval "EER thresholds" in
`metrics.json` are descriptive DET points.

**Split roles:** the original training loop used full dev for early stopping, so
dev is a **model-selection split** and its numbers are optimistic. Eval is the
clean estimate: it uses unseen attacks A07–A19 and, per the ASVspoof protocol,
different speakers. Exact speaker-overlap verification and train/dev/eval
leakage are E0.5, not this experiment.

## Question

How do the unweighted and weighted first-iteration checkpoints compare under the
same harness, and can the resulting table be frozen reproducibly?

## Method

- Evaluate both checkpoints on full dev and full eval with image input.
- Load each checkpoint strictly and record SHA-256, key remapping, missing and
  unexpected keys, parameter counts, head fingerprint, final head bias, and
  batch-norm step counters.
- Verify the training manifest hash, row count, and class balance.
- Verify the metric contract: experiment id, full run, attack subset, sample
  counts, dev-only threshold selection.
- Run a second full evaluation into a separate root and compare EER, AUC, and
  sample counts.
- Run a trained-vs-reinitialised head control on a deterministic dev subset.

Pre-registered success criteria:

1. One baseline table with both checkpoints and all four metric-contract fields.
2. Second run within ±1.0 absolute EER point (`0.01` fraction).
3. Both checkpoints load with zero missing/unexpected keys and 11,242,305
   parameters / 65,793 head parameters.
4. The classifier head is trained, not random: trained-head AUC ≥ 0.90,
   reinitialised-head AUC ≤ 0.70, gap ≥ 0.20 on the control subset.

## Commands

```bash
# Canonical frozen run (both checkpoints, full dev+eval)
make evaluate

# Second full run into /tmp/opencode/e0.3-rerun
make evaluate-rerun

# Manifest, checkpoint, contract, reproducibility, and head-control audit
uv run python scripts/freeze_baseline.py
```

The canonical run refreshes
`reports/baseline/best_audio_spoof_model_unweighted/` in place and creates
`reports/baseline/best_audio_spoof_model_weighted/`. The unweighted score CSVs
are byte-identical to the E0.2 run; only the report metadata changed.

## Checkpoint and load audit

| Checkpoint | SHA-256 | Keys (remapped) | Missing | Unexpected | Parameters | Head parameters |
|---|---|---:|---:|---:|---:|---:|
| unweighted | `abf8d9c9169d363bf9e9076b61689056a78d814c95de9634aea9acf96afc54d8` | 124 (124) | 0 | 0 | 11,242,305 | 65,793 |
| weighted | `214e2c813c4f48602ddf124590daf20a008f18385a5339029736a6c16837eb93` | 124 (124) | 0 | 0 | 11,242,305 | 65,793 |

Both checkpoints also record a distinct head-state fingerprint, a trained final
head bias, and matching batch-norm counters across all layers:

| Checkpoint | Head state SHA-256 | Final head bias | Batch-norm steps | Steps ÷ 315 |
|---|---|---:|---:|---:|
| unweighted | `de07fcb49e82ce28add233e10753f8885bfb1a11e0177cb8b89bcb6fc11685d4` | −0.234170 | 4,095 | 13.0 |
| weighted | `48c383336ddd257763adbbf00c94bd0a9047f4597375b7dfa3f683cf9422382a` | −0.500105 | 6,930 | 22.0 |

`315` is 10,080 manifest rows ÷ batch size 32. The counters are an audit signal,
not proof of the exact training schedule (see "Training provenance" below).

The training-manifest audit passed: 10,080 rows, 2,580 bonafide / 7,500 spoof,
SHA-256 matches the frozen value.

## Results

### Frozen baseline table

Threshold-free EER/AUC, all spoof clips against all bonafide clips:

| Checkpoint | Split | Attack subset | Threshold | Samples (bonafide/spoof) | EER | AUC |
|---|---|---|---|---:|---:|---:|
| unweighted | dev | A01–A06 | threshold-free | 24,844 (2,548 / 22,296) | **6.16%** | 0.9860 |
| unweighted | eval | A07–A19 | threshold-free | 71,237 (7,355 / 63,882) | **11.50%** | 0.9527 |
| weighted | dev | A01–A06 | threshold-free | 24,844 (2,548 / 22,296) | **5.42%** | 0.9881 |
| weighted | eval | A07–A19 | threshold-free | 71,237 (7,355 / 63,882) | **11.13%** | 0.9558 |

At the fixed `0.5` threshold:

| Checkpoint | Split | Attack subset | Threshold | Samples (bonafide/spoof) | Accuracy | Bonafide recall | Spoof recall | FPR | FNR |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| unweighted | dev | A01–A06 | `0.5` | 24,844 (2,548 / 22,296) | 96.51% | 80.18% | 98.38% | 19.82% | 1.62% |
| unweighted | eval | A07–A19 | `0.5` | 71,237 (7,355 / 63,882) | 90.21% | 83.45% | 90.99% | 16.55% | 9.01% |
| weighted | dev | A01–A06 | `0.5` | 24,844 (2,548 / 22,296) | 96.38% | 87.99% | 97.34% | 12.01% | 2.66% |
| weighted | eval | A07–A19 | `0.5` | 71,237 (7,355 / 63,882) | 88.05% | 90.58% | 87.76% | 9.42% | 12.24% |

At each checkpoint's dev-selected EER threshold:

| Checkpoint | Split | Attack subset | Threshold | Samples (bonafide/spoof) | Accuracy | Bonafide recall | Spoof recall | FPR | FNR |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| unweighted | dev | A01–A06 | `0.839741` | 24,844 (2,548 / 22,296) | 93.85% | 93.84% | 93.86% | 6.16% | 6.14% |
| unweighted | eval | A07–A19 | `0.839741` | 71,237 (7,355 / 63,882) | 83.84% | 94.98% | 82.55% | 5.02% | 17.45% |
| weighted | dev | A01–A06 | `0.752020` | 24,844 (2,548 / 22,296) | 94.59% | 94.58% | 94.59% | 5.42% | 5.41% |
| weighted | eval | A07–A19 | `0.752020` | 71,237 (7,355 / 63,882) | 83.91% | 95.32% | 82.60% | 4.68% | 17.40% |

Accuracy is misleading under the ~1:2.9 (dev) and ~1:8.7 (eval) imbalance.

### Per-attack EER

Per-attack EER uses that attack's spoof clips as positives and all bonafide
clips in the split as negatives. Split, attack subset, threshold, and sample
counts are as stated in the section heading and table.

Dev (A01–A06, threshold-free, 24,844 clips: 2,548 bonafide / 22,296 spoof,
3,716 spoof clips per attack):

| Attack | Unweighted EER | Weighted EER |
|---|---:|---:|
| A01 | 1.49% | 1.64% |
| A02 | 1.57% | 1.92% |
| A03 | 2.62% | 2.51% |
| A04 | 5.11% | 4.39% |
| A05 | 8.43% | 7.50% |
| A06 | 10.90% | 10.12% |

Eval (A07–A19, threshold-free, 71,237 clips: 7,355 bonafide / 63,882 spoof,
4,914 spoof clips per attack):

| Attack | Unweighted EER | Weighted EER |
|---|---:|---:|
| A07 | 1.30% | 1.26% |
| A08 | 2.95% | 2.83% |
| A09 | 1.04% | 1.10% |
| A10 | 6.65% | 5.84% |
| A11 | 3.01% | 2.93% |
| A12 | 6.98% | 5.62% |
| A13 | 5.41% | 5.00% |
| A14 | 4.46% | 4.42% |
| A15 | 5.70% | 5.11% |
| A16 | 4.50% | 3.97% |
| A17 | **36.24%** | **34.67%** |
| A18 | 18.30% | 18.50% |
| A19 | 18.30% | 17.99% |

AUCs per attack, per-speaker breakdowns, calibration diagnostics, and confusion
counts are in each checkpoint's `metrics.json` and `summary.md`.

### Per-speaker extremes

Per-speaker EER uses same-speaker bonafide clips as negatives. Split, attack
subset, and threshold are as above; counts are the speaker's scored clips.

| Split | Checkpoint | Worst five speakers (EER) |
|---|---|---|
| dev | unweighted | LA_0071 8.0%, LA_0069 6.5%, LA_0074 6.0%, LA_0077 5.3%, LA_0070 4.9% |
| dev | weighted | LA_0071 7.1%, LA_0069 5.8%, LA_0074 5.3%, LA_0077 4.6%, LA_0070 4.3% |
| eval | unweighted | LA_0038 21.8%, LA_0002 17.6%, LA_0032 17.6%, LA_0041 15.8%, LA_0040 14.7% |
| eval | weighted | LA_0038 22.4%, LA_0002 17.6%, LA_0044 16.3%, LA_0032 14.7%, LA_0041 14.5% |

Dev has 20 speakers, 10 with both classes; eval has 67 speakers, 48 with both
classes. Single-class speakers have no EER and are excluded from the extremes.

## Reproducibility

Second full run into `/tmp/opencode/e0.3-rerun`:

| Checkpoint | Split | EER first / second | Δ EER | Δ AUC | Score CSV |
|---|---|---:|---:|---:|---|
| unweighted | dev | 0.061554 / 0.061554 | 0.000000 | 0.000000 | byte-identical |
| unweighted | eval | 0.115024 / 0.115024 | 0.000000 | 0.000000 | byte-identical |
| weighted | dev | 0.054170 / 0.054170 | 0.000000 | 0.000000 | byte-identical |
| weighted | eval | 0.111349 / 0.111349 | 0.000000 | 0.000000 | byte-identical |

The tolerance was ±0.01 EER fraction (1.0 point). Both checkpoints reproduce at
zero difference. `metrics.json` and `summary.md` differ only in timestamps.

## Head control

Deterministic subset: split dev, A01–A06, 1,000 clips (500 bonafide / 500
spoof), threshold-free, seed 0. The selection is the first 500 records of each
class in source-row order.

| Checkpoint | Trained AUC | Trained EER | Reinitialised AUC | Reinitialised EER | Gap | Gate |
|---|---:|---:|---:|---:|---:|---|
| unweighted | 0.9994 | 1.40% | 0.5689 | 45.00% | 0.4304 | pass |
| weighted | 0.9994 | 1.40% | 0.5590 | 46.20% | 0.4404 | pass |

Criteria: trained AUC ≥ 0.90, reinitialised AUC ≤ 0.70, gap ≥ 0.20. Both pass,
so the head carries learned signal; the high AUC is not produced by an untrained
random head.

## Training provenance

Read from `2_model.ipynb` via the python-notebook MCP:

- Backbone: `torchvision` ResNet18 with ImageNet weights, frozen.
- Head: `Linear(512, 128) → ReLU → Dropout(0.5) → Linear(128, 1)`.
- Loss: `BCEWithLogitsLoss`.
- Optimizer: Adam, learning rate `1e-3`.
- Batch size 32; maximum 20 epochs; early stopping with patience 5 on full dev.
- Weighted run: `pos_weight = 2580 / 7500 = 0.344` on the spoof class, which
  makes bonafide roughly 2.91× the spoof weight.
- No augmentation and no recorded random seed.

Provenance is incomplete and is not claimed to be reproducible:

- The unweighted training cell and its output are missing from the notebook.
- The unweighted batch-norm counter is 4,095 steps = 13.0 epochs at 315
  steps/epoch, which is consistent with a short early-stopped run.
- The weighted batch-norm counter is 6,930 steps = 22.0 epochs at 315
  steps/epoch. This **conflicts** with the visible 20-epoch training loop, so
  the weighted checkpoint came from a re-executed or unrecorded run.
- Exact training schedules, seeds, and per-epoch histories are therefore
  unknown. The frozen baseline describes the checkpoints as they exist, not a
  claim that training can be replayed from the committed notebook.

### Notebook cross-check

The notebook's own eval report at `0.5` was compared with the harness:

| Checkpoint | Notebook accuracy | Notebook FP / FN | Harness accuracy | Harness FP / FN |
|---|---:|---|---:|---|
| unweighted | 0.9021 | 1,219 / 5,758 | 90.21% | 1,217 / 5,758 |
| weighted | 0.8805 | 693 / 7,819 | 88.05% | 693 / 7,820 |

Accuracy matches to four decimals. The unweighted harness finds two fewer false
positives; the weighted harness finds one more false negative. Both eval splits
have hundreds of clips with scores within 0.49–0.51, so a one- or two-clip
difference is plausible from small implementation differences in image loading
or borderline threshold handling. The cause is not established. The frozen
values are the harness counts, and notebook confusion counts should not be
quoted as exact.

## Observations

1. **The weighted checkpoint is slightly better threshold-free, but the gain is
   small.** Eval EER moves from 11.50% to 11.13% and dev from 6.16% to 5.42%.
   The per-attack pattern is nearly unchanged.
2. **The weighting mainly moves the operating point.** At `0.5`, weighted cuts
   eval FPR from 16.55% to 9.42% but lowers spoof recall from 90.99% to 87.76%.
   At the dev-EER threshold the two checkpoints are almost identical on eval
   (82.55% vs 82.60% spoof recall; FPR 5.02% vs 4.68%).
3. **A17 remains the failure mode for both.** EER is 36.24% (unweighted) and
   34.67% (weighted); A18/A19 sit at 18–19%. The aggregate eval EER is dominated
   by A17–A19, the advanced/unknown attacks.
4. **The dev threshold does not transfer cleanly.** Both dev-selected thresholds
   preserve low eval FPR (~5%) at the cost of ~17.4% FNR. There is no calibrated
   operating point yet (AGENTS.md G5).
5. **Dev is optimistic.** It was used for early stopping, so the dev/eval gap
   mixes model selection with genuine domain shift.

## Limitations

- Scores are uncalibrated; Brier and ECE are diagnostics only.
- Min t-DCF, confidence intervals, and significance tests are not implemented.
- Per-attack EER reuses all split bonafide clips as negatives.
- Dev is a model-selection split, not a clean holdout.
- Training schedules and seeds are not fully recoverable; the weighted
  checkpoint provenance is unexplained by the visible notebook run.
- No speaker-overlap or leakage audit yet (E0.5).
- `app.py` still loads with `strict=False`; the harness loads strictly.

## Artifacts

- `reports/baseline/README.md` — frozen baseline table and commands.
- `reports/baseline/best_audio_spoof_model_unweighted/` — refreshed E0.2 output.
- `reports/baseline/best_audio_spoof_model_weighted/` — first full run.
- `reports/baseline/freeze_report.json` — machine-readable audit.
- `src/spoof_detection/baseline.py` — manifest, contract, comparison, and
  head-control helpers.
- `scripts/freeze_baseline.py` — audit CLI.
- `tests/test_baseline.py` — 15 new tests; 83 tests pass in total.
- `make evaluate`, `make evaluate-rerun`, `make freeze-baseline`.
