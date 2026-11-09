# E0.2 — Baseline evaluation harness

**Status:** complete (2026-11-09)
**Checkpoint:** `Models/Itteration1/best_audio_spoof_model_unweighted.pth`
(SHA-256 `abf8d9c9169d363bf9e9076b61689056a78d814c95de9634aea9acf96afc54d8`)
**Git commit at run time:** `4bb5941f9968d1e6b9b224c4c210d4792c2e703c`
(working tree dirty; all E0.2 code was uncommitted)
**Preprocessing version:** `mel128-magma-4s-v1`
**Raw/source dataset:** ASVspoof 2019 LA, verified by
`manifests/asvspoof2019_la.meta.json`
**Training manifest for the checkpoint:** `manifests/train_seed42.csv`
(SHA-256 `22f98abdf3392e04f7df3632809f81d033d5687773946cd103f5066846ed195e`)

## Metric contract

| Split | Attack subset | Threshold(s) | Samples | Bonafide / spoof |
|---|---|---|---:|---:|
| dev | A01–A06 | threshold-free EER/AUC, fixed `0.5`, dev-selected `0.839741` | 24,844 | 2,548 / 22,296 |
| eval | A07–A19 | threshold-free EER/AUC, fixed `0.5`, dev-selected `0.839741` (transferred) | 71,237 | 7,355 / 63,882 |

The **only** data-selected operating threshold is the dev EER threshold,
`0.839741`. It is transferred to eval unchanged. Every eval EER threshold in
`metrics.json` is a descriptive DET-curve point, not an operating threshold.
Nothing was tuned on eval.

## Commands

Full run (repeated twice; the second run overwrote the first):

```bash
uv run python -m spoof_detection.evaluate \
  --checkpoint Models/Itteration1/best_audio_spoof_model_unweighted.pth \
  --asvspoof dev eval \
  --source image \
  --output-root reports/baseline \
  --device cuda --batch-size 64 --num-workers 4
```

Smoke test used for the image/audio parity check:

```bash
uv run python -m spoof_detection.evaluate \
  --checkpoint Models/Itteration1/best_audio_spoof_model_unweighted.pth \
  --asvspoof dev eval --source image --limit 100 \
  --output-dir /tmp/opencode/e0.2-smoke/image \
  --device cuda --batch-size 64 --num-workers 4

uv run python -m spoof_detection.evaluate \
  --checkpoint Models/Itteration1/best_audio_spoof_model_unweighted.pth \
  --asvspoof dev eval --source audio --limit 100 \
  --output-dir /tmp/opencode/e0.2-smoke/audio \
  --device cuda --batch-size 64 --num-workers 4
```

## Results

Threshold-free metrics over all spoof clips against all bonafide clips in the
split:

| Split | Attack subset | EER | AUC | Brier | ECE (10 bins) |
|---|---|---:|---:|---:|---:|
| dev | A01–A06 | **6.16%** | 0.9860 | 0.0260 | 0.0065 |
| eval | A07–A19 | **11.50%** | 0.9527 | 0.0699 | 0.0713 |

At the fixed `0.5` threshold:

| Split | Accuracy | Bonafide recall | Spoof recall | FPR | FNR |
|---|---:|---:|---:|---:|---:|
| dev | 96.51% | 80.18% | 98.38% | 19.82% | 1.62% |
| eval | 90.21% | 83.45% | 90.99% | 16.55% | 9.01% |

At the dev-selected EER threshold `0.839741`:

| Split | Accuracy | Bonafide recall | Spoof recall | FPR | FNR |
|---|---:|---:|---:|---:|---:|
| dev | 93.85% | 93.84% | 93.86% | 6.16% | 6.14% |
| eval | 83.84% | 94.98% | 82.55% | 5.02% | 17.45% |

Accuracy is misleading under the ~1:2.9 and ~1:8.7 class imbalance; per-class
recall and EER are the meaningful numbers.

### Per-attack EER

Each attack's spoof clips are scored against all bonafide clips in the split.

| Split | Attack | n spoof | EER | AUC |
|---|---|---:|---:|---:|
| dev | A01 | 3,716 | 1.49% | 0.9987 |
| dev | A02 | 3,716 | 1.57% | 0.9986 |
| dev | A03 | 3,716 | 2.62% | 0.9964 |
| dev | A04 | 3,716 | 5.11% | 0.9889 |
| dev | A05 | 3,716 | 8.43% | 0.9748 |
| dev | A06 | 3,716 | 10.90% | 0.9584 |
| eval | A07 | 4,914 | 1.30% | 0.9991 |
| eval | A08 | 4,914 | 2.95% | 0.9959 |
| eval | A09 | 4,914 | 1.04% | 0.9993 |
| eval | A10 | 4,914 | 6.65% | 0.9798 |
| eval | A11 | 4,914 | 3.01% | 0.9953 |
| eval | A12 | 4,914 | 6.98% | 0.9799 |
| eval | A13 | 4,914 | 5.41% | 0.9876 |
| eval | A14 | 4,914 | 4.46% | 0.9919 |
| eval | A15 | 4,914 | 5.70% | 0.9851 |
| eval | A16 | 4,914 | 4.50% | 0.9903 |
| eval | A17 | 4,914 | **36.24%** | 0.6907 |
| eval | A18 | 4,914 | 18.30% | 0.8938 |
| eval | A19 | 4,914 | 18.30% | 0.8964 |

Per-speaker breakdowns are in `metrics.json` under
`splits.<split>.metrics.per_speaker` and in `summary.md`.

## Observations

1. **Eval degrades but does not collapse.** Aggregate EER rises from 6.16% on
   dev to 11.50% on eval. The model was trained only on A01–A06, so A07–A16
   (unseen attacks) generalise well, with per-attack EER between 1.04% and 6.98%.
2. **A17 is the failure mode.** A17 alone reaches 36.24% EER (AUC 0.6907), and
   A18/A19 sit at 18.30%. The eval aggregate is dominated by these three
   attacks. A17–A19 are exactly the attacks that ASVspoof 2019 designates as
   unknown/advanced; this is the capability boundary to investigate in Phase 2.
3. **The dev EER threshold is high and does not transfer well.** The dev EER
   threshold is `0.839741`, reflecting the model's high spoof scores overall.
   Transferred to eval it keeps FPR at 5.02% but drops spoof recall to 82.55%
   (FNR 17.45%). At `0.5`, spoof recall is higher (90.99%) but FPR is 16.55%.
   Neither point is a calibrated operating point.
4. **Calibration is worse out of domain.** ECE rises from 0.0065 on dev to
   0.0713 on eval. Brier rises from 0.0260 to 0.0699. These are diagnostics of
   miscalibration, not evidence that scores are probabilities (AGENTS.md G5).

## Reproducibility

The full command was run twice on 2026-11-09. Results:

- `scores_dev.csv` and `scores_eval.csv`: byte-identical between runs
  (SHA-256 prefixes `1fbd18f8…` and `209a85c3…`).
- All values under `splits.{dev,eval}.metrics` in `metrics.json`: identical.
- `metrics.json` and `summary.md` differ only in the UTC timestamp; the
  `git.status_porcelain` record also reflects files created between runs.
- No missing or unexpected checkpoint keys: 124 state-dict keys, all 124
  remapped from `network.*` to `backbone.*`.
- Parameter counts: 11,242,305 parameters (11,176,512 backbone + 65,793 head),
  plus 9,620 batch-norm buffer elements — 11,251,925 tensor elements in total.
  `AGENTS.md` previously quoted the tensor total as the parameter count.

Runtime: 2m33s (first run) and 2m35s (second run) on an NVIDIA RTX 3050 Laptop
GPU, batch size 64, 4 DataLoader workers.

## Image vs audio input parity

On the deterministic 100-clip-per-split smoke subset (50 bonafide / 50 spoof),
reading the exported PNGs and re-rendering from source FLAC produced
**bit-identical logits and scores** for every clip:

| Split | n | max \|Δscore\| | identical logits |
|---|---:|---:|---|
| dev | 100 | 0.0 | yes |
| eval | 100 | 0.0 | yes |

This confirms that storage as RGBA PNG and the audio path do not diverge in the
current contract.

## Artifacts

- `src/spoof_detection/evaluate.py` — CLI, loaders, inference, report writers.
- `src/spoof_detection/metrics.py` — DET EER, AUC, threshold, calibration,
  per-attack and per-speaker metrics.
- `src/spoof_detection/model.py` — model definition and strict checkpoint load.
- `src/spoof_detection/preprocess.py` — inference transform and audio renderer.
- `src/spoof_detection/protocol.py` — generic train/dev/eval CM protocol loader.
- `tests/test_metrics.py`, `tests/test_model_loading.py`,
  `tests/test_evaluate.py` — 68 tests pass in total.
- `reports/baseline/best_audio_spoof_model_unweighted/` — `metrics.json`,
  `scores_dev.csv`, `scores_eval.csv`, `summary.md`.
- `make evaluate` — shortcut for the full unweighted dev+eval run.
- `spoof-evaluate` — console entry point for the same CLI.

## Limitations

- Scores are uncalibrated. Brier and ECE are reported as diagnostics only.
- Min t-DCF is not reported; ASV score files are not wired in yet.
- No confidence intervals or significance tests yet.
- Per-attack EER reuses all bonafide clips of the split as negatives, so the
  per-attack numbers are not directly comparable to attack-vs-attack rankings.
- The stored PNGs are the full-split exports, not the lost original 10,080
  training images. Membership of the training set comes from
  `manifests/train_seed42.csv`.
- `--limit` runs are smoke tests only; they use the first N/2 records per class
  in protocol order and must not be quoted as baseline results.
- Single-class manifests are supported for external smoke sets: threshold and
  score summaries are produced, while EER/AUC are reported as `n/a`.
