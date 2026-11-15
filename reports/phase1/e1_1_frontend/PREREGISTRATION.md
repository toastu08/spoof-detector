# E1.1 — Is the mel spectrogram necessary? (pre-registration)

**Status:** pre-registered before any probe was trained or evaluated.
**Experiment ID:** E1.1 (ROADMAP phase 1).
**Date:** 2026-11-16.
**Git commit:** unknown at pre-registration; the exact commit and dirty-tree
flag are recorded in each run's `config.json` / `metrics.json`.
**Data manifest:** `manifests/train_seed42.csv`
(SHA-256 `22f98abdf3392e04f7df3632809f81d033d5687773946cd103f5066846ed195e`,
10,080 rows: 2,580 bonafide / 7,500 spoof).

This document fixes the question, conditions, splits, metrics and decision rules
*before* results exist. Any later deviation must be logged in
`reports/phase1/e1_1_frontend/RESULTS.md` with a reason.

## Question

The current detector turns audio into a 128-band `magma` mel image and classifies
it with a frozen ImageNet ResNet18 plus a linear head. Is the **mel spectrogram
necessary**, or would *any* image-like time-frequency representation work equally
well? Concretely: is performance sensitive to mel resolution, and is a non-mel
representation competitive?

## Metric contract (AGENTS.md G4)

Every reported number states split, attack subset, threshold and sample count.

| Split | Role | Attack subset | Samples (bonafide / spoof) |
|---|---|---|---|
| train | fitted head only | A01–A06 | 10,080 (2,580 / 7,500) |
| dev | model selection (head epoch) | A01–A06 | 24,844 (2,548 / 22,296) |
| eval | clean estimate | A07–A19 | 71,237 (7,355 / 63,882) |

- **Primary metric:** threshold-free EER on **eval**. AUC is reported alongside.
- **Secondary metric:** EER on dev (model-selection split).
- **Threshold:** EER and AUC are threshold-free. The fixed `p >= 0.5` operating
  point is reported descriptively (`FPR@0.5`, spoof recall@0.5) and is **not**
  tuned. The dev EER threshold is recorded but is **not** transferred to eval as
  a claimed operating point in the primary table.
- Eval is never used for training, early stopping, head selection, or threshold
  selection.

## Conditions

Six front-ends, all sharing the audio contract (16 kHz, 4 s = 64,000 samples),
`power_to_db(ref=np.max)` (except MFCC), min-max normalisation, `magma` colormap,
and `origin='lower'` vertical flip. Only the 2-D representation changes.

| Condition | Representation | Expected shape (rows × frames) | Role |
|---|---|---|---|
| `mel64` | 64-band mel, dB | 64 × 126 | lower mel resolution |
| `mel128` | 128-band mel, dB | 128 × 126 | **control** (existing contract) |
| `mel256` | 256-band mel, dB | 256 × 126 | higher mel resolution |
| `linear_stft` | linear-frequency power STFT, dB | 1,025 × 126 | non-mel, same n_fft/hop |
| `mfcc40` | 40 MFCC coefficients (no deltas) | 40 × 126 | compact cepstral front-end |
| `randproj128` | fixed Gaussian projection of the linear dB STFT | 128 × 126 | random-feature control |

`mel128` must reproduce `data/dataset_img` pixel-for-pixel. The export records a
sampled parity check and a re-render determinism check per split.

## Fixed training recipe (head only)

- Backbone: `torchvision.models.resnet18(weights="DEFAULT")`, all parameters
  `requires_grad=False`. BatchNorm stays in **train** mode so running statistics
  update, matching `2_model.ipynb`.
- Head: `Linear(512,128) → ReLU → Dropout(0.5) → Linear(128,1)`, freshly and
  deterministically re-initialised with `torch.manual_seed(42)`.
- Loss: unweighted `BCEWithLogitsLoss`. Optimizer: Adam, `lr=1e-3`.
- Batch size 32, at most 20 epochs, early stopping with patience 5 on **full-dev
  BCE loss**. Best epoch = lowest full-dev loss.
- Seed 42 for Python/NumPy/torch, loader shuffling and worker seeding.
- Input transform: resize 224×224, ImageNet mean/std.

Deliberate differences from the notebook: a recorded seed (none was recorded
originally) and an identical deterministic head initialisation across all six
conditions. BatchNorm statistics become front-end-specific; this is a
consequence of the baseline-faithful train-mode BN and is documented, not hidden.

## Predictions / hypotheses

1. **H1 (resolution):** EER improves from `mel64` to `mel128` to `mel256`, with
   diminishing returns; large `mel64` degradation would indicate mel resolution
   matters.
2. **H2 (modality):** a non-mel representation (`linear_stft` or `mfcc40`) is
   within a small margin of `mel128` on eval, suggesting the backbone mostly uses
   generic spectral texture rather than the mel warping.
3. **H3 (control):** `randproj128` is clearly worse than `mel128`; if it is
   competitive, the result is suspicious of an image-statistics shortcut and must
   be investigated before any claim.

## Decision rules

- The primary comparison is the eval EER table across the six conditions, with
  sample counts and attack subset stated. Dev is reported for model selection.
- "Competitive" means the eval EER difference versus `mel128` is small relative
  to the per-attack variation already present in the frozen baseline (A07–A16
  EER 1.04–6.98%, vs 36.24% for A17 and 18–19% for A18/A19). No significance
  test is implemented; differences are reported descriptively.
- Any condition that changes the audio contract, the colormap, or the flip is out
  of scope (those are E1.2 and later).
- `mel128` failing pixel parity, or any condition failing the re-render
  determinism check, invalidates that condition's row and must be fixed before
  results are reported.

## Artifacts

- Images: `data/derived/e1_1_frontend/<frontend>/{train,dev,eval}/` (gitignored).
- Per-frontend manifests: `manifests/e1_1_frontend/<frontend>_<split>.csv`.
- Probes: `Models/Iteration1/e_1_1/<frontend>/` (checkpoint, `config.json`,
  `train_log.csv`).
- Evaluation: `reports/phase1/e1_1_frontend/runs/<frontend>/<checkpoint-stem>/`.
- Aggregate: `reports/phase1/e1_1_frontend/{summary.csv,per_attack_eer.csv,RESULTS.md}`.

## Limitations stated up front

- Head-only training on a frozen backbone; conclusions are about this transfer
  regime, not about front-ends in general.
- No confidence intervals or significance tests; per-attack spread is the only
  stability signal.
- Dev is a model-selection split (early stopping and threshold selection), so only
  eval is a clean estimate.
- The new image sets live outside `make audit-data`'s current scope; a later data
  audit extension is required before the images are treated as audited.
