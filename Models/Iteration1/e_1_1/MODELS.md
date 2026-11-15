# E1.1 front-end probes — model notes

**Experiment:** E1.1 "Is the mel spectrogram necessary?" (ROADMAP phase 1).
**Status:** checkpoints are produced by `scripts/train_e1_1_probes.py`; numbers are
filled in by `reports/phase1/e1_1_frontend/RESULTS.md` after evaluation.
**Date:** 2026-11-16.
**Git commit:** recorded per checkpoint in `<frontend>/config.json` (`git`) at
training time; unknown at authoring time.
**Pre-registration:** `reports/phase1/e1_1_frontend/PREREGISTRATION.md`.

## Training data

- Manifest: `manifests/train_seed42.csv`
  (SHA-256 `22f98abdf3392e04f7df3632809f81d033d5687773946cd103f5066846ed195e`).
- 10,080 clips: 2,580 bonafide / 7,500 spoof. Dev for early stopping: 24,844
  clips (2,548 / 22,296). Eval (clean estimate): 71,237 clips (7,355 / 63,882).
- Images: `data/derived/e1_1_frontend/<frontend>/{train,dev,eval}/`, regenerated
  from `data/LA` with the front-end renderer. `mel128` is pixel-identical to
  `data/dataset_img`.
- Per-frontend manifests: `manifests/e1_1_frontend/<frontend>_<split>.csv`.

## Preprocessing versions

| Front-end | Version string |
|---|---|
| mel64 | `mel64-magma-4s-v1` |
| mel128 | `mel128-magma-4s-v1` (existing contract) |
| mel256 | `mel256-magma-4s-v1` |
| linear_stft | `linear1025-magma-4s-v1` |
| mfcc40 | `mfcc40-magma-4s-v1` |
| randproj128 | `randproj128-magma-4s-v1` |

All share: 16 kHz, 4 s / 64,000 samples, min-max normalisation, `magma` colormap,
`origin='lower'` flip, resize 224×224, ImageNet mean/std.

## Architecture

| Component | Details |
|---|---|
| Backbone | `torchvision` ResNet18 (`weights="DEFAULT"`), frozen |
| Frozen layers | all backbone parameters `requires_grad=False`; BatchNorm runs in train mode (statistics update) |
| Head | `Linear(512,128) → ReLU → Dropout(0.5) → Linear(128,1)` |
| Head init | `torch.manual_seed(42)` + `reset_parameters()`, identical across conditions |
| Output | one logit; sigmoid outside the model; `p >= 0.5` = spoof |

Parameter counts are recorded per checkpoint by the strict loader (expected
11,242,305 total, 65,793 head; see `Models/Iteration1/MODELS.md`).

## Training recipe

- Loss: unweighted `BCEWithLogitsLoss`.
- Optimizer: Adam, `lr=1e-3`; batch size 32.
- Max 20 epochs; early stopping patience 5 on full-dev BCE loss.
- Seed: 42 (Python/NumPy/torch, loader generator, worker seeding).
- No augmentation; no class weights.
- Baseline-faithful deviations only: fixed seed and deterministic shared head
  init (the original notebook recorded no seed).

## Checkpoints

Expected paths, one directory per condition:

```
Models/Iteration1/e_1_1/<frontend>/resnet18_frozen_<frontend>.pth
Models/Iteration1/e_1_1/<frontend>/config.json
Models/Iteration1/e_1_1/<frontend>/train_log.csv
```

Checkpoints use the `network.` key prefix and load strictly through
`model.load_model_from_checkpoint` (missing and unexpected keys must both be
empty).

## Evaluation table

Reported in `reports/phase1/e1_1_frontend/RESULTS.md` and `summary.csv`. Contract
per row: split, attack subset, threshold, sample count.

| Front-end | Split | Attack subset | Threshold | Samples (bonafide/spoof) | EER | AUC |
|---|---|---|---:|---:|---:|---:|
| mel64 | dev | A01–A06 | threshold-free | 24,844 (2,548 / 22,296) | pending | pending |
| mel64 | eval | A07–A19 | threshold-free | 71,237 (7,355 / 63,882) | pending | pending |
| mel128 | dev | A01–A06 | threshold-free | 24,844 (2,548 / 22,296) | pending | pending |
| mel128 | eval | A07–A19 | threshold-free | 71,237 (7,355 / 63,882) | pending | pending |
| mel256 | dev | A01–A06 | threshold-free | 24,844 (2,548 / 22,296) | pending | pending |
| mel256 | eval | A07–A19 | threshold-free | 71,237 (7,355 / 63,882) | pending | pending |
| linear_stft | dev | A01–A06 | threshold-free | 24,844 (2,548 / 22,296) | pending | pending |
| linear_stft | eval | A07–A19 | threshold-free | 71,237 (7,355 / 63,882) | pending | pending |
| mfcc40 | dev | A01–A06 | threshold-free | 24,844 (2,548 / 22,296) | pending | pending |
| mfcc40 | eval | A07–A19 | threshold-free | 71,237 (7,355 / 63,882) | pending | pending |
| randproj128 | dev | A01–A06 | threshold-free | 24,844 (2,548 / 22,296) | pending | pending |
| randproj128 | eval | A07–A19 | threshold-free | 71,237 (7,355 / 63,882) | pending | pending |

`mel128` should reproduce the frozen E0.3 unweighted baseline up to training
nondeterminism: dev EER 6.16% / AUC 0.9860, eval EER 11.50% / AUC 0.9527. A large
gap would indicate a pipeline bug rather than a front-end effect.

## Known limitations and failure cases

- Head-only training on a frozen backbone; conclusions are about this regime.
- BatchNorm running statistics are updated from each front-end's images, so the
  backbone is not bit-identical to E0.3 after training. Only the initialisation is.
- No confidence intervals or significance tests; per-attack spread is descriptive.
- Dev is a model-selection split; eval is the clean estimate.
- The E1.1 images are outside `make audit-data`'s current scope.
