# E1.1 front-end probes — model notes

**Experiment:** E1.1 "Is the mel spectrogram necessary?" (ROADMAP phase 1).
**Status: complete (2026-11-23).** Six probes trained and evaluated; see
`reports/phase1/e1_1_frontend/README.md` for verdicts and interpretation.
**Date:** trained 2026-11-20; notes completed 2026-11-23.
**Git commit at training time:** `9881dbf6a5cafe8fe566f035d32b86b00a8cba21`,
working tree dirty (recorded as `"dirty": true` in every `config.json`). The
front-end and training code was committed afterwards in `477ae71`.
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

All six probes are written, one directory per condition (the `.pth` files are
gitignored; `config.json` and `train_log.csv` are tracked):

```
Models/Iteration1/e_1_1/<frontend>/resnet18_frozen_<frontend>.pth
Models/Iteration1/e_1_1/<frontend>/config.json
Models/Iteration1/e_1_1/<frontend>/train_log.csv
```

Checkpoints use the `network.` key prefix and load strictly through
`model.load_model_from_checkpoint` (missing and unexpected keys must both be
empty).

## Evaluation table

Reported in full in `reports/phase1/e1_1_frontend/README.md`, with generated
tables in `RESULTS.md`, `summary.csv` and `per_attack_eer.csv`. Contract per row:
split, attack subset, threshold, sample count. EER/AUC are threshold-free; the
`0.5` columns are a descriptive fixed operating point.

| Front-end | Split | Attack subset | Threshold | Samples (bonafide/spoof) | EER | AUC | Checkpoint SHA-256 (first 12) |
|---|---|---|---:|---:|---:|---:|---|
| mel64 | dev | A01–A06 | threshold-free | 24,844 (2,548 / 22,296) | 10.05% | 0.9669 | `30028f1aa47e` |
| mel64 | eval | A07–A19 | threshold-free | 71,237 (7,355 / 63,882) | 13.01% | 0.9434 | `30028f1aa47e` |
| mel128 | dev | A01–A06 | threshold-free | 24,844 (2,548 / 22,296) | 5.68% | 0.9876 | `c081a699bfcd` |
| mel128 | eval | A07–A19 | threshold-free | 71,237 (7,355 / 63,882) | 11.41% | 0.9536 | `c081a699bfcd` |
| mel256 | dev | A01–A06 | threshold-free | 24,844 (2,548 / 22,296) | 4.36% | 0.9919 | `2d96030622a9` |
| mel256 | eval | A07–A19 | threshold-free | 71,237 (7,355 / 63,882) | 11.31% | 0.9525 | `2d96030622a9` |
| linear_stft | dev | A01–A06 | threshold-free | 24,844 (2,548 / 22,296) | 5.69% | 0.9861 | `a9aeb0c6c406` |
| linear_stft | eval | A07–A19 | threshold-free | 71,237 (7,355 / 63,882) | 13.98% | 0.9371 | `a9aeb0c6c406` |
| mfcc40 | dev | A01–A06 | threshold-free | 24,844 (2,548 / 22,296) | 14.49% | 0.9327 | `4023e84c804b` |
| mfcc40 | eval | A07–A19 | threshold-free | 71,237 (7,355 / 63,882) | 18.56% | 0.8963 | `4023e84c804b` |
| randproj128 | dev | A01–A06 | threshold-free | 24,844 (2,548 / 22,296) | 19.71% | 0.8847 | `54cd0d8f707c` |
| randproj128 | eval | A07–A19 | threshold-free | 71,237 (7,355 / 63,882) | 17.04% | 0.9064 | `54cd0d8f707c` |

Training outcome per condition (from `train_log.csv`; selection is on full-dev
BCE loss, not EER):

| Front-end | Best epoch | Epochs run | Early stop triggered |
|---|---:|---:|---|
| mel64 | 18 | 20 | no (20-epoch cap) |
| mel128 | 20 | 20 | no (20-epoch cap) |
| mel256 | 15 | 20 | no (20-epoch cap) |
| linear_stft | 19 | 20 | no (20-epoch cap) |
| mfcc40 | 10 | 15 | yes |
| randproj128 | 11 | 16 | yes |

`mel128` reproduces the frozen E0.3 unweighted baseline (dev EER 6.16% / AUC
0.9860, eval EER 11.50% / AUC 0.9527) to within 0.48 dev and 0.09 eval EER points
with a freshly initialised head, confirming the pipeline is faithful.

## Known limitations and failure cases

- Head-only training on a frozen backbone; conclusions are about this regime.
- **Single seed per condition.** The 0.10-point `mel128`/`mel256` eval gap is a
  tie; the 1.6–2.6-point gaps exceed eval sampling error but not unmeasured
  training-seed variance. Only the ≥5-point gaps should be treated as robust.
- BatchNorm running statistics are updated from each front-end's images, so the
  backbone is not bit-identical to E0.3 after training. Only the initialisation is.
- No confidence intervals or significance tests; per-attack spread is descriptive.
- Dev is a model-selection split; eval is the clean estimate.
- The E1.1 images are outside `make audit-data`'s current scope.
