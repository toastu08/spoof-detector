# E1.4 shortcut probes (Iteration 1, nested experiment)

**Status:** prepared (2026-11-28); results pending. Report:
[`reports/phase1/e1_4_shortcuts/README.md`](../../reports/phase1/e1_4_shortcuts/README.md).
Pre-registration:
[`reports/phase1/e1_4_shortcuts/PREREGISTRATION.md`](../../reports/phase1/e1_4_shortcuts/PREREGISTRATION.md).
**Experiment:** E1.4 — can the model detect spoofing from silence/noise,
duration, or metadata alone?
**Training manifest:** `manifests/train_seed42.csv`
(SHA-256 `22f98abdf3392e04f7df3632809f81d033d5687773946cd103f5066846ed195e`,
10,080 rows: 2,580 bonafide / 7,500 spoof).
**Preprocessing version:** `mel128-magma-4s-v1` for the control; per-condition
versions are recorded in `reports/phase1/e1_4_shortcuts/export_manifest.json`.

## Architecture

Image conditions use the same frozen-backbone + fresh-head recipe as E1.1/E1.2:

```text
frozen ImageNet ResNet18 backbone (torchvision DEFAULT weights)
  -> Linear(512, 128)
  -> ReLU
  -> Dropout(p=0.5)
  -> Linear(128, 1)
  -> sigmoid (applied outside the model)
```

- Backbone parameters: 11,176,512, all frozen (`requires_grad=False`).
- Head parameters: 65,793, freshly and deterministically re-initialised with
  `torch.manual_seed(42)`.
- BatchNorm: train mode, baseline-faithful; running statistics adapt per
  condition.

Feature probes (`duration_only`, `metadata_system_id`, `metadata_speaker_id`)
use a single `nn.Linear(d, 1)` logistic regression (no backbone).

## Training config

Image conditions:

- Loss: unweighted `BCEWithLogitsLoss`.
- Optimizer / LR: Adam, `1e-3`.
- Batch size: 32.
- Epochs: at most 20; early stopping with patience 5 on full-dev BCE loss.
- Seed: 42 (one run per condition).
- Input: exported PNG -> RGB -> resize 224×224 -> ImageNet mean/std.

Feature probes:

- Loss: unweighted `BCEWithLogitsLoss`.
- Optimizer / LR: Adam, `1e-2`.
- Full-batch, at most 2,000 epochs; early stopping patience 200 on full-dev loss.
- Seed: 42. Features z-score standardised with train statistics.

## Checkpoints

```text
Models/Iteration1/e_1_4/<condition>/seed42/
  resnet18_frozen_<condition>_seed42.pth
  config.json
  train_log.csv

Models/Iteration1/e_1_4/<probe>/seed42/
  feature_probe.pth
  config.json
```

Image checkpoints use the `network.`-prefixed state dict understood by
`model.load_model_from_checkpoint`, which raises unless the missing and
unexpected key lists are empty. Feature-probe checkpoints are plain dicts
(`weights`, `bias`, `feature_mean`, `feature_std`, `speaker_categories`).

## Conditions

| Condition | Keep | Model |
|---|---|---|
| `full` | nothing masked (control) | frozen backbone + fresh head |
| `speech_only` | speech frames only | frozen backbone + fresh head |
| `nonspeech_only` | non-speech (silence/noise/padding) only | frozen backbone + fresh head |
| `duration_only` | clip duration feature | logistic regression |
| `metadata_system_id` | `system_id != "-"` feature | logistic regression |
| `metadata_speaker_id` | one-hot speaker feature | logistic regression |

## Evaluation table

<!-- TODO: fill after `make e1-4` runs. Use the G4 contract below. -->

EER/AUC are threshold-free. Eval is the clean estimate (A07–A19); dev is the
model-selection split (A01–A06). Fixed `p >= 0.5` columns are descriptive.

| Condition | Split | Attack subset | Threshold | Samples (bonafide/spoof) | EER | AUC |
|---|---|---|---|---|---:|---:|
| _pending_ | | | | | | |

## Known limitations

See `reports/phase1/e1_4_shortcuts/PREREGISTRATION.md` "Limitations stated up
front": single seed, energy-heuristic segmentation, feature probes are a
different model class, dev is a model-selection split, one corpus only, and the
E1.4 images are outside `make audit-data`'s current scope.
