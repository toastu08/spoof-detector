# E1.2 rendering-condition probes (Iteration 1, nested experiment)

**Status:** recipe fixed; training and evaluation pending.
**Experiment:** E1.2 — colormap, orientation, dB scaling and PNG encoding
ablation over the fixed `mel128` representation.
**Date prepared:** 2026-11-24.
**Git commit:** recorded in each run's `config.json`; fill the run table after
training.
**Training manifest:** `manifests/train_seed42.csv`
(SHA-256 `22f98abdf3392e04f7df3632809f81d033d5687773946cd103f5066846ed195e`,
10,080 rows: 2,580 bonafide / 7,500 spoof).
**Preprocessing version:** `mel128-magma-4s-v1` for the control; per-condition
versions are recorded in `reports/phase1/e1_2_rendering/export_manifest.json`.

## Architecture

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

## Training config

- Loss: unweighted `BCEWithLogitsLoss`.
- Optimizer / LR: Adam, `1e-3`.
- Batch size: 32.
- Epochs: at most 20; early stopping with patience 5 on full-dev BCE loss.
- Seed: 42 (one run per condition).
- Input: exported PNG -> RGB -> resize 224×224 -> ImageNet mean/std.

## Checkpoints

One directory per condition and seed:

```text
Models/Iteration1/e_1_2/<condition>/seed42/
  resnet18_frozen_<condition>_seed42.pth
  config.json
  train_log.csv
```

Checkpoints use the `network.`-prefixed state dict understood by
`model.load_model_from_checkpoint`, which raises unless the missing and
unexpected key lists are empty.

## Conditions

| Condition | Colormap | Orientation | dB scaling | Encoding |
|---|---|---|---|---|
| `magma_flip` | magma | low frequency bottom | per-clip min-max | RGBA |
| `viridis_flip` | viridis | low frequency bottom | per-clip min-max | RGBA |
| `gray_flip` | gray | low frequency bottom | per-clip min-max | RGBA |
| `rawdb_flip` | gray | low frequency bottom | fixed `[-80, 0]` dB | RGBA |
| `magma_noflip` | magma | low frequency top | per-clip min-max | RGBA |
| `viridis_noflip` | viridis | low frequency top | per-clip min-max | RGBA |
| `gray_noflip` | gray | low frequency top | per-clip min-max | RGBA |
| `magma_flip_rgb` | magma | low frequency bottom | per-clip min-max | RGB |

## Evaluation table

Fill after `make e1-2-evaluate` and `make e1-2-summarize`. Every row must state
split, attack subset, threshold and sample count.

| Condition | Split | Attack subset | Threshold | Samples (bonafide/spoof) | EER | AUC |
|---|---|---|---|---|---:|---:|
| pending | pending | pending | pending | pending | pending | pending |

## Known limitations

- Single seed; 1–3 point differences are within unmeasured training noise.
- Head-only training on a frozen backbone; conclusions do not transfer to
  fine-tuned or waveform models.
- Dev is a model-selection split; eval is the clean estimate.
- `rawdb_flip` removes the per-clip min-max stretch but keeps
  `power_to_db(ref=np.max)`, so it probes contrast normalisation, not absolute
  recording gain.
- The E1.2 images are outside `make audit-data`'s current scope.
