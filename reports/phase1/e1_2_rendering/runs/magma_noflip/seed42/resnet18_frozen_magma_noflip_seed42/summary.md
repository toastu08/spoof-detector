# E1.2 rendering condition: magma_noflip (seed 42): `resnet18_frozen_magma_noflip_seed42.pth`

Experiment: `E1.2`
Generated: `2026-11-27T14:03:48.103563+00:00`

## Metric contract

| Split | Attack subset | Operating thresholds | Samples | Balance |
|---|---|---|---:|---|
| `dev` | A01-A06 | fixed_0.5=0.500000, dev_eer=0.869372 | 24844 | 2548 / 22296 |
| `eval` | A07-A19 | fixed_0.5=0.500000, dev_eer=0.869372 | 71237 | 7355 / 63882 |

Threshold-free metrics: EER and AUC are computed over all spoof clips against all bonafide clips in the split. They are `n/a` when a split has only one class.

## Aggregate metrics

| Split | EER (threshold-free) | AUC (threshold-free) | Brier | ECE (10 bins) |
|---|---:|---:|---:|---:|
| `dev` | 6.01% | 0.9855 | 0.0263 | 0.0052 |
| `eval` | 11.92% | 0.9468 | 0.0815 | 0.0849 |

## Threshold metrics

| Split | Threshold | Value | Accuracy | Bonafide recall | Spoof recall | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|
| `dev` | `fixed_0.5` | 0.500000 | 96.45% | 81.12% | 98.20% | 18.88% | 1.80% |
| `dev` | `dev_eer` | 0.869372 | 94.00% | 94.00% | 94.00% | 6.00% | 6.00% |
| `eval` | `fixed_0.5` | 0.500000 | 88.93% | 85.28% | 89.35% | 14.72% | 10.65% |
| `eval` | `dev_eer` | 0.869372 | 81.90% | 95.43% | 80.34% | 4.57% | 19.66% |

## Per-attack EER

Per-attack EER uses that attack's spoof clips as positives and all bonafide clips in the split as negatives.

| Split | Attack | n_spoof | n_bonafide | EER | AUC |
|---|---|---:|---:|---:|---:|
| `dev` | A01 | 3716 | 2548 | 3.18% | 0.9956 |
| `dev` | A02 | 3716 | 2548 | 0.62% | 0.9997 |
| `dev` | A03 | 3716 | 2548 | 3.65% | 0.9938 |
| `dev` | A04 | 3716 | 2548 | 7.10% | 0.9792 |
| `dev` | A05 | 3716 | 2548 | 5.46% | 0.9879 |
| `dev` | A06 | 3716 | 2548 | 11.14% | 0.9569 |
| `eval` | A07 | 4914 | 7355 | 1.36% | 0.9991 |
| `eval` | A08 | 4914 | 7355 | 2.16% | 0.9974 |
| `eval` | A09 | 4914 | 7355 | 0.59% | 0.9998 |
| `eval` | A10 | 4914 | 7355 | 8.48% | 0.9710 |
| `eval` | A11 | 4914 | 7355 | 2.75% | 0.9961 |
| `eval` | A12 | 4914 | 7355 | 8.59% | 0.9707 |
| `eval` | A13 | 4914 | 7355 | 10.67% | 0.9568 |
| `eval` | A14 | 4914 | 7355 | 5.52% | 0.9865 |
| `eval` | A15 | 4914 | 7355 | 7.27% | 0.9788 |
| `eval` | A16 | 4914 | 7355 | 5.62% | 0.9862 |
| `eval` | A17 | 4914 | 7355 | 35.78% | 0.6978 |
| `eval` | A18 | 4914 | 7355 | 21.00% | 0.8708 |
| `eval` | A19 | 4914 | 7355 | 18.36% | 0.8975 |

## Per-speaker metrics

Per-speaker EER uses same-speaker bonafide clips as negatives. Recalls are at the fixed 0.5 threshold.

| Split | Speaker | n_spoof | n_bonafide | Spoof recall@0.5 | Bonafide recall@0.5 | EER |
|---|---|---:|---:|---:|---:|---:|
| `dev` | LA_0069 | 2484 | 154 | 97.91% | 89.61% | 4.04% |
| `dev` | LA_0070 | 1848 | 140 | 99.03% | 89.29% | 4.28% |
| `dev` | LA_0071 | 1848 | 140 | 98.48% | 80.71% | 6.49% |
| `dev` | LA_0072 | 2484 | 154 | 98.75% | 88.96% | 3.25% |
| `dev` | LA_0073 | 1848 | 140 | 98.48% | 91.43% | 3.57% |
| `dev` | LA_0074 | 2484 | 154 | 96.54% | 86.36% | 7.80% |
| `dev` | LA_0075 | 2484 | 154 | 98.19% | 90.91% | 3.21% |
| `dev` | LA_0076 | 1848 | 140 | 98.48% | 85.71% | 5.07% |
| `dev` | LA_0077 | 2484 | 154 | 98.51% | 79.22% | 4.04% |
| `dev` | LA_0078 | 2484 | 154 | 98.07% | 84.42% | 5.19% |
| `dev` | LA_0099 | 0 | 77 | n/a | 94.81% | n/a |
| `dev` | LA_0100 | 0 | 77 | n/a | 59.74% | n/a |
| `dev` | LA_0101 | 0 | 77 | n/a | 45.45% | n/a |
| `dev` | LA_0102 | 0 | 126 | n/a | 76.98% | n/a |
| `dev` | LA_0103 | 0 | 126 | n/a | 77.78% | n/a |
| `dev` | LA_0104 | 0 | 126 | n/a | 92.06% | n/a |
| `dev` | LA_0105 | 0 | 77 | n/a | 71.43% | n/a |
| `dev` | LA_0106 | 0 | 126 | n/a | 75.40% | n/a |
| `dev` | LA_0107 | 0 | 126 | n/a | 61.11% | n/a |
| `dev` | LA_0108 | 0 | 126 | n/a | 70.63% | n/a |
| `eval` | LA_0001 | 936 | 68 | 91.77% | 85.29% | 9.17% |
| `eval` | LA_0002 | 936 | 68 | 90.92% | 69.12% | 13.29% |
| `eval` | LA_0003 | 936 | 68 | 89.00% | 86.76% | 11.60% |
| `eval` | LA_0004 | 1638 | 146 | 88.03% | 88.36% | 11.65% |
| `eval` | LA_0005 | 936 | 68 | 89.96% | 94.12% | 8.47% |
| `eval` | LA_0006 | 1638 | 146 | 90.05% | 91.10% | 9.59% |
| `eval` | LA_0007 | 936 | 68 | 86.86% | 89.71% | 11.60% |
| `eval` | LA_0008 | 1638 | 146 | 87.79% | 82.19% | 14.40% |
| `eval` | LA_0009 | 1638 | 146 | 91.33% | 86.30% | 9.62% |
| `eval` | LA_0010 | 1638 | 146 | 89.50% | 71.92% | 15.07% |
| `eval` | LA_0011 | 936 | 68 | 89.96% | 94.12% | 8.79% |
| `eval` | LA_0012 | 1638 | 146 | 92.25% | 74.66% | 12.42% |
| `eval` | LA_0013 | 936 | 68 | 89.74% | 88.24% | 10.38% |
| `eval` | LA_0014 | 1638 | 146 | 88.77% | 85.62% | 11.80% |
| `eval` | LA_0015 | 936 | 68 | 89.64% | 91.18% | 9.95% |
| `eval` | LA_0016 | 1638 | 146 | 90.11% | 88.36% | 10.97% |
| `eval` | LA_0017 | 1638 | 146 | 89.87% | 94.52% | 9.53% |
| `eval` | LA_0018 | 936 | 68 | 88.68% | 92.65% | 10.12% |
| `eval` | LA_0019 | 1638 | 146 | 87.06% | 86.99% | 13.01% |
| `eval` | LA_0020 | 1638 | 146 | 91.27% | 93.84% | 7.58% |
| `eval` | LA_0021 | 936 | 68 | 89.32% | 83.82% | 13.24% |
| `eval` | LA_0022 | 1638 | 146 | 90.05% | 80.14% | 13.01% |
| `eval` | LA_0023 | 936 | 68 | 88.25% | 91.18% | 11.54% |
| `eval` | LA_0024 | 1638 | 146 | 89.26% | 84.25% | 12.33% |
| `eval` | LA_0025 | 936 | 68 | 88.14% | 91.18% | 11.54% |
| `eval` | LA_0026 | 1638 | 146 | 87.67% | 93.15% | 10.23% |
| `eval` | LA_0027 | 1638 | 146 | 86.94% | 83.56% | 14.92% |
| `eval` | LA_0028 | 936 | 68 | 91.03% | 95.59% | 7.36% |
| `eval` | LA_0029 | 1638 | 146 | 88.16% | 91.10% | 10.97% |
| `eval` | LA_0030 | 936 | 68 | 90.06% | 77.94% | 14.72% |
| `eval` | LA_0031 | 1638 | 146 | 88.83% | 90.41% | 10.27% |
| `eval` | LA_0032 | 936 | 68 | 87.29% | 72.06% | 17.58% |
| `eval` | LA_0033 | 1638 | 146 | 89.07% | 86.99% | 11.59% |
| `eval` | LA_0034 | 1638 | 146 | 91.03% | 74.66% | 13.69% |
| `eval` | LA_0035 | 1638 | 146 | 89.32% | 94.52% | 8.35% |
| `eval` | LA_0036 | 936 | 68 | 90.28% | 86.76% | 11.49% |
| `eval` | LA_0037 | 1638 | 146 | 90.72% | 90.41% | 9.43% |
| `eval` | LA_0038 | 936 | 68 | 91.88% | 54.41% | 22.03% |
| `eval` | LA_0039 | 1638 | 146 | 86.87% | 94.52% | 9.59% |
| `eval` | LA_0040 | 936 | 68 | 88.68% | 94.12% | 9.11% |
| `eval` | LA_0041 | 1638 | 146 | 89.56% | 76.03% | 15.07% |
| `eval` | LA_0042 | 1638 | 146 | 88.77% | 82.19% | 12.24% |
| `eval` | LA_0043 | 1638 | 146 | 91.03% | 75.34% | 14.24% |
| `eval` | LA_0044 | 936 | 68 | 89.64% | 70.59% | 18.01% |
| `eval` | LA_0045 | 1638 | 146 | 89.13% | 89.04% | 10.97% |
| `eval` | LA_0046 | 936 | 68 | 89.74% | 83.82% | 11.76% |
| `eval` | LA_0047 | 1638 | 146 | 88.83% | 86.30% | 12.21% |
| `eval` | LA_0048 | 936 | 68 | 87.93% | 89.71% | 10.28% |
| `eval` | LA_0049 | 0 | 95 | n/a | 81.05% | n/a |
| `eval` | LA_0050 | 0 | 113 | n/a | 78.76% | n/a |
| `eval` | LA_0051 | 0 | 113 | n/a | 81.42% | n/a |
| `eval` | LA_0052 | 0 | 95 | n/a | 97.89% | n/a |
| `eval` | LA_0053 | 0 | 113 | n/a | 79.65% | n/a |
| `eval` | LA_0054 | 0 | 95 | n/a | 82.11% | n/a |
| `eval` | LA_0055 | 0 | 113 | n/a | 77.88% | n/a |
| `eval` | LA_0056 | 0 | 95 | n/a | 92.63% | n/a |
| `eval` | LA_0057 | 0 | 95 | n/a | 76.84% | n/a |
| `eval` | LA_0058 | 0 | 113 | n/a | 86.73% | n/a |
| `eval` | LA_0059 | 0 | 95 | n/a | 77.89% | n/a |
| `eval` | LA_0060 | 0 | 113 | n/a | 83.19% | n/a |
| `eval` | LA_0061 | 0 | 113 | n/a | 93.81% | n/a |
| `eval` | LA_0063 | 0 | 113 | n/a | 93.81% | n/a |
| `eval` | LA_0064 | 0 | 95 | n/a | 57.89% | n/a |
| `eval` | LA_0065 | 0 | 95 | n/a | 94.74% | n/a |
| `eval` | LA_0066 | 0 | 113 | n/a | 91.15% | n/a |
| `eval` | LA_0067 | 0 | 113 | n/a | 92.04% | n/a |
| `eval` | LA_0068 | 0 | 95 | n/a | 84.21% | n/a |

## Checkpoint

- Path: `Models/Iteration1/e_1_2/magma_noflip/seed42/resnet18_frozen_magma_noflip_seed42.pth`
- SHA-256: `c64a3be108a3ef39026399014b527f327ea9f50fa5037edccca324fcef30a321`
- Parameters: 11242305 total (11176512 backbone, 65793 head)
- State-dict keys: 124 (124 remapped)
- Missing keys: none
- Unexpected keys: none
- Head state SHA-256: `bcbddcd0e948661a789eee009e88483a17f7f33a9467f1de9f0f6c4ec7cf0a3b`
- Final head bias: -0.5637
- Batch-norm steps: [6300]

## Provenance

- **dev**: {'dataset': 'ASVspoof 2019 LA', 'split': 'dev', 'source': 'image', 'data_root': 'data/LA', 'protocol_path': 'data/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt', 'protocol_sha256': '2fedaebf1ba3ae1a9bb7d3ec916804ac237b4eee063a27fa92e6549f03d129c7', 'image_root': 'data/derived/e1_2_rendering/magma_noflip', 'metadata_path': 'data/derived/e1_2_rendering/magma_noflip/dev/metadata_dev.csv', 'metadata_sha256': 'eaa1d4e3d73e3135892076e449fcc6ee5434d6e653bf0fb72d2255e57177c8e9'}
- **eval**: {'dataset': 'ASVspoof 2019 LA', 'split': 'eval', 'source': 'image', 'data_root': 'data/LA', 'protocol_path': 'data/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt', 'protocol_sha256': '84fce576efff092269a0a6ad247d2e9f2f7b4d4e25a4545efb2518da92708036', 'image_root': 'data/derived/e1_2_rendering/magma_noflip', 'metadata_path': 'data/derived/e1_2_rendering/magma_noflip/eval/metadata_eval.csv', 'metadata_sha256': '7e486bf22db024fbc94c000aa4b517d96be18c116ecadcda44857c3060866631'}

## Environment and command

- Git commit: `4fb53d685d23dc9f1040cf4744479bc4bbabd033` (dirty: True)
- Device: `cuda` (NVIDIA GeForce RTX 3050 Laptop GPU)
- Python / torch / torchvision: 3.14.8 / 2.15.0+cu132 / 0.29.2+cu132
- Batch size / workers: 64 / 4

```bash
'/home/toastu/Research/Spoof Detection/.venv/bin/python' -m spoof_detection.evaluate --checkpoint '/home/toastu/Research/Spoof Detection/Models/Iteration1/e_1_2/magma_noflip/seed42/resnet18_frozen_magma_noflip_seed42.pth' --asvspoof dev eval --source image --data-root '/home/toastu/Research/Spoof Detection/data/LA' --image-root '/home/toastu/Research/Spoof Detection/data/derived/e1_2_rendering/magma_noflip' --output-root '/home/toastu/Research/Spoof Detection/reports/phase1/e1_2_rendering/runs/magma_noflip/seed42' --frontend magma_noflip --experiment-id E1.2 --title 'E1.2 rendering condition: magma_noflip (seed 42)' --batch-size 64 --num-workers 4 --device auto
```

## Caveats

- Scores are uncalibrated; Brier and ECE are diagnostics, not evidence of calibrated probabilities.
- EER uses the ASVspoof DET-curve convention (spoof is positive). Per-attack EER reuses all bonafide clips as negatives; per-speaker EER uses same-speaker bonafides.
- Only the dev EER threshold is selected on data. The `dev_eer` threshold in an eval report is transferred from dev and is never selected on eval. Other EER thresholds are descriptive points on the DET curve, not operating thresholds.
- Minimum t-DCF and confidence intervals are not reported in E0.2.
