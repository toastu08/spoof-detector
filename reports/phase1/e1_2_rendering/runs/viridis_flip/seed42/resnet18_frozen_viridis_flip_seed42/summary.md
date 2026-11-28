# E1.2 rendering condition: viridis_flip (seed 42): `resnet18_frozen_viridis_flip_seed42.pth`

Experiment: `E1.2`
Generated: `2026-11-27T13:57:10.939165+00:00`

## Metric contract

| Split | Attack subset | Operating thresholds | Samples | Balance |
|---|---|---|---:|---|
| `dev` | A01-A06 | fixed_0.5=0.500000, dev_eer=0.854267 | 24844 | 2548 / 22296 |
| `eval` | A07-A19 | fixed_0.5=0.500000, dev_eer=0.854267 | 71237 | 7355 / 63882 |

Threshold-free metrics: EER and AUC are computed over all spoof clips against all bonafide clips in the split. They are `n/a` when a split has only one class.

## Aggregate metrics

| Split | EER (threshold-free) | AUC (threshold-free) | Brier | ECE (10 bins) |
|---|---:|---:|---:|---:|
| `dev` | 5.81% | 0.9875 | 0.0243 | 0.0062 |
| `eval` | 12.94% | 0.9392 | 0.0803 | 0.0805 |

## Threshold metrics

| Split | Threshold | Value | Accuracy | Bonafide recall | Spoof recall | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|
| `dev` | `fixed_0.5` | 0.500000 | 96.82% | 82.42% | 98.47% | 17.58% | 1.53% |
| `dev` | `dev_eer` | 0.854267 | 94.20% | 94.19% | 94.20% | 5.81% | 5.80% |
| `eval` | `fixed_0.5` | 0.500000 | 88.65% | 82.05% | 89.41% | 17.95% | 10.59% |
| `eval` | `dev_eer` | 0.854267 | 81.31% | 94.36% | 79.81% | 5.64% | 20.19% |

## Per-attack EER

Per-attack EER uses that attack's spoof clips as positives and all bonafide clips in the split as negatives.

| Split | Attack | n_spoof | n_bonafide | EER | AUC |
|---|---|---:|---:|---:|---:|
| `dev` | A01 | 3716 | 2548 | 1.68% | 0.9984 |
| `dev` | A02 | 3716 | 2548 | 2.63% | 0.9975 |
| `dev` | A03 | 3716 | 2548 | 2.39% | 0.9973 |
| `dev` | A04 | 3716 | 2548 | 4.90% | 0.9900 |
| `dev` | A05 | 3716 | 2548 | 7.50% | 0.9792 |
| `dev` | A06 | 3716 | 2548 | 10.31% | 0.9626 |
| `eval` | A07 | 4914 | 7355 | 2.08% | 0.9975 |
| `eval` | A08 | 4914 | 7355 | 2.16% | 0.9975 |
| `eval` | A09 | 4914 | 7355 | 1.22% | 0.9992 |
| `eval` | A10 | 4914 | 7355 | 11.34% | 0.9469 |
| `eval` | A11 | 4914 | 7355 | 2.89% | 0.9946 |
| `eval` | A12 | 4914 | 7355 | 9.49% | 0.9609 |
| `eval` | A13 | 4914 | 7355 | 12.55% | 0.9373 |
| `eval` | A14 | 4914 | 7355 | 5.41% | 0.9868 |
| `eval` | A15 | 4914 | 7355 | 9.32% | 0.9648 |
| `eval` | A16 | 4914 | 7355 | 4.52% | 0.9900 |
| `eval` | A17 | 4914 | 7355 | 38.95% | 0.6519 |
| `eval` | A18 | 4914 | 7355 | 19.50% | 0.8851 |
| `eval` | A19 | 4914 | 7355 | 18.19% | 0.8970 |

## Per-speaker metrics

Per-speaker EER uses same-speaker bonafide clips as negatives. Recalls are at the fixed 0.5 threshold.

| Split | Speaker | n_spoof | n_bonafide | Spoof recall@0.5 | Bonafide recall@0.5 | EER |
|---|---|---:|---:|---:|---:|---:|
| `dev` | LA_0069 | 2484 | 154 | 98.79% | 83.77% | 4.49% |
| `dev` | LA_0070 | 1848 | 140 | 98.48% | 81.43% | 6.41% |
| `dev` | LA_0071 | 1848 | 140 | 98.48% | 67.86% | 7.14% |
| `dev` | LA_0072 | 2484 | 154 | 98.35% | 81.17% | 5.98% |
| `dev` | LA_0073 | 1848 | 140 | 98.38% | 85.71% | 5.73% |
| `dev` | LA_0074 | 2484 | 154 | 97.14% | 88.31% | 5.07% |
| `dev` | LA_0075 | 2484 | 154 | 98.51% | 91.56% | 3.98% |
| `dev` | LA_0076 | 1848 | 140 | 98.76% | 90.71% | 3.57% |
| `dev` | LA_0077 | 2484 | 154 | 98.99% | 87.01% | 3.39% |
| `dev` | LA_0078 | 2484 | 154 | 98.87% | 87.66% | 3.90% |
| `dev` | LA_0099 | 0 | 77 | n/a | 84.42% | n/a |
| `dev` | LA_0100 | 0 | 77 | n/a | 72.73% | n/a |
| `dev` | LA_0101 | 0 | 77 | n/a | 61.04% | n/a |
| `dev` | LA_0102 | 0 | 126 | n/a | 76.19% | n/a |
| `dev` | LA_0103 | 0 | 126 | n/a | 73.81% | n/a |
| `dev` | LA_0104 | 0 | 126 | n/a | 84.13% | n/a |
| `dev` | LA_0105 | 0 | 77 | n/a | 81.82% | n/a |
| `dev` | LA_0106 | 0 | 126 | n/a | 89.68% | n/a |
| `dev` | LA_0107 | 0 | 126 | n/a | 76.19% | n/a |
| `dev` | LA_0108 | 0 | 126 | n/a | 86.51% | n/a |
| `eval` | LA_0001 | 936 | 68 | 93.38% | 64.71% | 16.15% |
| `eval` | LA_0002 | 936 | 68 | 92.52% | 57.35% | 19.12% |
| `eval` | LA_0003 | 936 | 68 | 92.09% | 80.88% | 13.24% |
| `eval` | LA_0004 | 1638 | 146 | 89.74% | 90.41% | 9.59% |
| `eval` | LA_0005 | 936 | 68 | 91.67% | 82.35% | 11.76% |
| `eval` | LA_0006 | 1638 | 146 | 87.48% | 91.10% | 10.94% |
| `eval` | LA_0007 | 936 | 68 | 90.38% | 79.41% | 12.08% |
| `eval` | LA_0008 | 1638 | 146 | 87.97% | 86.99% | 12.27% |
| `eval` | LA_0009 | 1638 | 146 | 89.74% | 79.45% | 13.69% |
| `eval` | LA_0010 | 1638 | 146 | 87.00% | 75.34% | 19.17% |
| `eval` | LA_0011 | 936 | 68 | 88.46% | 86.76% | 11.87% |
| `eval` | LA_0012 | 1638 | 146 | 91.82% | 67.12% | 15.75% |
| `eval` | LA_0013 | 936 | 68 | 89.74% | 83.82% | 10.65% |
| `eval` | LA_0014 | 1638 | 146 | 89.44% | 70.55% | 18.50% |
| `eval` | LA_0015 | 936 | 68 | 88.78% | 89.71% | 11.44% |
| `eval` | LA_0016 | 1638 | 146 | 89.87% | 89.04% | 10.27% |
| `eval` | LA_0017 | 1638 | 146 | 89.44% | 77.40% | 14.40% |
| `eval` | LA_0018 | 936 | 68 | 91.13% | 83.82% | 11.54% |
| `eval` | LA_0019 | 1638 | 146 | 86.81% | 82.88% | 15.07% |
| `eval` | LA_0020 | 1638 | 146 | 91.21% | 82.19% | 12.33% |
| `eval` | LA_0021 | 936 | 68 | 89.00% | 83.82% | 13.24% |
| `eval` | LA_0022 | 1638 | 146 | 89.13% | 72.60% | 14.46% |
| `eval` | LA_0023 | 936 | 68 | 91.56% | 76.47% | 14.78% |
| `eval` | LA_0024 | 1638 | 146 | 89.74% | 86.30% | 10.97% |
| `eval` | LA_0025 | 936 | 68 | 86.43% | 86.76% | 13.24% |
| `eval` | LA_0026 | 1638 | 146 | 87.55% | 91.78% | 10.91% |
| `eval` | LA_0027 | 1638 | 146 | 84.43% | 88.36% | 13.78% |
| `eval` | LA_0028 | 936 | 68 | 90.28% | 94.12% | 7.36% |
| `eval` | LA_0029 | 1638 | 146 | 90.05% | 92.47% | 8.91% |
| `eval` | LA_0030 | 936 | 68 | 92.09% | 73.53% | 13.24% |
| `eval` | LA_0031 | 1638 | 146 | 89.32% | 86.30% | 11.65% |
| `eval` | LA_0032 | 936 | 68 | 86.97% | 77.94% | 14.78% |
| `eval` | LA_0033 | 1638 | 146 | 89.87% | 82.88% | 12.45% |
| `eval` | LA_0034 | 1638 | 146 | 90.90% | 87.67% | 9.65% |
| `eval` | LA_0035 | 1638 | 146 | 88.28% | 89.04% | 11.13% |
| `eval` | LA_0036 | 936 | 68 | 88.35% | 94.12% | 10.28% |
| `eval` | LA_0037 | 1638 | 146 | 90.35% | 88.36% | 10.27% |
| `eval` | LA_0038 | 936 | 68 | 90.71% | 39.71% | 23.52% |
| `eval` | LA_0039 | 1638 | 146 | 85.53% | 93.84% | 10.14% |
| `eval` | LA_0040 | 936 | 68 | 92.09% | 82.35% | 11.81% |
| `eval` | LA_0041 | 1638 | 146 | 89.07% | 79.45% | 13.01% |
| `eval` | LA_0042 | 1638 | 146 | 88.95% | 74.66% | 13.72% |
| `eval` | LA_0043 | 1638 | 146 | 89.26% | 77.40% | 14.55% |
| `eval` | LA_0044 | 936 | 68 | 91.13% | 85.29% | 10.28% |
| `eval` | LA_0045 | 1638 | 146 | 90.35% | 85.62% | 10.97% |
| `eval` | LA_0046 | 936 | 68 | 91.77% | 72.06% | 14.35% |
| `eval` | LA_0047 | 1638 | 146 | 89.32% | 85.62% | 12.39% |
| `eval` | LA_0048 | 936 | 68 | 89.32% | 89.71% | 10.28% |
| `eval` | LA_0049 | 0 | 95 | n/a | 86.32% | n/a |
| `eval` | LA_0050 | 0 | 113 | n/a | 74.34% | n/a |
| `eval` | LA_0051 | 0 | 113 | n/a | 79.65% | n/a |
| `eval` | LA_0052 | 0 | 95 | n/a | 87.37% | n/a |
| `eval` | LA_0053 | 0 | 113 | n/a | 87.61% | n/a |
| `eval` | LA_0054 | 0 | 95 | n/a | 73.68% | n/a |
| `eval` | LA_0055 | 0 | 113 | n/a | 84.96% | n/a |
| `eval` | LA_0056 | 0 | 95 | n/a | 87.37% | n/a |
| `eval` | LA_0057 | 0 | 95 | n/a | 70.53% | n/a |
| `eval` | LA_0058 | 0 | 113 | n/a | 83.19% | n/a |
| `eval` | LA_0059 | 0 | 95 | n/a | 75.79% | n/a |
| `eval` | LA_0060 | 0 | 113 | n/a | 76.11% | n/a |
| `eval` | LA_0061 | 0 | 113 | n/a | 92.04% | n/a |
| `eval` | LA_0063 | 0 | 113 | n/a | 91.15% | n/a |
| `eval` | LA_0064 | 0 | 95 | n/a | 67.37% | n/a |
| `eval` | LA_0065 | 0 | 95 | n/a | 85.26% | n/a |
| `eval` | LA_0066 | 0 | 113 | n/a | 86.73% | n/a |
| `eval` | LA_0067 | 0 | 113 | n/a | 83.19% | n/a |
| `eval` | LA_0068 | 0 | 95 | n/a | 64.21% | n/a |

## Checkpoint

- Path: `Models/Iteration1/e_1_2/viridis_flip/seed42/resnet18_frozen_viridis_flip_seed42.pth`
- SHA-256: `4c8889effba341f9a78dd5f3059e2dc163be5d7735b6ff6b11e4b1fd6f22e0ed`
- Parameters: 11242305 total (11176512 backbone, 65793 head)
- State-dict keys: 124 (124 remapped)
- Missing keys: none
- Unexpected keys: none
- Head state SHA-256: `969c89929b631722985244160e385e1da2c58136cd3a176f24e0fac7755b0d0c`
- Final head bias: -0.3734
- Batch-norm steps: [6300]

## Provenance

- **dev**: {'dataset': 'ASVspoof 2019 LA', 'split': 'dev', 'source': 'image', 'data_root': 'data/LA', 'protocol_path': 'data/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt', 'protocol_sha256': '2fedaebf1ba3ae1a9bb7d3ec916804ac237b4eee063a27fa92e6549f03d129c7', 'image_root': 'data/derived/e1_2_rendering/viridis_flip', 'metadata_path': 'data/derived/e1_2_rendering/viridis_flip/dev/metadata_dev.csv', 'metadata_sha256': 'fcdbe57ee0d03af96875c86dd05045c868e5076bef8de69dbf90ccf42872ece2'}
- **eval**: {'dataset': 'ASVspoof 2019 LA', 'split': 'eval', 'source': 'image', 'data_root': 'data/LA', 'protocol_path': 'data/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt', 'protocol_sha256': '84fce576efff092269a0a6ad247d2e9f2f7b4d4e25a4545efb2518da92708036', 'image_root': 'data/derived/e1_2_rendering/viridis_flip', 'metadata_path': 'data/derived/e1_2_rendering/viridis_flip/eval/metadata_eval.csv', 'metadata_sha256': 'c421be44f280bef8f3c0017a11ac27e13f8f9beecb8d7f987af01c3520faf593'}

## Environment and command

- Git commit: `4fb53d685d23dc9f1040cf4744479bc4bbabd033` (dirty: True)
- Device: `cuda` (NVIDIA GeForce RTX 3050 Laptop GPU)
- Python / torch / torchvision: 3.14.8 / 2.15.0+cu132 / 0.29.2+cu132
- Batch size / workers: 64 / 4

```bash
'/home/toastu/Research/Spoof Detection/.venv/bin/python' -m spoof_detection.evaluate --checkpoint '/home/toastu/Research/Spoof Detection/Models/Iteration1/e_1_2/viridis_flip/seed42/resnet18_frozen_viridis_flip_seed42.pth' --asvspoof dev eval --source image --data-root '/home/toastu/Research/Spoof Detection/data/LA' --image-root '/home/toastu/Research/Spoof Detection/data/derived/e1_2_rendering/viridis_flip' --output-root '/home/toastu/Research/Spoof Detection/reports/phase1/e1_2_rendering/runs/viridis_flip/seed42' --frontend viridis_flip --experiment-id E1.2 --title 'E1.2 rendering condition: viridis_flip (seed 42)' --batch-size 64 --num-workers 4 --device auto
```

## Caveats

- Scores are uncalibrated; Brier and ECE are diagnostics, not evidence of calibrated probabilities.
- EER uses the ASVspoof DET-curve convention (spoof is positive). Per-attack EER reuses all bonafide clips as negatives; per-speaker EER uses same-speaker bonafides.
- Only the dev EER threshold is selected on data. The `dev_eer` threshold in an eval report is transferred from dev and is never selected on eval. Other EER thresholds are descriptive points on the DET curve, not operating thresholds.
- Minimum t-DCF and confidence intervals are not reported in E0.2.
