# E1.1 front-end probe: linear_stft: `resnet18_frozen_linear_stft.pth`

Experiment: `E1.1`
Generated: `2026-11-20T16:01:16.287084+00:00`

## Metric contract

| Split | Attack subset | Operating thresholds | Samples | Balance |
|---|---|---|---:|---|
| `dev` | A01-A06 | fixed_0.5=0.500000, dev_eer=0.894476 | 24844 | 2548 / 22296 |
| `eval` | A07-A19 | fixed_0.5=0.500000, dev_eer=0.894476 | 71237 | 7355 / 63882 |

Threshold-free metrics: EER and AUC are computed over all spoof clips against all bonafide clips in the split. They are `n/a` when a split has only one class.

## Aggregate metrics

| Split | EER (threshold-free) | AUC (threshold-free) | Brier | ECE (10 bins) |
|---|---:|---:|---:|---:|
| `dev` | 5.69% | 0.9861 | 0.0238 | 0.0043 |
| `eval` | 13.98% | 0.9371 | 0.0914 | 0.0998 |

## Threshold metrics

| Split | Threshold | Value | Accuracy | Bonafide recall | Spoof recall | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|
| `dev` | `fixed_0.5` | 0.500000 | 96.90% | 79.95% | 98.84% | 20.05% | 1.16% |
| `dev` | `dev_eer` | 0.894476 | 94.31% | 94.27% | 94.32% | 5.73% | 5.68% |
| `eval` | `fixed_0.5` | 0.500000 | 87.45% | 83.06% | 87.96% | 16.94% | 12.04% |
| `eval` | `dev_eer` | 0.894476 | 77.28% | 94.81% | 75.27% | 5.19% | 24.73% |

## Per-attack EER

Per-attack EER uses that attack's spoof clips as positives and all bonafide clips in the split as negatives.

| Split | Attack | n_spoof | n_bonafide | EER | AUC |
|---|---|---:|---:|---:|---:|
| `dev` | A01 | 3716 | 2548 | 1.37% | 0.9991 |
| `dev` | A02 | 3716 | 2548 | 4.28% | 0.9925 |
| `dev` | A03 | 3716 | 2548 | 6.05% | 0.9841 |
| `dev` | A04 | 3716 | 2548 | 4.31% | 0.9923 |
| `dev` | A05 | 3716 | 2548 | 11.11% | 0.9565 |
| `dev` | A06 | 3716 | 2548 | 4.28% | 0.9918 |
| `eval` | A07 | 4914 | 7355 | 0.98% | 0.9995 |
| `eval` | A08 | 4914 | 7355 | 2.01% | 0.9983 |
| `eval` | A09 | 4914 | 7355 | 2.67% | 0.9964 |
| `eval` | A10 | 4914 | 7355 | 13.82% | 0.9377 |
| `eval` | A11 | 4914 | 7355 | 7.02% | 0.9802 |
| `eval` | A12 | 4914 | 7355 | 16.81% | 0.9132 |
| `eval` | A13 | 4914 | 7355 | 16.22% | 0.9194 |
| `eval` | A14 | 4914 | 7355 | 9.77% | 0.9657 |
| `eval` | A15 | 4914 | 7355 | 12.96% | 0.9439 |
| `eval` | A16 | 4914 | 7355 | 3.25% | 0.9951 |
| `eval` | A17 | 4914 | 7355 | 35.57% | 0.6933 |
| `eval` | A18 | 4914 | 7355 | 22.00% | 0.8566 |
| `eval` | A19 | 4914 | 7355 | 6.94% | 0.9826 |

## Per-speaker metrics

Per-speaker EER uses same-speaker bonafide clips as negatives. Recalls are at the fixed 0.5 threshold.

| Split | Speaker | n_spoof | n_bonafide | Spoof recall@0.5 | Bonafide recall@0.5 | EER |
|---|---|---:|---:|---:|---:|---:|
| `dev` | LA_0069 | 2484 | 154 | 99.07% | 75.32% | 3.35% |
| `dev` | LA_0070 | 1848 | 140 | 98.59% | 77.86% | 7.14% |
| `dev` | LA_0071 | 1848 | 140 | 98.86% | 70.00% | 7.85% |
| `dev` | LA_0072 | 2484 | 154 | 99.07% | 85.71% | 5.72% |
| `dev` | LA_0073 | 1848 | 140 | 98.97% | 83.57% | 2.94% |
| `dev` | LA_0074 | 2484 | 154 | 97.58% | 94.16% | 3.90% |
| `dev` | LA_0075 | 2484 | 154 | 99.11% | 71.43% | 7.66% |
| `dev` | LA_0076 | 1848 | 140 | 98.70% | 84.29% | 4.99% |
| `dev` | LA_0077 | 2484 | 154 | 99.28% | 94.16% | 1.84% |
| `dev` | LA_0078 | 2484 | 154 | 99.07% | 92.86% | 3.09% |
| `dev` | LA_0099 | 0 | 77 | n/a | 79.22% | n/a |
| `dev` | LA_0100 | 0 | 77 | n/a | 76.62% | n/a |
| `dev` | LA_0101 | 0 | 77 | n/a | 66.23% | n/a |
| `dev` | LA_0102 | 0 | 126 | n/a | 53.97% | n/a |
| `dev` | LA_0103 | 0 | 126 | n/a | 74.60% | n/a |
| `dev` | LA_0104 | 0 | 126 | n/a | 63.49% | n/a |
| `dev` | LA_0105 | 0 | 77 | n/a | 79.22% | n/a |
| `dev` | LA_0106 | 0 | 126 | n/a | 83.33% | n/a |
| `dev` | LA_0107 | 0 | 126 | n/a | 84.92% | n/a |
| `dev` | LA_0108 | 0 | 126 | n/a | 93.65% | n/a |
| `eval` | LA_0001 | 936 | 68 | 88.35% | 79.41% | 14.35% |
| `eval` | LA_0002 | 936 | 68 | 90.06% | 58.82% | 19.23% |
| `eval` | LA_0003 | 936 | 68 | 91.45% | 76.47% | 15.94% |
| `eval` | LA_0004 | 1638 | 146 | 87.00% | 88.36% | 12.42% |
| `eval` | LA_0005 | 936 | 68 | 89.96% | 88.24% | 10.38% |
| `eval` | LA_0006 | 1638 | 146 | 88.22% | 82.19% | 13.72% |
| `eval` | LA_0007 | 936 | 68 | 90.28% | 79.41% | 13.24% |
| `eval` | LA_0008 | 1638 | 146 | 86.75% | 86.99% | 13.07% |
| `eval` | LA_0009 | 1638 | 146 | 89.07% | 69.18% | 19.17% |
| `eval` | LA_0010 | 1638 | 146 | 90.72% | 69.18% | 14.55% |
| `eval` | LA_0011 | 936 | 68 | 86.11% | 88.24% | 13.46% |
| `eval` | LA_0012 | 1638 | 146 | 87.79% | 81.51% | 14.30% |
| `eval` | LA_0013 | 936 | 68 | 87.61% | 72.06% | 16.31% |
| `eval` | LA_0014 | 1638 | 146 | 90.48% | 77.40% | 15.07% |
| `eval` | LA_0015 | 936 | 68 | 89.42% | 83.82% | 11.76% |
| `eval` | LA_0016 | 1638 | 146 | 90.66% | 88.36% | 10.11% |
| `eval` | LA_0017 | 1638 | 146 | 89.80% | 69.86% | 17.02% |
| `eval` | LA_0018 | 936 | 68 | 90.17% | 89.71% | 10.22% |
| `eval` | LA_0019 | 1638 | 146 | 86.26% | 72.60% | 18.47% |
| `eval` | LA_0020 | 1638 | 146 | 90.96% | 88.36% | 9.68% |
| `eval` | LA_0021 | 936 | 68 | 84.51% | 88.24% | 13.03% |
| `eval` | LA_0022 | 1638 | 146 | 88.46% | 82.88% | 13.66% |
| `eval` | LA_0023 | 936 | 68 | 88.03% | 80.88% | 14.72% |
| `eval` | LA_0024 | 1638 | 146 | 88.03% | 85.62% | 13.01% |
| `eval` | LA_0025 | 936 | 68 | 84.72% | 95.59% | 8.85% |
| `eval` | LA_0026 | 1638 | 146 | 89.87% | 85.62% | 10.97% |
| `eval` | LA_0027 | 1638 | 146 | 85.04% | 91.10% | 12.27% |
| `eval` | LA_0028 | 936 | 68 | 86.43% | 80.88% | 14.72% |
| `eval` | LA_0029 | 1638 | 146 | 87.18% | 96.58% | 8.97% |
| `eval` | LA_0030 | 936 | 68 | 90.60% | 80.88% | 14.72% |
| `eval` | LA_0031 | 1638 | 146 | 86.69% | 84.93% | 14.43% |
| `eval` | LA_0032 | 936 | 68 | 81.09% | 88.24% | 14.72% |
| `eval` | LA_0033 | 1638 | 146 | 89.38% | 78.77% | 15.07% |
| `eval` | LA_0034 | 1638 | 146 | 90.54% | 90.41% | 9.59% |
| `eval` | LA_0035 | 1638 | 146 | 86.26% | 84.93% | 14.55% |
| `eval` | LA_0036 | 936 | 68 | 85.90% | 88.24% | 13.24% |
| `eval` | LA_0037 | 1638 | 146 | 90.72% | 88.36% | 9.59% |
| `eval` | LA_0038 | 936 | 68 | 88.25% | 38.24% | 33.79% |
| `eval` | LA_0039 | 1638 | 146 | 85.47% | 89.04% | 13.63% |
| `eval` | LA_0040 | 936 | 68 | 83.97% | 94.12% | 10.33% |
| `eval` | LA_0041 | 1638 | 146 | 84.80% | 78.77% | 19.85% |
| `eval` | LA_0042 | 1638 | 146 | 87.18% | 82.88% | 15.63% |
| `eval` | LA_0043 | 1638 | 146 | 90.35% | 73.97% | 13.01% |
| `eval` | LA_0044 | 936 | 68 | 80.34% | 95.59% | 11.76% |
| `eval` | LA_0045 | 1638 | 146 | 88.64% | 93.15% | 8.91% |
| `eval` | LA_0046 | 936 | 68 | 87.71% | 82.35% | 14.40% |
| `eval` | LA_0047 | 1638 | 146 | 87.97% | 90.41% | 10.27% |
| `eval` | LA_0048 | 936 | 68 | 85.79% | 88.24% | 13.24% |
| `eval` | LA_0049 | 0 | 95 | n/a | 91.58% | n/a |
| `eval` | LA_0050 | 0 | 113 | n/a | 80.53% | n/a |
| `eval` | LA_0051 | 0 | 113 | n/a | 88.50% | n/a |
| `eval` | LA_0052 | 0 | 95 | n/a | 91.58% | n/a |
| `eval` | LA_0053 | 0 | 113 | n/a | 78.76% | n/a |
| `eval` | LA_0054 | 0 | 95 | n/a | 74.74% | n/a |
| `eval` | LA_0055 | 0 | 113 | n/a | 86.73% | n/a |
| `eval` | LA_0056 | 0 | 95 | n/a | 85.26% | n/a |
| `eval` | LA_0057 | 0 | 95 | n/a | 86.32% | n/a |
| `eval` | LA_0058 | 0 | 113 | n/a | 88.50% | n/a |
| `eval` | LA_0059 | 0 | 95 | n/a | 72.63% | n/a |
| `eval` | LA_0060 | 0 | 113 | n/a | 85.84% | n/a |
| `eval` | LA_0061 | 0 | 113 | n/a | 97.35% | n/a |
| `eval` | LA_0063 | 0 | 113 | n/a | 69.91% | n/a |
| `eval` | LA_0064 | 0 | 95 | n/a | 94.74% | n/a |
| `eval` | LA_0065 | 0 | 95 | n/a | 85.26% | n/a |
| `eval` | LA_0066 | 0 | 113 | n/a | 85.84% | n/a |
| `eval` | LA_0067 | 0 | 113 | n/a | 83.19% | n/a |
| `eval` | LA_0068 | 0 | 95 | n/a | 53.68% | n/a |

## Checkpoint

- Path: `Models/Iteration1/e_1_1/linear_stft/resnet18_frozen_linear_stft.pth`
- SHA-256: `a9aeb0c6c40657da33d68ce2c0af996ef60b013652465a10d0bc22e58c8afac1`
- Parameters: 11242305 total (11176512 backbone, 65793 head)
- State-dict keys: 124 (124 remapped)
- Missing keys: none
- Unexpected keys: none
- Head state SHA-256: `c19a4a9a18922e78b8e5d4d7879a2ad915576ee51018c37b6792b7854acd84d1`
- Final head bias: -0.3878
- Batch-norm steps: [5985]

## Provenance

- **dev**: {'dataset': 'ASVspoof 2019 LA', 'split': 'dev', 'source': 'image', 'data_root': 'data/LA', 'protocol_path': 'data/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt', 'protocol_sha256': '2fedaebf1ba3ae1a9bb7d3ec916804ac237b4eee063a27fa92e6549f03d129c7', 'image_root': 'data/derived/e1_1_frontend/linear_stft', 'metadata_path': 'data/derived/e1_1_frontend/linear_stft/dev/metadata_dev.csv', 'metadata_sha256': 'befc93505cc82ef4c0eb74b4721006254a81271a68036dbf27fe390ee7c3131e'}
- **eval**: {'dataset': 'ASVspoof 2019 LA', 'split': 'eval', 'source': 'image', 'data_root': 'data/LA', 'protocol_path': 'data/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt', 'protocol_sha256': '84fce576efff092269a0a6ad247d2e9f2f7b4d4e25a4545efb2518da92708036', 'image_root': 'data/derived/e1_1_frontend/linear_stft', 'metadata_path': 'data/derived/e1_1_frontend/linear_stft/eval/metadata_eval.csv', 'metadata_sha256': 'ef87efdd3bc428c16730c858b84fa907ba6cc2799175d1f8bc706f51092085d1'}

## Environment and command

- Git commit: `9881dbf6a5cafe8fe566f035d32b86b00a8cba21` (dirty: True)
- Device: `cuda` (NVIDIA GeForce RTX 3050 Laptop GPU)
- Python / torch / torchvision: 3.14.8 / 2.15.0+cu132 / 0.29.2+cu132
- Batch size / workers: 64 / 4

```bash
'/home/toastu/Research/Spoof Detection/.venv/bin/python' -m spoof_detection.evaluate --checkpoint '/home/toastu/Research/Spoof Detection/Models/Iteration1/e_1_1/linear_stft/resnet18_frozen_linear_stft.pth' --asvspoof dev eval --source image --data-root '/home/toastu/Research/Spoof Detection/data/LA' --image-root '/home/toastu/Research/Spoof Detection/data/derived/e1_1_frontend/linear_stft' --output-root '/home/toastu/Research/Spoof Detection/reports/phase1/e1_1_frontend/runs/linear_stft' --frontend linear_stft --experiment-id E1.1 --title 'E1.1 front-end probe: linear_stft' --batch-size 64 --num-workers 4 --device auto
```

## Caveats

- Scores are uncalibrated; Brier and ECE are diagnostics, not evidence of calibrated probabilities.
- EER uses the ASVspoof DET-curve convention (spoof is positive). Per-attack EER reuses all bonafide clips as negatives; per-speaker EER uses same-speaker bonafides.
- Only the dev EER threshold is selected on data. The `dev_eer` threshold in an eval report is transferred from dev and is never selected on eval. Other EER thresholds are descriptive points on the DET curve, not operating thresholds.
- Minimum t-DCF and confidence intervals are not reported in E0.2.
