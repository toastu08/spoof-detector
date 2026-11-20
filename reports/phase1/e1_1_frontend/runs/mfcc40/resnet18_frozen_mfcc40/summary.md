# E1.1 front-end probe: mfcc40: `resnet18_frozen_mfcc40.pth`

Experiment: `E1.1`
Generated: `2026-11-20T16:03:30.752279+00:00`

## Metric contract

| Split | Attack subset | Operating thresholds | Samples | Balance |
|---|---|---|---:|---|
| `dev` | A01-A06 | fixed_0.5=0.500000, dev_eer=0.844174 | 24844 | 2548 / 22296 |
| `eval` | A07-A19 | fixed_0.5=0.500000, dev_eer=0.844174 | 71237 | 7355 / 63882 |

Threshold-free metrics: EER and AUC are computed over all spoof clips against all bonafide clips in the split. They are `n/a` when a split has only one class.

## Aggregate metrics

| Split | EER (threshold-free) | AUC (threshold-free) | Brier | ECE (10 bins) |
|---|---:|---:|---:|---:|
| `dev` | 14.49% | 0.9327 | 0.0561 | 0.0141 |
| `eval` | 18.56% | 0.8963 | 0.0867 | 0.0683 |

## Threshold metrics

| Split | Threshold | Value | Accuracy | Bonafide recall | Spoof recall | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|
| `dev` | `fixed_0.5` | 0.500000 | 92.18% | 53.18% | 96.64% | 46.82% | 3.36% |
| `dev` | `dev_eer` | 0.844174 | 85.51% | 85.48% | 85.51% | 14.52% | 14.49% |
| `eval` | `fixed_0.5` | 0.500000 | 87.22% | 60.69% | 90.27% | 39.31% | 9.73% |
| `eval` | `dev_eer` | 0.844174 | 77.86% | 88.52% | 76.63% | 11.48% | 23.37% |

## Per-attack EER

Per-attack EER uses that attack's spoof clips as positives and all bonafide clips in the split as negatives.

| Split | Attack | n_spoof | n_bonafide | EER | AUC |
|---|---|---:|---:|---:|---:|
| `dev` | A01 | 3716 | 2548 | 5.01% | 0.9893 |
| `dev` | A02 | 3716 | 2548 | 6.72% | 0.9805 |
| `dev` | A03 | 3716 | 2548 | 6.55% | 0.9817 |
| `dev` | A04 | 3716 | 2548 | 13.70% | 0.9338 |
| `dev` | A05 | 3716 | 2548 | 14.21% | 0.9303 |
| `dev` | A06 | 3716 | 2548 | 28.61% | 0.7809 |
| `eval` | A07 | 4914 | 7355 | 6.98% | 0.9801 |
| `eval` | A08 | 4914 | 7355 | 5.64% | 0.9858 |
| `eval` | A09 | 4914 | 7355 | 3.24% | 0.9950 |
| `eval` | A10 | 4914 | 7355 | 9.77% | 0.9662 |
| `eval` | A11 | 4914 | 7355 | 5.97% | 0.9851 |
| `eval` | A12 | 4914 | 7355 | 10.48% | 0.9600 |
| `eval` | A13 | 4914 | 7355 | 11.58% | 0.9517 |
| `eval` | A14 | 4914 | 7355 | 11.29% | 0.9572 |
| `eval` | A15 | 4914 | 7355 | 7.57% | 0.9758 |
| `eval` | A16 | 4914 | 7355 | 12.78% | 0.9442 |
| `eval` | A17 | 4914 | 7355 | 40.00% | 0.6373 |
| `eval` | A18 | 4914 | 7355 | 40.21% | 0.6302 |
| `eval` | A19 | 4914 | 7355 | 36.25% | 0.6836 |

## Per-speaker metrics

Per-speaker EER uses same-speaker bonafide clips as negatives. Recalls are at the fixed 0.5 threshold.

| Split | Speaker | n_spoof | n_bonafide | Spoof recall@0.5 | Bonafide recall@0.5 | EER |
|---|---|---:|---:|---:|---:|---:|
| `dev` | LA_0069 | 2484 | 154 | 95.69% | 75.32% | 10.97% |
| `dev` | LA_0070 | 1848 | 140 | 97.35% | 56.43% | 13.58% |
| `dev` | LA_0071 | 1848 | 140 | 96.59% | 43.57% | 17.15% |
| `dev` | LA_0072 | 2484 | 154 | 96.94% | 59.74% | 13.64% |
| `dev` | LA_0073 | 1848 | 140 | 97.13% | 67.86% | 11.59% |
| `dev` | LA_0074 | 2484 | 154 | 96.94% | 55.84% | 12.85% |
| `dev` | LA_0075 | 2484 | 154 | 96.10% | 40.91% | 15.58% |
| `dev` | LA_0076 | 1848 | 140 | 96.81% | 58.57% | 12.13% |
| `dev` | LA_0077 | 2484 | 154 | 96.58% | 75.97% | 9.74% |
| `dev` | LA_0078 | 2484 | 154 | 96.62% | 66.88% | 13.00% |
| `dev` | LA_0099 | 0 | 77 | n/a | 50.65% | n/a |
| `dev` | LA_0100 | 0 | 77 | n/a | 58.44% | n/a |
| `dev` | LA_0101 | 0 | 77 | n/a | 25.97% | n/a |
| `dev` | LA_0102 | 0 | 126 | n/a | 34.13% | n/a |
| `dev` | LA_0103 | 0 | 126 | n/a | 52.38% | n/a |
| `dev` | LA_0104 | 0 | 126 | n/a | 45.24% | n/a |
| `dev` | LA_0105 | 0 | 77 | n/a | 37.66% | n/a |
| `dev` | LA_0106 | 0 | 126 | n/a | 43.65% | n/a |
| `dev` | LA_0107 | 0 | 126 | n/a | 44.44% | n/a |
| `dev` | LA_0108 | 0 | 126 | n/a | 40.48% | n/a |
| `eval` | LA_0001 | 936 | 68 | 92.09% | 76.47% | 11.60% |
| `eval` | LA_0002 | 936 | 68 | 93.38% | 22.06% | 28.23% |
| `eval` | LA_0003 | 936 | 68 | 94.34% | 48.53% | 18.91% |
| `eval` | LA_0004 | 1638 | 146 | 87.48% | 80.82% | 14.27% |
| `eval` | LA_0005 | 936 | 68 | 92.20% | 70.59% | 13.56% |
| `eval` | LA_0006 | 1638 | 146 | 89.74% | 60.27% | 18.50% |
| `eval` | LA_0007 | 936 | 68 | 93.70% | 63.24% | 14.40% |
| `eval` | LA_0008 | 1638 | 146 | 90.90% | 69.86% | 15.07% |
| `eval` | LA_0009 | 1638 | 146 | 89.99% | 59.59% | 19.08% |
| `eval` | LA_0010 | 1638 | 146 | 89.68% | 74.66% | 15.07% |
| `eval` | LA_0011 | 936 | 68 | 93.59% | 55.88% | 19.12% |
| `eval` | LA_0012 | 1638 | 146 | 89.19% | 85.62% | 12.33% |
| `eval` | LA_0013 | 936 | 68 | 92.31% | 58.82% | 22.03% |
| `eval` | LA_0014 | 1638 | 146 | 86.63% | 57.53% | 20.56% |
| `eval` | LA_0015 | 936 | 68 | 92.95% | 35.29% | 22.03% |
| `eval` | LA_0016 | 1638 | 146 | 88.52% | 79.45% | 15.07% |
| `eval` | LA_0017 | 1638 | 146 | 87.85% | 84.93% | 13.60% |
| `eval` | LA_0018 | 936 | 68 | 92.31% | 86.76% | 10.28% |
| `eval` | LA_0019 | 1638 | 146 | 90.17% | 52.05% | 21.80% |
| `eval` | LA_0020 | 1638 | 146 | 86.63% | 73.29% | 19.02% |
| `eval` | LA_0021 | 936 | 68 | 90.17% | 72.06% | 14.83% |
| `eval` | LA_0022 | 1638 | 146 | 92.49% | 51.37% | 16.46% |
| `eval` | LA_0023 | 936 | 68 | 92.52% | 80.88% | 11.76% |
| `eval` | LA_0024 | 1638 | 146 | 89.56% | 69.86% | 17.73% |
| `eval` | LA_0025 | 936 | 68 | 93.91% | 69.12% | 10.38% |
| `eval` | LA_0026 | 1638 | 146 | 90.29% | 74.66% | 14.40% |
| `eval` | LA_0027 | 1638 | 146 | 89.19% | 23.29% | 26.63% |
| `eval` | LA_0028 | 936 | 68 | 91.88% | 83.82% | 11.81% |
| `eval` | LA_0029 | 1638 | 146 | 87.18% | 70.55% | 17.66% |
| `eval` | LA_0030 | 936 | 68 | 94.02% | 48.53% | 20.28% |
| `eval` | LA_0031 | 1638 | 146 | 88.77% | 39.73% | 22.01% |
| `eval` | LA_0032 | 936 | 68 | 94.02% | 41.18% | 19.23% |
| `eval` | LA_0033 | 1638 | 146 | 88.46% | 67.81% | 19.20% |
| `eval` | LA_0034 | 1638 | 146 | 88.22% | 59.59% | 17.66% |
| `eval` | LA_0035 | 1638 | 146 | 87.85% | 75.34% | 16.43% |
| `eval` | LA_0036 | 936 | 68 | 91.13% | 47.06% | 20.60% |
| `eval` | LA_0037 | 1638 | 146 | 87.67% | 77.40% | 16.43% |
| `eval` | LA_0038 | 936 | 68 | 93.06% | 11.76% | 39.72% |
| `eval` | LA_0039 | 1638 | 146 | 87.85% | 70.55% | 17.29% |
| `eval` | LA_0040 | 936 | 68 | 93.27% | 44.12% | 17.74% |
| `eval` | LA_0041 | 1638 | 146 | 89.50% | 50.00% | 17.85% |
| `eval` | LA_0042 | 1638 | 146 | 91.94% | 61.64% | 16.58% |
| `eval` | LA_0043 | 1638 | 146 | 89.56% | 54.11% | 19.11% |
| `eval` | LA_0044 | 936 | 68 | 92.95% | 41.18% | 20.39% |
| `eval` | LA_0045 | 1638 | 146 | 90.35% | 78.08% | 14.49% |
| `eval` | LA_0046 | 936 | 68 | 92.74% | 83.82% | 10.28% |
| `eval` | LA_0047 | 1638 | 146 | 91.33% | 49.32% | 17.08% |
| `eval` | LA_0048 | 936 | 68 | 92.31% | 38.24% | 22.14% |
| `eval` | LA_0049 | 0 | 95 | n/a | 71.58% | n/a |
| `eval` | LA_0050 | 0 | 113 | n/a | 23.89% | n/a |
| `eval` | LA_0051 | 0 | 113 | n/a | 72.57% | n/a |
| `eval` | LA_0052 | 0 | 95 | n/a | 81.05% | n/a |
| `eval` | LA_0053 | 0 | 113 | n/a | 69.91% | n/a |
| `eval` | LA_0054 | 0 | 95 | n/a | 40.00% | n/a |
| `eval` | LA_0055 | 0 | 113 | n/a | 69.03% | n/a |
| `eval` | LA_0056 | 0 | 95 | n/a | 47.37% | n/a |
| `eval` | LA_0057 | 0 | 95 | n/a | 49.47% | n/a |
| `eval` | LA_0058 | 0 | 113 | n/a | 82.30% | n/a |
| `eval` | LA_0059 | 0 | 95 | n/a | 36.84% | n/a |
| `eval` | LA_0060 | 0 | 113 | n/a | 61.95% | n/a |
| `eval` | LA_0061 | 0 | 113 | n/a | 73.45% | n/a |
| `eval` | LA_0063 | 0 | 113 | n/a | 61.95% | n/a |
| `eval` | LA_0064 | 0 | 95 | n/a | 56.84% | n/a |
| `eval` | LA_0065 | 0 | 95 | n/a | 33.68% | n/a |
| `eval` | LA_0066 | 0 | 113 | n/a | 37.17% | n/a |
| `eval` | LA_0067 | 0 | 113 | n/a | 58.41% | n/a |
| `eval` | LA_0068 | 0 | 95 | n/a | 20.00% | n/a |

## Checkpoint

- Path: `Models/Iteration1/e_1_1/mfcc40/resnet18_frozen_mfcc40.pth`
- SHA-256: `4023e84c804bbeb8dbe8abe1bae92e70a1db4cfc5ec1925beeec8cbc8e4df138`
- Parameters: 11242305 total (11176512 backbone, 65793 head)
- State-dict keys: 124 (124 remapped)
- Missing keys: none
- Unexpected keys: none
- Head state SHA-256: `662046a9533ec19747fcd173c99d21e157f8e05029314c3302a150f64975a3b2`
- Final head bias: -0.2015
- Batch-norm steps: [3150]

## Provenance

- **dev**: {'dataset': 'ASVspoof 2019 LA', 'split': 'dev', 'source': 'image', 'data_root': 'data/LA', 'protocol_path': 'data/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt', 'protocol_sha256': '2fedaebf1ba3ae1a9bb7d3ec916804ac237b4eee063a27fa92e6549f03d129c7', 'image_root': 'data/derived/e1_1_frontend/mfcc40', 'metadata_path': 'data/derived/e1_1_frontend/mfcc40/dev/metadata_dev.csv', 'metadata_sha256': '53e90312369fb7a9db34a643213fb13634e2c8f9894519715085c0e8489e730a'}
- **eval**: {'dataset': 'ASVspoof 2019 LA', 'split': 'eval', 'source': 'image', 'data_root': 'data/LA', 'protocol_path': 'data/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt', 'protocol_sha256': '84fce576efff092269a0a6ad247d2e9f2f7b4d4e25a4545efb2518da92708036', 'image_root': 'data/derived/e1_1_frontend/mfcc40', 'metadata_path': 'data/derived/e1_1_frontend/mfcc40/eval/metadata_eval.csv', 'metadata_sha256': '1c2c80fb65fda60a3b874f0b356dca48cdcaf373e00452f26e5b37b1f6e89516'}

## Environment and command

- Git commit: `9881dbf6a5cafe8fe566f035d32b86b00a8cba21` (dirty: True)
- Device: `cuda` (NVIDIA GeForce RTX 3050 Laptop GPU)
- Python / torch / torchvision: 3.14.8 / 2.15.0+cu132 / 0.29.2+cu132
- Batch size / workers: 64 / 4

```bash
'/home/toastu/Research/Spoof Detection/.venv/bin/python' -m spoof_detection.evaluate --checkpoint '/home/toastu/Research/Spoof Detection/Models/Iteration1/e_1_1/mfcc40/resnet18_frozen_mfcc40.pth' --asvspoof dev eval --source image --data-root '/home/toastu/Research/Spoof Detection/data/LA' --image-root '/home/toastu/Research/Spoof Detection/data/derived/e1_1_frontend/mfcc40' --output-root '/home/toastu/Research/Spoof Detection/reports/phase1/e1_1_frontend/runs/mfcc40' --frontend mfcc40 --experiment-id E1.1 --title 'E1.1 front-end probe: mfcc40' --batch-size 64 --num-workers 4 --device auto
```

## Caveats

- Scores are uncalibrated; Brier and ECE are diagnostics, not evidence of calibrated probabilities.
- EER uses the ASVspoof DET-curve convention (spoof is positive). Per-attack EER reuses all bonafide clips as negatives; per-speaker EER uses same-speaker bonafides.
- Only the dev EER threshold is selected on data. The `dev_eer` threshold in an eval report is transferred from dev and is never selected on eval. Other EER thresholds are descriptive points on the DET curve, not operating thresholds.
- Minimum t-DCF and confidence intervals are not reported in E0.2.
