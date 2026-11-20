# E1.1 front-end probe: randproj128: `resnet18_frozen_randproj128.pth`

Experiment: `E1.1`
Generated: `2026-11-20T16:05:44.884263+00:00`

## Metric contract

| Split | Attack subset | Operating thresholds | Samples | Balance |
|---|---|---|---:|---|
| `dev` | A01-A06 | fixed_0.5=0.500000, dev_eer=0.802520 | 24844 | 2548 / 22296 |
| `eval` | A07-A19 | fixed_0.5=0.500000, dev_eer=0.802520 | 71237 | 7355 / 63882 |

Threshold-free metrics: EER and AUC are computed over all spoof clips against all bonafide clips in the split. They are `n/a` when a split has only one class.

## Aggregate metrics

| Split | EER (threshold-free) | AUC (threshold-free) | Brier | ECE (10 bins) |
|---|---:|---:|---:|---:|
| `dev` | 19.71% | 0.8847 | 0.0716 | 0.0300 |
| `eval` | 17.04% | 0.9064 | 0.0730 | 0.0321 |

## Threshold metrics

| Split | Threshold | Value | Accuracy | Bonafide recall | Spoof recall | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|
| `dev` | `fixed_0.5` | 0.500000 | 90.15% | 36.22% | 96.32% | 63.78% | 3.68% |
| `dev` | `dev_eer` | 0.802520 | 80.28% | 80.26% | 80.29% | 19.74% | 19.71% |
| `eval` | `fixed_0.5` | 0.500000 | 88.61% | 44.54% | 93.68% | 55.46% | 6.32% |
| `eval` | `dev_eer` | 0.802520 | 82.50% | 84.83% | 82.23% | 15.17% | 17.77% |

## Per-attack EER

Per-attack EER uses that attack's spoof clips as positives and all bonafide clips in the split as negatives.

| Split | Attack | n_spoof | n_bonafide | EER | AUC |
|---|---|---:|---:|---:|---:|
| `dev` | A01 | 3716 | 2548 | 3.34% | 0.9954 |
| `dev` | A02 | 3716 | 2548 | 21.66% | 0.8549 |
| `dev` | A03 | 3716 | 2548 | 7.15% | 0.9773 |
| `dev` | A04 | 3716 | 2548 | 8.64% | 0.9707 |
| `dev` | A05 | 3716 | 2548 | 28.73% | 0.7868 |
| `dev` | A06 | 3716 | 2548 | 33.16% | 0.7231 |
| `eval` | A07 | 4914 | 7355 | 2.67% | 0.9963 |
| `eval` | A08 | 4914 | 7355 | 4.12% | 0.9871 |
| `eval` | A09 | 4914 | 7355 | 2.20% | 0.9971 |
| `eval` | A10 | 4914 | 7355 | 3.56% | 0.9931 |
| `eval` | A11 | 4914 | 7355 | 4.20% | 0.9917 |
| `eval` | A12 | 4914 | 7355 | 4.05% | 0.9920 |
| `eval` | A13 | 4914 | 7355 | 6.41% | 0.9813 |
| `eval` | A14 | 4914 | 7355 | 4.21% | 0.9907 |
| `eval` | A15 | 4914 | 7355 | 5.19% | 0.9866 |
| `eval` | A16 | 4914 | 7355 | 7.59% | 0.9743 |
| `eval` | A17 | 4914 | 7355 | 43.12% | 0.5962 |
| `eval` | A18 | 4914 | 7355 | 41.03% | 0.6199 |
| `eval` | A19 | 4914 | 7355 | 36.91% | 0.6767 |

## Per-speaker metrics

Per-speaker EER uses same-speaker bonafide clips as negatives. Recalls are at the fixed 0.5 threshold.

| Split | Speaker | n_spoof | n_bonafide | Spoof recall@0.5 | Bonafide recall@0.5 | EER |
|---|---|---:|---:|---:|---:|---:|
| `dev` | LA_0069 | 2484 | 154 | 97.30% | 38.96% | 15.50% |
| `dev` | LA_0070 | 1848 | 140 | 95.35% | 38.57% | 19.90% |
| `dev` | LA_0071 | 1848 | 140 | 93.18% | 53.57% | 21.29% |
| `dev` | LA_0072 | 2484 | 154 | 97.30% | 60.39% | 11.58% |
| `dev` | LA_0073 | 1848 | 140 | 94.16% | 50.00% | 19.27% |
| `dev` | LA_0074 | 2484 | 154 | 96.90% | 37.01% | 17.04% |
| `dev` | LA_0075 | 2484 | 154 | 96.50% | 57.79% | 12.29% |
| `dev` | LA_0076 | 1848 | 140 | 95.83% | 41.43% | 21.27% |
| `dev` | LA_0077 | 2484 | 154 | 97.02% | 40.91% | 16.39% |
| `dev` | LA_0078 | 2484 | 154 | 97.91% | 20.13% | 18.07% |
| `dev` | LA_0099 | 0 | 77 | n/a | 51.95% | n/a |
| `dev` | LA_0100 | 0 | 77 | n/a | 31.17% | n/a |
| `dev` | LA_0101 | 0 | 77 | n/a | 11.69% | n/a |
| `dev` | LA_0102 | 0 | 126 | n/a | 10.32% | n/a |
| `dev` | LA_0103 | 0 | 126 | n/a | 29.37% | n/a |
| `dev` | LA_0104 | 0 | 126 | n/a | 40.48% | n/a |
| `dev` | LA_0105 | 0 | 77 | n/a | 22.08% | n/a |
| `dev` | LA_0106 | 0 | 126 | n/a | 20.63% | n/a |
| `dev` | LA_0107 | 0 | 126 | n/a | 19.84% | n/a |
| `dev` | LA_0108 | 0 | 126 | n/a | 24.60% | n/a |
| `eval` | LA_0001 | 936 | 68 | 94.87% | 73.53% | 11.87% |
| `eval` | LA_0002 | 936 | 68 | 93.80% | 17.65% | 20.82% |
| `eval` | LA_0003 | 936 | 68 | 96.15% | 48.53% | 12.03% |
| `eval` | LA_0004 | 1638 | 146 | 92.67% | 34.25% | 17.26% |
| `eval` | LA_0005 | 936 | 68 | 93.70% | 61.76% | 14.56% |
| `eval` | LA_0006 | 1638 | 146 | 94.75% | 39.73% | 15.75% |
| `eval` | LA_0007 | 936 | 68 | 95.19% | 41.18% | 13.24% |
| `eval` | LA_0008 | 1638 | 146 | 93.47% | 36.99% | 18.53% |
| `eval` | LA_0009 | 1638 | 146 | 92.61% | 41.10% | 16.31% |
| `eval` | LA_0010 | 1638 | 146 | 94.75% | 49.32% | 17.88% |
| `eval` | LA_0011 | 936 | 68 | 92.31% | 55.88% | 14.72% |
| `eval` | LA_0012 | 1638 | 146 | 93.47% | 56.85% | 13.84% |
| `eval` | LA_0013 | 936 | 68 | 93.91% | 42.65% | 19.39% |
| `eval` | LA_0014 | 1638 | 146 | 91.70% | 58.90% | 17.11% |
| `eval` | LA_0015 | 936 | 68 | 95.19% | 20.59% | 18.91% |
| `eval` | LA_0016 | 1638 | 146 | 93.59% | 58.22% | 15.72% |
| `eval` | LA_0017 | 1638 | 146 | 93.04% | 71.92% | 13.69% |
| `eval` | LA_0018 | 936 | 68 | 93.16% | 61.76% | 12.97% |
| `eval` | LA_0019 | 1638 | 146 | 93.96% | 49.32% | 15.07% |
| `eval` | LA_0020 | 1638 | 146 | 92.61% | 67.81% | 12.33% |
| `eval` | LA_0021 | 936 | 68 | 92.63% | 33.82% | 18.96% |
| `eval` | LA_0022 | 1638 | 146 | 94.32% | 20.55% | 19.85% |
| `eval` | LA_0023 | 936 | 68 | 92.41% | 50.00% | 17.64% |
| `eval` | LA_0024 | 1638 | 146 | 93.41% | 37.67% | 17.11% |
| `eval` | LA_0025 | 936 | 68 | 94.34% | 50.00% | 14.72% |
| `eval` | LA_0026 | 1638 | 146 | 93.59% | 63.01% | 13.69% |
| `eval` | LA_0027 | 1638 | 146 | 92.06% | 25.34% | 20.56% |
| `eval` | LA_0028 | 936 | 68 | 93.16% | 42.65% | 14.67% |
| `eval` | LA_0029 | 1638 | 146 | 93.10% | 43.84% | 15.75% |
| `eval` | LA_0030 | 936 | 68 | 94.44% | 22.06% | 22.03% |
| `eval` | LA_0031 | 1638 | 146 | 93.47% | 26.71% | 19.76% |
| `eval` | LA_0032 | 936 | 68 | 94.02% | 17.65% | 25.00% |
| `eval` | LA_0033 | 1638 | 146 | 93.04% | 57.53% | 16.58% |
| `eval` | LA_0034 | 1638 | 146 | 93.96% | 50.68% | 15.04% |
| `eval` | LA_0035 | 1638 | 146 | 94.02% | 54.79% | 14.37% |
| `eval` | LA_0036 | 936 | 68 | 93.38% | 35.29% | 20.34% |
| `eval` | LA_0037 | 1638 | 146 | 94.38% | 42.47% | 14.92% |
| `eval` | LA_0038 | 936 | 68 | 96.26% | 4.41% | 30.61% |
| `eval` | LA_0039 | 1638 | 146 | 92.67% | 56.85% | 15.07% |
| `eval` | LA_0040 | 936 | 68 | 94.12% | 19.12% | 20.92% |
| `eval` | LA_0041 | 1638 | 146 | 93.83% | 61.64% | 13.69% |
| `eval` | LA_0042 | 1638 | 146 | 94.14% | 43.15% | 17.82% |
| `eval` | LA_0043 | 1638 | 146 | 93.83% | 54.79% | 15.75% |
| `eval` | LA_0044 | 936 | 68 | 93.91% | 57.35% | 14.62% |
| `eval` | LA_0045 | 1638 | 146 | 94.26% | 35.62% | 15.72% |
| `eval` | LA_0046 | 936 | 68 | 95.51% | 57.35% | 11.87% |
| `eval` | LA_0047 | 1638 | 146 | 94.26% | 50.00% | 13.01% |
| `eval` | LA_0048 | 936 | 68 | 92.74% | 39.71% | 18.01% |
| `eval` | LA_0049 | 0 | 95 | n/a | 48.42% | n/a |
| `eval` | LA_0050 | 0 | 113 | n/a | 19.47% | n/a |
| `eval` | LA_0051 | 0 | 113 | n/a | 39.82% | n/a |
| `eval` | LA_0052 | 0 | 95 | n/a | 77.89% | n/a |
| `eval` | LA_0053 | 0 | 113 | n/a | 38.05% | n/a |
| `eval` | LA_0054 | 0 | 95 | n/a | 35.79% | n/a |
| `eval` | LA_0055 | 0 | 113 | n/a | 46.90% | n/a |
| `eval` | LA_0056 | 0 | 95 | n/a | 35.79% | n/a |
| `eval` | LA_0057 | 0 | 95 | n/a | 37.89% | n/a |
| `eval` | LA_0058 | 0 | 113 | n/a | 44.25% | n/a |
| `eval` | LA_0059 | 0 | 95 | n/a | 26.32% | n/a |
| `eval` | LA_0060 | 0 | 113 | n/a | 48.67% | n/a |
| `eval` | LA_0061 | 0 | 113 | n/a | 61.95% | n/a |
| `eval` | LA_0063 | 0 | 113 | n/a | 38.05% | n/a |
| `eval` | LA_0064 | 0 | 95 | n/a | 45.26% | n/a |
| `eval` | LA_0065 | 0 | 95 | n/a | 30.53% | n/a |
| `eval` | LA_0066 | 0 | 113 | n/a | 38.05% | n/a |
| `eval` | LA_0067 | 0 | 113 | n/a | 53.98% | n/a |
| `eval` | LA_0068 | 0 | 95 | n/a | 8.42% | n/a |

## Checkpoint

- Path: `Models/Iteration1/e_1_1/randproj128/resnet18_frozen_randproj128.pth`
- SHA-256: `54cd0d8f707cec6b8f733c991c6c7676ae456e17f9085a04c9813cc3f119dd12`
- Parameters: 11242305 total (11176512 backbone, 65793 head)
- State-dict keys: 124 (124 remapped)
- Missing keys: none
- Unexpected keys: none
- Head state SHA-256: `d44907bce7ac7fa78b34fd000497c6c355bc88807657cc1e9af5d878843be198`
- Final head bias: -0.2307
- Batch-norm steps: [3465]

## Provenance

- **dev**: {'dataset': 'ASVspoof 2019 LA', 'split': 'dev', 'source': 'image', 'data_root': 'data/LA', 'protocol_path': 'data/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt', 'protocol_sha256': '2fedaebf1ba3ae1a9bb7d3ec916804ac237b4eee063a27fa92e6549f03d129c7', 'image_root': 'data/derived/e1_1_frontend/randproj128', 'metadata_path': 'data/derived/e1_1_frontend/randproj128/dev/metadata_dev.csv', 'metadata_sha256': '8852cf2c510e68b76ee83e5aa6098f95236a330e5d0a7c4784cda610cecc5fd3'}
- **eval**: {'dataset': 'ASVspoof 2019 LA', 'split': 'eval', 'source': 'image', 'data_root': 'data/LA', 'protocol_path': 'data/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt', 'protocol_sha256': '84fce576efff092269a0a6ad247d2e9f2f7b4d4e25a4545efb2518da92708036', 'image_root': 'data/derived/e1_1_frontend/randproj128', 'metadata_path': 'data/derived/e1_1_frontend/randproj128/eval/metadata_eval.csv', 'metadata_sha256': '71aadfe9e8bfd5f5043ee07adf2d993ce7950194d8a864623baa6fa9c4a989de'}

## Environment and command

- Git commit: `9881dbf6a5cafe8fe566f035d32b86b00a8cba21` (dirty: True)
- Device: `cuda` (NVIDIA GeForce RTX 3050 Laptop GPU)
- Python / torch / torchvision: 3.14.8 / 2.15.0+cu132 / 0.29.2+cu132
- Batch size / workers: 64 / 4

```bash
'/home/toastu/Research/Spoof Detection/.venv/bin/python' -m spoof_detection.evaluate --checkpoint '/home/toastu/Research/Spoof Detection/Models/Iteration1/e_1_1/randproj128/resnet18_frozen_randproj128.pth' --asvspoof dev eval --source image --data-root '/home/toastu/Research/Spoof Detection/data/LA' --image-root '/home/toastu/Research/Spoof Detection/data/derived/e1_1_frontend/randproj128' --output-root '/home/toastu/Research/Spoof Detection/reports/phase1/e1_1_frontend/runs/randproj128' --frontend randproj128 --experiment-id E1.1 --title 'E1.1 front-end probe: randproj128' --batch-size 64 --num-workers 4 --device auto
```

## Caveats

- Scores are uncalibrated; Brier and ECE are diagnostics, not evidence of calibrated probabilities.
- EER uses the ASVspoof DET-curve convention (spoof is positive). Per-attack EER reuses all bonafide clips as negatives; per-speaker EER uses same-speaker bonafides.
- Only the dev EER threshold is selected on data. The `dev_eer` threshold in an eval report is transferred from dev and is never selected on eval. Other EER thresholds are descriptive points on the DET curve, not operating thresholds.
- Minimum t-DCF and confidence intervals are not reported in E0.2.
