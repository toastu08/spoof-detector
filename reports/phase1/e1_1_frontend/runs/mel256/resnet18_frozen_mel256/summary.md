# E1.1 front-end probe: mel256: `resnet18_frozen_mel256.pth`

Experiment: `E1.1`
Generated: `2026-11-20T15:58:03.039692+00:00`

## Metric contract

| Split | Attack subset | Operating thresholds | Samples | Balance |
|---|---|---|---:|---|
| `dev` | A01-A06 | fixed_0.5=0.500000, dev_eer=0.831765 | 24844 | 2548 / 22296 |
| `eval` | A07-A19 | fixed_0.5=0.500000, dev_eer=0.831765 | 71237 | 7355 / 63882 |

Threshold-free metrics: EER and AUC are computed over all spoof clips against all bonafide clips in the split. They are `n/a` when a split has only one class.

## Aggregate metrics

| Split | EER (threshold-free) | AUC (threshold-free) | Brier | ECE (10 bins) |
|---|---:|---:|---:|---:|
| `dev` | 4.36% | 0.9919 | 0.0188 | 0.0071 |
| `eval` | 11.31% | 0.9525 | 0.0814 | 0.0922 |

## Threshold metrics

| Split | Threshold | Value | Accuracy | Bonafide recall | Spoof recall | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|
| `dev` | `fixed_0.5` | 0.500000 | 97.59% | 87.21% | 98.78% | 12.79% | 1.22% |
| `dev` | `dev_eer` | 0.831765 | 95.63% | 95.60% | 95.64% | 4.40% | 4.36% |
| `eval` | `fixed_0.5` | 0.500000 | 88.77% | 88.47% | 88.80% | 11.53% | 11.20% |
| `eval` | `dev_eer` | 0.831765 | 82.33% | 95.74% | 80.79% | 4.26% | 19.21% |

## Per-attack EER

Per-attack EER uses that attack's spoof clips as positives and all bonafide clips in the split as negatives.

| Split | Attack | n_spoof | n_bonafide | EER | AUC |
|---|---|---:|---:|---:|---:|
| `dev` | A01 | 3716 | 2548 | 1.69% | 0.9987 |
| `dev` | A02 | 3716 | 2548 | 0.83% | 0.9996 |
| `dev` | A03 | 3716 | 2548 | 2.71% | 0.9964 |
| `dev` | A04 | 3716 | 2548 | 3.85% | 0.9935 |
| `dev` | A05 | 3716 | 2548 | 5.25% | 0.9876 |
| `dev` | A06 | 3716 | 2548 | 7.89% | 0.9756 |
| `eval` | A07 | 4914 | 7355 | 0.92% | 0.9995 |
| `eval` | A08 | 4914 | 7355 | 2.79% | 0.9962 |
| `eval` | A09 | 4914 | 7355 | 0.99% | 0.9993 |
| `eval` | A10 | 4914 | 7355 | 7.02% | 0.9796 |
| `eval` | A11 | 4914 | 7355 | 3.44% | 0.9942 |
| `eval` | A12 | 4914 | 7355 | 6.77% | 0.9802 |
| `eval` | A13 | 4914 | 7355 | 6.98% | 0.9779 |
| `eval` | A14 | 4914 | 7355 | 3.54% | 0.9940 |
| `eval` | A15 | 4914 | 7355 | 6.54% | 0.9822 |
| `eval` | A16 | 4914 | 7355 | 3.82% | 0.9929 |
| `eval` | A17 | 4914 | 7355 | 36.00% | 0.6857 |
| `eval` | A18 | 4914 | 7355 | 18.72% | 0.8844 |
| `eval` | A19 | 4914 | 7355 | 15.69% | 0.9158 |

## Per-speaker metrics

Per-speaker EER uses same-speaker bonafide clips as negatives. Recalls are at the fixed 0.5 threshold.

| Split | Speaker | n_spoof | n_bonafide | Spoof recall@0.5 | Bonafide recall@0.5 | EER |
|---|---|---:|---:|---:|---:|---:|
| `dev` | LA_0069 | 2484 | 154 | 98.83% | 90.91% | 3.25% |
| `dev` | LA_0070 | 1848 | 140 | 97.29% | 89.29% | 5.73% |
| `dev` | LA_0071 | 1848 | 140 | 99.19% | 72.86% | 7.14% |
| `dev` | LA_0072 | 2484 | 154 | 99.03% | 85.71% | 3.25% |
| `dev` | LA_0073 | 1848 | 140 | 99.24% | 92.86% | 2.86% |
| `dev` | LA_0074 | 2484 | 154 | 98.03% | 93.51% | 3.29% |
| `dev` | LA_0075 | 2484 | 154 | 98.43% | 92.21% | 2.61% |
| `dev` | LA_0076 | 1848 | 140 | 98.76% | 90.00% | 3.57% |
| `dev` | LA_0077 | 2484 | 154 | 99.44% | 89.61% | 3.37% |
| `dev` | LA_0078 | 2484 | 154 | 99.36% | 92.21% | 3.25% |
| `dev` | LA_0099 | 0 | 77 | n/a | 85.71% | n/a |
| `dev` | LA_0100 | 0 | 77 | n/a | 83.12% | n/a |
| `dev` | LA_0101 | 0 | 77 | n/a | 76.62% | n/a |
| `dev` | LA_0102 | 0 | 126 | n/a | 68.25% | n/a |
| `dev` | LA_0103 | 0 | 126 | n/a | 84.92% | n/a |
| `dev` | LA_0104 | 0 | 126 | n/a | 80.16% | n/a |
| `dev` | LA_0105 | 0 | 77 | n/a | 87.01% | n/a |
| `dev` | LA_0106 | 0 | 126 | n/a | 95.24% | n/a |
| `dev` | LA_0107 | 0 | 126 | n/a | 89.68% | n/a |
| `dev` | LA_0108 | 0 | 126 | n/a | 93.65% | n/a |
| `eval` | LA_0001 | 936 | 68 | 89.85% | 89.71% | 10.28% |
| `eval` | LA_0002 | 936 | 68 | 90.49% | 57.35% | 15.94% |
| `eval` | LA_0003 | 936 | 68 | 90.49% | 83.82% | 11.71% |
| `eval` | LA_0004 | 1638 | 146 | 87.85% | 93.84% | 8.91% |
| `eval` | LA_0005 | 936 | 68 | 90.81% | 97.06% | 4.40% |
| `eval` | LA_0006 | 1638 | 146 | 89.19% | 86.99% | 11.56% |
| `eval` | LA_0007 | 936 | 68 | 89.00% | 97.06% | 8.79% |
| `eval` | LA_0008 | 1638 | 146 | 88.03% | 93.84% | 10.27% |
| `eval` | LA_0009 | 1638 | 146 | 88.34% | 89.04% | 11.04% |
| `eval` | LA_0010 | 1638 | 146 | 89.32% | 80.82% | 14.98% |
| `eval` | LA_0011 | 936 | 68 | 87.39% | 88.24% | 12.08% |
| `eval` | LA_0012 | 1638 | 146 | 90.78% | 66.44% | 16.58% |
| `eval` | LA_0013 | 936 | 68 | 90.28% | 85.29% | 10.60% |
| `eval` | LA_0014 | 1638 | 146 | 89.26% | 75.34% | 15.04% |
| `eval` | LA_0015 | 936 | 68 | 89.96% | 92.65% | 8.74% |
| `eval` | LA_0016 | 1638 | 146 | 88.52% | 94.52% | 8.91% |
| `eval` | LA_0017 | 1638 | 146 | 88.83% | 81.51% | 13.69% |
| `eval` | LA_0018 | 936 | 68 | 89.00% | 89.71% | 10.54% |
| `eval` | LA_0019 | 1638 | 146 | 87.97% | 75.34% | 14.40% |
| `eval` | LA_0020 | 1638 | 146 | 89.99% | 93.84% | 7.61% |
| `eval` | LA_0021 | 936 | 68 | 86.54% | 97.06% | 7.63% |
| `eval` | LA_0022 | 1638 | 146 | 89.01% | 89.04% | 10.97% |
| `eval` | LA_0023 | 936 | 68 | 87.82% | 91.18% | 10.54% |
| `eval` | LA_0024 | 1638 | 146 | 89.93% | 95.21% | 8.85% |
| `eval` | LA_0025 | 936 | 68 | 88.03% | 100.00% | 5.99% |
| `eval` | LA_0026 | 1638 | 146 | 89.32% | 89.73% | 10.27% |
| `eval` | LA_0027 | 1638 | 146 | 87.06% | 94.52% | 8.91% |
| `eval` | LA_0028 | 936 | 68 | 89.00% | 98.53% | 4.66% |
| `eval` | LA_0029 | 1638 | 146 | 89.56% | 97.26% | 8.23% |
| `eval` | LA_0030 | 936 | 68 | 90.60% | 79.41% | 14.78% |
| `eval` | LA_0031 | 1638 | 146 | 86.87% | 86.99% | 13.07% |
| `eval` | LA_0032 | 936 | 68 | 85.90% | 91.18% | 11.87% |
| `eval` | LA_0033 | 1638 | 146 | 88.22% | 84.25% | 13.84% |
| `eval` | LA_0034 | 1638 | 146 | 90.66% | 91.78% | 8.91% |
| `eval` | LA_0035 | 1638 | 146 | 88.58% | 90.41% | 10.97% |
| `eval` | LA_0036 | 936 | 68 | 87.61% | 97.06% | 8.85% |
| `eval` | LA_0037 | 1638 | 146 | 90.17% | 95.21% | 7.52% |
| `eval` | LA_0038 | 936 | 68 | 90.17% | 42.65% | 20.50% |
| `eval` | LA_0039 | 1638 | 146 | 85.59% | 97.26% | 9.59% |
| `eval` | LA_0040 | 936 | 68 | 86.32% | 91.18% | 11.76% |
| `eval` | LA_0041 | 1638 | 146 | 88.52% | 76.71% | 15.66% |
| `eval` | LA_0042 | 1638 | 146 | 87.36% | 94.52% | 9.59% |
| `eval` | LA_0043 | 1638 | 146 | 90.17% | 83.56% | 11.71% |
| `eval` | LA_0044 | 936 | 68 | 86.86% | 88.24% | 11.76% |
| `eval` | LA_0045 | 1638 | 146 | 89.26% | 94.52% | 8.76% |
| `eval` | LA_0046 | 936 | 68 | 90.60% | 91.18% | 8.85% |
| `eval` | LA_0047 | 1638 | 146 | 89.13% | 93.84% | 9.03% |
| `eval` | LA_0048 | 936 | 68 | 88.25% | 88.24% | 11.76% |
| `eval` | LA_0049 | 0 | 95 | n/a | 94.74% | n/a |
| `eval` | LA_0050 | 0 | 113 | n/a | 85.84% | n/a |
| `eval` | LA_0051 | 0 | 113 | n/a | 82.30% | n/a |
| `eval` | LA_0052 | 0 | 95 | n/a | 94.74% | n/a |
| `eval` | LA_0053 | 0 | 113 | n/a | 95.58% | n/a |
| `eval` | LA_0054 | 0 | 95 | n/a | 86.32% | n/a |
| `eval` | LA_0055 | 0 | 113 | n/a | 87.61% | n/a |
| `eval` | LA_0056 | 0 | 95 | n/a | 93.68% | n/a |
| `eval` | LA_0057 | 0 | 95 | n/a | 85.26% | n/a |
| `eval` | LA_0058 | 0 | 113 | n/a | 92.92% | n/a |
| `eval` | LA_0059 | 0 | 95 | n/a | 74.74% | n/a |
| `eval` | LA_0060 | 0 | 113 | n/a | 87.61% | n/a |
| `eval` | LA_0061 | 0 | 113 | n/a | 98.23% | n/a |
| `eval` | LA_0063 | 0 | 113 | n/a | 95.58% | n/a |
| `eval` | LA_0064 | 0 | 95 | n/a | 91.58% | n/a |
| `eval` | LA_0065 | 0 | 95 | n/a | 90.53% | n/a |
| `eval` | LA_0066 | 0 | 113 | n/a | 91.15% | n/a |
| `eval` | LA_0067 | 0 | 113 | n/a | 93.81% | n/a |
| `eval` | LA_0068 | 0 | 95 | n/a | 72.63% | n/a |

## Checkpoint

- Path: `Models/Iteration1/e_1_1/mel256/resnet18_frozen_mel256.pth`
- SHA-256: `2d96030622a99e90ce3719fe8249e5fe65fcd0c0d670f947ca2f939b9715de68`
- Parameters: 11242305 total (11176512 backbone, 65793 head)
- State-dict keys: 124 (124 remapped)
- Missing keys: none
- Unexpected keys: none
- Head state SHA-256: `d8e74ab89dd58f92452a80c3cab3bba2447380d4f5600ae696efa07f2b530cbc`
- Final head bias: -0.4180
- Batch-norm steps: [4725]

## Provenance

- **dev**: {'dataset': 'ASVspoof 2019 LA', 'split': 'dev', 'source': 'image', 'data_root': 'data/LA', 'protocol_path': 'data/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt', 'protocol_sha256': '2fedaebf1ba3ae1a9bb7d3ec916804ac237b4eee063a27fa92e6549f03d129c7', 'image_root': 'data/derived/e1_1_frontend/mel256', 'metadata_path': 'data/derived/e1_1_frontend/mel256/dev/metadata_dev.csv', 'metadata_sha256': '7d3cc9c722634f515e6a57875eefc6370ac47ccd5cbbe0824879594ed7f45bbe'}
- **eval**: {'dataset': 'ASVspoof 2019 LA', 'split': 'eval', 'source': 'image', 'data_root': 'data/LA', 'protocol_path': 'data/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt', 'protocol_sha256': '84fce576efff092269a0a6ad247d2e9f2f7b4d4e25a4545efb2518da92708036', 'image_root': 'data/derived/e1_1_frontend/mel256', 'metadata_path': 'data/derived/e1_1_frontend/mel256/eval/metadata_eval.csv', 'metadata_sha256': 'd8d32b8ccb6e79ad90c4317324b1dccebfc36fb72146f14020d30b284891baef'}

## Environment and command

- Git commit: `9881dbf6a5cafe8fe566f035d32b86b00a8cba21` (dirty: True)
- Device: `cuda` (NVIDIA GeForce RTX 3050 Laptop GPU)
- Python / torch / torchvision: 3.14.8 / 2.15.0+cu132 / 0.29.2+cu132
- Batch size / workers: 64 / 4

```bash
'/home/toastu/Research/Spoof Detection/.venv/bin/python' -m spoof_detection.evaluate --checkpoint '/home/toastu/Research/Spoof Detection/Models/Iteration1/e_1_1/mel256/resnet18_frozen_mel256.pth' --asvspoof dev eval --source image --data-root '/home/toastu/Research/Spoof Detection/data/LA' --image-root '/home/toastu/Research/Spoof Detection/data/derived/e1_1_frontend/mel256' --output-root '/home/toastu/Research/Spoof Detection/reports/phase1/e1_1_frontend/runs/mel256' --frontend mel256 --experiment-id E1.1 --title 'E1.1 front-end probe: mel256' --batch-size 64 --num-workers 4 --device auto
```

## Caveats

- Scores are uncalibrated; Brier and ECE are diagnostics, not evidence of calibrated probabilities.
- EER uses the ASVspoof DET-curve convention (spoof is positive). Per-attack EER reuses all bonafide clips as negatives; per-speaker EER uses same-speaker bonafides.
- Only the dev EER threshold is selected on data. The `dev_eer` threshold in an eval report is transferred from dev and is never selected on eval. Other EER thresholds are descriptive points on the DET curve, not operating thresholds.
- Minimum t-DCF and confidence intervals are not reported in E0.2.
