# Baseline evaluation: `best_audio_spoof_model_unweighted.pth`

Generated: `2026-11-09T19:54:29.202924+00:00`

## Metric contract

| Split | Attack subset | Operating thresholds | Samples | Balance |
|---|---|---|---:|---|
| `dev` | A01-A06 | fixed_0.5=0.500000, dev_eer=0.839741 | 24844 | 2548 / 22296 |
| `eval` | A07-A19 | fixed_0.5=0.500000, dev_eer=0.839741 | 71237 | 7355 / 63882 |

Threshold-free metrics: EER and AUC are computed over all spoof clips against all bonafide clips in the split. They are `n/a` when a split has only one class.

## Aggregate metrics

| Split | EER (threshold-free) | AUC (threshold-free) | Brier | ECE (10 bins) |
|---|---:|---:|---:|---:|
| `dev` | 6.16% | 0.9860 | 0.0260 | 0.0065 |
| `eval` | 11.50% | 0.9527 | 0.0699 | 0.0713 |

## Threshold metrics

| Split | Threshold | Value | Accuracy | Bonafide recall | Spoof recall | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|
| `dev` | `fixed_0.5` | 0.500000 | 96.51% | 80.18% | 98.38% | 19.82% | 1.62% |
| `dev` | `dev_eer` | 0.839741 | 93.85% | 93.84% | 93.86% | 6.16% | 6.14% |
| `eval` | `fixed_0.5` | 0.500000 | 90.21% | 83.45% | 90.99% | 16.55% | 9.01% |
| `eval` | `dev_eer` | 0.839741 | 83.84% | 94.98% | 82.55% | 5.02% | 17.45% |

## Per-attack EER

Per-attack EER uses that attack's spoof clips as positives and all bonafide clips in the split as negatives.

| Split | Attack | n_spoof | n_bonafide | EER | AUC |
|---|---|---:|---:|---:|---:|
| `dev` | A01 | 3716 | 2548 | 1.49% | 0.9987 |
| `dev` | A02 | 3716 | 2548 | 1.57% | 0.9986 |
| `dev` | A03 | 3716 | 2548 | 2.62% | 0.9964 |
| `dev` | A04 | 3716 | 2548 | 5.11% | 0.9889 |
| `dev` | A05 | 3716 | 2548 | 8.43% | 0.9748 |
| `dev` | A06 | 3716 | 2548 | 10.90% | 0.9584 |
| `eval` | A07 | 4914 | 7355 | 1.30% | 0.9991 |
| `eval` | A08 | 4914 | 7355 | 2.95% | 0.9959 |
| `eval` | A09 | 4914 | 7355 | 1.04% | 0.9993 |
| `eval` | A10 | 4914 | 7355 | 6.65% | 0.9798 |
| `eval` | A11 | 4914 | 7355 | 3.01% | 0.9953 |
| `eval` | A12 | 4914 | 7355 | 6.98% | 0.9799 |
| `eval` | A13 | 4914 | 7355 | 5.41% | 0.9876 |
| `eval` | A14 | 4914 | 7355 | 4.46% | 0.9919 |
| `eval` | A15 | 4914 | 7355 | 5.70% | 0.9851 |
| `eval` | A16 | 4914 | 7355 | 4.50% | 0.9903 |
| `eval` | A17 | 4914 | 7355 | 36.24% | 0.6907 |
| `eval` | A18 | 4914 | 7355 | 18.30% | 0.8938 |
| `eval` | A19 | 4914 | 7355 | 18.30% | 0.8964 |

## Per-speaker metrics

Per-speaker EER uses same-speaker bonafide clips as negatives. Recalls are at the fixed 0.5 threshold.

| Split | Speaker | n_spoof | n_bonafide | Spoof recall@0.5 | Bonafide recall@0.5 | EER |
|---|---|---:|---:|---:|---:|---:|
| `dev` | LA_0069 | 2484 | 154 | 98.43% | 80.52% | 6.49% |
| `dev` | LA_0070 | 1848 | 140 | 98.00% | 90.00% | 4.88% |
| `dev` | LA_0071 | 1848 | 140 | 98.59% | 69.29% | 7.96% |
| `dev` | LA_0072 | 2484 | 154 | 98.59% | 81.82% | 4.41% |
| `dev` | LA_0073 | 1848 | 140 | 98.59% | 85.71% | 3.57% |
| `dev` | LA_0074 | 2484 | 154 | 97.26% | 85.06% | 5.98% |
| `dev` | LA_0075 | 2484 | 154 | 98.39% | 94.16% | 3.09% |
| `dev` | LA_0076 | 1848 | 140 | 98.92% | 81.43% | 3.57% |
| `dev` | LA_0077 | 2484 | 154 | 98.43% | 79.22% | 5.27% |
| `dev` | LA_0078 | 2484 | 154 | 98.71% | 88.96% | 3.25% |
| `dev` | LA_0099 | 0 | 77 | n/a | 87.01% | n/a |
| `dev` | LA_0100 | 0 | 77 | n/a | 68.83% | n/a |
| `dev` | LA_0101 | 0 | 77 | n/a | 50.65% | n/a |
| `dev` | LA_0102 | 0 | 126 | n/a | 52.38% | n/a |
| `dev` | LA_0103 | 0 | 126 | n/a | 68.25% | n/a |
| `dev` | LA_0104 | 0 | 126 | n/a | 80.95% | n/a |
| `dev` | LA_0105 | 0 | 77 | n/a | 84.42% | n/a |
| `dev` | LA_0106 | 0 | 126 | n/a | 92.86% | n/a |
| `dev` | LA_0107 | 0 | 126 | n/a | 80.16% | n/a |
| `dev` | LA_0108 | 0 | 126 | n/a | 83.33% | n/a |
| `eval` | LA_0001 | 936 | 68 | 94.76% | 80.88% | 8.85% |
| `eval` | LA_0002 | 936 | 68 | 93.06% | 55.88% | 17.64% |
| `eval` | LA_0003 | 936 | 68 | 93.16% | 85.29% | 8.85% |
| `eval` | LA_0004 | 1638 | 146 | 89.56% | 92.47% | 8.91% |
| `eval` | LA_0005 | 936 | 68 | 92.95% | 97.06% | 4.77% |
| `eval` | LA_0006 | 1638 | 146 | 90.96% | 89.04% | 9.50% |
| `eval` | LA_0007 | 936 | 68 | 91.77% | 94.12% | 7.47% |
| `eval` | LA_0008 | 1638 | 146 | 89.07% | 92.47% | 10.27% |
| `eval` | LA_0009 | 1638 | 146 | 90.96% | 85.62% | 10.97% |
| `eval` | LA_0010 | 1638 | 146 | 91.45% | 78.77% | 13.69% |
| `eval` | LA_0011 | 936 | 68 | 91.03% | 85.29% | 10.28% |
| `eval` | LA_0012 | 1638 | 146 | 92.74% | 53.42% | 14.40% |
| `eval` | LA_0013 | 936 | 68 | 92.74% | 83.82% | 10.06% |
| `eval` | LA_0014 | 1638 | 146 | 91.15% | 67.12% | 14.40% |
| `eval` | LA_0015 | 936 | 68 | 93.38% | 91.18% | 7.36% |
| `eval` | LA_0016 | 1638 | 146 | 90.66% | 91.10% | 9.43% |
| `eval` | LA_0017 | 1638 | 146 | 90.66% | 80.82% | 12.92% |
| `eval` | LA_0018 | 936 | 68 | 91.03% | 92.65% | 8.79% |
| `eval` | LA_0019 | 1638 | 146 | 91.03% | 72.60% | 12.48% |
| `eval` | LA_0020 | 1638 | 146 | 91.09% | 84.25% | 10.82% |
| `eval` | LA_0021 | 936 | 68 | 89.10% | 94.12% | 7.36% |
| `eval` | LA_0022 | 1638 | 146 | 90.42% | 76.03% | 13.01% |
| `eval` | LA_0023 | 936 | 68 | 91.03% | 79.41% | 13.13% |
| `eval` | LA_0024 | 1638 | 146 | 91.64% | 86.99% | 9.50% |
| `eval` | LA_0025 | 936 | 68 | 92.52% | 94.12% | 5.88% |
| `eval` | LA_0026 | 1638 | 146 | 89.99% | 93.84% | 9.06% |
| `eval` | LA_0027 | 1638 | 146 | 89.01% | 91.10% | 10.30% |
| `eval` | LA_0028 | 936 | 68 | 91.45% | 89.71% | 8.85% |
| `eval` | LA_0029 | 1638 | 146 | 92.55% | 93.15% | 6.84% |
| `eval` | LA_0030 | 936 | 68 | 92.52% | 75.00% | 13.08% |
| `eval` | LA_0031 | 1638 | 146 | 88.95% | 86.99% | 11.65% |
| `eval` | LA_0032 | 936 | 68 | 87.71% | 77.94% | 17.64% |
| `eval` | LA_0033 | 1638 | 146 | 89.13% | 79.45% | 13.53% |
| `eval` | LA_0034 | 1638 | 146 | 91.58% | 82.88% | 10.97% |
| `eval` | LA_0035 | 1638 | 146 | 90.29% | 89.73% | 10.17% |
| `eval` | LA_0036 | 936 | 68 | 91.88% | 92.65% | 7.36% |
| `eval` | LA_0037 | 1638 | 146 | 91.94% | 90.41% | 8.26% |
| `eval` | LA_0038 | 936 | 68 | 93.80% | 42.65% | 21.77% |
| `eval` | LA_0039 | 1638 | 146 | 87.00% | 95.89% | 8.91% |
| `eval` | LA_0040 | 936 | 68 | 90.71% | 77.94% | 14.72% |
| `eval` | LA_0041 | 1638 | 146 | 90.72% | 67.81% | 15.75% |
| `eval` | LA_0042 | 1638 | 146 | 89.32% | 84.93% | 13.01% |
| `eval` | LA_0043 | 1638 | 146 | 91.64% | 65.07% | 13.07% |
| `eval` | LA_0044 | 936 | 68 | 90.60% | 72.06% | 14.72% |
| `eval` | LA_0045 | 1638 | 146 | 90.66% | 87.67% | 10.42% |
| `eval` | LA_0046 | 936 | 68 | 93.38% | 85.29% | 8.69% |
| `eval` | LA_0047 | 1638 | 146 | 90.84% | 88.36% | 9.59% |
| `eval` | LA_0048 | 936 | 68 | 92.52% | 88.24% | 10.28% |
| `eval` | LA_0049 | 0 | 95 | n/a | 90.53% | n/a |
| `eval` | LA_0050 | 0 | 113 | n/a | 78.76% | n/a |
| `eval` | LA_0051 | 0 | 113 | n/a | 80.53% | n/a |
| `eval` | LA_0052 | 0 | 95 | n/a | 95.79% | n/a |
| `eval` | LA_0053 | 0 | 113 | n/a | 91.15% | n/a |
| `eval` | LA_0054 | 0 | 95 | n/a | 87.37% | n/a |
| `eval` | LA_0055 | 0 | 113 | n/a | 75.22% | n/a |
| `eval` | LA_0056 | 0 | 95 | n/a | 90.53% | n/a |
| `eval` | LA_0057 | 0 | 95 | n/a | 66.32% | n/a |
| `eval` | LA_0058 | 0 | 113 | n/a | 88.50% | n/a |
| `eval` | LA_0059 | 0 | 95 | n/a | 72.63% | n/a |
| `eval` | LA_0060 | 0 | 113 | n/a | 77.88% | n/a |
| `eval` | LA_0061 | 0 | 113 | n/a | 92.92% | n/a |
| `eval` | LA_0063 | 0 | 113 | n/a | 89.38% | n/a |
| `eval` | LA_0064 | 0 | 95 | n/a | 80.00% | n/a |
| `eval` | LA_0065 | 0 | 95 | n/a | 93.68% | n/a |
| `eval` | LA_0066 | 0 | 113 | n/a | 92.92% | n/a |
| `eval` | LA_0067 | 0 | 113 | n/a | 91.15% | n/a |
| `eval` | LA_0068 | 0 | 95 | n/a | 66.32% | n/a |

## Checkpoint

- Path: `Models/Itteration1/best_audio_spoof_model_unweighted.pth`
- SHA-256: `abf8d9c9169d363bf9e9076b61689056a78d814c95de9634aea9acf96afc54d8`
- Parameters: 11242305 total (11176512 backbone, 65793 head)
- State-dict keys: 124 (124 remapped)
- Missing keys: none
- Unexpected keys: none

## Provenance

- **dev**: {'dataset': 'ASVspoof 2019 LA', 'split': 'dev', 'source': 'image', 'data_root': 'data/LA', 'protocol_path': 'data/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt', 'protocol_sha256': '2fedaebf1ba3ae1a9bb7d3ec916804ac237b4eee063a27fa92e6549f03d129c7', 'image_root': 'data/dataset_img', 'metadata_path': 'data/dataset_img/dev/metadata_dev.csv', 'metadata_sha256': '6097d85f173bc0e852579f94c5c6551a12267e008feb1027f3ec576e105b3b36'}
- **eval**: {'dataset': 'ASVspoof 2019 LA', 'split': 'eval', 'source': 'image', 'data_root': 'data/LA', 'protocol_path': 'data/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt', 'protocol_sha256': '84fce576efff092269a0a6ad247d2e9f2f7b4d4e25a4545efb2518da92708036', 'image_root': 'data/dataset_img', 'metadata_path': 'data/dataset_img/eval/metadata_eval.csv', 'metadata_sha256': '1ae85a9663647add8d33f7860a96be06fde48785b296de8456baf4a3a27d58a6'}

## Environment and command

- Git commit: `4bb5941f9968d1e6b9b224c4c210d4792c2e703c` (dirty: True)
- Device: `cuda` (NVIDIA GeForce RTX 3050 Laptop GPU)
- Python / torch / torchvision: 3.14.8 / 2.15.0+cu132 / 0.29.2+cu132
- Batch size / workers: 64 / 4

```bash
'/home/toastu/Research/Spoof Detection/.venv/bin/python' -m spoof_detection.evaluate --checkpoint Models/Itteration1/best_audio_spoof_model_unweighted.pth --asvspoof dev eval --source image --output-root reports/baseline --device cuda --batch-size 64 --num-workers 4 --no-progress
```

## Caveats

- Scores are uncalibrated; Brier and ECE are diagnostics, not evidence of calibrated probabilities.
- EER uses the ASVspoof DET-curve convention (spoof is positive). Per-attack EER reuses all bonafide clips as negatives; per-speaker EER uses same-speaker bonafides.
- Only the dev EER threshold is selected on data. The `dev_eer` threshold in an eval report is transferred from dev and is never selected on eval. Other EER thresholds are descriptive points on the DET curve, not operating thresholds.
- Minimum t-DCF and confidence intervals are not reported in E0.2.
