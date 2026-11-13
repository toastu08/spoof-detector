# Frozen two-checkpoint baseline: `best_audio_spoof_model_weighted.pth`

Experiment: `E0.3`
Generated: `2026-11-13T21:41:28.020425+00:00`

## Metric contract

| Split | Attack subset | Operating thresholds | Samples | Balance |
|---|---|---|---:|---|
| `dev` | A01-A06 | fixed_0.5=0.500000, dev_eer=0.752020 | 24844 | 2548 / 22296 |
| `eval` | A07-A19 | fixed_0.5=0.500000, dev_eer=0.752020 | 71237 | 7355 / 63882 |

Threshold-free metrics: EER and AUC are computed over all spoof clips against all bonafide clips in the split. They are `n/a` when a split has only one class.

## Aggregate metrics

| Split | EER (threshold-free) | AUC (threshold-free) | Brier | ECE (10 bins) |
|---|---:|---:|---:|---:|
| `dev` | 5.42% | 0.9881 | 0.0261 | 0.0171 |
| `eval` | 11.13% | 0.9558 | 0.0856 | 0.0999 |

## Threshold metrics

| Split | Threshold | Value | Accuracy | Bonafide recall | Spoof recall | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|
| `dev` | `fixed_0.5` | 0.500000 | 96.38% | 87.99% | 97.34% | 12.01% | 2.66% |
| `dev` | `dev_eer` | 0.752020 | 94.59% | 94.58% | 94.59% | 5.42% | 5.41% |
| `eval` | `fixed_0.5` | 0.500000 | 88.05% | 90.58% | 87.76% | 9.42% | 12.24% |
| `eval` | `dev_eer` | 0.752020 | 83.91% | 95.32% | 82.60% | 4.68% | 17.40% |

## Per-attack EER

Per-attack EER uses that attack's spoof clips as positives and all bonafide clips in the split as negatives.

| Split | Attack | n_spoof | n_bonafide | EER | AUC |
|---|---|---:|---:|---:|---:|
| `dev` | A01 | 3716 | 2548 | 1.64% | 0.9986 |
| `dev` | A02 | 3716 | 2548 | 1.92% | 0.9982 |
| `dev` | A03 | 3716 | 2548 | 2.51% | 0.9967 |
| `dev` | A04 | 3716 | 2548 | 4.39% | 0.9919 |
| `dev` | A05 | 3716 | 2548 | 7.50% | 0.9790 |
| `dev` | A06 | 3716 | 2548 | 10.12% | 0.9644 |
| `eval` | A07 | 4914 | 7355 | 1.26% | 0.9992 |
| `eval` | A08 | 4914 | 7355 | 2.83% | 0.9960 |
| `eval` | A09 | 4914 | 7355 | 1.10% | 0.9995 |
| `eval` | A10 | 4914 | 7355 | 5.84% | 0.9847 |
| `eval` | A11 | 4914 | 7355 | 2.93% | 0.9962 |
| `eval` | A12 | 4914 | 7355 | 5.62% | 0.9851 |
| `eval` | A13 | 4914 | 7355 | 5.00% | 0.9892 |
| `eval` | A14 | 4914 | 7355 | 4.42% | 0.9917 |
| `eval` | A15 | 4914 | 7355 | 5.11% | 0.9879 |
| `eval` | A16 | 4914 | 7355 | 3.97% | 0.9932 |
| `eval` | A17 | 4914 | 7355 | 34.67% | 0.7104 |
| `eval` | A18 | 4914 | 7355 | 18.50% | 0.8918 |
| `eval` | A19 | 4914 | 7355 | 17.99% | 0.8999 |

## Per-speaker metrics

Per-speaker EER uses same-speaker bonafide clips as negatives. Recalls are at the fixed 0.5 threshold.

| Split | Speaker | n_spoof | n_bonafide | Spoof recall@0.5 | Bonafide recall@0.5 | EER |
|---|---|---:|---:|---:|---:|---:|
| `dev` | LA_0069 | 2484 | 154 | 97.30% | 87.01% | 5.80% |
| `dev` | LA_0070 | 1848 | 140 | 95.56% | 95.71% | 4.28% |
| `dev` | LA_0071 | 1848 | 140 | 97.56% | 84.29% | 7.14% |
| `dev` | LA_0072 | 2484 | 154 | 97.83% | 94.81% | 3.82% |
| `dev` | LA_0073 | 1848 | 140 | 97.78% | 94.29% | 2.86% |
| `dev` | LA_0074 | 2484 | 154 | 96.01% | 91.56% | 5.31% |
| `dev` | LA_0075 | 2484 | 154 | 97.26% | 97.40% | 2.61% |
| `dev` | LA_0076 | 1848 | 140 | 98.38% | 90.00% | 3.46% |
| `dev` | LA_0077 | 2484 | 154 | 97.79% | 91.56% | 4.59% |
| `dev` | LA_0078 | 2484 | 154 | 97.87% | 92.86% | 3.25% |
| `dev` | LA_0099 | 0 | 77 | n/a | 93.51% | n/a |
| `dev` | LA_0100 | 0 | 77 | n/a | 79.22% | n/a |
| `dev` | LA_0101 | 0 | 77 | n/a | 63.64% | n/a |
| `dev` | LA_0102 | 0 | 126 | n/a | 63.49% | n/a |
| `dev` | LA_0103 | 0 | 126 | n/a | 77.78% | n/a |
| `dev` | LA_0104 | 0 | 126 | n/a | 90.48% | n/a |
| `dev` | LA_0105 | 0 | 77 | n/a | 89.61% | n/a |
| `dev` | LA_0106 | 0 | 126 | n/a | 91.27% | n/a |
| `dev` | LA_0107 | 0 | 126 | n/a | 90.48% | n/a |
| `dev` | LA_0108 | 0 | 126 | n/a | 83.33% | n/a |
| `eval` | LA_0001 | 936 | 68 | 92.31% | 89.71% | 10.22% |
| `eval` | LA_0002 | 936 | 68 | 90.71% | 64.71% | 17.64% |
| `eval` | LA_0003 | 936 | 68 | 90.49% | 89.71% | 10.28% |
| `eval` | LA_0004 | 1638 | 146 | 86.94% | 97.95% | 8.23% |
| `eval` | LA_0005 | 936 | 68 | 90.71% | 98.53% | 7.10% |
| `eval` | LA_0006 | 1638 | 146 | 87.97% | 95.89% | 8.26% |
| `eval` | LA_0007 | 936 | 68 | 88.78% | 97.06% | 8.85% |
| `eval` | LA_0008 | 1638 | 146 | 85.23% | 95.21% | 10.42% |
| `eval` | LA_0009 | 1638 | 146 | 87.73% | 92.47% | 10.27% |
| `eval` | LA_0010 | 1638 | 146 | 86.45% | 87.67% | 13.01% |
| `eval` | LA_0011 | 936 | 68 | 88.03% | 94.12% | 10.17% |
| `eval` | LA_0012 | 1638 | 146 | 89.74% | 75.34% | 14.46% |
| `eval` | LA_0013 | 936 | 68 | 89.85% | 88.24% | 10.33% |
| `eval` | LA_0014 | 1638 | 146 | 87.48% | 84.93% | 13.56% |
| `eval` | LA_0015 | 936 | 68 | 90.17% | 94.12% | 7.36% |
| `eval` | LA_0016 | 1638 | 146 | 87.30% | 93.84% | 9.03% |
| `eval` | LA_0017 | 1638 | 146 | 87.73% | 89.73% | 11.74% |
| `eval` | LA_0018 | 936 | 68 | 89.10% | 95.59% | 8.85% |
| `eval` | LA_0019 | 1638 | 146 | 87.55% | 91.78% | 11.80% |
| `eval` | LA_0020 | 1638 | 146 | 87.85% | 90.41% | 10.79% |
| `eval` | LA_0021 | 936 | 68 | 85.15% | 95.59% | 8.85% |
| `eval` | LA_0022 | 1638 | 146 | 87.85% | 84.25% | 12.95% |
| `eval` | LA_0023 | 936 | 68 | 88.78% | 88.24% | 11.60% |
| `eval` | LA_0024 | 1638 | 146 | 89.19% | 92.47% | 10.27% |
| `eval` | LA_0025 | 936 | 68 | 88.57% | 98.53% | 9.11% |
| `eval` | LA_0026 | 1638 | 146 | 87.36% | 94.52% | 8.23% |
| `eval` | LA_0027 | 1638 | 146 | 85.35% | 93.84% | 10.79% |
| `eval` | LA_0028 | 936 | 68 | 87.93% | 92.65% | 10.22% |
| `eval` | LA_0029 | 1638 | 146 | 89.32% | 95.21% | 7.52% |
| `eval` | LA_0030 | 936 | 68 | 89.42% | 85.29% | 11.76% |
| `eval` | LA_0031 | 1638 | 146 | 85.59% | 93.15% | 11.65% |
| `eval` | LA_0032 | 936 | 68 | 82.91% | 86.76% | 14.72% |
| `eval` | LA_0033 | 1638 | 146 | 85.47% | 90.41% | 12.86% |
| `eval` | LA_0034 | 1638 | 146 | 87.97% | 93.84% | 10.14% |
| `eval` | LA_0035 | 1638 | 146 | 87.12% | 94.52% | 10.27% |
| `eval` | LA_0036 | 936 | 68 | 87.93% | 95.59% | 9.95% |
| `eval` | LA_0037 | 1638 | 146 | 89.26% | 95.21% | 7.52% |
| `eval` | LA_0038 | 936 | 68 | 90.06% | 47.06% | 22.41% |
| `eval` | LA_0039 | 1638 | 146 | 82.66% | 97.26% | 7.52% |
| `eval` | LA_0040 | 936 | 68 | 86.97% | 89.71% | 11.81% |
| `eval` | LA_0041 | 1638 | 146 | 87.79% | 81.51% | 14.52% |
| `eval` | LA_0042 | 1638 | 146 | 86.14% | 90.41% | 12.18% |
| `eval` | LA_0043 | 1638 | 146 | 88.34% | 84.25% | 12.89% |
| `eval` | LA_0044 | 936 | 68 | 87.07% | 82.35% | 16.26% |
| `eval` | LA_0045 | 1638 | 146 | 88.16% | 93.84% | 9.59% |
| `eval` | LA_0046 | 936 | 68 | 90.38% | 92.65% | 8.85% |
| `eval` | LA_0047 | 1638 | 146 | 87.42% | 93.84% | 9.59% |
| `eval` | LA_0048 | 936 | 68 | 89.53% | 92.65% | 8.85% |
| `eval` | LA_0049 | 0 | 95 | n/a | 95.79% | n/a |
| `eval` | LA_0050 | 0 | 113 | n/a | 84.07% | n/a |
| `eval` | LA_0051 | 0 | 113 | n/a | 84.96% | n/a |
| `eval` | LA_0052 | 0 | 95 | n/a | 97.89% | n/a |
| `eval` | LA_0053 | 0 | 113 | n/a | 98.23% | n/a |
| `eval` | LA_0054 | 0 | 95 | n/a | 92.63% | n/a |
| `eval` | LA_0055 | 0 | 113 | n/a | 82.30% | n/a |
| `eval` | LA_0056 | 0 | 95 | n/a | 95.79% | n/a |
| `eval` | LA_0057 | 0 | 95 | n/a | 85.26% | n/a |
| `eval` | LA_0058 | 0 | 113 | n/a | 93.81% | n/a |
| `eval` | LA_0059 | 0 | 95 | n/a | 83.16% | n/a |
| `eval` | LA_0060 | 0 | 113 | n/a | 84.96% | n/a |
| `eval` | LA_0061 | 0 | 113 | n/a | 97.35% | n/a |
| `eval` | LA_0063 | 0 | 113 | n/a | 94.69% | n/a |
| `eval` | LA_0064 | 0 | 95 | n/a | 92.63% | n/a |
| `eval` | LA_0065 | 0 | 95 | n/a | 95.79% | n/a |
| `eval` | LA_0066 | 0 | 113 | n/a | 95.58% | n/a |
| `eval` | LA_0067 | 0 | 113 | n/a | 94.69% | n/a |
| `eval` | LA_0068 | 0 | 95 | n/a | 73.68% | n/a |

## Checkpoint

- Path: `Models/Itteration1/best_audio_spoof_model_weighted.pth`
- SHA-256: `214e2c813c4f48602ddf124590daf20a008f18385a5339029736a6c16837eb93`
- Parameters: 11242305 total (11176512 backbone, 65793 head)
- State-dict keys: 124 (124 remapped)
- Missing keys: none
- Unexpected keys: none
- Head state SHA-256: `48c383336ddd257763adbbf00c94bd0a9047f4597375b7dfa3f683cf9422382a`
- Final head bias: -0.5001
- Batch-norm steps: [6930]

## Provenance

- **dev**: {'dataset': 'ASVspoof 2019 LA', 'split': 'dev', 'source': 'image', 'data_root': 'data/LA', 'protocol_path': 'data/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt', 'protocol_sha256': '2fedaebf1ba3ae1a9bb7d3ec916804ac237b4eee063a27fa92e6549f03d129c7', 'image_root': 'data/dataset_img', 'metadata_path': 'data/dataset_img/dev/metadata_dev.csv', 'metadata_sha256': '6097d85f173bc0e852579f94c5c6551a12267e008feb1027f3ec576e105b3b36'}
- **eval**: {'dataset': 'ASVspoof 2019 LA', 'split': 'eval', 'source': 'image', 'data_root': 'data/LA', 'protocol_path': 'data/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt', 'protocol_sha256': '84fce576efff092269a0a6ad247d2e9f2f7b4d4e25a4545efb2518da92708036', 'image_root': 'data/dataset_img', 'metadata_path': 'data/dataset_img/eval/metadata_eval.csv', 'metadata_sha256': '1ae85a9663647add8d33f7860a96be06fde48785b296de8456baf4a3a27d58a6'}

## Environment and command

- Git commit: `67a81104592090857ffd8b85de87edbc4ade9e07` (dirty: False)
- Device: `cuda` (NVIDIA GeForce RTX 3050 Laptop GPU)
- Python / torch / torchvision: 3.14.8 / 2.15.0+cu132 / 0.29.2+cu132
- Batch size / workers: 64 / 4

```bash
'/home/toastu/Research/Spoof Detection/.venv/bin/python' -m spoof_detection.evaluate --checkpoint Models/Itteration1/best_audio_spoof_model_unweighted.pth --checkpoint Models/Itteration1/best_audio_spoof_model_weighted.pth --asvspoof dev eval --source image --output-root reports/baseline --experiment-id E0.3 --title 'Frozen two-checkpoint baseline'
```

## Caveats

- Scores are uncalibrated; Brier and ECE are diagnostics, not evidence of calibrated probabilities.
- EER uses the ASVspoof DET-curve convention (spoof is positive). Per-attack EER reuses all bonafide clips as negatives; per-speaker EER uses same-speaker bonafides.
- Only the dev EER threshold is selected on data. The `dev_eer` threshold in an eval report is transferred from dev and is never selected on eval. Other EER thresholds are descriptive points on the DET curve, not operating thresholds.
- Minimum t-DCF and confidence intervals are not reported in E0.2.
