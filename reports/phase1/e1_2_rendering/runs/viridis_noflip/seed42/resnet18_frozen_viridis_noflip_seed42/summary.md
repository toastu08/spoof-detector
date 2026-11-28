# E1.2 rendering condition: viridis_noflip (seed 42): `resnet18_frozen_viridis_noflip_seed42.pth`

Experiment: `E1.2`
Generated: `2026-11-27T14:05:59.702816+00:00`

## Metric contract

| Split | Attack subset | Operating thresholds | Samples | Balance |
|---|---|---|---:|---|
| `dev` | A01-A06 | fixed_0.5=0.500000, dev_eer=0.855862 | 24844 | 2548 / 22296 |
| `eval` | A07-A19 | fixed_0.5=0.500000, dev_eer=0.855862 | 71237 | 7355 / 63882 |

Threshold-free metrics: EER and AUC are computed over all spoof clips against all bonafide clips in the split. They are `n/a` when a split has only one class.

## Aggregate metrics

| Split | EER (threshold-free) | AUC (threshold-free) | Brier | ECE (10 bins) |
|---|---:|---:|---:|---:|
| `dev` | 4.51% | 0.9911 | 0.0204 | 0.0037 |
| `eval` | 11.26% | 0.9530 | 0.0843 | 0.0905 |

## Threshold metrics

| Split | Threshold | Value | Accuracy | Bonafide recall | Spoof recall | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|
| `dev` | `fixed_0.5` | 0.500000 | 97.21% | 86.34% | 98.45% | 13.66% | 1.55% |
| `dev` | `dev_eer` | 0.855862 | 95.50% | 95.49% | 95.50% | 4.51% | 4.50% |
| `eval` | `fixed_0.5` | 0.500000 | 88.82% | 88.62% | 88.84% | 11.38% | 11.16% |
| `eval` | `dev_eer` | 0.855862 | 83.13% | 96.45% | 81.59% | 3.55% | 18.41% |

## Per-attack EER

Per-attack EER uses that attack's spoof clips as positives and all bonafide clips in the split as negatives.

| Split | Attack | n_spoof | n_bonafide | EER | AUC |
|---|---|---:|---:|---:|---:|
| `dev` | A01 | 3716 | 2548 | 1.29% | 0.9991 |
| `dev` | A02 | 3716 | 2548 | 0.94% | 0.9994 |
| `dev` | A03 | 3716 | 2548 | 2.83% | 0.9957 |
| `dev` | A04 | 3716 | 2548 | 3.97% | 0.9921 |
| `dev` | A05 | 3716 | 2548 | 5.81% | 0.9851 |
| `dev` | A06 | 3716 | 2548 | 8.21% | 0.9754 |
| `eval` | A07 | 4914 | 7355 | 0.91% | 0.9994 |
| `eval` | A08 | 4914 | 7355 | 2.16% | 0.9963 |
| `eval` | A09 | 4914 | 7355 | 0.73% | 0.9998 |
| `eval` | A10 | 4914 | 7355 | 7.90% | 0.9748 |
| `eval` | A11 | 4914 | 7355 | 1.71% | 0.9979 |
| `eval` | A12 | 4914 | 7355 | 8.31% | 0.9732 |
| `eval` | A13 | 4914 | 7355 | 10.05% | 0.9629 |
| `eval` | A14 | 4914 | 7355 | 4.05% | 0.9916 |
| `eval` | A15 | 4914 | 7355 | 5.72% | 0.9860 |
| `eval` | A16 | 4914 | 7355 | 3.46% | 0.9945 |
| `eval` | A17 | 4914 | 7355 | 37.96% | 0.6700 |
| `eval` | A18 | 4914 | 7355 | 17.93% | 0.8999 |
| `eval` | A19 | 4914 | 7355 | 13.31% | 0.9422 |

## Per-speaker metrics

Per-speaker EER uses same-speaker bonafide clips as negatives. Recalls are at the fixed 0.5 threshold.

| Split | Speaker | n_spoof | n_bonafide | Spoof recall@0.5 | Bonafide recall@0.5 | EER |
|---|---|---:|---:|---:|---:|---:|
| `dev` | LA_0069 | 2484 | 154 | 98.27% | 90.26% | 4.02% |
| `dev` | LA_0070 | 1848 | 140 | 98.43% | 88.57% | 4.28% |
| `dev` | LA_0071 | 1848 | 140 | 97.84% | 80.00% | 7.12% |
| `dev` | LA_0072 | 2484 | 154 | 99.03% | 92.86% | 3.09% |
| `dev` | LA_0073 | 1848 | 140 | 98.59% | 89.29% | 4.28% |
| `dev` | LA_0074 | 2484 | 154 | 98.07% | 92.21% | 3.90% |
| `dev` | LA_0075 | 2484 | 154 | 98.19% | 94.16% | 2.45% |
| `dev` | LA_0076 | 1848 | 140 | 98.27% | 89.29% | 3.44% |
| `dev` | LA_0077 | 2484 | 154 | 99.07% | 88.96% | 3.09% |
| `dev` | LA_0078 | 2484 | 154 | 98.59% | 93.51% | 2.67% |
| `dev` | LA_0099 | 0 | 77 | n/a | 90.91% | n/a |
| `dev` | LA_0100 | 0 | 77 | n/a | 74.03% | n/a |
| `dev` | LA_0101 | 0 | 77 | n/a | 62.34% | n/a |
| `dev` | LA_0102 | 0 | 126 | n/a | 76.19% | n/a |
| `dev` | LA_0103 | 0 | 126 | n/a | 87.30% | n/a |
| `dev` | LA_0104 | 0 | 126 | n/a | 92.06% | n/a |
| `dev` | LA_0105 | 0 | 77 | n/a | 77.92% | n/a |
| `dev` | LA_0106 | 0 | 126 | n/a | 80.95% | n/a |
| `dev` | LA_0107 | 0 | 126 | n/a | 70.63% | n/a |
| `dev` | LA_0108 | 0 | 126 | n/a | 92.06% | n/a |
| `eval` | LA_0001 | 936 | 68 | 92.52% | 85.29% | 10.28% |
| `eval` | LA_0002 | 936 | 68 | 91.24% | 73.53% | 11.76% |
| `eval` | LA_0003 | 936 | 68 | 90.92% | 92.65% | 8.79% |
| `eval` | LA_0004 | 1638 | 146 | 87.12% | 89.73% | 10.97% |
| `eval` | LA_0005 | 936 | 68 | 91.56% | 92.65% | 7.36% |
| `eval` | LA_0006 | 1638 | 146 | 87.97% | 97.95% | 8.14% |
| `eval` | LA_0007 | 936 | 68 | 88.46% | 89.71% | 11.65% |
| `eval` | LA_0008 | 1638 | 146 | 85.65% | 82.88% | 16.43% |
| `eval` | LA_0009 | 1638 | 146 | 90.48% | 88.36% | 10.97% |
| `eval` | LA_0010 | 1638 | 146 | 88.71% | 86.30% | 11.74% |
| `eval` | LA_0011 | 936 | 68 | 88.78% | 91.18% | 10.28% |
| `eval` | LA_0012 | 1638 | 146 | 90.17% | 84.25% | 11.65% |
| `eval` | LA_0013 | 936 | 68 | 90.92% | 89.71% | 10.28% |
| `eval` | LA_0014 | 1638 | 146 | 88.34% | 89.73% | 11.04% |
| `eval` | LA_0015 | 936 | 68 | 91.35% | 92.65% | 8.47% |
| `eval` | LA_0016 | 1638 | 146 | 90.54% | 91.10% | 9.50% |
| `eval` | LA_0017 | 1638 | 146 | 89.38% | 95.89% | 8.79% |
| `eval` | LA_0018 | 936 | 68 | 90.49% | 88.24% | 11.76% |
| `eval` | LA_0019 | 1638 | 146 | 86.32% | 94.52% | 10.27% |
| `eval` | LA_0020 | 1638 | 146 | 89.62% | 96.58% | 6.84% |
| `eval` | LA_0021 | 936 | 68 | 89.96% | 89.71% | 10.28% |
| `eval` | LA_0022 | 1638 | 146 | 88.22% | 89.04% | 10.97% |
| `eval` | LA_0023 | 936 | 68 | 90.17% | 83.82% | 11.65% |
| `eval` | LA_0024 | 1638 | 146 | 88.22% | 88.36% | 11.65% |
| `eval` | LA_0025 | 936 | 68 | 90.38% | 91.18% | 9.06% |
| `eval` | LA_0026 | 1638 | 146 | 87.48% | 92.47% | 11.10% |
| `eval` | LA_0027 | 1638 | 146 | 84.74% | 91.78% | 12.33% |
| `eval` | LA_0028 | 936 | 68 | 91.77% | 95.59% | 5.88% |
| `eval` | LA_0029 | 1638 | 146 | 87.36% | 92.47% | 9.74% |
| `eval` | LA_0030 | 936 | 68 | 88.03% | 75.00% | 14.72% |
| `eval` | LA_0031 | 1638 | 146 | 87.12% | 95.21% | 10.11% |
| `eval` | LA_0032 | 936 | 68 | 84.40% | 76.47% | 17.85% |
| `eval` | LA_0033 | 1638 | 146 | 89.38% | 91.10% | 9.59% |
| `eval` | LA_0034 | 1638 | 146 | 88.89% | 89.73% | 10.94% |
| `eval` | LA_0035 | 1638 | 146 | 88.58% | 92.47% | 8.88% |
| `eval` | LA_0036 | 936 | 68 | 89.74% | 80.88% | 12.08% |
| `eval` | LA_0037 | 1638 | 146 | 89.19% | 95.21% | 8.91% |
| `eval` | LA_0038 | 936 | 68 | 92.20% | 47.06% | 16.47% |
| `eval` | LA_0039 | 1638 | 146 | 85.71% | 97.26% | 8.94% |
| `eval` | LA_0040 | 936 | 68 | 90.60% | 83.82% | 11.76% |
| `eval` | LA_0041 | 1638 | 146 | 87.67% | 88.36% | 11.65% |
| `eval` | LA_0042 | 1638 | 146 | 88.22% | 84.93% | 12.89% |
| `eval` | LA_0043 | 1638 | 146 | 89.01% | 85.62% | 12.33% |
| `eval` | LA_0044 | 936 | 68 | 90.06% | 85.29% | 11.49% |
| `eval` | LA_0045 | 1638 | 146 | 88.71% | 94.52% | 8.23% |
| `eval` | LA_0046 | 936 | 68 | 91.99% | 92.65% | 7.42% |
| `eval` | LA_0047 | 1638 | 146 | 89.87% | 91.10% | 9.62% |
| `eval` | LA_0048 | 936 | 68 | 88.35% | 88.24% | 11.76% |
| `eval` | LA_0049 | 0 | 95 | n/a | 78.95% | n/a |
| `eval` | LA_0050 | 0 | 113 | n/a | 81.42% | n/a |
| `eval` | LA_0051 | 0 | 113 | n/a | 86.73% | n/a |
| `eval` | LA_0052 | 0 | 95 | n/a | 95.79% | n/a |
| `eval` | LA_0053 | 0 | 113 | n/a | 83.19% | n/a |
| `eval` | LA_0054 | 0 | 95 | n/a | 86.32% | n/a |
| `eval` | LA_0055 | 0 | 113 | n/a | 84.96% | n/a |
| `eval` | LA_0056 | 0 | 95 | n/a | 91.58% | n/a |
| `eval` | LA_0057 | 0 | 95 | n/a | 84.21% | n/a |
| `eval` | LA_0058 | 0 | 113 | n/a | 92.04% | n/a |
| `eval` | LA_0059 | 0 | 95 | n/a | 75.79% | n/a |
| `eval` | LA_0060 | 0 | 113 | n/a | 88.50% | n/a |
| `eval` | LA_0061 | 0 | 113 | n/a | 91.15% | n/a |
| `eval` | LA_0063 | 0 | 113 | n/a | 92.04% | n/a |
| `eval` | LA_0064 | 0 | 95 | n/a | 78.95% | n/a |
| `eval` | LA_0065 | 0 | 95 | n/a | 92.63% | n/a |
| `eval` | LA_0066 | 0 | 113 | n/a | 92.92% | n/a |
| `eval` | LA_0067 | 0 | 113 | n/a | 84.07% | n/a |
| `eval` | LA_0068 | 0 | 95 | n/a | 80.00% | n/a |

## Checkpoint

- Path: `Models/Iteration1/e_1_2/viridis_noflip/seed42/resnet18_frozen_viridis_noflip_seed42.pth`
- SHA-256: `65b494985b63a17c53c2485248f327846e680f52491ca47b08cdd9441febc3df`
- Parameters: 11242305 total (11176512 backbone, 65793 head)
- State-dict keys: 124 (124 remapped)
- Missing keys: none
- Unexpected keys: none
- Head state SHA-256: `34d973c7a7f3191aae687b9d95abd7ed3b4e0746f843998127d966209b8a25a8`
- Final head bias: -0.5032
- Batch-norm steps: [6300]

## Provenance

- **dev**: {'dataset': 'ASVspoof 2019 LA', 'split': 'dev', 'source': 'image', 'data_root': 'data/LA', 'protocol_path': 'data/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt', 'protocol_sha256': '2fedaebf1ba3ae1a9bb7d3ec916804ac237b4eee063a27fa92e6549f03d129c7', 'image_root': 'data/derived/e1_2_rendering/viridis_noflip', 'metadata_path': 'data/derived/e1_2_rendering/viridis_noflip/dev/metadata_dev.csv', 'metadata_sha256': 'a31235ee12a6cf21b46772d2052845ac9ba297f143f4e7589787639b094b88c9'}
- **eval**: {'dataset': 'ASVspoof 2019 LA', 'split': 'eval', 'source': 'image', 'data_root': 'data/LA', 'protocol_path': 'data/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt', 'protocol_sha256': '84fce576efff092269a0a6ad247d2e9f2f7b4d4e25a4545efb2518da92708036', 'image_root': 'data/derived/e1_2_rendering/viridis_noflip', 'metadata_path': 'data/derived/e1_2_rendering/viridis_noflip/eval/metadata_eval.csv', 'metadata_sha256': '1b936a883b2ca04df9ed5af4a923d5202f6f51e03650befb50b35f42443b685e'}

## Environment and command

- Git commit: `4fb53d685d23dc9f1040cf4744479bc4bbabd033` (dirty: True)
- Device: `cuda` (NVIDIA GeForce RTX 3050 Laptop GPU)
- Python / torch / torchvision: 3.14.8 / 2.15.0+cu132 / 0.29.2+cu132
- Batch size / workers: 64 / 4

```bash
'/home/toastu/Research/Spoof Detection/.venv/bin/python' -m spoof_detection.evaluate --checkpoint '/home/toastu/Research/Spoof Detection/Models/Iteration1/e_1_2/viridis_noflip/seed42/resnet18_frozen_viridis_noflip_seed42.pth' --asvspoof dev eval --source image --data-root '/home/toastu/Research/Spoof Detection/data/LA' --image-root '/home/toastu/Research/Spoof Detection/data/derived/e1_2_rendering/viridis_noflip' --output-root '/home/toastu/Research/Spoof Detection/reports/phase1/e1_2_rendering/runs/viridis_noflip/seed42' --frontend viridis_noflip --experiment-id E1.2 --title 'E1.2 rendering condition: viridis_noflip (seed 42)' --batch-size 64 --num-workers 4 --device auto
```

## Caveats

- Scores are uncalibrated; Brier and ECE are diagnostics, not evidence of calibrated probabilities.
- EER uses the ASVspoof DET-curve convention (spoof is positive). Per-attack EER reuses all bonafide clips as negatives; per-speaker EER uses same-speaker bonafides.
- Only the dev EER threshold is selected on data. The `dev_eer` threshold in an eval report is transferred from dev and is never selected on eval. Other EER thresholds are descriptive points on the DET curve, not operating thresholds.
- Minimum t-DCF and confidence intervals are not reported in E0.2.
