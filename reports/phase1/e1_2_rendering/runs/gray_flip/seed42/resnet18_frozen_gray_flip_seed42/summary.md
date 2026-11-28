# E1.2 rendering condition: gray_flip (seed 42): `resnet18_frozen_gray_flip_seed42.pth`

Experiment: `E1.2`
Generated: `2026-11-27T13:59:23.558511+00:00`

## Metric contract

| Split | Attack subset | Operating thresholds | Samples | Balance |
|---|---|---|---:|---|
| `dev` | A01-A06 | fixed_0.5=0.500000, dev_eer=0.899421 | 24844 | 2548 / 22296 |
| `eval` | A07-A19 | fixed_0.5=0.500000, dev_eer=0.899421 | 71237 | 7355 / 63882 |

Threshold-free metrics: EER and AUC are computed over all spoof clips against all bonafide clips in the split. They are `n/a` when a split has only one class.

## Aggregate metrics

| Split | EER (threshold-free) | AUC (threshold-free) | Brier | ECE (10 bins) |
|---|---:|---:|---:|---:|
| `dev` | 5.53% | 0.9874 | 0.0238 | 0.0036 |
| `eval` | 12.44% | 0.9420 | 0.0868 | 0.0888 |

## Threshold metrics

| Split | Threshold | Value | Accuracy | Bonafide recall | Spoof recall | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|
| `dev` | `fixed_0.5` | 0.500000 | 96.80% | 81.12% | 98.59% | 18.88% | 1.41% |
| `dev` | `dev_eer` | 0.899421 | 94.47% | 94.47% | 94.47% | 5.53% | 5.53% |
| `eval` | `fixed_0.5` | 0.500000 | 88.34% | 85.41% | 88.68% | 14.59% | 11.32% |
| `eval` | `dev_eer` | 0.899421 | 80.68% | 95.76% | 78.94% | 4.24% | 21.06% |

## Per-attack EER

Per-attack EER uses that attack's spoof clips as positives and all bonafide clips in the split as negatives.

| Split | Attack | n_spoof | n_bonafide | EER | AUC |
|---|---|---:|---:|---:|---:|
| `dev` | A01 | 3716 | 2548 | 2.08% | 0.9979 |
| `dev` | A02 | 3716 | 2548 | 2.16% | 0.9978 |
| `dev` | A03 | 3716 | 2548 | 3.73% | 0.9944 |
| `dev` | A04 | 3716 | 2548 | 4.20% | 0.9919 |
| `dev` | A05 | 3716 | 2548 | 7.10% | 0.9804 |
| `dev` | A06 | 3716 | 2548 | 10.12% | 0.9618 |
| `eval` | A07 | 4914 | 7355 | 1.12% | 0.9991 |
| `eval` | A08 | 4914 | 7355 | 2.89% | 0.9952 |
| `eval` | A09 | 4914 | 7355 | 1.12% | 0.9992 |
| `eval` | A10 | 4914 | 7355 | 9.24% | 0.9654 |
| `eval` | A11 | 4914 | 7355 | 2.83% | 0.9952 |
| `eval` | A12 | 4914 | 7355 | 9.01% | 0.9652 |
| `eval` | A13 | 4914 | 7355 | 15.40% | 0.9186 |
| `eval` | A14 | 4914 | 7355 | 3.24% | 0.9943 |
| `eval` | A15 | 4914 | 7355 | 6.12% | 0.9831 |
| `eval` | A16 | 4914 | 7355 | 3.83% | 0.9920 |
| `eval` | A17 | 4914 | 7355 | 39.82% | 0.6420 |
| `eval` | A18 | 4914 | 7355 | 18.79% | 0.8907 |
| `eval` | A19 | 4914 | 7355 | 17.32% | 0.9056 |

## Per-speaker metrics

Per-speaker EER uses same-speaker bonafide clips as negatives. Recalls are at the fixed 0.5 threshold.

| Split | Speaker | n_spoof | n_bonafide | Spoof recall@0.5 | Bonafide recall@0.5 | EER |
|---|---|---:|---:|---:|---:|---:|
| `dev` | LA_0069 | 2484 | 154 | 98.79% | 88.31% | 3.25% |
| `dev` | LA_0070 | 1848 | 140 | 98.43% | 87.14% | 4.28% |
| `dev` | LA_0071 | 1848 | 140 | 98.97% | 67.86% | 7.14% |
| `dev` | LA_0072 | 2484 | 154 | 98.55% | 76.62% | 4.55% |
| `dev` | LA_0073 | 1848 | 140 | 98.38% | 90.00% | 3.41% |
| `dev` | LA_0074 | 2484 | 154 | 97.30% | 95.45% | 3.25% |
| `dev` | LA_0075 | 2484 | 154 | 98.59% | 92.21% | 3.13% |
| `dev` | LA_0076 | 1848 | 140 | 98.59% | 85.00% | 4.28% |
| `dev` | LA_0077 | 2484 | 154 | 98.99% | 72.73% | 7.13% |
| `dev` | LA_0078 | 2484 | 154 | 99.32% | 87.01% | 3.90% |
| `dev` | LA_0099 | 0 | 77 | n/a | 92.21% | n/a |
| `dev` | LA_0100 | 0 | 77 | n/a | 63.64% | n/a |
| `dev` | LA_0101 | 0 | 77 | n/a | 63.64% | n/a |
| `dev` | LA_0102 | 0 | 126 | n/a | 50.00% | n/a |
| `dev` | LA_0103 | 0 | 126 | n/a | 76.98% | n/a |
| `dev` | LA_0104 | 0 | 126 | n/a | 78.57% | n/a |
| `dev` | LA_0105 | 0 | 77 | n/a | 84.42% | n/a |
| `dev` | LA_0106 | 0 | 126 | n/a | 93.65% | n/a |
| `dev` | LA_0107 | 0 | 126 | n/a | 74.60% | n/a |
| `dev` | LA_0108 | 0 | 126 | n/a | 88.10% | n/a |
| `eval` | LA_0001 | 936 | 68 | 92.31% | 79.41% | 11.76% |
| `eval` | LA_0002 | 936 | 68 | 90.49% | 64.71% | 19.17% |
| `eval` | LA_0003 | 936 | 68 | 91.77% | 85.29% | 10.38% |
| `eval` | LA_0004 | 1638 | 146 | 88.83% | 95.21% | 8.94% |
| `eval` | LA_0005 | 936 | 68 | 89.21% | 95.59% | 9.06% |
| `eval` | LA_0006 | 1638 | 146 | 87.48% | 92.47% | 9.71% |
| `eval` | LA_0007 | 936 | 68 | 89.00% | 83.82% | 13.24% |
| `eval` | LA_0008 | 1638 | 146 | 87.85% | 86.99% | 12.42% |
| `eval` | LA_0009 | 1638 | 146 | 89.44% | 87.67% | 11.65% |
| `eval` | LA_0010 | 1638 | 146 | 87.61% | 78.77% | 15.13% |
| `eval` | LA_0011 | 936 | 68 | 85.26% | 91.18% | 11.49% |
| `eval` | LA_0012 | 1638 | 146 | 90.23% | 73.97% | 15.75% |
| `eval` | LA_0013 | 936 | 68 | 90.81% | 85.29% | 11.60% |
| `eval` | LA_0014 | 1638 | 146 | 89.87% | 71.23% | 15.20% |
| `eval` | LA_0015 | 936 | 68 | 88.25% | 86.76% | 11.92% |
| `eval` | LA_0016 | 1638 | 146 | 88.34% | 85.62% | 12.33% |
| `eval` | LA_0017 | 1638 | 146 | 87.24% | 82.88% | 14.98% |
| `eval` | LA_0018 | 936 | 68 | 89.00% | 92.65% | 10.28% |
| `eval` | LA_0019 | 1638 | 146 | 89.56% | 80.82% | 12.30% |
| `eval` | LA_0020 | 1638 | 146 | 88.52% | 86.30% | 12.89% |
| `eval` | LA_0021 | 936 | 68 | 88.89% | 86.76% | 11.87% |
| `eval` | LA_0022 | 1638 | 146 | 89.68% | 84.93% | 12.24% |
| `eval` | LA_0023 | 936 | 68 | 88.35% | 91.18% | 10.28% |
| `eval` | LA_0024 | 1638 | 146 | 90.96% | 87.67% | 9.46% |
| `eval` | LA_0025 | 936 | 68 | 88.14% | 94.12% | 8.85% |
| `eval` | LA_0026 | 1638 | 146 | 86.69% | 91.78% | 12.21% |
| `eval` | LA_0027 | 1638 | 146 | 85.96% | 95.89% | 9.59% |
| `eval` | LA_0028 | 936 | 68 | 87.71% | 95.59% | 8.85% |
| `eval` | LA_0029 | 1638 | 146 | 89.56% | 94.52% | 7.67% |
| `eval` | LA_0030 | 936 | 68 | 89.42% | 83.82% | 11.97% |
| `eval` | LA_0031 | 1638 | 146 | 86.08% | 91.10% | 10.97% |
| `eval` | LA_0032 | 936 | 68 | 84.51% | 92.65% | 12.03% |
| `eval` | LA_0033 | 1638 | 146 | 87.18% | 83.56% | 14.43% |
| `eval` | LA_0034 | 1638 | 146 | 91.33% | 89.73% | 9.56% |
| `eval` | LA_0035 | 1638 | 146 | 88.83% | 91.78% | 10.20% |
| `eval` | LA_0036 | 936 | 68 | 88.57% | 89.71% | 11.44% |
| `eval` | LA_0037 | 1638 | 146 | 88.16% | 94.52% | 9.65% |
| `eval` | LA_0038 | 936 | 68 | 89.74% | 42.65% | 26.54% |
| `eval` | LA_0039 | 1638 | 146 | 86.08% | 91.78% | 11.65% |
| `eval` | LA_0040 | 936 | 68 | 89.21% | 72.06% | 15.05% |
| `eval` | LA_0041 | 1638 | 146 | 88.10% | 78.08% | 13.75% |
| `eval` | LA_0042 | 1638 | 146 | 88.83% | 78.77% | 14.40% |
| `eval` | LA_0043 | 1638 | 146 | 88.77% | 71.92% | 15.01% |
| `eval` | LA_0044 | 936 | 68 | 90.06% | 75.00% | 14.88% |
| `eval` | LA_0045 | 1638 | 146 | 90.23% | 86.99% | 10.39% |
| `eval` | LA_0046 | 936 | 68 | 89.21% | 94.12% | 7.63% |
| `eval` | LA_0047 | 1638 | 146 | 90.72% | 91.78% | 9.00% |
| `eval` | LA_0048 | 936 | 68 | 86.00% | 88.24% | 11.87% |
| `eval` | LA_0049 | 0 | 95 | n/a | 83.16% | n/a |
| `eval` | LA_0050 | 0 | 113 | n/a | 85.84% | n/a |
| `eval` | LA_0051 | 0 | 113 | n/a | 76.11% | n/a |
| `eval` | LA_0052 | 0 | 95 | n/a | 98.95% | n/a |
| `eval` | LA_0053 | 0 | 113 | n/a | 95.58% | n/a |
| `eval` | LA_0054 | 0 | 95 | n/a | 83.16% | n/a |
| `eval` | LA_0055 | 0 | 113 | n/a | 85.84% | n/a |
| `eval` | LA_0056 | 0 | 95 | n/a | 91.58% | n/a |
| `eval` | LA_0057 | 0 | 95 | n/a | 70.53% | n/a |
| `eval` | LA_0058 | 0 | 113 | n/a | 81.42% | n/a |
| `eval` | LA_0059 | 0 | 95 | n/a | 87.37% | n/a |
| `eval` | LA_0060 | 0 | 113 | n/a | 73.45% | n/a |
| `eval` | LA_0061 | 0 | 113 | n/a | 97.35% | n/a |
| `eval` | LA_0063 | 0 | 113 | n/a | 88.50% | n/a |
| `eval` | LA_0064 | 0 | 95 | n/a | 60.00% | n/a |
| `eval` | LA_0065 | 0 | 95 | n/a | 94.74% | n/a |
| `eval` | LA_0066 | 0 | 113 | n/a | 83.19% | n/a |
| `eval` | LA_0067 | 0 | 113 | n/a | 92.04% | n/a |
| `eval` | LA_0068 | 0 | 95 | n/a | 77.89% | n/a |

## Checkpoint

- Path: `Models/Iteration1/e_1_2/gray_flip/seed42/resnet18_frozen_gray_flip_seed42.pth`
- SHA-256: `8ad060fbc063c6719f391d5954589f090f80c919ddcab7080850b6b405e4efdb`
- Parameters: 11242305 total (11176512 backbone, 65793 head)
- State-dict keys: 124 (124 remapped)
- Missing keys: none
- Unexpected keys: none
- Head state SHA-256: `9f8d2f31184dc3c195ef8b020d44dc4baf47173a00c0d2bbdf3a4a3414badbcb`
- Final head bias: -0.3889
- Batch-norm steps: [5670]

## Provenance

- **dev**: {'dataset': 'ASVspoof 2019 LA', 'split': 'dev', 'source': 'image', 'data_root': 'data/LA', 'protocol_path': 'data/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt', 'protocol_sha256': '2fedaebf1ba3ae1a9bb7d3ec916804ac237b4eee063a27fa92e6549f03d129c7', 'image_root': 'data/derived/e1_2_rendering/gray_flip', 'metadata_path': 'data/derived/e1_2_rendering/gray_flip/dev/metadata_dev.csv', 'metadata_sha256': '15148d2bdb0c666ee94dbbb2aed50cd5dddb1db4f27dc957f59855954e926f0e'}
- **eval**: {'dataset': 'ASVspoof 2019 LA', 'split': 'eval', 'source': 'image', 'data_root': 'data/LA', 'protocol_path': 'data/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt', 'protocol_sha256': '84fce576efff092269a0a6ad247d2e9f2f7b4d4e25a4545efb2518da92708036', 'image_root': 'data/derived/e1_2_rendering/gray_flip', 'metadata_path': 'data/derived/e1_2_rendering/gray_flip/eval/metadata_eval.csv', 'metadata_sha256': '4b085dd52fe4ee55ea482b752b19ef1d4886fb2182d8b05a5d57e46668fc6920'}

## Environment and command

- Git commit: `4fb53d685d23dc9f1040cf4744479bc4bbabd033` (dirty: True)
- Device: `cuda` (NVIDIA GeForce RTX 3050 Laptop GPU)
- Python / torch / torchvision: 3.14.8 / 2.15.0+cu132 / 0.29.2+cu132
- Batch size / workers: 64 / 4

```bash
'/home/toastu/Research/Spoof Detection/.venv/bin/python' -m spoof_detection.evaluate --checkpoint '/home/toastu/Research/Spoof Detection/Models/Iteration1/e_1_2/gray_flip/seed42/resnet18_frozen_gray_flip_seed42.pth' --asvspoof dev eval --source image --data-root '/home/toastu/Research/Spoof Detection/data/LA' --image-root '/home/toastu/Research/Spoof Detection/data/derived/e1_2_rendering/gray_flip' --output-root '/home/toastu/Research/Spoof Detection/reports/phase1/e1_2_rendering/runs/gray_flip/seed42' --frontend gray_flip --experiment-id E1.2 --title 'E1.2 rendering condition: gray_flip (seed 42)' --batch-size 64 --num-workers 4 --device auto
```

## Caveats

- Scores are uncalibrated; Brier and ECE are diagnostics, not evidence of calibrated probabilities.
- EER uses the ASVspoof DET-curve convention (spoof is positive). Per-attack EER reuses all bonafide clips as negatives; per-speaker EER uses same-speaker bonafides.
- Only the dev EER threshold is selected on data. The `dev_eer` threshold in an eval report is transferred from dev and is never selected on eval. Other EER thresholds are descriptive points on the DET curve, not operating thresholds.
- Minimum t-DCF and confidence intervals are not reported in E0.2.
