# E1.2 rendering condition: gray_noflip (seed 42): `resnet18_frozen_gray_noflip_seed42.pth`

Experiment: `E1.2`
Generated: `2026-11-27T14:08:13.517125+00:00`

## Metric contract

| Split | Attack subset | Operating thresholds | Samples | Balance |
|---|---|---|---:|---|
| `dev` | A01-A06 | fixed_0.5=0.500000, dev_eer=0.825407 | 24844 | 2548 / 22296 |
| `eval` | A07-A19 | fixed_0.5=0.500000, dev_eer=0.825407 | 71237 | 7355 / 63882 |

Threshold-free metrics: EER and AUC are computed over all spoof clips against all bonafide clips in the split. They are `n/a` when a split has only one class.

## Aggregate metrics

| Split | EER (threshold-free) | AUC (threshold-free) | Brier | ECE (10 bins) |
|---|---:|---:|---:|---:|
| `dev` | 5.10% | 0.9895 | 0.0226 | 0.0067 |
| `eval` | 11.35% | 0.9512 | 0.0890 | 0.1013 |

## Threshold metrics

| Split | Threshold | Value | Accuracy | Bonafide recall | Spoof recall | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|
| `dev` | `fixed_0.5` | 0.500000 | 96.93% | 86.50% | 98.12% | 13.50% | 1.88% |
| `dev` | `dev_eer` | 0.825407 | 94.90% | 94.86% | 94.90% | 5.14% | 5.10% |
| `eval` | `fixed_0.5` | 0.500000 | 87.95% | 89.68% | 87.75% | 10.32% | 12.25% |
| `eval` | `dev_eer` | 0.825407 | 82.02% | 96.13% | 80.39% | 3.87% | 19.61% |

## Per-attack EER

Per-attack EER uses that attack's spoof clips as positives and all bonafide clips in the split as negatives.

| Split | Attack | n_spoof | n_bonafide | EER | AUC |
|---|---|---:|---:|---:|---:|
| `dev` | A01 | 3716 | 2548 | 1.49% | 0.9991 |
| `dev` | A02 | 3716 | 2548 | 0.91% | 0.9996 |
| `dev` | A03 | 3716 | 2548 | 2.55% | 0.9967 |
| `dev` | A04 | 3716 | 2548 | 4.98% | 0.9893 |
| `dev` | A05 | 3716 | 2548 | 5.01% | 0.9894 |
| `dev` | A06 | 3716 | 2548 | 9.93% | 0.9630 |
| `eval` | A07 | 4914 | 7355 | 1.02% | 0.9994 |
| `eval` | A08 | 4914 | 7355 | 2.32% | 0.9969 |
| `eval` | A09 | 4914 | 7355 | 0.49% | 0.9999 |
| `eval` | A10 | 4914 | 7355 | 7.59% | 0.9740 |
| `eval` | A11 | 4914 | 7355 | 2.27% | 0.9974 |
| `eval` | A12 | 4914 | 7355 | 9.45% | 0.9636 |
| `eval` | A13 | 4914 | 7355 | 12.10% | 0.9464 |
| `eval` | A14 | 4914 | 7355 | 3.32% | 0.9946 |
| `eval` | A15 | 4914 | 7355 | 5.45% | 0.9873 |
| `eval` | A16 | 4914 | 7355 | 4.07% | 0.9924 |
| `eval` | A17 | 4914 | 7355 | 35.31% | 0.7020 |
| `eval` | A18 | 4914 | 7355 | 17.31% | 0.8991 |
| `eval` | A19 | 4914 | 7355 | 16.30% | 0.9128 |

## Per-speaker metrics

Per-speaker EER uses same-speaker bonafide clips as negatives. Recalls are at the fixed 0.5 threshold.

| Split | Speaker | n_spoof | n_bonafide | Spoof recall@0.5 | Bonafide recall@0.5 | EER |
|---|---|---:|---:|---:|---:|---:|
| `dev` | LA_0069 | 2484 | 154 | 98.27% | 92.21% | 3.90% |
| `dev` | LA_0070 | 1848 | 140 | 98.86% | 85.71% | 4.99% |
| `dev` | LA_0071 | 1848 | 140 | 98.32% | 85.71% | 5.73% |
| `dev` | LA_0072 | 2484 | 154 | 98.43% | 86.36% | 5.19% |
| `dev` | LA_0073 | 1848 | 140 | 98.81% | 92.14% | 4.28% |
| `dev` | LA_0074 | 2484 | 154 | 95.89% | 93.51% | 4.67% |
| `dev` | LA_0075 | 2484 | 154 | 97.58% | 92.21% | 3.74% |
| `dev` | LA_0076 | 1848 | 140 | 98.27% | 90.71% | 3.71% |
| `dev` | LA_0077 | 2484 | 154 | 98.63% | 83.77% | 3.78% |
| `dev` | LA_0078 | 2484 | 154 | 98.59% | 97.40% | 2.59% |
| `dev` | LA_0099 | 0 | 77 | n/a | 89.61% | n/a |
| `dev` | LA_0100 | 0 | 77 | n/a | 77.92% | n/a |
| `dev` | LA_0101 | 0 | 77 | n/a | 66.23% | n/a |
| `dev` | LA_0102 | 0 | 126 | n/a | 73.81% | n/a |
| `dev` | LA_0103 | 0 | 126 | n/a | 76.98% | n/a |
| `dev` | LA_0104 | 0 | 126 | n/a | 90.48% | n/a |
| `dev` | LA_0105 | 0 | 77 | n/a | 80.52% | n/a |
| `dev` | LA_0106 | 0 | 126 | n/a | 93.65% | n/a |
| `dev` | LA_0107 | 0 | 126 | n/a | 76.19% | n/a |
| `dev` | LA_0108 | 0 | 126 | n/a | 85.71% | n/a |
| `eval` | LA_0001 | 936 | 68 | 90.17% | 88.24% | 10.33% |
| `eval` | LA_0002 | 936 | 68 | 89.21% | 66.18% | 16.05% |
| `eval` | LA_0003 | 936 | 68 | 88.68% | 92.65% | 8.85% |
| `eval` | LA_0004 | 1638 | 146 | 87.42% | 94.52% | 9.53% |
| `eval` | LA_0005 | 936 | 68 | 90.92% | 94.12% | 6.04% |
| `eval` | LA_0006 | 1638 | 146 | 88.40% | 95.21% | 8.23% |
| `eval` | LA_0007 | 936 | 68 | 86.43% | 89.71% | 11.76% |
| `eval` | LA_0008 | 1638 | 146 | 86.39% | 83.56% | 15.07% |
| `eval` | LA_0009 | 1638 | 146 | 90.23% | 83.56% | 12.21% |
| `eval` | LA_0010 | 1638 | 146 | 87.55% | 84.25% | 13.81% |
| `eval` | LA_0011 | 936 | 68 | 86.32% | 92.65% | 11.92% |
| `eval` | LA_0012 | 1638 | 146 | 91.58% | 70.55% | 14.40% |
| `eval` | LA_0013 | 936 | 68 | 89.21% | 91.18% | 10.28% |
| `eval` | LA_0014 | 1638 | 146 | 87.79% | 89.04% | 11.04% |
| `eval` | LA_0015 | 936 | 68 | 87.82% | 92.65% | 10.28% |
| `eval` | LA_0016 | 1638 | 146 | 90.11% | 91.10% | 8.91% |
| `eval` | LA_0017 | 1638 | 146 | 89.74% | 88.36% | 10.97% |
| `eval` | LA_0018 | 936 | 68 | 89.64% | 97.06% | 7.36% |
| `eval` | LA_0019 | 1638 | 146 | 83.94% | 93.84% | 11.65% |
| `eval` | LA_0020 | 1638 | 146 | 88.58% | 95.21% | 8.23% |
| `eval` | LA_0021 | 936 | 68 | 88.25% | 92.65% | 10.28% |
| `eval` | LA_0022 | 1638 | 146 | 88.22% | 86.99% | 12.33% |
| `eval` | LA_0023 | 936 | 68 | 89.10% | 91.18% | 9.95% |
| `eval` | LA_0024 | 1638 | 146 | 88.71% | 95.21% | 8.91% |
| `eval` | LA_0025 | 936 | 68 | 86.22% | 95.59% | 8.53% |
| `eval` | LA_0026 | 1638 | 146 | 82.05% | 97.95% | 9.50% |
| `eval` | LA_0027 | 1638 | 146 | 83.58% | 93.84% | 11.65% |
| `eval` | LA_0028 | 936 | 68 | 87.39% | 98.53% | 5.72% |
| `eval` | LA_0029 | 1638 | 146 | 87.97% | 97.95% | 7.37% |
| `eval` | LA_0030 | 936 | 68 | 85.04% | 75.00% | 17.64% |
| `eval` | LA_0031 | 1638 | 146 | 84.43% | 96.58% | 7.67% |
| `eval` | LA_0032 | 936 | 68 | 80.66% | 83.82% | 16.47% |
| `eval` | LA_0033 | 1638 | 146 | 89.01% | 89.04% | 10.97% |
| `eval` | LA_0034 | 1638 | 146 | 90.66% | 94.52% | 7.70% |
| `eval` | LA_0035 | 1638 | 146 | 89.19% | 93.84% | 9.56% |
| `eval` | LA_0036 | 936 | 68 | 87.82% | 82.35% | 13.40% |
| `eval` | LA_0037 | 1638 | 146 | 89.01% | 92.47% | 8.76% |
| `eval` | LA_0038 | 936 | 68 | 89.96% | 47.06% | 22.03% |
| `eval` | LA_0039 | 1638 | 146 | 86.63% | 95.89% | 10.14% |
| `eval` | LA_0040 | 936 | 68 | 85.15% | 91.18% | 11.49% |
| `eval` | LA_0041 | 1638 | 146 | 86.39% | 84.93% | 14.30% |
| `eval` | LA_0042 | 1638 | 146 | 87.00% | 89.73% | 11.68% |
| `eval` | LA_0043 | 1638 | 146 | 89.13% | 84.93% | 11.68% |
| `eval` | LA_0044 | 936 | 68 | 88.78% | 83.82% | 13.24% |
| `eval` | LA_0045 | 1638 | 146 | 86.81% | 95.21% | 8.91% |
| `eval` | LA_0046 | 936 | 68 | 90.28% | 91.18% | 10.01% |
| `eval` | LA_0047 | 1638 | 146 | 87.61% | 93.84% | 9.59% |
| `eval` | LA_0048 | 936 | 68 | 87.39% | 88.24% | 12.92% |
| `eval` | LA_0049 | 0 | 95 | n/a | 85.26% | n/a |
| `eval` | LA_0050 | 0 | 113 | n/a | 89.38% | n/a |
| `eval` | LA_0051 | 0 | 113 | n/a | 88.50% | n/a |
| `eval` | LA_0052 | 0 | 95 | n/a | 92.63% | n/a |
| `eval` | LA_0053 | 0 | 113 | n/a | 84.96% | n/a |
| `eval` | LA_0054 | 0 | 95 | n/a | 92.63% | n/a |
| `eval` | LA_0055 | 0 | 113 | n/a | 92.92% | n/a |
| `eval` | LA_0056 | 0 | 95 | n/a | 94.74% | n/a |
| `eval` | LA_0057 | 0 | 95 | n/a | 88.42% | n/a |
| `eval` | LA_0058 | 0 | 113 | n/a | 87.61% | n/a |
| `eval` | LA_0059 | 0 | 95 | n/a | 90.53% | n/a |
| `eval` | LA_0060 | 0 | 113 | n/a | 89.38% | n/a |
| `eval` | LA_0061 | 0 | 113 | n/a | 98.23% | n/a |
| `eval` | LA_0063 | 0 | 113 | n/a | 92.92% | n/a |
| `eval` | LA_0064 | 0 | 95 | n/a | 71.58% | n/a |
| `eval` | LA_0065 | 0 | 95 | n/a | 95.79% | n/a |
| `eval` | LA_0066 | 0 | 113 | n/a | 92.92% | n/a |
| `eval` | LA_0067 | 0 | 113 | n/a | 84.07% | n/a |
| `eval` | LA_0068 | 0 | 95 | n/a | 85.26% | n/a |

## Checkpoint

- Path: `Models/Iteration1/e_1_2/gray_noflip/seed42/resnet18_frozen_gray_noflip_seed42.pth`
- SHA-256: `bcbed29544a2869d45bfe893441757970ed0f14f80fde50e28c435a16948550b`
- Parameters: 11242305 total (11176512 backbone, 65793 head)
- State-dict keys: 124 (124 remapped)
- Missing keys: none
- Unexpected keys: none
- Head state SHA-256: `41ea4d6d24b5bdadf382e71390620cdd28a881780b906498137b62cbe9c9eb7b`
- Final head bias: -0.5652
- Batch-norm steps: [6300]

## Provenance

- **dev**: {'dataset': 'ASVspoof 2019 LA', 'split': 'dev', 'source': 'image', 'data_root': 'data/LA', 'protocol_path': 'data/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt', 'protocol_sha256': '2fedaebf1ba3ae1a9bb7d3ec916804ac237b4eee063a27fa92e6549f03d129c7', 'image_root': 'data/derived/e1_2_rendering/gray_noflip', 'metadata_path': 'data/derived/e1_2_rendering/gray_noflip/dev/metadata_dev.csv', 'metadata_sha256': '82db0beb84426fc09a5d67d38344790d736c4597c9a5971c390e0f22d5c256bb'}
- **eval**: {'dataset': 'ASVspoof 2019 LA', 'split': 'eval', 'source': 'image', 'data_root': 'data/LA', 'protocol_path': 'data/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt', 'protocol_sha256': '84fce576efff092269a0a6ad247d2e9f2f7b4d4e25a4545efb2518da92708036', 'image_root': 'data/derived/e1_2_rendering/gray_noflip', 'metadata_path': 'data/derived/e1_2_rendering/gray_noflip/eval/metadata_eval.csv', 'metadata_sha256': 'c1c87a5e7cac11cec7e3a408d142e45eb749db098a36a23f0ad31b2931fd6840'}

## Environment and command

- Git commit: `4fb53d685d23dc9f1040cf4744479bc4bbabd033` (dirty: True)
- Device: `cuda` (NVIDIA GeForce RTX 3050 Laptop GPU)
- Python / torch / torchvision: 3.14.8 / 2.15.0+cu132 / 0.29.2+cu132
- Batch size / workers: 64 / 4

```bash
'/home/toastu/Research/Spoof Detection/.venv/bin/python' -m spoof_detection.evaluate --checkpoint '/home/toastu/Research/Spoof Detection/Models/Iteration1/e_1_2/gray_noflip/seed42/resnet18_frozen_gray_noflip_seed42.pth' --asvspoof dev eval --source image --data-root '/home/toastu/Research/Spoof Detection/data/LA' --image-root '/home/toastu/Research/Spoof Detection/data/derived/e1_2_rendering/gray_noflip' --output-root '/home/toastu/Research/Spoof Detection/reports/phase1/e1_2_rendering/runs/gray_noflip/seed42' --frontend gray_noflip --experiment-id E1.2 --title 'E1.2 rendering condition: gray_noflip (seed 42)' --batch-size 64 --num-workers 4 --device auto
```

## Caveats

- Scores are uncalibrated; Brier and ECE are diagnostics, not evidence of calibrated probabilities.
- EER uses the ASVspoof DET-curve convention (spoof is positive). Per-attack EER reuses all bonafide clips as negatives; per-speaker EER uses same-speaker bonafides.
- Only the dev EER threshold is selected on data. The `dev_eer` threshold in an eval report is transferred from dev and is never selected on eval. Other EER thresholds are descriptive points on the DET curve, not operating thresholds.
- Minimum t-DCF and confidence intervals are not reported in E0.2.
