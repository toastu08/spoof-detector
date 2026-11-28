# E1.2 rendering condition: magma_flip_rgb (seed 42): `resnet18_frozen_magma_flip_rgb_seed42.pth`

Experiment: `E1.2`
Generated: `2026-11-27T14:10:27.530204+00:00`

## Metric contract

| Split | Attack subset | Operating thresholds | Samples | Balance |
|---|---|---|---:|---|
| `dev` | A01-A06 | fixed_0.5=0.500000, dev_eer=0.916409 | 24844 | 2548 / 22296 |
| `eval` | A07-A19 | fixed_0.5=0.500000, dev_eer=0.916409 | 71237 | 7355 / 63882 |

Threshold-free metrics: EER and AUC are computed over all spoof clips against all bonafide clips in the split. They are `n/a` when a split has only one class.

## Aggregate metrics

| Split | EER (threshold-free) | AUC (threshold-free) | Brier | ECE (10 bins) |
|---|---:|---:|---:|---:|
| `dev` | 5.68% | 0.9876 | 0.0250 | 0.0085 |
| `eval` | 11.41% | 0.9536 | 0.0648 | 0.0571 |

## Threshold metrics

| Split | Threshold | Value | Accuracy | Bonafide recall | Spoof recall | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|
| `dev` | `fixed_0.5` | 0.500000 | 96.61% | 77.63% | 98.78% | 22.37% | 1.22% |
| `dev` | `dev_eer` | 0.916409 | 94.33% | 94.31% | 94.33% | 5.69% | 5.67% |
| `eval` | `fixed_0.5` | 0.500000 | 91.08% | 80.39% | 92.32% | 19.61% | 7.68% |
| `eval` | `dev_eer` | 0.916409 | 83.54% | 94.93% | 82.23% | 5.07% | 17.77% |

## Per-attack EER

Per-attack EER uses that attack's spoof clips as positives and all bonafide clips in the split as negatives.

| Split | Attack | n_spoof | n_bonafide | EER | AUC |
|---|---|---:|---:|---:|---:|
| `dev` | A01 | 3716 | 2548 | 1.72% | 0.9986 |
| `dev` | A02 | 3716 | 2548 | 1.42% | 0.9990 |
| `dev` | A03 | 3716 | 2548 | 2.67% | 0.9965 |
| `dev` | A04 | 3716 | 2548 | 5.06% | 0.9897 |
| `dev` | A05 | 3716 | 2548 | 6.83% | 0.9827 |
| `dev` | A06 | 3716 | 2548 | 11.11% | 0.9588 |
| `eval` | A07 | 4914 | 7355 | 1.37% | 0.9989 |
| `eval` | A08 | 4914 | 7355 | 3.22% | 0.9951 |
| `eval` | A09 | 4914 | 7355 | 1.16% | 0.9992 |
| `eval` | A10 | 4914 | 7355 | 6.73% | 0.9794 |
| `eval` | A11 | 4914 | 7355 | 3.58% | 0.9944 |
| `eval` | A12 | 4914 | 7355 | 6.69% | 0.9803 |
| `eval` | A13 | 4914 | 7355 | 5.19% | 0.9876 |
| `eval` | A14 | 4914 | 7355 | 4.52% | 0.9915 |
| `eval` | A15 | 4914 | 7355 | 6.12% | 0.9832 |
| `eval` | A16 | 4914 | 7355 | 4.56% | 0.9902 |
| `eval` | A17 | 4914 | 7355 | 34.39% | 0.7119 |
| `eval` | A18 | 4914 | 7355 | 18.56% | 0.8920 |
| `eval` | A19 | 4914 | 7355 | 18.82% | 0.8935 |

## Per-speaker metrics

Per-speaker EER uses same-speaker bonafide clips as negatives. Recalls are at the fixed 0.5 threshold.

| Split | Speaker | n_spoof | n_bonafide | Spoof recall@0.5 | Bonafide recall@0.5 | EER |
|---|---|---:|---:|---:|---:|---:|
| `dev` | LA_0069 | 2484 | 154 | 98.59% | 74.03% | 5.84% |
| `dev` | LA_0070 | 1848 | 140 | 98.38% | 89.29% | 4.99% |
| `dev` | LA_0071 | 1848 | 140 | 99.03% | 71.43% | 6.43% |
| `dev` | LA_0072 | 2484 | 154 | 98.95% | 81.17% | 3.90% |
| `dev` | LA_0073 | 1848 | 140 | 99.13% | 87.14% | 3.57% |
| `dev` | LA_0074 | 2484 | 154 | 97.99% | 85.71% | 5.76% |
| `dev` | LA_0075 | 2484 | 154 | 98.67% | 90.26% | 3.90% |
| `dev` | LA_0076 | 1848 | 140 | 99.46% | 76.43% | 4.36% |
| `dev` | LA_0077 | 2484 | 154 | 98.79% | 75.97% | 5.19% |
| `dev` | LA_0078 | 2484 | 154 | 99.07% | 88.31% | 2.75% |
| `dev` | LA_0099 | 0 | 77 | n/a | 85.71% | n/a |
| `dev` | LA_0100 | 0 | 77 | n/a | 63.64% | n/a |
| `dev` | LA_0101 | 0 | 77 | n/a | 42.86% | n/a |
| `dev` | LA_0102 | 0 | 126 | n/a | 49.21% | n/a |
| `dev` | LA_0103 | 0 | 126 | n/a | 65.08% | n/a |
| `dev` | LA_0104 | 0 | 126 | n/a | 76.98% | n/a |
| `dev` | LA_0105 | 0 | 77 | n/a | 80.52% | n/a |
| `dev` | LA_0106 | 0 | 126 | n/a | 88.10% | n/a |
| `dev` | LA_0107 | 0 | 126 | n/a | 79.37% | n/a |
| `dev` | LA_0108 | 0 | 126 | n/a | 78.57% | n/a |
| `eval` | LA_0001 | 936 | 68 | 96.26% | 76.47% | 8.74% |
| `eval` | LA_0002 | 936 | 68 | 94.76% | 55.88% | 17.42% |
| `eval` | LA_0003 | 936 | 68 | 94.55% | 85.29% | 10.06% |
| `eval` | LA_0004 | 1638 | 146 | 91.21% | 88.36% | 9.06% |
| `eval` | LA_0005 | 936 | 68 | 94.55% | 91.18% | 5.83% |
| `eval` | LA_0006 | 1638 | 146 | 92.67% | 88.36% | 10.23% |
| `eval` | LA_0007 | 936 | 68 | 92.84% | 83.82% | 9.06% |
| `eval` | LA_0008 | 1638 | 146 | 90.72% | 91.10% | 8.91% |
| `eval` | LA_0009 | 1638 | 146 | 92.31% | 80.82% | 10.79% |
| `eval` | LA_0010 | 1638 | 146 | 93.35% | 74.66% | 14.40% |
| `eval` | LA_0011 | 936 | 68 | 91.77% | 75.00% | 10.49% |
| `eval` | LA_0012 | 1638 | 146 | 93.89% | 48.63% | 15.75% |
| `eval` | LA_0013 | 936 | 68 | 94.12% | 82.35% | 10.54% |
| `eval` | LA_0014 | 1638 | 146 | 92.43% | 64.38% | 14.40% |
| `eval` | LA_0015 | 936 | 68 | 93.91% | 89.71% | 8.85% |
| `eval` | LA_0016 | 1638 | 146 | 92.49% | 84.25% | 10.17% |
| `eval` | LA_0017 | 1638 | 146 | 91.82% | 79.45% | 13.53% |
| `eval` | LA_0018 | 936 | 68 | 92.09% | 89.71% | 8.63% |
| `eval` | LA_0019 | 1638 | 146 | 92.00% | 76.71% | 13.60% |
| `eval` | LA_0020 | 1638 | 146 | 92.43% | 83.56% | 10.33% |
| `eval` | LA_0021 | 936 | 68 | 90.49% | 91.18% | 8.90% |
| `eval` | LA_0022 | 1638 | 146 | 92.12% | 72.60% | 12.89% |
| `eval` | LA_0023 | 936 | 68 | 92.20% | 77.94% | 13.29% |
| `eval` | LA_0024 | 1638 | 146 | 92.61% | 82.88% | 9.03% |
| `eval` | LA_0025 | 936 | 68 | 93.91% | 89.71% | 7.52% |
| `eval` | LA_0026 | 1638 | 146 | 91.39% | 90.41% | 9.50% |
| `eval` | LA_0027 | 1638 | 146 | 89.99% | 89.04% | 10.27% |
| `eval` | LA_0028 | 936 | 68 | 92.52% | 91.18% | 8.85% |
| `eval` | LA_0029 | 1638 | 146 | 93.65% | 89.73% | 8.23% |
| `eval` | LA_0030 | 936 | 68 | 93.27% | 61.76% | 11.76% |
| `eval` | LA_0031 | 1638 | 146 | 90.42% | 83.56% | 11.56% |
| `eval` | LA_0032 | 936 | 68 | 87.71% | 79.41% | 16.10% |
| `eval` | LA_0033 | 1638 | 146 | 90.90% | 78.08% | 13.75% |
| `eval` | LA_0034 | 1638 | 146 | 93.10% | 81.51% | 10.30% |
| `eval` | LA_0035 | 1638 | 146 | 91.64% | 84.93% | 10.79% |
| `eval` | LA_0036 | 936 | 68 | 93.16% | 85.29% | 8.85% |
| `eval` | LA_0037 | 1638 | 146 | 92.98% | 91.10% | 8.23% |
| `eval` | LA_0038 | 936 | 68 | 94.34% | 41.18% | 26.27% |
| `eval` | LA_0039 | 1638 | 146 | 88.95% | 95.21% | 9.06% |
| `eval` | LA_0040 | 936 | 68 | 91.35% | 73.53% | 11.97% |
| `eval` | LA_0041 | 1638 | 146 | 92.92% | 67.12% | 15.20% |
| `eval` | LA_0042 | 1638 | 146 | 91.27% | 78.08% | 12.45% |
| `eval` | LA_0043 | 1638 | 146 | 93.28% | 59.59% | 13.53% |
| `eval` | LA_0044 | 936 | 68 | 92.20% | 73.53% | 14.72% |
| `eval` | LA_0045 | 1638 | 146 | 92.06% | 86.30% | 8.94% |
| `eval` | LA_0046 | 936 | 68 | 93.48% | 85.29% | 8.95% |
| `eval` | LA_0047 | 1638 | 146 | 92.19% | 85.62% | 10.27% |
| `eval` | LA_0048 | 936 | 68 | 92.63% | 83.82% | 8.74% |
| `eval` | LA_0049 | 0 | 95 | n/a | 88.42% | n/a |
| `eval` | LA_0050 | 0 | 113 | n/a | 76.11% | n/a |
| `eval` | LA_0051 | 0 | 113 | n/a | 73.45% | n/a |
| `eval` | LA_0052 | 0 | 95 | n/a | 94.74% | n/a |
| `eval` | LA_0053 | 0 | 113 | n/a | 89.38% | n/a |
| `eval` | LA_0054 | 0 | 95 | n/a | 77.89% | n/a |
| `eval` | LA_0055 | 0 | 113 | n/a | 69.91% | n/a |
| `eval` | LA_0056 | 0 | 95 | n/a | 89.47% | n/a |
| `eval` | LA_0057 | 0 | 95 | n/a | 68.42% | n/a |
| `eval` | LA_0058 | 0 | 113 | n/a | 86.73% | n/a |
| `eval` | LA_0059 | 0 | 95 | n/a | 68.42% | n/a |
| `eval` | LA_0060 | 0 | 113 | n/a | 69.91% | n/a |
| `eval` | LA_0061 | 0 | 113 | n/a | 90.27% | n/a |
| `eval` | LA_0063 | 0 | 113 | n/a | 85.84% | n/a |
| `eval` | LA_0064 | 0 | 95 | n/a | 78.95% | n/a |
| `eval` | LA_0065 | 0 | 95 | n/a | 90.53% | n/a |
| `eval` | LA_0066 | 0 | 113 | n/a | 88.50% | n/a |
| `eval` | LA_0067 | 0 | 113 | n/a | 83.19% | n/a |
| `eval` | LA_0068 | 0 | 95 | n/a | 65.26% | n/a |

## Checkpoint

- Path: `Models/Iteration1/e_1_2/magma_flip_rgb/seed42/resnet18_frozen_magma_flip_rgb_seed42.pth`
- SHA-256: `404e40573e70b6db6dacc0ada0101fda4f9e0c4513e36b106f68ce4b2f114d0a`
- Parameters: 11242305 total (11176512 backbone, 65793 head)
- State-dict keys: 124 (124 remapped)
- Missing keys: none
- Unexpected keys: none
- Head state SHA-256: `36f2a37c5b9a6cbefd5149895e754dd31addde0adf4eae72a4a1e507d00db36d`
- Final head bias: -0.5761
- Batch-norm steps: [6300]

## Provenance

- **dev**: {'dataset': 'ASVspoof 2019 LA', 'split': 'dev', 'source': 'image', 'data_root': 'data/LA', 'protocol_path': 'data/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt', 'protocol_sha256': '2fedaebf1ba3ae1a9bb7d3ec916804ac237b4eee063a27fa92e6549f03d129c7', 'image_root': 'data/derived/e1_2_rendering/magma_flip_rgb', 'metadata_path': 'data/derived/e1_2_rendering/magma_flip_rgb/dev/metadata_dev.csv', 'metadata_sha256': '9821ff1e33c0a7e42de68dfcb552859c61c4ffaaebfeea1a881afdb16a055d99'}
- **eval**: {'dataset': 'ASVspoof 2019 LA', 'split': 'eval', 'source': 'image', 'data_root': 'data/LA', 'protocol_path': 'data/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt', 'protocol_sha256': '84fce576efff092269a0a6ad247d2e9f2f7b4d4e25a4545efb2518da92708036', 'image_root': 'data/derived/e1_2_rendering/magma_flip_rgb', 'metadata_path': 'data/derived/e1_2_rendering/magma_flip_rgb/eval/metadata_eval.csv', 'metadata_sha256': '767140793e44b9b71354ed05795449e4543204d5994857707d5d977bf0899796'}

## Environment and command

- Git commit: `4fb53d685d23dc9f1040cf4744479bc4bbabd033` (dirty: True)
- Device: `cuda` (NVIDIA GeForce RTX 3050 Laptop GPU)
- Python / torch / torchvision: 3.14.8 / 2.15.0+cu132 / 0.29.2+cu132
- Batch size / workers: 64 / 4

```bash
'/home/toastu/Research/Spoof Detection/.venv/bin/python' -m spoof_detection.evaluate --checkpoint '/home/toastu/Research/Spoof Detection/Models/Iteration1/e_1_2/magma_flip_rgb/seed42/resnet18_frozen_magma_flip_rgb_seed42.pth' --asvspoof dev eval --source image --data-root '/home/toastu/Research/Spoof Detection/data/LA' --image-root '/home/toastu/Research/Spoof Detection/data/derived/e1_2_rendering/magma_flip_rgb' --output-root '/home/toastu/Research/Spoof Detection/reports/phase1/e1_2_rendering/runs/magma_flip_rgb/seed42' --frontend magma_flip_rgb --experiment-id E1.2 --title 'E1.2 rendering condition: magma_flip_rgb (seed 42)' --batch-size 64 --num-workers 4 --device auto
```

## Caveats

- Scores are uncalibrated; Brier and ECE are diagnostics, not evidence of calibrated probabilities.
- EER uses the ASVspoof DET-curve convention (spoof is positive). Per-attack EER reuses all bonafide clips as negatives; per-speaker EER uses same-speaker bonafides.
- Only the dev EER threshold is selected on data. The `dev_eer` threshold in an eval report is transferred from dev and is never selected on eval. Other EER thresholds are descriptive points on the DET curve, not operating thresholds.
- Minimum t-DCF and confidence intervals are not reported in E0.2.
