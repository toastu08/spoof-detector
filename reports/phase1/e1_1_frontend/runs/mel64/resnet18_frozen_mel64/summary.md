# E1.1 front-end probe: mel64: `resnet18_frozen_mel64.pth`

Experiment: `E1.1`
Generated: `2026-11-20T15:53:32.519181+00:00`

## Metric contract

| Split | Attack subset | Operating thresholds | Samples | Balance |
|---|---|---|---:|---|
| `dev` | A01-A06 | fixed_0.5=0.500000, dev_eer=0.836578 | 24844 | 2548 / 22296 |
| `eval` | A07-A19 | fixed_0.5=0.500000, dev_eer=0.836578 | 71237 | 7355 / 63882 |

Threshold-free metrics: EER and AUC are computed over all spoof clips against all bonafide clips in the split. They are `n/a` when a split has only one class.

## Aggregate metrics

| Split | EER (threshold-free) | AUC (threshold-free) | Brier | ECE (10 bins) |
|---|---:|---:|---:|---:|
| `dev` | 10.05% | 0.9669 | 0.0398 | 0.0099 |
| `eval` | 13.01% | 0.9434 | 0.0659 | 0.0560 |

## Threshold metrics

| Split | Threshold | Value | Accuracy | Bonafide recall | Spoof recall | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|
| `dev` | `fixed_0.5` | 0.500000 | 94.69% | 70.49% | 97.46% | 29.51% | 2.54% |
| `dev` | `dev_eer` | 0.836578 | 89.96% | 89.95% | 89.96% | 10.05% | 10.04% |
| `eval` | `fixed_0.5` | 0.500000 | 90.67% | 75.34% | 92.44% | 24.66% | 7.56% |
| `eval` | `dev_eer` | 0.836578 | 83.04% | 92.56% | 81.94% | 7.44% | 18.06% |

## Per-attack EER

Per-attack EER uses that attack's spoof clips as positives and all bonafide clips in the split as negatives.

| Split | Attack | n_spoof | n_bonafide | EER | AUC |
|---|---|---:|---:|---:|---:|
| `dev` | A01 | 3716 | 2548 | 2.79% | 0.9966 |
| `dev` | A02 | 3716 | 2548 | 3.64% | 0.9936 |
| `dev` | A03 | 3716 | 2548 | 3.50% | 0.9936 |
| `dev` | A04 | 3716 | 2548 | 9.66% | 0.9683 |
| `dev` | A05 | 3716 | 2548 | 11.81% | 0.9560 |
| `dev` | A06 | 3716 | 2548 | 18.33% | 0.8936 |
| `eval` | A07 | 4914 | 7355 | 3.38% | 0.9943 |
| `eval` | A08 | 4914 | 7355 | 3.93% | 0.9931 |
| `eval` | A09 | 4914 | 7355 | 0.47% | 0.9999 |
| `eval` | A10 | 4914 | 7355 | 6.81% | 0.9817 |
| `eval` | A11 | 4914 | 7355 | 3.07% | 0.9958 |
| `eval` | A12 | 4914 | 7355 | 5.58% | 0.9860 |
| `eval` | A13 | 4914 | 7355 | 5.27% | 0.9870 |
| `eval` | A14 | 4914 | 7355 | 6.09% | 0.9835 |
| `eval` | A15 | 4914 | 7355 | 4.60% | 0.9910 |
| `eval` | A16 | 4914 | 7355 | 8.18% | 0.9733 |
| `eval` | A17 | 4914 | 7355 | 33.56% | 0.7198 |
| `eval` | A18 | 4914 | 7355 | 21.79% | 0.8587 |
| `eval` | A19 | 4914 | 7355 | 26.94% | 0.7999 |

## Per-speaker metrics

Per-speaker EER uses same-speaker bonafide clips as negatives. Recalls are at the fixed 0.5 threshold.

| Split | Speaker | n_spoof | n_bonafide | Spoof recall@0.5 | Bonafide recall@0.5 | EER |
|---|---|---:|---:|---:|---:|---:|
| `dev` | LA_0069 | 2484 | 154 | 97.75% | 79.22% | 8.45% |
| `dev` | LA_0070 | 1848 | 140 | 98.38% | 72.86% | 9.43% |
| `dev` | LA_0071 | 1848 | 140 | 96.97% | 60.00% | 14.42% |
| `dev` | LA_0072 | 2484 | 154 | 98.35% | 68.83% | 9.03% |
| `dev` | LA_0073 | 1848 | 140 | 98.59% | 82.86% | 6.30% |
| `dev` | LA_0074 | 2484 | 154 | 94.81% | 83.12% | 9.09% |
| `dev` | LA_0075 | 2484 | 154 | 96.74% | 73.38% | 9.88% |
| `dev` | LA_0076 | 1848 | 140 | 97.84% | 72.86% | 8.72% |
| `dev` | LA_0077 | 2484 | 154 | 98.11% | 79.87% | 6.49% |
| `dev` | LA_0078 | 2484 | 154 | 97.58% | 85.71% | 5.19% |
| `dev` | LA_0099 | 0 | 77 | n/a | 63.64% | n/a |
| `dev` | LA_0100 | 0 | 77 | n/a | 49.35% | n/a |
| `dev` | LA_0101 | 0 | 77 | n/a | 40.26% | n/a |
| `dev` | LA_0102 | 0 | 126 | n/a | 53.97% | n/a |
| `dev` | LA_0103 | 0 | 126 | n/a | 61.11% | n/a |
| `dev` | LA_0104 | 0 | 126 | n/a | 74.60% | n/a |
| `dev` | LA_0105 | 0 | 77 | n/a | 62.34% | n/a |
| `dev` | LA_0106 | 0 | 126 | n/a | 76.19% | n/a |
| `dev` | LA_0107 | 0 | 126 | n/a | 53.97% | n/a |
| `dev` | LA_0108 | 0 | 126 | n/a | 78.57% | n/a |
| `eval` | LA_0001 | 936 | 68 | 96.05% | 72.06% | 11.60% |
| `eval` | LA_0002 | 936 | 68 | 96.05% | 44.12% | 17.85% |
| `eval` | LA_0003 | 936 | 68 | 92.52% | 67.65% | 16.15% |
| `eval` | LA_0004 | 1638 | 146 | 91.76% | 80.14% | 11.50% |
| `eval` | LA_0005 | 936 | 68 | 94.66% | 86.76% | 7.36% |
| `eval` | LA_0006 | 1638 | 146 | 92.74% | 90.41% | 8.29% |
| `eval` | LA_0007 | 936 | 68 | 93.16% | 79.41% | 10.49% |
| `eval` | LA_0008 | 1638 | 146 | 91.94% | 86.99% | 11.10% |
| `eval` | LA_0009 | 1638 | 146 | 93.28% | 68.49% | 13.01% |
| `eval` | LA_0010 | 1638 | 146 | 90.84% | 75.34% | 15.07% |
| `eval` | LA_0011 | 936 | 68 | 94.34% | 75.00% | 13.24% |
| `eval` | LA_0012 | 1638 | 146 | 93.71% | 34.93% | 20.56% |
| `eval` | LA_0013 | 936 | 68 | 92.74% | 67.65% | 13.24% |
| `eval` | LA_0014 | 1638 | 146 | 92.43% | 67.12% | 15.75% |
| `eval` | LA_0015 | 936 | 68 | 94.02% | 88.24% | 8.85% |
| `eval` | LA_0016 | 1638 | 146 | 92.37% | 83.56% | 9.59% |
| `eval` | LA_0017 | 1638 | 146 | 92.06% | 70.55% | 13.84% |
| `eval` | LA_0018 | 936 | 68 | 92.20% | 82.35% | 10.28% |
| `eval` | LA_0019 | 1638 | 146 | 90.48% | 71.23% | 14.43% |
| `eval` | LA_0020 | 1638 | 146 | 92.49% | 81.51% | 11.13% |
| `eval` | LA_0021 | 936 | 68 | 90.81% | 91.18% | 8.90% |
| `eval` | LA_0022 | 1638 | 146 | 93.22% | 78.08% | 12.30% |
| `eval` | LA_0023 | 936 | 68 | 93.48% | 70.59% | 13.29% |
| `eval` | LA_0024 | 1638 | 146 | 92.74% | 78.77% | 11.04% |
| `eval` | LA_0025 | 936 | 68 | 93.80% | 88.24% | 9.11% |
| `eval` | LA_0026 | 1638 | 146 | 92.12% | 90.41% | 8.79% |
| `eval` | LA_0027 | 1638 | 146 | 90.60% | 71.92% | 14.40% |
| `eval` | LA_0028 | 936 | 68 | 92.63% | 92.65% | 7.36% |
| `eval` | LA_0029 | 1638 | 146 | 92.61% | 83.56% | 10.27% |
| `eval` | LA_0030 | 936 | 68 | 94.23% | 52.94% | 13.51% |
| `eval` | LA_0031 | 1638 | 146 | 90.90% | 81.51% | 12.98% |
| `eval` | LA_0032 | 936 | 68 | 90.71% | 61.76% | 17.42% |
| `eval` | LA_0033 | 1638 | 146 | 91.21% | 84.93% | 12.92% |
| `eval` | LA_0034 | 1638 | 146 | 92.67% | 84.25% | 9.53% |
| `eval` | LA_0035 | 1638 | 146 | 91.70% | 83.56% | 10.79% |
| `eval` | LA_0036 | 936 | 68 | 92.95% | 80.88% | 11.76% |
| `eval` | LA_0037 | 1638 | 146 | 92.61% | 86.99% | 9.71% |
| `eval` | LA_0038 | 936 | 68 | 93.48% | 27.94% | 22.25% |
| `eval` | LA_0039 | 1638 | 146 | 90.72% | 86.99% | 10.36% |
| `eval` | LA_0040 | 936 | 68 | 92.20% | 67.65% | 17.48% |
| `eval` | LA_0041 | 1638 | 146 | 92.19% | 52.05% | 17.82% |
| `eval` | LA_0042 | 1638 | 146 | 90.78% | 69.86% | 15.75% |
| `eval` | LA_0043 | 1638 | 146 | 92.49% | 75.34% | 12.33% |
| `eval` | LA_0044 | 936 | 68 | 93.48% | 58.82% | 13.24% |
| `eval` | LA_0045 | 1638 | 146 | 91.58% | 85.62% | 10.30% |
| `eval` | LA_0046 | 936 | 68 | 94.66% | 85.29% | 10.44% |
| `eval` | LA_0047 | 1638 | 146 | 92.06% | 80.14% | 12.30% |
| `eval` | LA_0048 | 936 | 68 | 93.06% | 70.59% | 13.46% |
| `eval` | LA_0049 | 0 | 95 | n/a | 80.00% | n/a |
| `eval` | LA_0050 | 0 | 113 | n/a | 70.80% | n/a |
| `eval` | LA_0051 | 0 | 113 | n/a | 70.80% | n/a |
| `eval` | LA_0052 | 0 | 95 | n/a | 87.37% | n/a |
| `eval` | LA_0053 | 0 | 113 | n/a | 79.65% | n/a |
| `eval` | LA_0054 | 0 | 95 | n/a | 72.63% | n/a |
| `eval` | LA_0055 | 0 | 113 | n/a | 65.49% | n/a |
| `eval` | LA_0056 | 0 | 95 | n/a | 80.00% | n/a |
| `eval` | LA_0057 | 0 | 95 | n/a | 64.21% | n/a |
| `eval` | LA_0058 | 0 | 113 | n/a | 81.42% | n/a |
| `eval` | LA_0059 | 0 | 95 | n/a | 60.00% | n/a |
| `eval` | LA_0060 | 0 | 113 | n/a | 69.91% | n/a |
| `eval` | LA_0061 | 0 | 113 | n/a | 83.19% | n/a |
| `eval` | LA_0063 | 0 | 113 | n/a | 79.65% | n/a |
| `eval` | LA_0064 | 0 | 95 | n/a | 60.00% | n/a |
| `eval` | LA_0065 | 0 | 95 | n/a | 72.63% | n/a |
| `eval` | LA_0066 | 0 | 113 | n/a | 82.30% | n/a |
| `eval` | LA_0067 | 0 | 113 | n/a | 75.22% | n/a |
| `eval` | LA_0068 | 0 | 95 | n/a | 68.42% | n/a |

## Checkpoint

- Path: `Models/Iteration1/e_1_1/mel64/resnet18_frozen_mel64.pth`
- SHA-256: `30028f1aa47e85c0777363735ae34146f6e2233dee57900f87d7a7c8ba074b2c`
- Parameters: 11242305 total (11176512 backbone, 65793 head)
- State-dict keys: 124 (124 remapped)
- Missing keys: none
- Unexpected keys: none
- Head state SHA-256: `d89db757e1ad753b56dcaf483d7c49185fadd1019ae7bdfedfcb3d2a5d2fe4bc`
- Final head bias: -0.4018
- Batch-norm steps: [5670]

## Provenance

- **dev**: {'dataset': 'ASVspoof 2019 LA', 'split': 'dev', 'source': 'image', 'data_root': 'data/LA', 'protocol_path': 'data/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt', 'protocol_sha256': '2fedaebf1ba3ae1a9bb7d3ec916804ac237b4eee063a27fa92e6549f03d129c7', 'image_root': 'data/derived/e1_1_frontend/mel64', 'metadata_path': 'data/derived/e1_1_frontend/mel64/dev/metadata_dev.csv', 'metadata_sha256': '8c6653bfc6d979a52b3b9cad64ae1104e1a24d7f3118fba4f1bfaff546346993'}
- **eval**: {'dataset': 'ASVspoof 2019 LA', 'split': 'eval', 'source': 'image', 'data_root': 'data/LA', 'protocol_path': 'data/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt', 'protocol_sha256': '84fce576efff092269a0a6ad247d2e9f2f7b4d4e25a4545efb2518da92708036', 'image_root': 'data/derived/e1_1_frontend/mel64', 'metadata_path': 'data/derived/e1_1_frontend/mel64/eval/metadata_eval.csv', 'metadata_sha256': 'a54839d28edbd47d4b121e166ac117a4c99d6c971d2513aa7513c45837723f6a'}

## Environment and command

- Git commit: `9881dbf6a5cafe8fe566f035d32b86b00a8cba21` (dirty: True)
- Device: `cuda` (NVIDIA GeForce RTX 3050 Laptop GPU)
- Python / torch / torchvision: 3.14.8 / 2.15.0+cu132 / 0.29.2+cu132
- Batch size / workers: 64 / 4

```bash
'/home/toastu/Research/Spoof Detection/.venv/bin/python' -m spoof_detection.evaluate --checkpoint '/home/toastu/Research/Spoof Detection/Models/Iteration1/e_1_1/mel64/resnet18_frozen_mel64.pth' --asvspoof dev eval --source image --data-root '/home/toastu/Research/Spoof Detection/data/LA' --image-root '/home/toastu/Research/Spoof Detection/data/derived/e1_1_frontend/mel64' --output-root '/home/toastu/Research/Spoof Detection/reports/phase1/e1_1_frontend/runs/mel64' --frontend mel64 --experiment-id E1.1 --title 'E1.1 front-end probe: mel64' --batch-size 64 --num-workers 4 --device auto
```

## Caveats

- Scores are uncalibrated; Brier and ECE are diagnostics, not evidence of calibrated probabilities.
- EER uses the ASVspoof DET-curve convention (spoof is positive). Per-attack EER reuses all bonafide clips as negatives; per-speaker EER uses same-speaker bonafides.
- Only the dev EER threshold is selected on data. The `dev_eer` threshold in an eval report is transferred from dev and is never selected on eval. Other EER thresholds are descriptive points on the DET curve, not operating thresholds.
- Minimum t-DCF and confidence intervals are not reported in E0.2.
