# E1.2 — Do colormap and orientation matter? (pre-registration)

**Status:** pre-registered before any rendering condition was exported, trained
or evaluated.
**Experiment ID:** E1.2 (ROADMAP phase 1).
**Date:** 2026-11-24.
**Git commit:** unknown at pre-registration; the exact commit and dirty-tree flag
are recorded in each run's `export_manifest.json`, `config.json` and
`metrics.json`.
**Data manifest:** `manifests/train_seed42.csv`
(SHA-256 `22f98abdf3392e04f7df3632809f81d033d5687773946cd103f5066846ed195e`,
10,080 rows: 2,580 bonafide / 7,500 spoof).

This document fixes the question, conditions, splits, metrics and decision rules
*before* results exist. Any later deviation must be logged in
`reports/phase1/e1_2_rendering/README.md`.

## Question

E1.1 held the rendering pipeline fixed and found that mel warping is not
required for baseline-like in-domain performance. E1.2 asks the complementary
question: is the frozen ImageNet ResNet18 reading the **spectral content**, or
incidental **image statistics** introduced by the `magma` colormap and the
vertical flip?

The working hypothesis is that a colormap with rich multi-channel RGB variation
(`magma`, `viridis`) fits ImageNet features better than grayscale, and that
removing the flip changes performance if orientation is a shortcut.

## Metric contract (AGENTS.md G4)

Every reported number states split, attack subset, threshold and sample count.

| Split | Role | Attack subset | Samples (bonafide / spoof) |
|---|---|---|---|
| train | fitted head only | A01–A06 | 10,080 (2,580 / 7,500) |
| dev | model selection (head epoch) | A01–A06 | 24,844 (2,548 / 22,296) |
| eval | clean estimate | A07–A19 | 71,237 (7,355 / 63,882) |

- **Primary metric:** threshold-free EER on **eval**. AUC is reported alongside.
- **Secondary metric:** EER on dev (model-selection split).
- **Threshold:** EER and AUC are threshold-free. The fixed `p >= 0.5` operating
  point is reported descriptively (`FPR@0.5`, spoof recall@0.5) and is **not**
  tuned. The dev EER threshold is recorded but is not transferred to eval as a
  claimed operating point in the primary table.
- Eval is never used for training, early stopping, head selection, or threshold
  selection.
- **Seeds:** one run per condition (seed 42). The report must state that the
  1–3 point differences are not resolvable against unmeasured training-seed
  variance; only differences of at least 5 points are treated as robust.

## Conditions

All conditions use the same 128-band mel representation
(`power_to_db(ref=np.max)`, 16 kHz, 4 s = 64,000 samples). Only rendering changes.

| Condition | Colormap | Orientation | dB scaling | PNG encoding | Role |
|---|---|---|---|---|---|
| `magma_flip` | magma | low frequency at bottom (`origin='lower'`) | per-clip min-max | RGBA | **control** (existing contract) |
| `viridis_flip` | viridis | low frequency at bottom | per-clip min-max | RGBA | rich-RGB alternative |
| `gray_flip` | gray | low frequency at bottom | per-clip min-max | RGBA | no chroma |
| `rawdb_flip` | gray | low frequency at bottom | fixed `[-80, 0]` dB, no min-max | RGBA | contrast-normalisation probe |
| `magma_noflip` | magma | low frequency at top (`origin='upper'`) | per-clip min-max | RGBA | orientation |
| `viridis_noflip` | viridis | low frequency at top | per-clip min-max | RGBA | orientation × colormap |
| `gray_noflip` | gray | low frequency at top | per-clip min-max | RGBA | orientation × colormap |
| `magma_flip_rgb` | magma | low frequency at bottom | per-clip min-max | RGB (PIL) | encoding control |

`magma_flip` must reproduce the E1.1 `mel128` export and `data/dataset_img`
pixel-for-pixel. The export records a sampled control parity check, a RGB/RGBA
encoding parity check, flip-pair parity checks, and a re-render determinism check
per condition and split.

`rawdb_flip` still uses `power_to_db(ref=np.max)`, which is part of the fixed mel
contract. It removes only the per-clip min-max contrast stretch, so it probes
contrast normalisation, **not** absolute recording gain. A true absolute-gain
probe would require `ref=1.0` and is out of scope.

## Fixed training recipe (head only)

- Backbone: `torchvision.models.resnet18(weights="DEFAULT")`, all parameters
  `requires_grad=False`. BatchNorm stays in **train** mode so running statistics
  update, matching `2_model.ipynb`.
- Head: `Linear(512,128) → ReLU → Dropout(0.5) → Linear(128,1)`, freshly and
  deterministically re-initialised with `torch.manual_seed(42)`.
- Loss: unweighted `BCEWithLogitsLoss`. Optimizer: Adam, `lr=1e-3`.
- Batch size 32, at most 20 epochs, early stopping with patience 5 on **full-dev
  BCE loss**. Best epoch = lowest full-dev loss.
- Seed 42 for Python/NumPy/torch, loader shuffling and worker seeding.
- Input transform: resize 224×224, ImageNet mean/std.

Deliberate difference from the notebook: a recorded seed (none was recorded
originally). BatchNorm statistics become condition-specific because of
baseline-faithful train-mode BN; this is documented, not hidden.

## Predictions / hypotheses

1. **H1 (colormap):** `magma_flip` and `viridis_flip` are close to each other and
   better than `gray_flip`; grayscale losing materially would support the
   ImageNet-RGB-statistics reading.
2. **H2 (orientation):** each no-flip condition is close to its flipped
   counterpart; a large gap would identify orientation as a shortcut.
3. **H3 (contrast normalisation):** `rawdb_flip` differs from `gray_flip`. A
   material `rawdb_flip` advantage would indicate that removing per-clip contrast
   normalisation preserves spoofing or channel information, a shortcut concern.
4. **H4 (encoding):** `magma_flip_rgb` is pixel-identical to `magma_flip` and
   therefore differs by approximately zero. Any material difference is a
   pipeline bug, not a result.

The report will also tabulate each colormap's actual RGB locus spread
(per-channel range, mean channel standard deviation, saturation and luminance
range) so “more RGB variation” is measured rather than assumed.

## Decision rules

- The primary comparison is the eval EER table across the eight conditions, with
  sample counts and attack subset stated. Dev is reported for model selection.
- **Colormap cosmetic:** the three flipped conditions are within **1.0 point**
  mean eval EER of each other.
- **Colormap difference claimed:** only at a mean eval EER gap of **≥5 points**.
  Gaps of 1–3 points are labelled suggestive; 3–5 points are labelled modest and
  unresolved for a single seed.
- **Orientation not a shortcut:** each no-flip condition is within **1.5 points**
  of its flip counterpart. A gap of **≥5 points** is an orientation shortcut and
  must be documented.
- **Contrast shortcut signal:** `rawdb_flip` at least **3 points** better than
  `gray_flip`.
- **Encoding control:** `magma_flip_rgb` within **0.5 points** of `magma_flip`;
  otherwise the export is invalidated and investigated before any other row is
  reported.
- **Validity gates:** `magma_flip` must reproduce the E1.1 `mel128` eval EER
  (11.41%) within 0.5 points; control and encoding parity must pass on every
  split; every condition must pass the re-render determinism check. A failed gate
  invalidates that condition's row and must be fixed before results are reported.
- No significance test is implemented. Per-attack spread and the single-seed
  limitation are the stability signals.

## Artifacts

- Images: `data/derived/e1_2_rendering/<condition>/{train,dev,eval}/` (gitignored).
- Per-condition manifests: `manifests/e1_2_rendering/<condition>_<split>.csv`
  (gitignored).
- Probes: `Models/Iteration1/e_1_2/<condition>/seed42/` (checkpoint,
  `config.json`, `train_log.csv`).
- Evaluation: `reports/phase1/e1_2_rendering/runs/<condition>/seed42/`.
- Aggregate: `reports/phase1/e1_2_rendering/{summary.csv,
  summary_by_condition.csv, per_attack_eer.csv, per_attack_by_condition.csv,
  colormap_statistics.csv, RESULTS.md, export_manifest.json}`.
- Final report: `reports/phase1/e1_2_rendering/README.md`.

## Limitations stated up front

- Head-only training on a frozen backbone; conclusions are about this transfer
  regime, not about rendering in general.
- **Single seed.** No confidence intervals or significance tests; the E1.1
  experience is that 1–3 point differences are within unmeasured training noise.
- Dev is a model-selection split (early stopping and threshold selection), so only
  eval is a clean estimate.
- The 224×224 resize and ImageNet normalisation are held fixed. Grayscale
  therefore becomes slightly non-gray after ImageNet normalisation because the
  per-channel means and standard deviations differ.
- The fixed DB range is `[-80, 0]` relative to each clip's peak, not an absolute
  recording level.
- One corpus only (ASVspoof 2019 LA); A17–A19 difficulty is a property of this
  corpus's attack families.
- The E1.2 image sets are outside `make audit-data`'s current scope. They are
  rendered from the audited `data/LA` with the audited audio contract, and
  `magma_flip` is pixel-identical to the audited export, but the new PNGs are not
  themselves hash-audited.
