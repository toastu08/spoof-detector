# E1.4 — Is the model using spoofing cues or shortcuts? (pre-registration)

**Status:** pre-registered before any shortcut condition was exported, trained
or evaluated.
**Experiment ID:** E1.4 (ROADMAP phase 1).
**Date:** 2026-11-28.
**Git commit:** unknown at pre-registration; the exact commit and dirty-tree flag
are recorded in each run's `export_manifest.json`, `config.json` and
`metrics.json`.
**Data manifest:** `manifests/train_seed42.csv`
(SHA-256 `22f98abdf3392e04f7df3632809f81d033d5687773946cd103f5066846ed195e`,
10,080 rows: 2,580 bonafide / 7,500 spoof).

This document fixes the question, conditions, splits, metrics and decision rules
*before* results exist. Any later deviation must be logged in
`reports/phase1/e1_4_shortcuts/README.md`.

## Question

E1.1 held the rendering fixed and found mel warping is not required in-domain;
E1.2 found colormap, contrast normalisation and PNG encoding to be irrelevant,
and orientation a small colormap-dependent interaction. Neither experiment could
tell us whether the frozen backbone is reading **speech** or incidental
**non-speech content** (silence, noise, the zero-padding introduced by the
4-second fix) or whether the corpus makes the label recoverable from
**duration** or **metadata** alone.

E1.4 asks: can spoofing be detected from silence/noise segments, duration, or
metadata alone? If any of these reaches materially above-chance performance, the
full-image baseline is potentially contaminated by a shortcut and must be
reported with that caveat.

## Metric contract (AGENTS.md G4)

Every reported number states split, attack subset, threshold and sample count.

| Split | Role | Attack subset | Samples (bonafide / spoof) |
|---|---|---|---|
| train | fitted head/classifier only | A01–A06 | 10,080 (2,580 / 7,500) |
| dev | model selection | A01–A06 | 24,844 (2,548 / 22,296) |
| eval | clean estimate | A07–A19 | 71,237 (7,355 / 63,882) |

- **Primary metric:** threshold-free EER on **eval**. AUC is reported alongside.
- **Secondary metric:** EER on dev (model-selection split).
- **Threshold:** EER and AUC are threshold-free. The fixed `p >= 0.5` operating
  point is reported descriptively (`FPR@0.5`, spoof recall@0.5) and is **not**
  tuned. Dev-selected EER thresholds are recorded but not transferred to eval as
  a claimed operating point.
- Eval is never used for training, early stopping, or selection.
- **Seeds:** one run per condition (seed 42). Only differences of at least the
  pre-registered bars below are treated as robust; smaller gaps are suggestive.

## Segmentation and masking (fixed before any export)

- The fixed 4-second waveform is loaded exactly as the baseline does
  (`load_fixed_audio`, 16 kHz, 64,000 samples).
- Frame energy is `librosa.feature.rms(frame_length=2048, hop_length=512,
  center=True)` converted to dB relative to the frame peak
  (`amplitude_to_db(ref=np.max, top_db=None)`), yielding 126 frames aligned to
  the mel time axis.
- A frame is **speech** when its RMS energy is within `SPEECH_DB_GAP = 30` dB of
  the clip's per-frame energy peak; otherwise it is **non-speech** (silence,
  noise, or zero-padding).
- Masking is a hard time mask applied *after* the mel is computed and min-max
  normalised. Kept frames are pixel-identical to the corresponding frames of the
  `full` image; removed frames are set to `0.0` (the colormap bottom). A
  constant-energy clip has every frame marked speech, so its `nonspeech_only`
  image is a constant (all-black) image — the degenerate case is handled, not
  special-cased.

## Conditions

Three image conditions over the fixed 128-band mel representation, plus three
feature probes and one attribution analysis on the frozen baseline.

| Condition | Input | Model | Probes |
|---|---|---|---|
| `full` | unmodified mel128 (control) | frozen backbone + fresh head | control; must reproduce E1.1 `mel128` |
| `speech_only` | speech frames only | frozen backbone + fresh head | does speech carry everything? |
| `nonspeech_only` | silence + noise + padding only | frozen backbone + fresh head | the central shortcut probe |
| `duration_only` | clip duration in seconds (before fix) | logistic regression | duration shortcut |
| `metadata_system_id` | `system_id != "-"` (spoof-system presence) | logistic regression | metadata ceiling |
| `metadata_speaker_id` | one-hot speaker identity | logistic regression | speaker leak (expected none) |
| `silence_trim` | bonafide clips, silence trimmed | frozen baseline (no retrain) | does the trained model lean on silence? |

### Image conditions (A)

Same frozen-backbone recipe as E1.1/E1.2: `torchvision` ResNet18
(`weights="DEFAULT"`), all backbone parameters frozen but BatchNorm in train
mode, head `Linear(512,128) → ReLU → Dropout(0.5) → Linear(128,1)` freshly
re-initialised with `torch.manual_seed(42)`, unweighted `BCEWithLogitsLoss`,
Adam `lr=1e-3`, batch size 32, at most 20 epochs, early stopping patience 5 on
full-dev BCE loss, seed 42.

### Feature probes (B)

A single `nn.Linear(d, 1)` logistic regression with `BCEWithLogitsLoss`, Adam
`lr=1e-2`, up to 2,000 full-batch epochs, early stopping patience 200 on full-dev
BCE loss, seed 42. Features are z-score standardised with train statistics.

- `duration_only`: 1 feature, the clip duration in seconds read from the source
  FLAC header before the 4-second fix.
- `metadata_system_id`: 1 binary feature, `1` iff `system_id != "-"`. Because the
  CM protocol labels every bonafide row `-` and every spoof row `A01`–`A19`, this
  feature is the label; it documents the ceiling the image model cannot reach.
- `metadata_speaker_id`: one-hot over train speakers; unseen dev/eval speakers
  map to the zero vector, so the model cannot transfer a speaker-specific signal
  across splits.

### Attribution (C)

`silence_trim` scores each bonafide clip twice with the frozen **unweighted**
baseline (`Models/Iteration1/best_audio_spoof_model_unweighted.pth`): once on the
fixed 4-second input and once after leading/trailing non-speech frames are
removed and the clip re-fixed to 4 seconds. It reports the paired score shift
and the fraction of clips that flip across `p = 0.5`. Frame-level trim
boundaries are approximate (documented in README.md).

## Validity gates

1. `full` is pixel-identical to `data/dataset_img` (sampled clips per split).
2. `speech_only` and `nonspeech_only` masks are exact complements (every frame in
   exactly one).
3. The `full` probe reproduces E1.1 `mel128` eval EER (11.41%) within 0.5 points.
4. Re-render determinism passes for every condition and split.

A failed gate invalidates that condition's row and must be fixed before results
are reported.

## Decision rules

- **Silence/noise shortcut claimed** iff `nonspeech_only` eval AUC **≥ 0.70**;
  **clean** iff **≤ 0.60**; between 0.60 and 0.70 is ambiguous and unresolved for
  a single seed.
- **Speech sufficiency** iff `speech_only` is within **1.5 EER points** of `full`
  on eval.
- **Duration shortcut flagged** iff `duration_only` eval AUC **≥ 0.60**.
- **Metadata ceiling documented** when `metadata_system_id` eval AUC **≥ 0.95**
  (expected, and *not* a claim that the image model sees metadata); **speaker
  leak ruled out** when `metadata_speaker_id` eval AUC **≤ 0.60** (expected).
- **Silence reliance in the trained model** is evidenced if `silence_trim` shows
  a positive median score shift (trimming silence pushes bonafide toward spoof)
  and a non-trivial flip rate; a near-zero shift and flip rate support "the
  trained model does not lean on silence".

## Artifacts

- Images: `data/derived/e1_4_shortcuts/<condition>/{train,dev,eval}/` (gitignored).
- Per-condition manifests: `manifests/e1_4_shortcuts/<condition>_<split>.csv`
  (gitignored).
- Image probes: `Models/Iteration1/e_1_4/<condition>/seed42/` (checkpoint,
  `config.json`, `train_log.csv`).
- Feature probes: `Models/Iteration1/e_1_4/<probe>/seed42/` (`feature_probe.pth`,
  `config.json`).
- Evaluation: `reports/phase1/e1_4_shortcuts/runs/<condition>/seed42/` (image) and
  `reports/phase1/e1_4_shortcuts/runs/<probe>/seed42/metrics.json` (feature).
- Attribution: `reports/phase1/e1_4_shortcuts/silence_trim_bonafide.{csv,json}`.
- Aggregate: `reports/phase1/e1_4_shortcuts/{summary.csv, per_attack_eer.csv,
  feature_probes.csv, RESULTS.md, export_manifest.json}`.
- Final report: `reports/phase1/e1_4_shortcuts/README.md`.

## Limitations stated up front

- **Single seed.** No confidence intervals or significance tests; the E1.1/E1.2
  experience is that 1–3 point gaps are within unmeasured training noise.
- The speech/non-speech split is an energy heuristic (30 dB below the frame
  peak), not a phonetic VAD; it can mislabel low-energy speech and loud noise.
- Masked frames are set to the colormap bottom, which removes *content* but does
  not preserve the absolute frame count as a visible cue in the same way the
  baseline sees it (the 4-second padding boundary is still present in
  `nonspeech_only`).
- Feature probes are a different model class (logistic regression), so their
  numbers are not directly comparable to the ResNet18 EER/AUC beyond the
  "above chance" reading.
- Dev is a model-selection split; only eval is a clean estimate.
- One corpus only (ASVspoof 2019 LA); silence/noise patterns are a property of
  this corpus's recording and spoofing conditions.
- The E1.4 image sets are outside `make audit-data`'s current scope. They are
  rendered from the audited `data/LA` with the audited audio contract, and
  `full` is pixel-identical to the audited export, but the new PNGs are not
  themselves hash-audited.
