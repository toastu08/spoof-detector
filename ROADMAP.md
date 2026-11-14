# ROADMAP.md

Prioritized research roadmap for the audio spoof detection project.

Read [`AGENTS.md`](./AGENTS.md) first. It contains the operating rules, verified
current state, and hard guardrails. This document assumes those rules.

Last updated: 2026-11-14.

**Current status (2026-11-14):** E0.1, E0.2, E0.3, E0.4, and E0.5 are complete.
The 10,080-clip training subset is reconstructed at
`data/derived/train_seed42/` with manifest `manifests/train_seed42.csv`; the
evaluation harness scores full dev/eval reproducibly; the two-checkpoint baseline
is frozen at `reports/baseline/README.md`; preprocessing, checkpoint-loading, and
app contract tests pass via `make test`; and the data and leakage audit finds no
train/dev/eval leakage (`reports/audit/data_audit.md`). Phase 0 is complete, so
Phase 1 can start; E1.4 is the next priority.

---

## 1. Central research question

> How much of audio spoof detection can be achieved with a frozen ImageNet
> ResNet18 reading magma-coloured mel spectrograms, and where does that approach
> break?

The current model is a useful probe, not a finished detector. The roadmap is
designed to answer four questions in order:

1. **Is the baseline real and reproducible?**
2. **Why does it work?**
3. **Where does it stop working?**
4. **What should replace or extend it?**

Do not skip ahead. A finding from Phase 2 is only meaningful if the Phase 0
baseline is trustworthy.

---

## 2. Guiding hypotheses

These are working hypotheses, not conclusions.

- **H1. Frozen features are sufficient for the easy cases.** ImageNet features
  already encode enough texture and edge statistics to separate many vocoder
  artefacts from natural speech.
- **H2. The model may be reading artefacts, not speech.** Performance could be
  driven by codec noise, silence, padding, duration, or colormap statistics
  rather than speaker-independent spoofing cues.
- **H3. Generalisation will collapse under channel shift.** A model trained on
  ASVspoof 2019 LA is expected to degrade on unseen codecs, languages, and
  recording conditions.
- **H4. More data will help less than a better front-end or backbone.** The
  frozen linear probe may saturate quickly; the next gain may come from
  unfreezing, a waveform model, or a self-supervised speech model.
- **H5. Calibration is currently unmeasured.** The sigmoid score and the app's
  `max(p, 1-p)` are not calibrated probabilities.

Each experiment below should make at least one hypothesis more or less likely.

---

## 3. How to use this roadmap

### Phase 0 is a gate

Phase 0 must pass before any Phase 1-4 experiment is treated as a result. The
gate exists because the original training images were overwritten. The training
manifest (E0.1), evaluation harness (E0.2), frozen two-checkpoint baseline
(E0.3), contract tests (E0.4), and data/leakage audit (E0.5) now exist, so the
gate is closed and new numbers can be compared to a recorded baseline.

### Experiment format

Every experiment is written as:

- **Question**: what we want to learn.
- **Method**: the smallest credible test.
- **Success criterion**: what counts as a useful answer, decided before running.
- **Artifact**: what must exist in the repository afterwards.

### Metric contract

Every result table must state:

1. Split: train, dev, eval, or external.
2. Attack subset: A01-A06, A07-A19, all eval attacks, or a named corpus.
3. Threshold: exact value, or "threshold-free" for EER/AUC.
4. Sample count and class balance.

Report at least:

- EER (threshold-free, primary)
- AUC (threshold-free, secondary)
- Accuracy and per-class recall at a stated threshold
- Per-attack EER, not just an aggregate
- Calibration metrics when probabilities are discussed (ECE, Brier)

Report both the fixed `0.5` threshold and the dev-selected EER threshold. Select
thresholds on dev only; never tune on eval or external test data. Min t-DCF is
out of scope until ASV scores are available.

### Decision rules

- A change is kept only if it improves EER on dev **and** does not degrade the
  worst per-attack EER without a documented trade-off.
- A result is not a result until it is reproducible from a recorded command,
  config, seed, and data manifest.
- Negative results are recorded. They are how the capability envelope is mapped.
- Do not compare numbers across different preprocessing versions, splits, or
  thresholds.

---

## 4. Phase 0 - Baseline harness and reproducibility (gate)

**Goal:** make the existing 10,080-clip baseline exactly reproducible and
measurable before changing anything.

**Why this is first:** the current `data/dataset_img/train` is the full 25,380-image
export, not the original training subset, and the original PNGs are gone. E0.1
has reconstructed the subset and frozen its manifest; the evaluation harness,
baseline table, contract tests, and leakage audit now exist, so the Phase 0 gate
is closed.

### E0.1 Reconstruct the 10,080-clip training manifest

**Status: complete (2026-11-07).** Report:
[`reports/phase0/E0.1/E0.1_provenance.md`](reports/phase0/E0.1/E0.1_provenance.md).

- **Question:** Can we recover the exact first-iteration training set from source
  audio?
- **Method:** Follow the recipe recorded in `1.1_data_prep-train-old.ipynb`.
  Regenerate images from `data/LA/` into a new directory
  (`data/derived/train_seed42/`, never overwriting existing data). Record source
  FLAC path, protocol row, speaker, attack, label, and SHA-256 for every clip.
- **Success criterion:** Exactly 10,080 rows, with 2,580 bonafide and 7,500
  spoof; attack and speaker coverage match the original; regenerated images are
  pixel-identical to any surviving reference samples.
- **Artifact:** `manifests/train_seed42.csv` plus a regeneration script and a
  short provenance note. If exact reconstruction fails, document why and stop.
- **Result:** 10,080 rows (2,580 / 7,500); 20/20 notebook anchors match; all 20
  speakers and A01–A06 covered; 10,080/10,080 regenerated images pixel-identical
  to the full-split export. Manifest SHA-256
  `22f98abdf3392e04f7df3632809f81d033d5687773946cd103f5066846ed195e`.
- **Caveat:** the original 10,080 PNGs are lost, so pixel parity is against the
  later full-split export; membership and order are proven by the notebook
  anchors. Use `manifests/train_seed42.csv` for all training.

### E0.2 Build the evaluation harness

**Status: complete (2026-11-09).** Report:
[`reports/phase0/E0.2/README.md`](reports/phase0/E0.2/README.md). Baseline
output: `reports/baseline/best_audio_spoof_model_unweighted/`.

- **Question:** What is the baseline performance on dev and eval, including
  per-attack behaviour?
- **Method:** Add a batch inference CLI in `src/spoof_detection/evaluate.py`.
  It loads a checkpoint, verifies the key mapping, scores every image in a split,
  and writes per-clip scores plus aggregate metrics. Use the ASVspoof CM protocol
  and metadata row order. Do not train or tune anything.
- **Success criterion:** One command reproduces the same metrics twice; output
  includes EER, AUC, accuracy at 0.5, per-class recall, and per-attack EER with
  sample counts.
- **Artifact:** `src/spoof_detection/evaluate.py`,
  `reports/baseline/<checkpoint>/metrics.json`, and a readable Markdown summary.
- **Result:** Unweighted checkpoint, image input, dev/eval full splits. Dev
  (A01–A06, 24,844 clips, 2,548/22,296): EER 6.16%, AUC 0.9860. Eval (A07–A19,
  71,237 clips, 7,355/63,882): EER 11.50%, AUC 0.9527. Per-attack eval EER is
  1.04%–6.98% for A07–A16 but 36.24% for A17 and 18.30% for A18/A19. Dev EER
  threshold `0.839741` transfers to eval at 5.02% FPR / 82.55% spoof recall.
  Two full runs produced byte-identical score CSVs and identical metrics.
  Reading stored PNGs and re-rendering from source audio gave bit-identical
  scores on a 100-clip-per-split smoke subset.

### E0.3 Freeze the baseline

**Status: complete (2026-11-13).** Report:
[`reports/phase0/E0.3/README.md`](reports/phase0/E0.3/README.md). Frozen table:
[`reports/baseline/README.md`](reports/baseline/README.md). Audit:
`reports/baseline/freeze_report.json`.

- **Question:** How do the unweighted and weighted first-iteration checkpoints
  compare under the same harness?
- **Method:** Evaluate both checkpoints on dev and eval. Confirm checkpoint
  parameter counts and that the head is not randomly initialized. Record the git
  commit and manifest hash.
- **Success criterion:** A single baseline table exists for both checkpoints,
  with all four metric-contract fields. A second run reproduces the table within
  a pre-registered tolerance (suggested: +/-1.0 absolute EER point).
- **Artifact:** `reports/baseline/README.md` with the frozen baseline table and
  the exact commands used.
- **Result:** Both checkpoints load strictly (124/124 keys remapped, zero
  missing/unexpected, 11,242,305 parameters, 65,793 head parameters). Dev
  (A01–A06, 24,844 clips, 2,548/22,296) and eval (A07–A19, 71,237 clips,
  7,355/63,882), threshold-free: unweighted dev EER 6.16% / AUC 0.9860, eval
  EER 11.50% / AUC 0.9527; weighted dev EER 5.42% / AUC 0.9881, eval EER 11.13%
  / AUC 0.9558. At `0.5`, weighting cuts eval FPR from 16.55% to 9.42% but lowers
  spoof recall from 90.99% to 87.76%. The second full run reproduced every EER
  and AUC at exactly zero difference and every score CSV byte-for-byte. Head
  control on 1,000 dev clips (500/500): trained AUC 0.9994 vs reinitialised AUC
  0.5689 (unweighted) and 0.5590 (weighted); both pass the pre-registered gate.
  Dev is labelled a model-selection split because the notebook used it for early
  stopping; eval is the clean estimate.

### E0.4 Preprocessing and model-load contract tests

- **Question:** Can a future change silently break inference?
- **Method:** Add tests for: audio loading and 64,000-sample fixing; mel and dB
  computation; min-max normalization; colormap and vertical flip; resize and
  ImageNet normalization; checkpoint remap with zero missing/unexpected keys;
  app rejection of unsupported file types; cleanup of temporary files.
- **Success criterion:** Tests pass in a clean environment. A deliberately
  broken key remap fails the test instead of loading silently.
- **Artifact:** `tests/test_preprocessing.py`, `tests/test_model_loading.py`,
  and `tests/test_app_contract.py`.
- **Status: complete (2026-11-14).** `make test` ran 116 tests at E0.4 and
  passes (138 now with the E0.5 audit suite). The
  app now loads its checkpoint through the strict package loader and
  refuses to start on a missing or mismatched state dict; a deliberately broken
  prefix raises instead of loading silently. A full image-input rerun of both
  checkpoints on dev and eval reproduced all four score CSVs byte-for-byte and
  every EER/AUC at zero difference, and image vs source-audio input gave
  bit-identical logits on a deterministic 100-clip-per-split subset. Report:
  `reports/phase0/E0.4/README.md`.

### E0.5 Data and leakage audit

- **Question:** Is the evaluation set genuinely held out?
- **Method:** Check protocol row alignment, duplicate files, speaker overlap
  between train/dev/eval, and whether any dev/eval clip appears in the training
  manifest. Confirm the 838 `_A` enrollment files are excluded from CM scoring.
  Distinguish documented dev reuse from eval leakage: E0.3 selected the
  operating threshold on dev and the original training loop used full dev for
  early stopping, so dev is a model-selection split. The audit must confirm that
  **no eval clip** entered training, early stopping, or threshold selection.
- **Success criterion:** A written audit states exactly which files are used for
  each split, documents the dev reuse, and confirms no eval leakage into training
  or threshold selection.
- **Artifact:** `reports/audit/data_audit.md` with counts and known exclusions.
- **Status: complete (2026-11-14).** All 15 checks pass. No cross-split speaker,
  filename, audio-hash, or image-hash collisions; zero training clips in dev/eval;
  the 838 enrollment files are excluded; and every operating threshold traces to
  dev only. The only duplicate content is the expected byte-identical overlap
  between `data/derived/train_seed42/` and the full train export, both labelled
  `train`. Reproduce with `make audit-data`. Reports:
  `reports/audit/data_audit.md` and `reports/phase0/E0.5/README.md`.

### Phase 0 exit criteria

Phase 0 is complete only when all of the following are true:

- [x] The 10,080-clip manifest exists and is reproducible from source audio.
      (E0.1, 2026-11-07.)
- [x] The evaluation harness runs on dev and eval from one documented command.
      (E0.2, 2026-11-09: `make evaluate`.)
- [x] Both checkpoints have a frozen baseline table with per-attack EER.
      (E0.3, 2026-11-13: `reports/baseline/README.md`.)
- [x] Preprocessing and model-loading tests pass.
      (E0.4, 2026-11-14: `make test`.)
- [x] The data audit finds no train/dev/eval leakage.
      (E0.5, 2026-11-14: `make audit-data`; `reports/audit/data_audit.md`.)

If any item fails, fix it before starting Phase 1.

---

## 5. Phase 1 - Why does it work?

**Goal:** explain the mechanism behind the baseline. Separate genuine spoofing
cues from dataset and preprocessing shortcuts.

### E1.1 Is the mel spectrogram necessary?

- **Question:** Does the model need a mel spectrogram, or would any image-like
  time-frequency representation work?
- **Method:** With the backbone frozen and the same train/dev/eval splits, compare
  mel spectrograms at `n_mels` 64/128/256, linear-frequency spectrograms, MFCC
  images, and a random-projection control. Keep the classifier head and training
  budget fixed.
- **Success criterion:** An EER table that shows whether performance is sensitive
  to mel resolution and whether a non-mel representation is competitive.
- **Artifact:** `reports/phase1/e1_1_frontend/` with configs and result tables.

### E1.2 Do colormap and orientation matter?

- **Question:** Is the model reading spectral content, or incidental image
  statistics introduced by `magma` and the vertical flip?
- **Method:** Ablate colormap (`magma`, `viridis`, grayscale, raw dB) and
  orientation (flipped vs not flipped). Retrain the head for each condition on
  the same split. Include an RGBA-vs-RGB check.
- **Success criterion:** If grayscale matches `magma`, the colormap is cosmetic.
  If removing the flip causes a large EER change, orientation is a shortcut and
  must be documented.
- **Artifact:** `reports/phase1/e1_2_rendering/` with an ablation matrix.

### E1.3 Is the frozen backbone the right compute trade-off?

- **Question:** How much performance is left on the table by freezing ResNet18?
- **Method:** Compare four conditions at equal data: head-only training, unfreeze
  `layer4`, unfreeze the last two blocks, and full fine-tuning. Use the same
  schedule and report GPU-hours.
- **Success criterion:** An EER-vs-GPU-hour Pareto table that either confirms the
  frozen choice or identifies a better operating point.
- **Artifact:** `Models/IterationN/` checkpoints and
  `reports/phase1/e1_3_finetuning/`.

### E1.4 Is the model using spoofing cues or shortcuts?

- **Question:** Can the model detect spoofing from silence, noise, duration, or
  metadata alone?
- **Method:** Train control probes on: silence-only segments, noise-only
  segments, non-speech segments, duration-only features, and metadata-only
  features. Also evaluate on bonafide clips with silence trimmed. Compare against
  the full-image baseline.
- **Success criterion:** Quantify above-chance shortcut performance. If any
  control probe reaches a materially non-trivial AUC, the baseline result is
  contaminated and must be reported with that caveat.
- **Artifact:** `reports/phase1/e1_4_shortcuts/` with control-probe results.

### E1.5 How sensitive is the model to duration and padding?

- **Question:** Does the 4-second fixed-length contract create artefacts that the
  model exploits?
- **Method:** Evaluate the same clips at 1, 2, 4, and 8 seconds, and compare
  padding, tiling, and centre-cropping. Keep the model fixed; change only the
  input construction.
- **Success criterion:** An EER-vs-duration curve and a recommendation for the
  canonical input contract.
- **Artifact:** `reports/phase1/e1_5_duration/` and an updated preprocessing note
  if the contract changes.

---

## 6. Phase 2 - Capability envelope

**Goal:** measure where the model works and where it fails. This phase produces
the evidence needed to decide whether to scale the current approach.

### E2.1 Cross-dataset evaluation

- **Question:** Does performance survive a change of corpus and recording
  condition?
- **Method:** Evaluate the frozen baseline on ASVspoof 2019 LA eval (in-domain),
  ASVspoof 2021 DF/LA, WaveFake, In-the-Wild, and any licensed external corpus
  with bonafide controls. Do not fine-tune on the target corpus.
- **Success criterion:** EER and AUC per corpus, with the domain gap stated
  explicitly. Identify the worst-performing corpus and attack family.
- **Artifact:** `reports/phase2/e2_1_cross_dataset/` with a benchmark table and
  licensing notes.

### E2.2 Codec and channel robustness

- **Question:** How quickly does performance degrade under realistic delivery
  conditions?
- **Method:** Transcode eval audio to MP3 (128/64/32 kbps), AAC, and Opus;
  resample to 8/16/22.05/44.1 kHz; add noise at 20/10/5/0 dB SNR; apply reverb and
  clipping. Recompute spectrograms with the training pipeline.
- **Success criterion:** An EER-vs-degradation matrix and a documented breaking
  point. Report which transformation hurts most.
- **Artifact:** `reports/phase2/e2_2_robustness/` with scripts and matrices.

### E2.3 Unseen attack families

- **Question:** Does the model generalise to vocoders and TTS systems it has
  never seen?
- **Method:** Train on A01-A06 only, evaluate on A07-A19. Also run a
  leave-one-attack-out analysis. Keep the training set and preprocessing fixed.
- **Success criterion:** Seen-vs-unseen EER gap and a per-attack table. A large gap
  means the model is fitting attack-specific artefacts.
- **Artifact:** `reports/phase2/e2_3_unseen_attacks/` with per-attack EER and
  confidence intervals where sample sizes allow.

### E2.4 Speaker and language shift

- **Question:** Is the detector speaker-independent and language-independent?
- **Method:** Use speaker-disjoint splits where the corpus permits. Evaluate on
  multilingual external data with bonafide controls. Report per-speaker and
  per-language metrics.
- **Success criterion:** Quantified gap between matched and mismatched
  speaker/language conditions; identify any systematic bias.
- **Artifact:** `reports/phase2/e2_4_speaker_language/`.

### E2.5 Calibration and thresholding

- **Question:** Can the sigmoid score be turned into a trustworthy probability?
- **Method:** Compute reliability diagrams, ECE, and Brier score on dev. Fit
  temperature scaling or isotonic regression on dev only, then evaluate on eval.
  Report accuracy and false-positive rate at both 0.5 and the dev-selected EER
  threshold.
- **Success criterion:** Calibration error is reduced without changing the ranking
  (AUC). Only after this passes may the app call its output "calibrated".
- **Artifact:** `reports/phase2/e2_5_calibration/` and, if successful, an updated
  inference contract.

---

## 7. Phase 3 - Scaling and architecture

**Goal:** decide whether to invest in more data, more compute, a different
front-end, or a different model family.

### E3.1 Data scaling curve

- **Question:** How much does performance improve with more training data?
- **Method:** Train the frozen-head model on 2.5k, 5k, 10k, 25k, and the full
  available training set. Keep epochs, preprocessing, and evaluation fixed.
- **Success criterion:** A learning curve with marginal gain per doubling of data.
  If the curve is flat after 10k, more data is not the priority.
- **Artifact:** `Models/IterationN/` checkpoints and
  `reports/phase3/e3_1_data_scaling/`.

### E3.2 Unfreezing schedule

- **Question:** What is the best accuracy-per-GPU-hour way to fine-tune?
- **Method:** Compare gradual unfreezing, last-block-only, full fine-tuning, and
  frozen batch-norm variants. Use discriminative learning rates and early
  stopping on dev.
- **Success criterion:** EER and GPU-hours for each schedule; a recommended
  default for future iterations.
- **Artifact:** `reports/phase3/e3_2_unfreezing/` and `MODELS.md` entries.

### E3.3 Class imbalance and loss functions

- **Question:** Does the 1:2.9 bonafide/spoof imbalance distort the decision
  boundary?
- **Method:** Compare unweighted cross-entropy, class weights, balanced sampling,
  focal loss, and a ranking/AUC-style loss. Evaluate per-class recall and EER,
  not accuracy alone.
- **Success criterion:** A loss comparison that states which operating point is
  best for equal error and which is best for high spoof recall.
- **Artifact:** `reports/phase3/e3_3_losses/` with per-class metrics.

### E3.4 Backbone bake-off

- **Question:** Is a frozen ImageNet ResNet18 the right feature extractor?
- **Method:** Under the same data and evaluation harness, compare ResNet18,
  EfficientNet-B0, a spectrogram transformer (AST), a self-supervised speech
  model (e.g. wav2vec2 features), AASIST, and RawNet2. Include at least one
  waveform-native baseline.
- **Success criterion:** EER vs parameter count vs latency. A non-image model must
  beat the image baseline by a meaningful margin to justify the added complexity.
- **Artifact:** `reports/phase3/e3_4_backbones/` with a comparison table.

### E3.5 Front-end search

- **Question:** Is the current mel/`magma` front-end near-optimal?
- **Method:** Sweep `n_mels`, hop length, power vs amplitude, dB reference, PCEN,
  CQT, and a learnable mel front-end. Retrain the head for each condition.
- **Success criterion:** Best EER per unit of compute, with the winning front-end
  documented as a new preprocessing version.
- **Artifact:** `reports/phase3/e3_5_frontend/` and an updated contract if the
  front-end changes.

---

## 8. Phase 4 - Applications and deployment

**Goal:** turn the research model into a demo that is honest about its limits and
safe to expose.

### E4.1 Streaming and chunked inference

- **Question:** Can the detector run on short windows with acceptable latency?
- **Method:** Evaluate 1-4 second windows with overlap and score aggregation.
  Measure real-time factor on CPU and GPU.
- **Success criterion:** Real-time factor below 1 on the target hardware and a
  documented EER delta versus full 4-second inference.
- **Artifact:** `reports/phase4/e4_1_streaming/` with latency and EER tables.

### E4.2 Explainability and failure analysis

- **Question:** What evidence does the model use, and where does it fail?
- **Method:** Apply Grad-CAM and time-frequency occlusion to the mel image.
  Build a failure gallery covering the worst per-attack and external cases.
- **Success criterion:** A human-auditable explanation for a sample of correct and
  incorrect predictions, plus a list of systematic failure modes.
- **Artifact:** `reports/phase4/e4_2_explainability/` with figures and case notes.

### E4.3 Red-team evaluation

- **Question:** How easy is it to evade the detector?
- **Method:** Test new TTS/VC systems, adversarial perturbations, replay,
  splicing, and codec chains. Measure attack success rate at a fixed false-positive
  rate.
- **Success criterion:** A documented attack-success curve and a clear statement of
  what the detector does not protect against.
- **Artifact:** `reports/phase4/e4_3_redteam/`.

### E4.4 Deployment hardening

- **Question:** Is the demo safe and operable outside a trusted laptop?
- **Method:** Add upload size limits, MIME sniffing, CORS allowlisting, safe temp
  file handling, model versioning, health checks, and structured logging. Package
  the app in a container. Load-test it.
- **Success criterion:** A security checklist passes and the app returns a clear
  error for malformed, oversized, and unsupported uploads.
- **Artifact:** `deploy/` configuration, `tests/test_app_contract.py`, and an
  operations note.

---

## 9. Cross-cutting risks

| Risk | Why it matters | Mitigation |
|---|---|---|
| Training-set reconstruction drift | Baseline membership can silently change | E0.1 passed; always use `manifests/train_seed42.csv` and verify its SHA-256 |
| Eval overfitting | Numbers become optimistic | Select thresholds on dev only; freeze eval until a release |
| Codec confound | Model may learn channel, not spoofing | E2.2 and E1.4 controls |
| Label noise | ASVspoof labels may not reflect all real-world spoofing | Audit suspicious clips; report per-attack results |
| External corpus licensing | Cannot redistribute audio | Store manifests and scripts, not restricted audio |
| Compute limits | Full fine-tuning and large backbones are expensive | Report GPU-hours; prefer accuracy-per-compute comparisons |
| Notebook information loss | Original hyperparameters may stay undocumented | Notebooks are readable via MCP; mark inferred facts and re-derive the schedule |
| Silent checkpoint mismatch | Random head can look like a result | G7: assert zero missing/unexpected keys |
| Uncalibrated confidence | Misleading demo output | E2.5; never call the sigmoid calibrated before it passes |
| Small external test set | Eight clips cannot support broad claims | Use it only as a smoke test; add controlled corpora in E2.1 |

---

## 10. Prioritized next actions

E0.1–E0.5 are complete, so Phase 0 is closed. The remaining prioritized actions
are:

| Priority | Experiment | Reason |
|---:|---|---|
| 1 | E1.4 | Cheapest high-information test: is the baseline using shortcuts? |
| 2 | E1.2 | Clarifies whether the image rendering is doing real work |
| 3 | E2.1 | Establishes the domain gap before investing in scaling |
| 4 | E1.3 | Tests whether unfreezing is worth the compute |
| 5 | E3.1 | Guides the data-vs-architecture decision |
| 6 | E2.5 | Required before the app can report honest confidence |

Do not start a lower-priority experiment while a higher-priority gate item is
unresolved, unless the user explicitly reprioritizes.

---

## 11. Experiment log template

Copy this into `reports/<phase>/<experiment-id>/README.md`:

````markdown
# <Experiment ID> - <Title>

## Question


## Hypothesis


## Data
- Manifest:
- Manifest SHA-256:
- Split:
- Attack subset:
- Sample count / class balance:

## Preprocessing version


## Model
- Checkpoint(s):
- Frozen layers:
- Parameters:

## Training config
- Loss:
- Class weights:
- Optimizer / LR:
- Batch size:
- Epochs:
- Seed:

## Evaluation
- Threshold:
- EER:
- AUC:
- Accuracy / per-class recall:
- Per-attack EER:
- Calibration (ECE / Brier):

## Result summary


## Interpretation


## Limitations and failure cases


## Artifact paths


## Reproduction command

```bash
# exact command
```
````

---

## 12. Definition of done for a phase

A phase is done when:

- [ ] Every experiment has a report using the template above.
- [ ] Every metric follows the contract in section 3.
- [ ] The winning and losing conditions are both documented.
- [ ] Artifacts are reproducible from a recorded command and manifest hash.
- [ ] New checkpoints live under `Models/IterationN/` with a complete
      `MODELS.md`.
- [ ] Known limitations are written down before the next phase starts.
