# AGENTS.md

Operating guide for anyone (human or AI) working in this repository.

Read this file before changing code, data, models, or experiment claims. For the
prioritized research plan, read [`ROADMAP.md`](./ROADMAP.md). This file records
what is true today; the roadmap records what we intend to test next.

Last verified: 2026-11-14. Notebook MCP status re-verified 2026-11-13.

**Current status (2026-11-14):** E0.1, E0.2, E0.3, E0.4, and E0.5 complete. The
10,080-clip training subset is reconstructed at `data/derived/train_seed42/`
with manifest `manifests/train_seed42.csv`. The evaluation harness scores full
dev/eval from one command, both first-iteration checkpoints have a frozen,
reproducibility-audited baseline under `reports/baseline/`, and preprocessing,
checkpoint-loading, and app-contract tests pass via `make test`. The data and
leakage audit (`reports/audit/data_audit.md`) finds no train/dev/eval leakage, so
Phase 0 is complete and Phase 1 can start.

---

## 1. Project summary

This is a small research project on **audio spoofing / deepfake speech detection**.
The current detector converts a 4-second audio clip into a mel-spectrogram image,
colours it with the `magma` colormap, and classifies the image with a ResNet18.

The current model is effectively a **frozen ImageNet-pretrained ResNet18 feature
extractor plus a small linear probe**. The convolutional backbone was intentionally
frozen to save compute, and the hypothesis that this would work was confirmed by
the first iteration. The model outputs one sigmoid score; the positive class is
**spoof**, so `p >= 0.5` means "spoof" and `p < 0.5` means "bonafide".

The repository contains the original notebooks, exported spectrogram datasets,
two first-iteration checkpoints, a FastAPI demo app, and an ad hoc external test
set of synthetic speech clips. The 10,080-clip first-iteration training subset
has been reconstructed from source audio and recorded in
`manifests/train_seed42.csv` (section 4.4).

---

## 2. Research goals

### Primary goal

Build a trustworthy, reproducible evaluation of image-based audio spoof detection,
then determine how far the frozen-backbone approach generalises beyond the data it
was trained on.

### Near-term goals

1. Reconstruct the exact 10,080-clip training subset and make the baseline
   reproducible from source audio. **Complete (E0.1).**
2. Establish a baseline evaluation harness with per-attack, per-speaker, and
   threshold-aware reporting, and freeze the two-checkpoint baseline table.
   **Complete (E0.2/E0.3).**
3. Explain why the frozen ImageNet backbone works: spectral content, colormap
   statistics, or dataset shortcuts.
4. Measure the capability envelope under channel, codec, language, speaker, and
   unseen-attack shift.

### Longer-term goals

1. Decide whether to scale the current approach, change the architecture, or
   replace the image front-end with a waveform/self-supervised model.
2. Calibrate scores and define an operating threshold suitable for a real
   application.
3. Produce a documented demo that reports uncertainty and failure modes rather
   than a single uncalibrated "confidence" number.

### Non-goals for now

- Building a production-grade anti-spoofing system.
- Training large models from scratch.
- Optimising the demo UI.
- Claiming state-of-the-art performance on ASVspoof or any external benchmark.

---

## 3. Repository map

| Path | Purpose | Status |
|---|---|---|
| `1.1_data_prep-train-old.ipynb` | Canonical record of the original 10,080-clip training subset and its preprocessing | Read-only reference; access via python-notebook MCP (see G1) |
| `1.1_data_prep-train.ipynb` | Current train export notebook | Accessible via python-notebook MCP |
| `1.2_data_prep-dev.ipynb` | Dev export notebook | Accessible via python-notebook MCP |
| `1.3_data_prep-eval.ipynb` | Eval export notebook | Accessible via python-notebook MCP |
| `2_model.ipynb` | Original training/evaluation notebook | Accessible via python-notebook MCP |
| `app.py` | FastAPI inference demo | Working; strict checkpoint load and shared preprocessing; see section 4.5 |
| `templates/index.html` | Browser UI for the demo | Working |
| `Models/Itteration1/` | First-iteration checkpoints and `MODELS.md` | Existing; keep spelling for compatibility |
| `best_audio_spoof_model_unweighted.pth` | Root checkpoint actually loaded by `app.py` | Existing; do not overwrite |
| `data/LA/` | Extracted ASVspoof 2019 LA audio and protocols | Existing; do not overwrite |
| `data/archive.zip` | Kaggle download zip of the extracted `data/LA` tree | Deleted 2026-11-07 after verification; hashes kept in `manifests/asvspoof2019_la.meta.json` |
| `data/dataset_img/` | Exported spectrogram PNGs and metadata for train/dev/eval | Existing; do not overwrite |
| `data/derived/train_seed42/` | Reconstructed 10,080-image first-iteration training set (E0.1) | New; 257 MB; gitignored |
| `manifests/` | Data manifests and metadata | `train_seed42.csv` + `.meta.json`; `asvspoof2019_la.meta.json` (dataset verification) |
| `reports/` | Experiment reports | `phase0/E0.1/` … `phase0/E0.5/`; `audit/` |
| `reports/audit/` | E0.5 data/leakage audit report | `data_audit.md` + `data_audit.json` (PASS) |
| `scripts/` | Thin CLI entry points | `reproduce_train_seed42.py`, `verify_asvspoof2019_la.py`, `freeze_baseline.py`, `audit_data.py` |
| `tests/test1/` | Eight ElevenLabs clips plus `test1-results.csv` | Existing; do not overwrite |
| `tests/` | Automated tests | E0.1 reconstruction, dataset validation, metrics, model loading, evaluation, baseline audit, data audit, preprocessing contract, app contract |
| `src/spoof_detection/` | Reusable package code | `protocol.py`, `sampling.py`, `preprocess.py`, `dataset.py`, `model.py`, `metrics.py`, `evaluate.py`, `baseline.py`, `audit.py` |
| `reports/baseline/` | Frozen two-checkpoint `metrics.json`, per-clip scores, `summary.md`, and `freeze_report.json` | E0.3 two-checkpoint dev+eval baseline |
| `pyproject.toml`, `uv.lock` | Project metadata and locked dependencies | Source of truth for dependencies |
| `makefile` | `make app`, `make test`, `make verify-data`, `make audit-data`, `make evaluate`, `make evaluate-rerun`, `make freeze-baseline` | Working |
| `README.md` | Setup, data acquisition, demo, and limitations | Written; keep commands in sync |

---

## 4. Verified current state

Everything in this section was checked against the repository on 2026-11-13,
except where explicitly labelled "inferred" or "unverified".

### 4.1 Data inventory

ASVspoof 2019 Logical Access (LA) counts:

| Split | Bonafide | Spoof | Total |
|---|---:|---:|---:|
| Train | 2,580 | 22,800 | 25,380 |
| Dev | 2,548 | 22,296 | 24,844 |
| Eval | 7,355 | 63,882 | 71,237 |
| **Total** | **12,483** | **108,978** | **121,461** |

The current `data/dataset_img/{train,dev,eval}` exports contain the **full**
split counts above, stored as:

```text
data/dataset_img/<split>/bonafide/*.png
data/dataset_img/<split>/spoofed/*.png
data/dataset_img/<split>/metadata_<split>.csv
```

Metadata columns are `speaker_id,system_id,label,img_path`, where
`label=0` is bonafide and `label=1` is spoof. Metadata row order matches the
ASVspoof CM protocol order. The metadata does **not** include the original audio
file name.

Other data facts:

- `data/LA/` contains 838 extra `_A`-prefixed enrollment FLACs that are not in
  the CM protocols (142 dev, 696 eval). Do not mix them into CM evaluation.
- `data/archive.zip` was the Kaggle download zip of the extracted `data/LA`
  tree; it was deleted on 2026-11-07 after verification. Its hashes remain in
  `manifests/asvspoof2019_la.meta.json`.
- The original 10,080-image training subset has been **regenerated** from source
  audio at `data/derived/train_seed42/` (2,580 bonafide + 7,500 spoof, 257 MB,
  gitignored). The current `data/dataset_img/train` remains the later full
  25,380-image export and was not modified.
- The authoritative membership and provenance record is
  `manifests/train_seed42.csv` (SHA-256
  `22f98abdf3392e04f7df3632809f81d033d5687773946cd103f5066846ed195e`).
- Dataset acquisition and verification are documented in `README.md`.
  `scripts/verify_asvspoof2019_la.py` checks the `data/LA` layout, counts, and
  protocol hashes, and writes `manifests/asvspoof2019_la.meta.json`.

### 4.2 Audio-to-image contract

Training images are **126 x 128 RGBA PNGs** (width x height). The model receives
224 x 224 RGB tensors after resizing and ImageNet normalization.

The verified preprocessing chain is:

1. Load audio with librosa at **16 kHz**.
2. Fix the length to **64,000 samples** (4 seconds); short clips are padded,
   long clips are truncated.
3. Compute a mel spectrogram with **`n_mels=128`**.
4. Convert to decibels with **`power_to_db(ref=np.max)`**.
5. Min-max normalise the dB values to `[0, 1]`.
6. Map through the **`magma`** colormap.
7. Flip vertically (`np.flipud` / `origin='lower'`).
8. Save as an RGBA PNG during dataset export.
9. At inference: convert to RGB, resize to **224 x 224**, convert to tensor, and
   normalise with ImageNet mean/std:
   `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`.

The rendered 126 x 128 image was verified pixel-identical to training
preprocessing on sampled clips. The app then resizes to 224 x 224 and applies
ImageNet normalization; treat that as part of the contract and re-verify it
against the training loader before changing it.

**Do not change any step above without an explicit ablation experiment.** A
silent preprocessing change invalidates every prior checkpoint and metric.

### 4.3 Model and checkpoints

Architecture:

```text
ResNet18 convolutions
  -> Linear(512, 128)
  -> ReLU
  -> Dropout(p=0.5)
  -> Linear(128, 1)
  -> sigmoid (applied outside the model)
```

Verified parameter facts:

- Model parameters: **11,242,305** (11,176,512 backbone + 65,793 head).
- Batch-norm buffers: **9,620** elements (`running_mean`, `running_var`,
  `num_batches_tracked`), so the checkpoint holds 11,251,925 tensor elements in
  total. Earlier notes quoted the tensor total as the parameter count.
- Trainable parameters in the current setup: **65,793** (the head only).
- All convolutional and batch-norm affine weights in the backbone are
  bit-identical to `torchvision.models.resnet18(weights="DEFAULT")`.
- Only batch-norm running statistics and the replaced head differ from the
  torchvision initialization.

Checkpoints:

| Path | Weighting | Notes |
|---|---|---|
| `Models/Itteration1/best_audio_spoof_model_unweighted.pth` | No class weights | First-iteration unweighted model |
| `Models/Itteration1/best_audio_spoof_model_weighted.pth` | Spoof `pos_weight` 0.344 (≈2.91× bonafide) | First-iteration weighted model |
| `best_audio_spoof_model_unweighted.pth` | No class weights | Root duplicate loaded by `app.py` |

The checkpoints use a `network.` key prefix. Both the evaluation harness and
`app.py` load them through `model.load_model_from_checkpoint`, which remaps the
prefix and raises unless the missing and unexpected key lists are both empty.
Do not trust a "loaded successfully" print without checking those lists; the
strict loader makes the check unavoidable.

Training configuration was read from `2_model.ipynb` via the python-notebook MCP
on 2026-11-13:

- Backbone frozen; head
  `Linear(512, 128) -> ReLU -> Dropout(0.5) -> Linear(128, 1)`.
- `BCEWithLogitsLoss`; Adam with `lr=1e-3`; batch size 32; maximum 20 epochs;
  early stopping with patience 5 on full dev; no augmentation; no recorded seed.
- The weighted run applies `pos_weight=2580/7500=0.344` to the spoof class,
  i.e. roughly 2.91x the bonafide weight.

Exact schedules are still **not reproducible**. The unweighted training cell and
output are missing from the notebook. `num_batches_tracked` is 4,095 steps for
the unweighted checkpoint (13.0 epochs at 315 steps/epoch) and 6,930 for the
weighted checkpoint (22.0 epochs), but the visible loop has a 20-epoch maximum.
The weighted checkpoint therefore came from a re-executed or unrecorded run. Do
not claim training can be replayed from the committed notebook.

All four checkpoint facts below were verified by the E0.3 strict-load audit:
124 state-dict keys, 124 remapped, zero missing/unexpected keys, 11,242,305
parameters, 65,793 head parameters. The root
`best_audio_spoof_model_unweighted.pth` is byte-identical to the `Models`
unweighted copy. Each checkpoint also has a recorded head-state SHA-256, final
head bias, and batch-norm counters; see `reports/baseline/freeze_report.json`.

### 4.4 Training-subset provenance (E0.1, complete)

The first iteration was trained on **10,080 clips**:

- 2,580 bonafide
- 7,500 spoof

**Status: reconstructed and verified on 2026-11-07.** The exact subset is on disk
at `data/derived/train_seed42/` and recorded in `manifests/train_seed42.csv`. The
canonical recipe remains `1.1_data_prep-train-old.ipynb` (execution_count 9) and
is implemented in `src/spoof_detection/sampling.py` (`build_seed42_subset`).

Verified facts:

| Check | Result |
|---|---|
| Subset size / balance | 10,080 (2,580 bonafide / 7,500 spoof) |
| Notebook anchor rows | 20 / 20 exact matches |
| Speaker coverage | all 20 train speakers, both classes |
| Attack coverage | A01–A06 (1,243 / 1,235 / 1,295 / 1,232 / 1,252 / 1,243) |
| Unique clips | 10,080 |
| Pixel parity | 10,080 / 10,080 identical to the full-split export |
| Manifest SHA-256 | `22f98abdf3392e04f7df3632809f81d033d5687773946cd103f5066846ed195e` |

Provenance constraints that still apply:

- The subset was drawn from the full train split with `random_state=42`; the
  shuffle that defines `sample_<i>.png` also uses seed 42.
- `metadata_train.csv` has no `audio_file_name` column, so provenance requires
  joining back to the protocol by `protocol_row`, not by filename.
- The original 10,080 PNGs no longer exist. Current images were regenerated from
  `data/LA/`; pixel parity is proven against the later full-split export, not the
  lost originals. Membership and order are proven by the 20 notebook anchors.
- `data/dataset_img/train` is the full 25,380-image export. Do not silently train
  on it and compare the result to the 10,080-image baseline.

Regenerate and verify with:

```bash
uv run python scripts/reproduce_train_seed42.py
uv run python -m unittest discover -s tests -p "test_*.py" -v
```

Full report: `reports/phase0/E0.1/E0.1_provenance.md`. The script refuses to
overwrite a non-empty output directory without `--force`, and `--force` only
removes paths inside `data/derived/`.

### 4.5 App behaviour

`app.py` is a FastAPI demo:

- Loads `best_audio_spoof_model_unweighted.pth` from the repository root through
  `model.load_model_from_checkpoint`, which raises on a missing or mismatched
  state dict. The app therefore fails at startup instead of silently running
  with randomly initialised weights.
- Uses the package preprocessing helpers (`generate_mel_spectrogram`,
  `render_mel_rgb`, and `build_inference_transform`); it no longer duplicates
  the model or preprocessing code.
- Accepts `.flac`, `.wav`, and `.mp3` uploads.
- Writes every upload to a temporary file with suffix `.flac`, regardless of the
  original format, then deletes it in `finally`.
- Applies the preprocessing chain in section 4.2.
- Computes `probability = sigmoid(logit)` and predicts spoof when
  `probability >= 0.5`.
- Reports `confidence = max(probability, 1 - probability)`, rounded to two
  decimal places as a percentage.
- Returns `raw_probability` and a base64 PNG of the generated spectrogram.
- Has no upload size limit.
- Enables CORS for all origins, methods, and headers.

The reported "confidence" is **not calibrated**. It is only the distance of the
sigmoid score from the decision boundary. Do not describe it as a probability
that the prediction is correct.

### 4.6 External smoke test

`tests/test1/` contains eight ElevenLabs-generated clips and
`tests/test1-results.csv`. All eight clips are spoofed; there are **no bonafide
human controls**.

With the unweighted checkpoint at threshold `p >= 0.5`, 7 of 8 clips were
labelled spoof. `Sarah-Educational.mp3` received raw `p = 0.3964` and was
labelled bonafide. This is an ad hoc smoke test, not a benchmark: it measures
recall on one eight-clip synthetic set and says nothing about specificity, EER,
or robustness.

### 4.7 Environment

Verified locally:

| Component | Version / status |
|---|---|
| Python | 3.14.8 (`requires-python >= 3.14`) |
| torch | 2.15.0+cu132 |
| torchvision | 0.29.2 |
| CUDA | Available |
| librosa | 1.0.0 |
| numpy | 2.5.4 |
| pandas | 3.1.1 |
| matplotlib | 3.11.2 |
| fastapi | 0.162.0 |

Use the project virtual environment at `.venv/` or `uv run`. Do not install
packages globally.

### 4.8 Baseline evaluation (E0.2) and frozen baseline (E0.3)

`src/spoof_detection/evaluate.py` is the batch evaluation entry point. It loads a
checkpoint through `model.load_model_from_checkpoint`, which remaps `network.*`
to `backbone.*` and raises unless the missing and unexpected key lists are both
empty. It reads either the stored PNGs (`--source image`, default) or re-renders
source FLACs through the verified preprocessing chain (`--source audio`), and it
writes `metrics.json`, `scores_<split>.csv`, and `summary.md` per checkpoint.

**Frozen E0.3 baseline** (commit `67a81104592090857ffd8b85de87edbc4ade9e07`,
manifest SHA-256
`22f98abdf3392e04f7df3632809f81d033d5687773946cd103f5066846ed195e`,
image input, full splits):

| Checkpoint | Split | Attack subset | Threshold | Samples (bonafide/spoof) | EER | AUC |
|---|---|---|---|---:|---:|---:|
| unweighted | dev | A01–A06 | threshold-free | 24,844 (2,548 / 22,296) | 6.16% | 0.9860 |
| unweighted | eval | A07–A19 | threshold-free | 71,237 (7,355 / 63,882) | 11.50% | 0.9527 |
| weighted | dev | A01–A06 | threshold-free | 24,844 (2,548 / 22,296) | 5.42% | 0.9881 |
| weighted | eval | A07–A19 | threshold-free | 71,237 (7,355 / 63,882) | 11.13% | 0.9558 |

The dev EER thresholds are `0.839741` (unweighted) and `0.752020` (weighted),
selected per checkpoint on dev only. Transferred unchanged to eval they give
5.02% FPR / 82.55% spoof recall (unweighted) and 4.68% / 82.60% (weighted). At
the fixed `0.5` threshold, unweighted eval gives 16.55% FPR / 90.99% spoof
recall; weighted gives 9.42% / 87.76%, so weighting mainly shifts the operating
point rather than improving threshold-free separation.

Eval per-attack EER is 1.04%–6.98% for A07–A16, 36.24% (unweighted) / 34.67%
(weighted) for A17, and 18–19% for A18/A19. Two full runs produced byte-identical
score CSVs and identical EER/AUC values for both checkpoints. On a
100-clip-per-split smoke subset, image and audio inputs produced bit-identical
logits and scores.

**Dev is a model-selection split, not a clean holdout:** the notebook used full
dev for early stopping, and E0.3 selects the operating threshold on dev. Eval is
the clean estimate. No eval data is used for training, early stopping, or
threshold selection.

The E0.3 audit (`scripts/freeze_baseline.py`, output
`reports/baseline/freeze_report.json`) verifies the manifest hash, metric
contract, run-to-run tolerance (±0.01 EER fraction), and a trained-vs-random head
control on a deterministic 1,000-clip dev subset (500/500). Trained AUC is
0.9994 for both checkpoints; a reinitialised head gets 0.5689 / 0.5590, so the
head is trained rather than random. The notebook's own eval at `0.5` agrees on
accuracy to four decimals but differs by 2 false positives (unweighted) and 1
false negative (weighted); the harness counts are the frozen values. Full
reports: `reports/baseline/README.md`,
`reports/phase0/E0.3/README.md`, `reports/phase0/E0.2/README.md`.

The harness supports a generic labelled CSV via `--manifest`, a deterministic
class-stratified `--limit` for smoke tests, `--threshold-from` to reuse a
previous dev threshold for eval-only or manifest runs, and
`--experiment-id`/`--title` to label output metadata. Only the dev EER threshold
is selected on data; eval is never used for threshold selection.

### 4.9 Contract tests (E0.4, complete)

`tests/test_preprocessing.py` locks audio loading/padding, mel and dB computation,
min-max normalisation, `magma` rendering and vertical flip, resize plus ImageNet
normalisation, and pixel parity with the stored dev PNGs.
`tests/test_model_loading.py` covers synthetic and real checkpoint remaps and
fails on a deliberately broken prefix. `tests/test_app_contract.py` covers
unsupported file rejection, temp-file cleanup, strict checkpoint loading, and the
response contract (`p >= 0.5` is spoof; `confidence = max(p, 1-p)`).

`make test` ran 116 tests at E0.4 (138 now, including the E0.5 audit suite). The
full dev/eval image rerun after the preprocessing refactor is byte-identical to
`reports/baseline/`. Full report: `reports/phase0/E0.4/README.md`.

### 4.10 Data and leakage audit (E0.5, complete)

`src/spoof_detection/audit.py` plus `scripts/audit_data.py` implement a
read-only data and leakage audit. `make audit-data` hashes all 122,299 audio
files and 131,541 images (the full train/dev/eval exports plus
`data/derived/train_seed42/`), checks protocol/metadata alignment, and writes
`reports/audit/data_audit.md` and `reports/audit/data_audit.json`. It exits
nonzero on any failure; `--skip-hashes` runs structural checks only and marks
B.3/B.4 skipped.

Result on 2026-11-14: **all 15 checks PASS.** Verified facts:

| Check | Result |
|---|---|
| A.1 layout/protocol hashes | PASS |
| A.2–A.4 row alignment | PASS: 25,380 / 24,844 / 71,237 rows; 0 missing, 0 orphaned images |
| B.1 speaker and filename isolation | PASS: no pairwise overlap |
| B.2 attack coverage | PASS: train/dev A01–A06; eval A07–A19, no overlap |
| B.3 audio duplicates | PASS: 122,299 files hashed, 0 cross-split, 0 within-split |
| B.4 image duplicates | PASS: 131,541 files hashed, 0 cross-split; 10,080 within-train groups, all explained as derived-subset/full-export pairs, 0 unexplained |
| C.1–C.3 manifest | PASS: hash `22f98abd…`, 10,080 rows, 10,080/10,080 audio+image hashes match, zero name/speaker/content overlap with dev or eval |
| D.1 enrollment exclusion | PASS: 142 dev + 696 eval `_A` files excluded |
| E.1–E.3 baseline lineage | PASS: `selected_on: dev`, `source: in_run_dev`, eval threshold equals the re-derived dev threshold, artifacts match `freeze_report.json` |

**No train/dev/eval leakage was found.** Dev remains a documented
**model-selection split** (early stopping in `2_model.ipynb` and E0.3 threshold
selection); eval was read only by the post-training inference cells (23 and 25)
and is the clean estimate. Notebook evidence is in
`reports/phase0/E0.5/README.md`. The audit catches only exact byte duplicates
and cannot speak to uncommitted historical runs. `make test` runs 138 tests
including `tests/test_audit.py` (22 tests and a real-data structural audit).

---

## 5. Guardrails (hard rules)

These are not suggestions. If a task conflicts with one of them, stop and ask.

### G1. Notebook access goes through the python-notebook MCP

The python-notebook MCP server is fully working as of 2026-11-13; all eight tools
are usable. Use its tools (`initialize_workspace`, `list_notebooks`,
`read_notebook`, `read_cell`, `read_cell_output`, `read_notebook_outputs`,
`create_notebook`, `add_cell`, `edit_cell`) rather than reading notebook JSON
with Python, `jq`, `grep`, or `git show`.

Verified working on 2026-11-13:

| Tool | Status |
|---|---|
| `initialize_workspace` | Working; must be the first call in a session |
| `list_notebooks` | Working; returns absolute paths, includes `.ipynb_checkpoints/` copies |
| `read_notebook`, `read_cell`, `read_cell_output`, `read_notebook_outputs` | Working |
| `create_notebook`, `add_cell`, `edit_cell` | Working, but paths must stay inside the workspace root |

The historical `'map' object has no attribute 'close'` failure came from wrapping
`Path.rglob()` (a generator) in `contextlib.closing()`; that wrapper has been
removed, so `initialize_workspace` and `list_notebooks` both work.

Behaviour notes:

- **Call `initialize_workspace` first**, with the absolute repository path:
  `initialize_workspace("/home/toastu/Research/Spoof Detection")`. Every other
  tool refuses to run until it has been called, and relative paths are rejected.
  Its return value already lists the notebooks found, so it doubles as a first
  listing.
- `list_notebooks(directory=".")` takes a workspace-relative subdirectory and
  rejects paths that escape the workspace root (e.g. `"../.."`).
- Write tools reject absolute paths (and `../` traversal) outside the repository
  root. Use workspace-relative paths.
- `create_notebook` writes a markdown title cell at index 0 plus a code cell.
- `edit_cell` replaces source text but does not change the cell's type.

The server lives at `/home/toastu/python-notebook-mcp/server.py`, outside this
repository. After editing it, restart OpenCode so the MCP server reloads: a
still-running `server.py` process keeps the old code loaded and the same error
will persist even though the file is fixed. If any other notebook tool fails,
report the error instead of falling back to parsing notebook JSON.

For training-subset provenance, reference `1.1_data_prep-train-old.ipynb` via the
MCP; do not reproduce a full seed-42 join snippet in documentation or code unless
the user asks for it.

### G2. Do not destroy data or existing artifacts

Do not overwrite, delete, move, or regenerate in place:

- anything under `data/`
- any existing `.pth` checkpoint
- anything under `tests/`
- the original eight external test clips or their results CSV

Create new files in new paths. If a rebuild requires replacing an existing
artifact, ask first. Creating a new, clearly named derived-data directory such as
`data/derived/<experiment>/` is allowed; overwriting an existing file under
`data/` is not.

### G3. Preprocessing parity

Inference preprocessing must remain identical to training preprocessing
(section 4.2). If an experiment changes the front-end, it must retrain the model
and document the change. Do not mix a new front-end with an old checkpoint.

### G4. Metric reporting contract

Never report a metric without stating all of:

1. Split: train, dev, eval, or external.
2. Attack subset: e.g. A01-A06, A07-A19, all eval attacks, or a named external
   corpus.
3. Threshold: the exact decision threshold, or "threshold-free" for EER/AUC.
4. Sample count and class balance.

If any of these is unknown, write "unknown" rather than omitting it. Do not
compare numbers produced under different splits, subsets, or thresholds.

### G5. No calibrated-confidence claims

The sigmoid output is a score, not a calibrated probability. The app's
"confidence" is `max(p, 1-p)`. Do not call either one calibrated, do not present
it as the probability that the prediction is correct, and do not use it for
risk-based decisions until a calibration experiment has been run and documented.

### G6. Model iteration layout

New model iterations go under `Models/IterationN/` (correct spelling) and must
include a `MODELS.md` with at least:

- date and git commit
- training data manifest and hash
- preprocessing version
- architecture and which layers were frozen
- loss, class weights, optimizer, learning rate, batch size, epochs, seed
- checkpoint filenames and parameter counts
- evaluation table with split, attack subset, threshold, sample count
- known limitations and failure cases

Do not overwrite `Models/Itteration1/`; keep its existing spelling for
compatibility.

### G7. No silent checkpoint mismatches

When loading a checkpoint, verify that missing and unexpected keys are empty, or
explicitly list and justify them. `strict=False` is acceptable only with an
assertion or a printed diff that a reviewer can inspect.

### G8. No secrets or large binary commits

Do not commit credentials, tokens, datasets, `.pth` files, or generated images.
Update `.gitignore` before adding large artifacts. Use Git LFS only if the user
approves it.

### G9. Reproducibility

Record the exact command, config, seed, and environment used for every experiment.
If a result cannot be reproduced from the repository, label it "not reproducible"
and do not use it as a baseline.

---

## 6. Conventions

### Code

- Python 3.14, 4-space indentation, type hints on public functions.
- Put reusable logic in `src/spoof_detection/`, not in notebooks or `app.py`.
- Keep `app.py` thin: it imports the model definition, strict loader, and
  preprocessing helpers from `src/spoof_detection/` rather than duplicating them.
- Prefer explicit paths resolved from the repository root; do not depend on the
  caller's current working directory.
- Use `pathlib.Path` for new filesystem code.
- Add tests under `tests/` for any preprocessing, metric, or loading change.

### Experiment artifacts

- Put generated reports in `reports/<phase>/<experiment-id>/`.
- Put data manifests in `manifests/`.
- Put new checkpoints in `Models/IterationN/`.
- Every report must open with the metric contract from G4 and state the git
  commit and data manifest hash.
- Prefer CSV/JSON for machine-readable results and Markdown for interpretation.

### Naming

- New model iterations: `Models/Iteration2/`, `Models/Iteration3/`, ...
- Experiment IDs: `E0.1`, `E1.2`, `E2.3`, matching `ROADMAP.md`.
- Checkpoints: include iteration, weighting, and date, e.g.
  `resnet18_frozen_unweighted_2026-11-06.pth`.
- Do not rename existing files to "fix" spelling or style.

### Git

- Keep commits focused and describe the experiment or fix.
- Do not commit generated data or checkpoints.
- Do not rewrite history or run destructive git commands on the user's changes.
- The repository currently has one initial commit and many untracked or modified
  files. Treat all of them as user work; do not clean them up without permission.

---

## 7. Commands

```bash
# Install / sync dependencies
uv sync

# Run the demo (http://localhost:8000)
make app

# Run a Python script in the project environment
uv run python path/to/script.py

# Check the environment
uv run python -c "import torch, torchvision, librosa; print(torch.__version__, torchvision.__version__, librosa.__version__)"

# Verify the ASVspoof 2019 LA layout, counts, and protocol hashes
uv run python scripts/verify_asvspoof2019_la.py

# Validate an archive too (use --accept-any-archive for an unofficial re-zip)
uv run python scripts/verify_asvspoof2019_la.py --archive data/downloads/LA.zip

# Reconstruct and verify the E0.1 training subset (~3 min, 257 MB)
uv run python scripts/reproduce_train_seed42.py

# Score both checkpoints on full dev + eval (~5 min on the RTX 3050)
make evaluate

# Second full run for the E0.3 reproducibility check
make evaluate-rerun

# Audit manifest, contract, run-to-run tolerance, and head control
uv run python scripts/freeze_baseline.py

# All three of the above in sequence
make freeze-baseline

# Run the E0.5 data and leakage audit (~44 s, hashes 122,299 FLACs + 131,541 PNGs)
make audit-data
# equivalent
uv run python scripts/audit_data.py
# structural-only run; B.3/B.4 are marked skipped
uv run python scripts/audit_data.py --skip-hashes

# Smoke-test the harness on a deterministic 100-clip-per-split subset
uv run python -m spoof_detection.evaluate \
  --checkpoint Models/Itteration1/best_audio_spoof_model_unweighted.pth \
  --asvspoof dev eval --source image --limit 100 \
  --output-dir /tmp/opencode/e0.2-smoke

# Run the full unit and contract test suite
make test

# Equivalent direct command
uv run python -m unittest discover -s tests -p "test_*.py" -v
```

There is still no CI pipeline; that is the only open infrastructure item. Phase 0
is complete.

---

## 8. Known gaps and open questions

1. **Training provenance is incomplete.** `2_model.ipynb` confirms the
   architecture, `BCEWithLogitsLoss`, Adam `lr=1e-3`, batch size 32, 20-epoch
   maximum, early stopping on dev, weighted `pos_weight`, no augmentation, and no
   recorded seed. However, the unweighted training cell/output is missing, and
   the weighted batch-norm counter (6,930 steps ≈ 22 epochs) exceeds the visible
   20-epoch loop. Exact schedules are not reproducible; see section 4.3.
2. **Regenerated images, not original PNGs.** E0.1 rebuilt the subset at
   `data/derived/train_seed42/` and verified 10,080/10,080 pixel parity against
   the full-split export. Parity with the lost original PNGs cannot be proven.
   Use the manifest for all training.
3. **Evaluation harness exists but is incomplete for a benchmark.** E0.2/E0.3
   compute EER, AUC, threshold metrics, per-attack and per-speaker breakdowns on
   dev/eval (`src/spoof_detection/evaluate.py`). Min t-DCF, confidence intervals,
   and significance tests are not implemented.
4. **No bonafide controls in the external set.** Specificity and calibration are
   unmeasured outside ASVspoof.
5. **Uncalibrated scores.** The app's confidence is not meaningful as a
   probability.
6. **Silent-load risk in `app.py` resolved (E0.4).** Both the harness and the app
   use `model.load_model_from_checkpoint`, which raises on missing or unexpected
   keys. The app now refuses to start with a broken checkpoint.
7. **App hardening.** No upload size limit, CORS is fully open, and all uploads
   are written with a `.flac` suffix.
8. **Leakage audit complete (E0.5).** No train/dev/eval leakage was found; see
   section 4.10 and `reports/audit/data_audit.md`. Dev remains a
   model-selection split because the original loop early-stopped on it and E0.3
   selected the threshold on it. The audit detects only exact byte duplicates, so
   a re-encoded copy would not be caught, and it cannot speak to uncommitted
   historical runs.
9. **Class imbalance.** Training was 2,580 bonafide vs 7,500 spoof (about 1:2.9).
   Accuracy alone is misleading; report per-class recall and EER.
10. **Archive artifact.** `data/archive.zip` was deleted on 2026-11-07 after
    verification; only its hashes remain, in
    `manifests/asvspoof2019_la.meta.json`.
11. **Project scaffolding is mostly complete.** `src/spoof_detection/` has
    protocol, sampling, preprocessing, dataset-validation, model-loading,
    metrics, evaluation, baseline-audit, and data-audit modules with unit tests.
    E0.4 added preprocessing and app-contract tests; E0.5 added `audit.py` and
    `tests/test_audit.py`, so `make test` runs 138 tests. `README.md` documents
    setup, data acquisition, evaluation, and the audit. There is still no CI
    pipeline.

---

## 9. Where to start

For any new task:

1. Read this file and the relevant section of `ROADMAP.md`.
2. Check `git status` before touching files; the working tree contains user work.
3. If the task involves training data, use `manifests/train_seed42.csv` and the
   images in `data/derived/train_seed42/`; do not re-derive the seed-42 subset or
   substitute the full 25,380-image export. Verify the raw dataset with
   `scripts/verify_asvspoof2019_la.py` if counts or paths look wrong.
4. If the task involves a metric, write down the split, attack subset,
   threshold, and sample count **before** running the experiment.
5. For a baseline comparison, use `reports/baseline/README.md` and keep the
   preprocessing contract, training manifest, and split definitions identical.
   Run `make test` after any preprocessing, metric, loading, or app change,
   `make freeze-baseline` after any evaluation or checkpoint-loading change, and
   `make audit-data` after any data, manifest, or split change.
6. If the task involves a new model, create `Models/IterationN/MODELS.md` first.
7. If the task requires notebook access, use the python-notebook MCP tools under
   the rules in G1; do not parse notebook JSON directly. If a tool errors, report
   it rather than working around it.

---

## 10. Related documents

- [`ROADMAP.md`](./ROADMAP.md): prioritized research roadmap, phases 0-4.
- [`Models/Itteration1/MODELS.md`](./Models/Itteration1/MODELS.md): first-iteration
  model notes.
- [`reports/phase0/E0.1/E0.1_provenance.md`](./reports/phase0/E0.1/E0.1_provenance.md):
  E0.1 reconstruction provenance, verification, and limitations.
- [`reports/phase0/E0.3/README.md`](./reports/phase0/E0.3/README.md): frozen
  two-checkpoint baseline and audit.
- [`reports/phase0/E0.4/README.md`](./reports/phase0/E0.4/README.md):
  preprocessing, checkpoint-loading, and app-contract tests.
- [`reports/audit/data_audit.md`](./reports/audit/data_audit.md): canonical E0.5
  data and leakage audit report.
- [`reports/phase0/E0.5/README.md`](./reports/phase0/E0.5/README.md): E0.5
  experiment log, notebook lineage evidence, and limitations.
