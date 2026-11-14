# E0.5 — Data and leakage audit

**Status:** complete (2026-11-14)
**Canonical audit:** [`reports/audit/data_audit.md`](../../audit/data_audit.md)
and `reports/audit/data_audit.json`
**Git commit at run time:** `ef487b0` (with the E0.5 changes uncommitted in the
working tree)
**Preprocessing version:** `mel128-magma-4s-v1`
**Training manifest:** `manifests/train_seed42.csv`
(SHA-256 `22f98abdf3392e04f7df3632809f81d033d5687773946cd103f5066846ed195e`)
**Raw dataset:** ASVspoof 2019 LA, verified by
`manifests/asvspoof2019_la.meta.json`

## Metric contract

This experiment runs no new model evaluation, so no EER/AUC value is
introduced. The audit checks existing artifacts under the following contract:

| Split | Attack subset | Threshold | Samples | Bonafide / spoof |
|---|---|---|---:|---:|
| train | A01–A06 | not applicable (audit) | 25,380 | 2,580 / 22,800 |
| dev | A01–A06 | not applicable (audit); baseline thresholds audited as artifacts | 24,844 | 2,548 / 22,296 |
| eval | A07–A19 | not applicable (audit); baseline thresholds audited as artifacts | 71,237 | 7,355 / 63,882 |
| train (first iteration) | A01–A06 | not applicable (audit) | 10,080 | 2,580 / 7,500 |

The baseline thresholds audited are the fixed `0.5` and the dev-selected EER
thresholds `0.839741` (unweighted) and `0.752020` (weighted). Eval never
selected a threshold.

## Question

Is the evaluation set genuinely held out? Specifically:

1. Do train, dev, and eval overlap in speakers, audio file names, or file
   content?
2. Did any dev or eval clip enter the 10,080-clip training set?
3. Did eval enter the original training loop, early stopping, or E0.3 threshold
   selection?
4. Are the 838 `_A` enrollment files excluded from CM scoring?

## Hypothesis

Eval is held out at the clip, speaker, and content level, and no eval data
entered training, early stopping, or threshold selection. Dev is reused by
design (early stopping and threshold selection) and must be reported as a
model-selection split rather than as a clean holdout.

## Method

`scripts/audit_data.py` implements the audit in `src/spoof_detection/audit.py`.
It is read-only over `data/`, `Models/`, and `reports/baseline/`, and writes only
`reports/audit/data_audit.{md,json}`. It performs:

- **A. Layout and alignment.** Dataset layout, counts, and protocol hashes;
  metadata row *i* against protocol row *i* on speaker, system, and label;
  image filename index; image existence, class folder, and orphan count.
- **B. Cross-split identity.** Pairwise speaker and filename intersections;
  attack-system coverage; SHA-256 of every audio file and every stored or
  derived image, grouped to find cross-split and within-split duplicates.
- **C. Training manifest containment.** The 10,080-row manifest is re-checked
  against the train protocol, all 10,080 audio and image hashes are recomputed,
  and the manifest is intersected with dev and eval by clip name, speaker, and
  audio content hash.
- **D. Enrollment exclusion.** Non-protocol FLACs are counted and checked
  against the documented 142 dev / 696 eval enrollment files.
- **E. Baseline lineage.** Each `metrics.json` threshold is re-derived from
  `scores_dev.csv`, checked for `selected_on: dev` and `source: in_run_dev`, and
  compared with the eval operating threshold. Score CSV rows are aligned to
  their protocol split, and checkpoint/manifest identities are compared with the
  frozen E0.3 audit.
- **F. Notebook lineage.** Notebook cells are read through the python-notebook
  MCP (AGENTS.md G1); the evidence is recorded below and is not parsed by the
  audit code.

Pre-registered success criteria:

1. Every A–E check passes.
2. Zero cross-split name, speaker, audio-hash, or image-hash collisions.
3. The training manifest intersects dev/eval in zero clips by name and zero by
   audio content hash.
4. All recorded operating thresholds trace to dev only.
5. Notebook evidence shows eval is read only after training for reporting.

## Notebook lineage evidence

Read via the python-notebook MCP on 2026-11-14.

| Notebook | Cell | Evidence |
|---|---:|---|
| `2_model.ipynb` | 10 | `df = pd.read_csv("data/metadata_train.csv")` — the 10,080-row training export |
| `2_model.ipynb` | 12 | `train_dataset`/`train_loader` built from `df` |
| `2_model.ipynb` | 16–17 | `df_dev = pd.read_csv("data/metadata_dev.csv")`; validation loader built from full dev |
| `2_model.ipynb` | 21 | Early-stopping loop monitors `val_loss` from the dev loader and saves `best_audio_spoof_model_weighted.pth` |
| `2_model.ipynb` | 23, 25 | Eval metadata is read only in post-training inference cells, for the unweighted and weighted checkpoints |
| `1.1_data_prep-train-old.ipynb` | 2, 11, 17, 20 | Reads only the train protocol/FLACs; applies seed-42 sampling; exports the 10,080 images and `data/metadata_train.csv` |
| `1.1_data_prep-train.ipynb` | 3, 13–14, 23, 28 | Reads only the train protocol/FLACs. Cell 13 computes the seed-42 frame but cell 14 leaves `df_train` unchanged, so the export is the full 25,380-image train split |
| `1.2_data_prep-dev.ipynb` | 3, 13, 16 | Reads only the dev protocol/FLACs; exports `data/dataset_img/dev` and `metadata_dev.csv` |
| `1.3_data_prep-eval.ipynb` | 3, 13, 16 | Reads only the eval protocol/FLACs; exports `data/dataset_img/eval` and `metadata_eval.csv` |

No notebook builds a training or early-stopping loader from eval. Eval is used
only for post-training reporting, and no eval-derived quantity was used to select
a checkpoint or threshold.

## Results

All checks pass. Full machine-readable evidence:
`reports/audit/data_audit.json`.

| Check | Result |
|---|---|
| A.1 layout and protocol hashes | PASS |
| A.2–A.4 train/dev/eval row alignment | PASS: 25,380 / 24,844 / 71,237 rows; 0 missing, 0 orphaned images |
| B.1 speaker and filename isolation | PASS: pairwise speaker and filename intersections are empty |
| B.2 attack coverage | PASS: train/dev A01–A06 by design; eval A07–A19, no overlap |
| B.3 audio content duplicates | PASS: 122,299 files hashed; 0 cross-split groups, 0 within-split groups |
| B.4 image content duplicates | PASS: 131,541 files hashed; 0 cross-split groups; 10,080 within-train groups, all of them the expected derived-subset/full-export pairs; 0 unexplained |
| C.1 manifest integrity | PASS: hash `22f98abd…`, 10,080 rows (2,580 / 7,500), all rows match their train protocol row |
| C.2 manifest file/hash integrity | PASS: 10,080/10,080 audio and image hashes match |
| C.3 manifest isolation | PASS: zero name, speaker, or content-hash overlap with dev or eval |
| D.1 enrollment exclusion | PASS: 142 dev + 696 eval `_A` files excluded from protocols and image exports |
| E.1 threshold lineage | PASS: `selected_on: dev`, `source: in_run_dev`; eval threshold equals the re-derived dev threshold |
| E.2 score/protocol alignment | PASS: all four baseline score CSVs map 1:1 to their protocol split |
| E.3 baseline artifact identity | PASS: checkpoint hashes, parameter counts, and manifest hash match the E0.3 freeze report |

## Result summary

**No eval leakage was found.** Eval is isolated from train and dev at the
speaker, filename, audio-content, and rendered-image levels, and no eval clip or
derived quantity entered training, early stopping, or threshold selection.

**Dev is a documented model-selection split.** The original training loop
early-stopped on full dev, and E0.3 selected the operating threshold on full dev.
Dev numbers are therefore optimistic; eval remains the clean estimate.

## Interpretation

- The Phase 0 exit criterion "the data audit finds no train/dev/eval leakage" is
  satisfied in the sense defined by the roadmap: no clip-level train/dev/eval
  overlap, and no eval involvement in training or selection. The dev reuse is
  process-level contamination that is documented rather than hidden.
- The only duplicate content is the expected byte-identical overlap between the
  reconstructed `data/derived/train_seed42/` images and their counterparts in
  the full `data/dataset_img/train` export. Both are labelled `train`, so this
  is not cross-split leakage.
- The 838 `_A` enrollment files are present on disk as expected and were never
  scored: they are absent from the CM protocols, the image exports, and the
  baseline score CSVs.
- Weighted/unweighted threshold differences are operating-point choices made on
  dev; E0.3's eval EER and AUC remain threshold-free.

## Limitations and failure cases

- Content hashing catches only exact byte duplicates. A re-encoded or resampled
  copy of a clip in another split would not be detected by SHA-256.
- Full re-rendering of all 96,081 dev/eval audio clips was not repeated; the
  audit relies on the stored exports, the E0.4 pixel-parity test, and the frozen
  image-vs-audio smoke parity.
- "No eval leakage" is established by clip identity, artifact lineage, and the
  notebook record. It cannot prove anything about uncommitted historical runs
  that left no artifact.
- Dev contamination from early stopping and threshold selection cannot be undone
  retroactively; it is recorded as a limitation, not repaired.
- The audit does not cover the eight external `tests/test1` clips; they are an
  ad hoc all-spoof smoke set, not a benchmark.

## Artifact paths

- `src/spoof_detection/audit.py` — audit checks and report renderer
- `scripts/audit_data.py` — read-only CLI
- `tests/test_audit.py` — 22 tests, including a real-data structural audit
- `reports/audit/data_audit.md` — canonical Markdown report
- `reports/audit/data_audit.json` — machine-readable evidence
- `makefile` — `make audit-data`

## Reproduction command

```bash
make audit-data
# equivalent
uv run python scripts/audit_data.py
```

The full run hashes 122,299 audio files and 131,541 images (~10.7 GB) and took
about 44 seconds on the development machine. Add `--skip-hashes` for a
structural-only run; the report then marks B.3/B.4 as skipped.
