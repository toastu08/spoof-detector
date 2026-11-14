# Spoof Detection

Research code for **audio spoofing / deepfake speech detection** on ASVspoof 2019
Logical Access (LA). The current baseline renders a 4-second clip as a
magma-coloured mel spectrogram and classifies it with a frozen ImageNet-pretrained
ResNet18 plus a small linear head.

> **Status (2026-11-14):** Phase 0, E0.1–E0.5 complete. The 10,080-clip
> training subset is reconstructed at `data/derived/train_seed42/` with manifest
> `manifests/train_seed42.csv`. The evaluation harness is in
> `src/spoof_detection/evaluate.py`, the frozen two-checkpoint baseline is in
> [`reports/baseline/README.md`](reports/baseline/README.md), the
> preprocessing/checkpoint/app contract tests pass via `make test`, and the data
> and leakage audit ([`reports/audit/data_audit.md`](reports/audit/data_audit.md))
> finds no train/dev/eval leakage.

The sigmoid output is a **score, not a calibrated probability**. The demo's
"confidence" is `max(p, 1 - p)` and must not be read as the probability that a
prediction is correct.

## Requirements

- Python 3.14
- [`uv`](https://docs.astral.sh/uv/)
- Roughly 16 GB of free disk space for the ASVspoof archive and its extracted
  tree
- A CUDA-capable GPU is useful for training but not required for the demo

## Quick start

```bash
uv sync
# Download and verify the dataset (see "Data" below)
uv run python scripts/verify_asvspoof2019_la.py
make app          # http://localhost:8000
```

Run the tests with:

```bash
make test
```

This is equivalent to
`uv run python -m unittest discover -s tests -p "test_*.py" -v`. Some tests read
`data/LA` and `data/dataset_img/train`, so run them after the dataset is in
place.

## Data: ASVspoof 2019 LA

The code expects the dataset at **`data/LA`** (repository-relative, not
`/data/LA`). `data/` is gitignored; the dataset is never committed.

The dataset is distributed under the **Open Data Commons Attribution License
(ODC-By 1.0)**. The canonical record is:

- **DOI:** [`10.7488/ds/2555`](https://doi.org/10.7488/ds/2555)
- **Publisher:** University of Edinburgh DataShare
- **Zenodo mirror:** <https://zenodo.org/records/6906306>

### Option A: official source (recommended)

The official `LA.zip` is 7,640,952,520 bytes with MD5
`30c98f11d8b2bc21f2c257bfd78bb5c5`.

```bash
mkdir -p data/downloads

# Zenodo mirror (single file, supports resume)
curl -L -C - -o data/downloads/LA.zip \
  'https://zenodo.org/records/6906306/files/LA.zip?download=1'

# Check the published MD5 before extracting
md5sum data/downloads/LA.zip
# expect: 30c98f11d8b2bc21f2c257bfd78bb5c5

# The archive contains a top-level LA/ directory, so extract into data/
unzip -q data/downloads/LA.zip -d data/

uv run python scripts/verify_asvspoof2019_la.py \
  --archive data/downloads/LA.zip
```

The publisher's direct bitstream is also available:

```bash
curl -L -C - -o data/downloads/LA.zip \
  'https://datashare.ed.ac.uk/server/api/core/bitstreams/a9f87c35-f055-4015-80e2-2fdff0d46269/content'
```

### Option B: Kaggle mirror (alternative)

The initial setup of this repository used the third-party Kaggle copy
[`anishsarkar22/asvpoof-2019-dataset-la`](https://www.kaggle.com/datasets/anishsarkar22/asvpoof-2019-dataset-la).
It is the extracted LA tree re-packaged as a zip, not the official `LA.zip`, so
its archive hash is different. The copy originally used here was
`data/archive.zip` (MD5 `793d7b00e00ec0466acfd85800b714fd`); it was deleted after
verification and its hashes are recorded in
`manifests/asvspoof2019_la.meta.json`.

Kaggle requires an account and the Kaggle CLI credentials
(`~/.kaggle/kaggle.json`):

```bash
kaggle datasets download \
  -d anishsarkar22/asvpoof-2019-dataset-la \
  -p data/downloads

unzip -q data/downloads/asvpoof-2019-dataset-la.zip -d data/

uv run python scripts/verify_asvspoof2019_la.py \
  --archive data/downloads/asvpoof-2019-dataset-la.zip \
  --accept-any-archive
```

`--accept-any-archive` only relaxes the archive MD5 check; the extracted
contents are still validated against the expected counts and protocol hashes.
The official source above is preferred for reproducibility and attribution.

### Verify an existing installation

```bash
# Validate data/LA only (fast; no 7.6 GB hashing)
uv run python scripts/verify_asvspoof2019_la.py

# Validate the archive too
uv run python scripts/verify_asvspoof2019_la.py --archive data/downloads/LA.zip

# Shortcut (does not rewrite the provenance manifest)
make verify-data
```

On success the script writes `manifests/asvspoof2019_la.meta.json`, which records
the source, archive hashes, per-split counts, and protocol SHA-256 values.

### Expected layout

```text
data/LA/
├── README.LA.txt
├── ASVspoof2019_LA_train/
│   ├── LICENSE.txt
│   └── flac/                        # 25,380 LA_T_*.flac
├── ASVspoof2019_LA_dev/
│   ├── LICENSE.txt
│   └── flac/                        # 24,986 = 24,844 CM + 142 LA_D_A*.flac
├── ASVspoof2019_LA_eval/
│   ├── LICENSE.txt
│   └── flac/                        # 71,933 = 71,237 CM + 696 LA_E_A*.flac
├── ASVspoof2019_LA_cm_protocols/    # 3 countermeasure protocol files
├── ASVspoof2019_LA_asv_protocols/   # 10 ASV protocol files
└── ASVspoof2019_LA_asv_scores/      # 2 baseline ASV score files
```

| Split | CM rows | Bonafide | Spoof | FLACs on disk | Extra `_A` |
|---|---:|---:|---:|---:|---:|
| train | 25,380 | 2,580 | 22,800 | 25,380 | 0 |
| dev | 24,844 | 2,548 | 22,296 | 24,986 | 142 |
| eval | 71,237 | 7,355 | 63,882 | 71,933 | 696 |
| **total** | **121,461** | **12,483** | **108,978** | **122,299** | **838** |

The `_A` files are ASV enrollment audio and are **not** part of the CM
protocols. Do not mix them into CM evaluation.

### License and citation

The dataset is released under the ODC Attribution License (ODC-By 1.0). If you
use it, cite:

> Yamagishi, J., Todisco, M., Sahidullah, M., Delgado, H., Wang, X., Evans, N.,
> Kinnunen, T., Lee, K. A., Vestman, V., & Nautsch, A. (2019). *ASVspoof 2019:
> The 3rd Automatic Speaker Verification Spoofing and Countermeasures Challenge
> database*. University of Edinburgh. <https://doi.org/10.7488/ds/2555>

## Repository layout

| Path | Purpose |
|---|---|
| `1.1_data_prep-train-old.ipynb` | Canonical record of the original 10,080-clip subset |
| `1.1_data_prep-train.ipynb`, `1.2_data_prep-dev.ipynb`, `1.3_data_prep-eval.ipynb` | Spectrogram export notebooks |
| `2_model.ipynb` | Original training/evaluation notebook |
| `app.py`, `templates/` | FastAPI inference demo |
| `src/spoof_detection/` | Reusable protocol, sampling, preprocessing, model-loading, metrics, and evaluation code |
| `scripts/` | Reproduction and dataset-verification entry points |
| `manifests/` | Data manifests and provenance metadata |
| `reports/` | Experiment reports |
| `Models/` | Model iteration notes and (gitignored) checkpoints |
| `data/` | Downloaded/derived data (gitignored) |

## Baseline evaluation

Score both first-iteration checkpoints on the full dev and eval splits:

```bash
make evaluate
# equivalent:
uv run python -m spoof_detection.evaluate \
  --checkpoint Models/Itteration1/best_audio_spoof_model_unweighted.pth \
  --checkpoint Models/Itteration1/best_audio_spoof_model_weighted.pth \
  --asvspoof dev eval --source image --output-root reports/baseline \
  --experiment-id E0.3 --title "Frozen two-checkpoint baseline"
```

The command writes `metrics.json`, per-clip `scores_<split>.csv`, and
`summary.md` under `reports/baseline/<checkpoint-stem>/`. It verifies each
checkpoint key mapping with zero missing/unexpected keys, selects only the dev
EER threshold, and never tunes on eval. The frozen table and exact numbers are
in [`reports/baseline/README.md`](reports/baseline/README.md); the experiment
report is [`reports/phase0/E0.3/README.md`](reports/phase0/E0.3/README.md).

`make evaluate-rerun` writes a second full run to `/tmp/opencode/e0.3-rerun`, and
`make freeze-baseline` runs both plus `scripts/freeze_baseline.py`, which audits
the manifest hash, metric contract, run-to-run EER tolerance, and a
trained-vs-reinitialised head control. See
[`reports/phase0/E0.2/README.md`](reports/phase0/E0.2/README.md) for the harness
design and the unweighted-only first results.

## Data and leakage audit

Confirm that eval is genuinely held out before trusting any eval number:

```bash
make audit-data
# equivalent:
uv run python scripts/audit_data.py
```

The audit is read-only over `data/`, `Models/`, and `reports/baseline/`. It
checks protocol/metadata row alignment, pairwise speaker and filename isolation,
SHA-256 content duplicates across all 122,299 audio files and 131,541 images,
training-manifest containment and identity, exclusion of the 838 `_A` enrollment
files, and baseline threshold lineage (dev selection only). It writes
`reports/audit/data_audit.md` and `reports/audit/data_audit.json` and exits
nonzero if any check fails. Add `--skip-hashes` for a structural-only run that
marks checks B.3/B.4 as skipped. The canonical result and its limitations are in
[`reports/phase0/E0.5/README.md`](reports/phase0/E0.5/README.md).

The audit finds **no train/dev/eval leakage**: zero cross-split speaker,
filename, audio-hash, or image-hash collisions, and zero training clips in
dev/eval. Dev is still a documented **model-selection split** (early stopping and
the E0.3 threshold were chosen on dev); eval is the clean estimate.

To score an external corpus, provide a labelled CSV with a `label` column and a
path column (`image_path`/`img_path` for `--source image`, or
`source_flac`/`audio_path`/`file_path`/`path` for `--source audio`):

```bash
uv run python -m spoof_detection.evaluate \
  --checkpoint Models/Itteration1/best_audio_spoof_model_unweighted.pth \
  --manifest path/to/manifest.csv --source image --split-name my_corpus \
  --output-dir reports/external/my_corpus
```

`--limit N` produces a deterministic class-stratified smoke-test subset and
names the output directory `<checkpoint>_partialN`. Partial runs are not
baseline results. Manifests may be single-class; EER and AUC are then reported
as `n/a`, while threshold and score summaries are still produced.

## Demo

```bash
make app
```

Open <http://localhost:8000> and upload a `.flac`, `.wav`, or `.mp3` clip. The
app returns a spoof score, a thresholded label at `p >= 0.5`, and the generated
mel spectrogram. The score is uncalibrated; see `AGENTS.md` and `ROADMAP.md`. The
app loads its checkpoint strictly and refuses to start with a missing or
mismatched state dict.

## Tests

```bash
make test
```

The suite covers E0.1 reconstruction, dataset validation, metrics, model loading,
evaluation, baseline audit, data audit, preprocessing, and app contracts:

- `tests/test_preprocessing.py` locks audio loading/padding, mel and dB
  computation, min-max normalisation, `magma` rendering and vertical flip, resize
  plus ImageNet normalisation, and parity with stored dev PNGs.
- `tests/test_model_loading.py` checks synthetic and real checkpoint remaps,
  missing/unexpected key failures, parameter counts, and frozen head identity.
- `tests/test_app_contract.py` checks unsupported file rejection, temp-file
  cleanup, strict checkpoint loading, and the response contract.
- `tests/test_audit.py` checks split alignment, speaker/filename isolation,
  duplicate-hash classification, manifest containment, enrollment exclusion, and
  baseline threshold lineage, plus a real-data structural audit.

The dataset validator tests use a miniature synthetic tree and do not require the
7.6 GB download. The E0.1 reconstruction test reads the real `data/LA` and
`data/dataset_img/train` exports. Tests that need the ASVspoof protocols, image
exports, or local checkpoints skip when those artifacts are missing.

## Known limitations

- The sigmoid score is not calibrated. Do not use the demo's "confidence" for
  risk-based decisions.
- The external smoke test in `tests/test1/` contains eight spoofed clips and no
  bonafide controls; it measures neither specificity nor EER.
- The evaluation harness currently reports EER, AUC, threshold metrics, and
  diagnostics but not min t-DCF, confidence intervals, or significance tests.
- Both first-iteration checkpoints now have a frozen two-model baseline; the
  weighted model improves threshold-free EER only slightly and mainly shifts the
  operating point. Dev was used for early stopping, so it is a model-selection
  split rather than a clean holdout.
- The data audit proves eval is isolated by clip identity, speaker, content hash,
  and artifact lineage. It cannot detect re-encoded or resampled copies, and it
  cannot prove anything about uncommitted historical runs that left no artifact.
- The current model is a frozen ImageNet backbone with a small linear head. It is
  a research probe, not a production anti-spoofing system.