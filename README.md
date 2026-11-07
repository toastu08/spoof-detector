# Spoof Detection

Research code for **audio spoofing / deepfake speech detection** on ASVspoof 2019
Logical Access (LA). The current baseline renders a 4-second clip as a
magma-coloured mel spectrogram and classifies it with a frozen ImageNet-pretrained
ResNet18 plus a small linear head.

> **Status (2026-11-07):** Phase 0, E0.1 complete. The 10,080-clip training
> subset is reconstructed at `data/derived/train_seed42/` with manifest
> `manifests/train_seed42.csv`. The evaluation harness (E0.2), frozen baseline
> table (E0.3), contract tests (E0.4), and leakage audit (E0.5) are still open.

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
uv run python -m unittest discover -s tests -p "test_*.py" -v
```

Some tests read `data/LA` and `data/dataset_img/train`, so run them after the
dataset is in place.

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
| `src/spoof_detection/` | Reusable protocol, sampling, preprocessing, and dataset-validation code |
| `scripts/` | Reproduction and dataset-verification entry points |
| `manifests/` | Data manifests and provenance metadata |
| `reports/` | Experiment reports |
| `Models/` | Model iteration notes and (gitignored) checkpoints |
| `data/` | Downloaded/derived data (gitignored) |

## Demo

```bash
make app
```

Open <http://localhost:8000> and upload a `.flac`, `.wav`, or `.mp3` clip. The
app returns a spoof score, a thresholded label at `p >= 0.5`, and the generated
mel spectrogram. The score is uncalibrated; see `AGENTS.md` and `ROADMAP.md`.

## Tests

```bash
uv run python -m unittest discover -s tests -p "test_*.py" -v
```

The dataset validator tests use a miniature synthetic tree and do not require the
7.6 GB download. The E0.1 reconstruction test reads the real `data/LA` and
`data/dataset_img/train` exports.

## Known limitations

- The sigmoid score is not calibrated. Do not use the demo's "confidence" for
  risk-based decisions.
- The external smoke test in `tests/test1/` contains eight spoofed clips and no
  bonafide controls; it measures neither specificity nor EER.
- There is no automated evaluation harness yet (E0.2), so there is no frozen
  baseline table for dev/eval.
- The current model is a frozen ImageNet backbone with a small linear head. It is
  a research probe, not a production anti-spoofing system.
