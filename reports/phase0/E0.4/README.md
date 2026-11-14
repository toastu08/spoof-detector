# E0.4 — Preprocessing, checkpoint-loading, and app-contract tests

**Status:** complete (2026-11-14)
**Git commit at run time:** `9c00d3c6f6f6b53c9a7892eb5e8320b336ddae94`
(with the E0.4 changes uncommitted in the working tree)
**Preprocessing version:** `mel128-magma-4s-v1`
**Training manifest:** `manifests/train_seed42.csv`
(SHA-256 `22f98abdf3392e04f7df3632809f81d033d5687773946cd103f5066846ed195e`)
**Frozen baseline:** [`reports/baseline/README.md`](../../baseline/README.md)

## Metric contract

| Split | Attack subset | Threshold(s) | Samples | Bonafide / spoof |
|---|---|---|---:|---:|
| dev | A01–A06 | threshold-free, fixed `0.5`, dev-selected EER threshold | 24,844 | 2,548 / 22,296 |
| eval | A07–A19 | threshold-free, fixed `0.5`, dev-selected EER threshold | 71,237 | 7,355 / 63,882 |
| dev (smoke) | A01–A06, deterministic 100-clip subset | threshold-free | 100 | 50 / 50 |
| eval (smoke) | A07–A19, deterministic 100-clip subset | threshold-free | 100 | 50 / 50 |

The dev EER threshold is selected on dev only. Eval is never used for
threshold selection. The smoke rows are parity checks, not benchmark results.
Dev remains a **model-selection split** because the original training loop used
it for early stopping; eval is the clean estimate.

## Question

Can a future change silently break inference through the audio-to-image
pipeline, the checkpoint loader, or the FastAPI app — and does the refactor that
makes those paths testable preserve the frozen baseline exactly?

## Method

- Refactor the app to use the package model definition, the strict
  `load_model_from_checkpoint`, and the package preprocessing helpers instead of
  duplicating them.
- Extract behavior-preserving helpers (`load_fixed_audio`, `normalize_mel_db`,
  `colorize_mel`, `render_mel_rgb`) so each contract step is testable without
  changing the stored-image export path.
- Add preprocessing tests against synthetic audio and real dev clips, including
  pixel parity with the stored export.
- Extend checkpoint tests with the real first-iteration artifacts, frozen head
  fingerprints, and a deliberately broken key prefix.
- Add app tests that call the endpoint coroutine directly with a stub model and a
  real `UploadFile`, checking rejection, cleanup, strict loading, and the
  response contract.
- Rerun both checkpoints on full dev and eval with image input into
  `/tmp/opencode/e0.4-rerun` and byte-compare with the frozen baseline.
- Rerun a deterministic 100-clip-per-split subset from images and from source
  audio and compare logits and scores row by row.

Nothing under `data/` or `Models/` was modified, no checkpoint was overwritten,
and no metric in `reports/baseline/` was rewritten.

Pre-registered success criteria:

1. `make test` passes in the project environment.
2. A deliberately broken `network.` prefix raises instead of loading silently.
3. Full dev/eval score CSVs are byte-identical to the frozen baseline, and
   EER/AUC are identical.
4. Image and audio inputs produce bit-identical logits and scores on the smoke
   subset.
5. The app rejects unsupported extensions before creating a temp file, always
   cleans up temp files, and uses `p >= 0.5` for spoof.

## Test matrix

| File | Tests | What it locks |
|---|---:|---|
| `tests/test_preprocessing.py` | 19 | 16 kHz load, 64,000-sample pad/truncate, 128-band mel, `power_to_db(ref=np.max)`, min-max normalisation including silence, `magma` + vertical flip, 224×224 resize, ImageNet normalisation, contract constants, and pixel parity with the stored dev PNGs |
| `tests/test_model_loading.py` | 12 | Synthetic and real checkpoint remaps, exact prefix handling, missing/unexpected-key failure, 124 keys / 11,242,305 parameters / 65,793 head parameters, frozen head SHA-256 and bias, batch-norm counters, root checkpoint byte identity |
| `tests/test_app_contract.py` | 8 | Unsupported-extension rejection before temp-file creation, `.flac`/`.wav`/`.mp3` acceptance, temp-file cleanup on success and failure, strict-load report, spoof/bonafide responses, `p = 0.5` boundary, spectrogram PNG response |

The other 77 tests cover E0.1 reconstruction, dataset validation, metrics,
evaluation, and the baseline audit. Total: **116 tests**.

## Commands

```bash
# Full suite
make test

# Full rerun (both checkpoints, full dev + eval, image input)
uv run python -m spoof_detection.evaluate \
  --checkpoint Models/Itteration1/best_audio_spoof_model_unweighted.pth \
  --checkpoint Models/Itteration1/best_audio_spoof_model_weighted.pth \
  --asvspoof dev eval --source image \
  --output-root /tmp/opencode/e0.4-rerun \
  --experiment-id E0.4 --title "E0.4 preprocessing parity rerun"

# Image-vs-audio smoke on a deterministic 100-clip-per-split subset
uv run python -m spoof_detection.evaluate \
  --checkpoint Models/Itteration1/best_audio_spoof_model_unweighted.pth \
  --asvspoof dev eval --source image --limit 100 \
  --output-root /tmp/opencode/e0.4-image-smoke

uv run python -m spoof_detection.evaluate \
  --checkpoint Models/Itteration1/best_audio_spoof_model_unweighted.pth \
  --asvspoof dev eval --source audio --limit 100 \
  --output-root /tmp/opencode/e0.4-audio-smoke
```

## Results

### Test suite

`make test` ran **116 tests in ~19 s: OK**. No test was skipped in this
environment; tests that need the ASVspoof protocols, image exports, or local
checkpoints skip only when those artifacts are absent.

### Full dev/eval rerun

| Checkpoint | Split | Attack subset | Threshold | Samples (bonafide/spoof) | EER | AUC | Score CSV |
|---|---|---|---|---:|---:|---:|---|
| unweighted | dev | A01–A06 | threshold-free | 24,844 (2,548 / 22,296) | 6.16% | 0.9860 | byte-identical |
| unweighted | eval | A07–A19 | threshold-free | 71,237 (7,355 / 63,882) | 11.50% | 0.9527 | byte-identical |
| weighted | dev | A01–A06 | threshold-free | 24,844 (2,548 / 22,296) | 5.42% | 0.9881 | byte-identical |
| weighted | eval | A07–A19 | threshold-free | 71,237 (7,355 / 63,882) | 11.13% | 0.9558 | byte-identical |

All four `scores_{dev,eval}.csv` files are byte-for-byte identical to
`reports/baseline/`, and every EER/AUC value matches to full floating-point
precision. The refactor therefore changed code structure, not model input or
scores.

### Image-vs-audio smoke parity

| Split | Samples (bonafide/spoof) | Rows aligned | Labels equal | Max abs logit diff | Max abs score diff |
|---|---:|---|---|---:|---:|
| dev | 100 (50 / 50) | yes | yes | 0.0 | 0.0 |
| eval | 100 (50 / 50) | yes | yes | 0.0 | 0.0 |

The image and audio paths produce bit-identical logits on both subsets.

### Checkpoint and app loading

Both real checkpoints load with 124 state-dict keys, 124 remapped keys, zero
missing keys, zero unexpected keys, 11,242,305 parameters, and 65,793 head
parameters. The root app checkpoint is byte-identical to the `Models` unweighted
copy. Importing `app` reports the same strict-load summary, and a deliberately
broken prefix raises `CheckpointError` before the app can serve predictions.

### App contract

The endpoint returns `prediction`, `confidence`, `raw_probability`, and a
base64 PNG. `p >= 0.5` is labelled spoof, `p < 0.5` bonafide, and
`confidence = max(p, 1-p)` as a percentage. Unsupported extensions are rejected
with HTTP 400 before any temp file is created, and every success/failure path
removes its temp file.

## Artifacts

- `tests/test_preprocessing.py`
- `tests/test_model_loading.py` (extended)
- `tests/test_app_contract.py`
- `src/spoof_detection/preprocess.py` (testable helpers; export path unchanged)
- `app.py` (strict loader and shared preprocessing)
- `makefile` (`make test`)
- `reports/phase0/E0.4/README.md` (this report)

## Limitations

- App tests call the endpoint coroutine directly with a stub model; they do not
  run an HTTP client or push audio through the real checkpoint end to end.
  Real-model inference is covered by the checkpoint tests and the evaluation
  smoke runs.
- The demo still has no upload size limit, allows all CORS origins, and writes
  every upload with a `.flac` suffix.
- The reported "confidence" remains `max(p, 1-p)`, not a calibrated probability.
- The full-rerun and smoke outputs live under `/tmp/opencode/` and are not
  committed; the frozen numbers in `reports/baseline/` remain the reference.
- There is still no CI pipeline; `make test` is a local command.
