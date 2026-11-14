# E0.5 — Data and leakage audit

**Status:** PASS  
**Generated:** 2026-11-14T16:49:44.255351+00:00  
**Git commit:** `ef487b0f7e482e5829eadf194b51f8570192eb9f` (working tree dirty)
**Training manifest:** `manifests/train_seed42.csv` (SHA-256 `22f98abdf3392e04f7df3632809f81d033d5687773946cd103f5066846ed195e`)  
**Command:** `uv run python scripts/audit_data.py`

## Metric contract

| Split | Attack subset | Threshold | Samples | Bonafide / spoof |
|---|---|---|---:|---:|
| train | A01–A06 | not applicable (audit) | 25,380 | 2,580 / 22,800 |
| dev | A01–A06 | not applicable (audit) | 24,844 | 2,548 / 22,296 |
| eval | A07–A19 | not applicable (audit) | 71,237 | 7,355 / 63,882 |

Baseline lineage rows are audited threshold-free: the dev-selected EER threshold and the fixed `0.5` threshold are checked as artifacts, not re-reported as new performance numbers.

## Inputs and hashes

| Input | SHA-256 |
|---|---|
| `manifests/train_seed42.csv` | `22f98abdf3392e04f7df3632809f81d033d5687773946cd103f5066846ed195e` |
| `data/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt` | `65a289e4009400b42bb4f1e47ac60938d93f7f595452d1f220d35defc3de8ab8` |
| `data/dataset_img/train/metadata_train.csv` | `4ad4e41451c2ce84dab1df49ee17175ce8a6d74734e2995a8ea917de20ce47cd` |
| `data/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt` | `2fedaebf1ba3ae1a9bb7d3ec916804ac237b4eee063a27fa92e6549f03d129c7` |
| `data/dataset_img/dev/metadata_dev.csv` | `6097d85f173bc0e852579f94c5c6551a12267e008feb1027f3ec576e105b3b36` |
| `data/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt` | `84fce576efff092269a0a6ad247d2e9f2f7b4d4e25a4545efb2518da92708036` |
| `data/dataset_img/eval/metadata_eval.csv` | `1ae85a9663647add8d33f7860a96be06fde48785b296de8456baf4a3a27d58a6` |

## Inventory

| Split | Protocol rows | Bonafide | Spoof | Speakers | FLAC files | Enrollment | Images |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 25,380 | 2,580 | 22,800 | 20 | 25,380 | 0 | 25,380 |
| dev | 24,844 | 2,548 | 22,296 | 20 | 24,986 | 142 | 24,844 |
| eval | 71,237 | 7,355 | 63,882 | 67 | 71,933 | 696 | 71,237 |

## Checks

| ID | Check | Status | Detail |
|---|---|---|---|
| A.1 | ASVspoof 2019 LA layout and protocol hashes | PASS | layout, counts, and protocol SHA-256 values match the dataset spec |
| A.2 | train metadata and image alignment | PASS | 25,380 rows align 1:1 with the train protocol; 25,380 images present, 0 missing, 0 orphaned |
| A.3 | dev metadata and image alignment | PASS | 24,844 rows align 1:1 with the dev protocol; 24,844 images present, 0 missing, 0 orphaned |
| A.4 | eval metadata and image alignment | PASS | 71,237 rows align 1:1 with the eval protocol; 71,237 images present, 0 missing, 0 orphaned |
| B.1 | speaker and audio filename isolation | PASS | no speaker or audio filename overlap between train, dev, and eval |
| B.2 | attack-system coverage and unseen eval attacks | PASS | train and dev share A01-A06 by design; eval uses only unseen A07-A19 |
| B.3 | audio content duplicates | PASS | 122,299 files hashed; no cross-split audio duplicates; 0 unexplained within-split groups |
| B.4 | image content duplicates | PASS | 131,541 files hashed; no cross-split image duplicates; 0 unexplained within-split groups |
| C.1 | training manifest integrity against the train protocol | PASS | 10,080 rows (2,580 bonafide / 7,500 spoof) match the frozen manifest hash and their train protocol rows |
| C.2 | training manifest file and hash integrity | PASS | all 10,080 source FLACs and derived images exist and match their recorded SHA-256 |
| C.3 | training manifest isolation from dev and eval | PASS | zero training clips, speakers, or audio contents appear in dev or eval |
| D.1 | enrollment-file exclusion | PASS | 838 enrollment FLACs (142 dev / 696 eval) are excluded from the CM protocols and all image exports |
| E.1 | baseline threshold lineage (dev only) | PASS | every checkpoint selects its EER threshold on dev only and transfers it unchanged to eval |
| E.2 | baseline score/protocol alignment | PASS | every baseline score CSV maps 1:1 to its protocol split with the correct labels, speakers, systems, and source paths |
| E.3 | baseline artifact identity | PASS | both baseline checkpoints and the training manifest match the frozen E0.3 audit |

## Duplicate-hash audit

| Check | Files hashed | Unique hashes | Cross-split groups | Within-split groups | Explained within-split | Unexplained within-split |
|---|---:|---:|---:|---:|---:|---:|
| B.3 | 122,299 | 122,299 | 0 | 0 | 0 | 0 |
| B.4 | 131,541 | 121,461 | 0 | 10,080 | 10,080 | 0 |

The image check covers `data/dataset_img/{train,dev,eval}` and the reconstructed `data/derived/train_seed42/`. The derived train subset is labelled `train`; its byte-identical overlap with the full train export is the only permitted within-split duplicate.

## Conclusion

- `B.3` PASS: 122,299 files hashed, 0 cross-split duplicate groups, 0 explained within-split groups.
- `B.4` PASS: 131,541 files hashed, 0 cross-split duplicate groups, 10,080 explained within-split groups.
- `C.3` confirms zero training clips, speakers, or audio contents appear in dev or eval.
- `E.1`–`E.3` confirm both checkpoints selected their threshold on dev only and that the baseline artifacts match the frozen E0.3 audit.
- Dev remains a **documented model-selection split**; eval is clean.
- Raw status: `PASS`.

## Dev reuse and eval isolation

- The original training loop early-stopped on **full dev** (`2_model.ipynb` cells 16–17 and 21).
- E0.3 selected the operating threshold on **full dev** and transferred it unchanged to eval.
- Dev is therefore a **model-selection split**, not a clean holdout. This is documented in `reports/baseline/README.md`.
- `eval` was read only by the post-training inference cells (`2_model.ipynb` cells 23 and 25). No eval clip entered training, early stopping, or threshold selection.
- Notebook cell evidence is recorded in `reports/phase0/E0.5/README.md`.

## Reproduction

```bash
uv run python scripts/audit_data.py
```

The audit is read-only: it hashes and compares existing artifacts and writes only this report and its JSON companion.
