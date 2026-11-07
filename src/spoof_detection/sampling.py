"""Deterministic reconstruction of the first-iteration 10,080-clip training set.

The recipe is copied from ``1.1_data_prep-train-old.ipynb`` (execution_count 9)::

    df_real = df_train[df_train['label'] == 0]
    df_fake = df_train[df_train['label'] == 1]

    df_fake_sampled = df_fake.sample(n=7500, random_state=42)

    df_train_balanced = pd.concat([df_real, df_fake_sampled])
    df_train_balanced = df_train_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

The second shuffle defines the exported ``sample_<index>.png`` filenames.
"""

from __future__ import annotations

import pandas as pd

DEFAULT_SEED = 42
DEFAULT_N_SPOOF = 7_500

EXPECTED_SUBSET_ROWS = 10_080
EXPECTED_SUBSET_COUNTS = {0: 2_580, 1: 7_500}

# (speaker_id, system_id, label) tuples captured from the surviving output of
# 1.1_data_prep-train-old.ipynb.  Ten come from the final 10,080-row display
# (head and tail); ten come from the seedless sample(10) of the balanced frame,
# whose row indices are fixed by the seed-42 shuffle.
SEED42_ANCHORS: dict[int, tuple[str, str, int]] = {
    0: ("LA_0098", "A05", 1),
    1: ("LA_0079", "-", 0),
    2: ("LA_0089", "-", 0),
    3: ("LA_0080", "A03", 1),
    4: ("LA_0088", "A05", 1),
    3_748: ("LA_0083", "-", 0),
    7_473: ("LA_0088", "A05", 1),
    9_500: ("LA_0085", "A01", 1),
    3_028: ("LA_0087", "A05", 1),
    9_487: ("LA_0080", "A02", 1),
    8_204: ("LA_0098", "-", 0),
    2_208: ("LA_0079", "A03", 1),
    375: ("LA_0084", "-", 0),
    9_505: ("LA_0079", "A02", 1),
    687: ("LA_0083", "A01", 1),
    10_075: ("LA_0098", "A05", 1),
    10_076: ("LA_0083", "A02", 1),
    10_077: ("LA_0096", "A01", 1),
    10_078: ("LA_0085", "-", 0),
    10_079: ("LA_0082", "A06", 1),
}

REQUIRED_COLUMNS = {
    "speaker_id",
    "audio_file_name",
    "system_id",
    "label",
    "source_flac",
}


def build_seed42_subset(
    df: pd.DataFrame,
    n_spoof: int = DEFAULT_N_SPOOF,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Reproduce the 10,080-clip seed-42 training subset.

    ``df`` must be the protocol loaded in original row order.  The returned
    frame is in final shuffled order with ``sample_index`` (0..10079) and the
    original ``protocol_row`` for each clip.
    """
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"input dataframe is missing columns: {sorted(missing)}")

    df = df.drop(columns=["protocol_row"], errors="ignore").copy()

    df_real = df[df["label"] == 0]
    df_fake = df[df["label"] == 1]

    expected_bonafide = EXPECTED_SUBSET_COUNTS[0]
    if len(df_real) != expected_bonafide:
        raise ValueError(
            f"expected {expected_bonafide} bonafide rows, found {len(df_real)}"
        )
    if n_spoof > len(df_fake):
        raise ValueError(
            f"cannot sample {n_spoof} spoof rows from a pool of {len(df_fake)}"
        )

    df_fake_sampled = df_fake.sample(n=n_spoof, random_state=seed)

    combined = pd.concat([df_real, df_fake_sampled])
    combined = combined.sample(frac=1, random_state=seed)

    protocol_rows = combined.index.to_numpy()
    balanced = combined.reset_index(drop=True)
    balanced.insert(0, "protocol_row", protocol_rows)
    balanced.insert(0, "sample_index", range(len(balanced)))
    return balanced


def validate_seed42_subset(subset: pd.DataFrame) -> list[str]:
    """Return a list of problems with a reconstructed subset (empty = valid)."""
    problems: list[str] = []

    if len(subset) != EXPECTED_SUBSET_ROWS:
        problems.append(
            f"expected {EXPECTED_SUBSET_ROWS} rows, found {len(subset)}"
        )

    counts = subset["label"].value_counts().to_dict()
    for label, expected in EXPECTED_SUBSET_COUNTS.items():
        if counts.get(label, 0) != expected:
            problems.append(
                f"expected {expected} rows with label={label}, found {counts.get(label, 0)}"
            )

    if list(subset["sample_index"]) != list(range(len(subset))):
        problems.append("sample_index is not a contiguous 0-based range")

    if subset["audio_file_name"].duplicated().any():
        problems.append("subset contains duplicate audio_file_name values")

    if subset["protocol_row"].duplicated().any():
        problems.append("subset contains duplicate protocol_row values")

    for sample_index, expected in SEED42_ANCHORS.items():
        if sample_index >= len(subset):
            problems.append(f"anchor sample_index {sample_index} is out of range")
            continue
        row = subset.iloc[sample_index]
        got = (row["speaker_id"], row["system_id"], int(row["label"]))
        if got != expected:
            problems.append(
                f"anchor {sample_index}: expected {expected}, found {got}"
            )

    attack_set = set(subset.loc[subset["label"] == 1, "system_id"].unique())
    if attack_set != {"A01", "A02", "A03", "A04", "A05", "A06"}:
        problems.append(f"unexpected attack set: {sorted(attack_set)}")

    if subset["speaker_id"].nunique() != 20:
        problems.append(
            f"expected 20 speakers, found {subset['speaker_id'].nunique()}"
        )

    return problems


def subset_summary(subset: pd.DataFrame) -> dict[str, object]:
    """Small JSON-serialisable summary of a reconstructed subset."""
    spoof = subset[subset["label"] == 1]
    return {
        "rows": len(subset),
        "label_counts": {
            "bonafide": int((subset["label"] == 0).sum()),
            "spoof": int((subset["label"] == 1).sum()),
        },
        "speakers": int(subset["speaker_id"].nunique()),
        "bonafide_speakers": int(
            subset.loc[subset["label"] == 0, "speaker_id"].nunique()
        ),
        "spoof_speakers": int(
            subset.loc[subset["label"] == 1, "speaker_id"].nunique()
        ),
        "attack_counts": {
            str(k): int(v)
            for k, v in sorted(spoof["system_id"].value_counts().items())
        },
    }
