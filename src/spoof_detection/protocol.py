"""ASVspoof 2019 LA countermeasure protocol loading.

The protocol file is whitespace-separated with five columns::

    SPEAKER_ID AUDIO_FILE_NAME - SYSTEM_ID KEY

Row order is preserved and exposed as ``protocol_row``.  This matters because
the exported image metadata drops ``audio_file_name``; provenance for the
original training subset can only be recovered by row position in the protocol.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_PROTOCOL_PATH = (
    REPO_ROOT
    / "data"
    / "LA"
    / "ASVspoof2019_LA_cm_protocols"
    / "ASVspoof2019.LA.cm.train.trn.txt"
)
DEFAULT_AUDIO_DIR = REPO_ROOT / "data" / "LA" / "ASVspoof2019_LA_train" / "flac"

PROTOCOL_COLUMNS = ["speaker_id", "audio_file_name", "ignored", "system_id", "label"]
LABEL_MAP = {"bonafide": 0, "spoof": 1}

EXPECTED_TRAIN_ROWS = 25_380
EXPECTED_TRAIN_COUNTS = {0: 2_580, 1: 22_800}


def _repo_relative(path: Path, repo_root: Path) -> str:
    """Return ``path`` relative to ``repo_root`` when possible, else absolute."""
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(Path(repo_root).resolve()))
    except ValueError:
        return str(resolved)


def load_train_protocol(
    protocol_path: str | Path | None = None,
    audio_dir: str | Path | None = None,
    repo_root: str | Path | None = None,
) -> pd.DataFrame:
    """Load the ASVspoof 2019 LA CM training protocol in its original row order.

    Returns a dataframe with columns ``protocol_row``, ``speaker_id``,
    ``audio_file_name``, ``system_id``, ``label`` (0 = bonafide, 1 = spoof) and
    ``source_flac`` (path relative to ``repo_root`` where possible).
    """
    repo_root = Path(repo_root) if repo_root is not None else REPO_ROOT
    protocol_path = Path(protocol_path) if protocol_path is not None else DEFAULT_PROTOCOL_PATH
    audio_dir = Path(audio_dir) if audio_dir is not None else DEFAULT_AUDIO_DIR

    df = pd.read_csv(
        protocol_path,
        sep=" ",
        header=None,
        names=PROTOCOL_COLUMNS,
    )
    df = df.drop(columns=["ignored"])

    df["label"] = df["label"].map(LABEL_MAP)
    if df["label"].isna().any():
        bad = df.loc[df["label"].isna(), "speaker_id"].tolist()
        raise ValueError(f"Unrecognised label values in protocol for speakers: {bad[:5]}")

    df.insert(0, "protocol_row", range(len(df)))
    df["source_flac"] = [
        _repo_relative(audio_dir / f"{name}.flac", repo_root)
        for name in df["audio_file_name"]
    ]
    return df


def validate_train_protocol(df: pd.DataFrame) -> list[str]:
    """Return a list of problems with a loaded training protocol (empty = valid)."""
    problems: list[str] = []

    if len(df) != EXPECTED_TRAIN_ROWS:
        problems.append(f"expected {EXPECTED_TRAIN_ROWS} protocol rows, found {len(df)}")

    counts = df["label"].value_counts().to_dict()
    for label, expected in EXPECTED_TRAIN_COUNTS.items():
        if counts.get(label, 0) != expected:
            problems.append(
                f"expected {expected} rows with label={label}, found {counts.get(label, 0)}"
            )

    if list(df["protocol_row"]) != list(range(len(df))):
        problems.append("protocol_row is not a contiguous 0-based range")

    if df["audio_file_name"].duplicated().any():
        problems.append("protocol contains duplicate audio_file_name values")

    return problems
