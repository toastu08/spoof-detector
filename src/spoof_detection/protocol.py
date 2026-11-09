"""ASVspoof 2019 LA countermeasure protocol loading.

The protocol file is whitespace-separated with five columns::

    SPEAKER_ID AUDIO_FILE_NAME - SYSTEM_ID KEY

Row order is preserved and exposed as ``protocol_row``.  This matters because
the exported image metadata drops ``audio_file_name``; provenance for the
original training subset can only be recovered by row position in the protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "LA"
CM_PROTOCOLS_DIRNAME = "ASVspoof2019_LA_cm_protocols"

PROTOCOL_COLUMNS = ["speaker_id", "audio_file_name", "ignored", "system_id", "label"]
LABEL_MAP = {"bonafide": 0, "spoof": 1}


@dataclass(frozen=True)
class CMProtocolSpec:
    """Expected layout and counts for one ASVspoof 2019 LA CM split."""

    name: str
    protocol_filename: str
    split_dirname: str
    flac_dirname: str
    expected_rows: int
    expected_counts: tuple[tuple[int, int], ...]


SPLIT_SPECS: dict[str, CMProtocolSpec] = {
    "train": CMProtocolSpec(
        name="train",
        protocol_filename="ASVspoof2019.LA.cm.train.trn.txt",
        split_dirname="ASVspoof2019_LA_train",
        flac_dirname="flac",
        expected_rows=25_380,
        expected_counts=((0, 2_580), (1, 22_800)),
    ),
    "dev": CMProtocolSpec(
        name="dev",
        protocol_filename="ASVspoof2019.LA.cm.dev.trl.txt",
        split_dirname="ASVspoof2019_LA_dev",
        flac_dirname="flac",
        expected_rows=24_844,
        expected_counts=((0, 2_548), (1, 22_296)),
    ),
    "eval": CMProtocolSpec(
        name="eval",
        protocol_filename="ASVspoof2019.LA.cm.eval.trl.txt",
        split_dirname="ASVspoof2019_LA_eval",
        flac_dirname="flac",
        expected_rows=71_237,
        expected_counts=((0, 7_355), (1, 63_882)),
    ),
}

DEFAULT_PROTOCOL_PATH = (
    DEFAULT_DATA_ROOT
    / CM_PROTOCOLS_DIRNAME
    / SPLIT_SPECS["train"].protocol_filename
)
DEFAULT_AUDIO_DIR = (
    DEFAULT_DATA_ROOT
    / SPLIT_SPECS["train"].split_dirname
    / SPLIT_SPECS["train"].flac_dirname
)

EXPECTED_TRAIN_ROWS = SPLIT_SPECS["train"].expected_rows
EXPECTED_TRAIN_COUNTS = dict(SPLIT_SPECS["train"].expected_counts)


def _get_spec(split: str) -> CMProtocolSpec:
    """Return the split specification or raise a helpful error."""
    try:
        return SPLIT_SPECS[split]
    except KeyError:
        raise ValueError(
            f"unknown ASVspoof split {split!r}; expected one of {sorted(SPLIT_SPECS)}"
        ) from None


def protocol_path_for(
    split: str,
    data_root: str | Path = DEFAULT_DATA_ROOT,
) -> Path:
    """Return the CM protocol path for ``split`` under ``data_root``."""
    spec = _get_spec(split)
    return Path(data_root) / CM_PROTOCOLS_DIRNAME / spec.protocol_filename


def audio_dir_for(
    split: str,
    data_root: str | Path = DEFAULT_DATA_ROOT,
) -> Path:
    """Return the FLAC directory for ``split`` under ``data_root``."""
    spec = _get_spec(split)
    return Path(data_root) / spec.split_dirname / spec.flac_dirname


def _repo_relative(path: Path, repo_root: Path) -> str:
    """Return ``path`` relative to ``repo_root`` when possible, else absolute."""
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(Path(repo_root).resolve()))
    except ValueError:
        return str(resolved)


def load_cm_protocol(
    split: str,
    data_root: str | Path | None = None,
    protocol_path: str | Path | None = None,
    audio_dir: str | Path | None = None,
    repo_root: str | Path | None = None,
) -> pd.DataFrame:
    """Load an ASVspoof 2019 LA CM protocol in its original row order.

    Returns a dataframe with columns ``protocol_row``, ``speaker_id``,
    ``audio_file_name``, ``system_id``, ``label`` (0 = bonafide, 1 = spoof) and
    ``source_flac`` (path relative to ``repo_root`` where possible).
    """
    spec = _get_spec(split)
    data_root = Path(data_root) if data_root is not None else DEFAULT_DATA_ROOT
    protocol_path = (
        Path(protocol_path)
        if protocol_path is not None
        else protocol_path_for(split, data_root)
    )
    audio_dir = (
        Path(audio_dir)
        if audio_dir is not None
        else audio_dir_for(split, data_root)
    )
    repo_root = Path(repo_root) if repo_root is not None else REPO_ROOT

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


def load_train_protocol(
    protocol_path: str | Path | None = None,
    audio_dir: str | Path | None = None,
    repo_root: str | Path | None = None,
) -> pd.DataFrame:
    """Load the ASVspoof 2019 LA CM training protocol in its original row order.

    This is a backwards-compatible wrapper around :func:`load_cm_protocol`.
    """
    return load_cm_protocol(
        "train",
        protocol_path=protocol_path,
        audio_dir=audio_dir,
        repo_root=repo_root,
    )


def validate_cm_protocol(df: pd.DataFrame, split: str) -> list[str]:
    """Return a list of problems with a loaded CM protocol (empty = valid)."""
    spec = _get_spec(split)
    problems: list[str] = []
    expected_counts = dict(spec.expected_counts)

    if len(df) != spec.expected_rows:
        problems.append(
            f"expected {spec.expected_rows} protocol rows, found {len(df)}"
        )

    counts = df["label"].value_counts().to_dict()
    for label, expected in sorted(expected_counts.items()):
        if counts.get(label, 0) != expected:
            problems.append(
                f"expected {expected} rows with label={label}, "
                f"found {counts.get(label, 0)}"
            )

    if df["label"].isna().any():
        problems.append("protocol contains unlabelled rows")

    if list(df["protocol_row"]) != list(range(len(df))):
        problems.append("protocol_row is not a contiguous 0-based range")

    if df["audio_file_name"].duplicated().any():
        problems.append("protocol contains duplicate audio_file_name values")

    return problems


def validate_train_protocol(df: pd.DataFrame) -> list[str]:
    """Return a list of problems with a loaded training protocol (empty = valid)."""
    return validate_cm_protocol(df, "train")
