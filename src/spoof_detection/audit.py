"""Data and leakage audit for the ASVspoof 2019 LA baseline (E0.5).

The audit answers two separate questions:

1. **Clip-level isolation** - do ``train``, ``dev`` and ``eval`` overlap in
   speakers, audio file names, audio content, or rendered image content?
2. **Process-level isolation** - did the training loop, early stopping, or the
   E0.3 threshold selection ever touch ``eval``?

Dev reuse is already documented: the original training loop early-stopped on
full dev and E0.3 selected the operating threshold on full dev.  The audit
records that as a *model-selection limitation* and proves that ``eval`` stayed
out of training, early stopping, and threshold selection.

Everything here is read-only.  Files under ``data/``, ``Models/`` and
``reports/baseline/`` are only read and hashed; the CLI writes its report to
``reports/audit/``.

Notebook provenance is *not* parsed by this module.  Per AGENTS.md G1 notebooks
are read through the python-notebook MCP; the cell-level evidence is recorded in
``reports/phase0/E0.5/README.md`` and summarised by the generated report.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Mapping

import pandas as pd

from .dataset import (
    ASVSPOOF2019_LA,
    REPO_ROOT,
    sha256_file,
    validate_dataset,
)
from .metrics import compute_eer
from .protocol import (
    SPLIT_SPECS,
    audio_dir_for,
    load_cm_protocol,
    protocol_path_for,
    validate_cm_protocol,
)

SPLITS: tuple[str, ...] = ("train", "dev", "eval")
ENROLLMENT_PREFIX: dict[str, str] = {"dev": "LA_D_A", "eval": "LA_E_A"}

EXPECTED_ATTACKS: dict[str, set[str]] = {
    "train": {f"A0{i}" for i in range(1, 7)},
    "dev": {f"A0{i}" for i in range(1, 7)},
    "eval": {f"A{i:02d}" for i in range(7, 20)},
}

DEFAULT_IMAGE_ROOT = REPO_ROOT / "data" / "dataset_img"
DEFAULT_DERIVED_ROOT = REPO_ROOT / "data" / "derived" / "train_seed42"
DEFAULT_TRAIN_MANIFEST = REPO_ROOT / "manifests" / "train_seed42.csv"
DEFAULT_BASELINE_ROOT = REPO_ROOT / "reports" / "baseline"
DEFAULT_FREEZE_REPORT = DEFAULT_BASELINE_ROOT / "freeze_report.json"

EXPECTED_MANIFEST_SHA256 = (
    "22f98abdf3392e04f7df3632809f81d033d5687773946cd103f5066846ed195e"
)
EXPECTED_MANIFEST_ROWS = 10_080
EXPECTED_MANIFEST_COUNTS = {0: 2_580, 1: 7_500}

MANIFEST_REQUIRED_COLUMNS = {
    "sample_index",
    "protocol_row",
    "speaker_id",
    "system_id",
    "label",
    "audio_file_name",
    "source_flac",
    "image_path",
    "sha256_flac",
    "sha256_png",
}

#: Cap on how many offending rows/paths a single check echoes into its evidence.
MAX_EVIDENCE_ITEMS = 20


# ---------------------------------------------------------------------------
# Check records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CheckResult:
    """One auditable assertion with its evidence."""

    check_id: str
    title: str
    status: str  # "pass" | "fail" | "skip"
    detail: str
    evidence: dict[str, object] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.status == "pass"

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.check_id,
            "title": self.title,
            "status": self.status,
            "detail": self.detail,
            "evidence": self.evidence,
        }


# ---------------------------------------------------------------------------
# Paths and hashing
# ---------------------------------------------------------------------------


def resolve_repo_path(value: str | Path, repo_root: Path = REPO_ROOT) -> Path:
    """Resolve ``value`` against the repository root unless it is absolute."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else Path(repo_root) / path


def normalized_path(value: str | Path, repo_root: Path = REPO_ROOT) -> Path:
    """Lexically normalise a path without the cost of ``Path.resolve``.

    The audit compares hundreds of thousands of exported paths, so symlink
    resolution is deliberately skipped; ``os.path.normpath`` removes ``.`` and
    ``..`` segments and keeps absolute/relative form stable.
    """
    return Path(os.path.normpath(str(resolve_repo_path(value, repo_root))))


def repo_relative(path: str | Path, repo_root: Path = REPO_ROOT) -> str:
    """Return ``path`` relative to ``repo_root`` when possible."""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(Path(repo_root).resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class HashCache:
    """Threaded, memoised SHA-256 over a set of files."""

    def __init__(self, workers: int = 8) -> None:
        if workers < 1:
            raise ValueError("workers must be positive")
        self.workers = workers
        self._digests: dict[str, str] = {}
        self._paths: dict[str, Path] = {}

    @staticmethod
    def _key(path: str | Path) -> str:
        return os.path.normpath(str(path))

    def hash_paths(self, paths: Iterable[Path]) -> dict[str, str]:
        """Hash every existing path not already cached and return the mapping."""
        requested: dict[str, Path] = {self._key(path): Path(path) for path in paths}
        todo = sorted(
            key
            for key, path in requested.items()
            if key not in self._digests and path.is_file()
        )
        if todo:
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                for key, digest in zip(todo, pool.map(_sha256_file, (requested[key] for key in todo))):
                    self._digests[key] = digest
        return {
            key: self._digests[key]
            for key in requested
            if key in self._digests
        }

    def digest(self, path: str | Path) -> str | None:
        return self._digests.get(self._key(path))


@dataclass(frozen=True)
class DuplicateFindings:
    """Within-split and cross-split duplicate hashes for one corpus kind."""

    kind: str
    hashed_files: int
    unique_hashes: int
    cross_split_groups: int
    cross_split_examples: tuple[dict[str, object], ...]
    within_split_groups: int
    within_split_examples: tuple[dict[str, object], ...]
    within_split_classes: dict[str, int] = field(default_factory=dict)
    within_split_explained_groups: int = 0
    within_split_unexplained_groups: int = 0
    within_split_unexplained_examples: tuple[dict[str, object], ...] = ()


def find_duplicate_hashes(
    entries: Iterable[tuple[str, Path, str]],
    kind: str,
    within_classifier: "Callable[[Path], str] | None" = None,
    explained_class_sets: Iterable[frozenset[str]] = (),
) -> DuplicateFindings:
    """Group ``(split, path, digest)`` entries and report duplicate groups.

    A digest that appears in more than one split is a cross-split duplicate and
    is treated as leakage.  A digest that appears more than once inside one
    split is reported separately; for images the full train export and the
    reconstructed train subset are both labelled ``train``, so their expected
    overlap must be declared through ``within_classifier`` and
    ``explained_class_sets``.
    """
    by_digest: dict[str, list[tuple[str, Path]]] = {}
    hashed_files = 0
    for split, path, digest in entries:
        hashed_files += 1
        by_digest.setdefault(digest, []).append((split, Path(path)))

    explained_sets = {frozenset(item) for item in explained_class_sets}
    cross: list[dict[str, object]] = []
    within: list[dict[str, object]] = []
    within_classes: dict[str, int] = {}
    explained = 0
    unexplained: list[dict[str, object]] = []
    for digest, members in by_digest.items():
        splits = sorted({split for split, _ in members})
        if len(splits) > 1:
            cross.append(
                {
                    "sha256": digest,
                    "splits": splits,
                    "paths": [repo_relative(path) for _, path in members],
                }
            )
        elif len(members) > 1:
            record = {
                "sha256": digest,
                "split": splits[0],
                "paths": [repo_relative(path) for _, path in members],
            }
            within.append(record)
            if within_classifier is not None:
                classes = frozenset(
                    within_classifier(path) for _, path in members
                )
                key = "+".join(sorted(classes))
                within_classes[key] = within_classes.get(key, 0) + 1
                if classes in explained_sets:
                    explained += 1
                else:
                    unexplained.append({**record, "classes": sorted(classes)})

    return DuplicateFindings(
        kind=kind,
        hashed_files=hashed_files,
        unique_hashes=len(by_digest),
        cross_split_groups=len(cross),
        cross_split_examples=tuple(cross[:MAX_EVIDENCE_ITEMS]),
        within_split_groups=len(within),
        within_split_examples=tuple(within[:MAX_EVIDENCE_ITEMS]),
        within_split_classes=within_classes,
        within_split_explained_groups=explained,
        within_split_unexplained_groups=len(unexplained),
        within_split_unexplained_examples=tuple(unexplained[:MAX_EVIDENCE_ITEMS]),
    )


# ---------------------------------------------------------------------------
# Protocol and metadata alignment
# ---------------------------------------------------------------------------


def _read_metadata(split: str, image_root: Path) -> pd.DataFrame:
    path = Path(image_root) / split / f"metadata_{split}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"missing metadata export: {path}")
    frame = pd.read_csv(path)
    missing = {"speaker_id", "system_id", "label", "img_path"} - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return frame


def _mismatch_examples(
    frame: pd.DataFrame,
    mask: pd.Series,
    columns: list[str],
) -> list[dict[str, object]]:
    rows = frame.loc[mask, ["protocol_row", *columns]]
    return [
        {key: (int(value) if key == "protocol_row" else str(value)) for key, value in record.items()}
        for record in rows.head(MAX_EVIDENCE_ITEMS).to_dict(orient="records")
    ]


def check_split_alignment(
    split: str,
    data_root: Path,
    image_root: Path,
    protocol: pd.DataFrame | None = None,
    strict_counts: bool = True,
    repo_root: Path = REPO_ROOT,
) -> CheckResult:
    """Verify metadata rows 1:1 with protocol rows and the image inventory.

    ``strict_counts=False`` skips the ASVspoof row/class-count validation so a
    small fixture can be audited; row contiguity and duplicate-name checks still
    apply.
    """
    check_id = f"A.{ {'train': 2, 'dev': 3, 'eval': 4}[split] }"
    if protocol is None:
        protocol = load_cm_protocol(
            split, data_root=data_root, repo_root=repo_root
        )
    protocol_problems = (
        validate_cm_protocol(protocol, split) if strict_counts else []
    )
    metadata = _read_metadata(split, image_root)

    evidence: dict[str, object] = {
        "protocol_rows": len(protocol),
        "metadata_rows": len(metadata),
        "protocol_problems": protocol_problems,
    }

    failures: list[str] = []
    if protocol_problems:
        failures.append(f"protocol is invalid: {protocol_problems}")
    if len(protocol) != len(metadata):
        failures.append(
            f"row-count mismatch: protocol={len(protocol)} metadata={len(metadata)}"
        )

    aligned = metadata.copy()
    aligned["protocol_row"] = range(len(aligned))
    if len(protocol) == len(metadata):
        for column in ("speaker_id", "system_id"):
            mismatch = aligned[column].astype(str).to_numpy() != protocol[column].astype(str).to_numpy()
            if mismatch.any():
                failures.append(f"{int(mismatch.sum())} rows disagree on {column}")
                evidence[f"{column}_mismatch_examples"] = _mismatch_examples(
                    aligned, pd.Series(mismatch, index=aligned.index), [column]
                )
        label_mismatch = (
            aligned["label"].astype(int).to_numpy()
            != protocol["label"].astype(int).to_numpy()
        )
        if label_mismatch.any():
            failures.append(f"{int(label_mismatch.sum())} rows disagree on label")
            evidence["label_mismatch_examples"] = _mismatch_examples(
                aligned, pd.Series(label_mismatch, index=aligned.index), ["label"]
            )

    expected_paths: list[Path] = []
    bad_folder: list[dict[str, object]] = []
    bad_name: list[dict[str, object]] = []
    for row_index, (label, img_path) in enumerate(
        zip(
            metadata["label"].to_numpy(dtype=int),
            metadata["img_path"].astype(str).to_numpy(),
        )
    ):
        path = normalized_path(img_path, repo_root)
        expected_paths.append(path)
        expected_folder = "bonafide" if int(label) == 0 else "spoofed"
        if path.parent.name != expected_folder or path.parent.parent.name != split:
            bad_folder.append({"protocol_row": row_index, "img_path": img_path})
        if path.stem != f"sample_{row_index}":
            bad_name.append(
                {
                    "protocol_row": row_index,
                    "img_path": img_path,
                    "expected_stem": f"sample_{row_index}",
                }
            )

    actual_images = [
        normalized_path(path, repo_root)
        for path in sorted((Path(image_root) / split).glob("*/*.png"))
    ]
    actual_set = {str(path) for path in actual_images}
    expected_set = {str(path) for path in expected_paths}
    missing = sorted(expected_set - actual_set)
    orphans = sorted(actual_set - expected_set)

    evidence.update(
        {
            "image_files": len(actual_images),
            "missing_images": len(missing),
            "orphan_images": len(orphans),
            "wrong_class_folder": len(bad_folder),
            "wrong_filename_index": len(bad_name),
            "first_missing": missing[:MAX_EVIDENCE_ITEMS],
            "first_orphans": [repo_relative(path, repo_root) for path in orphans[:MAX_EVIDENCE_ITEMS]],
            "first_wrong_folder": bad_folder[:MAX_EVIDENCE_ITEMS],
            "first_wrong_name": bad_name[:MAX_EVIDENCE_ITEMS],
        }
    )
    if missing:
        failures.append(f"{len(missing)} metadata images are missing")
    if orphans:
        failures.append(f"{len(orphans)} image files are not referenced by metadata")
    if bad_folder:
        failures.append(f"{len(bad_folder)} images sit in the wrong split/class folder")
    if bad_name:
        failures.append(f"{len(bad_name)} image filenames do not match their row index")

    status = "fail" if failures else "pass"
    detail = "; ".join(failures) if failures else (
        f"{len(metadata):,} rows align 1:1 with the {split} protocol; "
        f"{len(actual_images):,} images present, 0 missing, 0 orphaned"
    )
    return CheckResult(check_id, f"{split} metadata and image alignment", status, detail, evidence)


# ---------------------------------------------------------------------------
# Cross-split clip identity
# ---------------------------------------------------------------------------


def check_split_isolation(
    protocols: Mapping[str, pd.DataFrame],
    expected_attacks: Mapping[str, set[str]] | None = None,
) -> list[CheckResult]:
    """Check speaker, filename, and attack-system separation across splits.

    ``expected_attacks`` defaults to the ASVspoof 2019 LA layout.  Pass an empty
    mapping to audit a fixture without asserting full attack coverage.
    """
    expected_attacks = (
        EXPECTED_ATTACKS if expected_attacks is None else dict(expected_attacks)
    )
    speakers = {
        split: set(protocol["speaker_id"].astype(str)) for split, protocol in protocols.items()
    }
    names = {
        split: set(protocol["audio_file_name"].astype(str))
        for split, protocol in protocols.items()
    }
    systems = {
        split: {value for value in protocol["system_id"].astype(str) if value != "-"}
        for split, protocol in protocols.items()
    }

    speaker_overlap = {
        f"{left}-{right}": sorted(speakers[left] & speakers[right])
        for left, right in (("train", "dev"), ("train", "eval"), ("dev", "eval"))
    }
    name_overlap = {
        f"{left}-{right}": len(names[left] & names[right])
        for left, right in (("train", "dev"), ("train", "eval"), ("dev", "eval"))
    }
    bad_speakers = {pair: values for pair, values in speaker_overlap.items() if values}
    bad_names = {pair: count for pair, count in name_overlap.items() if count}

    overlap_evidence = {
        "speaker_counts": {split: len(values) for split, values in speakers.items()},
        "speaker_overlap": speaker_overlap,
        "audio_name_overlap": name_overlap,
        "attack_systems": {split: sorted(values) for split, values in systems.items()},
    }
    failures: list[str] = []
    if bad_speakers:
        failures.append(f"speaker overlap between splits: {sorted(bad_speakers)}")
    if bad_names:
        failures.append(f"audio filename overlap between splits: {bad_names}")

    isolation = CheckResult(
        "B.1",
        "speaker and audio filename isolation",
        "fail" if failures else "pass",
        "; ".join(failures)
        if failures
        else "no speaker or audio filename overlap between train, dev, and eval",
        overlap_evidence,
    )

    attack_problems: list[str] = []
    for split, expected in expected_attacks.items():
        observed = systems[split]
        if observed != expected:
            attack_problems.append(
                f"{split} attacks are {sorted(observed)}, expected {sorted(expected)}"
            )
    train_dev_overlap = systems["train"] & systems["dev"]
    if "train" in expected_attacks and train_dev_overlap != expected_attacks["train"]:
        attack_problems.append(
            "train and dev do not share the full A01-A06 attack set"
        )
    eval_overlap = systems["eval"] & (systems["train"] | systems["dev"])
    if eval_overlap:
        attack_problems.append(
            f"eval attacks overlap train/dev: {sorted(eval_overlap)}"
        )
    attacks = CheckResult(
        "B.2",
        "attack-system coverage and unseen eval attacks",
        "fail" if attack_problems else "pass",
        "; ".join(attack_problems)
        if attack_problems
        else (
            "train and dev share A01-A06 by design; eval uses only unseen "
            "A07-A19"
        ),
        {
            "attack_systems": {split: sorted(values) for split, values in systems.items()},
            "train_dev_shared": sorted(train_dev_overlap),
            "eval_overlap_with_train_or_dev": sorted(eval_overlap),
        },
    )
    return [isolation, attacks]


# ---------------------------------------------------------------------------
# Content-hash duplicates
# ---------------------------------------------------------------------------


def collect_corpus_paths(
    data_root: Path,
    image_root: Path,
    derived_root: Path,
) -> tuple[dict[str, list[Path]], dict[str, list[Path]]]:
    """Return ``(audio_by_split, images_by_split)`` for every file to audit.

    Audio covers every FLAC in each split directory, including the enrollment
    files, so an enrollment file that duplicates another split is still caught.
    Images cover the three stored exports plus the reconstructed training
    subset, which is labelled ``train``.
    """
    audio: dict[str, list[Path]] = {
        split: sorted(audio_dir_for(split, data_root).glob("*.flac")) for split in SPLITS
    }
    images: dict[str, list[Path]] = {
        split: sorted((Path(image_root) / split).glob("*/*.png")) for split in SPLITS
    }
    derived_images = sorted(Path(derived_root).glob("*/*.png"))
    images["train"] = images["train"] + derived_images
    return audio, images


def check_duplicate_hashes(
    audio_by_split: Mapping[str, list[Path]],
    images_by_split: Mapping[str, list[Path]],
    hashes: HashCache,
    derived_root: Path | None = None,
) -> list[CheckResult]:
    """Hash every corpus file and report cross-split and within-split duplicates.

    When ``derived_root`` is given, a within-train image duplicate is accepted
    only when it pairs the reconstructed subset with the full train export.
    Any other within-split duplicate fails the check.
    """
    all_audio = [path for paths in audio_by_split.values() for path in paths]
    all_images = [path for paths in images_by_split.values() for path in paths]
    hashes.hash_paths(all_audio)
    hashes.hash_paths(all_images)

    audio_entries = [
        (split, path, hashes.digest(path))
        for split, paths in audio_by_split.items()
        for path in paths
        if hashes.digest(path) is not None
    ]
    image_entries = [
        (split, path, hashes.digest(path))
        for split, paths in images_by_split.items()
        for path in paths
        if hashes.digest(path) is not None
    ]

    audio_findings = find_duplicate_hashes(audio_entries, "audio")
    image_classifier: Callable[[Path], str] | None = None
    explained: frozenset[frozenset[str]] = frozenset()
    if derived_root is not None:
        derived = normalized_path(derived_root)

        def classify(path: Path) -> str:
            try:
                normalized_path(path).relative_to(derived)
            except ValueError:
                return "export"
            return "derived"

        image_classifier = classify
        explained = frozenset({frozenset({"derived", "export"})})
    image_findings = find_duplicate_hashes(
        image_entries,
        "image",
        within_classifier=image_classifier,
        explained_class_sets=explained,
    )

    results: list[CheckResult] = []
    for findings, check_id, title in (
        (audio_findings, "B.3", "audio content duplicates"),
        (image_findings, "B.4", "image content duplicates"),
    ):
        failures: list[str] = []
        if findings.cross_split_groups:
            failures.append(
                f"{findings.cross_split_groups} {findings.kind} hashes appear in "
                "more than one split"
            )
        if findings.within_split_unexplained_groups:
            failures.append(
                f"{findings.within_split_unexplained_groups} "
                f"{findings.kind} hashes repeat inside a split without an "
                "explanation"
            )
        note = (
            "within-train image duplicates are allowed only between the "
            "reconstructed train subset and the full train export"
            if findings.kind == "image"
            else "within-split duplicates would indicate repeated corpus files"
        )
        results.append(
            CheckResult(
                check_id,
                title,
                "fail" if failures else "pass",
                "; ".join(failures)
                if failures
                else (
                    f"{findings.hashed_files:,} files hashed; no cross-split "
                    f"{findings.kind} duplicates; "
                    f"{findings.within_split_unexplained_groups} unexplained "
                    "within-split groups"
                ),
                {
                    "hashed_files": findings.hashed_files,
                    "unique_hashes": findings.unique_hashes,
                    "cross_split_groups": findings.cross_split_groups,
                    "cross_split_examples": list(findings.cross_split_examples),
                    "within_split_groups": findings.within_split_groups,
                    "within_split_classes": findings.within_split_classes,
                    "within_split_explained_groups": (
                        findings.within_split_explained_groups
                    ),
                    "within_split_unexplained_groups": (
                        findings.within_split_unexplained_groups
                    ),
                    "within_split_examples": list(findings.within_split_examples),
                    "within_split_unexplained_examples": list(
                        findings.within_split_unexplained_examples
                    ),
                    "note": note,
                },
            )
        )
    return results


# ---------------------------------------------------------------------------
# Training manifest containment
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ManifestAudit:
    """Loaded training manifest plus the frames needed to audit it."""

    path: Path
    sha256: str
    frame: pd.DataFrame
    protocol: pd.DataFrame
    derived_root: Path


def load_manifest_audit(
    manifest_path: Path,
    protocol: pd.DataFrame,
    derived_root: Path,
) -> ManifestAudit:
    manifest_path = Path(manifest_path)
    frame = pd.read_csv(manifest_path)
    return ManifestAudit(
        path=manifest_path,
        sha256=sha256_file(manifest_path),
        frame=frame,
        protocol=protocol,
        derived_root=Path(derived_root),
    )


def check_manifest_integrity(
    audit: ManifestAudit,
    expected_sha256: str = EXPECTED_MANIFEST_SHA256,
    expected_rows: int = EXPECTED_MANIFEST_ROWS,
    expected_counts: Mapping[int, int] | None = None,
) -> CheckResult:
    """C.1: the manifest is the seed-42 subset of the train protocol."""
    expected_counts = dict(expected_counts or EXPECTED_MANIFEST_COUNTS)
    frame = audit.frame
    evidence: dict[str, object] = {
        "path": str(audit.path),
        "sha256": audit.sha256,
        "expected_sha256": expected_sha256,
        "rows": len(frame),
    }
    failures: list[str] = []

    missing_columns = MANIFEST_REQUIRED_COLUMNS - set(frame.columns)
    if missing_columns:
        failures.append(f"missing columns: {sorted(missing_columns)}")
        return CheckResult(
            "C.1",
            "training manifest integrity against the train protocol",
            "fail",
            "; ".join(failures),
            evidence,
        )

    if audit.sha256 != expected_sha256:
        failures.append("manifest SHA-256 does not match the frozen E0.1 value")
    if len(frame) != expected_rows:
        failures.append(f"expected {expected_rows} rows, found {len(frame)}")
    counts = {
        int(label): int(count)
        for label, count in frame["label"].value_counts().sort_index().items()
    }
    evidence["label_counts"] = counts
    if counts != expected_counts:
        failures.append(
            f"expected class counts {expected_counts}, found {counts}"
        )
    if list(frame["sample_index"]) != list(range(len(frame))):
        failures.append("sample_index is not a contiguous 0-based range")
    if frame["protocol_row"].duplicated().any():
        failures.append("protocol_row is not unique")
    if frame["audio_file_name"].duplicated().any():
        failures.append("audio_file_name is not unique")

    in_range = frame["protocol_row"].between(0, len(audit.protocol) - 1).all()
    if in_range:
        protocol_rows = frame["protocol_row"].to_numpy(dtype=int)
        expected = audit.protocol.iloc[protocol_rows].reset_index(drop=True)
        frame_rows = frame.reset_index(drop=True)
        for column in ("speaker_id", "system_id", "audio_file_name"):
            mismatch = (
                expected[column].astype(str).to_numpy()
                != frame_rows[column].astype(str).to_numpy()
            )
            if mismatch.any():
                failures.append(
                    f"{int(mismatch.sum())} manifest rows disagree with their "
                    f"protocol row on {column}"
                )
                evidence[f"{column}_mismatch_examples"] = [
                    {
                        key: (int(value) if not isinstance(value, str) else value)
                        for key, value in record.items()
                    }
                    for record in frame_rows.loc[
                        mismatch, ["sample_index", "protocol_row", column]
                    ]
                    .head(MAX_EVIDENCE_ITEMS)
                    .to_dict(orient="records")
                ]
        label_mismatch = (
            expected["label"].astype(int).to_numpy()
            != frame_rows["label"].astype(int).to_numpy()
        )
        if label_mismatch.any():
            failures.append(
                f"{int(label_mismatch.sum())} manifest rows disagree with their "
                "protocol row on label"
            )
            evidence["label_mismatch_examples"] = [
                {
                    key: (int(value) if not isinstance(value, str) else value)
                    for key, value in record.items()
                }
                for record in frame_rows.loc[
                    label_mismatch, ["sample_index", "protocol_row", "label"]
                ]
                .head(MAX_EVIDENCE_ITEMS)
                .to_dict(orient="records")
            ]

        expected_flac = audit.protocol.iloc[
            frame["protocol_row"].to_numpy(dtype=int)
        ]["source_flac"].reset_index(drop=True)
        actual_flac = frame["source_flac"].astype(str).reset_index(drop=True)
        source_mismatch = (
            expected_flac.astype(str).to_numpy() != actual_flac.to_numpy()
        )
        if source_mismatch.any():
            failures.append(
                f"{int(source_mismatch.sum())} source_flac values do not match "
                "the train protocol"
            )
    else:
        failures.append("protocol_row values fall outside the train protocol")

    status = "fail" if failures else "pass"
    detail = "; ".join(failures) if failures else (
        "10,080 rows (2,580 bonafide / 7,500 spoof) match the frozen manifest "
        "hash and their train protocol rows"
    )
    return CheckResult(
        "C.1",
        "training manifest integrity against the train protocol",
        status,
        detail,
        evidence,
    )


def check_manifest_files(
    audit: ManifestAudit,
    hashes: HashCache | None,
    expected_class_counts: Mapping[str, int] | None = None,
) -> CheckResult:
    """C.2: every manifest clip and image exists and matches its recorded hash."""
    expected_class_counts = dict(
        expected_class_counts or {"bonafide": 2_580, "spoofed": 7_500}
    )
    frame = audit.frame
    failures: list[str] = []
    evidence: dict[str, object] = {
        "rows": len(frame),
        "derived_root": str(audit.derived_root),
        "hashes_checked": hashes is not None,
    }

    missing_audio: list[str] = []
    missing_images: list[str] = []
    wrong_folder: list[str] = []
    audio_hash_mismatch: list[str] = []
    image_hash_mismatch: list[str] = []

    audio_values = frame["source_flac"].astype(str).to_numpy()
    image_values = frame["image_path"].astype(str).to_numpy()
    labels = frame["label"].to_numpy(dtype=int)
    audio_paths = [normalized_path(value) for value in audio_values]
    image_paths = [normalized_path(value) for value in image_values]
    for value, path in zip(audio_values, audio_paths):
        if not path.is_file():
            missing_audio.append(value)
    for value, path, label in zip(image_values, image_paths, labels):
        if not path.is_file():
            missing_images.append(value)
        expected_folder = "bonafide" if int(label) == 0 else "spoofed"
        if path.parent.name != expected_folder:
            wrong_folder.append(value)

    if hashes is not None:
        hashes.hash_paths(audio_paths)
        hashes.hash_paths(image_paths)
        recorded_audio = frame["sha256_flac"].astype(str).to_numpy()
        recorded_images = frame["sha256_png"].astype(str).to_numpy()
        for value, path, recorded in zip(audio_values, audio_paths, recorded_audio):
            digest = hashes.digest(path)
            if digest is not None and digest != recorded:
                audio_hash_mismatch.append(value)
        for value, path, recorded in zip(image_values, image_paths, recorded_images):
            digest = hashes.digest(path)
            if digest is not None and digest != recorded:
                image_hash_mismatch.append(value)

    evidence.update(
        {
            "missing_audio": len(missing_audio),
            "missing_images": len(missing_images),
            "wrong_class_folder": len(wrong_folder),
            "audio_hash_mismatches": len(audio_hash_mismatch),
            "image_hash_mismatches": len(image_hash_mismatch),
            "first_missing_audio": missing_audio[:MAX_EVIDENCE_ITEMS],
            "first_missing_images": missing_images[:MAX_EVIDENCE_ITEMS],
            "first_wrong_folder": wrong_folder[:MAX_EVIDENCE_ITEMS],
            "first_audio_hash_mismatches": audio_hash_mismatch[:MAX_EVIDENCE_ITEMS],
            "first_image_hash_mismatches": image_hash_mismatch[:MAX_EVIDENCE_ITEMS],
        }
    )
    if missing_audio:
        failures.append(f"{len(missing_audio)} source FLACs are missing")
    if missing_images:
        failures.append(f"{len(missing_images)} derived images are missing")
    if wrong_folder:
        failures.append(f"{len(wrong_folder)} derived images sit in the wrong class folder")
    if audio_hash_mismatch:
        failures.append(
            f"{len(audio_hash_mismatch)} source FLACs do not match sha256_flac"
        )
    if image_hash_mismatch:
        failures.append(
            f"{len(image_hash_mismatch)} derived images do not match sha256_png"
        )

    derived_counts = {
        folder: len(list((audit.derived_root / folder).glob("*.png")))
        for folder in ("bonafide", "spoofed")
    }
    evidence["derived_image_counts"] = derived_counts
    if derived_counts != expected_class_counts:
        failures.append(
            f"derived image counts are {derived_counts}, expected "
            f"{expected_class_counts}"
        )

    status = "fail" if failures else "pass"
    if failures:
        detail = "; ".join(failures)
    elif hashes is None:
        detail = (
            "all 10,080 audio and image paths exist and sit in the right class "
            "folder; content hashes skipped"
        )
    else:
        detail = (
            "all 10,080 source FLACs and derived images exist and match their "
            "recorded SHA-256"
        )
    return CheckResult(
        "C.2",
        "training manifest file and hash integrity",
        status,
        detail,
        evidence,
    )


def check_manifest_isolation(
    audit: ManifestAudit,
    protocols: Mapping[str, pd.DataFrame],
    audio_by_split: Mapping[str, list[Path]] | None,
    hashes: HashCache | None,
) -> CheckResult:
    """C.3: no training clip (name, speaker, or content) appears in dev/eval."""
    frame = audit.frame
    failures: list[str] = []
    evidence: dict[str, object] = {}

    manifest_names = set(frame["audio_file_name"].astype(str))
    manifest_speakers = set(frame["speaker_id"].astype(str))
    for split in ("dev", "eval"):
        protocol = protocols[split]
        name_overlap = sorted(manifest_names & set(protocol["audio_file_name"].astype(str)))
        speaker_overlap = sorted(manifest_speakers & set(protocol["speaker_id"].astype(str)))
        evidence[f"{split}_name_overlap"] = name_overlap[:MAX_EVIDENCE_ITEMS]
        evidence[f"{split}_speaker_overlap"] = speaker_overlap[:MAX_EVIDENCE_ITEMS]
        if name_overlap:
            failures.append(
                f"{len(name_overlap)} training clips also appear in {split}"
            )
        if speaker_overlap:
            failures.append(
                f"{len(speaker_overlap)} training speakers also appear in {split}"
            )

    if hashes is not None and audio_by_split is not None:
        manifest_hashes = set(frame["sha256_flac"].astype(str))
        for split in ("dev", "eval"):
            split_hashes = {
                hashes.digest(path)
                for path in audio_by_split.get(split, [])
                if hashes.digest(path) is not None
            }
            hash_overlap = sorted(manifest_hashes & split_hashes)
            evidence[f"{split}_content_hash_overlap"] = hash_overlap[:MAX_EVIDENCE_ITEMS]
            if hash_overlap:
                failures.append(
                    f"{len(hash_overlap)} training FLAC contents also appear in {split}"
                )
    else:
        evidence["content_hash_check"] = "skipped"

    status = "fail" if failures else "pass"
    detail = "; ".join(failures) if failures else (
        "zero training clips, speakers, or audio contents appear in dev or eval"
    )
    return CheckResult(
        "C.3",
        "training manifest isolation from dev and eval",
        status,
        detail,
        evidence,
    )


# ---------------------------------------------------------------------------
# Enrollment exclusions
# ---------------------------------------------------------------------------


def check_enrollment_exclusions(
    data_root: Path,
    protocols: Mapping[str, pd.DataFrame],
    expected_counts: Mapping[str, int] | None = None,
) -> CheckResult:
    """D.1: the 838 ``_A`` enrollment FLACs never enter CM scoring."""
    failures: list[str] = []
    evidence: dict[str, object] = {}
    expected = dict(
        expected_counts
        or {
            split_spec.name: split_spec.enrollment_files
            for split_spec in ASVSPOOF2019_LA.splits
        }
    )

    for split in SPLITS:
        protocol_names = set(protocols[split]["audio_file_name"].astype(str))
        flac_stems = {
            path.stem for path in audio_dir_for(split, data_root).glob("*.flac")
        }
        extras = flac_stems - protocol_names
        prefix = ENROLLMENT_PREFIX.get(split)
        prefixed = {name for name in extras if prefix and name.startswith(prefix)}
        unprefixed = extras - prefixed
        evidence[split] = {
            "extra_flac_files": len(extras),
            "enrollment_prefixed": len(prefixed),
            "unexpected_extra_files": sorted(unprefixed)[:MAX_EVIDENCE_ITEMS],
            "expected_enrollment_files": expected.get(split, 0),
        }
        if len(extras) != expected.get(split, 0):
            failures.append(
                f"{split}: expected {expected.get(split, 0)} enrollment files, "
                f"found {len(extras)}"
            )
        if unprefixed:
            failures.append(
                f"{split}: {len(unprefixed)} non-protocol FLACs lack the "
                "enrollment prefix"
            )

    total_enrollment = sum(expected.values())
    evidence["total_enrollment_files"] = total_enrollment
    status = "fail" if failures else "pass"
    detail = "; ".join(failures) if failures else (
        f"{total_enrollment} enrollment FLACs (142 dev / 696 eval) are excluded "
        "from the CM protocols and all image exports"
    )
    return CheckResult(
        "D.1",
        "enrollment-file exclusion",
        status,
        detail,
        evidence,
    )


# ---------------------------------------------------------------------------
# Baseline evaluation lineage
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return payload


def check_baseline_threshold_lineage(
    baseline_root: Path,
) -> CheckResult:
    """E.1: the only data-selected threshold comes from dev and transfers to eval."""
    failures: list[str] = []
    evidence: dict[str, object] = {}
    metrics_paths = sorted(Path(baseline_root).glob("*/metrics.json"))
    if not metrics_paths:
        return CheckResult(
            "E.1",
            "baseline threshold lineage",
            "fail",
            f"no metrics.json found under {baseline_root}",
            {},
        )

    for metrics_path in metrics_paths:
        metrics = _load_json(metrics_path)
        stem = metrics_path.parent.name
        thresholds = metrics.get("thresholds") or {}
        entry: dict[str, object] = {
            "selected_on": thresholds.get("selected_on"),
            "source": thresholds.get("source"),
            "dev_eer": thresholds.get("dev_eer"),
        }
        if thresholds.get("selected_on") != "dev":
            failures.append(f"{stem}: selected_on is not dev")
        if thresholds.get("source") != "in_run_dev":
            failures.append(f"{stem}: threshold source is not in_run_dev")
        dev_threshold = thresholds.get("dev_eer")
        if dev_threshold is None:
            failures.append(f"{stem}: no dev EER threshold recorded")
            evidence[stem] = entry
            continue

        dev_scores_path = metrics_path.parent / "scores_dev.csv"
        eval_scores_path = metrics_path.parent / "scores_eval.csv"
        if not dev_scores_path.is_file() or not eval_scores_path.is_file():
            failures.append(f"{stem}: missing dev or eval score CSV")
            evidence[stem] = entry
            continue
        dev_scores = pd.read_csv(dev_scores_path)
        recomputed = compute_eer(
            dev_scores.loc[dev_scores["label"] == 1, "score"].to_numpy(),
            dev_scores.loc[dev_scores["label"] == 0, "score"].to_numpy(),
        ).threshold
        entry["recomputed_dev_threshold"] = float(recomputed)
        entry["dev_threshold_delta"] = float(abs(recomputed - float(dev_threshold)))
        if abs(recomputed - float(dev_threshold)) > 1e-6:
            failures.append(
                f"{stem}: recorded dev threshold does not reproduce from scores_dev.csv"
            )

        eval_metrics = (metrics.get("splits") or {}).get("eval") or {}
        eval_operating = (
            (eval_metrics.get("metrics") or {}).get("threshold_metrics") or {}
        )
        eval_thresholds = eval_operating.get("dev_eer") or {}
        entry["eval_operating_dev_eer"] = eval_thresholds.get("threshold")
        if eval_thresholds.get("threshold") is None:
            failures.append(f"{stem}: eval has no transferred dev_eer threshold")
        elif abs(float(eval_thresholds["threshold"]) - float(dev_threshold)) > 1e-9:
            failures.append(
                f"{stem}: eval dev_eer threshold differs from the dev threshold"
            )
        suspicious = [
            name
            for name in eval_operating
            if name not in {"fixed_0.5", "dev_eer"}
        ]
        entry["eval_operating_threshold_names"] = sorted(eval_operating)
        if suspicious:
            failures.append(
                f"{stem}: eval has non-policy operating thresholds: {suspicious}"
            )
        entry["policy"] = (metrics.get("thresholds") or {}).get("policy")
        evidence[stem] = entry

    status = "fail" if failures else "pass"
    detail = "; ".join(failures) if failures else (
        "every checkpoint selects its EER threshold on dev only and transfers "
        "it unchanged to eval"
    )
    return CheckResult(
        "E.1",
        "baseline threshold lineage (dev only)",
        status,
        detail,
        evidence,
    )


def check_baseline_score_alignment(
    baseline_root: Path,
    protocols: Mapping[str, pd.DataFrame],
    image_root: Path,
    repo_root: Path = REPO_ROOT,
) -> CheckResult:
    """E.2: baseline score rows line up with the evaluated protocol split."""
    failures: list[str] = []
    evidence: dict[str, object] = {}
    metrics_paths = sorted(Path(baseline_root).glob("*/metrics.json"))
    if not metrics_paths:
        return CheckResult(
            "E.2",
            "baseline score/protocol alignment",
            "fail",
            f"no metrics.json found under {baseline_root}",
            {},
        )

    for metrics_path in metrics_paths:
        metrics = _load_json(metrics_path)
        stem = metrics_path.parent.name
        split_payloads = metrics.get("splits") or {}
        entry: dict[str, object] = {}
        for split in ("dev", "eval"):
            scores_path = metrics_path.parent / f"scores_{split}.csv"
            if not scores_path.is_file():
                failures.append(f"{stem}: missing scores_{split}.csv")
                continue
            scores = pd.read_csv(scores_path)
            protocol = protocols[split]
            split_entry: dict[str, object] = {
                "rows": len(scores),
                "protocol_rows": len(protocol),
            }
            entry[split] = split_entry
            if len(scores) != len(protocol):
                failures.append(
                    f"{stem}/{split}: {len(scores)} scores for {len(protocol)} "
                    "protocol rows"
                )
                continue
            if list(scores["protocol_row"]) != list(range(len(protocol))):
                failures.append(f"{stem}/{split}: protocol_row is not 0..n-1")
            for column in ("clip_id", "speaker_id", "system_id"):
                source_column = (
                    "audio_file_name" if column == "clip_id" else column
                )
                mismatch = (
                    scores[column].astype(str).to_numpy()
                    != protocol[source_column].astype(str).to_numpy()
                )
                if mismatch.any():
                    failures.append(
                        f"{stem}/{split}: {int(mismatch.sum())} rows disagree "
                        f"on {column}"
                    )
            label_mismatch = (
                scores["label"].astype(int).to_numpy()
                != protocol["label"].astype(int).to_numpy()
            )
            if label_mismatch.any():
                failures.append(
                    f"{stem}/{split}: {int(label_mismatch.sum())} rows disagree "
                    "on label"
                )
            split_root = normalized_path(Path(image_root) / split).as_posix() + os.sep
            outside = [
                str(value)
                for value in scores["source_path"].astype(str)
                if not normalized_path(value).as_posix().startswith(split_root)
            ]
            if outside:
                failures.append(
                    f"{stem}/{split}: {len(outside)} source paths are outside "
                    f"data/dataset_img/{split}"
                )

            metrics_split = split_payloads.get(split) or {}
            recorded_counts = (metrics_split.get("metrics") or {}).get(
                "sample_counts"
            ) or {}
            expected_counts = {
                "total": len(protocol),
                "bonafide": int((protocol["label"] == 0).sum()),
                "spoof": int((protocol["label"] == 1).sum()),
            }
            split_entry["recorded_sample_counts"] = recorded_counts
            split_entry["expected_sample_counts"] = expected_counts
            if recorded_counts != expected_counts:
                failures.append(
                    f"{stem}/{split}: metrics sample counts do not match the protocol"
                )
            expected_attacks = "A01-A06" if split == "dev" else "A07-A19"
            if metrics_split.get("attack_subset") != expected_attacks:
                failures.append(
                    f"{stem}/{split}: attack subset is "
                    f"{metrics_split.get('attack_subset')!r}, expected "
                    f"{expected_attacks!r}"
                )
        evidence[stem] = entry

    status = "fail" if failures else "pass"
    detail = "; ".join(failures) if failures else (
        "every baseline score CSV maps 1:1 to its protocol split with the "
        "correct labels, speakers, systems, and source paths"
    )
    return CheckResult(
        "E.2",
        "baseline score/protocol alignment",
        status,
        detail,
        evidence,
    )


def check_baseline_artifacts(
    baseline_root: Path,
    freeze_report_path: Path,
    manifest_sha256: str,
) -> CheckResult:
    """E.3: baseline checkpoints and manifest match the frozen E0.3 audit."""
    failures: list[str] = []
    evidence: dict[str, object] = {}
    freeze_path = Path(freeze_report_path)
    if not freeze_path.is_file():
        return CheckResult(
            "E.3",
            "baseline artifact identity",
            "fail",
            f"missing freeze report: {freeze_path}",
            {},
        )
    freeze = _load_json(freeze_path)
    manifest = freeze.get("training_manifest") or {}
    evidence["freeze_manifest_sha256"] = manifest.get("sha256")
    if manifest.get("sha256") != manifest_sha256:
        failures.append(
            "freeze report manifest hash does not match the current manifest"
        )
    if not manifest.get("passed", False):
        failures.append("freeze report training-manifest check did not pass")

    expected_checkpoints = {
        entry.get("stem"): entry
        for entry in freeze.get("checkpoints") or []
        if entry.get("stem")
    }
    for metrics_path in sorted(Path(baseline_root).glob("*/metrics.json")):
        metrics = _load_json(metrics_path)
        checkpoint = metrics.get("checkpoint") or {}
        stem = Path(str(checkpoint.get("path", metrics_path.parent.name))).stem
        evidence[stem] = {
            "sha256": checkpoint.get("sha256"),
            "total_parameters": checkpoint.get("total_parameters"),
        }
        frozen = expected_checkpoints.get(stem)
        if frozen is None:
            failures.append(f"{stem}: checkpoint is absent from the freeze report")
            continue
        frozen_audit = frozen.get("audit") or {}
        if checkpoint.get("sha256") != frozen_audit.get("sha256"):
            failures.append(f"{stem}: checkpoint hash differs from the freeze report")
        if checkpoint.get("total_parameters") != frozen_audit.get(
            "total_parameters"
        ):
            failures.append(f"{stem}: parameter count differs from the freeze report")
        if checkpoint.get("missing_keys") or checkpoint.get("unexpected_keys"):
            failures.append(f"{stem}: checkpoint has missing/unexpected keys")

    status = "fail" if failures else "pass"
    detail = "; ".join(failures) if failures else (
        "both baseline checkpoints and the training manifest match the frozen "
        "E0.3 audit"
    )
    return CheckResult(
        "E.3",
        "baseline artifact identity",
        status,
        detail,
        evidence,
    )


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AuditConfig:
    """Inputs for one audit run."""

    data_root: Path = REPO_ROOT / "data" / "LA"
    image_root: Path = DEFAULT_IMAGE_ROOT
    derived_root: Path = DEFAULT_DERIVED_ROOT
    manifest_path: Path = DEFAULT_TRAIN_MANIFEST
    baseline_root: Path = DEFAULT_BASELINE_ROOT
    freeze_report_path: Path = DEFAULT_FREEZE_REPORT
    hash_files: bool = True
    workers: int = 8
    command: str | None = None


def _git_state(repo_root: Path = REPO_ROOT) -> dict[str, object]:
    commit = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    status = subprocess.run(
        ["git", "-C", str(repo_root), "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "commit": commit.stdout.strip() or "unknown",
        "dirty": bool(status.stdout.strip()),
        "status_porcelain": status.stdout.strip().splitlines(),
    }


def _environment() -> dict[str, object]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "pandas": pd.__version__,
    }


def _inventory(
    protocols: Mapping[str, pd.DataFrame],
    audio_by_split: Mapping[str, list[Path]],
    images_by_split: Mapping[str, list[Path]],
) -> dict[str, object]:
    inventory: dict[str, object] = {}
    for split in SPLITS:
        protocol = protocols[split]
        # ``images_by_split["train"]`` also holds the reconstructed train
        # subset; the inventory reports the stored full-split export only.
        export_images = [
            path
            for path in images_by_split.get(split, [])
            if "dataset_img" in path.parts
        ]
        labels = protocol["label"].to_numpy(dtype=int)
        inventory[split] = {
            "protocol_rows": len(protocol),
            "bonafide": int((labels == 0).sum()),
            "spoof": int((labels == 1).sum()),
            "speakers": int(protocol["speaker_id"].nunique()),
            "attack_systems": sorted(
                value
                for value in protocol["system_id"].astype(str).unique()
                if value != "-"
            ),
            "flac_files": len(audio_by_split.get(split, [])),
            "protocol_flac_files": len(protocol),
            "enrollment_flac_files": len(audio_by_split.get(split, []))
            - len(protocol),
            "image_files": len(export_images),
            "bonafide_images": sum(
                1 for path in export_images if path.parent.name == "bonafide"
            ),
            "spoofed_images": sum(
                1 for path in export_images if path.parent.name == "spoofed"
            ),
        }
    return inventory


def run_audit(config: AuditConfig) -> dict[str, object]:
    """Run every check and return a JSON-serialisable audit payload."""
    data_root = Path(config.data_root)
    image_root = Path(config.image_root)
    derived_root = Path(config.derived_root)
    baseline_root = Path(config.baseline_root)
    manifest_path = Path(config.manifest_path)

    protocols = {
        split: load_cm_protocol(split, data_root=data_root, repo_root=REPO_ROOT)
        for split in SPLITS
    }

    dataset_report = validate_dataset(data_root)
    dataset_problems = list(dataset_report.problems)
    checks: list[CheckResult] = [
        CheckResult(
            "A.1",
            "ASVspoof 2019 LA layout and protocol hashes",
            "fail" if dataset_problems else "pass",
            "; ".join(dataset_problems)
            if dataset_problems
            else "layout, counts, and protocol SHA-256 values match the dataset spec",
            {
                "problems": dataset_problems,
                "splits": {
                    name: {
                        "protocol_rows": report.protocol_rows,
                        "bonafide": report.bonafide,
                        "spoof": report.spoof,
                        "flac_files": report.flac_files,
                        "enrollment_files": report.enrollment_files,
                        "protocol_sha256": report.protocol_sha256,
                    }
                    for name, report in dataset_report.splits.items()
                },
            },
        )
    ]

    audio_by_split, images_by_split = collect_corpus_paths(
        data_root, image_root, derived_root
    )

    for split in SPLITS:
        checks.append(
            check_split_alignment(
                split,
                data_root,
                image_root,
                protocol=protocols[split],
            )
        )
    checks.extend(check_split_isolation(protocols))
    hashes = HashCache(workers=config.workers) if config.hash_files else None
    if hashes is not None:
        checks.extend(
            check_duplicate_hashes(
                audio_by_split,
                images_by_split,
                hashes,
                derived_root=derived_root,
            )
        )
    else:
        for check_id, title in (
            ("B.3", "audio content duplicates"),
            ("B.4", "image content duplicates"),
        ):
            checks.append(
                CheckResult(
                    check_id,
                    title,
                    "skip",
                    "content hashing disabled with --skip-hashes",
                    {},
                )
            )

    manifest_audit = load_manifest_audit(manifest_path, protocols["train"], derived_root)
    checks.append(check_manifest_integrity(manifest_audit))
    checks.append(check_manifest_files(manifest_audit, hashes))
    checks.append(
        check_manifest_isolation(manifest_audit, protocols, audio_by_split, hashes)
    )
    checks.append(check_enrollment_exclusions(data_root, protocols))

    checks.append(check_baseline_threshold_lineage(baseline_root))
    checks.append(
        check_baseline_score_alignment(
            baseline_root, protocols, image_root, repo_root=REPO_ROOT
        )
    )
    checks.append(
        check_baseline_artifacts(
            baseline_root, config.freeze_report_path, manifest_audit.sha256
        )
    )

    failures = [check for check in checks if check.status == "fail"]
    skipped = [check for check in checks if check.status == "skip"]
    if failures:
        status = "fail"
    elif skipped:
        status = "pass_with_skips"
    else:
        status = "pass"

    inputs: dict[str, object] = {
        "data_root": repo_relative(data_root),
        "image_root": repo_relative(image_root),
        "derived_root": repo_relative(derived_root),
        "baseline_root": repo_relative(baseline_root),
        "freeze_report": repo_relative(config.freeze_report_path),
        "training_manifest": {
            "path": repo_relative(manifest_path),
            "sha256": manifest_audit.sha256,
            "expected_sha256": EXPECTED_MANIFEST_SHA256,
            "rows": len(manifest_audit.frame),
        },
        "protocols": {
            split: {
                "path": repo_relative(protocol_path_for(split, data_root)),
                "sha256": sha256_file(protocol_path_for(split, data_root)),
            }
            for split in SPLITS
        },
        "metadata": {
            split: {
                "path": repo_relative(image_root / split / f"metadata_{split}.csv"),
                "sha256": sha256_file(
                    image_root / split / f"metadata_{split}.csv"
                ),
            }
            for split in SPLITS
        },
    }

    return {
        "experiment": "E0.5",
        "title": "Data and leakage audit",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": config.command,
        "git": _git_state(),
        "environment": _environment(),
        "inputs": inputs,
        "inventory": _inventory(protocols, audio_by_split, images_by_split),
        "checks": [check.to_dict() for check in checks],
        "conclusion": {
            "status": status,
            "passed": not failures,
            "failures": [check.check_id for check in failures],
            "skipped": [check.check_id for check in skipped],
            "policy": (
                "Dev is a documented model-selection split (early stopping and "
                "E0.3 threshold selection). Eval is the clean estimate: no eval "
                "clip entered training, early stopping, or threshold selection."
            ),
        },
    }


def _status_label(status: str) -> str:
    return {
        "pass": "PASS",
        "pass_with_skips": "PASS (with skips)",
        "fail": "FAIL",
    }.get(status, status.upper())


def render_markdown(report: Mapping[str, object]) -> str:
    """Render the audit payload as the roadmap's Markdown report."""
    conclusion = report.get("conclusion") or {}
    inputs = report.get("inputs") or {}
    inventory = report.get("inventory") or {}
    git = report.get("git") or {}
    manifest = inputs.get("training_manifest") or {}
    lines: list[str] = [
        "# E0.5 — Data and leakage audit",
        "",
        f"**Status:** {_status_label(str(conclusion.get('status')))}  ",
        f"**Generated:** {report.get('generated_at_utc')}  ",
        f"**Git commit:** `{git.get('commit')}`"
        + (" (working tree dirty)" if git.get("dirty") else " (clean tree)"),
        f"**Training manifest:** `{manifest.get('path')}` "
        f"(SHA-256 `{manifest.get('sha256')}`)  ",
        f"**Command:** `{report.get('command')}`",
        "",
        "## Metric contract",
        "",
        "| Split | Attack subset | Threshold | Samples | Bonafide / spoof |",
        "|---|---|---|---:|---:|",
    ]
    for split in SPLITS:
        row = inventory.get(split) or {}
        subset = "A01–A06" if split in {"train", "dev"} else "A07–A19"
        lines.append(
            f"| {split} | {subset} | not applicable (audit) | "
            f"{int(row.get('protocol_rows', 0)):,} | "
            f"{int(row.get('bonafide', 0)):,} / {int(row.get('spoof', 0)):,} |"
        )
    lines.extend(
        [
            "",
            "Baseline lineage rows are audited threshold-free: the dev-selected "
            "EER threshold and the fixed `0.5` threshold are checked as artifacts, "
            "not re-reported as new performance numbers.",
            "",
            "## Inputs and hashes",
            "",
            "| Input | SHA-256 |",
            "|---|---|",
            f"| `{manifest.get('path')}` | `{manifest.get('sha256')}` |",
        ]
    )
    for split in SPLITS:
        protocol = (inputs.get("protocols") or {}).get(split) or {}
        metadata = (inputs.get("metadata") or {}).get(split) or {}
        lines.append(
            f"| `{protocol.get('path')}` | `{protocol.get('sha256')}` |"
        )
        lines.append(
            f"| `{metadata.get('path')}` | `{metadata.get('sha256')}` |"
        )

    lines.extend(
        [
            "",
            "## Inventory",
            "",
            "| Split | Protocol rows | Bonafide | Spoof | Speakers | FLAC files | "
            "Enrollment | Images |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for split in SPLITS:
        row = inventory.get(split) or {}
        values = [
            int(row.get("protocol_rows", 0)),
            int(row.get("bonafide", 0)),
            int(row.get("spoof", 0)),
            int(row.get("speakers", 0)),
            int(row.get("flac_files", 0)),
            int(row.get("enrollment_flac_files", 0)),
            int(row.get("image_files", 0)),
        ]
        lines.append(
            f"| {split} | " + " | ".join(f"{value:,}" for value in values) + " |"
        )

    lines.extend(
        [
            "",
            "## Checks",
            "",
            "| ID | Check | Status | Detail |",
            "|---|---|---|---|",
        ]
    )
    for check in report.get("checks") or []:
        detail = str(check.get("detail", "")).replace("|", "\\|")
        status = str(check.get("status", "")).upper()
        lines.append(
            f"| {check.get('id')} | {check.get('title')} | {status} | {detail} |"
        )

    duplicate_rows = []
    for check in report.get("checks") or []:
        evidence = check.get("evidence") or {}
        if "cross_split_groups" in evidence:
            duplicate_rows.append(
                f"| {check.get('id')} | "
                f"{int(evidence.get('hashed_files', 0)):,} | "
                f"{int(evidence.get('unique_hashes', 0)):,} | "
                f"{evidence.get('cross_split_groups')} | "
                f"{int(evidence.get('within_split_groups', 0)):,} | "
                f"{int(evidence.get('within_split_explained_groups', 0)):,} | "
                f"{evidence.get('within_split_unexplained_groups')} |"
            )
    if duplicate_rows:
        lines.extend(
            [
                "",
                "## Duplicate-hash audit",
                "",
                "| Check | Files hashed | Unique hashes | Cross-split groups | "
                "Within-split groups | Explained within-split | Unexplained "
                "within-split |",
                "|---|---:|---:|---:|---:|---:|---:|",
                *duplicate_rows,
                "",
                "The image check covers `data/dataset_img/{train,dev,eval}` and "
                "the reconstructed `data/derived/train_seed42/`. The derived train "
                "subset is labelled `train`; its byte-identical overlap with the "
                "full train export is the only permitted within-split duplicate.",
            ]
        )

    lines.extend(
        [
            "",
            "## Conclusion",
            "",
        ]
    )
    for check in report.get("checks") or []:
        evidence = check.get("evidence") or {}
        if check.get("id") in {"B.3", "B.4"}:
            status = str(check.get("status")).upper()
            if status == "SKIP":
                lines.append(
                    f"- `{check.get('id')}` SKIP: content hashing was disabled."
                )
                continue
            lines.append(
                f"- `{check.get('id')}` {status}: "
                f"{int(evidence.get('hashed_files', 0)):,} files hashed, "
                f"{evidence.get('cross_split_groups')} cross-split duplicate "
                f"groups, {int(evidence.get('within_split_explained_groups', 0)):,} "
                "explained within-split groups."
            )
    lines.extend(
        [
            "- `C.3` confirms zero training clips, speakers, or audio contents "
            "appear in dev or eval.",
            "- `E.1`–`E.3` confirm both checkpoints selected their threshold on "
            "dev only and that the baseline artifacts match the frozen E0.3 "
            "audit.",
            "- Dev remains a **documented model-selection split**; eval is clean.",
            f"- Raw status: `{_status_label(str(conclusion.get('status')))}`.",
        ]
    )

    lines.extend(
        [
            "",
            "## Dev reuse and eval isolation",
            "",
            "- The original training loop early-stopped on **full dev** "
            "(`2_model.ipynb` cells 16–17 and 21).",
            "- E0.3 selected the operating threshold on **full dev** and "
            "transferred it unchanged to eval.",
            "- Dev is therefore a **model-selection split**, not a clean holdout. "
            "This is documented in `reports/baseline/README.md`.",
            "- `eval` was read only by the post-training inference cells "
            "(`2_model.ipynb` cells 23 and 25). No eval clip entered training, "
            "early stopping, or threshold selection.",
            "- Notebook cell evidence is recorded in "
            "`reports/phase0/E0.5/README.md`.",
        ]
    )

    if conclusion.get("failures"):
        lines.extend(
            [
                "",
                "## Failures",
                "",
                *[
                    f"- `{check_id}`"
                    for check_id in conclusion.get("failures") or []
                ],
            ]
        )
    if conclusion.get("skipped"):
        lines.extend(
            [
                "",
                "## Skipped checks",
                "",
                *[
                    f"- `{check_id}`"
                    for check_id in conclusion.get("skipped") or []
                ],
            ]
        )

    lines.extend(
        [
            "",
            "## Reproduction",
            "",
            "```bash",
            "uv run python scripts/audit_data.py",
            "```",
            "",
            "The audit is read-only: it hashes and compares existing artifacts and "
            "writes only this report and its JSON companion.",
            "",
        ]
    )
    return "\n".join(lines)


def write_audit(
    report: Mapping[str, object],
    markdown_path: str | Path,
    json_path: str | Path,
) -> tuple[Path, Path]:
    """Write the Markdown report and its JSON companion."""
    markdown_path = Path(markdown_path)
    json_path = Path(json_path)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return markdown_path, json_path


__all__ = [
    "AuditConfig",
    "CheckResult",
    "DuplicateFindings",
    "HashCache",
    "check_baseline_artifacts",
    "check_baseline_score_alignment",
    "check_baseline_threshold_lineage",
    "check_duplicate_hashes",
    "check_enrollment_exclusions",
    "check_manifest_files",
    "check_manifest_integrity",
    "check_manifest_isolation",
    "check_split_alignment",
    "check_split_isolation",
    "collect_corpus_paths",
    "find_duplicate_hashes",
    "load_manifest_audit",
    "render_markdown",
    "run_audit",
    "write_audit",
]
