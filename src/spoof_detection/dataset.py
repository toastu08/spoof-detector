"""ASVspoof 2019 LA dataset layout, validation, and provenance.

The dataset is downloaded manually (see ``README.md``); this module does not
download or extract anything.  It verifies that an existing ``data/LA`` tree,
or the archive it came from, matches the layout and counts used by the rest of
the project.

The expected tree is::

    data/LA/
    ├── README.LA.txt
    ├── ASVspoof2019_LA_train/{LICENSE.txt,flac/}
    ├── ASVspoof2019_LA_dev/{LICENSE.txt,flac/}
    ├── ASVspoof2019_LA_eval/{LICENSE.txt,flac/}
    ├── ASVspoof2019_LA_cm_protocols/
    ├── ASVspoof2019_LA_asv_protocols/
    └── ASVspoof2019_LA_asv_scores/

The dev and eval FLAC directories contain enrollment files (``LA_D_A*.flac``
and ``LA_E_A*.flac``) that are not part of the countermeasure protocols.  They
are expected, counted separately, and must not be mixed into CM evaluation.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "LA"
DEFAULT_MANIFEST_PATH = REPO_ROOT / "manifests" / "asvspoof2019_la.meta.json"

OFFICIAL_DOI = "10.7488/ds/2555"
OFFICIAL_LICENSE = "ODC Attribution License (ODC-By 1.0)"
OFFICIAL_ARCHIVE_SIZE = 7_640_952_520
OFFICIAL_ARCHIVE_MD5 = "30c98f11d8b2bc21f2c257bfd78bb5c5"
OFFICIAL_ARCHIVE_URL = (
    "https://datashare.ed.ac.uk/server/api/core/bitstreams/"
    "a9f87c35-f055-4015-80e2-2fdff0d46269/content"
)
ZENODO_ARCHIVE_URL = (
    "https://zenodo.org/records/6906306/files/LA.zip?download=1"
)

# The Kaggle copy is a third-party re-upload.  Its zip is a re-packaging of the
# extracted tree, so its MD5 differs from the official LA.zip.  The extracted
# contents are still validated against the same counts and protocol hashes.
KNOWN_UNOFFICIAL_ARCHIVES = {
    "793d7b00e00ec0466acfd85800b714fd": (
        "Kaggle: anishsarkar22/asvpoof-2019-dataset-la "
        "(third-party re-upload; extracted content is validated)"
    ),
}

OS_METADATA_NAMES = frozenset({".DS_Store", "Thumbs.db", "desktop.ini"})
OS_METADATA_DIRS = frozenset({"__MACOSX"})


@dataclass(frozen=True)
class SplitSpec:
    """Expected contents of one ASVspoof 2019 LA split."""

    name: str
    dirname: str
    protocol_name: str
    protocol_rows: int
    bonafide: int
    spoof: int
    flac_files: int
    enrollment_files: int
    protocol_sha256: str
    enrollment_prefix: str | None = None
    flac_dirname: str = "flac"


@dataclass(frozen=True)
class DatasetSpec:
    """Expected ASVspoof 2019 LA dataset layout and checksums."""

    name: str
    splits: tuple[SplitSpec, ...]
    cm_protocols_dirname: str
    asv_protocols_dirname: str
    asv_scores_dirname: str
    asv_protocols_count: int
    asv_scores_count: int
    readme_name: str
    readme_sha256: str
    license_name: str
    license_sha256: str
    total_files: int | None
    total_bytes: int | None
    archive_size: int
    archive_md5: str


ASVSPOOF2019_LA = DatasetSpec(
    name="ASVspoof 2019 Logical Access (LA)",
    splits=(
        SplitSpec(
            name="train",
            dirname="ASVspoof2019_LA_train",
            protocol_name="ASVspoof2019.LA.cm.train.trn.txt",
            protocol_rows=25_380,
            bonafide=2_580,
            spoof=22_800,
            flac_files=25_380,
            enrollment_files=0,
            protocol_sha256=(
                "65a289e4009400b42bb4f1e47ac60938d93f7f595452d1f220d35defc3de8ab8"
            ),
        ),
        SplitSpec(
            name="dev",
            dirname="ASVspoof2019_LA_dev",
            protocol_name="ASVspoof2019.LA.cm.dev.trl.txt",
            protocol_rows=24_844,
            bonafide=2_548,
            spoof=22_296,
            flac_files=24_986,
            enrollment_files=142,
            protocol_sha256=(
                "2fedaebf1ba3ae1a9bb7d3ec916804ac237b4eee063a27fa92e6549f03d129c7"
            ),
            enrollment_prefix="LA_D_A",
        ),
        SplitSpec(
            name="eval",
            dirname="ASVspoof2019_LA_eval",
            protocol_name="ASVspoof2019.LA.cm.eval.trl.txt",
            protocol_rows=71_237,
            bonafide=7_355,
            spoof=63_882,
            flac_files=71_933,
            enrollment_files=696,
            protocol_sha256=(
                "84fce576efff092269a0a6ad247d2e9f2f7b4d4e25a4545efb2518da92708036"
            ),
            enrollment_prefix="LA_E_A",
        ),
    ),
    cm_protocols_dirname="ASVspoof2019_LA_cm_protocols",
    asv_protocols_dirname="ASVspoof2019_LA_asv_protocols",
    asv_scores_dirname="ASVspoof2019_LA_asv_scores",
    asv_protocols_count=10,
    asv_scores_count=2,
    readme_name="README.LA.txt",
    readme_sha256="c99a875d94d91734c2468eff13a1d3a60991eb308335caa1750be1020c2748fe",
    license_name="LICENSE.txt",
    license_sha256="99a4d3e0db6f9a5cd8871f9737b207af01de333f57c6269a11ce755227c1555d",
    total_files=122_318,
    total_bytes=7_655_650_274,
    archive_size=OFFICIAL_ARCHIVE_SIZE,
    archive_md5=OFFICIAL_ARCHIVE_MD5,
)


@dataclass(frozen=True)
class ProtocolSummary:
    """Parsed countermeasure protocol."""

    rows: int
    bonafide: int
    spoof: int
    names: frozenset[str]
    systems: Counter[str]


@dataclass
class SplitReport:
    """Validation result for one split."""

    name: str
    protocol_rows: int
    bonafide: int
    spoof: int
    flac_files: int
    enrollment_files: int
    protocol_sha256: str
    missing_protocol_files: int
    unexpected_files: int
    problems: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.problems


@dataclass
class DatasetReport:
    """Validation result for an extracted ``data/LA`` tree."""

    root: Path
    total_files: int
    total_bytes: int
    readme_sha256: str
    license_sha256: str
    splits: dict[str, SplitReport]
    problems: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.problems


@dataclass
class ArchiveReport:
    """Validation result for an ``LA.zip``-style archive."""

    path: Path
    size_bytes: int
    md5: str
    sha256: str
    is_official: bool
    source: str
    total_files: int
    total_bytes: int
    readme_sha256: str
    license_sha256: str
    splits: dict[str, SplitReport]
    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.problems


def hash_file(
    path: str | Path,
    algorithms: tuple[str, ...] = ("sha256",),
    chunk_size: int = 1 << 20,
) -> dict[str, str]:
    """Return the requested hashes of ``path`` in one pass."""
    hashers = {name: hashlib.new(name) for name in algorithms}
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            for hasher in hashers.values():
                hasher.update(chunk)
    return {name: hasher.hexdigest() for name, hasher in hashers.items()}


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 hex digest of ``path``."""
    return hash_file(path, ("sha256",))["sha256"]


def repo_relative(path: str | Path, repo_root: str | Path = REPO_ROOT) -> str:
    """Return ``path`` relative to the repository root when possible."""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(Path(repo_root).resolve()).as_posix()
    except ValueError:
        return str(resolved)


def parse_cm_protocol(text: str, source: str = "protocol") -> ProtocolSummary:
    """Parse an ASVspoof 2019 LA countermeasure protocol.

    The format is ``SPEAKER_ID AUDIO_FILE_NAME - SYSTEM_ID KEY``.  Row order is
    preserved by callers; this function only validates and summarises.
    """
    names: list[str] = []
    labels: Counter[str] = Counter()
    systems: Counter[str] = Counter()

    for line_number, raw_line in enumerate(text.splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(
                f"{source}:{line_number}: expected 5 columns, found {len(parts)}"
            )
        _speaker, audio_file, ignored, system_id, label = parts
        if ignored != "-":
            raise ValueError(
                f"{source}:{line_number}: expected '-' in column 3, found {ignored!r}"
            )
        if label not in {"bonafide", "spoof"}:
            raise ValueError(
                f"{source}:{line_number}: unknown label {label!r}"
            )
        names.append(audio_file)
        labels[label] += 1
        systems[system_id] += 1

    if len(names) != len(set(names)):
        raise ValueError(f"{source}: protocol contains duplicate audio file names")

    return ProtocolSummary(
        rows=len(names),
        bonafide=labels["bonafide"],
        spoof=labels["spoof"],
        names=frozenset(names),
        systems=systems,
    )


def _iter_dataset_files(root: Path) -> Iterable[Path]:
    """Yield dataset files, ignoring common OS metadata."""
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.name in OS_METADATA_NAMES:
            continue
        if any(part in OS_METADATA_DIRS for part in path.parts):
            continue
        yield path


def _summarize_split(
    split: SplitSpec,
    protocol: ProtocolSummary,
    flac_names: set[str],
    protocol_sha256: str,
    source: str,
) -> SplitReport:
    """Compare one split against its spec and return a report."""
    problems: list[str] = []
    missing = protocol.names - flac_names
    unexpected = flac_names - protocol.names
    if split.enrollment_prefix:
        enrollment = {
            name for name in unexpected if name.startswith(split.enrollment_prefix)
        }
    else:
        enrollment = set()
    bad_unexpected = unexpected - enrollment

    if protocol.rows != split.protocol_rows:
        problems.append(
            f"{source}: expected {split.protocol_rows} protocol rows, "
            f"found {protocol.rows}"
        )
    if protocol.bonafide != split.bonafide:
        problems.append(
            f"{source}: expected {split.bonafide} bonafide rows, "
            f"found {protocol.bonafide}"
        )
    if protocol.spoof != split.spoof:
        problems.append(
            f"{source}: expected {split.spoof} spoof rows, found {protocol.spoof}"
        )
    if len(flac_names) != split.flac_files:
        problems.append(
            f"{source}: expected {split.flac_files} FLAC files, "
            f"found {len(flac_names)}"
        )
    if len(enrollment) != split.enrollment_files:
        problems.append(
            f"{source}: expected {split.enrollment_files} enrollment FLAC files, "
            f"found {len(enrollment)}"
        )
    if missing:
        problems.append(
            f"{source}: {len(missing)} protocol FLAC files are missing, "
            f"e.g. {sorted(missing)[:3]}"
        )
    if bad_unexpected:
        problems.append(
            f"{source}: {len(bad_unexpected)} unexpected FLAC files are not in "
            f"the protocol, e.g. {sorted(bad_unexpected)[:3]}"
        )
    if protocol_sha256 != split.protocol_sha256:
        problems.append(
            f"{source}: protocol SHA-256 mismatch: expected "
            f"{split.protocol_sha256}, found {protocol_sha256}"
        )

    return SplitReport(
        name=split.name,
        protocol_rows=protocol.rows,
        bonafide=protocol.bonafide,
        spoof=protocol.spoof,
        flac_files=len(flac_names),
        enrollment_files=len(enrollment),
        protocol_sha256=protocol_sha256,
        missing_protocol_files=len(missing),
        unexpected_files=len(bad_unexpected),
        problems=problems,
    )


def validate_dataset(
    root: str | Path = DEFAULT_DATA_ROOT,
    spec: DatasetSpec = ASVSPOOF2019_LA,
) -> DatasetReport:
    """Validate an extracted ASVspoof 2019 LA tree against ``spec``."""
    root = Path(root)
    problems: list[str] = []
    splits: dict[str, SplitReport] = {}

    if not root.is_dir():
        problems.append(f"data root does not exist or is not a directory: {root}")
        return DatasetReport(
            root=root,
            total_files=0,
            total_bytes=0,
            readme_sha256="",
            license_sha256="",
            splits={},
            problems=problems,
        )

    expected_top = {
        spec.readme_name,
        spec.cm_protocols_dirname,
        spec.asv_protocols_dirname,
        spec.asv_scores_dirname,
    }
    expected_top |= {split.dirname for split in spec.splits}
    actual_top = {
        path.name
        for path in root.iterdir()
        if path.name not in OS_METADATA_NAMES
        and path.name not in OS_METADATA_DIRS
    }
    missing_top = expected_top - actual_top
    unexpected_top = actual_top - expected_top
    if missing_top:
        problems.append(f"missing top-level entries: {sorted(missing_top)}")
    if unexpected_top:
        problems.append(f"unexpected top-level entries: {sorted(unexpected_top)}")

    files = list(_iter_dataset_files(root))
    total_files = len(files)
    total_bytes = sum(path.stat().st_size for path in files)
    if spec.total_files is not None and total_files != spec.total_files:
        problems.append(
            f"expected {spec.total_files} files, found {total_files}"
        )
    if spec.total_bytes is not None and total_bytes != spec.total_bytes:
        problems.append(
            f"expected {spec.total_bytes} bytes, found {total_bytes}"
        )

    readme_path = root / spec.readme_name
    readme_sha256 = ""
    if readme_path.is_file():
        readme_sha256 = sha256_file(readme_path)
        if readme_sha256 != spec.readme_sha256:
            problems.append(
                f"{spec.readme_name}: SHA-256 mismatch: expected "
                f"{spec.readme_sha256}, found {readme_sha256}"
            )
    else:
        problems.append(f"missing {spec.readme_name}")

    cm_protocols_dir = root / spec.cm_protocols_dirname
    expected_protocols = {split.protocol_name for split in spec.splits}
    if cm_protocols_dir.is_dir():
        actual_protocols = {
            path.name
            for path in cm_protocols_dir.iterdir()
            if path.is_file() and path.name not in OS_METADATA_NAMES
        }
        missing_protocols = expected_protocols - actual_protocols
        unexpected_protocols = actual_protocols - expected_protocols
        if missing_protocols:
            problems.append(
                f"missing CM protocol files: {sorted(missing_protocols)}"
            )
        if unexpected_protocols:
            problems.append(
                f"unexpected CM protocol files: {sorted(unexpected_protocols)}"
            )
    else:
        problems.append(f"missing directory: {spec.cm_protocols_dirname}")

    license_shas: set[str] = set()
    for split in spec.splits:
        split_dir = root / split.dirname
        flac_dir = split_dir / split.flac_dirname
        protocol_path = root / spec.cm_protocols_dirname / split.protocol_name
        license_path = split_dir / spec.license_name

        if not split_dir.is_dir():
            problems.append(f"missing split directory: {split.dirname}")
            continue
        if not flac_dir.is_dir():
            problems.append(
                f"missing FLAC directory: {split.dirname}/{split.flac_dirname}"
            )
            continue
        if not protocol_path.is_file():
            problems.append(
                f"missing protocol: {spec.cm_protocols_dirname}/{split.protocol_name}"
            )
            continue

        expected_split_entries = {spec.license_name, split.flac_dirname}
        actual_split_entries = {
            path.name
            for path in split_dir.iterdir()
            if path.name not in OS_METADATA_NAMES
        }
        unexpected_entries = actual_split_entries - expected_split_entries
        if unexpected_entries:
            problems.append(
                f"{split.dirname}: unexpected entries: {sorted(unexpected_entries)}"
            )

        if not license_path.is_file():
            problems.append(f"missing license: {split.dirname}/{spec.license_name}")
        else:
            license_sha256 = sha256_file(license_path)
            license_shas.add(license_sha256)
            if license_sha256 != spec.license_sha256:
                problems.append(
                    f"{split.dirname}/{spec.license_name}: SHA-256 mismatch: "
                    f"expected {spec.license_sha256}, found {license_sha256}"
                )

        try:
            protocol = parse_cm_protocol(
                protocol_path.read_text(encoding="utf-8"),
                source=repo_relative(protocol_path),
            )
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            problems.append(str(exc))
            continue

        non_flac = [
            path.name
            for path in flac_dir.iterdir()
            if path.is_file()
            and path.suffix != ".flac"
            and path.name not in OS_METADATA_NAMES
        ]
        if non_flac:
            problems.append(
                f"{split.dirname}/{split.flac_dirname}: non-FLAC files: "
                f"{sorted(non_flac)[:5]}"
            )

        flac_names = {path.stem for path in flac_dir.glob("*.flac")}
        protocol_sha256 = sha256_file(protocol_path)
        split_report = _summarize_split(
            split,
            protocol,
            flac_names,
            protocol_sha256,
            source=split.name,
        )
        splits[split.name] = split_report
        problems.extend(split_report.problems)

    for dirname, expected_count in (
        (spec.asv_protocols_dirname, spec.asv_protocols_count),
        (spec.asv_scores_dirname, spec.asv_scores_count),
    ):
        directory = root / dirname
        if not directory.is_dir():
            problems.append(f"missing directory: {dirname}")
            continue
        count = sum(
            1
            for path in directory.iterdir()
            if path.is_file() and path.name not in OS_METADATA_NAMES
        )
        if count != expected_count:
            problems.append(
                f"{dirname}: expected {expected_count} files, found {count}"
            )

    license_sha256 = ""
    if len(license_shas) == 1:
        license_sha256 = next(iter(license_shas))
    elif len(license_shas) > 1:
        problems.append("split LICENSE.txt files are not identical")

    return DatasetReport(
        root=root,
        total_files=total_files,
        total_bytes=total_bytes,
        readme_sha256=readme_sha256,
        license_sha256=license_sha256,
        splits=splits,
        problems=problems,
    )


def archive_source_label(md5: str, is_official: bool) -> str:
    """Return a human-readable source label for an archive checksum."""
    if is_official:
        return "University of Edinburgh DataShare / Zenodo mirror (official LA.zip)"
    return KNOWN_UNOFFICIAL_ARCHIVES.get(md5, "unofficial archive")


def validate_archive(
    path: str | Path,
    spec: DatasetSpec = ASVSPOOF2019_LA,
    allow_unofficial: bool = False,
) -> ArchiveReport:
    """Validate an ``LA.zip``-style archive without extracting it.

    If ``allow_unofficial`` is false, the archive must match the official
    ``LA.zip`` size and MD5.  If true, an unofficial archive is accepted but its
    extracted contents are still checked against ``spec``.
    """
    path = Path(path)
    problems: list[str] = []
    warnings: list[str] = []

    if not path.is_file():
        problems.append(f"archive does not exist or is not a file: {path}")
        return ArchiveReport(
            path=path,
            size_bytes=0,
            md5="",
            sha256="",
            is_official=False,
            source="",
            total_files=0,
            total_bytes=0,
            readme_sha256="",
            license_sha256="",
            splits={},
            problems=problems,
        )

    size_bytes = path.stat().st_size
    digests = hash_file(path, ("md5", "sha256"))
    md5 = digests["md5"]
    sha256 = digests["sha256"]
    is_official = size_bytes == spec.archive_size and md5 == spec.archive_md5
    source = archive_source_label(md5, is_official)

    if not is_official:
        if allow_unofficial:
            warnings.append(
                f"archive is not the official LA.zip; accepting {source} "
                "because --accept-any-archive was given"
            )
        else:
            problems.append(
                "archive does not match the official LA.zip "
                f"(expected size {spec.archive_size} and MD5 {spec.archive_md5}; "
                f"found size {size_bytes} and MD5 {md5}); pass "
                "--accept-any-archive to validate its contents anyway"
            )

    total_files = 0
    total_bytes = 0
    readme_sha256 = ""
    license_shas: set[str] = set()
    license_members: set[str] = set()
    protocol_bytes: dict[str, bytes] = {}
    split_flacs: dict[str, set[str]] = {split.name: set() for split in spec.splits}
    asv_protocol_count = 0
    asv_score_count = 0
    unexpected_members: list[str] = []
    splits: dict[str, SplitReport] = {}

    try:
        with zipfile.ZipFile(path) as archive:
            for info in archive.infolist():
                if info.is_dir():
                    continue
                name = info.filename
                if not name.startswith("LA/"):
                    problems.append(
                        f"archive member is outside the expected LA/ root: {name}"
                    )
                    continue
                rel_str = name[3:]
                if not rel_str:
                    continue
                rel_path = PurePosixPath(rel_str)
                if rel_path.is_absolute() or ".." in rel_path.parts:
                    problems.append(
                        f"archive member has an unsafe path: {name}"
                    )
                    continue

                total_files += 1
                total_bytes += info.file_size

                if rel_str == spec.readme_name:
                    readme_sha256 = hashlib.sha256(archive.read(info)).hexdigest()
                    continue

                matched = False
                for split in spec.splits:
                    flac_prefix = f"{split.dirname}/{split.flac_dirname}/"
                    if rel_str.startswith(flac_prefix) and rel_str.endswith(".flac"):
                        split_flacs[split.name].add(rel_path.stem)
                        matched = True
                        break
                    if rel_str == f"{split.dirname}/{spec.license_name}":
                        license_shas.add(
                            hashlib.sha256(archive.read(info)).hexdigest()
                        )
                        license_members.add(split.name)
                        matched = True
                        break
                    if rel_str == (
                        f"{spec.cm_protocols_dirname}/{split.protocol_name}"
                    ):
                        protocol_bytes[split.name] = archive.read(info)
                        matched = True
                        break
                if matched:
                    continue

                if rel_str.startswith(f"{spec.asv_protocols_dirname}/"):
                    asv_protocol_count += 1
                elif rel_str.startswith(f"{spec.asv_scores_dirname}/"):
                    asv_score_count += 1
                else:
                    unexpected_members.append(name)
    except zipfile.BadZipFile:
        problems.append(f"not a valid zip archive: {path}")
    except OSError as exc:
        problems.append(f"could not read archive {path}: {exc}")

    if unexpected_members:
        problems.append(
            f"archive contains unexpected members, e.g. {sorted(unexpected_members)[:5]}"
        )
    if spec.total_files is not None and total_files != spec.total_files:
        problems.append(
            f"archive: expected {spec.total_files} files, found {total_files}"
        )
    if spec.total_bytes is not None and total_bytes != spec.total_bytes:
        problems.append(
            f"archive: expected {spec.total_bytes} uncompressed bytes, "
            f"found {total_bytes}"
        )
    if readme_sha256 == "":
        problems.append(f"archive: missing {spec.readme_name}")
    elif readme_sha256 != spec.readme_sha256:
        problems.append(
            f"archive: {spec.readme_name} SHA-256 mismatch: expected "
            f"{spec.readme_sha256}, found {readme_sha256}"
        )
    for split in spec.splits:
        if split.name not in license_members:
            problems.append(
                f"archive: missing {split.dirname}/{spec.license_name}"
            )
    if len(license_shas) != 1:
        problems.append(
            f"archive: expected one distinct {spec.license_name}, found "
            f"{len(license_shas)}"
        )
    elif next(iter(license_shas)) != spec.license_sha256:
        problems.append(
            f"archive: {spec.license_name} SHA-256 mismatch: expected "
            f"{spec.license_sha256}, found {next(iter(license_shas))}"
        )
    if asv_protocol_count != spec.asv_protocols_count:
        problems.append(
            f"archive: expected {spec.asv_protocols_count} ASV protocol files, "
            f"found {asv_protocol_count}"
        )
    if asv_score_count != spec.asv_scores_count:
        problems.append(
            f"archive: expected {spec.asv_scores_count} ASV score files, "
            f"found {asv_score_count}"
        )

    for split in spec.splits:
        raw_protocol = protocol_bytes.get(split.name)
        if raw_protocol is None:
            problems.append(
                f"archive: missing protocol "
                f"{spec.cm_protocols_dirname}/{split.protocol_name}"
            )
            continue
        try:
            protocol = parse_cm_protocol(
                raw_protocol.decode("utf-8"),
                source=f"archive:{split.protocol_name}",
            )
        except (UnicodeDecodeError, ValueError) as exc:
            problems.append(str(exc))
            continue
        protocol_sha256 = hashlib.sha256(raw_protocol).hexdigest()
        split_report = _summarize_split(
            split,
            protocol,
            split_flacs[split.name],
            protocol_sha256,
            source=f"archive:{split.name}",
        )
        splits[split.name] = split_report
        problems.extend(split_report.problems)

    license_sha256 = ""
    if len(license_shas) == 1:
        license_sha256 = next(iter(license_shas))

    return ArchiveReport(
        path=path,
        size_bytes=size_bytes,
        md5=md5,
        sha256=sha256,
        is_official=is_official,
        source=source,
        total_files=total_files,
        total_bytes=total_bytes,
        readme_sha256=readme_sha256,
        license_sha256=license_sha256,
        splits=splits,
        problems=problems,
        warnings=warnings,
    )


def build_manifest(
    dataset_report: DatasetReport,
    archive_report: ArchiveReport | None = None,
    command: str | None = None,
) -> dict[str, object]:
    """Build a JSON-serialisable provenance manifest."""
    split_payload = {
        name: {
            "protocol_rows": report.protocol_rows,
            "bonafide": report.bonafide,
            "spoof": report.spoof,
            "flac_files": report.flac_files,
            "enrollment_files": report.enrollment_files,
            "protocol_sha256": report.protocol_sha256,
            "missing_protocol_files": report.missing_protocol_files,
            "unexpected_files": report.unexpected_files,
        }
        for name, report in dataset_report.splits.items()
    }

    manifest: dict[str, object] = {
        "dataset": "ASVspoof 2019 Logical Access (LA)",
        "doi": OFFICIAL_DOI,
        "license": OFFICIAL_LICENSE,
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "data_root": repo_relative(dataset_report.root),
        "validation": {
            "status": "pass" if dataset_report.valid else "fail",
            "problems": list(dataset_report.problems),
        },
        "counts": {
            "total_files": dataset_report.total_files,
            "total_bytes": dataset_report.total_bytes,
            "readme_sha256": dataset_report.readme_sha256,
            "license_sha256": dataset_report.license_sha256,
        },
        "splits": split_payload,
        "archive": None,
    }

    if archive_report is not None:
        manifest["archive"] = {
            "path": repo_relative(archive_report.path),
            "size_bytes": archive_report.size_bytes,
            "md5": archive_report.md5,
            "sha256": archive_report.sha256,
            "is_official": archive_report.is_official,
            "source": archive_report.source,
            "total_files": archive_report.total_files,
            "total_bytes": archive_report.total_bytes,
            "validation": {
                "status": "pass" if archive_report.valid else "fail",
                "problems": list(archive_report.problems),
                "warnings": list(archive_report.warnings),
            },
        }

    return manifest


def write_manifest(
    path: str | Path = DEFAULT_MANIFEST_PATH,
    manifest: dict[str, object] | None = None,
) -> Path:
    """Write a provenance manifest as pretty-printed JSON."""
    if manifest is None:
        raise ValueError("manifest is required")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return path
