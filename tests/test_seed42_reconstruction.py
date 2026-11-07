"""Fast checks for the E0.1 seed-42 training-subset reconstruction."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from spoof_detection.preprocess import (  # noqa: E402
    generate_mel_spectrogram,
    render_mel_spectrogram_png_bytes,
)
from spoof_detection.protocol import (  # noqa: E402
    load_train_protocol,
    validate_train_protocol,
)
from spoof_detection.sampling import (  # noqa: E402
    SEED42_ANCHORS,
    build_seed42_subset,
    subset_summary,
    validate_seed42_subset,
)

REFERENCE_DIR = REPO_ROOT / "data" / "dataset_img" / "train"


class TrainSeed42ReconstructionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = load_train_protocol()
        cls.subset = build_seed42_subset(cls.protocol)

    def test_protocol_matches_expected_counts(self) -> None:
        self.assertEqual(validate_train_protocol(self.protocol), [])
        self.assertEqual(len(self.protocol), 25_380)
        counts = self.protocol["label"].value_counts().to_dict()
        self.assertEqual(counts, {1: 22_800, 0: 2_580})

    def test_protocol_bonafide_block_is_first(self) -> None:
        self.assertTrue((self.protocol.loc[:2_579, "label"] == 0).all())
        self.assertTrue((self.protocol.loc[2_580:, "label"] == 1).all())

    def test_subset_counts_and_anchors(self) -> None:
        self.assertEqual(validate_seed42_subset(self.subset), [])
        self.assertEqual(len(self.subset), 10_080)
        counts = self.subset["label"].value_counts().to_dict()
        self.assertEqual(counts, {1: 7_500, 0: 2_580})
        self.assertEqual(len(SEED42_ANCHORS), 20)
        for sample_index, expected in SEED42_ANCHORS.items():
            row = self.subset.iloc[sample_index]
            got = (row["speaker_id"], row["system_id"], int(row["label"]))
            self.assertEqual(got, expected, f"anchor {sample_index}")

    def test_subset_covers_all_speakers_and_attacks(self) -> None:
        summary = subset_summary(self.subset)
        self.assertEqual(summary["speakers"], 20)
        self.assertEqual(summary["bonafide_speakers"], 20)
        self.assertEqual(summary["spoof_speakers"], 20)
        self.assertEqual(
            set(summary["attack_counts"]),
            {"A01", "A02", "A03", "A04", "A05", "A06"},
        )

    def test_no_duplicate_clips(self) -> None:
        self.assertFalse(self.subset["audio_file_name"].duplicated().any())
        self.assertFalse(self.subset["protocol_row"].duplicated().any())

    def test_regenerated_images_match_reference_export(self) -> None:
        references = [
            (row.protocol_row, row.label, row.source_flac)
            for row in self.subset.itertuples(index=False)
            if row.protocol_row in {0, 2_580, 25_379}
        ]
        if not references or not all(
            (REFERENCE_DIR / ("bonafide" if label == 0 else "spoofed")
             / f"sample_{protocol_row}.png").exists()
            for protocol_row, label, _ in references
        ):
            self.skipTest("full-split reference export is not available")

        import matplotlib.image as mpimg

        for protocol_row, label, source_flac in references:
            spec_db = generate_mel_spectrogram(REPO_ROOT / source_flac)
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "sample.png"
                path.write_bytes(render_mel_spectrogram_png_bytes(spec_db))
                generated = mpimg.imread(path)

            folder = "bonafide" if label == 0 else "spoofed"
            reference = mpimg.imread(
                REFERENCE_DIR / folder / f"sample_{protocol_row}.png"
            )
            self.assertTrue(
                generated.shape == reference.shape
                and np.array_equal(generated, reference),
                f"pixel mismatch for protocol row {protocol_row}",
            )


if __name__ == "__main__":
    unittest.main()
