"""Tests for the E1.1 head-only probe training loop."""

from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from spoof_detection.model import (  # noqa: E402
    AudioSpoofDetector,
    load_model_from_checkpoint,
)
from spoof_detection.train_probe import (  # noqa: E402
    ProbeTrainingConfig,
    build_probe_model,
    load_probe_frame,
    save_checkpoint,
    train_probe,
    verify_backbone_parity,
)


def _write_images(root: Path, count: int, prefix: str = "img") -> list[dict[str, object]]:
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    rows: list[dict[str, object]] = []
    for index in range(count):
        label = index % 2
        folder = "bonafide" if label == 0 else "spoofed"
        folder_dir = root / folder
        folder_dir.mkdir(exist_ok=True)
        path = folder_dir / f"{prefix}_{index}.png"
        pixels = rng.integers(0, 256, size=(16, 16, 3), dtype=np.uint8)
        Image.fromarray(pixels, mode="RGB").save(path)
        rows.append(
            {
                "image_path": str(path),
                "label": label,
                "speaker_id": f"speaker_{index % 2}",
                "system_id": "-" if label == 0 else "A01",
                "protocol_row": index,
            }
        )
    return rows


class ProbeModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.manual_seed(0)
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp_dir.name)
        cls.reference = cls.root / "reference.pth"
        model = AudioSpoofDetector()
        torch.save(
            {
                key.replace("backbone.", "network.", 1): value
                for key, value in model.state_dict().items()
            },
            cls.reference,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp_dir.cleanup()

    def test_head_is_trainable_and_backbone_is_frozen(self) -> None:
        model = build_probe_model(
            seed=42,
            reference_checkpoint=self.reference,
            device="cpu",
            backbone_weights="checkpoint",
        )
        for name, parameter in model.backbone.named_parameters():
            if name.startswith("fc."):
                self.assertTrue(parameter.requires_grad, name)
            else:
                self.assertFalse(parameter.requires_grad, name)
        self.assertTrue(model.training)

    def test_head_initialisation_is_deterministic(self) -> None:
        first = build_probe_model(
            seed=42,
            reference_checkpoint=self.reference,
            device="cpu",
            backbone_weights="checkpoint",
        )
        second = build_probe_model(
            seed=42,
            reference_checkpoint=self.reference,
            device="cpu",
            backbone_weights="checkpoint",
        )
        for key in first.backbone.fc.state_dict():
            self.assertTrue(
                torch.equal(
                    first.backbone.fc.state_dict()[key],
                    second.backbone.fc.state_dict()[key],
                ),
                key,
            )

    def test_verify_backbone_parity_against_reference(self) -> None:
        model = build_probe_model(
            seed=7,
            reference_checkpoint=self.reference,
            device="cpu",
            backbone_weights="checkpoint",
        )
        report = verify_backbone_parity(model, self.reference)
        self.assertEqual(report["mismatched_keys"], [])
        self.assertGreater(report["checked_keys"], 0)

    def test_freeze_bn_switches_batch_norm_to_eval(self) -> None:
        model = build_probe_model(
            seed=1,
            reference_checkpoint=self.reference,
            device="cpu",
            freeze_bn=True,
            backbone_weights="checkpoint",
        )
        batch_norms = [
            module
            for module in model.backbone.modules()
            if isinstance(module, torch.nn.BatchNorm2d)
        ]
        self.assertTrue(batch_norms)
        self.assertTrue(all(not module.training for module in batch_norms))


class LoadProbeFrameTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_limit_is_class_stratified(self) -> None:
        rows = _write_images(self.root / "images", 10)
        manifest = self.root / "manifest.csv"
        pd.DataFrame(rows).to_csv(manifest, index=False)
        frame = load_probe_frame(manifest, self.root, limit=4)
        self.assertEqual(len(frame), 4)
        self.assertEqual(sorted(frame["label"].unique().tolist()), [0, 1])
        self.assertEqual(int((frame["label"] == 0).sum()), 2)

    def test_img_path_column_is_accepted(self) -> None:
        rows = _write_images(self.root / "images", 2)
        for row in rows:
            row["img_path"] = row.pop("image_path")
        manifest = self.root / "manifest.csv"
        pd.DataFrame(rows).to_csv(manifest, index=False)
        frame = load_probe_frame(manifest, self.root)
        self.assertTrue(frame["image_path"].map(Path.is_file).all())

    def test_missing_image_is_rejected(self) -> None:
        rows = _write_images(self.root / "images", 2)
        (Path(rows[0]["image_path"])).unlink()
        manifest = self.root / "manifest.csv"
        pd.DataFrame(rows).to_csv(manifest, index=False)
        with self.assertRaises(FileNotFoundError):
            load_probe_frame(manifest, self.root)

    def test_missing_label_column_is_rejected(self) -> None:
        rows = _write_images(self.root / "images", 2)
        manifest = self.root / "manifest.csv"
        pd.DataFrame(rows).drop(columns=["label"]).to_csv(manifest, index=False)
        with self.assertRaises(ValueError):
            load_probe_frame(manifest, self.root)


class ProbeTrainingEndToEndTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp_dir.name)
        cls.reference = cls.root / "reference.pth"
        torch.manual_seed(0)
        model = AudioSpoofDetector()
        torch.save(
            {
                key.replace("backbone.", "network.", 1): value
                for key, value in model.state_dict().items()
            },
            cls.reference,
        )

        train_rows = _write_images(cls.root / "train", 4, prefix="train")
        dev_rows = _write_images(cls.root / "dev", 4, prefix="dev")
        cls.train_manifest = cls.root / "train.csv"
        cls.dev_manifest = cls.root / "dev.csv"
        pd.DataFrame(train_rows).to_csv(cls.train_manifest, index=False)
        pd.DataFrame(dev_rows).to_csv(cls.dev_manifest, index=False)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp_dir.cleanup()

    def test_one_epoch_writes_strict_checkpoint_and_config(self) -> None:
        checkpoint = self.root / "out" / "resnet18_frozen_mel64.pth"
        result = train_probe(
            ProbeTrainingConfig(
                frontend="mel64",
                train_manifest=self.train_manifest,
                dev_manifest=self.dev_manifest,
                checkpoint_out=checkpoint,
                reference_checkpoint=self.reference,
                backbone_weights="checkpoint",
                batch_size=2,
                max_epochs=1,
                patience=5,
                num_workers=0,
                device="cpu",
            )
        )
        self.assertTrue(checkpoint.is_file())
        self.assertEqual(result.best_epoch, 1)
        self.assertEqual(result.epochs_run, 1)
        self.assertTrue(result.log_path.is_file())
        self.assertTrue(result.config_path.is_file())

        _model, report = load_model_from_checkpoint(checkpoint, device="cpu")
        self.assertEqual(list(report.missing_keys), [])
        self.assertEqual(list(report.unexpected_keys), [])

    def test_existing_checkpoint_requires_force(self) -> None:
        checkpoint = self.root / "guard" / "resnet18_frozen_mel64.pth"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"placeholder")
        with self.assertRaises(FileExistsError):
            train_probe(
                ProbeTrainingConfig(
                    frontend="mel64",
                    train_manifest=self.train_manifest,
                    dev_manifest=self.dev_manifest,
                    checkpoint_out=checkpoint,
                    reference_checkpoint=self.reference,
                    backbone_weights="checkpoint",
                    batch_size=2,
                    max_epochs=1,
                    num_workers=0,
                    device="cpu",
                )
            )


class SaveCheckpointTest(unittest.TestCase):
    def test_save_round_trips_strictly(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            reference_model = AudioSpoofDetector()
            checkpoint = root / "probe.pth"
            report = save_checkpoint(reference_model, checkpoint)
            self.assertEqual(report["missing_keys"], [])
            self.assertEqual(report["unexpected_keys"], [])
            self.assertEqual(
                report["total_parameters"],
                sum(p.numel() for p in reference_model.parameters()),
            )


if __name__ == "__main__":
    unittest.main()
