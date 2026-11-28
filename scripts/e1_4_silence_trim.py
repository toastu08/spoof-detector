#!/usr/bin/env python3
"""Bonafide silence-trimmed attribution for E1.4 (ROADMAP E1.4).

Reuses the frozen unweighted baseline (no retraining).  For each bonafide clip
in dev and eval it scores the original fixed 4-second input and a
silence-trimmed version (leading/trailing non-speech frames removed, then
re-fixed to 4 s), and reports the paired score shift.

A systematic *increase* in the spoof score after trimming would mean the model
leans on leading/trailing silence to say "bonafide".  This complements the
``nonspeech_only`` control probe, which asks whether silence can be trained into
a head; this script asks whether the *already-trained* model reads it.

Outputs ``<output>.csv`` (per clip) and ``<output>.json`` (summary).

Usage::

    uv run python scripts/e1_4_silence_trim.py
    uv run python scripts/e1_4_silence_trim.py --limit 200
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

import librosa  # noqa: E402

from spoof_detection.frontend_export import build_split_records  # noqa: E402
from spoof_detection.model import load_model_from_checkpoint  # noqa: E402
from spoof_detection.preprocess import (  # noqa: E402
    N_MELS,
    SAMPLE_RATE,
    TARGET_LENGTH,
    build_inference_transform,
    load_fixed_audio,
    render_mel_rgb,
)
from spoof_detection.shortcuts import HOP_LENGTH, speech_mask  # noqa: E402
from spoof_detection.train_probe import resolve_device  # noqa: E402

DEFAULT_CHECKPOINT = (
    "Models/Iteration1/best_audio_spoof_model_unweighted.pth"
)
DEFAULT_TRAIN_MANIFEST = "manifests/train_seed42.csv"
DEFAULT_OUTPUT = "reports/phase1/e1_4_shortcuts/silence_trim_bonafide"


def _mel_db(y_fixed: np.ndarray, sr: int) -> np.ndarray:
    mel = librosa.feature.melspectrogram(y=y_fixed, sr=sr, n_mels=N_MELS)
    return librosa.power_to_db(mel, ref=np.max)


def _trim_waveform(y_fixed: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Remove leading/trailing non-speech frames, then re-fix to 4 seconds."""
    indices = np.nonzero(mask)[0]
    if indices.size == 0:
        trimmed = y_fixed
    else:
        first = int(indices[0])
        last = int(indices[-1])
        start = first * HOP_LENGTH
        end = min(int(y_fixed.size), (last + 1) * HOP_LENGTH)
        trimmed = y_fixed[start:end]
    return librosa.util.fix_length(data=trimmed, size=TARGET_LENGTH)


def _score(model: torch.nn.Module, spec_db: np.ndarray, device: torch.device) -> float:
    rgb = render_mel_rgb(spec_db)
    tensor = build_inference_transform()(Image.fromarray(rgb))
    with torch.inference_mode():
        logit = model(tensor.unsqueeze(0).to(device)).reshape(-1)
    return float(torch.sigmoid(logit).item())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help=f"frozen baseline checkpoint (default: {DEFAULT_CHECKPOINT})",
    )
    parser.add_argument(
        "--train-manifest",
        default=DEFAULT_TRAIN_MANIFEST,
        help=f"seed-42 training manifest (default: {DEFAULT_TRAIN_MANIFEST})",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("dev", "eval"),
        default=["dev", "eval"],
        help="splits to analyse (default: dev eval)",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"output CSV/JSON base path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="inference device (default: auto)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="smoke only: analyse the first N bonafide clips per split",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = REPO_ROOT
    checkpoint = (repo_root / args.checkpoint).resolve()
    if not checkpoint.is_file():
        print(f"error: checkpoint not found: {checkpoint}", file=sys.stderr)
        return 2
    train_manifest = (repo_root / args.train_manifest).resolve()

    device = resolve_device(args.device)
    model, report = load_model_from_checkpoint(checkpoint, device=device)

    rows: list[dict[str, object]] = []
    for split in args.splits:
        records = build_split_records(
            split,
            repo_root=repo_root,
            train_manifest=train_manifest,
            data_root=repo_root / "data" / "LA",
        )
        bonafide = records[records["label"] == 0].reset_index(drop=True)
        if args.limit is not None:
            bonafide = bonafide.head(args.limit)

        for row in bonafide.itertuples(index=False):
            y_fixed, sr = load_fixed_audio(row.source_path)
            mask = speech_mask(y_fixed, sr)
            n_speech = int(mask.sum())
            n_frames = int(mask.size)

            original_spec = _mel_db(y_fixed, sr)
            original_score = _score(model, original_spec, device)

            trimmed = _trim_waveform(y_fixed, mask)
            trimmed_spec = _mel_db(trimmed, sr)
            trimmed_score = _score(model, trimmed_spec, device)

            rows.append(
                {
                    "split": split,
                    "protocol_row": int(row.protocol_row),
                    "speaker_id": str(row.speaker_id),
                    "system_id": str(row.system_id),
                    "n_frames": n_frames,
                    "n_speech_frames": n_speech,
                    "original_score": original_score,
                    "trimmed_score": trimmed_score,
                    "delta_score": trimmed_score - original_score,
                    "flipped": bool(original_score < 0.5 <= trimmed_score),
                }
            )

    frame = pd.DataFrame(rows)
    if frame.empty:
        print("error: no bonafide clips analysed", file=sys.stderr)
        return 2

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output.with_suffix(".csv")
    json_path = output.with_suffix(".json")
    frame.to_csv(csv_path, index=False)

    deltas = frame["delta_score"].to_numpy(dtype=float)
    summary = {
        "experiment": "E1.4",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": report.sha256,
        "n_clips": int(len(frame)),
        "splits": sorted(frame["split"].unique().tolist()),
        "original_mean": float(frame["original_score"].mean()),
        "trimmed_mean": float(frame["trimmed_score"].mean()),
        "delta_mean": float(deltas.mean()),
        "delta_median": float(np.median(deltas)),
        "delta_min": float(deltas.min()),
        "delta_max": float(deltas.max()),
        "n_flipped": int(frame["flipped"].sum()),
        "n_originally_spoof": int((frame["original_score"] >= 0.5).sum()),
        "n_originally_bonafide": int((frame["original_score"] < 0.5).sum()),
    }
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")
    print(f"n_clips={summary['n_clips']} delta_mean={summary['delta_mean']:.4f} "
          f"delta_median={summary['delta_median']:.4f} "
          f"n_flipped={summary['n_flipped']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
