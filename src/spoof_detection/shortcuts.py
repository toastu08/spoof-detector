"""Silence/noise shortcut conditions for the E1.4 ablation.

E1.4 asks whether the frozen ImageNet ResNet18 detects spoofing from the
*speech* content of a clip, or from incidental non-speech content (silence,
noise, padding) that survives the fixed 4-second mel contract.  This module
defines three image conditions over the fixed 128-band mel representation:

* ``full`` - the existing mel128/magma/flip contract (control);
* ``speech_only`` - speech frames kept, non-speech frames masked to the colormap
  bottom;
* ``nonspeech_only`` - non-speech frames kept, speech frames masked to the
  colormap bottom.

A frame is "speech" when its RMS energy lies within ``SPEECH_DB_GAP`` of the
clip's per-frame energy peak.  The mask is a hard time mask applied *after* the
mel is computed and min-max normalised, so the kept content is pixel-identical
to the corresponding frames of the ``full`` image and the removed content is
replaced by a neutral (colormap-bottom) region.

``full`` delegates to :func:`spoof_detection.preprocess.save_mel_spectrogram_png`
verbatim so it stays byte/pixel-compatible with ``data/dataset_img`` (AGENTS.md
G3).  The masked conditions are new image sets and only need to be deterministic
and correctly masked, not identical to any existing export.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np

from .preprocess import (
    COLORMAP,
    IMAGENET_MEAN,
    IMAGENET_STD,
    INFERENCE_SIZE,
    N_MELS,
    PREPROCESSING_VERSION,
    SAMPLE_RATE,
    TARGET_LENGTH,
    TARGET_SECONDS,
    load_fixed_audio,
    normalize_mel_db,
    save_mel_spectrogram_png,
)

# Frame geometry must match the mel contract exactly (librosa defaults used by
# ``melspectrogram``): n_fft / frame_length 2048, hop_length 512, center=True,
# producing exactly 126 time frames for 4 s at 16 kHz.
FRAME_LENGTH = 2048
HOP_LENGTH = 512
TIME_FRAMES = 126

# A frame is "speech" when its RMS energy is within this many dB of the clip's
# per-frame energy peak.  Everything below that is non-speech (silence, noise or
# the zero-padding introduced by the 4-second fix).
SPEECH_DB_GAP = 30.0


@dataclass(frozen=True)
class ShortcutSpec:
    """Metadata for one E1.4 image condition."""

    name: str
    keep: str  # "all" | "speech" | "nonspeech"
    version: str
    description: str


SHORTCUT_ORDER = (
    "full",
    "speech_only",
    "nonspeech_only",
)

SHORTCUTS: dict[str, ShortcutSpec] = {
    "full": ShortcutSpec(
        name="full",
        keep="all",
        version=PREPROCESSING_VERSION,
        description="Existing 128-band mel contract (control: no masking)",
    ),
    "speech_only": ShortcutSpec(
        name="speech_only",
        keep="speech",
        version="mel128-magma-speech-only-4s-v1",
        description="128-band mel, speech frames kept, non-speech masked to black",
    ),
    "nonspeech_only": ShortcutSpec(
        name="nonspeech_only",
        keep="nonspeech",
        version="mel128-magma-nonspeech-only-4s-v1",
        description="128-band mel, non-speech frames kept, speech masked to black",
    ),
}


def get_shortcut_spec(name: str) -> ShortcutSpec:
    """Return the spec for ``name`` or raise a helpful error."""
    try:
        return SHORTCUTS[name]
    except KeyError:
        raise ValueError(
            f"unknown shortcut condition {name!r}; "
            f"expected one of {list(SHORTCUT_ORDER)}"
        ) from None


def _mel_db(y_fixed: np.ndarray, sr: int) -> np.ndarray:
    """Return the dB-scaled 128-band mel spectrogram of already-fixed audio.

    This mirrors :func:`spoof_detection.preprocess.generate_mel_spectrogram`
    exactly, but takes the fixed waveform so the caller can decode audio once
    and reuse it for the energy mask.
    """
    mel = librosa.feature.melspectrogram(y=y_fixed, sr=sr, n_mels=N_MELS)
    return librosa.power_to_db(mel, ref=np.max)


def frame_energy_db(y_fixed: np.ndarray, sr: int) -> np.ndarray:
    """Return per-frame RMS energy in dB relative to the clip peak.

    The result has shape ``(TIME_FRAMES,)`` and is aligned to the mel time axis
    (same frame_length, hop_length and centering).
    """
    rms = librosa.feature.rms(
        y=y_fixed,
        frame_length=FRAME_LENGTH,
        hop_length=HOP_LENGTH,
        center=True,
    )
    rms = np.asarray(rms).reshape(-1)
    return librosa.amplitude_to_db(rms, ref=np.max, top_db=None)


def speech_mask(
    y_fixed: np.ndarray,
    sr: int,
    speech_db_gap: float = SPEECH_DB_GAP,
) -> np.ndarray:
    """Return a boolean ``(TIME_FRAMES,)`` mask of speech frames.

    A frame is speech when its RMS energy is within ``speech_db_gap`` of the
    clip's per-frame energy peak.  A constant-energy clip has every frame marked
    speech, which makes ``nonspeech_only`` a constant (all-black) image - the
    degenerate case is handled downstream rather than special-cased here.
    """
    db = frame_energy_db(y_fixed, sr)
    return db >= (float(np.max(db)) - speech_db_gap)


def masked_db(
    name: str,
    spec_db: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Return the dB/array to render for one shortcut condition.

    ``full`` returns the raw dB matrix (autoscaled at save time).  The masked
    conditions return the min-max normalised matrix with the removed frames set
    to ``0.0`` (the colormap bottom), so the kept frames are pixel-identical to
    the corresponding frames of the ``full`` image.
    """
    spec = get_shortcut_spec(name)
    if spec.keep == "all":
        return np.asarray(spec_db)

    mask = np.asarray(mask, dtype=bool).reshape(-1)
    normalized = normalize_mel_db(np.asarray(spec_db)).copy()
    if spec.keep == "speech":
        normalized[:, ~mask] = 0.0
    elif spec.keep == "nonspeech":
        normalized[:, mask] = 0.0
    else:
        raise ValueError(f"unsupported keep {spec.keep!r} for {name}")
    return normalized


def save_shortcut_png(
    name: str,
    spec_db: np.ndarray,
    mask: np.ndarray,
    output_path: str | Path,
) -> Path:
    """Render one clip under condition ``name`` and save it as a PNG.

    ``full`` uses the exact existing autoscale path.  The masked conditions save
    the already-normalised ``[0, 1]`` array with explicit ``vmin=0`` / ``vmax=1``
    so the mask value cannot be re-normalised.
    """
    spec = get_shortcut_spec(name)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    values = masked_db(name, spec_db, mask)
    if spec.keep == "all":
        return save_mel_spectrogram_png(values, output_path)
    return save_mel_spectrogram_png(
        values,
        output_path,
        vmin=0.0,
        vmax=1.0,
    )


def render_shortcut_rgb(
    name: str,
    spec_db: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Return the ``uint8`` RGB image a saved condition decodes to.

    Used for pixel-level checks and tests; mirrors :func:`save_shortcut_png`.
    """
    from .preprocess import colorize_mel

    spec = get_shortcut_spec(name)
    values = masked_db(name, spec_db, mask)
    if spec.keep == "all":
        return colorize_mel(normalize_mel_db(values))
    return colorize_mel(values)


def shortcut_payload(name: str) -> dict[str, object]:
    """Return the JSON-serialisable provenance block for ``name``."""
    spec = get_shortcut_spec(name)
    return {
        "version": spec.version,
        "condition": spec.name,
        "description": spec.description,
        "representation": "mel128",
        "representation_version": PREPROCESSING_VERSION,
        "params": {
            "n_mels": N_MELS,
            "n_fft": FRAME_LENGTH,
            "hop_length": HOP_LENGTH,
            "power_to_db_ref": "np.max",
            "speech_db_gap": SPEECH_DB_GAP,
            "mask_keep": spec.keep,
            "mask_fill": "colormap bottom (0.0 after min-max normalisation)",
        },
        "colormap": COLORMAP,
        "orientation": "lower",
        "flip": True,
        "encoding": "RGBA",
        "sample_rate": SAMPLE_RATE,
        "target_seconds": TARGET_SECONDS,
        "target_length": TARGET_LENGTH,
        "resize": [INFERENCE_SIZE, INFERENCE_SIZE],
        "imagenet_mean": list(IMAGENET_MEAN),
        "imagenet_std": list(IMAGENET_STD),
    }
