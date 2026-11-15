"""Time-frequency front-end registry for the E1.1 front-end ablation.

E1.1 asks whether the frozen ResNet18 needs a 128-band mel spectrogram, or
whether any image-like time-frequency representation works.  Every front-end in
this module shares the same audio contract (16 kHz, 4 s, 64,000 samples) and the
same rendering path (min-max normalisation, ``magma`` colormap, vertical flip),
so the only variable is the 2-D time-frequency representation.

``mel128`` is the existing contract and must remain pixel-identical to
``data/dataset_img``.  It reuses :mod:`spoof_detection.preprocess` verbatim; do
not change that path without a documented ablation (AGENTS.md G3).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np

from .preprocess import (
    COLORMAP,
    IMAGENET_MEAN,
    IMAGENET_STD,
    INFERENCE_SIZE,
    PREPROCESSING_VERSION,
    SAMPLE_RATE,
    TARGET_LENGTH,
    TARGET_SECONDS,
    colorize_mel,
    load_fixed_audio,
    normalize_mel_db,
    save_mel_spectrogram_png,
)

# The 4 s / 16 kHz contract with librosa's default n_fft=2048, hop_length=512 and
# center=True produces exactly 126 time frames for every representation.
TIME_FRAMES = 126

# Linear-frequency STFT settings.  These match the mel front-end's n_fft/hop so
# that only the frequency binning differs.
LINEAR_N_FFT = 2048
LINEAR_HOP_LENGTH = 512
LINEAR_FREQUENCY_BINS = 1 + LINEAR_N_FFT // 2

MFCC_N_MFCC = 40

# Fixed random-projection control: the matrix is derived from the seed, never
# from data, and is identical for train, dev and eval.
RANDOM_PROJECTION_SEED = 42
RANDOM_PROJECTION_ROWS = 128


@dataclass(frozen=True)
class FrontendSpec:
    """Metadata for one front-end condition."""

    name: str
    kind: str
    version: str
    description: str
    expected_shape: tuple[int, int]
    params: dict[str, object]


def _mel(y: np.ndarray, sr: int, n_mels: int) -> np.ndarray:
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    return librosa.power_to_db(mel, ref=np.max)


def _linear_stft(y: np.ndarray, sr: int) -> np.ndarray:
    stft = librosa.stft(
        y,
        n_fft=LINEAR_N_FFT,
        hop_length=LINEAR_HOP_LENGTH,
    )
    return librosa.power_to_db(np.abs(stft) ** 2, ref=np.max)


def _mfcc(y: np.ndarray, sr: int) -> np.ndarray:
    return librosa.feature.mfcc(y=y, sr=sr, n_mfcc=MFCC_N_MFCC)


def _random_projection(y: np.ndarray, sr: int) -> np.ndarray:
    linear_db = _linear_stft(y, sr)
    matrix = random_projection_matrix(cols=linear_db.shape[0])
    return matrix @ linear_db


_COMPUTE = {
    "mel": _mel,
    "linear_stft": _linear_stft,
    "mfcc": _mfcc,
    "random_projection": _random_projection,
}


def random_projection_matrix(
    rows: int = RANDOM_PROJECTION_ROWS,
    cols: int = LINEAR_FREQUENCY_BINS,
    seed: int = RANDOM_PROJECTION_SEED,
) -> np.ndarray:
    """Return the fixed Gaussian projection matrix used by ``randproj128``.

    The matrix depends only on ``seed`` and the shape, so train, dev and eval
    always use the same projection.
    """
    rng = np.random.default_rng(seed)
    return rng.standard_normal((rows, cols), dtype=np.float64)


def random_projection_sha256(
    rows: int = RANDOM_PROJECTION_ROWS,
    cols: int = LINEAR_FREQUENCY_BINS,
    seed: int = RANDOM_PROJECTION_SEED,
) -> str:
    """Return a stable SHA-256 over the projection matrix bytes and shape."""
    matrix = random_projection_matrix(rows=rows, cols=cols, seed=seed)
    hasher = hashlib.sha256()
    hasher.update(str(tuple(matrix.shape)).encode("utf-8"))
    hasher.update(matrix.tobytes())
    return hasher.hexdigest()


FRONTEND_ORDER = (
    "mel64",
    "mel128",
    "mel256",
    "linear_stft",
    "mfcc40",
    "randproj128",
)

FRONTENDS: dict[str, FrontendSpec] = {
    "mel64": FrontendSpec(
        name="mel64",
        kind="mel",
        version="mel64-magma-4s-v1",
        description="64-band mel spectrogram, dB, magma, flipped",
        expected_shape=(64, TIME_FRAMES),
        params={"n_mels": 64, "n_fft": 2048, "hop_length": 512, "power_to_db_ref": "np.max"},
    ),
    "mel128": FrontendSpec(
        name="mel128",
        kind="mel",
        version=PREPROCESSING_VERSION,
        description="Existing 128-band mel contract (control)",
        expected_shape=(128, TIME_FRAMES),
        params={"n_mels": 128, "n_fft": 2048, "hop_length": 512, "power_to_db_ref": "np.max"},
    ),
    "mel256": FrontendSpec(
        name="mel256",
        kind="mel",
        version="mel256-magma-4s-v1",
        description="256-band mel spectrogram, dB, magma, flipped",
        expected_shape=(256, TIME_FRAMES),
        params={"n_mels": 256, "n_fft": 2048, "hop_length": 512, "power_to_db_ref": "np.max"},
    ),
    "linear_stft": FrontendSpec(
        name="linear_stft",
        kind="linear_stft",
        version="linear1025-magma-4s-v1",
        description="Linear-frequency power STFT in dB, magma, flipped",
        expected_shape=(LINEAR_FREQUENCY_BINS, TIME_FRAMES),
        params={
            "n_fft": LINEAR_N_FFT,
            "hop_length": LINEAR_HOP_LENGTH,
            "frequency_bins": LINEAR_FREQUENCY_BINS,
            "power_to_db_ref": "np.max",
        },
    ),
    "mfcc40": FrontendSpec(
        name="mfcc40",
        kind="mfcc",
        version="mfcc40-magma-4s-v1",
        description="40 MFCC coefficients (no deltas), magma, flipped",
        expected_shape=(MFCC_N_MFCC, TIME_FRAMES),
        params={"n_mfcc": MFCC_N_MFCC, "n_fft": 2048, "hop_length": 512},
    ),
    "randproj128": FrontendSpec(
        name="randproj128",
        kind="random_projection",
        version="randproj128-magma-4s-v1",
        description=(
            "128-row fixed Gaussian projection of the linear dB STFT (control)"
        ),
        expected_shape=(RANDOM_PROJECTION_ROWS, TIME_FRAMES),
        params={
            "rows": RANDOM_PROJECTION_ROWS,
            "cols": LINEAR_FREQUENCY_BINS,
            "seed": RANDOM_PROJECTION_SEED,
            "source": "linear_stft_db",
        },
    ),
}


def get_frontend_spec(name: str) -> FrontendSpec:
    """Return the spec for ``name`` or raise a helpful error."""
    try:
        return FRONTENDS[name]
    except KeyError:
        raise ValueError(
            f"unknown front-end {name!r}; expected one of {list(FRONTEND_ORDER)}"
        ) from None


def compute_frontend(
    name: str,
    file_path: str | Path,
    target_seconds: int = TARGET_SECONDS,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """Return the raw 2-D representation for one audio clip.

    The result is a float array of shape ``(rows, TIME_FRAMES)`` and is not yet
    normalised or colourised.  Rendering happens in :func:`render_frontend_rgb`
    and :func:`save_frontend_png`.
    """
    spec = get_frontend_spec(name)
    y, sr = load_fixed_audio(file_path, target_seconds, sample_rate)
    if spec.kind == "mel":
        n_mels = int(spec.params["n_mels"])
        return _COMPUTE["mel"](y, sr, n_mels=n_mels)
    return _COMPUTE[spec.kind](y, sr)


def render_frontend_rgb(name: str, file_path: str | Path) -> np.ndarray:
    """Render one clip to the ``uint8`` RGB image the model sees."""
    return colorize_mel(normalize_mel_db(compute_frontend(name, file_path)))


def save_frontend_png(
    name: str,
    file_path: str | Path,
    output_path: str | Path,
    cmap: str = COLORMAP,
) -> Path:
    """Render one clip and save it as the exported PNG.

    This uses the same ``plt.imsave(..., origin='lower')`` path as the original
    dataset export, so ``mel128`` is byte/pixel-compatible with
    ``data/dataset_img``.
    """
    representation = compute_frontend(name, file_path)
    return save_mel_spectrogram_png(representation, output_path, cmap=cmap)


def frontend_payload(name: str) -> dict[str, object]:
    """Return the JSON-serialisable provenance block for ``name``."""
    spec = get_frontend_spec(name)
    payload: dict[str, object] = {
        "version": spec.version,
        "frontend": spec.name,
        "description": spec.description,
        "params": dict(spec.params),
        "sample_rate": SAMPLE_RATE,
        "target_seconds": TARGET_SECONDS,
        "target_length": TARGET_LENGTH,
        "colormap": COLORMAP,
        "render": "plt.imsave(cmap='magma', origin='lower')",
        "resize": [INFERENCE_SIZE, INFERENCE_SIZE],
        "imagenet_mean": list(IMAGENET_MEAN),
        "imagenet_std": list(IMAGENET_STD),
    }
    if spec.kind == "random_projection":
        payload["random_projection_sha256"] = random_projection_sha256(
            rows=int(spec.params["rows"]),
            cols=int(spec.params["cols"]),
            seed=int(spec.params["seed"]),
        )
    return payload
