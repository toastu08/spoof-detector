"""Audio-to-image preprocessing contract for the frozen ResNet18 baseline.

This module mirrors the verified training pipeline exactly:

1. load audio at 16 kHz (mono),
2. fix the length to 64,000 samples (4 s) by zero-padding/truncating,
3. compute a 128-band mel spectrogram,
4. convert to dB with ``power_to_db(ref=np.max)``,
5. render it with the ``magma`` colormap, flipped vertically (``origin='lower'``).

Anything that changes these defaults changes the model input contract.  Do not
alter this module without a documented ablation experiment (AGENTS.md G3).
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import librosa
import numpy as np

SAMPLE_RATE = 16_000
TARGET_SECONDS = 4
TARGET_LENGTH = SAMPLE_RATE * TARGET_SECONDS
N_MELS = 128
COLORMAP = "magma"


def generate_mel_spectrogram(
    file_path: str | Path,
    target_seconds: int = TARGET_SECONDS,
    sample_rate: int = SAMPLE_RATE,
    n_mels: int = N_MELS,
) -> np.ndarray:
    """Return the dB-scaled mel spectrogram used by the baseline.

    The result has shape ``(n_mels, 126)`` for 4 s at 16 kHz with librosa's
    default ``n_fft=2048`` / ``hop_length=512`` / ``center=True``.
    """
    target_length = target_seconds * sample_rate

    y, sr = librosa.load(str(file_path), sr=sample_rate)
    y_fixed = librosa.util.fix_length(data=y, size=target_length)
    mel_spec = librosa.feature.melspectrogram(y=y_fixed, sr=sr, n_mels=n_mels)
    return librosa.power_to_db(mel_spec, ref=np.max)


def save_mel_spectrogram_png(
    spec_db: np.ndarray,
    output_path: str | Path,
    cmap: str = COLORMAP,
) -> Path:
    """Render ``spec_db`` to an RGBA PNG exactly as the dataset export did."""
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(output_path, spec_db, cmap=cmap, origin="lower")
    return output_path


def render_mel_spectrogram_png_bytes(
    spec_db: np.ndarray,
    cmap: str = COLORMAP,
) -> bytes:
    """Render ``spec_db`` to PNG bytes without touching the filesystem."""
    import matplotlib.pyplot as plt

    buffer = BytesIO()
    plt.imsave(buffer, spec_db, cmap=cmap, origin="lower", format="png")
    return buffer.getvalue()
