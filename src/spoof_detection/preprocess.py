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
from typing import TYPE_CHECKING

import librosa
import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from torchvision.transforms import Compose

SAMPLE_RATE = 16_000
TARGET_SECONDS = 4
TARGET_LENGTH = SAMPLE_RATE * TARGET_SECONDS
N_MELS = 128
COLORMAP = "magma"

# Version string recorded in evaluation reports.  Bump only with a documented
# ablation that changes the audio-to-image contract (AGENTS.md G3).
PREPROCESSING_VERSION = "mel128-magma-4s-v1"

# Model input contract: the 126 x 128 rendered image is resized to 224 x 224,
# converted to a tensor, and ImageNet-normalised.
INFERENCE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


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


def load_mel_image(file_path: str | Path) -> Image.Image:
    """Render a source audio file to the RGB image the baseline sees.

    This uses the same renderer as dataset export, so the result is identical
    to reading the exported PNG.  It is slower than reading the PNG because it
    decodes audio and recomputes the mel spectrogram.
    """
    spec_db = generate_mel_spectrogram(file_path)
    png_bytes = render_mel_spectrogram_png_bytes(spec_db)
    with Image.open(BytesIO(png_bytes)) as image:
        image.load()
        return image.convert("RGB")


def build_inference_transform() -> "Compose":
    """Return the exact image transform used at inference time."""
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize((INFERENCE_SIZE, INFERENCE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=list(IMAGENET_MEAN),
                std=list(IMAGENET_STD),
            ),
        ]
    )
