"""Rendering-condition registry for the E1.2 colormap/orientation ablation.

E1.2 holds the 128-band mel representation and the frozen-backbone training
recipe fixed and changes only how the dB matrix is turned into a PNG:

* **colormap** - ``magma`` (the existing contract), ``viridis``, or ``gray``;
* **orientation** - ``origin='lower'`` (the existing contract, low frequency at
  the bottom of the PNG) or ``origin='upper'`` (no flip, low frequency at the
  top);
* **dB scaling** - per-clip min-max (existing contract) or a fixed ``[-80, 0]``
  dB range with no per-clip contrast stretch;
* **PNG encoding** - RGBA (the existing export) or RGB.

The fixed-range raw-dB condition still uses ``power_to_db(ref=np.max)`` because
that belongs to the fixed mel contract; it removes only the per-clip min-max
stretch.  It therefore tests contrast normalisation, not absolute recording
gain.  A true absolute-gain probe would require ``ref=1.0`` and is out of scope.

``magma_flip`` must stay byte/pixel-compatible with the E1.1 ``mel128`` export
and ``data/dataset_img``; it delegates to :func:`spoof_detection.preprocess.save_mel_spectrogram_png`
verbatim.  Do not change the default rendering path without a documented
ablation (AGENTS.md G3).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .preprocess import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    INFERENCE_SIZE,
    N_MELS,
    PREPROCESSING_VERSION,
    SAMPLE_RATE,
    TARGET_LENGTH,
    TARGET_SECONDS,
    colorize_mel,
    normalize_mel_db,
    save_mel_spectrogram_png,
)

# Fixed dB window for the raw-dB condition.  The mel contract already uses
# power_to_db(ref=np.max), so 0 dB is the clip peak and -80 dB is the default
# librosa dynamic-range floor.
RAW_DB_RANGE = (-80.0, 0.0)

DEFAULT_ORIGIN = "lower"


@dataclass(frozen=True)
class RenderingSpec:
    """Metadata for one rendering condition."""

    name: str
    colormap: str
    flip: bool
    encoding: str
    normalization: str
    db_range: tuple[float, float] | None
    version: str
    description: str


RENDER_CONDITION_ORDER = (
    "magma_flip",
    "viridis_flip",
    "gray_flip",
    "rawdb_flip",
    "magma_noflip",
    "viridis_noflip",
    "gray_noflip",
    "magma_flip_rgb",
)

RENDER_CONDITIONS: dict[str, RenderingSpec] = {
    "magma_flip": RenderingSpec(
        name="magma_flip",
        colormap="magma",
        flip=True,
        encoding="rgba",
        normalization="per_clip_minmax",
        db_range=None,
        version=PREPROCESSING_VERSION,
        description="Existing 128-band mel contract (control: magma, flipped, RGBA)",
    ),
    "viridis_flip": RenderingSpec(
        name="viridis_flip",
        colormap="viridis",
        flip=True,
        encoding="rgba",
        normalization="per_clip_minmax",
        db_range=None,
        version="mel128-viridis-flip-4s-v1",
        description="128-band mel, viridis, flipped, RGBA",
    ),
    "gray_flip": RenderingSpec(
        name="gray_flip",
        colormap="gray",
        flip=True,
        encoding="rgba",
        normalization="per_clip_minmax",
        db_range=None,
        version="mel128-gray-flip-4s-v1",
        description="128-band mel, grayscale, flipped, RGBA",
    ),
    "rawdb_flip": RenderingSpec(
        name="rawdb_flip",
        colormap="gray",
        flip=True,
        encoding="rgba",
        normalization="fixed_db_range",
        db_range=RAW_DB_RANGE,
        version="mel128-gray-fixed80-flip-4s-v1",
        description=(
            "128-band mel, grayscale, flipped, RGBA, fixed [-80, 0] dB range "
            "with no per-clip min-max stretch"
        ),
    ),
    "magma_noflip": RenderingSpec(
        name="magma_noflip",
        colormap="magma",
        flip=False,
        encoding="rgba",
        normalization="per_clip_minmax",
        db_range=None,
        version="mel128-magma-noflip-4s-v1",
        description="128-band mel, magma, unflipped, RGBA",
    ),
    "viridis_noflip": RenderingSpec(
        name="viridis_noflip",
        colormap="viridis",
        flip=False,
        encoding="rgba",
        normalization="per_clip_minmax",
        db_range=None,
        version="mel128-viridis-noflip-4s-v1",
        description="128-band mel, viridis, unflipped, RGBA",
    ),
    "gray_noflip": RenderingSpec(
        name="gray_noflip",
        colormap="gray",
        flip=False,
        encoding="rgba",
        normalization="per_clip_minmax",
        db_range=None,
        version="mel128-gray-noflip-4s-v1",
        description="128-band mel, grayscale, unflipped, RGBA",
    ),
    "magma_flip_rgb": RenderingSpec(
        name="magma_flip_rgb",
        colormap="magma",
        flip=True,
        encoding="rgb",
        normalization="per_clip_minmax",
        db_range=None,
        version="mel128-magma-flip-rgbenc-4s-v1",
        description=(
            "Existing magma/flip rendering saved as RGB instead of RGBA "
            "(encoding control; pixels must match magma_flip)"
        ),
    ),
}


def get_rendering_spec(name: str) -> RenderingSpec:
    """Return the spec for ``name`` or raise a helpful error."""
    try:
        return RENDER_CONDITIONS[name]
    except KeyError:
        raise ValueError(
            f"unknown rendering condition {name!r}; "
            f"expected one of {list(RENDER_CONDITION_ORDER)}"
        ) from None


def origin_for(spec: RenderingSpec) -> str:
    """Return the matplotlib ``origin`` for ``spec``."""
    return DEFAULT_ORIGIN if spec.flip else "upper"


def _normalized(spec: RenderingSpec, spec_db: np.ndarray) -> np.ndarray:
    """Return the ``[0, 1]`` normalised matrix for a rendering condition."""
    values = np.asarray(spec_db)
    if spec.normalization == "per_clip_minmax":
        return normalize_mel_db(values)
    if spec.normalization == "fixed_db_range":
        low, high = spec.db_range  # type: ignore[misc]
        return np.clip((values - low) / (high - low), 0.0, 1.0)
    raise ValueError(
        f"unsupported normalization {spec.normalization!r} for {spec.name}"
    )


def render_rendering_rgb(name: str, spec_db: np.ndarray) -> np.ndarray:
    """Return the ``uint8`` RGB image a saved condition decodes to.

    This mirrors the PNG writers exactly: matplotlib's autoscaled ``imsave`` is
    equivalent to a per-clip min-max normalisation, and the fixed-range
    condition passes explicit ``vmin=0`` / ``vmax=1`` so it cannot autoscale.
    """
    spec = get_rendering_spec(name)
    return colorize_mel(
        _normalized(spec, spec_db),
        cmap=spec.colormap,
        flip=spec.flip,
    )


def save_rendering_png(
    name: str,
    spec_db: np.ndarray,
    output_path: str | Path,
) -> Path:
    """Render one dB matrix under condition ``name`` and save it as a PNG."""
    spec = get_rendering_spec(name)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    values = np.asarray(spec_db)
    origin = origin_for(spec)
    if spec.encoding == "rgba":
        if spec.normalization == "per_clip_minmax":
            # Exactly the existing export path: plt.imsave autoscales the dB
            # matrix per image, which is the control contract.
            return save_mel_spectrogram_png(
                values,
                output_path,
                cmap=spec.colormap,
                origin=origin,
            )
        return save_mel_spectrogram_png(
            _normalized(spec, values),
            output_path,
            cmap=spec.colormap,
            origin=origin,
            vmin=0.0,
            vmax=1.0,
        )

    if spec.encoding == "rgb":
        Image.fromarray(
            render_rendering_rgb(name, values),
            mode="RGB",
        ).save(output_path)
        return output_path

    raise ValueError(
        f"unsupported encoding {spec.encoding!r} for {spec.name}"
    )


def rendering_payload(name: str) -> dict[str, object]:
    """Return the JSON-serialisable provenance block for ``name``."""
    spec = get_rendering_spec(name)
    return {
        "version": spec.version,
        "condition": spec.name,
        "description": spec.description,
        "representation": "mel128",
        "representation_version": PREPROCESSING_VERSION,
        "params": {
            "n_mels": N_MELS,
            "n_fft": 2048,
            "hop_length": 512,
            "power_to_db_ref": "np.max",
        },
        "colormap": spec.colormap,
        "orientation": origin_for(spec),
        "flip": spec.flip,
        "encoding": spec.encoding.upper(),
        "normalization": spec.normalization,
        "db_range": (
            list(spec.db_range) if spec.db_range is not None else None
        ),
        "sample_rate": SAMPLE_RATE,
        "target_seconds": TARGET_SECONDS,
        "target_length": TARGET_LENGTH,
        "resize": [INFERENCE_SIZE, INFERENCE_SIZE],
        "imagenet_mean": list(IMAGENET_MEAN),
        "imagenet_std": list(IMAGENET_STD),
    }
