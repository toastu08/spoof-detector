"""Model definition and strict checkpoint loading for the baseline detector.

The first-iteration checkpoints store parameters under a ``network.`` prefix,
while the model exposes the ResNet18 backbone as ``backbone.``.  Loading with
``strict=False`` can silently leave the classifier head randomly initialised
(AGENTS.md G7), so this module diffs the remapped state dict against the model
before loading and raises on any mismatch.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import torch
import torch.nn as nn
from torchvision.models import resnet18

HEAD_PARAMETERS = 65_793
# Parameter elements only.  The full checkpoint holds 9,620 additional buffer
# elements (batch-norm running statistics and ``num_batches_tracked``), which is
# why the tensor total is 11,251,925.
TOTAL_PARAMETERS = 11_242_305


class CheckpointError(RuntimeError):
    """Raised when a checkpoint does not match the model exactly."""


@dataclass(frozen=True)
class CheckpointLoadReport:
    """Everything a reviewer needs to know about a checkpoint load."""

    path: str
    sha256: str
    state_dict_keys: int
    remapped_keys: int
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]
    total_parameters: int
    backbone_parameters: int
    head_parameters: int
    head_state_sha256: str
    head_last_bias: float
    batch_norm_steps: tuple[int, ...]


class AudioSpoofDetector(nn.Module):
    """Frozen ImageNet ResNet18 with the first-iteration linear head."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = resnet18(weights=None)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 hex digest of ``path``."""
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def remap_state_dict_keys(
    state_dict: Mapping[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], int]:
    """Rename checkpoint ``network.*`` keys to model ``backbone.*`` keys."""
    remapped: dict[str, torch.Tensor] = {}
    remapped_count = 0
    for key, value in state_dict.items():
        if key.startswith("network."):
            remapped[f"backbone.{key[len('network.'):]}"] = value
            remapped_count += 1
        else:
            remapped[key] = value
    return remapped, remapped_count


def _state_dict_fingerprint(
    state_dict: Mapping[str, torch.Tensor],
    prefixes: tuple[str, ...],
) -> str:
    """Return a stable SHA-256 over the selected tensors and their key names."""
    hasher = hashlib.sha256()
    for key in sorted(state_dict):
        if not key.startswith(prefixes):
            continue
        tensor = state_dict[key].detach().cpu().contiguous()
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.float()
        hasher.update(key.encode("utf-8"))
        hasher.update(str(tensor.dtype).encode("utf-8"))
        hasher.update(str(tuple(tensor.shape)).encode("utf-8"))
        hasher.update(tensor.numpy().tobytes())
    return hasher.hexdigest()


def _batch_norm_steps(
    state_dict: Mapping[str, torch.Tensor],
) -> tuple[int, ...]:
    """Return the unique ``num_batches_tracked`` values in the checkpoint."""
    return tuple(
        sorted(
            {
                int(value.reshape(-1)[0].item())
                for key, value in state_dict.items()
                if key.endswith(".num_batches_tracked")
            }
        )
    )


def _head_last_bias(state_dict: Mapping[str, torch.Tensor]) -> float:
    """Return the final head bias, a cheap checkpoint-trained sanity signal."""
    key = "backbone.fc.3.bias"
    if key not in state_dict:
        return float("nan")
    return float(state_dict[key].detach().cpu().reshape(-1)[0].item())


def _extract_state_dict(raw: object) -> Mapping[str, torch.Tensor]:
    """Return a state dict from the common checkpoint wrapper shapes."""
    if isinstance(raw, nn.Module):
        return raw.state_dict()
    if not isinstance(raw, Mapping):
        raise CheckpointError(
            f"checkpoint must contain a state dict, found {type(raw).__name__}"
        )
    for wrapper in ("state_dict", "model_state_dict"):
        nested = raw.get(wrapper)
        if isinstance(nested, Mapping):
            return nested  # type: ignore[return-value]
    return raw  # type: ignore[return-value]


def parameter_counts(model: nn.Module) -> tuple[int, int, int]:
    """Return ``(total, backbone, head)`` parameter counts."""
    total = sum(parameter.numel() for parameter in model.parameters())
    head = sum(
        parameter.numel() for parameter in model.backbone.fc.parameters()
    )
    return total, total - head, head


def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[AudioSpoofDetector, CheckpointLoadReport]:
    """Load a checkpoint with a verified, exact key mapping.

    Raises :class:`CheckpointError` if any model parameter is missing from the
    checkpoint or any checkpoint key is unexpected.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    try:
        raw = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        # Older PyTorch versions do not accept ``weights_only``.
        raw = torch.load(checkpoint_path, map_location="cpu")

    raw_state_dict = _extract_state_dict(raw)
    state_dict, remapped_count = remap_state_dict_keys(raw_state_dict)
    head_state_sha256 = _state_dict_fingerprint(state_dict, ("backbone.fc.",))
    batch_norm_steps = _batch_norm_steps(state_dict)
    head_last_bias = _head_last_bias(state_dict)

    model = AudioSpoofDetector()
    model_keys = set(model.state_dict().keys())
    checkpoint_keys = set(state_dict.keys())
    missing = tuple(sorted(model_keys - checkpoint_keys))
    unexpected = tuple(sorted(checkpoint_keys - model_keys))

    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing keys ({len(missing)}): {list(missing[:5])}")
        if unexpected:
            details.append(
                f"unexpected keys ({len(unexpected)}): {list(unexpected[:5])}"
            )
        raise CheckpointError(
            f"checkpoint does not match {type(model).__name__}: "
            + "; ".join(details)
        )

    model.load_state_dict(state_dict, strict=True)
    total, backbone, head = parameter_counts(model)
    model.to(device)
    model.eval()
    model.requires_grad_(False)

    report = CheckpointLoadReport(
        path=str(checkpoint_path),
        sha256=sha256_file(checkpoint_path),
        state_dict_keys=len(state_dict),
        remapped_keys=remapped_count,
        missing_keys=missing,
        unexpected_keys=unexpected,
        total_parameters=total,
        backbone_parameters=backbone,
        head_parameters=head,
        head_state_sha256=head_state_sha256,
        head_last_bias=head_last_bias,
        batch_norm_steps=batch_norm_steps,
    )
    return model, report
