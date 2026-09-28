"""Multimodal feature encoder.

Ref: Sec. III-A (the multimodal evidence record x_p), Sec. IV-E (modality
ablation: clinical variables alone 0.702, + molecular 0.741, + histopathology
0.756, full multimodal 0.765).

Two details matter for the ablation. The modality mask is applied before
pooling, so withholding a modality removes it rather than feeding zeros into a
learned projection; and a modality that is absent for some records is dropped
from the mean over present modalities instead of being counted as an observation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from pathanchor.cohort.schema import (
    FEATURE_WIDTH_CLINICAL,
    FEATURE_WIDTH_HISTOLOGY,
    FEATURE_WIDTH_IMAGE,
    FEATURE_WIDTH_MOLECULAR,
)
from pathanchor.utils.torch_helpers import reset_linear

MODALITY_SLICES: dict[str, tuple[int, int]] = {
    "clinical": (0, FEATURE_WIDTH_CLINICAL),
    "molecular": (FEATURE_WIDTH_CLINICAL, FEATURE_WIDTH_CLINICAL + FEATURE_WIDTH_MOLECULAR),
    "histology": (
        FEATURE_WIDTH_CLINICAL + FEATURE_WIDTH_MOLECULAR,
        FEATURE_WIDTH_CLINICAL + FEATURE_WIDTH_MOLECULAR + FEATURE_WIDTH_HISTOLOGY,
    ),
    "image": (
        FEATURE_WIDTH_CLINICAL + FEATURE_WIDTH_MOLECULAR + FEATURE_WIDTH_HISTOLOGY,
        FEATURE_WIDTH_CLINICAL
        + FEATURE_WIDTH_MOLECULAR
        + FEATURE_WIDTH_HISTOLOGY
        + FEATURE_WIDTH_IMAGE,
    ),
}

MODALITY_ORDER: tuple[str, ...] = ("clinical", "molecular", "histology", "image")

FEATURE_DIM: int = sum(high - low for low, high in MODALITY_SLICES.values())


@dataclass(frozen=True, slots=True)
class EncoderConfig:
    feature_dim: int = FEATURE_DIM
    hidden: int = 64
    blocks: int = 2
    dropout: float = 0.1


def modality_mask(
    present: Tensor,
) -> Tensor:
    """Expand a per-modality presence matrix to per-feature presence.

    ``present`` has shape (B, 4) in MODALITY_ORDER order.
    """
    parts: list[Tensor] = []
    for position, name in enumerate(MODALITY_ORDER):
        low, high = MODALITY_SLICES[name]
        width = high - low
        parts.append(present[:, position : position + 1].expand(-1, width))
    return torch.cat(parts, dim=-1)


class ModalityBlock(nn.Module):
    """One residual widening block over the concatenated feature vector."""

    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.projection = nn.Linear(width, width)
        self.gate = nn.Linear(width, width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: Tensor) -> Tensor:
        hidden = self.norm(values)
        hidden = torch.nn.functional.gelu(self.projection(hidden)) * torch.sigmoid(
            self.gate(hidden)
        )
        residual: Tensor = values + self.dropout(hidden)
        return residual


class MultimodalEncoder(nn.Module):
    """Projects an evidence record to the conditioning embedding z_p."""

    def __init__(self, config: EncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.input_projection = nn.Linear(config.feature_dim, config.hidden)
        self.blocks = nn.ModuleList(
            [ModalityBlock(config.hidden, config.dropout) for _ in range(config.blocks)]
        )
        self.output_norm = nn.LayerNorm(config.hidden)
        reset_linear(self.input_projection)

    def forward(self, features: Tensor, mask: Tensor) -> Tensor:
        expanded = mask if mask.shape[-1] == features.shape[-1] else modality_mask(mask)
        masked = features * expanded
        hidden = torch.tanh(self.input_projection(masked))
        for block in self.blocks:
            hidden = block(hidden)
        encoded: Tensor = self.output_norm(hidden)
        return encoded

    def pooled_modalities(self, features: Tensor, mask: Tensor) -> Tensor:
        """Per-modality means over the features that are present.

        Used by the modality ablation so a withheld modality contributes no
        channel rather than a zero channel.
        """
        expanded = mask if mask.shape[-1] == features.shape[-1] else modality_mask(mask)
        means: list[Tensor] = []
        for name in MODALITY_ORDER:
            low, high = MODALITY_SLICES[name]
            block = features[:, low:high] * expanded[:, low:high]
            width = torch.clamp(expanded[:, low:high].sum(dim=1, keepdim=True), min=1.0)
            means.append(block.sum(dim=1, keepdim=True) / width)
        stacked = torch.cat(means, dim=1)
        present = torch.stack(
            [
                (
                    expanded[:, MODALITY_SLICES[name][0] : MODALITY_SLICES[name][1]].sum(dim=1) > 0
                ).to(stacked.dtype)
                for name in MODALITY_ORDER
            ],
            dim=1,
        )
        denominator = torch.clamp(present.sum(dim=1, keepdim=True), min=1.0)
        pooled: Tensor = (stacked * present).sum(dim=1, keepdim=True) / denominator
        return pooled


class ResponseHead(nn.Module):
    """Maps z_p to a response logit."""

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.hidden = nn.Linear(hidden, hidden // 2)
        self.output = nn.Linear(hidden // 2, 1)
        reset_linear(self.hidden)
        reset_linear(self.output)

    def forward(self, encoded: Tensor) -> Tensor:
        activated = torch.tanh(self.hidden(encoded))
        logit: Tensor = self.output(activated).squeeze(-1)
        return logit
