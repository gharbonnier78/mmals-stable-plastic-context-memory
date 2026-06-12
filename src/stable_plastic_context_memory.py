"""Reference interface for the proposed MMALS stable-plastic context memory.

This module is intentionally decoupled from any dataset name or image encoder.
It accepts branch probabilities and returns a fused context posterior plus an
auditable decision record. It is a design reference, not a trained production
component.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


_EPS = 1e-8


def _normalize_probabilities(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 2:
        raise ValueError(f"Expected [batch, contexts], got shape={tuple(x.shape)}")
    if torch.any(x < 0):
        raise ValueError("Probabilities must be non-negative")
    return x / x.sum(dim=-1, keepdim=True).clamp_min(_EPS)


def _entropy(p: torch.Tensor) -> torch.Tensor:
    p = p.clamp_min(_EPS)
    return -(p * p.log()).sum(dim=-1, keepdim=True)


@dataclass(frozen=True)
class ContextMemoryConfig:
    mode: str = "legacy"
    legacy_single_weight: float = 0.05
    stable_weight: float = 0.50
    plastic_weight: float = 0.40
    single_weight: float = 0.10
    metadata_dim: int = 6
    adaptive_hidden_dim: int = 16

    def validate(self) -> None:
        if self.mode not in {"legacy", "stable_plastic", "adaptive"}:
            raise ValueError(f"Unsupported mode: {self.mode}")
        for name in (
            "legacy_single_weight",
            "stable_weight",
            "plastic_weight",
            "single_weight",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        fixed_sum = self.stable_weight + self.plastic_weight + self.single_weight
        if abs(fixed_sum - 1.0) > 1e-6:
            raise ValueError("stable_weight + plastic_weight + single_weight must equal 1")


class StablePlasticContextMemory(nn.Module):
    """Backward-compatible context-posterior fusion.

    Inputs are already-normalized or non-negative branch scores. The module
    normalizes them internally. In legacy mode the historical two-branch blend
    is reproduced exactly, apart from the final normalization required if the
    caller supplies non-normalized scores.
    """

    def __init__(self, config: ContextMemoryConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.adaptive_gate = nn.Sequential(
            nn.Linear(config.metadata_dim, config.adaptive_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.adaptive_hidden_dim, 3),
        )

    def forward(
        self,
        *,
        q_single: torch.Tensor,
        q_legacy_prototype: Optional[torch.Tensor] = None,
        q_stable: Optional[torch.Tensor] = None,
        q_plastic: Optional[torch.Tensor] = None,
        metadata: Optional[torch.Tensor] = None,
        mode: Optional[str] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor | str]]:
        active_mode = mode or self.config.mode
        if active_mode not in {"legacy", "stable_plastic", "adaptive"}:
            raise ValueError(f"Unsupported mode: {active_mode}")

        q_single = _normalize_probabilities(q_single)

        if active_mode == "legacy":
            if q_legacy_prototype is None:
                raise ValueError("legacy mode requires q_legacy_prototype")
            q_legacy_prototype = _normalize_probabilities(q_legacy_prototype)
            if q_legacy_prototype.shape != q_single.shape:
                raise ValueError("Legacy branch shapes must match")
            w = float(self.config.legacy_single_weight)
            posterior = w * q_single + (1.0 - w) * q_legacy_prototype
            posterior = _normalize_probabilities(posterior)
            weights = posterior.new_tensor([w, 0.0, 1.0 - w]).expand(len(posterior), -1)
            audit: Dict[str, torch.Tensor | str] = {
                "mode": active_mode,
                "fusion_weights_single_plastic_stable": weights,
                "branch_disagreement": (q_single - q_legacy_prototype).abs().mean(dim=-1),
                "posterior_entropy": _entropy(posterior).squeeze(-1),
            }
            return posterior, audit

        if q_stable is None or q_plastic is None:
            raise ValueError(f"{active_mode} mode requires q_stable and q_plastic")
        q_stable = _normalize_probabilities(q_stable)
        q_plastic = _normalize_probabilities(q_plastic)
        if not (q_stable.shape == q_plastic.shape == q_single.shape):
            raise ValueError("All branch shapes must match")

        if active_mode == "stable_plastic":
            weights = q_single.new_tensor(
                [self.config.single_weight, self.config.plastic_weight, self.config.stable_weight]
            ).expand(len(q_single), -1)
        else:
            if metadata is None:
                # Dataset-agnostic fallback metadata based only on branch behavior.
                metadata = torch.cat(
                    [
                        _entropy(q_single),
                        _entropy(q_plastic),
                        _entropy(q_stable),
                        (q_stable - q_plastic).abs().mean(dim=-1, keepdim=True),
                        q_stable.max(dim=-1, keepdim=True).values,
                        q_plastic.max(dim=-1, keepdim=True).values,
                    ],
                    dim=-1,
                )
            if metadata.ndim != 2 or metadata.shape[0] != q_single.shape[0]:
                raise ValueError("metadata must be [batch, metadata_dim]")
            if metadata.shape[1] != self.config.metadata_dim:
                raise ValueError(
                    f"metadata feature count must be {self.config.metadata_dim}, got {metadata.shape[1]}"
                )
            weights = torch.softmax(self.adaptive_gate(metadata), dim=-1)

        posterior = (
            weights[:, 0:1] * q_single
            + weights[:, 1:2] * q_plastic
            + weights[:, 2:3] * q_stable
        )
        posterior = _normalize_probabilities(posterior)
        audit = {
            "mode": active_mode,
            "fusion_weights_single_plastic_stable": weights,
            "stable_plastic_disagreement": (q_stable - q_plastic).abs().mean(dim=-1),
            "posterior_entropy": _entropy(posterior).squeeze(-1),
            "single_winner": q_single.argmax(dim=-1),
            "plastic_winner": q_plastic.argmax(dim=-1),
            "stable_winner": q_stable.argmax(dim=-1),
            "deployed_winner": posterior.argmax(dim=-1),
        }
        return posterior, audit
