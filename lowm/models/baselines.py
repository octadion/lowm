"""Baseline energy models for LOWM-Synth ranking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import nn


@dataclass(frozen=True)
class BaselineConfig:
    object_dim: int = 7
    action_dim: int = 2
    hidden_dim: int = 128
    token_dim: int = 128
    context_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.0


def baseline_config_from_mapping(config: Mapping[str, Any]) -> BaselineConfig:
    model = dict(config.get("model", {}))
    return BaselineConfig(
        object_dim=int(model.get("object_dim", BaselineConfig.object_dim)),
        action_dim=int(model.get("action_dim", BaselineConfig.action_dim)),
        hidden_dim=int(model.get("hidden_dim", BaselineConfig.hidden_dim)),
        token_dim=int(model.get("token_dim", BaselineConfig.token_dim)),
        context_dim=int(model.get("context_dim", BaselineConfig.context_dim)),
        num_layers=int(model.get("num_layers", BaselineConfig.num_layers)),
        dropout=float(model.get("dropout", BaselineConfig.dropout)),
    )


def _mlp(input_dim: int, hidden_dim: int, output_dim: int, num_layers: int, dropout: float = 0.0) -> nn.Sequential:
    if num_layers < 1:
        raise ValueError("num_layers must be >= 1")
    layers: list[nn.Module] = []
    dim = input_dim
    for _ in range(max(0, num_layers - 1)):
        layers.append(nn.Linear(dim, hidden_dim))
        layers.append(nn.SiLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        dim = hidden_dim
    layers.append(nn.Linear(dim, output_dim))
    return nn.Sequential(*layers)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor, dims: tuple[int, ...]) -> torch.Tensor:
    mask = mask.to(dtype=values.dtype)
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(-1)
    weighted = values * mask
    denom = mask.sum(dim=dims).clamp_min(1.0)
    return weighted.sum(dim=dims) / denom


def candidate_transition_tokens(
    cand_states: torch.Tensor,
    cand_actions: torch.Tensor,
    cand_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build per-object transition tokens.

    Inputs:
    cand_states: [B, M, H+1, N, D]
    cand_actions: [B, M, H, N, 2]
    cand_mask: [B, M, H+1, N]

    Returns:
    tokens: [B, M, H, N, 23]
    token_mask: [B, M, H, N]
    """

    s_t = cand_states[:, :, :-1]
    s_tp1 = cand_states[:, :, 1:]
    delta = s_tp1 - s_t
    token_mask = cand_mask[:, :, :-1] * cand_mask[:, :, 1:]
    tokens = torch.cat([s_t, cand_actions, s_tp1, delta], dim=-1)
    return tokens, token_mask


def context_transition_tokens(
    context_states: torch.Tensor,
    context_actions: torch.Tensor,
    context_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build context transition tokens.

    Inputs:
    context_states: [B, K, 2, N, D]
    context_actions: [B, K, N, 2]
    context_mask: [B, K, 2, N]

    Returns:
    tokens: [B, K, N, 23]
    token_mask: [B, K, N]
    """

    s_t = context_states[:, :, 0]
    s_tp1 = context_states[:, :, 1]
    delta = s_tp1 - s_t
    token_mask = context_mask[:, :, 0] * context_mask[:, :, 1]
    tokens = torch.cat([s_t, context_actions, s_tp1, delta], dim=-1)
    return tokens, token_mask


class CandidateEncoder(nn.Module):
    def __init__(self, cfg: BaselineConfig) -> None:
        super().__init__()
        token_input_dim = cfg.object_dim * 3 + cfg.action_dim
        self.token_mlp = _mlp(token_input_dim, cfg.hidden_dim, cfg.token_dim, cfg.num_layers, cfg.dropout)

    def forward(self, cand_states: torch.Tensor, cand_actions: torch.Tensor, cand_mask: torch.Tensor) -> torch.Tensor:
        tokens, token_mask = candidate_transition_tokens(cand_states, cand_actions, cand_mask)
        encoded = self.token_mlp(tokens)
        return _masked_mean(encoded, token_mask, dims=(2, 3))


class ContextEncoder(nn.Module):
    def __init__(self, cfg: BaselineConfig) -> None:
        super().__init__()
        token_input_dim = cfg.object_dim * 3 + cfg.action_dim
        self.token_mlp = _mlp(token_input_dim, cfg.hidden_dim, cfg.context_dim, cfg.num_layers, cfg.dropout)

    def forward(self, context_states: torch.Tensor, context_actions: torch.Tensor, context_mask: torch.Tensor) -> torch.Tensor:
        tokens, token_mask = context_transition_tokens(context_states, context_actions, context_mask)
        encoded = self.token_mlp(tokens)
        return _masked_mean(encoded, token_mask, dims=(1, 2))


class FixedEnergyModel(nn.Module):
    """Energy baseline without context or latent operator."""

    def __init__(self, cfg: BaselineConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or BaselineConfig()
        self.candidate_encoder = CandidateEncoder(self.cfg)
        self.energy_head = _mlp(self.cfg.token_dim, self.cfg.hidden_dim, 1, self.cfg.num_layers, self.cfg.dropout)

    def forward(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        candidate_embed = self.candidate_encoder(batch["cand_states"], batch["cand_actions"], batch["cand_mask"])
        return self.energy_head(candidate_embed).squeeze(-1)


class DirectContextEnergyModel(nn.Module):
    """Strong baseline that scores candidates directly from context C."""

    def __init__(self, cfg: BaselineConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or BaselineConfig()
        self.candidate_encoder = CandidateEncoder(self.cfg)
        self.context_encoder = ContextEncoder(self.cfg)
        head_input_dim = self.cfg.token_dim + self.cfg.context_dim
        self.energy_head = _mlp(head_input_dim, self.cfg.hidden_dim, 1, self.cfg.num_layers, self.cfg.dropout)

    def forward(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        candidate_embed = self.candidate_encoder(batch["cand_states"], batch["cand_actions"], batch["cand_mask"])
        context_embed = self.context_encoder(batch["context_states"], batch["context_actions"], batch["context_mask"])
        context_embed = context_embed[:, None, :].expand(-1, candidate_embed.shape[1], -1)
        return self.energy_head(torch.cat([candidate_embed, context_embed], dim=-1)).squeeze(-1)


def build_baseline(name: str, cfg: BaselineConfig | None = None) -> nn.Module:
    normalized = name.strip().lower().replace("-", "_")
    if normalized in {"fixed", "fixed_energy"}:
        return FixedEnergyModel(cfg)
    if normalized in {"context", "direct_context", "direct_context_energy"}:
        return DirectContextEnergyModel(cfg)
    raise ValueError(f"unknown baseline '{name}'")
