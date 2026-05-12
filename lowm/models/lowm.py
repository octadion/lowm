"""Latent Operator World Model (LOWM) v0."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import nn


@dataclass(frozen=True)
class LOWMConfig:
    object_dim: int = 7
    action_dim: int = 2
    lambda_dim: int = 16
    hidden_dim: int = 128
    context_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.0
    use_pairwise_energy: bool = True
    use_mu_eval: bool = True
    energy_reduction: str = "sum"


def lowm_config_from_mapping(config: Mapping[str, Any]) -> LOWMConfig:
    model = dict(config.get("model", {}))
    lambda_dim = int(model.get("lambda_dim", LOWMConfig.lambda_dim))
    if lambda_dim not in {4, 8, 16, 32}:
        raise ValueError("lambda_dim must be one of {4, 8, 16, 32}")
    return LOWMConfig(
        object_dim=int(model.get("object_dim", LOWMConfig.object_dim)),
        action_dim=int(model.get("action_dim", LOWMConfig.action_dim)),
        lambda_dim=lambda_dim,
        hidden_dim=int(model.get("hidden_dim", LOWMConfig.hidden_dim)),
        context_dim=int(model.get("context_dim", LOWMConfig.context_dim)),
        num_layers=int(model.get("num_layers", LOWMConfig.num_layers)),
        dropout=float(model.get("dropout", LOWMConfig.dropout)),
        use_pairwise_energy=bool(model.get("use_pairwise_energy", LOWMConfig.use_pairwise_energy)),
        use_mu_eval=bool(model.get("use_mu_eval", LOWMConfig.use_mu_eval)),
        energy_reduction=str(model.get("energy_reduction", LOWMConfig.energy_reduction)),
    )


def _mlp(input_dim: int, hidden_dim: int, output_dim: int, num_layers: int, dropout: float) -> nn.Sequential:
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
    mask = mask.to(values.dtype)
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(-1)
    denom = mask.sum(dim=dims).clamp_min(1.0)
    return (values * mask).sum(dim=dims) / denom


class DeepSetsContextEncoder(nn.Module):
    """DeepSets encoder for context transitions C."""

    def __init__(self, cfg: LOWMConfig) -> None:
        super().__init__()
        token_dim = cfg.object_dim * 3 + cfg.action_dim
        self.token_mlp = _mlp(token_dim, cfg.hidden_dim, cfg.context_dim, cfg.num_layers, cfg.dropout)
        self.mu_head = nn.Linear(cfg.context_dim, cfg.lambda_dim)
        self.logvar_head = nn.Linear(cfg.context_dim, cfg.lambda_dim)

    def forward(
        self,
        context_states: torch.Tensor,
        context_actions: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        s_t = context_states[:, :, 0]
        s_next = context_states[:, :, 1]
        delta = s_next - s_t
        token_mask = context_mask[:, :, 0] * context_mask[:, :, 1]
        tokens = torch.cat([s_t, context_actions, s_next, delta], dim=-1)
        encoded = self.token_mlp(tokens)
        pooled = _masked_mean(encoded, token_mask, dims=(1, 2))
        mu = self.mu_head(pooled)
        logvar = self.logvar_head(pooled).clamp(min=-8.0, max=8.0)
        return mu, logvar


class ObjectCentricEnergy(nn.Module):
    """Self and relation energy for candidate trajectories under lambda."""

    def __init__(self, cfg: LOWMConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.self_mlp = _mlp(
            cfg.object_dim * 2 + cfg.action_dim + cfg.lambda_dim,
            cfg.hidden_dim,
            1,
            cfg.num_layers,
            cfg.dropout,
        )
        self.rel_mlp = _mlp(
            cfg.object_dim * 4 + cfg.lambda_dim,
            cfg.hidden_dim,
            1,
            cfg.num_layers,
            cfg.dropout,
        )

    def forward(
        self,
        cand_states: torch.Tensor,
        cand_actions: torch.Tensor,
        cand_mask: torch.Tensor,
        lambda_sample: torch.Tensor,
    ) -> torch.Tensor:
        if cand_states.ndim != 5:
            raise ValueError(f"cand_states must be [B,M,H+1,N,D], got {tuple(cand_states.shape)}")
        bsz, num_candidates, horizon_plus_one, num_objects, _ = cand_states.shape
        horizon = horizon_plus_one - 1
        s_t = cand_states[:, :, :-1]
        s_next = cand_states[:, :, 1:]
        transition_mask = cand_mask[:, :, :-1] * cand_mask[:, :, 1:]

        lam_self = lambda_sample[:, None, None, None, :].expand(bsz, num_candidates, horizon, num_objects, -1)
        self_input = torch.cat([s_t, cand_actions, s_next, lam_self], dim=-1)
        self_energy = self.self_mlp(self_input).squeeze(-1) * transition_mask
        energy = self_energy.sum(dim=(2, 3))

        if self.cfg.use_pairwise_energy and num_objects > 1:
            idx_i, idx_j = torch.triu_indices(num_objects, num_objects, offset=1, device=cand_states.device)
            s_i = s_t.index_select(dim=3, index=idx_i)
            s_j = s_t.index_select(dim=3, index=idx_j)
            next_i = s_next.index_select(dim=3, index=idx_i)
            next_j = s_next.index_select(dim=3, index=idx_j)
            pair_mask = transition_mask.index_select(dim=3, index=idx_i) * transition_mask.index_select(dim=3, index=idx_j)
            lam_rel = lambda_sample[:, None, None, None, :].expand(bsz, num_candidates, horizon, idx_i.numel(), -1)
            rel_input = torch.cat([s_i, s_j, next_i, next_j, lam_rel], dim=-1)
            rel_energy = self.rel_mlp(rel_input).squeeze(-1) * pair_mask
            energy = energy + rel_energy.sum(dim=(2, 3))

        if self.cfg.energy_reduction == "mean":
            denom = transition_mask.sum(dim=(2, 3)).clamp_min(1.0)
            energy = energy / denom
        elif self.cfg.energy_reduction != "sum":
            raise ValueError("energy_reduction must be 'sum' or 'mean'")
        return energy


class LOWM(nn.Module):
    """Latent Operator World Model.

    The model infers q_phi(lambda | C), samples a latent operator, and scores
    candidate future trajectories with an object-centric energy function.
    """

    def __init__(self, cfg: LOWMConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or LOWMConfig()
        self.context_encoder = DeepSetsContextEncoder(self.cfg)
        self.energy_model = ObjectCentricEnergy(self.cfg)

    def encode_lambda(
        self,
        context_states: torch.Tensor,
        context_actions: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.context_encoder(context_states, context_actions, context_mask)
        if (not self.training) and self.cfg.use_mu_eval:
            lambda_sample = mu
        else:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            lambda_sample = mu + eps * std
        return mu, logvar, lambda_sample

    def energy(
        self,
        cand_states: torch.Tensor,
        cand_actions: torch.Tensor,
        cand_mask: torch.Tensor,
        lambda_sample: torch.Tensor,
    ) -> torch.Tensor:
        return self.energy_model(cand_states, cand_actions, cand_mask, lambda_sample)

    def energy_matrix(
        self,
        pos_states: torch.Tensor,
        pos_actions: torch.Tensor,
        pos_mask: torch.Tensor,
        lambdas: torch.Tensor,
    ) -> torch.Tensor:
        """Pair each positive trajectory tau_i with each latent operator lambda_j.

        Returns E[i, j] = E(tau_i, lambda_j).
        """

        if pos_states.ndim != 4:
            raise ValueError(f"pos_states must be [B,H+1,N,D], got {tuple(pos_states.shape)}")
        batch_size = pos_states.shape[0]
        traj_states = pos_states[:, None].expand(batch_size, batch_size, *pos_states.shape[1:]).reshape(
            batch_size * batch_size, 1, *pos_states.shape[1:]
        )
        traj_actions = pos_actions[:, None].expand(batch_size, batch_size, *pos_actions.shape[1:]).reshape(
            batch_size * batch_size, 1, *pos_actions.shape[1:]
        )
        traj_mask = pos_mask[:, None].expand(batch_size, batch_size, *pos_mask.shape[1:]).reshape(
            batch_size * batch_size, 1, *pos_mask.shape[1:]
        )
        paired_lambdas = lambdas[None, :, :].expand(batch_size, batch_size, -1).reshape(batch_size * batch_size, -1)
        return self.energy(traj_states, traj_actions, traj_mask, paired_lambdas).reshape(batch_size, batch_size)

    def forward(self, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        mu, logvar, lambda_sample = self.encode_lambda(
            batch["context_states"],
            batch["context_actions"],
            batch["context_mask"],
        )
        energies = self.energy(batch["cand_states"], batch["cand_actions"], batch["cand_mask"], lambda_sample)
        return {
            "energies": energies,
            "mu": mu,
            "logvar": logvar,
            "lambda": lambda_sample,
        }

    def score(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Compatibility helper for evaluators that expect energies only."""

        return self.forward(batch)["energies"]
