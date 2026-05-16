"""LOWM-G: operator-conditioned proposal dynamics model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import nn

from lowm.models.lowm import DeepSetsContextEncoder, LOWMConfig


@dataclass(frozen=True)
class LOWMGConfig:
    object_dim: int = 7
    action_dim: int = 2
    lambda_dim: int = 16
    context_dim: int = 128
    proposal_hidden_dim: int = 128
    proposal_num_layers: int = 2
    proposal_noise_dim: int = 8
    proposal_use_noise: bool = True
    proposal_dropout: float = 0.0
    use_mu: bool = True


def lowm_g_config_from_mapping(config: Mapping[str, Any]) -> LOWMGConfig:
    model = dict(config.get("model", {}))
    proposal = dict(config.get("proposal", {}))
    lambda_dim = int(model.get("lambda_dim", proposal.get("lambda_dim", LOWMGConfig.lambda_dim)))
    if lambda_dim not in {4, 8, 16, 32}:
        raise ValueError("lambda_dim must be one of {4, 8, 16, 32}")
    return LOWMGConfig(
        object_dim=int(model.get("object_dim", proposal.get("object_dim", LOWMGConfig.object_dim))),
        action_dim=int(model.get("action_dim", proposal.get("action_dim", LOWMGConfig.action_dim))),
        lambda_dim=lambda_dim,
        context_dim=int(model.get("context_dim", proposal.get("context_dim", LOWMGConfig.context_dim))),
        proposal_hidden_dim=int(model.get("proposal_hidden_dim", proposal.get("hidden_dim", LOWMGConfig.proposal_hidden_dim))),
        proposal_num_layers=int(model.get("proposal_num_layers", proposal.get("num_layers", LOWMGConfig.proposal_num_layers))),
        proposal_noise_dim=int(model.get("proposal_noise_dim", proposal.get("noise_dim", LOWMGConfig.proposal_noise_dim))),
        proposal_use_noise=bool(model.get("proposal_use_noise", proposal.get("use_noise", LOWMGConfig.proposal_use_noise))),
        proposal_dropout=float(model.get("proposal_dropout", proposal.get("dropout", LOWMGConfig.proposal_dropout))),
        use_mu=bool(model.get("proposal_use_mu", proposal.get("use_mu", LOWMGConfig.use_mu))),
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


class OperatorConditionedProposalModel(nn.Module):
    """Autoregressive object-centric proposal dynamics conditioned on lambda."""

    def __init__(self, cfg: LOWMGConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or LOWMGConfig()
        lowm_cfg = LOWMConfig(
            object_dim=self.cfg.object_dim,
            action_dim=self.cfg.action_dim,
            lambda_dim=self.cfg.lambda_dim,
            hidden_dim=self.cfg.proposal_hidden_dim,
            context_dim=self.cfg.context_dim,
            num_layers=self.cfg.proposal_num_layers,
            dropout=self.cfg.proposal_dropout,
            use_pairwise_energy=False,
        )
        self.context_encoder = DeepSetsContextEncoder(lowm_cfg)
        noise_dim = self.cfg.proposal_noise_dim if self.cfg.proposal_use_noise else 0
        step_input_dim = self.cfg.object_dim + self.cfg.action_dim + self.cfg.lambda_dim + noise_dim
        self.step_mlp = _mlp(step_input_dim, self.cfg.proposal_hidden_dim, 4, self.cfg.proposal_num_layers, self.cfg.proposal_dropout)

    def encode_lambda(
        self,
        context_states: torch.Tensor,
        context_actions: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.context_encoder(context_states, context_actions, context_mask)
        if self.cfg.use_mu or not self.training:
            lam = mu
        else:
            lam = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return mu, logvar, lam

    def _noise(
        self,
        batch_size: int,
        horizon: int,
        num_objects: int,
        device: torch.device,
        dtype: torch.dtype,
        noise: torch.Tensor | None,
        noise_scale: float,
    ) -> torch.Tensor:
        zdim = self.cfg.proposal_noise_dim
        if not self.cfg.proposal_use_noise or zdim <= 0:
            return torch.zeros(batch_size, horizon, num_objects, 0, device=device, dtype=dtype)
        if noise is None:
            return torch.randn(batch_size, horizon, num_objects, zdim, device=device, dtype=dtype) * float(noise_scale)
        noise = noise.to(device=device, dtype=dtype)
        if noise.ndim == 2:
            return noise[:, None, None, :].expand(batch_size, horizon, num_objects, zdim) * float(noise_scale)
        if noise.ndim == 4:
            return noise * float(noise_scale)
        raise ValueError("noise must be [B,z_dim] or [B,H,N,z_dim]")

    def forward(
        self,
        context_states: torch.Tensor,
        context_actions: torch.Tensor,
        context_mask: torch.Tensor,
        initial_state: torch.Tensor,
        actions: torch.Tensor,
        mask: torch.Tensor,
        noise: torch.Tensor | None = None,
        noise_scale: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        if initial_state.ndim != 3:
            raise ValueError("initial_state must be [B,N,D]")
        batch_size, horizon, num_objects, _ = actions.shape
        if initial_state.shape[0] != batch_size:
            raise ValueError("initial_state and actions batch sizes differ")
        mu, logvar, lam = self.encode_lambda(context_states, context_actions, context_mask)
        z = self._noise(batch_size, horizon, num_objects, actions.device, actions.dtype, noise, noise_scale)
        state = initial_state
        states = [initial_state]
        for t in range(horizon):
            lam_obj = lam[:, None, :].expand(batch_size, num_objects, -1)
            step_input = torch.cat([state, actions[:, t], lam_obj, z[:, t]], dim=-1)
            delta = self.step_mlp(step_input)
            active = (mask[:, t] * mask[:, t + 1]).unsqueeze(-1).to(delta.dtype)
            next_state = state.clone()
            next_state_dyn = next_state[:, :, 0:4] + delta * active
            next_state = torch.cat([next_state_dyn, initial_state[:, :, 4:]], dim=-1)
            states.append(next_state)
            state = next_state
        pred_states = torch.stack(states, dim=1)
        pred_states[:, 0] = initial_state
        return {
            "pred_states": pred_states,
            "pred_mask": mask,
            "mu": mu,
            "logvar": logvar,
            "lambda": lam,
        }


def masked_rollout_mse(pred_states: torch.Tensor, target_states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    active = mask[:, 1:, :, None].to(pred_states.dtype)
    diff = (pred_states[:, 1:, :, 0:4] - target_states[:, 1:, :, 0:4]) * active
    denom = active.sum().clamp_min(1.0) * 4.0
    return diff.square().sum() / denom


def masked_delta_mse(pred_states: torch.Tensor, target_states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    pred_delta = pred_states[:, 1:, :, 0:4] - pred_states[:, :-1, :, 0:4]
    true_delta = target_states[:, 1:, :, 0:4] - target_states[:, :-1, :, 0:4]
    active = (mask[:, 1:] * mask[:, :-1])[:, :, :, None].to(pred_states.dtype)
    denom = active.sum().clamp_min(1.0) * 4.0
    return ((pred_delta - true_delta).square() * active).sum() / denom


def rollout_mse_by_horizon(pred_states: torch.Tensor, target_states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    active = mask[:, 1:, :, None].to(pred_states.dtype)
    diff = (pred_states[:, 1:, :, 0:4] - target_states[:, 1:, :, 0:4]) * active
    denom = active.sum(dim=(0, 2, 3)).clamp_min(1.0) * 4.0
    return diff.square().sum(dim=(0, 2, 3)) / denom
