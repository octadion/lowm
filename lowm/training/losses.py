"""Loss functions for energy ranking."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def nce_ranking_loss(energies: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Cross entropy over negative energies.

    Lower energy means more coherent, so logits are `-energies`.
    """

    if energies.ndim != 2:
        raise ValueError(f"energies must have shape [B,M], got {tuple(energies.shape)}")
    if labels.ndim != 1:
        raise ValueError(f"labels must have shape [B], got {tuple(labels.shape)}")
    return F.cross_entropy(-energies, labels)


def kl_standard_normal_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Mean KL(q(lambda|C) || N(0,I)) over the batch."""

    if mu.shape != logvar.shape:
        raise ValueError("mu and logvar must have the same shape")
    kl_per_sample = 0.5 * torch.sum(torch.exp(logvar) + mu.square() - 1.0 - logvar, dim=-1)
    return kl_per_sample.mean()


def law_stability_loss(mu_a: torch.Tensor, mu_b: torch.Tensor) -> torch.Tensor:
    """Encourage two context subsets from the same sample to infer the same operator."""

    if mu_a.shape != mu_b.shape:
        raise ValueError("mu_a and mu_b must have the same shape")
    return (mu_a - mu_b).square().mean()


def lowm_total_loss(
    energies: torch.Tensor,
    labels: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta_kl: float,
    stability: torch.Tensor | None = None,
    alpha_stable: float = 0.0,
    occl_loss: torch.Tensor | None = None,
    alpha_occl: float = 0.0,
) -> dict[str, torch.Tensor]:
    nce = nce_ranking_loss(energies, labels)
    kl = kl_standard_normal_loss(mu, logvar)
    stable = stability if stability is not None else torch.zeros((), device=energies.device, dtype=energies.dtype)
    occl = occl_loss if occl_loss is not None else torch.zeros((), device=energies.device, dtype=energies.dtype)
    total = nce + float(alpha_occl) * occl + float(beta_kl) * kl + float(alpha_stable) * stable
    return {"total": total, "nce": nce, "kl": kl, "stability": stable, "occl": occl}


def operator_coherence_contrastive_loss(E_matrix: torch.Tensor, temperature: float = 1.0) -> dict[str, torch.Tensor]:
    if E_matrix.ndim != 2 or E_matrix.shape[0] != E_matrix.shape[1]:
        raise ValueError(f"E_matrix must have shape [B,B], got {tuple(E_matrix.shape)}")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    logits = -E_matrix / float(temperature)
    labels = torch.arange(E_matrix.shape[0], device=E_matrix.device)
    tau_to_lambda = F.cross_entropy(logits, labels)
    lambda_to_tau = F.cross_entropy(logits.T, labels)
    occl = 0.5 * (tau_to_lambda + lambda_to_tau)
    return {
        "occl_loss": occl,
        "tau_to_lambda_loss": tau_to_lambda,
        "lambda_to_tau_loss": lambda_to_tau,
        "occl_acc_tau_to_lambda": (torch.argmax(logits, dim=1) == labels).float().mean(),
        "occl_acc_lambda_to_tau": (torch.argmax(logits.T, dim=1) == labels).float().mean(),
    }
