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


def denoising_energy_shaping_loss(
    model,
    batch: dict[str, torch.Tensor],
    lambdas: torch.Tensor,
    data_std_per_dim: torch.Tensor,
    noise_stds: list[float] | tuple[float, ...] = (0.01, 0.03, 0.05, 0.1),
    batch_fraction: float = 1.0,
    clip_grad_target: float | None = None,
    future_only: bool = True,
    use_dsm: bool = True,
    use_denoise_rank: bool = True,
    denoise_rank_margin: float = 1.0,
    use_grad_reg: bool = False,
    create_graph: bool = True,
) -> dict[str, torch.Tensor]:
    """Local trajectory-energy shaping around clean positive futures.

    DSM uses only state dimensions [p_x, p_y, v_x, v_y]. Static attributes stay
    fixed and actions are not noised. The returned scalar losses are unweighted;
    callers apply alpha coefficients.
    """

    clean = batch["pos_states"]
    actions = batch["pos_actions"]
    mask = batch["pos_mask"]
    batch_size = int(clean.shape[0])
    if batch_size == 0:
        zero = torch.zeros((), device=clean.device, dtype=clean.dtype)
        return {
            "dsm_loss": zero,
            "denoise_rank_loss": zero,
            "grad_reg_loss": zero,
            "dsm_grad_norm": zero,
            "dsm_target_norm": zero,
            "dsm_cosine_to_clean_direction": zero,
            "clean_energy": zero,
            "noisy_energy": zero,
            "clean_noisy_gap": zero,
            "clean_noisy_pair_acc": zero,
        }

    frac = float(batch_fraction)
    take = max(1, min(batch_size, int(round(batch_size * frac))))
    clean = clean[:take]
    actions = actions[:take]
    mask = mask[:take]
    lambdas = lambdas[:take]

    std_choices = torch.tensor(list(noise_stds), device=clean.device, dtype=clean.dtype)
    if std_choices.numel() == 0:
        raise ValueError("noise_stds must be non-empty")
    choice_idx = torch.randint(0, std_choices.numel(), (take,), device=clean.device)
    noise_scale = std_choices[choice_idx].view(take, 1, 1, 1)
    data_std = data_std_per_dim.to(device=clean.device, dtype=clean.dtype).view(1, 1, 1, 4).clamp_min(1e-6)
    sigma = noise_scale * data_std

    noisy = clean.detach().clone()
    dyn_noise = torch.randn_like(noisy[:, 1:, :, 0:4]) * sigma
    noisy[:, 1:, :, 0:4] = noisy[:, 1:, :, 0:4] + dyn_noise
    noisy[:, 1:, :, 0:2] = noisy[:, 1:, :, 0:2].clamp(0.0, 1.0)
    noisy.requires_grad_(True)

    clean_energy = model.energy(clean[:, None], actions[:, None], mask[:, None], lambdas).squeeze(1)
    noisy_energy = model.energy(noisy[:, None], actions[:, None], mask[:, None], lambdas).squeeze(1)
    grad = torch.autograd.grad(
        noisy_energy.sum(),
        noisy,
        create_graph=create_graph,
        retain_graph=True,
        only_inputs=True,
    )[0]

    time_slice = slice(1, None) if future_only else slice(None)
    active = mask[:, time_slice, :, None].to(clean.dtype)
    grad_dyn = grad[:, time_slice, :, 0:4]
    target = (noisy[:, time_slice, :, 0:4] - clean[:, time_slice, :, 0:4]) / sigma.square()
    if clip_grad_target is not None:
        target = target.clamp(min=-float(clip_grad_target), max=float(clip_grad_target))
    denom = active.sum().clamp_min(1.0) * 4.0
    dsm = ((grad_dyn - target).square() * active).sum() / denom if use_dsm else torch.zeros((), device=clean.device, dtype=clean.dtype)
    grad_reg = (grad_dyn.square() * active).sum() / denom if use_grad_reg else torch.zeros((), device=clean.device, dtype=clean.dtype)
    denoise_rank = (
        F.softplus(clean_energy - noisy_energy + float(denoise_rank_margin)).mean()
        if use_denoise_rank
        else torch.zeros((), device=clean.device, dtype=clean.dtype)
    )

    flat_mask = active.expand_as(grad_dyn).reshape(take, -1)
    neg_grad_flat = (-grad_dyn * active).reshape(take, -1)
    clean_dir_flat = ((clean[:, time_slice, :, 0:4] - noisy[:, time_slice, :, 0:4]) * active).reshape(take, -1)
    dot = (neg_grad_flat * clean_dir_flat).sum(dim=1)
    grad_norm = torch.sqrt((neg_grad_flat.square() * flat_mask).sum(dim=1).clamp_min(1e-12))
    target_norm = torch.sqrt(((target * active).reshape(take, -1).square() * flat_mask).sum(dim=1).clamp_min(1e-12))
    clean_norm = torch.sqrt((clean_dir_flat.square() * flat_mask).sum(dim=1).clamp_min(1e-12))
    cosine = dot / (grad_norm * clean_norm).clamp_min(1e-12)

    return {
        "dsm_loss": dsm,
        "denoise_rank_loss": denoise_rank,
        "grad_reg_loss": grad_reg,
        "dsm_grad_norm": grad_norm.mean().detach(),
        "dsm_target_norm": target_norm.mean().detach(),
        "dsm_cosine_to_clean_direction": cosine.mean().detach(),
        "clean_energy": clean_energy.mean().detach(),
        "noisy_energy": noisy_energy.mean().detach(),
        "clean_noisy_gap": (noisy_energy - clean_energy).mean().detach(),
        "clean_noisy_pair_acc": (clean_energy < noisy_energy).float().mean().detach(),
    }


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
