"""Probe latent/context representations for hidden operator information."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from lowm.data.dataset import LOWMSynthRankingDataset, make_ranking_dataloader, ranking_config_from_mapping
from lowm.data.operators import OP_PARAM_NAMES
from lowm.eval.evaluate_all import _ensure_split, _move_batch_to_device, _resolve_checkpoint_path, load_run_model


METRIC_VERSION = "paper1_operator_probe_v1"


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def _context_representation(model: torch.nn.Module, batch: Mapping[str, torch.Tensor], model_type: str) -> torch.Tensor:
    """Return mu/lambda-like context representation for models that expose one."""

    if model_type == "lowm" or hasattr(model, "encode_lambda"):
        if not hasattr(model, "context_encoder"):
            raise ValueError("LOWM-like model does not expose context_encoder")
        mu, _ = model.context_encoder(batch["context_states"], batch["context_actions"], batch["context_mask"])
        return mu
    if hasattr(model, "context_encoder"):
        return model.context_encoder(batch["context_states"], batch["context_actions"], batch["context_mask"])
    raise ValueError(
        "operator representation probing requires a context/latent encoder; "
        f"model_type='{model_type}' is not compatible"
    )


def _collect_representations(
    model: torch.nn.Module,
    model_type: str,
    dataset: LOWMSynthRankingDataset,
    batch_size: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    loader = make_ranking_dataloader(dataset, batch_size=batch_size, shuffle=False)
    embeddings: list[np.ndarray] = []
    op_ids: list[np.ndarray] = []
    op_params: list[np.ndarray] = []
    num_objects: list[np.ndarray] = []
    episodes: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            batch = _move_batch_to_device(batch, device)
            reps = _context_representation(model, batch, model_type).detach().cpu().numpy()
            query_episode = batch["query_episode"].detach().cpu().numpy().astype(np.int64)
            embeddings.append(reps.astype(np.float32))
            op_ids.append(batch["query_op_id"].detach().cpu().numpy().astype(np.int64))
            op_params.append(batch["query_op_params"].detach().cpu().numpy().astype(np.float32))
            num_objects.append(dataset.num_objects[query_episode].astype(np.int64))
            episodes.append(query_episode)

    if not embeddings:
        raise ValueError("no samples collected for operator representation probe")
    return {
        "embeddings": np.concatenate(embeddings, axis=0),
        "op_id": np.concatenate(op_ids, axis=0),
        "op_params": np.concatenate(op_params, axis=0),
        "num_objects": np.concatenate(num_objects, axis=0),
        "query_episode": np.concatenate(episodes, axis=0),
    }


def _stratified_split(labels: np.ndarray, test_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train: list[int] = []
    test: list[int] = []
    for label in np.unique(labels):
        idx = np.where(labels == label)[0]
        rng.shuffle(idx)
        n_test = int(round(len(idx) * test_fraction)) if len(idx) > 1 else 0
        n_test = min(max(n_test, 1 if len(idx) >= 4 else 0), max(0, len(idx) - 1))
        test.extend(int(i) for i in idx[:n_test])
        train.extend(int(i) for i in idx[n_test:])
    if not test and len(labels) > 1:
        perm = rng.permutation(len(labels))
        test = [int(perm[0])]
        train = [int(i) for i in perm[1:]]
    if not train:
        train = list(test)
    if not test:
        test = list(train)
    return np.asarray(train, dtype=np.int64), np.asarray(test, dtype=np.int64)


def _standardize(train_x: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (train_x - mean) / std, (test_x - mean) / std, mean.squeeze(0), std.squeeze(0)


def _fit_linear_classifier(
    x: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    seed: int,
    epochs: int,
) -> dict[str, Any]:
    classes = np.unique(y)
    class_to_idx = {int(label): idx for idx, label in enumerate(classes)}
    y_mapped = np.asarray([class_to_idx[int(label)] for label in y], dtype=np.int64)
    x_train, x_test, _, _ = _standardize(x[train_idx], x[test_idx])
    y_train = y_mapped[train_idx]
    y_test = y_mapped[test_idx]

    if len(classes) == 1:
        return {
            "accuracy": 1.0,
            "train_accuracy": 1.0,
            "classes": classes.astype(int).tolist(),
            "confusion": np.asarray([[len(test_idx)]], dtype=np.int64),
            "correct_count": int(len(test_idx)),
            "test_count": int(len(test_idx)),
        }

    torch.manual_seed(seed)
    model = nn.Linear(x.shape[1], len(classes))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.05, weight_decay=1e-4)
    x_train_t = torch.from_numpy(x_train.astype(np.float32))
    y_train_t = torch.from_numpy(y_train)
    x_test_t = torch.from_numpy(x_test.astype(np.float32))
    for _ in range(max(1, int(epochs))):
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(model(x_train_t), y_train_t)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        train_pred = model(x_train_t).argmax(dim=1).cpu().numpy()
        test_pred = model(x_test_t).argmax(dim=1).cpu().numpy()
    confusion = np.zeros((len(classes), len(classes)), dtype=np.int64)
    for target, pred in zip(y_test, test_pred):
        confusion[int(target), int(pred)] += 1
    return {
        "accuracy": float(np.mean(test_pred == y_test)) if len(y_test) else 0.0,
        "train_accuracy": float(np.mean(train_pred == y_train)) if len(y_train) else 0.0,
        "classes": classes.astype(int).tolist(),
        "confusion": confusion,
        "correct_count": int(np.sum(test_pred == y_test)),
        "test_count": int(len(y_test)),
    }


class _MLPProbe(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _fit_mlp_classifier(
    x: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    seed: int,
    hidden_dim: int,
    epochs: int,
    lr: float,
) -> dict[str, Any]:
    classes = np.unique(y)
    class_to_idx = {int(label): idx for idx, label in enumerate(classes)}
    y_mapped = np.asarray([class_to_idx[int(label)] for label in y], dtype=np.int64)
    x_train, x_test, _, _ = _standardize(x[train_idx], x[test_idx])
    y_train = y_mapped[train_idx]
    y_test = y_mapped[test_idx]

    if len(classes) == 1:
        return {
            "accuracy": 1.0,
            "train_accuracy": 1.0,
            "classes": classes.astype(int).tolist(),
            "confusion": np.asarray([[len(test_idx)]], dtype=np.int64),
            "correct_count": int(len(test_idx)),
            "test_count": int(len(test_idx)),
        }

    torch.manual_seed(seed)
    model = _MLPProbe(x.shape[1], len(classes), hidden_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=1e-4)
    x_train_t = torch.from_numpy(x_train.astype(np.float32))
    y_train_t = torch.from_numpy(y_train)
    x_test_t = torch.from_numpy(x_test.astype(np.float32))
    for _ in range(max(1, int(epochs))):
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(model(x_train_t), y_train_t)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        train_pred = model(x_train_t).argmax(dim=1).cpu().numpy()
        test_pred = model(x_test_t).argmax(dim=1).cpu().numpy()
    confusion = np.zeros((len(classes), len(classes)), dtype=np.int64)
    for target, pred in zip(y_test, test_pred):
        confusion[int(target), int(pred)] += 1
    return {
        "accuracy": float(np.mean(test_pred == y_test)) if len(y_test) else 0.0,
        "train_accuracy": float(np.mean(train_pred == y_train)) if len(y_train) else 0.0,
        "classes": classes.astype(int).tolist(),
        "confusion": confusion,
        "correct_count": int(np.sum(test_pred == y_test)),
        "test_count": int(len(y_test)),
    }


def _fit_ridge_regression(
    x: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    alpha: float,
) -> dict[str, Any]:
    x_train, x_test, _, _ = _standardize(x[train_idx], x[test_idx])
    y_train = y[train_idx]
    y_test = y[test_idx]
    x_train_aug = np.concatenate([x_train, np.ones((x_train.shape[0], 1), dtype=x_train.dtype)], axis=1)
    x_test_aug = np.concatenate([x_test, np.ones((x_test.shape[0], 1), dtype=x_test.dtype)], axis=1)
    penalty = np.eye(x_train_aug.shape[1], dtype=np.float64) * float(alpha)
    penalty[-1, -1] = 0.0
    lhs = x_train_aug.T @ x_train_aug + penalty
    rhs = x_train_aug.T @ y_train
    try:
        weights = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        weights = np.linalg.pinv(lhs) @ rhs
    pred = x_test_aug @ weights
    mse_per_dim = np.mean((pred - y_test) ** 2, axis=0)
    var_per_dim = np.var(y_test, axis=0)
    r2_per_dim = np.where(var_per_dim > 1e-12, 1.0 - mse_per_dim / np.maximum(var_per_dim, 1e-12), 0.0)
    return {
        "pred": pred.astype(np.float32),
        "mse": float(np.mean(mse_per_dim)),
        "r2": float(np.mean(r2_per_dim)),
        "mse_per_dim": mse_per_dim.astype(float).tolist(),
        "r2_per_dim": r2_per_dim.astype(float).tolist(),
    }


def _fit_mlp_regression(
    x: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    seed: int,
    hidden_dim: int,
    epochs: int,
    lr: float,
) -> dict[str, Any]:
    x_train, x_test, _, _ = _standardize(x[train_idx], x[test_idx])
    y_train = y[train_idx]
    y_test = y[test_idx]
    y_mean = y_train.mean(axis=0, keepdims=True)
    y_std = y_train.std(axis=0, keepdims=True)
    y_std = np.where(y_std < 1e-6, 1.0, y_std)
    y_train_std = (y_train - y_mean) / y_std

    torch.manual_seed(seed)
    model = _MLPProbe(x.shape[1], y.shape[1], hidden_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=1e-4)
    x_train_t = torch.from_numpy(x_train.astype(np.float32))
    y_train_t = torch.from_numpy(y_train_std.astype(np.float32))
    x_test_t = torch.from_numpy(x_test.astype(np.float32))
    for _ in range(max(1, int(epochs))):
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.mse_loss(model(x_train_t), y_train_t)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        pred_std = model(x_test_t).cpu().numpy()
    pred = pred_std * y_std + y_mean
    mse_per_dim = np.mean((pred - y_test) ** 2, axis=0)
    var_per_dim = np.var(y_test, axis=0)
    r2_per_dim = np.where(var_per_dim > 1e-12, 1.0 - mse_per_dim / np.maximum(var_per_dim, 1e-12), 0.0)
    return {
        "pred": pred.astype(np.float32),
        "mse": float(np.mean(mse_per_dim)),
        "r2": float(np.mean(r2_per_dim)),
        "mse_per_dim": mse_per_dim.astype(float).tolist(),
        "r2_per_dim": r2_per_dim.astype(float).tolist(),
    }


def _binned_param_accuracy(y_train: np.ndarray, y_test: np.ndarray, pred: np.ndarray, bins: int) -> dict[str, Any]:
    per_dim: list[float] = []
    valid_per_dim: list[int] = []
    total_correct = 0
    total_count = 0
    for dim in range(y_train.shape[1]):
        unique = np.unique(y_train[:, dim])
        if len(unique) < 2:
            per_dim.append(0.0)
            valid_per_dim.append(0)
            continue
        quantiles = np.linspace(0, 1, int(bins) + 1)[1:-1]
        edges = np.unique(np.quantile(y_train[:, dim], quantiles))
        if len(edges) == 0:
            per_dim.append(0.0)
            valid_per_dim.append(0)
            continue
        true_bins = np.digitize(y_test[:, dim], edges)
        pred_bins = np.digitize(pred[:, dim], edges)
        correct = int(np.sum(true_bins == pred_bins))
        count = int(len(true_bins))
        total_correct += correct
        total_count += count
        per_dim.append(float(correct / max(1, count)))
        valid_per_dim.append(1)
    return {
        "accuracy": float(total_correct / max(1, total_count)),
        "accuracy_per_dim": per_dim,
        "valid_per_dim": valid_per_dim,
    }


def _bin_edges_by_dim(y_train: np.ndarray, bins: int) -> list[np.ndarray | None]:
    edges_by_dim: list[np.ndarray | None] = []
    for dim in range(y_train.shape[1]):
        if len(np.unique(y_train[:, dim])) < 2:
            edges_by_dim.append(None)
            continue
        quantiles = np.linspace(0, 1, int(bins) + 1)[1:-1]
        edges = np.unique(np.quantile(y_train[:, dim], quantiles))
        edges_by_dim.append(edges if len(edges) else None)
    return edges_by_dim


def _binned_labels(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.digitize(values, edges).astype(np.int64)


def _fit_binned_param_classifiers(
    x: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    seed: int,
    bins: int,
    probe_kind: str,
    hidden_dim: int,
    epochs: int,
    lr: float,
) -> dict[str, Any]:
    edges_by_dim = _bin_edges_by_dim(y[train_idx], bins)
    per_dim: list[float] = []
    valid_per_dim: list[int] = []
    total_correct = 0
    total_count = 0
    for dim, edges in enumerate(edges_by_dim):
        if edges is None:
            per_dim.append(0.0)
            valid_per_dim.append(0)
            continue
        train_bins = _binned_labels(y[train_idx, dim], edges)
        test_bins = _binned_labels(y[test_idx, dim], edges)
        if len(np.unique(train_bins)) < 2:
            per_dim.append(0.0)
            valid_per_dim.append(0)
            continue
        x_pair = np.concatenate([x[train_idx], x[test_idx]], axis=0)
        y_pair = np.concatenate([train_bins, test_bins], axis=0)
        pair_train_idx = np.arange(len(train_idx))
        pair_test_idx = np.arange(len(train_idx), len(train_idx) + len(test_idx))
        if probe_kind == "linear":
            result = _fit_linear_classifier(x_pair, y_pair, pair_train_idx, pair_test_idx, seed + dim + 1, epochs)
        else:
            result = _fit_mlp_classifier(
                x_pair,
                y_pair,
                pair_train_idx,
                pair_test_idx,
                seed + dim + 1,
                hidden_dim,
                epochs,
                lr,
            )
        acc = float(result["accuracy"])
        per_dim.append(acc)
        valid_per_dim.append(1)
        total_correct += int(result.get("correct_count", 0))
        total_count += int(result.get("test_count", len(test_bins)))
    return {
        "accuracy": float(total_correct / max(1, total_count)),
        "accuracy_per_dim": per_dim,
        "valid_per_dim": valid_per_dim,
    }


def _pca_2d(x: np.ndarray) -> tuple[np.ndarray, list[float]]:
    centered = x - x.mean(axis=0, keepdims=True)
    if centered.shape[0] < 2:
        return np.zeros((centered.shape[0], 2), dtype=np.float32), [0.0, 0.0]
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    n_components = min(2, vt.shape[0])
    projected = centered @ vt[:n_components].T
    if n_components < 2:
        projected = np.pad(projected, ((0, 0), (0, 2 - n_components)))
    variances = singular_values**2 / max(1, centered.shape[0] - 1)
    total = float(np.sum(variances))
    ratios = (variances[:2] / total).tolist() if total > 0 else [0.0, 0.0]
    while len(ratios) < 2:
        ratios.append(0.0)
    return projected.astype(np.float32), [float(x) for x in ratios]


def _silhouette_score(x: np.ndarray, labels: np.ndarray, max_samples: int, seed: int) -> float:
    unique = np.unique(labels)
    if len(unique) < 2 or len(labels) < 3:
        return 0.0
    rng = np.random.default_rng(seed)
    if len(labels) > max_samples:
        idx = rng.choice(np.arange(len(labels)), size=max_samples, replace=False)
        x = x[idx]
        labels = labels[idx]
    distances = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=-1)
    scores: list[float] = []
    for i in range(len(labels)):
        same = labels == labels[i]
        other = labels != labels[i]
        if np.sum(same) <= 1 or not np.any(other):
            continue
        a = float(np.mean(distances[i, same & (np.arange(len(labels)) != i)]))
        b = min(float(np.mean(distances[i, labels == label])) for label in np.unique(labels[other]))
        denom = max(a, b)
        if denom > 1e-12:
            scores.append((b - a) / denom)
    return float(np.mean(scores)) if scores else 0.0


def _correlation_matrix(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    corr = np.zeros((x.shape[1], y.shape[1]), dtype=np.float32)
    for i in range(x.shape[1]):
        xi = x[:, i]
        if float(np.std(xi)) < 1e-12:
            continue
        for j in range(y.shape[1]):
            yj = y[:, j]
            if float(np.std(yj)) < 1e-12:
                continue
            corr[i, j] = float(np.corrcoef(xi, yj)[0, 1])
    return corr


def _write_probe_csv(metrics: Mapping[str, Any], path: Path) -> None:
    rows: list[dict[str, Any]] = [
        {"metric": "op_id_probe_accuracy", "value": metrics.get("op_id_probe_accuracy", 0.0)},
        {"metric": "op_id_probe_train_accuracy", "value": metrics.get("op_id_probe_train_accuracy", 0.0)},
        {"metric": "op_param_mse", "value": metrics.get("op_param_mse", 0.0)},
        {"metric": "op_param_r2", "value": metrics.get("op_param_r2", 0.0)},
        {"metric": "binned_param_accuracy", "value": metrics.get("binned_param_accuracy", 0.0)},
        {"metric": "linear_op_id_probe_accuracy", "value": metrics.get("linear_op_id_probe_accuracy", 0.0)},
        {"metric": "linear_op_param_r2", "value": metrics.get("linear_op_param_r2", 0.0)},
        {"metric": "linear_binned_param_accuracy", "value": metrics.get("linear_binned_param_accuracy", 0.0)},
        {"metric": "mlp_op_id_probe_accuracy", "value": metrics.get("mlp_op_id_probe_accuracy", 0.0)},
        {"metric": "mlp_op_param_r2", "value": metrics.get("mlp_op_param_r2", 0.0)},
        {"metric": "mlp_binned_param_accuracy", "value": metrics.get("mlp_binned_param_accuracy", 0.0)},
        {"metric": "op_id_silhouette", "value": metrics.get("op_id_silhouette", 0.0)},
    ]
    for name, value in zip(OP_PARAM_NAMES, metrics.get("op_param_mse_per_dim", [])):
        rows.append({"metric": f"{name}_mse", "value": value})
    for name, value in zip(OP_PARAM_NAMES, metrics.get("op_param_r2_per_dim", [])):
        rows.append({"metric": f"{name}_r2", "value": value})
    for name, value in zip(OP_PARAM_NAMES, metrics.get("binned_param_accuracy_per_dim", [])):
        rows.append({"metric": f"{name}_binned_accuracy", "value": value})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerows(rows)


def _plot_pca(projected: np.ndarray, labels: np.ndarray, ratios: list[float], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    scatter = ax.scatter(projected[:, 0], projected[:, 1], c=labels, cmap="tab10", s=18, alpha=0.85)
    ax.set_xlabel(f"PC1 ({ratios[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({ratios[1] * 100:.1f}%)")
    ax.set_title("Context representation PCA by operator")
    legend = ax.legend(*scatter.legend_elements(), title="op_id", loc="best", fontsize=8)
    ax.add_artist(legend)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_confusion(confusion: np.ndarray, classes: list[int], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    im = ax.imshow(confusion, cmap="Blues", aspect="auto")
    ax.set_xlabel("Predicted op_id")
    ax.set_ylabel("True op_id")
    ax.set_xticks(np.arange(len(classes)), [str(c) for c in classes])
    ax.set_yticks(np.arange(len(classes)), [str(c) for c in classes])
    for i in range(confusion.shape[0]):
        for j in range(confusion.shape[1]):
            value = int(confusion[i, j])
            ax.text(j, i, str(value), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_correlation(corr: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, max(3.0, 0.25 * corr.shape[0])))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1.0, vmax=1.0, aspect="auto")
    ax.set_xlabel("operator parameter")
    ax.set_ylabel("representation dim")
    ax.set_xticks(np.arange(len(OP_PARAM_NAMES)), OP_PARAM_NAMES, rotation=25, ha="right")
    ax.set_yticks(np.arange(corr.shape[0]))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _write_summary(metrics: Mapping[str, Any], path: Path) -> None:
    lines = [
        "# Operator Representation Probe Summary",
        "",
        f"- Samples: {metrics.get('num_samples', 0)}",
        f"- Representation dim: {metrics.get('embedding_dim', 0)}",
        f"- Probe type: {metrics.get('probe_type', 'linear')}",
        f"- Linear op_id accuracy: {metrics.get('linear_op_id_probe_accuracy', 0.0):.4f}",
        f"- Linear op_param R2: {metrics.get('linear_op_param_r2', 0.0):.4f}",
        f"- Linear binned param accuracy: {metrics.get('linear_binned_param_accuracy', 0.0):.4f}",
        f"- MLP op_id accuracy: {metrics.get('mlp_op_id_probe_accuracy', 0.0):.4f}",
        f"- MLP op_param R2: {metrics.get('mlp_op_param_r2', 0.0):.4f}",
        f"- MLP binned param accuracy: {metrics.get('mlp_binned_param_accuracy', 0.0):.4f}",
        f"- op_id silhouette: {metrics.get('op_id_silhouette', 0.0):.4f}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def probe_operator_representation(
    run_dir: Path,
    split: str = "val",
    checkpoint_name: str = "best_law_pair.pt",
    model_type: str | None = None,
    batch_size: int | None = None,
    num_samples: int | None = None,
    seed: int | None = None,
    device_name: str = "auto",
    probe_type: str = "linear",
    probe_seed: int = 0,
    probe_hidden_dim: int = 64,
    probe_epochs: int = 100,
    probe_lr: float = 1e-3,
    test_fraction: float = 0.3,
    ridge_alpha: float = 1e-2,
    param_bins: int = 4,
) -> dict[str, Any]:
    if probe_type not in {"linear", "mlp", "both"}:
        raise ValueError("probe_type must be one of {'linear', 'mlp', 'both'}")
    device = _resolve_device(device_name)
    checkpoint_path = _resolve_checkpoint_path(run_dir, checkpoint_name)
    model, config, detected = load_run_model(run_dir, model_type=model_type, checkpoint_name=checkpoint_name, device=device)
    split_path = _ensure_split(config, split)
    ranking_cfg = ranking_config_from_mapping(config)
    ranking_cfg = replace(ranking_cfg, seed=int(seed if seed is not None else ranking_cfg.seed))
    eval_cfg = dict(config.get("evaluation", {}))
    sample_count = num_samples if num_samples is not None else eval_cfg.get("num_samples", config.get("training", {}).get("val_samples"))
    dataset = LOWMSynthRankingDataset(split_path, ranking_cfg, num_samples=int(sample_count) if sample_count else None)
    bs = int(batch_size or config.get("training", {}).get("batch_size", 64))

    collected = _collect_representations(model, detected, dataset, bs, device)
    embeddings = collected["embeddings"]
    op_id = collected["op_id"]
    op_params = collected["op_params"]
    split_seed = int(probe_seed)
    train_idx, test_idx = _stratified_split(op_id, test_fraction=test_fraction, seed=split_seed)

    run_linear = probe_type in {"linear", "both"}
    run_mlp = probe_type in {"mlp", "both"}
    linear_classifier: dict[str, Any] | None = None
    linear_regression: dict[str, Any] | None = None
    linear_binned: dict[str, Any] | None = None
    mlp_classifier: dict[str, Any] | None = None
    mlp_regression: dict[str, Any] | None = None
    mlp_binned: dict[str, Any] | None = None

    if run_linear:
        linear_classifier = _fit_linear_classifier(embeddings, op_id, train_idx, test_idx, seed=split_seed, epochs=probe_epochs)
        linear_regression = _fit_ridge_regression(embeddings, op_params, train_idx, test_idx, alpha=ridge_alpha)
        linear_binned = _fit_binned_param_classifiers(
            embeddings,
            op_params,
            train_idx,
            test_idx,
            seed=split_seed,
            bins=param_bins,
            probe_kind="linear",
            hidden_dim=probe_hidden_dim,
            epochs=probe_epochs,
            lr=probe_lr,
        )
    if run_mlp:
        mlp_classifier = _fit_mlp_classifier(
            embeddings,
            op_id,
            train_idx,
            test_idx,
            seed=split_seed,
            hidden_dim=probe_hidden_dim,
            epochs=probe_epochs,
            lr=probe_lr,
        )
        mlp_regression = _fit_mlp_regression(
            embeddings,
            op_params,
            train_idx,
            test_idx,
            seed=split_seed + 1009,
            hidden_dim=probe_hidden_dim,
            epochs=probe_epochs,
            lr=probe_lr,
        )
        mlp_binned = _fit_binned_param_classifiers(
            embeddings,
            op_params,
            train_idx,
            test_idx,
            seed=split_seed + 2017,
            bins=param_bins,
            probe_kind="mlp",
            hidden_dim=probe_hidden_dim,
            epochs=probe_epochs,
            lr=probe_lr,
        )

    primary_classifier = linear_classifier if linear_classifier is not None else mlp_classifier
    primary_regression = linear_regression if linear_regression is not None else mlp_regression
    primary_binned = linear_binned if linear_binned is not None else mlp_binned
    if primary_classifier is None or primary_regression is None or primary_binned is None:
        raise ValueError("no probe metrics were produced")
    projected, pca_ratios = _pca_2d(embeddings)
    corr = _correlation_matrix(embeddings, op_params)
    silhouette = _silhouette_score(embeddings, op_id, max_samples=512, seed=split_seed)

    metrics: dict[str, Any] = {
        "metric_version": METRIC_VERSION,
        "model_type": detected,
        "split": split,
        "checkpoint_requested": checkpoint_name,
        "checkpoint_used": checkpoint_path.name,
        "checkpoint_stem": checkpoint_path.stem,
        "ranking_seed": ranking_cfg.seed,
        "probe_type": probe_type,
        "probe_seed": split_seed,
        "probe_hidden_dim": int(probe_hidden_dim),
        "probe_epochs": int(probe_epochs),
        "probe_lr": float(probe_lr),
        "num_samples": int(len(embeddings)),
        "embedding_dim": int(embeddings.shape[1]),
        "train_samples": int(len(train_idx)),
        "test_samples": int(len(test_idx)),
        "op_id_probe_accuracy": primary_classifier["accuracy"],
        "op_id_probe_train_accuracy": primary_classifier["train_accuracy"],
        "op_id_classes": primary_classifier["classes"],
        "op_param_mse": primary_regression["mse"],
        "op_param_r2": primary_regression["r2"],
        "op_param_mse_per_dim": primary_regression["mse_per_dim"],
        "op_param_r2_per_dim": primary_regression["r2_per_dim"],
        "binned_param_accuracy": primary_binned["accuracy"],
        "binned_param_accuracy_per_dim": primary_binned["accuracy_per_dim"],
        "binned_param_valid_per_dim": primary_binned["valid_per_dim"],
        "linear_op_id_probe_accuracy": linear_classifier["accuracy"] if linear_classifier else 0.0,
        "linear_op_id_probe_train_accuracy": linear_classifier["train_accuracy"] if linear_classifier else 0.0,
        "linear_op_param_mse": linear_regression["mse"] if linear_regression else 0.0,
        "linear_op_param_r2": linear_regression["r2"] if linear_regression else 0.0,
        "linear_op_param_mse_per_dim": linear_regression["mse_per_dim"] if linear_regression else [],
        "linear_op_param_r2_per_dim": linear_regression["r2_per_dim"] if linear_regression else [],
        "linear_binned_param_accuracy": linear_binned["accuracy"] if linear_binned else 0.0,
        "linear_binned_param_accuracy_per_dim": linear_binned["accuracy_per_dim"] if linear_binned else [],
        "linear_binned_param_valid_per_dim": linear_binned["valid_per_dim"] if linear_binned else [],
        "mlp_op_id_probe_accuracy": mlp_classifier["accuracy"] if mlp_classifier else 0.0,
        "mlp_op_id_probe_train_accuracy": mlp_classifier["train_accuracy"] if mlp_classifier else 0.0,
        "mlp_op_param_mse": mlp_regression["mse"] if mlp_regression else 0.0,
        "mlp_op_param_r2": mlp_regression["r2"] if mlp_regression else 0.0,
        "mlp_op_param_mse_per_dim": mlp_regression["mse_per_dim"] if mlp_regression else [],
        "mlp_op_param_r2_per_dim": mlp_regression["r2_per_dim"] if mlp_regression else [],
        "mlp_binned_param_accuracy": mlp_binned["accuracy"] if mlp_binned else 0.0,
        "mlp_binned_param_accuracy_per_dim": mlp_binned["accuracy_per_dim"] if mlp_binned else [],
        "mlp_binned_param_valid_per_dim": mlp_binned["valid_per_dim"] if mlp_binned else [],
        "op_id_silhouette": silhouette,
        "pca_explained_variance_ratio": pca_ratios,
    }

    out_dir = run_dir / "eval" / split / "operator_probe"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "lambda_embeddings.npy", embeddings)
    np.save(out_dir / "pca_lambda.npy", projected)
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    _write_probe_csv(metrics, out_dir / "probe_results.csv")
    _plot_pca(projected, op_id, pca_ratios, out_dir / "pca_lambda.png")
    _plot_confusion(primary_classifier["confusion"], primary_classifier["classes"], out_dir / "op_id_confusion.png")
    if linear_classifier is not None:
        _plot_confusion(linear_classifier["confusion"], linear_classifier["classes"], out_dir / "linear_op_id_confusion.png")
    if mlp_classifier is not None:
        _plot_confusion(mlp_classifier["confusion"], mlp_classifier["classes"], out_dir / "mlp_op_id_confusion.png")
    _plot_correlation(corr, out_dir / "lambda_param_correlation.png")
    _write_summary(metrics, out_dir / "probe_summary.md")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--checkpoint", type=str, default="best_law_pair.pt")
    parser.add_argument("--model_type", type=str, default=None, choices=["direct_context_energy", "lowm"])
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--probe-type", type=str, default="linear", choices=["linear", "mlp", "both"])
    parser.add_argument("--probe-seed", type=int, default=0)
    parser.add_argument("--probe-hidden-dim", type=int, default=64)
    parser.add_argument("--probe-epochs", type=int, default=100)
    parser.add_argument("--probe-lr", type=float, default=1e-3)
    parser.add_argument("--test-fraction", type=float, default=0.3)
    parser.add_argument("--ridge-alpha", type=float, default=1e-2)
    parser.add_argument("--param-bins", type=int, default=4)
    args = parser.parse_args()
    metrics = probe_operator_representation(
        args.run,
        split=args.split,
        checkpoint_name=args.checkpoint,
        model_type=args.model_type,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        seed=args.seed,
        device_name=args.device,
        probe_type=args.probe_type,
        probe_seed=args.probe_seed,
        probe_hidden_dim=args.probe_hidden_dim,
        probe_epochs=args.probe_epochs,
        probe_lr=args.probe_lr,
        test_fraction=args.test_fraction,
        ridge_alpha=args.ridge_alpha,
        param_bins=args.param_bins,
    )
    print(
        json.dumps(
            {
                "op_id_probe_accuracy": metrics["op_id_probe_accuracy"],
                "op_param_r2": metrics["op_param_r2"],
                "op_param_mse": metrics["op_param_mse"],
                "binned_param_accuracy": metrics["binned_param_accuracy"],
                "linear_op_id_probe_accuracy": metrics["linear_op_id_probe_accuracy"],
                "mlp_op_id_probe_accuracy": metrics["mlp_op_id_probe_accuracy"],
                "linear_binned_param_accuracy": metrics["linear_binned_param_accuracy"],
                "mlp_binned_param_accuracy": metrics["mlp_binned_param_accuracy"],
                "linear_op_param_r2": metrics["linear_op_param_r2"],
                "mlp_op_param_r2": metrics["mlp_op_param_r2"],
                "op_id_silhouette": metrics["op_id_silhouette"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
