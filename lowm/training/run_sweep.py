"""Run LOWM-OCCL ablation sweeps."""

from __future__ import annotations

import argparse
import itertools
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml

from lowm.eval.evaluate_all import evaluate_run
from lowm.eval.ebtwm_inference import evaluate_ebtwm_inference
from lowm.eval.evaluate_coherence_stratification import evaluate_coherence_stratification
from lowm.eval.evaluate_cophy_ranking import evaluate_cophy_ranking
from lowm.eval.evaluate_energy_matrix import evaluate_energy_matrix
from lowm.eval.evaluate_law_mismatch_only import evaluate_law_mismatch_only
from lowm.eval.evaluate_occl_alignment import evaluate_occl_alignment
from lowm.training.train_baseline import train_baseline
from lowm.training.train_lowm import train_lowm


SWEEP_TO_CONFIG_PATH = {
    "alpha_occl": ("training", "alpha_occl"),
    "lambda_dim": ("model", "lambda_dim"),
    "use_pairwise_energy": ("model", "use_pairwise_energy"),
    "use_stability": ("training", "use_stability"),
    "beta_kl": ("training", "beta_kl"),
    "seed": ("training", "seed"),
    "negative_types": ("ranking", "negative_types"),
    "selection_metric": ("training", "selection_metric"),
    "use_occl": ("training", "use_occl"),
    "use_dsm": ("training", "use_dsm"),
    "alpha_dsm": ("training", "alpha_dsm"),
    "use_denoise_rank": ("training", "use_denoise_rank"),
    "alpha_denoise_rank": ("training", "alpha_denoise_rank"),
    "use_grad_reg": ("training", "use_grad_reg"),
    "alpha_grad_reg": ("training", "alpha_grad_reg"),
}
METADATA_ONLY_PARAMS = {"negative_set", "component", "name", "variant", "model_type", "scenario", "mode"}


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return loaded


def _set_nested(config: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    cursor = config
    for key in path[:-1]:
        cursor = cursor.setdefault(key, {})
    cursor[path[-1]] = value


def _deep_update(base: dict[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _fmt_value(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def run_name_from_params(params: Mapping[str, Any]) -> str:
    if params.get("variant"):
        model = _fmt_value(params.get("model_type", "lowm"))
        return f"{model}_{_fmt_value(params['variant'])}_seed{_fmt_value(params.get('seed', 0))}"
    if "alpha_dsm" in params:
        return f"lowm_ebtwm_alpha{_fmt_value(params.get('alpha_dsm'))}_seed{_fmt_value(params.get('seed', 0))}"
    if params.get("component"):
        return f"lowm_component_{_fmt_value(params['component'])}_seed{_fmt_value(params.get('seed', 0))}"
    if params.get("negative_set"):
        return (
            f"lowm_negs_{_fmt_value(params['negative_set'])}"
            f"_alpha{_fmt_value(params.get('alpha_occl', 1.0))}"
            f"_lambda{_fmt_value(params.get('lambda_dim', 16))}"
            f"_seed{_fmt_value(params.get('seed', 0))}"
        )
    prefix = (
        f"lowm_occl_alpha{_fmt_value(params.get('alpha_occl', 1.0))}"
        f"_lambda{_fmt_value(params.get('lambda_dim', 16))}"
        f"_seed{_fmt_value(params.get('seed', 0))}"
    )
    suffix = (
        f"_pair{_fmt_value(params.get('use_pairwise_energy', True))}"
        f"_stab{_fmt_value(params.get('use_stability', True))}"
        f"_beta{_fmt_value(params.get('beta_kl', 1e-4))}"
    )
    return prefix + suffix


def expand_sweep(sweep: Mapping[str, Any]) -> list[dict[str, Any]]:
    if "variants" in sweep:
        variants = sweep["variants"]
        if not isinstance(variants, list) or not variants:
            raise ValueError("variants must be a non-empty list")
        return [dict(variant) for variant in variants]
    values = dict(sweep.get("parameters", {}))
    if not values:
        raise ValueError("sweep config must define parameters")
    keys = list(values)
    combos = []
    for product in itertools.product(*[values[key] for key in keys]):
        combos.append(dict(zip(keys, product)))
    return combos


def build_run_config(base_config: Mapping[str, Any], params: Mapping[str, Any], sweep_dir: Path, overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    config = deepcopy(dict(base_config))
    if overrides:
        _deep_update(config, overrides)
    for key, value in params.items():
        if key in METADATA_ONLY_PARAMS:
            continue
        if key not in SWEEP_TO_CONFIG_PATH:
            raise ValueError(f"unsupported sweep parameter '{key}'")
        _set_nested(config, SWEEP_TO_CONFIG_PATH[key], value)
    seed = int(params.get("seed", config.get("training", {}).get("seed", config.get("seed", 0))))
    config["seed"] = seed
    _set_nested(config, ("training", "seed"), seed)
    _set_nested(config, ("training", "output_dir"), str(sweep_dir / "runs"))
    _set_nested(config, ("training", "run_name"), run_name_from_params(params))
    model_type = str(params.get("model_type", "lowm"))
    if model_type in {"fixed_energy", "direct_context_energy"}:
        _set_nested(config, ("training", "baseline"), model_type)
    else:
        _set_nested(config, ("training", "use_occl"), bool(params.get("use_occl", config.get("training", {}).get("use_occl", True))))
    config["sweep_params"] = dict(params)
    return config


def _write_run_config(config: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(dict(config), f, sort_keys=False)


def run_sweep(config_path: Path, dry_run: bool = False, max_runs: int | None = None) -> list[Path]:
    sweep = _load_yaml(config_path)
    base_config_path = Path(sweep.get("base_config", "configs/train_lowm_occl.yaml"))
    base_config = _load_yaml(base_config_path)
    sweep_dir = Path(sweep.get("sweep_dir", "runs/lowm_synth_v0/lowm_occl_ablation"))
    eval_cfg = dict(sweep.get("evaluation", {}))
    split = str(eval_cfg.get("split", "val"))
    raw_splits = eval_cfg.get("splits")
    splits = [str(item) for item in raw_splits] if isinstance(raw_splits, list) and raw_splits else [split]
    eval_num_samples = eval_cfg.get("num_samples")
    eval_batch_size = eval_cfg.get("batch_size")
    device = str(eval_cfg.get("device", "auto"))
    ranking_checkpoint = str(eval_cfg.get("ranking_checkpoint", "best_law_pair.pt"))
    law_checkpoint = str(eval_cfg.get("law_checkpoint", "best_law_pair.pt"))
    occl_checkpoint = str(eval_cfg.get("occl_checkpoint", "best_occl_acc.pt"))
    run_ebtwm = bool(eval_cfg.get("ebtwm_inference", False))
    run_coherence = bool(eval_cfg.get("coherence_stratification", False))
    run_energy_matrix = bool(eval_cfg.get("energy_matrix", False))
    run_cophy = bool(eval_cfg.get("cophy_ranking", False))
    ebtwm_cfg = dict(eval_cfg.get("ebtwm", {}))
    coherence_cfg = dict(eval_cfg.get("coherence", {}))
    energy_matrix_cfg = dict(eval_cfg.get("energy_matrix_config", {}))
    cophy_cfg = dict(eval_cfg.get("cophy", {}))
    evaluate = bool(eval_cfg.get("enabled", True))

    configs_dir = sweep_dir / "configs"
    run_paths: list[Path] = []
    combos = expand_sweep(sweep)
    if max_runs is not None:
        combos = combos[:max_runs]

    for params in combos:
        run_config = build_run_config(base_config, params, sweep_dir, sweep.get("overrides", {}))
        run_name = str(run_config["training"]["run_name"])
        generated_config = configs_dir / f"{run_name}.yaml"
        _write_run_config(run_config, generated_config)
        run_dir = Path(run_config["training"]["output_dir"]) / run_name
        run_paths.append(run_dir)
        print(f"prepared {run_name}")
        if dry_run or bool(sweep.get("dry_run", False)):
            continue
        model_type = str(params.get("model_type", "lowm"))
        if model_type in {"fixed_energy", "direct_context_energy"}:
            train_baseline(generated_config)
        else:
            train_lowm(generated_config)
        if evaluate:
            for eval_split in splits:
                evaluate_run(run_dir, split=eval_split, checkpoint_name=ranking_checkpoint, num_samples=eval_num_samples, batch_size=eval_batch_size, device_name=device)
                evaluate_law_mismatch_only(run_dir, split=eval_split, checkpoint_name=law_checkpoint, num_samples=eval_num_samples, batch_size=eval_batch_size, device_name=device)
                if run_coherence:
                    evaluate_coherence_stratification(
                        run_dir,
                        split=eval_split,
                        checkpoint_name=str(coherence_cfg.get("checkpoint", ranking_checkpoint)),
                        num_samples=coherence_cfg.get("num_samples", eval_num_samples),
                        batch_size=coherence_cfg.get("batch_size", eval_batch_size),
                        seed=coherence_cfg.get("seed"),
                        device_name=device,
                    )
                if run_energy_matrix:
                    evaluate_energy_matrix(
                        run_dir,
                        split=eval_split,
                        checkpoint_name=str(energy_matrix_cfg.get("checkpoint", ranking_checkpoint)),
                        matrix_size=int(energy_matrix_cfg.get("matrix_size", 16)),
                        max_batches=int(energy_matrix_cfg.get("max_batches", 8)),
                        num_samples=energy_matrix_cfg.get("num_samples"),
                        seed=energy_matrix_cfg.get("seed"),
                        device_name=device,
                    )
                if run_cophy:
                    evaluate_cophy_ranking(
                        run_dir,
                        split=eval_split,
                        scenario=cophy_cfg.get("scenario"),
                        checkpoint_name=str(cophy_cfg.get("checkpoint", ranking_checkpoint)),
                        num_samples=cophy_cfg.get("num_samples", eval_num_samples),
                        batch_size=cophy_cfg.get("batch_size", eval_batch_size),
                        seed=cophy_cfg.get("seed"),
                        matrix_size=int(cophy_cfg.get("matrix_size", 16)),
                        max_batches=int(cophy_cfg.get("max_batches", 8)),
                        device_name=device,
                    )
                if model_type == "lowm":
                    evaluate_occl_alignment(run_dir, split=eval_split, checkpoint_name=occl_checkpoint, num_samples=eval_num_samples, batch_size=eval_batch_size, device_name=device)
                if run_ebtwm and model_type == "lowm":
                    evaluate_ebtwm_inference(
                        run_dir,
                        split=eval_split,
                        checkpoint=ranking_checkpoint,
                        num_samples=int(ebtwm_cfg.get("num_samples", eval_num_samples or 100)),
                        num_steps=int(ebtwm_cfg.get("num_steps", 100)),
                        step_size=float(ebtwm_cfg.get("step_size", 1e-2)),
                        noise_std=float(ebtwm_cfg.get("noise_std", 0.05)),
                        corruption_type=str(ebtwm_cfg.get("corruption_type", "gaussian")),
                        horizon=ebtwm_cfg.get("horizon"),
                        device=device,
                        mode=str(ebtwm_cfg.get("mode", "denoise")),
                        eta_smooth=float(ebtwm_cfg.get("eta_smooth", 0.01)),
                        eta_anchor=float(ebtwm_cfg.get("eta_anchor", 0.01)),
                        eta_bounds=float(ebtwm_cfg.get("eta_bounds", 0.0)),
                        langevin_noise=float(ebtwm_cfg.get("langevin_noise", 0.0)),
                        seed=int(ebtwm_cfg.get("seed", 0)),
                        skip_preflight=bool(ebtwm_cfg.get("skip_preflight", False)),
                    )

    manifest = {"config": str(config_path), "runs": [str(path) for path in run_paths]}
    sweep_dir.mkdir(parents=True, exist_ok=True)
    with (sweep_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    return run_paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-runs", type=int, default=None)
    args = parser.parse_args()
    paths = run_sweep(args.config, dry_run=args.dry_run, max_runs=args.max_runs)
    print(f"prepared {len(paths)} runs")


if __name__ == "__main__":
    main()
