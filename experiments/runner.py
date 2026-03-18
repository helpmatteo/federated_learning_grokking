"""Multi-seed experiment runner with early abort and adaptive budgets."""

import json
import math
import os
from dataclasses import dataclass, replace

import numpy as np

from experiments.grokking_metrics import extract_grokking_results, summarize_seeds


@dataclass
class RunConfig:
    """Adaptive step budgets derived from T_base and T_max (Section 3.2)."""
    t_base: int = 8000
    t_max: int = 30000

    @property
    def s_fl(self) -> int:
        raw = math.ceil(1.5 * self.t_max / 1000) * 1000
        return min(50_000, raw)

    @property
    def s_rescue(self) -> int:
        return min(80_000, 2 * self.t_max)


def should_abort(step: int, train_acc: float, test_acc: float,
                 t_base: int, t_max: int) -> bool:
    """Check early abort conditions (Section 3.2).

    Rule 1: Memorisation failure — train_acc < 50% by min(2*T_base, 15000)
    Rule 2: Generalisation hopeless — train_acc ~100% and test_acc < 5% by T_max
    """
    mem_deadline = min(2 * t_base, 15000)
    if step >= mem_deadline and train_acc < 50.0:
        return True

    if step >= t_max and train_acc >= 99.9 and test_acc < 5.0:
        return True

    return False


def run_single_centralized(cfg, label: str = "") -> dict:
    """Run one centralized experiment, return history + grokking metrics."""
    from centralized.train import train
    print(f"\n--- Centralized run: {label} seed={cfg.seed} ---")
    history, model = train(cfg)
    metrics = extract_grokking_results(history)
    metrics["history_path"] = os.path.join(
        cfg.output_dir,
        f"history_{cfg.task}_{cfg.optimizer}_p{cfg.p}_N{cfg.hidden_width}_a{cfg.alpha}_s{cfg.seed}.json"
    )
    return metrics


def run_single_federated(cfg, label: str = "") -> dict:
    """Run one federated experiment, return history + grokking metrics."""
    from federated.train import fed_train
    print(f"\n--- Federated run: {label} seed={cfg.seed} ---")
    history, model = fed_train(cfg)
    metrics = extract_grokking_results(history)
    return metrics


def run_multi_seed(run_fn, cfg_template, seeds: list, cfg_overrides: dict = None,
                   label: str = "") -> dict:
    """Run an experiment across multiple seeds and aggregate results."""
    per_seed = []
    for seed in seeds:
        overrides = {"seed": seed}
        if cfg_overrides:
            overrides.update(cfg_overrides)
        cfg = replace(cfg_template, **overrides)
        result = run_fn(cfg, label=f"{label} seed={seed}")
        per_seed.append(result)
        print(f"  -> T_grok={result['t_grok']}, T_50={result['t_50']}, "
              f"final_acc={result['final_test_acc']:.1f}%")

    summary = summarize_seeds(per_seed)
    return {
        "label": label,
        "per_seed": per_seed,
        "summary": summary,
    }


def save_experiment_results(results, output_path: str):
    """Save experiment results to JSON, handling inf values."""
    def _sanitize(obj):
        if isinstance(obj, float) and math.isinf(obj):
            return "inf"
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitize(v) for v in obj]
        return obj

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(_sanitize(results), f, indent=2)
    print(f"Results saved to {output_path}")
