"""Multi-seed experiment runner with adaptive step budgets.

NOTE: an early-abort mechanism (`should_abort`) previously lived here. It was
unit-tested but never wired into any experiment, so no run was ever truncated
by it. It has been removed rather than enabled: every run in the archived
results went to its full budget, and keeping the analysis free of
truncation bias is worth more than the compute an abort rule would save.
Runs that do not grok within budget are handled as right-censored
observations in the survival statistics instead.
"""

import json
import math
import os
from dataclasses import dataclass, replace

import numpy as np

from fedgrok.analysis.grokking_metrics import extract_grokking_results, summarize_seeds


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


def run_single_centralized(cfg, label: str = "") -> dict:
    """Run one centralized experiment, return history + grokking metrics."""
    from fedgrok.training.centralized import train
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
    from fedgrok.training.federated import fed_train
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
