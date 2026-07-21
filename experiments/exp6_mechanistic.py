"""Experiment 6: Mechanistic analysis (post-hoc on runs from Exps 1-5).

No new training runs. Selects representative runs and produces analysis data.
"""

import json
import os
import glob
import numpy as np


def load_history(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def select_representative_runs(exp_results_dir: str) -> dict:
    """Select 4 representative runs for mechanistic analysis."""
    return {
        "healthy_grok": None,
        "boundary_grok": None,
        "failed_grok": None,
        "rescued_grok": None,
    }


def analyze_drift_vs_grokking(results_dirs: list, output_dir: str):
    """Cross-run scatter plot data: mean client drift vs grokking step."""
    from fedgrok.analysis.grokking_metrics import compute_t_grok

    data_points = []
    for d in results_dirs:
        for path in glob.glob(os.path.join(d, "**", "history_fed_*.json"), recursive=True):
            h = load_history(path)
            if "mean_client_drift" not in h or not h["mean_client_drift"]:
                continue
            steps = h.get("total_steps", [])
            test_accs = h.get("test_acc", [])
            t_grok = compute_t_grok(steps, test_accs)
            mean_drift = float(np.mean(h["mean_client_drift"]))
            data_points.append({
                "path": path,
                "t_grok": t_grok,
                "mean_drift": mean_drift,
            })

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "drift_vs_grokking.json"), "w") as f:
        json.dump(data_points, f, indent=2, default=str)

    print(f"Collected {len(data_points)} data points for drift vs grokking analysis")
    return data_points


if __name__ == "__main__":
    analyze_drift_vs_grokking(
        results_dirs=["results/experiments/exp2_aggregation", "results/experiments/exp3_heterogeneity",
                       "results/experiments/exp4_optimization", "results/experiments/exp5_algorithms"],
        output_dir="results/experiments/exp6_mechanistic",
    )
