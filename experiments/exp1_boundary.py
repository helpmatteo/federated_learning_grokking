"""Experiment 1: Centralized grokking phase boundary.

Design: alpha in {0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5}
seeds = {42, 123, 456}, S_max = 100,000 steps. 24 runs.
"""

from dataclasses import replace
from fedgrok.core.config import Config
from fedgrok.training.runner import run_single_centralized, run_multi_seed, save_experiment_results


ALPHAS = [0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5]
SEEDS = [42, 123, 456]
S_MAX = 100_000


def run_exp1(hidden_width: int = 256,
             output_dir: str = "results/experiments/exp1_boundary"):
    """Run centralized phase boundary experiment.

    Returns dict with alpha_crit, t_max, and per-alpha results.
    """
    base_cfg = Config(
        task="addition",
        p=97,
        optimizer="gd",
        lr=50.0,
        weight_decay=0.0,
        momentum=0.0,
        hidden_width=hidden_width,
        epochs=S_MAX,
        log_every=100,
        output_dir=output_dir,
    )

    all_results = []

    for alpha in ALPHAS:
        result = run_multi_seed(
            run_fn=run_single_centralized,
            cfg_template=replace(base_cfg, alpha=alpha),
            seeds=SEEDS,
            label=f"alpha={alpha}",
        )
        result["alpha"] = alpha
        all_results.append(result)

    # Determine alpha_crit: smallest alpha where all seeds grok
    alpha_crit = None
    for r in sorted(all_results, key=lambda x: x["alpha"]):
        if r["summary"]["n_grokked"] == len(SEEDS):
            alpha_crit = r["alpha"]
            break

    # T_max: max T_grok across all alpha that grok
    t_max = 0
    for r in all_results:
        for seed_result in r["per_seed"]:
            if seed_result["t_grok"] < float("inf"):
                t_max = max(t_max, int(seed_result["t_grok"]))

    print(f"\n{'='*60}")
    print(f"EXPERIMENT 1 RESULTS")
    print(f"  alpha_crit = {alpha_crit}")
    print(f"  T_max = {t_max}")
    print(f"{'='*60}")

    output = {
        "alpha_crit": alpha_crit,
        "t_max": t_max,
        "per_alpha": all_results,
    }
    save_experiment_results(output, f"{output_dir}/exp1_results.json")
    return output


if __name__ == "__main__":
    run_exp1()
