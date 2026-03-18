"""Experiment 0: Width validation — verify N=256 works across alpha range.

Design: N in {100, 128, 200, 256}, alpha in {0.1, 0.3, 0.5}, seeds={42}
S_max = 50,000 steps, centralized only. 12 runs total.
"""

from dataclasses import replace
from core.config import Config
from experiments.runner import run_single_centralized, save_experiment_results


WIDTHS = [100, 128, 200, 256]
ALPHAS = [0.1, 0.3, 0.5]
SEEDS = [42]
S_MAX = 50_000


def run_exp0(output_dir: str = "results/exp0_width"):
    """Run width validation experiment."""
    base_cfg = Config(
        task="addition",
        p=97,
        optimizer="gd",
        lr=50.0,
        weight_decay=0.0,
        momentum=0.0,
        log_every=100,
        output_dir=output_dir,
    )

    all_results = []

    for alpha in ALPHAS:
        for width in WIDTHS:
            for seed in SEEDS:
                cfg = replace(
                    base_cfg,
                    alpha=alpha,
                    hidden_width=width,
                    epochs=S_MAX,
                    seed=seed,
                )
                label = f"N={width}_a={alpha}_s={seed}"
                result = run_single_centralized(cfg, label=label)
                result["alpha"] = alpha
                result["width"] = width
                result["seed"] = seed
                all_results.append(result)
                print(f"  N={width}, alpha={alpha}: T_grok={result['t_grok']}, "
                      f"T_50={result['t_50']}, final_acc={result['final_test_acc']:.1f}%")

    save_experiment_results(all_results, f"{output_dir}/exp0_results.json")

    # Determine recommended N*
    for width in WIDTHS:
        width_results = [r for r in all_results if r["width"] == width]
        all_grok = all(r["t_grok"] < float("inf") for r in width_results)
        if all_grok:
            print(f"\nRecommended N* = {width} (groks at all tested alphas)")
            break

    return all_results


if __name__ == "__main__":
    run_exp0()
