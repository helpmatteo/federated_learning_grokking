"""Experiment 2: Aggregation effect & FL phase boundary.

Three conditions per (alpha, K):
  (a) Centralized-full: standard centralized with n_train
  (b) Centralized-reduced: centralized with n_train/K
  (c) FL (IID): K clients, FedAvg, full participation

Usage:
    # Run a single (alpha, K) cell — maximally parallelisable
    python run_experiment.py exp2 --alpha 0.25 --K 10 --t_max 50000

    # Run one alpha across all K values
    python run_experiment.py exp2 --alpha 0.25 --t_max 50000

    # Run everything (sequential)
    python run_experiment.py exp2 --alphas 0.20,0.25,0.30,0.35,0.50 --t_max 50000
"""

from dataclasses import replace
from core.config import Config
from federated.config import FedConfig
from experiments.runner import (
    run_single_centralized, run_single_federated,
    run_multi_seed, save_experiment_results, RunConfig,
)

DEFAULT_ALPHAS = [0.20, 0.25, 0.30, 0.35, 0.50]
K_VALUES = [2, 5, 10, 20, 50, 97]
SEEDS = [42, 123, 456]
LOCAL_EPOCHS = 5


def run_exp2_cell(alpha: float, K: int, hidden_width: int = 256,
                  t_max: int = 50000, output_dir: str = "results/exp2_aggregation"):
    """Run all three conditions for a single (alpha, K) pair.

    This is the atomic unit of work — call this in parallel across (alpha, K) pairs.
    Results are saved to per-cell JSON files that can be assembled later.
    """
    rc = RunConfig(t_base=8000, t_max=t_max)
    s_fl = rc.s_fl
    num_rounds = s_fl // LOCAL_EPOCHS

    print(f"\n{'='*70}")
    print(f"Exp 2: alpha={alpha}, K={K}")
    print(f"{'='*70}")

    # (a) Centralized-full
    cent_full_cfg = Config(
        task="addition", p=97, optimizer="gd", lr=50.0,
        weight_decay=0.0, momentum=0.0, hidden_width=hidden_width,
        epochs=100_000, log_every=100,
        alpha=alpha, output_dir=f"{output_dir}/cent_full",
    )
    result_a = run_multi_seed(
        run_fn=run_single_centralized,
        cfg_template=cent_full_cfg,
        seeds=SEEDS,
        label=f"cent_full a={alpha}",
    )

    # (b) Centralized-reduced: alpha_eff = alpha / K
    alpha_eff = alpha / K
    cent_red_cfg = Config(
        task="addition", p=97, optimizer="gd", lr=50.0,
        weight_decay=0.0, momentum=0.0, hidden_width=hidden_width,
        epochs=100_000, log_every=100,
        alpha=alpha_eff, output_dir=f"{output_dir}/cent_reduced",
    )
    result_b = run_multi_seed(
        run_fn=run_single_centralized,
        cfg_template=cent_red_cfg,
        seeds=SEEDS,
        label=f"cent_reduced a={alpha}/K={K} (eff={alpha_eff:.4f})",
    )

    # (c) FL (IID)
    fed_cfg = FedConfig(
        task="addition", p=97, optimizer="gd", lr=50.0,
        weight_decay=0.0, momentum=0.0, hidden_width=hidden_width,
        alpha=alpha, num_clients=K, num_rounds=num_rounds,
        local_epochs=LOCAL_EPOCHS, fraction_train=1.0,
        partition="iid", strategy="fedavg",
        output_dir=f"{output_dir}/fl_iid",
    )
    result_c = run_multi_seed(
        run_fn=run_single_federated,
        cfg_template=fed_cfg,
        seeds=SEEDS,
        label=f"fl_iid a={alpha} K={K}",
    )

    result = {
        "alpha": alpha,
        "K": K,
        "cent_full": result_a,
        "cent_reduced": result_b,
        "fl_iid": result_c,
    }

    save_experiment_results(result, f"{output_dir}/exp2_a{alpha}_K{K}.json")
    return result


def run_exp2(alphas: list = None, k_values: list = None, hidden_width: int = 256,
             t_max: int = 50000, output_dir: str = "results/exp2_aggregation"):
    """Run Exp 2 across all (alpha, K) pairs sequentially."""
    if alphas is None:
        alphas = DEFAULT_ALPHAS
    if k_values is None:
        k_values = K_VALUES

    all_results = []
    for alpha in alphas:
        for K in k_values:
            result = run_exp2_cell(alpha, K, hidden_width=hidden_width,
                                   t_max=t_max, output_dir=output_dir)
            all_results.append(result)

    save_experiment_results(all_results, f"{output_dir}/exp2_results.json")
    return all_results


if __name__ == "__main__":
    run_exp2()
