"""Experiment 2: Aggregation effect & FL phase boundary.

Three conditions per (alpha, K):
  (a) Centralized-full: standard centralized with n_train
  (b) Centralized-reduced: centralized with n_train/K
  (c) FL (IID): K clients, FedAvg, full participation
"""

from dataclasses import replace
from core.config import Config
from federated.config import FedConfig
from experiments.runner import (
    run_single_centralized, run_single_federated,
    run_multi_seed, save_experiment_results, RunConfig,
)

DEFAULT_ALPHAS = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
K_VALUES = [2, 5, 10, 20, 50, 97]
SEEDS = [42, 123, 456]
LOCAL_EPOCHS = 5


def run_exp2(alphas: list = None, hidden_width: int = 256,
             t_max: int = 30000, output_dir: str = "results/exp2_aggregation"):
    if alphas is None:
        alphas = DEFAULT_ALPHAS

    rc = RunConfig(t_base=8000, t_max=t_max)
    s_fl = rc.s_fl
    num_rounds = s_fl // LOCAL_EPOCHS

    all_results = []

    # (a) Centralized-full: K-independent — run once per alpha, cache
    cent_full_cache = {}
    for alpha in alphas:
        cent_full_cfg = Config(
            task="addition", p=97, optimizer="gd", lr=50.0,
            weight_decay=0.0, momentum=0.0, hidden_width=hidden_width,
            epochs=100_000, log_every=100,
            alpha=alpha, output_dir=f"{output_dir}/cent_full",
        )
        cent_full_cache[alpha] = run_multi_seed(
            run_fn=run_single_centralized,
            cfg_template=cent_full_cfg,
            seeds=SEEDS,
            label=f"cent_full a={alpha}",
        )

    for alpha in alphas:
        for K in K_VALUES:
            print(f"\n{'='*70}")
            print(f"Exp 2: alpha={alpha}, K={K}")
            print(f"{'='*70}")

            result_a = cent_full_cache[alpha]

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

            all_results.append({
                "alpha": alpha,
                "K": K,
                "cent_full": result_a,
                "cent_reduced": result_b,
                "fl_iid": result_c,
            })

    save_experiment_results(all_results, f"{output_dir}/exp2_results.json")
    return all_results


if __name__ == "__main__":
    run_exp2()
