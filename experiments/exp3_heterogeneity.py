"""Experiment 3: Heterogeneity at the phase boundary.

3a: Dirichlet sweep (continuous heterogeneity knob)
3b: Structured partition (operand vs target vs IID)
"""

from dataclasses import replace
from federated.config import FedConfig
from experiments.runner import (
    run_single_federated, run_multi_seed,
    save_experiment_results, RunConfig,
)

SEEDS = [42, 123, 456]
DIRICHLET_ALPHAS = [0.01, 0.1, 0.5, 1.0, 10.0, 1000.0]
STRUCTURED_PARTITIONS = ["iid", "operand", "target"]
LOCAL_EPOCHS = 5


def run_exp3a(alphas: list, k_primary: int = 10,
              hidden_width: int = 256, t_max: int = 30000,
              output_dir: str = "results/exp3_heterogeneity"):
    """3a: Dirichlet sweep at K_primary."""
    rc = RunConfig(t_base=8000, t_max=t_max)
    s_fl = rc.s_fl
    num_rounds = s_fl // LOCAL_EPOCHS
    results = []

    for alpha in alphas:
        for dir_alpha in DIRICHLET_ALPHAS:
            cfg = FedConfig(
                task="addition", p=97, optimizer="gd", lr=50.0,
                weight_decay=0.0, momentum=0.0, hidden_width=hidden_width,
                alpha=alpha, num_clients=k_primary, num_rounds=num_rounds,
                local_epochs=LOCAL_EPOCHS, fraction_train=1.0,
                partition="dirichlet", dirichlet_alpha=dir_alpha,
                strategy="fedavg",
                output_dir=f"{output_dir}/exp3a",
            )
            result = run_multi_seed(
                run_fn=run_single_federated,
                cfg_template=cfg,
                seeds=SEEDS,
                label=f"3a a={alpha} dir={dir_alpha} K={k_primary}",
            )
            result["alpha"] = alpha
            result["dirichlet_alpha"] = dir_alpha
            result["K"] = k_primary
            results.append(result)

    save_experiment_results(results, f"{output_dir}/exp3a_results.json")
    return results


def run_exp3a_k_validation(alphas: list, k_secondary: int = 50,
                           hidden_width: int = 256, t_max: int = 30000,
                           output_dir: str = "results/exp3_heterogeneity"):
    """3a K-validation: reduced sweep at K_secondary."""
    rc = RunConfig(t_base=8000, t_max=t_max)
    reduced_alphas = alphas[:3]
    reduced_dir = [0.1, 1.0, 1000.0]
    num_rounds = rc.s_fl // LOCAL_EPOCHS
    results = []

    for alpha in reduced_alphas:
        for dir_alpha in reduced_dir:
            cfg = FedConfig(
                task="addition", p=97, optimizer="gd", lr=50.0,
                weight_decay=0.0, momentum=0.0, hidden_width=hidden_width,
                alpha=alpha, num_clients=k_secondary, num_rounds=num_rounds,
                local_epochs=LOCAL_EPOCHS, fraction_train=1.0,
                partition="dirichlet", dirichlet_alpha=dir_alpha,
                strategy="fedavg",
                output_dir=f"{output_dir}/exp3a_kval",
            )
            result = run_multi_seed(
                run_fn=run_single_federated,
                cfg_template=cfg,
                seeds=SEEDS,
                label=f"3a-kval a={alpha} dir={dir_alpha} K={k_secondary}",
            )
            result["alpha"] = alpha
            result["dirichlet_alpha"] = dir_alpha
            result["K"] = k_secondary
            results.append(result)

    save_experiment_results(results, f"{output_dir}/exp3a_kval_results.json")
    return results


def run_exp3b(alphas: list, k_primary: int = 10,
              hidden_width: int = 256, t_max: int = 30000,
              output_dir: str = "results/exp3_heterogeneity"):
    """3b: Structured partition comparison (IID vs operand vs target)."""
    rc = RunConfig(t_base=8000, t_max=t_max)
    num_rounds = rc.s_fl // LOCAL_EPOCHS
    results = []

    for alpha in alphas:
        for partition in STRUCTURED_PARTITIONS:
            cfg = FedConfig(
                task="addition", p=97, optimizer="gd", lr=50.0,
                weight_decay=0.0, momentum=0.0, hidden_width=hidden_width,
                alpha=alpha, num_clients=k_primary, num_rounds=num_rounds,
                local_epochs=LOCAL_EPOCHS, fraction_train=1.0,
                partition=partition,
                strategy="fedavg",
                output_dir=f"{output_dir}/exp3b",
            )
            result = run_multi_seed(
                run_fn=run_single_federated,
                cfg_template=cfg,
                seeds=SEEDS,
                label=f"3b a={alpha} part={partition} K={k_primary}",
            )
            result["alpha"] = alpha
            result["partition"] = partition
            result["K"] = k_primary
            results.append(result)

    save_experiment_results(results, f"{output_dir}/exp3b_results.json")
    return results


if __name__ == "__main__":
    alphas = [0.1, 0.15, 0.2, 0.3, 0.5]
    run_exp3a(alphas)
    run_exp3a_k_validation(alphas)
    run_exp3b(alphas)
