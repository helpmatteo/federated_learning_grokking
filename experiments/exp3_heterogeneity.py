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


def run_exp3a_cell(alpha: float, dir_alpha: float, k: int = 10,
                   hidden_width: int = 256, t_max: int = 50000,
                   output_dir: str = "results/exp3_heterogeneity"):
    """Run a single (alpha, dir_alpha) cell for 3a."""
    rc = RunConfig(t_base=8000, t_max=t_max)
    num_rounds = rc.s_fl // LOCAL_EPOCHS

    print(f"\n{'='*70}")
    print(f"Exp 3a: alpha={alpha}, dir_alpha={dir_alpha}, K={k}")
    print(f"{'='*70}")

    cfg = FedConfig(
        task="addition", p=97, optimizer="gd", lr=50.0,
        weight_decay=0.0, momentum=0.0, hidden_width=hidden_width,
        alpha=alpha, num_clients=k, num_rounds=num_rounds,
        local_epochs=LOCAL_EPOCHS, fraction_train=1.0,
        partition="dirichlet", dirichlet_alpha=dir_alpha,
        strategy="fedavg",
        output_dir=f"{output_dir}/exp3a",
    )
    result = run_multi_seed(
        run_fn=run_single_federated,
        cfg_template=cfg,
        seeds=SEEDS,
        label=f"3a a={alpha} dir={dir_alpha} K={k}",
    )
    result["alpha"] = alpha
    result["dirichlet_alpha"] = dir_alpha
    result["K"] = k

    save_experiment_results(result, f"{output_dir}/exp3a_a{alpha}_dir{dir_alpha}_K{k}.json")
    return result


def run_exp3a(alphas: list, k_primary: int = 10,
              hidden_width: int = 256, t_max: int = 50000,
              output_dir: str = "results/exp3_heterogeneity"):
    """3a: Dirichlet sweep at K_primary."""
    results = []
    for alpha in alphas:
        for dir_alpha in DIRICHLET_ALPHAS:
            result = run_exp3a_cell(alpha, dir_alpha, k=k_primary,
                                    hidden_width=hidden_width, t_max=t_max,
                                    output_dir=output_dir)
            results.append(result)

    save_experiment_results(results, f"{output_dir}/exp3a_results.json")
    return results


def run_exp3a_k_validation(alphas: list = None, k_secondary: int = 20,
                           hidden_width: int = 256, t_max: int = 50000,
                           output_dir: str = "results/exp3_heterogeneity"):
    """3a K-validation: reduced sweep at K_secondary."""
    reduced_alphas = alphas if alphas else [0.25, 0.30, 0.50]
    reduced_dir = [0.1, 1.0, 1000.0]
    results = []

    for alpha in reduced_alphas:
        for dir_alpha in reduced_dir:
            result = run_exp3a_cell(alpha, dir_alpha, k=k_secondary,
                                    hidden_width=hidden_width, t_max=t_max,
                                    output_dir=output_dir)
            results.append(result)

    save_experiment_results(results, f"{output_dir}/exp3a_kval_results.json")
    return results


def run_exp3b_cell(alpha: float, partition: str, k: int = 10,
                   hidden_width: int = 256, t_max: int = 50000,
                   output_dir: str = "results/exp3_heterogeneity"):
    """Run a single (alpha, partition) cell for 3b."""
    rc = RunConfig(t_base=8000, t_max=t_max)
    num_rounds = rc.s_fl // LOCAL_EPOCHS

    print(f"\n{'='*70}")
    print(f"Exp 3b: alpha={alpha}, partition={partition}, K={k}")
    print(f"{'='*70}")

    cfg = FedConfig(
        task="addition", p=97, optimizer="gd", lr=50.0,
        weight_decay=0.0, momentum=0.0, hidden_width=hidden_width,
        alpha=alpha, num_clients=k, num_rounds=num_rounds,
        local_epochs=LOCAL_EPOCHS, fraction_train=1.0,
        partition=partition,
        strategy="fedavg",
        output_dir=f"{output_dir}/exp3b",
    )
    result = run_multi_seed(
        run_fn=run_single_federated,
        cfg_template=cfg,
        seeds=SEEDS,
        label=f"3b a={alpha} part={partition} K={k}",
    )
    result["alpha"] = alpha
    result["partition"] = partition
    result["K"] = k

    save_experiment_results(result, f"{output_dir}/exp3b_a{alpha}_{partition}_K{k}.json")
    return result


def run_exp3b(alphas: list, k_primary: int = 10,
              hidden_width: int = 256, t_max: int = 50000,
              output_dir: str = "results/exp3_heterogeneity"):
    """3b: Structured partition comparison (IID vs operand vs target)."""
    results = []
    for alpha in alphas:
        for partition in STRUCTURED_PARTITIONS:
            result = run_exp3b_cell(alpha, partition, k=k_primary,
                                    hidden_width=hidden_width, t_max=t_max,
                                    output_dir=output_dir)
            results.append(result)

    save_experiment_results(results, f"{output_dir}/exp3b_results.json")
    return results


if __name__ == "__main__":
    alphas = [0.20, 0.25, 0.30, 0.35, 0.50]
    run_exp3a(alphas)
    run_exp3a_k_validation(alphas)
    run_exp3b(alphas)
