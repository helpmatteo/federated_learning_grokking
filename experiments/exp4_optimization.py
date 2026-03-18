"""Experiment 4: Optimization fragmentation.

4a: Drift accumulation x heterogeneity (vary E, fixed S)
4b: Partial participation x heterogeneity (vary f, fixed E)
4c: Compute vs communication budget (fixed R, vary E)
"""

from dataclasses import replace
from federated.config import FedConfig
from experiments.runner import (
    run_single_federated, run_multi_seed,
    save_experiment_results, RunConfig,
)

SEEDS = [42, 123, 456]


def run_exp4a(alphas: list, k: int = 10,
              noniid_setting: dict = None,
              hidden_width: int = 256, t_max: int = 30000,
              output_dir: str = "results/exp4_optimization"):
    """4a: Drift accumulation x heterogeneity. Fixed total S, vary E."""
    rc = RunConfig(t_base=8000, t_max=t_max)
    s_fl = rc.s_fl
    e_values = [5, 10, 25, 50]

    if noniid_setting is None:
        noniid_setting = {"partition": "dirichlet", "dirichlet_alpha": 1.0}

    results = []

    for alpha in alphas:
        for E in e_values:
            num_rounds = s_fl // E
            for het_label, het_cfg in [("iid", {"partition": "iid"}),
                                        ("noniid", noniid_setting)]:
                cfg = FedConfig(
                    task="addition", p=97, optimizer="gd", lr=50.0,
                    weight_decay=0.0, momentum=0.0, hidden_width=hidden_width,
                    alpha=alpha, num_clients=k, num_rounds=num_rounds,
                    local_epochs=E, fraction_train=1.0,
                    strategy="fedavg",
                    output_dir=f"{output_dir}/exp4a",
                    **het_cfg,
                )
                result = run_multi_seed(
                    run_fn=run_single_federated,
                    cfg_template=cfg,
                    seeds=SEEDS,
                    label=f"4a a={alpha} E={E} {het_label}",
                )
                result["alpha"] = alpha
                result["E"] = E
                result["heterogeneity"] = het_label
                results.append(result)

    save_experiment_results(results, f"{output_dir}/exp4a_results.json")
    return results


def run_exp4b(alphas: list, k: int = 10,
              noniid_setting: dict = None,
              hidden_width: int = 256, t_max: int = 30000,
              output_dir: str = "results/exp4_optimization"):
    """4b: Partial participation x heterogeneity. Fixed E=5, vary f."""
    rc = RunConfig(t_base=8000, t_max=t_max)
    s_fl = rc.s_fl
    E = 5
    num_rounds = s_fl // E
    f_values = [0.2, 0.4, 0.6, 1.0]

    if noniid_setting is None:
        noniid_setting = {"partition": "dirichlet", "dirichlet_alpha": 1.0}

    results = []

    for alpha in alphas:
        for f in f_values:
            for het_label, het_cfg in [("iid", {"partition": "iid"}),
                                        ("noniid", noniid_setting)]:
                cfg = FedConfig(
                    task="addition", p=97, optimizer="gd", lr=50.0,
                    weight_decay=0.0, momentum=0.0, hidden_width=hidden_width,
                    alpha=alpha, num_clients=k, num_rounds=num_rounds,
                    local_epochs=E, fraction_train=f,
                    strategy="fedavg",
                    output_dir=f"{output_dir}/exp4b",
                    **het_cfg,
                )
                result = run_multi_seed(
                    run_fn=run_single_federated,
                    cfg_template=cfg,
                    seeds=SEEDS,
                    label=f"4b a={alpha} f={f} {het_label}",
                )
                result["alpha"] = alpha
                result["f"] = f
                result["heterogeneity"] = het_label
                results.append(result)

    save_experiment_results(results, f"{output_dir}/exp4b_results.json")
    return results


def run_exp4c(alphas: list, k: int = 10,
              hidden_width: int = 256,
              output_dir: str = "results/exp4_optimization"):
    """4c: Compute vs communication. Fixed R=2000, IID, vary E."""
    R = 2000
    e_values = [1, 5, 10, 25]
    results = []

    for alpha in alphas:
        for E in e_values:
            cfg = FedConfig(
                task="addition", p=97, optimizer="gd", lr=50.0,
                weight_decay=0.0, momentum=0.0, hidden_width=hidden_width,
                alpha=alpha, num_clients=k, num_rounds=R,
                local_epochs=E, fraction_train=1.0,
                partition="iid", strategy="fedavg",
                output_dir=f"{output_dir}/exp4c",
            )
            result = run_multi_seed(
                run_fn=run_single_federated,
                cfg_template=cfg,
                seeds=SEEDS,
                label=f"4c a={alpha} E={E} R={R} S={R*E}",
            )
            result["alpha"] = alpha
            result["E"] = E
            result["R"] = R
            result["S"] = R * E
            results.append(result)

    save_experiment_results(results, f"{output_dir}/exp4c_results.json")
    return results


if __name__ == "__main__":
    alphas = [0.2, 0.3, 0.5]
    run_exp4a(alphas)
    run_exp4b(alphas)
    run_exp4c(alphas)
