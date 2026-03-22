"""Experiment 4: Optimization fragmentation.

4a: Drift accumulation x heterogeneity (vary E, fixed S)
4b: Partial participation x heterogeneity (vary f, fixed E)
4c: Compute vs communication budget (fixed R, vary E)

Usage:
    # Single cell (maximally parallelisable)
    python run_experiment.py exp4a --alpha 0.30 --E 10 --het iid --t_max 50000
    python run_experiment.py exp4b --alpha 0.30 --frac 0.4 --het noniid --t_max 50000
    python run_experiment.py exp4c --alpha 0.30 --E 10 --t_max 50000

    # Full sweep (sequential)
    python run_experiment.py exp4a --alphas 0.25,0.30,0.50 --t_max 50000
"""

from federated.config import FedConfig
from experiments.runner import (
    run_single_federated, run_multi_seed,
    save_experiment_results, RunConfig,
)

DEFAULT_ALPHAS = [0.25, 0.30, 0.50]
SEEDS = [42, 123, 456]
NONIID_SETTING = {"partition": "dirichlet", "dirichlet_alpha": 0.1}


def run_exp4a_cell(alpha: float, E: int, het: str, k: int = 10,
                   hidden_width: int = 256, t_max: int = 50000,
                   output_dir: str = "results/exp4_optimization"):
    """Run a single (alpha, E, het) cell for 4a."""
    rc = RunConfig(t_base=8000, t_max=t_max)
    s_fl = rc.s_fl
    num_rounds = s_fl // E

    het_cfg = {"partition": "iid"} if het == "iid" else NONIID_SETTING

    print(f"\n{'='*70}")
    print(f"Exp 4a: alpha={alpha}, E={E}, het={het}, K={k}, R={num_rounds}")
    print(f"{'='*70}")

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
        label=f"4a a={alpha} E={E} {het}",
    )
    result["alpha"] = alpha
    result["E"] = E
    result["heterogeneity"] = het

    save_experiment_results(result, f"{output_dir}/exp4a_a{alpha}_E{E}_{het}.json")
    return result


def run_exp4a(alphas: list = None, k: int = 10,
              hidden_width: int = 256, t_max: int = 50000,
              output_dir: str = "results/exp4_optimization"):
    """4a: Drift accumulation x heterogeneity. Fixed total S, vary E."""
    if alphas is None:
        alphas = DEFAULT_ALPHAS
    e_values = [5, 10, 25, 50]
    results = []

    for alpha in alphas:
        for E in e_values:
            for het in ["iid", "noniid"]:
                result = run_exp4a_cell(alpha, E, het, k=k,
                                        hidden_width=hidden_width, t_max=t_max,
                                        output_dir=output_dir)
                results.append(result)

    save_experiment_results(results, f"{output_dir}/exp4a_results.json")
    return results


def run_exp4b_cell(alpha: float, frac: float, het: str, k: int = 10,
                   hidden_width: int = 256, t_max: int = 50000,
                   output_dir: str = "results/exp4_optimization"):
    """Run a single (alpha, f, het) cell for 4b."""
    rc = RunConfig(t_base=8000, t_max=t_max)
    s_fl = rc.s_fl
    E = 5
    num_rounds = s_fl // E

    het_cfg = {"partition": "iid"} if het == "iid" else NONIID_SETTING

    print(f"\n{'='*70}")
    print(f"Exp 4b: alpha={alpha}, f={frac}, het={het}, K={k}, R={num_rounds}")
    print(f"{'='*70}")

    cfg = FedConfig(
        task="addition", p=97, optimizer="gd", lr=50.0,
        weight_decay=0.0, momentum=0.0, hidden_width=hidden_width,
        alpha=alpha, num_clients=k, num_rounds=num_rounds,
        local_epochs=E, fraction_train=frac,
        strategy="fedavg",
        output_dir=f"{output_dir}/exp4b",
        **het_cfg,
    )
    result = run_multi_seed(
        run_fn=run_single_federated,
        cfg_template=cfg,
        seeds=SEEDS,
        label=f"4b a={alpha} f={frac} {het}",
    )
    result["alpha"] = alpha
    result["f"] = frac
    result["heterogeneity"] = het

    save_experiment_results(result, f"{output_dir}/exp4b_a{alpha}_f{frac}_{het}.json")
    return result


def run_exp4b(alphas: list = None, k: int = 10,
              hidden_width: int = 256, t_max: int = 50000,
              output_dir: str = "results/exp4_optimization"):
    """4b: Partial participation x heterogeneity. Fixed E=5, vary f."""
    if alphas is None:
        alphas = DEFAULT_ALPHAS
    f_values = [0.2, 0.4, 0.6, 1.0]
    results = []

    for alpha in alphas:
        for f in f_values:
            for het in ["iid", "noniid"]:
                result = run_exp4b_cell(alpha, f, het, k=k,
                                        hidden_width=hidden_width, t_max=t_max,
                                        output_dir=output_dir)
                results.append(result)

    save_experiment_results(results, f"{output_dir}/exp4b_results.json")
    return results


def run_exp4c_cell(alpha: float, E: int, k: int = 10,
                   hidden_width: int = 256,
                   output_dir: str = "results/exp4_optimization"):
    """Run a single (alpha, E) cell for 4c."""
    R = 2000

    print(f"\n{'='*70}")
    print(f"Exp 4c: alpha={alpha}, E={E}, R={R}, S={R*E}, K={k}")
    print(f"{'='*70}")

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

    save_experiment_results(result, f"{output_dir}/exp4c_a{alpha}_E{E}.json")
    return result


def run_exp4c(alphas: list = None, k: int = 10,
              hidden_width: int = 256,
              output_dir: str = "results/exp4_optimization"):
    """4c: Compute vs communication. Fixed R=2000, IID, vary E."""
    if alphas is None:
        alphas = DEFAULT_ALPHAS
    e_values = [1, 5, 10, 25]
    results = []

    for alpha in alphas:
        for E in e_values:
            result = run_exp4c_cell(alpha, E, k=k,
                                    hidden_width=hidden_width,
                                    output_dir=output_dir)
            results.append(result)

    save_experiment_results(results, f"{output_dir}/exp4c_results.json")
    return results


if __name__ == "__main__":
    alphas = DEFAULT_ALPHAS
    run_exp4a(alphas)
    run_exp4b(alphas)
    run_exp4c(alphas)
