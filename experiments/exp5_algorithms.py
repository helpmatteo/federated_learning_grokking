"""Experiment 5: Algorithm comparison & rescue.

For hard settings from Exps 2-4 where FedAvg fails or is delayed,
test: FedAvg, FedProx (mu sweep), FedAdam (server_lr sweep), FedAvg+WD.

Hard settings (from Exp 2-4 results):
  H1: α=0.25, K=10, E=25, IID          — 2/3 grok (easy rescue)
  H2: α=0.25, K=10, E=25, non-IID      — 1/3 grok (hard rescue)
  H3: α=0.30, K=10, E=50, non-IID      — 3/3 at T=31.8k (acceleration)

Usage:
    # Single cell
    python run_experiment.py exp5 --setting H1 --algorithm FedProx-0.01 --t_max 50000

    # Full sweep (sequential)
    python run_experiment.py exp5 --t_max 50000
"""

from fedgrok.core.fed_config import FedConfig
from fedgrok.training.runner import (
    run_single_federated, run_multi_seed,
    save_experiment_results, RunConfig,
)

SEEDS = [42, 123, 456]

HARD_SETTINGS = {
    "H1": {"alpha": 0.25, "K": 10, "partition": "iid",
            "local_epochs": 25, "fraction_train": 1.0},
    "H2": {"alpha": 0.25, "K": 10, "partition": "dirichlet",
            "dirichlet_alpha": 0.1, "local_epochs": 25, "fraction_train": 1.0},
    "H3": {"alpha": 0.30, "K": 10, "partition": "dirichlet",
            "dirichlet_alpha": 0.1, "local_epochs": 50, "fraction_train": 1.0},
}

ALGORITHMS = [
    ("FedAvg", "fedavg", {}),
    ("FedProx-0.001", "fedprox", {"proximal_mu": 0.001}),
    ("FedProx-0.01", "fedprox", {"proximal_mu": 0.01}),
    ("FedProx-0.1", "fedprox", {"proximal_mu": 0.1}),
    ("FedProx-1.0", "fedprox", {"proximal_mu": 1.0}),
    ("FedAdam-0.01", "fedadam", {"server_lr": 0.01, "tau": 1e-3}),
    ("FedAdam-0.1", "fedadam", {"server_lr": 0.1, "tau": 1e-3}),
    ("FedAdam-1.0", "fedadam", {"server_lr": 1.0, "tau": 1e-3}),
    # Weight-decay arms are labelled by the effective per-step shrinkage
    # lr*lambda, which is what is comparable across optimizers and learning
    # rates (see fedgrok.core.utils.check_decay_stability).
    #
    # These previously ran at lambda in {0.01, 0.1, 1.0}. At lr=50 that is
    # lr*lambda in {0.5, 5, 50} -- weights halved or sign-flipped every single
    # step -- so all nine cells reported T_grok=inf for purely numerical
    # reasons. The published grokking band is lr*lambda in [1e-4, 1e-3], which
    # at lr=50 means lambda in [2e-6, 2e-5].
    ("FedAvg+WD-lr1e-5", "fedavg", {"weight_decay": 2e-7}),   # timescale ~1e5 steps
    ("FedAvg+WD-lr1e-4", "fedavg", {"weight_decay": 2e-6}),   # Omnigrok band
    ("FedAvg+WD-lr1e-3", "fedavg", {"weight_decay": 2e-5}),   # Nanda/Power band
    ("FedAvg+WD-lr1e-2", "fedavg", {"weight_decay": 2e-4}),   # expect memorisation suppressed
]


def run_exp5_cell(setting_label: str, algo_label: str,
                  hidden_width: int = 256, t_max: int = 50000,
                  output_dir: str = "results/experiments/exp5_algorithms"):
    """Run a single (setting, algorithm) cell."""
    setting = HARD_SETTINGS[setting_label]
    algo_match = [a for a in ALGORITHMS if a[0] == algo_label]
    if not algo_match:
        raise ValueError(f"Unknown algorithm: {algo_label}. Options: {[a[0] for a in ALGORITHMS]}")
    _, strategy, algo_kwargs = algo_match[0]

    rc = RunConfig(t_base=8000, t_max=t_max)
    s_rescue = rc.s_rescue
    E = setting["local_epochs"]
    num_rounds = s_rescue // E

    print(f"\n{'='*70}")
    print(f"Exp 5: {setting_label} × {algo_label}")
    print(f"  alpha={setting['alpha']}, K={setting['K']}, E={E}, "
          f"partition={setting['partition']}, R={num_rounds}, S_rescue={s_rescue}")
    print(f"{'='*70}")

    cfg = FedConfig(
        task="addition", p=97, optimizer="gd", lr=50.0,
        weight_decay=algo_kwargs.get("weight_decay", 0.0),
        momentum=0.0, hidden_width=hidden_width,
        alpha=setting["alpha"],
        num_clients=setting["K"],
        num_rounds=num_rounds,
        local_epochs=E,
        fraction_train=setting.get("fraction_train", 1.0),
        partition=setting["partition"],
        dirichlet_alpha=setting.get("dirichlet_alpha", 0.5),
        strategy=strategy,
        proximal_mu=algo_kwargs.get("proximal_mu", 0.0),
        server_lr=algo_kwargs.get("server_lr", 1.0),
        tau=algo_kwargs.get("tau", 1e-3),
        output_dir=f"{output_dir}/{setting_label}",
    )
    result = run_multi_seed(
        run_fn=run_single_federated,
        cfg_template=cfg,
        seeds=SEEDS,
        label=f"5 {setting_label} {algo_label}",
    )
    result["setting"] = setting_label
    result["algorithm"] = algo_label

    save_experiment_results(result, f"{output_dir}/exp5_{setting_label}_{algo_label}.json")
    return result


def run_exp5(settings: list = None, algorithms: list = None,
             hidden_width: int = 256, t_max: int = 50000,
             output_dir: str = "results/experiments/exp5_algorithms"):
    """Run Exp 5 across all (setting, algorithm) pairs."""
    if settings is None:
        settings = list(HARD_SETTINGS.keys())
    if algorithms is None:
        algorithms = [a[0] for a in ALGORITHMS]

    results = []
    for setting_label in settings:
        for algo_label in algorithms:
            result = run_exp5_cell(setting_label, algo_label,
                                   hidden_width=hidden_width, t_max=t_max,
                                   output_dir=output_dir)
            results.append(result)

    save_experiment_results(results, f"{output_dir}/exp5_results.json")
    return results


if __name__ == "__main__":
    run_exp5()
