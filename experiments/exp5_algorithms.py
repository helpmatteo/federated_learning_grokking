"""Experiment 5: Algorithm comparison & rescue.

For hard settings from Exps 2-4 where FedAvg fails or is delayed,
test: FedAvg, FedProx (mu sweep), FedAdam (server_lr sweep), FedAvg+WD.
"""

from dataclasses import replace
from federated.config import FedConfig
from experiments.runner import (
    run_single_federated, run_multi_seed,
    save_experiment_results, RunConfig,
)

SEEDS = [42, 123, 456]

ALGORITHMS = [
    ("FedAvg", "fedavg", {}),
    ("FedProx-0.001", "fedprox", {"proximal_mu": 0.001}),
    ("FedProx-0.01", "fedprox", {"proximal_mu": 0.01}),
    ("FedProx-0.1", "fedprox", {"proximal_mu": 0.1}),
    ("FedProx-1.0", "fedprox", {"proximal_mu": 1.0}),
    ("FedAdam-0.01", "fedadam", {"server_lr": 0.01, "tau": 1e-3}),
    ("FedAdam-0.1", "fedadam", {"server_lr": 0.1, "tau": 1e-3}),
    ("FedAdam-1.0", "fedadam", {"server_lr": 1.0, "tau": 1e-3}),
    ("FedAvg+WD-0.01", "fedavg", {"weight_decay": 0.01}),
    ("FedAvg+WD-0.1", "fedavg", {"weight_decay": 0.1}),
    ("FedAvg+WD-1.0", "fedavg", {"weight_decay": 1.0}),
]


def run_exp5(hard_settings: list, hidden_width: int = 256,
             t_max: int = 30000,
             output_dir: str = "results/exp5_algorithms"):
    rc = RunConfig(t_base=8000, t_max=t_max)
    s_rescue = rc.s_rescue
    results = []

    for setting in hard_settings:
        for algo_label, strategy, algo_kwargs in ALGORITHMS:
            E = setting.get("local_epochs", 5)
            num_rounds = s_rescue // E

            cfg = FedConfig(
                task="addition", p=97, optimizer="gd", lr=50.0,
                weight_decay=algo_kwargs.get("weight_decay", 0.0),
                momentum=0.0, hidden_width=hidden_width,
                alpha=setting["alpha"],
                num_clients=setting["K"],
                num_rounds=num_rounds,
                local_epochs=E,
                fraction_train=setting.get("fraction_train", 1.0),
                partition=setting.get("partition", "iid"),
                dirichlet_alpha=setting.get("dirichlet_alpha", 0.5),
                strategy=strategy,
                proximal_mu=algo_kwargs.get("proximal_mu", 0.0),
                server_lr=algo_kwargs.get("server_lr", 1.0),
                tau=algo_kwargs.get("tau", 1e-3),
                output_dir=f"{output_dir}/{setting['label']}",
            )
            result = run_multi_seed(
                run_fn=run_single_federated,
                cfg_template=cfg,
                seeds=SEEDS,
                label=f"5 {setting['label']} {algo_label}",
            )
            result["setting"] = setting["label"]
            result["algorithm"] = algo_label
            results.append(result)

    save_experiment_results(results, f"{output_dir}/exp5_results.json")
    return results


if __name__ == "__main__":
    hard_settings = [
        {"label": "H1", "alpha": 0.15, "K": 20, "partition": "iid",
         "local_epochs": 5, "fraction_train": 1.0},
        {"label": "H2", "alpha": 0.15, "K": 10, "partition": "dirichlet",
         "local_epochs": 5, "fraction_train": 1.0, "dirichlet_alpha": 0.1},
        {"label": "H3", "alpha": 0.3, "K": 10, "partition": "iid",
         "local_epochs": 25, "fraction_train": 0.4},
    ]
    run_exp5(hard_settings)
