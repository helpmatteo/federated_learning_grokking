"""Entry point for federated grokking experiments.

Usage examples:
    # Default: FedAvg with 5 IID clients, GD, 2000 rounds
    python fed_main.py

    # Non-IID by operand (fragments Fourier structure)
    python fed_main.py --partition operand

    # More clients, fewer rounds
    python fed_main.py --num_clients 10 --num_rounds 100

    # AdamW optimizer
    python fed_main.py --optimizer adamw

    # Compare partition strategies
    python fed_main.py --sweep partition

    # Compare number of clients
    python fed_main.py --sweep num_clients
"""

import argparse
import os
from federated.config import FedConfig
from federated.train import fed_train
from federated.visualize import (
    plot_fed_vs_centralized,
    plot_partition_comparison,
    plot_client_scaling,
    plot_local_epochs,
    plot_dirichlet_sweep,
    plot_participation_sweep,
    plot_breaking_point,
    plot_fedprox_sweep,
    load_history,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Federated grokking experiments")
    # Base config
    parser.add_argument("--p", type=int, default=97)
    parser.add_argument("--task", type=str, default="addition")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--hidden_width", type=int, default=128)
    parser.add_argument("--activation", type=str, default="quadratic")
    parser.add_argument("--optimizer", type=str, default="gd", choices=["gd", "adamw"])
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="results/baselines/federated")
    parser.add_argument("--save_weights", action="store_true")
    # FL config
    parser.add_argument("--num_clients", type=int, default=5)
    parser.add_argument("--num_rounds", type=int, default=2000)
    parser.add_argument("--local_epochs", type=int, default=5)
    parser.add_argument("--fraction_train", type=float, default=1.0)
    parser.add_argument("--partition", type=str, default="iid",
                        choices=["iid", "operand", "target", "dirichlet"])
    parser.add_argument("--dirichlet_alpha", type=float, default=0.5,
                        help="Dirichlet concentration parameter (only used when --partition dirichlet)")
    parser.add_argument("--proximal_mu", type=float, default=0.0,
                        help="FedProx proximal term strength (0 = FedAvg)")
    parser.add_argument("--no_plot", action="store_true")
    parser.add_argument("--sweep", type=str, default=None,
                        choices=["partition", "num_clients", "local_epochs", "dirichlet",
                                 "participation", "breaking_point", "fedprox"],
                        help="Run a sweep instead of a single experiment")
    return parser.parse_args()


def build_config(args) -> FedConfig:
    lr = args.lr if args.lr is not None else (1e-4 if args.optimizer == "adamw" else 50.0)
    wd = args.weight_decay if args.weight_decay is not None else (1.0 if args.optimizer == "adamw" else 0.0)
    return FedConfig(
        p=args.p, task=args.task, alpha=args.alpha,
        hidden_width=args.hidden_width, activation=args.activation,
        optimizer=args.optimizer, lr=lr, weight_decay=wd,
        momentum=args.momentum, seed=args.seed,
        output_dir=args.output_dir, save_weights=args.save_weights,
        num_clients=args.num_clients, num_rounds=args.num_rounds,
        local_epochs=args.local_epochs, fraction_train=args.fraction_train,
        partition=args.partition, dirichlet_alpha=args.dirichlet_alpha,
        proximal_mu=args.proximal_mu,
        _lr_set=args.lr is not None,
        _wd_set=args.weight_decay is not None,
    )


def single_run(cfg: FedConfig, plot=True):
    print(f"\n{'='*60}")
    print(f"FEDERATED | Task: {cfg.task}  |  Partition: {cfg.partition}  "
          f"|  K={cfg.num_clients}")
    print(f"Rounds: {cfg.num_rounds}  |  Local epochs: {cfg.local_epochs}  "
          f"|  Optimizer: {cfg.optimizer}  |  lr={cfg.lr}")
    print(f"p={cfg.p}  |  N={cfg.hidden_width}  |  \u03b1={cfg.alpha}")
    print(f"{'='*60}\n")

    history, model = fed_train(cfg)

    if plot:
        # Try to load centralized baseline for comparison
        central_tag = f"{cfg.task}_{cfg.optimizer}_p{cfg.p}_N{cfg.hidden_width}_a{cfg.alpha}"
        central_path = os.path.join("results/baselines/centralized", f"history_{central_tag}.json")
        if os.path.exists(central_path):
            central_history = load_history(central_path)
            tag = f"{cfg.partition}_K{cfg.num_clients}"
            plot_fed_vs_centralized(history, central_history, cfg.output_dir, tag)
        else:
            print(f"No centralized baseline found at {central_path}")
            print("Run `python main.py` first to generate it.")

    return history


def sweep_partition(base_cfg: FedConfig):
    """Compare IID, operand, and target partition strategies."""
    histories = []
    labels = []
    for part in ["iid", "operand", "target"]:
        cfg = FedConfig(
            p=base_cfg.p, task=base_cfg.task, alpha=base_cfg.alpha,
            hidden_width=base_cfg.hidden_width, activation=base_cfg.activation,
            optimizer=base_cfg.optimizer, lr=base_cfg.lr,
            weight_decay=base_cfg.weight_decay, momentum=base_cfg.momentum,
            seed=base_cfg.seed, output_dir=base_cfg.output_dir,
            num_clients=base_cfg.num_clients, num_rounds=base_cfg.num_rounds,
            local_epochs=base_cfg.local_epochs, fraction_train=base_cfg.fraction_train,
            partition=part,
            _lr_set=True, _wd_set=True,
        )
        h = single_run(cfg, plot=False)
        histories.append(h)
        labels.append(part.upper())

    plot_partition_comparison(histories, labels, base_cfg.output_dir)


def sweep_num_clients(base_cfg: FedConfig):
    """Compare different numbers of clients."""
    client_counts = [2, 5, 10, 20]
    histories = []
    for k in client_counts:
        cfg = FedConfig(
            p=base_cfg.p, task=base_cfg.task, alpha=base_cfg.alpha,
            hidden_width=base_cfg.hidden_width, activation=base_cfg.activation,
            optimizer=base_cfg.optimizer, lr=base_cfg.lr,
            weight_decay=base_cfg.weight_decay, momentum=base_cfg.momentum,
            seed=base_cfg.seed, output_dir=base_cfg.output_dir,
            num_clients=k, num_rounds=base_cfg.num_rounds,
            local_epochs=base_cfg.local_epochs, fraction_train=base_cfg.fraction_train,
            partition=base_cfg.partition,
            _lr_set=True, _wd_set=True,
        )
        h = single_run(cfg, plot=False)
        histories.append(h)

    plot_client_scaling(histories, client_counts, base_cfg.output_dir)


def sweep_local_epochs(base_cfg: FedConfig):
    """Compare grokking dynamics across local epoch counts (IID partition)."""
    local_epoch_counts = [1, 5, 10, 20, 50]
    histories = []
    for le in local_epoch_counts:
        cfg = FedConfig(
            p=base_cfg.p, task=base_cfg.task, alpha=base_cfg.alpha,
            hidden_width=base_cfg.hidden_width, activation=base_cfg.activation,
            optimizer=base_cfg.optimizer, lr=base_cfg.lr,
            weight_decay=base_cfg.weight_decay, momentum=base_cfg.momentum,
            seed=base_cfg.seed, output_dir=base_cfg.output_dir,
            num_clients=base_cfg.num_clients, num_rounds=base_cfg.num_rounds,
            local_epochs=le, fraction_train=base_cfg.fraction_train,
            partition="iid",
            _lr_set=True, _wd_set=True,
        )
        h = single_run(cfg, plot=False)
        histories.append(h)

    plot_local_epochs(histories, local_epoch_counts, base_cfg.output_dir)


def sweep_dirichlet(base_cfg: FedConfig):
    """Sweep Dirichlet concentration parameter to explore heterogeneity effects."""
    dirichlet_alphas = [0.1, 0.5, 1.0, 5.0, 100.0]
    histories = []
    for da in dirichlet_alphas:
        cfg = FedConfig(
            p=base_cfg.p, task=base_cfg.task, alpha=base_cfg.alpha,
            hidden_width=base_cfg.hidden_width, activation=base_cfg.activation,
            optimizer=base_cfg.optimizer, lr=base_cfg.lr,
            weight_decay=base_cfg.weight_decay, momentum=base_cfg.momentum,
            seed=base_cfg.seed, output_dir=base_cfg.output_dir,
            num_clients=base_cfg.num_clients, num_rounds=base_cfg.num_rounds,
            local_epochs=base_cfg.local_epochs, fraction_train=base_cfg.fraction_train,
            partition="dirichlet", dirichlet_alpha=da,
            _lr_set=True, _wd_set=True,
        )
        h = single_run(cfg, plot=False)
        histories.append(h)

    plot_dirichlet_sweep(histories, dirichlet_alphas, base_cfg.output_dir)


def sweep_participation(base_cfg: FedConfig):
    """Sweep participation rate on a non-IID partition to reveal heterogeneity effects."""
    fractions = [0.2, 0.4, 0.6, 0.8, 1.0]
    partition = base_cfg.partition if base_cfg.partition != "iid" else "target"
    histories = []
    for ft in fractions:
        cfg = FedConfig(
            p=base_cfg.p, task=base_cfg.task, alpha=base_cfg.alpha,
            hidden_width=base_cfg.hidden_width, activation=base_cfg.activation,
            optimizer=base_cfg.optimizer, lr=base_cfg.lr,
            weight_decay=base_cfg.weight_decay, momentum=base_cfg.momentum,
            seed=base_cfg.seed, output_dir=base_cfg.output_dir,
            num_clients=base_cfg.num_clients, num_rounds=base_cfg.num_rounds,
            local_epochs=base_cfg.local_epochs, fraction_train=ft,
            partition=partition, dirichlet_alpha=base_cfg.dirichlet_alpha,
            _lr_set=True, _wd_set=True,
        )
        h = single_run(cfg, plot=False)
        histories.append(h)

    plot_participation_sweep(histories, fractions, partition, base_cfg.output_dir)


def sweep_breaking_point(base_cfg: FedConfig):
    """Stress-test federated grokking along three axes to find where it breaks.

    Sub-experiment 1 — Client scaling (IID, full participation):
        K in {5, 10, 20, 50, 97}.  At K=97 each client has ~48 training samples.

    Sub-experiment 2 — Partial participation (K=20, target partition):
        fraction_train in {0.2, 0.4, 0.6, 0.8, 1.0}.

    Sub-experiment 3 — Extreme local epochs / client drift (K=5, IID):
        local_epochs in {50, 100, 200} with num_rounds adjusted so
        total_steps >= 20 000 (2x the typical grokking threshold).
    """
    results = {}

    def _make_cfg(**overrides):
        kw = dict(
            p=base_cfg.p, task=base_cfg.task, alpha=base_cfg.alpha,
            hidden_width=base_cfg.hidden_width, activation=base_cfg.activation,
            optimizer=base_cfg.optimizer, lr=base_cfg.lr,
            weight_decay=base_cfg.weight_decay, momentum=base_cfg.momentum,
            seed=base_cfg.seed, output_dir=base_cfg.output_dir,
            num_clients=base_cfg.num_clients, num_rounds=base_cfg.num_rounds,
            local_epochs=base_cfg.local_epochs, fraction_train=base_cfg.fraction_train,
            partition=base_cfg.partition, dirichlet_alpha=base_cfg.dirichlet_alpha,
            _lr_set=True, _wd_set=True,
        )
        kw.update(overrides)
        return FedConfig(**kw)

    # ── Sub-experiment 1: Client scaling ─────────────────────────────────
    print("\n" + "=" * 70)
    print("BREAKING POINT — Sub-experiment 1: Client Scaling")
    print("=" * 70)
    client_counts = [5, 10, 20, 50, 97]
    client_histories, client_labels = [], []
    for k in client_counts:
        cfg = _make_cfg(num_clients=k, partition="iid",
                        fraction_train=1.0, local_epochs=5, num_rounds=2000)
        h = single_run(cfg, plot=False)
        client_histories.append(h)
        client_labels.append(f"K={k}")
    results["clients"] = {"histories": client_histories, "labels": client_labels}
    plot_client_scaling(client_histories, client_counts, base_cfg.output_dir)

    # ── Sub-experiment 2: Partial participation ──────────────────────────
    print("\n" + "=" * 70)
    print("BREAKING POINT — Sub-experiment 2: Partial Participation (K=20, target)")
    print("=" * 70)
    fractions = [0.2, 0.4, 0.6, 0.8, 1.0]
    part_histories, part_labels = [], []
    for ft in fractions:
        cfg = _make_cfg(num_clients=20, partition="target",
                        fraction_train=ft, local_epochs=5, num_rounds=2000)
        h = single_run(cfg, plot=False)
        part_histories.append(h)
        part_labels.append(f"ft={ft}")
    results["participation"] = {"histories": part_histories, "labels": part_labels}
    plot_participation_sweep(part_histories, fractions, "target", base_cfg.output_dir)

    # ── Sub-experiment 3: Extreme local epochs ───────────────────────────
    print("\n" + "=" * 70)
    print("BREAKING POINT — Sub-experiment 3: Extreme Local Epochs (K=5, IID)")
    print("=" * 70)
    # Target ~20k total steps minimum so grokking has room to appear.
    le_configs = [
        (50,  400),   # 50 * 400  = 20 000 steps
        (100, 200),   # 100 * 200 = 20 000 steps
        (200, 100),   # 200 * 100 = 20 000 steps
    ]
    le_histories, le_labels = [], []
    for le, nr in le_configs:
        cfg = _make_cfg(num_clients=5, partition="iid",
                        fraction_train=1.0, local_epochs=le, num_rounds=nr)
        h = single_run(cfg, plot=False)
        le_histories.append(h)
        le_labels.append(f"le={le}")
    results["local_epochs"] = {"histories": le_histories, "labels": le_labels}

    # ── Combined summary plot ────────────────────────────────────────────
    plot_breaking_point(results, base_cfg.output_dir)


def sweep_fedprox(base_cfg: FedConfig):
    """Sweep FedProx proximal strength mu to test if regularization kills grokking.

    mu=0 is FedAvg (baseline). Increasing mu penalizes client drift from the
    global model, which may suppress the weight growth that drives grokking.
    """
    mu_values = [0.0, 0.001, 0.01, 0.1, 1.0, 10.0]
    histories = []
    for mu in mu_values:
        cfg = FedConfig(
            p=base_cfg.p, task=base_cfg.task, alpha=base_cfg.alpha,
            hidden_width=base_cfg.hidden_width, activation=base_cfg.activation,
            optimizer=base_cfg.optimizer, lr=base_cfg.lr,
            weight_decay=base_cfg.weight_decay, momentum=base_cfg.momentum,
            seed=base_cfg.seed, output_dir=base_cfg.output_dir,
            num_clients=base_cfg.num_clients, num_rounds=base_cfg.num_rounds,
            local_epochs=base_cfg.local_epochs, fraction_train=base_cfg.fraction_train,
            partition=base_cfg.partition, dirichlet_alpha=base_cfg.dirichlet_alpha,
            proximal_mu=mu,
            _lr_set=True, _wd_set=True,
        )
        h = single_run(cfg, plot=False)
        histories.append(h)

    plot_fedprox_sweep(histories, mu_values, base_cfg.output_dir)


def main():
    args = parse_args()
    cfg = build_config(args)

    if args.sweep == "partition":
        sweep_partition(cfg)
    elif args.sweep == "num_clients":
        sweep_num_clients(cfg)
    elif args.sweep == "local_epochs":
        sweep_local_epochs(cfg)
    elif args.sweep == "dirichlet":
        sweep_dirichlet(cfg)
    elif args.sweep == "participation":
        sweep_participation(cfg)
    elif args.sweep == "breaking_point":
        sweep_breaking_point(cfg)
    elif args.sweep == "fedprox":
        sweep_fedprox(cfg)
    else:
        single_run(cfg, plot=not args.no_plot)


if __name__ == "__main__":
    main()
