"""Entry point for reproducing Gromov (2023) grokking experiments.

Usage examples:
    # Default: GD on modular addition, p=97, N=100, α=0.5, 40k epochs
    python main.py

    # AdamW (auto-sets lr=1e-3, wd=1.0, fewer epochs needed)
    python main.py --optimizer adamw --epochs 5000

    # Different task
    python main.py --task multiplication --alpha 0.7

    # Sweep widths (reproduces Figure 4b)
    python main.py --sweep width

    # Sweep optimizers (reproduces Figure 4a comparison)
    python main.py --sweep optimizer
"""

import argparse
import os
from config import Config
from train import train
from visualize import plot_training_dynamics, plot_ipr, plot_compare_histories


def parse_args():
    parser = argparse.ArgumentParser(description="Grokking modular arithmetic")
    parser.add_argument("--p", type=int, default=97)
    parser.add_argument("--task", type=str, default="addition",
                        choices=["addition", "subtraction", "division",
                                 "x2_plus_y2", "x_plus_y_squared",
                                 "multiplication", "x2_y2_xy", "x3_xy2_y"])
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--hidden_width", type=int, default=100)
    parser.add_argument("--activation", type=str, default="quadratic",
                        choices=["quadratic", "relu", "gelu", "abs", "quartic"])
    parser.add_argument("--optimizer", type=str, default="gd", choices=["gd", "adamw"])
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--save_weights", action="store_true")
    parser.add_argument("--no_plot", action="store_true")
    parser.add_argument("--sweep", type=str, default=None,
                        choices=["width", "optimizer"],
                        help="Run a sweep instead of a single experiment")
    return parser.parse_args()


def single_run(cfg: Config, plot=True):
    """Run one experiment and optionally plot."""
    print(f"\n{'='*60}")
    print(f"Task: {cfg.task}  |  Optimizer: {cfg.optimizer}  |  p={cfg.p}  "
          f"|  N={cfg.hidden_width}  |  α={cfg.alpha}  |  epochs={cfg.epochs}")
    print(f"lr={cfg.lr}  |  wd={cfg.weight_decay}  |  momentum={cfg.momentum}  "
          f"|  activation={cfg.activation}")
    print(f"{'='*60}\n")

    history, model = train(cfg)

    if plot:
        tag = f"{cfg.task}_{cfg.optimizer}_N{cfg.hidden_width}"
        os.makedirs(cfg.output_dir, exist_ok=True)
        plot_training_dynamics(history, cfg.output_dir, tag)
        plot_ipr(history, cfg.output_dir, tag)

    return history


def sweep_width(base_cfg: Config):
    """Reproduce Figure 4b: test accuracy vs hidden width."""
    widths = [20, 40, 60, 80, 100, 120, 140]
    paths = []

    for n in widths:
        cfg = Config(
            p=base_cfg.p, task=base_cfg.task, alpha=base_cfg.alpha,
            hidden_width=n, activation=base_cfg.activation,
            optimizer=base_cfg.optimizer, lr=base_cfg.lr,
            weight_decay=base_cfg.weight_decay, momentum=base_cfg.momentum,
            epochs=base_cfg.epochs, log_every=base_cfg.log_every,
            seed=base_cfg.seed, output_dir=base_cfg.output_dir,
        )
        single_run(cfg, plot=False)
        tag = f"{cfg.task}_{cfg.optimizer}_p{cfg.p}_N{n}_a{cfg.alpha}"
        paths.append(os.path.join(cfg.output_dir, f"history_{tag}.json"))

    labels = [f"N={n}" for n in widths]
    plot_compare_histories(paths, labels, base_cfg.output_dir,
                           title="Test Accuracy vs Width (Fig. 4b)")


def sweep_optimizer(base_cfg: Config):
    """Reproduce Figure 4a-style comparison across optimizers."""
    configs = [
        ("GD", "gd", 0.0, 0.0, 50.0),
        ("GD+mom=0.9", "gd", 0.9, 0.0, 50.0),
        ("GD+wd", "gd", 0.0, 0.01, 50.0),
        ("AdamW", "adamw", 0.0, 1.0, 1e-4),
    ]
    paths = []

    for label, opt, mom, wd, lr in configs:
        cfg = Config(
            p=base_cfg.p, task=base_cfg.task, alpha=base_cfg.alpha,
            hidden_width=base_cfg.hidden_width, activation=base_cfg.activation,
            optimizer=opt, lr=lr, weight_decay=wd, momentum=mom,
            epochs=base_cfg.epochs, log_every=base_cfg.log_every,
            seed=base_cfg.seed, output_dir=base_cfg.output_dir,
            _lr_set=True, _wd_set=True, _epochs_set=True,
        )
        single_run(cfg, plot=False)
        tag = f"{cfg.task}_{opt}_p{cfg.p}_N{cfg.hidden_width}_a{cfg.alpha}"
        paths.append(os.path.join(cfg.output_dir, f"history_{tag}.json"))

    labels = [c[0] for c in configs]
    plot_compare_histories(paths, labels, base_cfg.output_dir,
                           title="Optimizer Comparison (Fig. 4a)")


def main():
    args = parse_args()

    # Build config, marking which values the user explicitly set
    cfg = Config(
        p=args.p, task=args.task, alpha=args.alpha,
        hidden_width=args.hidden_width, activation=args.activation,
        optimizer=args.optimizer,
        lr=args.lr if args.lr is not None else 50.0,
        weight_decay=args.weight_decay if args.weight_decay is not None else 0.0,
        momentum=args.momentum,
        epochs=args.epochs if args.epochs is not None else 10_000,
        log_every=args.log_every,
        seed=args.seed, output_dir=args.output_dir, save_weights=args.save_weights,
        _lr_set=args.lr is not None,
        _wd_set=args.weight_decay is not None,
        _epochs_set=args.epochs is not None,
    )
    cfg.apply_adamw_defaults()

    if args.sweep == "width":
        sweep_width(cfg)
    elif args.sweep == "optimizer":
        sweep_optimizer(cfg)
    else:
        single_run(cfg, plot=not args.no_plot)


if __name__ == "__main__":
    main()
