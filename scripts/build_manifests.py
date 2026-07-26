"""Generate the versioned run manifests under manifests/.

Each function builds one manifest (a JSONL of run specs) for a tier of the
plan. Regenerating is deterministic — run ids are content hashes — so
re-running never changes ids and the launcher's resume stays valid.

    python scripts/build_manifests.py            # write all
    python scripts/build_manifests.py t0_wd_grid # write one
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fedgrok.manifest import expand_grid, write_manifest

MANIFEST_DIR = os.path.join(os.path.dirname(__file__), "..", "manifests")
SEEDS5 = [42, 123, 456, 789, 1011]
SEEDS3 = [42, 123, 456]


def t0_wd_grid():
    """Tier 0: loss x optimizer x weight-decay grid on setup A (mod-97 MLP).

    Fixes the coupled-WD defect and answers the lazy->rich / numerical-stability
    critiques. Weight-decay values are chosen so lr*wd (the comparable quantity)
    is log-spaced; see fedgrok.core.utils.check_decay_stability.

      GD    (lr=50)   : wd in {0, 2e-7, 2e-6, 2e-5, 2e-4} -> lr*wd {0,1e-5..1e-2}
      AdamW (lr=1e-3) : wd in {0, 0.01, 0.1, 1.0}         -> lr*wd {0,1e-5..1e-3}

    Centralized, 5 seeds. p=97, quadratic MLP, MSE (loss is fixed here; the CE
    arm needs the Phase 2/3 loss registry and is generated separately later).
    """
    specs = []
    specs += expand_grid(
        {"mode": "centralized", "task": "addition", "p": 97, "alpha": 0.5,
         "hidden_width": 256, "activation": "quadratic", "optimizer": "gd",
         "lr": 50.0, "epochs": 50000, "log_every": 100},
        {"weight_decay": [0.0, 2e-7, 2e-6, 2e-5, 2e-4], "seed": SEEDS5},
        tags={"tier": "T0", "group": "wd_grid", "experiment": "wd"},
    )
    specs += expand_grid(
        {"mode": "centralized", "task": "addition", "p": 97, "alpha": 0.5,
         "hidden_width": 256, "activation": "quadratic", "optimizer": "adamw",
         "lr": 1e-3, "epochs": 50000, "log_every": 100},
        {"weight_decay": [0.0, 0.01, 0.1, 1.0], "seed": SEEDS5},
        tags={"tier": "T0", "group": "wd_grid", "experiment": "wd"},
    )
    return specs


def t0_poly_pilot():
    """Tier 0 pilot: do x2+y2 and x2+xy+y2 grok centrally before FL compute?

    Doshi et al. (2406.03495) separate learnable from non-learnable modular
    polynomials for this architecture; a task that does not grok centrally can
    tell us nothing about FL, so gate the operation set on this.
    """
    return expand_grid(
        {"mode": "centralized", "p": 97, "alpha": 0.5, "hidden_width": 256,
         "activation": "quadratic", "optimizer": "gd", "lr": 50.0,
         "epochs": 50000, "log_every": 100},
        {"task": ["x2_plus_y2", "x2_y2_xy"], "seed": SEEDS3},
        tags={"tier": "T0", "group": "poly_pilot", "experiment": "pilot"},
    )


def t0_mnist_wd_band():
    """Locate the Omnigrok Goldilocks weight-decay band for MNIST-1k.

    The pipeline is verified (it reaches ~91% test), but a clean grok — fast
    memorisation then *delayed* generalisation — only appears in a narrow decay
    band: too weak and test never catches up within budget; too strong and there
    is no delay at all. This sweep over lr*wd (with large init) finds the band.
    Centralized, MSE, 3-layer MLP width 200, minibatch, 3 seeds.
    """
    return expand_grid(
        {"mode": "centralized", "dataset": "mnist", "model": "mlp",
         "hidden_width": 200, "n_layers": 3, "init_scale": 9.0,
         "loss": "mse", "optimizer": "adamw", "lr": 1e-3, "batch_size": 200,
         "n_train": 1000, "n_test": 5000, "epochs": 20000, "log_every": 100},
        # lr*wd in {1e-5, 3e-5, 1e-4, 3e-4, 1e-3}
        {"weight_decay": [0.01, 0.03, 0.1, 0.3, 1.0], "seed": SEEDS3},
        tags={"tier": "T0", "group": "mnist_wd_band", "experiment": "mnist"},
    )


def t3_server_lr_calibration():
    """Fair server-LR tuning for the server-optimiser strategies.

    The original exp5 tuned FedAdam's server LR but not the others', so its
    ~10x "speedup" was partly a tuning artifact. This sweeps server_lr for every
    tunable strategy on one representative heterogeneous cell, 3 seeds, so each
    method can be fixed at its own best LR before the main comparison. FedAvg /
    FedProx / SCAFFOLD have no server-LR knob (implicitly 1.0) and are the fixed
    references. FedAvgM additionally needs its momentum, swept lightly here.
    """
    cell = {"mode": "federated", "task": "addition", "p": 97, "alpha": 0.3,
            "hidden_width": 256, "num_clients": 10, "local_epochs": 5,
            "partition": "dirichlet", "dirichlet_alpha": 0.1, "num_rounds": 10000,
            "lr": 50.0, "eval_every": 20}
    specs = []
    for strat in ("fedadam", "fedyogi"):
        specs += expand_grid(
            {**cell, "strategy": strat},
            {"server_lr": [0.01, 0.1, 0.3, 1.0], "seed": SEEDS3},
            tags={"tier": "T3", "group": "server_lr_cal", "experiment": "cal",
                  "algorithm": strat},
        )
    specs += expand_grid(
        {**cell, "strategy": "fedavgm"},
        {"server_lr": [0.1, 0.3, 1.0], "server_momentum": [0.0, 0.9],
         "seed": SEEDS3},
        tags={"tier": "T3", "group": "server_lr_cal", "experiment": "cal",
              "algorithm": "fedavgm"},
    )
    return specs


BUILDERS = {
    "t0_wd_grid": t0_wd_grid,
    "t0_poly_pilot": t0_poly_pilot,
    "t0_mnist_wd_band": t0_mnist_wd_band,
    "t3_server_lr_calibration": t3_server_lr_calibration,
}


def main():
    os.makedirs(MANIFEST_DIR, exist_ok=True)
    which = sys.argv[1:] or list(BUILDERS)
    for name in which:
        if name not in BUILDERS:
            raise SystemExit(f"Unknown manifest {name}; options: {list(BUILDERS)}")
        specs = BUILDERS[name]()
        path = os.path.join(MANIFEST_DIR, name + ".jsonl")
        write_manifest(specs, path)
        print(f"{name}: {len(specs)} runs -> {os.path.relpath(path)}")


if __name__ == "__main__":
    main()
