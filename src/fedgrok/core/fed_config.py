"""Federated learning configuration extending the base Config."""

from dataclasses import dataclass
from typing import Literal
from fedgrok.core.config import Config


@dataclass
class FedConfig(Config):
    # --- Federated learning ---
    num_clients: int = 5                  # number of FL clients
    num_rounds: int = 2000                # FedAvg communication rounds
    local_epochs: int = 5                 # local SGD steps per client per round
    fraction_train: float = 1.0           # fraction of clients selected per round
    partition: Literal[
        "iid", "operand", "target", "dirichlet", "label_block", "coset"
    ] = "iid"
    dirichlet_alpha: float = 0.5          # concentration param for Dirichlet partition
                                          # (α→∞: IID, α→0: one class per client)
    proximal_mu: float = 0.0              # FedProx proximal term strength
                                          # (0.0 = FedAvg, >0 = FedProx)
    strategy: Literal[
        "fedavg", "fedprox", "fedadam", "fedavgm", "fedyogi", "scaffold", "feddyn"
    ] = "fedavg"
    server_lr: float = 1.0               # server-side learning rate (FedAdam/Yogi/AvgM)
    server_momentum: float = 0.0         # server momentum (FedAvgM); DiLoCo's outer Nesterov
    tau: float = 1e-3                     # adaptivity parameter (FedAdam/FedYogi)
    feddyn_alpha: float = 0.01           # FedDyn dynamic-regularisation strength
    track_client_drift: bool = True       # enable per-round drift logging
    persist_local_opt_state: bool = False # keep each client's optimizer state across
                                          #   rounds instead of rebuilding it.
                                          #   False = standard FedAvg semantics and
                                          #   the historical behaviour. For GD at
                                          #   momentum=0 the two are identical; for
                                          #   AdamW they are NOT — a fresh optimizer
                                          #   makes every round E bias-corrected
                                          #   cold-start Adam steps, so the E axis
                                          #   measures optimizer restart mixed with
                                          #   client drift. Set True to separate them.
    eval_every: int = 1                   # run global evaluation every N rounds
                                          # (1 = every round; higher = fewer curve
                                          #  points, proportionally faster). Round 0,
                                          #  the final round, and checkpoint rounds
                                          #  are always evaluated.
    checkpoint_every: int = 0                # save checkpoints every N rounds (0 = disabled)
    checkpoint_client_weights: bool = False  # save per-client W1 weights at checkpoints

    # Override defaults for federated setting
    hidden_width: int = 128               # slightly overparameterized for FL
    output_dir: str = "results/baselines/federated"

    # NOTE: `epochs`, `log_every`, and `_epochs_set` are inherited from Config
    # but unused in the federated setting (replaced by num_rounds/local_epochs).
