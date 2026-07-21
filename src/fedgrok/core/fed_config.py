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
        "iid", "operand", "target", "dirichlet"
    ] = "iid"
    dirichlet_alpha: float = 0.5          # concentration param for Dirichlet partition
                                          # (α→∞: IID, α→0: one class per client)
    proximal_mu: float = 0.0              # FedProx proximal term strength
                                          # (0.0 = FedAvg, >0 = FedProx)
    strategy: Literal[
        "fedavg", "fedprox", "fedadam"
    ] = "fedavg"
    server_lr: float = 1.0               # server-side learning rate (FedAdam)
    tau: float = 1e-3                     # FedAdam adaptivity parameter
    track_client_drift: bool = True       # enable per-round drift logging
    checkpoint_every: int = 0                # save checkpoints every N rounds (0 = disabled)
    checkpoint_client_weights: bool = False  # save per-client W1 weights at checkpoints

    # Override defaults for federated setting
    hidden_width: int = 128               # slightly overparameterized for FL
    output_dir: str = "results/baselines/federated"

    # NOTE: `epochs`, `log_every`, and `_epochs_set` are inherited from Config
    # but unused in the federated setting (replaced by num_rounds/local_epochs).
