"""Federated learning configuration extending the base Config."""

from dataclasses import dataclass
from typing import Literal
from core.config import Config


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

    # Override defaults for federated setting
    hidden_width: int = 128               # slightly overparameterized for FL
    output_dir: str = "results/federated"

    # NOTE: `epochs`, `log_every`, and `_epochs_set` are inherited from Config
    # but unused in the federated setting (replaced by num_rounds/local_epochs).
