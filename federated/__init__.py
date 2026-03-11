"""Federated learning package."""

from federated.config import FedConfig
from federated.dataset import make_federated_datasets
from federated.train import fed_train
from federated.visualize import (
    plot_fed_vs_centralized,
    plot_partition_comparison,
    plot_client_scaling,
    plot_local_epochs,
    plot_dirichlet_sweep,
    plot_participation_sweep,
    load_history,
)
