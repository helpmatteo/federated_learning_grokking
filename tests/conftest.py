"""Shared fixtures for all test modules."""

import pytest
import torch
import numpy as np

from core.config import Config
from core.model import GrokNet
from federated.config import FedConfig


# Use a small prime to keep tests fast
SMALL_P = 7


@pytest.fixture
def small_cfg():
    """Minimal Config with p=7 for fast tests."""
    return Config(p=SMALL_P, hidden_width=16, epochs=10, log_every=5, seed=42)


@pytest.fixture
def small_fed_cfg():
    """Minimal FedConfig with p=7, 3 clients, 2 rounds for fast tests."""
    return FedConfig(
        p=SMALL_P,
        hidden_width=16,
        num_clients=3,
        num_rounds=2,
        local_epochs=2,
        seed=42,
    )


@pytest.fixture
def small_model():
    """GrokNet with p=7, N=16."""
    return GrokNet(input_dim=2 * SMALL_P, hidden_width=16, output_dim=SMALL_P)


@pytest.fixture
def random_batch():
    """Random input batch for p=7."""
    torch.manual_seed(0)
    x = torch.randn(10, 2 * SMALL_P)
    return x


@pytest.fixture
def trained_model_with_grads(small_model, random_batch):
    """A model that has had forward + backward called (has gradients)."""
    model = small_model
    x = random_batch
    out = model(x)
    loss = out.sum()
    loss.backward()
    return model
