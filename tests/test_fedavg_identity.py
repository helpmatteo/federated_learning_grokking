"""FedAvg at E=1 is an exact identity with centralized full-batch GD.

With one local step, full participation, full-batch local gradients and
n_k/n aggregation weights:

    w' = sum_k (n_k/n) * (w - lr * grad L_k(w))
       = w - lr * sum_k (n_k/n) * grad L_k(w)
       = w - lr * grad L(w)

because sum_k (n_k/n) * (1/n_k) * sum_{i in k} grad l_i  =  (1/n) sum_i grad l_i.

Nothing in that derivation assumes the partition is IID -- the n_k/n weights
cancel the per-client normalisation whatever the shards look like. So the
identity holds under Dirichlet, operand and target partitions too, and the
whole of this study's federated behaviour lives in the E > 1 deviation.

These tests exist for two reasons: they are the cheapest possible check that
the federated implementation is correct, and the identity itself is a stated
result in the write-up (it is why "FedAvg preserves grokking" is not on its own
a finding).
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from fedgrok.core.config import Config
from fedgrok.core.fed_config import FedConfig
from fedgrok.core.utils import make_optimizer, make_targets_onehot, get_device
from fedgrok.data.partition import make_federated_datasets
from fedgrok.training.federated import (
    GrokClient,
    _cfg_to_fit_config,
    _make_model,
    _model_to_ndarrays,
    _ndarrays_to_state_dict,
    _dataset_cache,
)

# p=17 (not 7): with only 24 training samples at p=7, the extreme non-IID
# partitions leave some clients empty at K=4. p=17 gives 144 samples, enough for
# every partition scheme here to keep all four clients non-empty -- which is what
# lets the identity be tested against the pathological Dirichlet(0.01) case.
SMALL_P = 17
STABLE_LR = 1.0

# Partitions spanning IID through pathological. dirichlet_alpha=0.01 gives
# clients that hold essentially one class each.
PARTITIONS = [
    ("iid", 0.5),
    ("dirichlet", 1.0),
    ("dirichlet", 0.1),
    ("dirichlet", 0.01),
    ("operand", 0.5),
    ("target", 0.5),
]


def _fed_cfg(partition, dirichlet_alpha, num_clients=4, local_epochs=1):
    return FedConfig(
        task="addition", p=SMALL_P, alpha=0.5, seed=42, hidden_width=16,
        num_clients=num_clients, num_rounds=1, local_epochs=local_epochs,
        fraction_train=1.0, partition=partition,
        dirichlet_alpha=dirichlet_alpha, lr=STABLE_LR, weight_decay=0.0,
        momentum=0.0,
    )


def _centralized_step(cfg, init_ndarrays, x_train, y_train):
    """One full-batch GD step from `init_ndarrays`, returned as ndarrays."""
    device = get_device()
    model = _make_model(cfg)
    model.load_state_dict(_ndarrays_to_state_dict(init_ndarrays, model))
    model.to(device)

    x_train = x_train.to(device)
    y_onehot = make_targets_onehot(y_train.to(device), cfg.p)

    optimizer = make_optimizer(model, cfg)
    loss_fn = nn.MSELoss()

    optimizer.zero_grad()
    loss_fn(model(x_train), y_onehot).backward()
    optimizer.step()

    return _model_to_ndarrays(model)


def _fedavg_round(cfg, init_ndarrays):
    """One FedAvg round through the real client code path, aggregated n_k/n."""
    fit_config = _cfg_to_fit_config(cfg, server_round=1)

    updates, weights = [], []
    for partition_id in range(cfg.num_clients):
        client = GrokClient(partition_id=partition_id)
        client_ndarrays, n_examples, _ = client.fit(init_ndarrays, fit_config)
        updates.append(client_ndarrays)
        weights.append(n_examples)

    total = float(sum(weights))
    return [
        sum(w * layer[i] for w, layer in zip(weights, updates)) / total
        for i in range(len(init_ndarrays))
    ]


class TestFedAvgE1Identity:

    @pytest.mark.parametrize("partition,dirichlet_alpha", PARTITIONS)
    def test_one_round_equals_one_centralized_step(self, partition, dirichlet_alpha):
        """The identity holds for every partition scheme, IID or not."""
        _dataset_cache.clear()
        cfg = _fed_cfg(partition, dirichlet_alpha)

        _, x_train_full, y_train_full, _, _ = make_federated_datasets(cfg)

        torch.manual_seed(cfg.seed)
        init_ndarrays = _model_to_ndarrays(_make_model(cfg))

        fed = _fedavg_round(cfg, init_ndarrays)
        cent = _centralized_step(cfg, init_ndarrays, x_train_full, y_train_full)

        for layer_idx, (a, b) in enumerate(zip(fed, cent)):
            np.testing.assert_allclose(
                a, b, rtol=1e-4, atol=1e-6,
                err_msg=(f"layer {layer_idx} diverged for partition={partition} "
                         f"(dirichlet_alpha={dirichlet_alpha})"),
            )

    @pytest.mark.parametrize("num_clients", [2, 4, 12])
    def test_identity_is_independent_of_client_count(self, num_clients):
        """Splitting the same data more ways changes nothing at E=1."""
        _dataset_cache.clear()
        cfg = _fed_cfg("dirichlet", 0.1, num_clients=num_clients)

        _, x_train_full, y_train_full, _, _ = make_federated_datasets(cfg)

        torch.manual_seed(cfg.seed)
        init_ndarrays = _model_to_ndarrays(_make_model(cfg))

        fed = _fedavg_round(cfg, init_ndarrays)
        cent = _centralized_step(cfg, init_ndarrays, x_train_full, y_train_full)

        for a, b in zip(fed, cent):
            np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-6)

    def test_multiple_local_steps_do_diverge(self):
        """Guard against a vacuous test: at E>1 the identity must break.

        If this passed, the E=1 tests above would be proving nothing -- and the
        entire study, which lives in the E>1 deviation, would have no signal.
        """
        _dataset_cache.clear()
        cfg = _fed_cfg("dirichlet", 0.01, local_epochs=5)

        _, x_train_full, y_train_full, _, _ = make_federated_datasets(cfg)

        torch.manual_seed(cfg.seed)
        init_ndarrays = _model_to_ndarrays(_make_model(cfg))

        fed = _fedavg_round(cfg, init_ndarrays)
        cent = _centralized_step(cfg, init_ndarrays, x_train_full, y_train_full)

        max_diff = max(float(np.abs(a - b).max()) for a, b in zip(fed, cent))
        assert max_diff > 1e-5, (
            "FedAvg with E=5 matched a single centralized step; the identity "
            "test is not discriminating."
        )
