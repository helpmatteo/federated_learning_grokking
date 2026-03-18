import pytest
from federated.config import FedConfig


class TestFedConfigStrategy:
    def test_default_strategy_is_fedavg(self):
        cfg = FedConfig()
        assert cfg.strategy == "fedavg"

    def test_fedadam_params(self):
        cfg = FedConfig(strategy="fedadam", server_lr=0.1, tau=1e-3)
        assert cfg.strategy == "fedadam"
        assert cfg.server_lr == 0.1
        assert cfg.tau == 1e-3

    def test_track_client_drift_default_true(self):
        cfg = FedConfig()
        assert cfg.track_client_drift is True

    def test_weight_decay_in_fl(self):
        cfg = FedConfig(weight_decay=0.1)
        assert cfg.weight_decay == 0.1

    def test_strategy_preserved_through_inheritance(self):
        cfg = FedConfig(strategy="fedprox", proximal_mu=0.1)
        assert cfg.strategy == "fedprox"
        assert cfg.proximal_mu == 0.1
