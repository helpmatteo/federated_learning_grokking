import pytest
from unittest.mock import MagicMock
from federated.config import FedConfig
from federated.train import _build_strategy


class TestBuildStrategy:
    def test_fedavg_default(self):
        cfg = FedConfig(p=7, hidden_width=16, num_clients=3, strategy="fedavg")
        strategy = _build_strategy(cfg, init_params=MagicMock(), evaluate_fn=lambda *a: (0, {}))
        from flwr.server.strategy import FedAvg
        assert isinstance(strategy, FedAvg)

    def test_fedadam_strategy(self):
        cfg = FedConfig(p=7, hidden_width=16, num_clients=3,
                        strategy="fedadam", server_lr=0.1, tau=1e-3)
        strategy = _build_strategy(cfg, init_params=MagicMock(), evaluate_fn=lambda *a: (0, {}))
        from flwr.server.strategy import FedAdam
        assert isinstance(strategy, FedAdam)

    def test_fedprox_uses_fedavg_strategy(self):
        cfg = FedConfig(p=7, hidden_width=16, num_clients=3,
                        strategy="fedprox", proximal_mu=0.1)
        strategy = _build_strategy(cfg, init_params=MagicMock(), evaluate_fn=lambda *a: (0, {}))
        from flwr.server.strategy import FedAvg
        assert isinstance(strategy, FedAvg)

    def test_with_metrics_aggregation(self):
        cfg = FedConfig(p=7, hidden_width=16, num_clients=3, strategy="fedavg")
        mock_fn = MagicMock()
        strategy = _build_strategy(cfg, init_params=MagicMock(),
                                   evaluate_fn=lambda *a: (0, {}),
                                   fit_metrics_aggregation_fn=mock_fn)
        from flwr.server.strategy import FedAvg
        assert isinstance(strategy, FedAvg)
