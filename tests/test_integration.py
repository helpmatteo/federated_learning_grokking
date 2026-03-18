"""Integration tests: verify experiment pipeline runs end-to-end with tiny configs."""
import pytest
import os
import tempfile

from core.config import Config
from federated.config import FedConfig
from experiments.runner import run_single_centralized, run_single_federated, run_multi_seed
from experiments.grokking_metrics import extract_grokking_results


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


class TestCentralizedPipeline:
    def test_single_run_produces_metrics(self, tmp_dir):
        cfg = Config(p=7, hidden_width=16, epochs=50, log_every=10,
                     seed=42, output_dir=tmp_dir)
        result = run_single_centralized(cfg, label="test")
        assert "t_grok" in result
        assert "t_50" in result
        assert "final_test_acc" in result

    def test_multi_seed_produces_summary(self, tmp_dir):
        cfg = Config(p=7, hidden_width=16, epochs=50, log_every=10,
                     output_dir=tmp_dir)
        result = run_multi_seed(
            run_fn=run_single_centralized,
            cfg_template=cfg,
            seeds=[42, 123],
            label="test",
        )
        assert result["summary"]["n_seeds"] == 2
        assert "t_grok_mean" in result["summary"]


class TestFederatedPipeline:
    def test_fedavg_run(self, tmp_dir):
        cfg = FedConfig(p=7, hidden_width=16, num_clients=2,
                        num_rounds=3, local_epochs=2,
                        seed=42, output_dir=tmp_dir, strategy="fedavg")
        result = run_single_federated(cfg, label="test")
        assert "t_grok" in result

    def test_fedadam_run(self, tmp_dir):
        cfg = FedConfig(p=7, hidden_width=16, num_clients=2,
                        num_rounds=3, local_epochs=2,
                        seed=42, output_dir=tmp_dir,
                        strategy="fedadam", server_lr=0.1, tau=1e-3)
        result = run_single_federated(cfg, label="test")
        assert "t_grok" in result

    def test_fedprox_run(self, tmp_dir):
        cfg = FedConfig(p=7, hidden_width=16, num_clients=2,
                        num_rounds=3, local_epochs=2,
                        seed=42, output_dir=tmp_dir,
                        strategy="fedprox", proximal_mu=0.1)
        result = run_single_federated(cfg, label="test")
        assert "t_grok" in result

    def test_drift_metrics_in_history(self, tmp_dir):
        """Verify that client drift and weight divergence are tracked."""
        from federated.train import fed_train
        cfg = FedConfig(p=7, hidden_width=16, num_clients=2,
                        num_rounds=3, local_epochs=2,
                        seed=42, output_dir=tmp_dir, strategy="fedavg")
        history, _ = fed_train(cfg)
        assert "mean_client_drift" in history
        assert "client_weight_divergence" in history
        # num_rounds + 1 (Flower evaluates initial params at round 0)
        assert len(history["mean_client_drift"]) == 4
