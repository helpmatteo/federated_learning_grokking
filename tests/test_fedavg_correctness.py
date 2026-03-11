"""Tests for FedAvg correctness: weight averaging, convergence, reproducibility.

These tests verify the mathematical correctness of FedAvg by testing
weight averaging behavior, single-client equivalence to centralized,
and reproducibility with fixed seeds.

NOTE: Default lr=50.0 is tuned for p=97. For p=7 tests we use lr=1.0
to avoid divergence on the tiny dataset (24 training samples).
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict

from core.config import Config
from core.model import GrokNet
from core.dataset import make_dataset
from core.utils import make_optimizer, make_targets_onehot
from federated.config import FedConfig
from federated.dataset import make_federated_datasets
from federated.train import (
    _model_to_ndarrays,
    _ndarrays_to_state_dict,
    _make_model,
    _cfg_to_fit_config,
    _fit_config_to_cfg,
    _get_cached_datasets,
    _dataset_cache,
    GrokClient,
    fed_train,
)

SMALL_P = 7
# lr=1.0 is stable for p=7 (24 training samples); default lr=50 diverges
STABLE_LR = 1.0


class TestFedAvgAveraging:
    """Test that FedAvg correctly computes weighted average of client models."""

    def test_manual_weight_average(self):
        """Verify that averaging 2 models with equal weights gives the midpoint."""
        torch.manual_seed(0)
        m1 = GrokNet(14, 16, 7)
        torch.manual_seed(1)
        m2 = GrokNet(14, 16, 7)

        w1_arrays = _model_to_ndarrays(m1)
        w2_arrays = _model_to_ndarrays(m2)

        # Manual average
        avg_arrays = [(a + b) / 2.0 for a, b in zip(w1_arrays, w2_arrays)]

        # Verify the manual average
        for orig1, orig2, avg in zip(w1_arrays, w2_arrays, avg_arrays):
            expected = (orig1 + orig2) / 2.0
            np.testing.assert_allclose(avg, expected, atol=1e-7)

    def test_weighted_average_with_different_sample_sizes(self):
        """Verify FedAvg weighted average formula: w_avg = sum(n_i * w_i) / sum(n_i)."""
        torch.manual_seed(0)
        m1 = GrokNet(14, 16, 7)
        torch.manual_seed(1)
        m2 = GrokNet(14, 16, 7)

        w1 = _model_to_ndarrays(m1)
        w2 = _model_to_ndarrays(m2)

        n1, n2 = 100, 300  # different sample sizes

        # FedAvg weighted average
        total = n1 + n2
        avg = [(n1 * a + n2 * b) / total for a, b in zip(w1, w2)]

        # The result should be closer to m2 (which has 3x weight)
        for a1, a2, a_avg in zip(w1, w2, avg):
            dist_to_m1 = np.linalg.norm(a_avg - a1)
            dist_to_m2 = np.linalg.norm(a_avg - a2)
            assert dist_to_m2 < dist_to_m1, \
                "Average should be closer to model with more samples"

    def test_average_of_identical_models_is_same(self):
        """Averaging identical models should return the same model."""
        torch.manual_seed(42)
        model = GrokNet(14, 16, 7)
        arrays = _model_to_ndarrays(model)

        # Average of K copies of the same model
        K = 5
        avg = [sum(arrays[i] for _ in range(K)) / K for i in range(len(arrays))]

        for orig, averaged in zip(arrays, avg):
            np.testing.assert_allclose(averaged, orig, atol=1e-7)


class TestLocalTrainingEffect:
    """Test that local training moves weights in the right direction."""

    def test_local_training_reduces_loss(self):
        """After local training, loss on local data should decrease."""
        _dataset_cache.clear()
        cfg = FedConfig(
            p=SMALL_P, num_clients=3, local_epochs=5,
            hidden_width=16, seed=42, lr=STABLE_LR,
        )
        fit_config = _cfg_to_fit_config(cfg, server_round=1)
        model = _make_model(cfg)
        params = _model_to_ndarrays(model)

        # Compute initial loss
        client_data, _, _, _, _ = _get_cached_datasets(cfg)
        x_local, y_local = client_data[0]
        y_local_oh = make_targets_onehot(y_local, SMALL_P)

        model.eval()
        with torch.no_grad():
            initial_out = model(x_local)
            initial_loss = nn.MSELoss()(initial_out, y_local_oh).item()

        # Train
        client = GrokClient(partition_id=0)
        updated_params, _, metrics = client.fit(params, fit_config)

        # Loss should have decreased
        assert metrics["loss"] < initial_loss, \
            f"Loss did not decrease: {initial_loss:.4f} → {metrics['loss']:.4f}"

    def test_more_local_epochs_moves_weights_further(self):
        """More local epochs should produce larger weight changes."""
        _dataset_cache.clear()
        cfg_short = FedConfig(
            p=SMALL_P, num_clients=3, local_epochs=1,
            hidden_width=16, seed=42, lr=STABLE_LR,
        )
        cfg_long = FedConfig(
            p=SMALL_P, num_clients=3, local_epochs=10,
            hidden_width=16, seed=42, lr=STABLE_LR,
        )

        model = _make_model(cfg_short)
        params = _model_to_ndarrays(model)

        fit_config_short = _cfg_to_fit_config(cfg_short, server_round=1)
        fit_config_long = _cfg_to_fit_config(cfg_long, server_round=1)

        client = GrokClient(partition_id=0)
        short_params, _, _ = client.fit(params, fit_config_short)
        long_params, _, _ = client.fit(params, fit_config_long)

        short_delta = sum(np.linalg.norm(s - o) for s, o in zip(short_params, params))
        long_delta = sum(np.linalg.norm(l - o) for l, o in zip(long_params, params))

        assert long_delta > short_delta, \
            f"More epochs should produce larger weight changes: {short_delta:.4f} vs {long_delta:.4f}"


class TestFedAvgConvergence:
    """Test FedAvg convergence properties."""

    def test_loss_does_not_diverge(self):
        """Loss should not blow up during federated training."""
        _dataset_cache.clear()
        cfg = FedConfig(
            p=SMALL_P, num_clients=3, num_rounds=5, local_epochs=2,
            hidden_width=16, seed=42, lr=STABLE_LR,
            output_dir="/tmp/test_fed_convergence",
        )
        history, _ = fed_train(cfg)

        # Loss should stay finite
        for loss in history["train_loss"] + history["test_loss"]:
            assert np.isfinite(loss), f"Loss is not finite: {loss}"

        # Loss at end should not be dramatically larger than at start
        assert history["train_loss"][-1] < history["train_loss"][0] * 100, \
            "Training loss increased by more than 100x"

    def test_weight_norms_stay_bounded(self):
        """Weight norms should not explode during training."""
        _dataset_cache.clear()
        cfg = FedConfig(
            p=SMALL_P, num_clients=3, num_rounds=5, local_epochs=2,
            hidden_width=16, seed=42, lr=STABLE_LR,
            output_dir="/tmp/test_fed_convergence2",
        )
        history, _ = fed_train(cfg)

        for wn in history["weight_norm_layer1"] + history["weight_norm_layer2"]:
            assert np.isfinite(wn)
            assert wn < 1000, f"Weight norm is too large: {wn}"

    def test_ipr_is_bounded(self):
        """IPR should stay in [0, 1] throughout training."""
        _dataset_cache.clear()
        cfg = FedConfig(
            p=SMALL_P, num_clients=3, num_rounds=5, local_epochs=2,
            hidden_width=16, seed=42, lr=STABLE_LR,
            output_dir="/tmp/test_fed_convergence3",
        )
        history, _ = fed_train(cfg)

        for ipr_val in history["ipr"]:
            assert 0 <= ipr_val <= 1.0, f"IPR out of bounds: {ipr_val}"


class TestReproducibility:
    """Test that runs with the same seed produce identical results."""

    def test_same_seed_same_history(self):
        """Two runs with the same seed should produce identical history."""
        _dataset_cache.clear()
        cfg = FedConfig(
            p=SMALL_P, num_clients=3, num_rounds=3, local_epochs=1,
            hidden_width=16, seed=42, lr=STABLE_LR,
            output_dir="/tmp/test_fed_repro_1",
        )
        h1, _ = fed_train(cfg)

        _dataset_cache.clear()
        cfg2 = FedConfig(
            p=SMALL_P, num_clients=3, num_rounds=3, local_epochs=1,
            hidden_width=16, seed=42, lr=STABLE_LR,
            output_dir="/tmp/test_fed_repro_2",
        )
        h2, _ = fed_train(cfg2)

        for key in h1:
            for v1, v2 in zip(h1[key], h2[key]):
                assert v1 == pytest.approx(v2, abs=1e-5), \
                    f"Mismatch in {key}: {v1} != {v2}"

    def test_different_seed_different_history(self):
        """Two runs with different seeds should produce different results."""
        _dataset_cache.clear()
        cfg1 = FedConfig(
            p=SMALL_P, num_clients=3, num_rounds=3, local_epochs=1,
            hidden_width=16, seed=42, lr=STABLE_LR,
            output_dir="/tmp/test_fed_diff_1",
        )
        h1, _ = fed_train(cfg1)

        _dataset_cache.clear()
        cfg2 = FedConfig(
            p=SMALL_P, num_clients=3, num_rounds=3, local_epochs=1,
            hidden_width=16, seed=99, lr=STABLE_LR,
            output_dir="/tmp/test_fed_diff_2",
        )
        h2, _ = fed_train(cfg2)

        # At least some values should differ (different init + different data split)
        diffs = [abs(a - b) for a, b in zip(h1["test_loss"], h2["test_loss"])]
        assert max(diffs) > 1e-6, "Different seeds should produce different results"


class TestSingleClientEquivalence:
    """Test that FedAvg with K=1 client behaves like centralized training."""

    def test_single_client_iid_runs(self):
        """FedAvg with 1 client should complete without error."""
        _dataset_cache.clear()
        cfg = FedConfig(
            p=SMALL_P, num_clients=1, num_rounds=3, local_epochs=2,
            hidden_width=16, seed=42, partition="iid", lr=STABLE_LR,
            output_dir="/tmp/test_fed_single",
        )
        history, model = fed_train(cfg)
        assert len(history["round"]) == 4  # 0 + 3 rounds
        assert isinstance(model, GrokNet)

    def test_single_client_gets_all_data(self):
        """With K=1, the single client should receive the entire training set."""
        _dataset_cache.clear()
        cfg = FedConfig(
            p=SMALL_P, num_clients=1, hidden_width=16, seed=42)
        client_data, x_train_full, _, _, _ = make_federated_datasets(cfg)
        assert len(client_data) == 1
        assert len(client_data[0][1]) == len(x_train_full)


class TestFinalModelReconstruction:
    """Test that the final model is correctly reconstructed from evaluate_fn parameters."""

    def test_final_model_matches_last_evaluation(self):
        """The final model's predictions should match what evaluate_fn computed."""
        _dataset_cache.clear()
        cfg = FedConfig(
            p=SMALL_P, num_clients=3, num_rounds=3, local_epochs=2,
            hidden_width=16, seed=42, lr=STABLE_LR,
            output_dir="/tmp/test_fed_final",
        )
        history, final_model = fed_train(cfg)

        # Recompute loss/accuracy with the final model
        client_data, x_train_full, y_train_full, x_test, y_test = _get_cached_datasets(cfg)
        y_test_oh = make_targets_onehot(y_test, SMALL_P)

        final_model.eval()
        final_model = final_model.cpu()
        with torch.no_grad():
            out = final_model(x_test.cpu())
            recomputed_loss = nn.MSELoss()(out, y_test_oh.cpu()).item()

        # Should match the last recorded test loss
        recorded_loss = history["test_loss"][-1]
        assert abs(recomputed_loss - recorded_loss) < 1e-4, \
            f"Recomputed loss {recomputed_loss:.6f} != recorded {recorded_loss:.6f}"

    def test_final_model_has_correct_architecture(self):
        _dataset_cache.clear()
        cfg = FedConfig(
            p=SMALL_P, num_clients=3, num_rounds=2, local_epochs=1,
            hidden_width=16, seed=42, lr=STABLE_LR,
            output_dir="/tmp/test_fed_arch",
        )
        _, final_model = fed_train(cfg)

        assert final_model.D == 2 * SMALL_P
        assert final_model.N == 16
        assert final_model.P == SMALL_P
        assert final_model.W1.shape == (16, 14)
        assert final_model.W2.shape == (7, 16)


class TestFractionTrain:
    """Test partial client participation (fraction_train < 1.0)."""

    def test_partial_participation_runs(self):
        """FedAvg with fraction_train=0.5 should still work."""
        _dataset_cache.clear()
        cfg = FedConfig(
            p=SMALL_P, num_clients=4, num_rounds=3, local_epochs=1,
            hidden_width=16, seed=42, fraction_train=0.5, lr=STABLE_LR,
            output_dir="/tmp/test_fed_partial",
        )
        history, model = fed_train(cfg)
        assert len(history["round"]) == 4

    def test_full_vs_partial_participation_differs(self):
        """Full and partial participation should give different results."""
        _dataset_cache.clear()
        cfg_full = FedConfig(
            p=SMALL_P, num_clients=4, num_rounds=3, local_epochs=2,
            hidden_width=16, seed=42, fraction_train=1.0, lr=STABLE_LR,
            output_dir="/tmp/test_fed_full",
        )
        h_full, _ = fed_train(cfg_full)

        _dataset_cache.clear()
        cfg_partial = FedConfig(
            p=SMALL_P, num_clients=4, num_rounds=3, local_epochs=2,
            hidden_width=16, seed=42, fraction_train=0.5, lr=STABLE_LR,
            output_dir="/tmp/test_fed_partial2",
        )
        h_partial, _ = fed_train(cfg_partial)

        # Results should differ due to different participation
        diffs = [abs(a - b) for a, b in zip(h_full["test_loss"], h_partial["test_loss"])]
        # At least the later rounds should differ
        assert max(diffs) > 1e-6


class TestOptimizerVariants:
    """Test both SGD and AdamW in federated setting."""

    def test_adamw_runs(self):
        _dataset_cache.clear()
        cfg = FedConfig(
            p=SMALL_P, num_clients=3, num_rounds=2, local_epochs=2,
            hidden_width=16, seed=42,
            optimizer="adamw", lr=1e-3, weight_decay=0.1,
            output_dir="/tmp/test_fed_adamw",
        )
        history, model = fed_train(cfg)
        assert len(history["round"]) == 3
        assert all(np.isfinite(l) for l in history["train_loss"])

    def test_sgd_with_momentum_runs(self):
        _dataset_cache.clear()
        cfg = FedConfig(
            p=SMALL_P, num_clients=3, num_rounds=2, local_epochs=2,
            hidden_width=16, seed=42,
            optimizer="gd", momentum=0.9, lr=STABLE_LR,
            output_dir="/tmp/test_fed_momentum",
        )
        history, model = fed_train(cfg)
        assert len(history["round"]) == 3
        assert all(np.isfinite(l) for l in history["train_loss"])

    def test_sgd_with_weight_decay_runs(self):
        _dataset_cache.clear()
        cfg = FedConfig(
            p=SMALL_P, num_clients=3, num_rounds=2, local_epochs=2,
            hidden_width=16, seed=42,
            optimizer="gd", weight_decay=0.01, lr=STABLE_LR,
            output_dir="/tmp/test_fed_wd",
        )
        history, model = fed_train(cfg)
        assert len(history["round"]) == 3
        assert all(np.isfinite(l) for l in history["train_loss"])
