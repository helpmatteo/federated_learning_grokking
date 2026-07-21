"""Tests for federated/train.py: weight serialization, dataset cache, GrokClient, fed_train."""

import pytest
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict

from fedgrok.models.groknet import GrokNet
from fedgrok.core.utils import make_targets_onehot, make_optimizer
from fedgrok.core.fed_config import FedConfig
from fedgrok.data.partition import make_federated_datasets
from fedgrok.training.federated import (
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
# lr=1.0 is stable for p=7; default lr=50 is tuned for p=97 and diverges here
STABLE_LR = 1.0


# ── Weight serialization ─────────────────────────────────────────────────────

class TestWeightSerialization:
    def test_model_to_ndarrays_returns_list(self, small_model):
        arrays = _model_to_ndarrays(small_model)
        assert isinstance(arrays, list)
        assert len(arrays) == 2  # W1, W2

    def test_model_to_ndarrays_shapes(self, small_model):
        arrays = _model_to_ndarrays(small_model)
        assert arrays[0].shape == (16, 14)  # W1: (N, 2p)
        assert arrays[1].shape == (7, 16)   # W2: (P, N)

    def test_model_to_ndarrays_are_numpy(self, small_model):
        arrays = _model_to_ndarrays(small_model)
        for arr in arrays:
            assert isinstance(arr, np.ndarray)

    def test_ndarrays_to_state_dict_roundtrip(self, small_model):
        """model → ndarrays → state_dict should preserve weights exactly."""
        original_w1 = small_model.W1.data.clone()
        original_w2 = small_model.W2.data.clone()

        arrays = _model_to_ndarrays(small_model)
        state_dict = _ndarrays_to_state_dict(arrays, small_model)

        assert "W1" in state_dict
        assert "W2" in state_dict
        assert torch.allclose(state_dict["W1"], original_w1)
        assert torch.allclose(state_dict["W2"], original_w2)

    def test_full_roundtrip_through_model(self, small_model):
        """model → ndarrays → state_dict → new model: weights should match."""
        original_w1 = small_model.W1.data.clone()
        original_w2 = small_model.W2.data.clone()

        arrays = _model_to_ndarrays(small_model)

        new_model = GrokNet(14, 16, 7)
        state_dict = _ndarrays_to_state_dict(arrays, new_model)
        new_model.load_state_dict(state_dict)

        assert torch.allclose(new_model.W1.data, original_w1)
        assert torch.allclose(new_model.W2.data, original_w2)

    def test_roundtrip_preserves_forward_pass(self, small_model, random_batch):
        """Serialization roundtrip should produce identical model outputs."""
        original_out = small_model(random_batch)

        arrays = _model_to_ndarrays(small_model)
        new_model = GrokNet(14, 16, 7, activation=small_model.activation)
        state_dict = _ndarrays_to_state_dict(arrays, new_model)
        new_model.load_state_dict(state_dict)
        new_out = new_model(random_batch)

        assert torch.allclose(original_out, new_out, atol=1e-6)

    def test_model_to_ndarrays_on_cpu(self):
        """_model_to_ndarrays should always return CPU arrays."""
        model = GrokNet(14, 16, 7)
        # Even on MPS/CUDA, should work
        arrays = _model_to_ndarrays(model)
        for arr in arrays:
            assert isinstance(arr, np.ndarray)


# ── _make_model ───────────────────────────────────────────────────────────────

class TestMakeModel:
    def test_creates_groknet(self, small_fed_cfg):
        model = _make_model(small_fed_cfg)
        assert isinstance(model, GrokNet)

    def test_correct_dimensions(self, small_fed_cfg):
        model = _make_model(small_fed_cfg)
        assert model.D == 2 * small_fed_cfg.p
        assert model.N == small_fed_cfg.hidden_width
        assert model.P == small_fed_cfg.p

    def test_model_on_cpu(self, small_fed_cfg):
        model = _make_model(small_fed_cfg)
        assert next(model.parameters()).device == torch.device("cpu")


# ── Dataset cache ─────────────────────────────────────────────────────────────

class TestDatasetCache:
    def setup_method(self):
        """Clear the global cache before each test."""
        _dataset_cache.clear()

    def test_cache_hit(self):
        cfg = FedConfig(p=SMALL_P, num_clients=3, hidden_width=16, seed=42)
        result1 = _get_cached_datasets(cfg)
        result2 = _get_cached_datasets(cfg)
        # Should be the exact same objects
        assert result1 is result2

    def test_cache_miss_on_different_partition(self):
        cfg1 = FedConfig(p=SMALL_P, num_clients=3, partition="iid", hidden_width=16)
        cfg2 = FedConfig(p=SMALL_P, num_clients=3, partition="operand", hidden_width=16)
        result1 = _get_cached_datasets(cfg1)
        result2 = _get_cached_datasets(cfg2)
        assert result1 is not result2

    def test_cache_miss_on_different_num_clients(self):
        cfg1 = FedConfig(p=SMALL_P, num_clients=3, hidden_width=16)
        cfg2 = FedConfig(p=SMALL_P, num_clients=5, hidden_width=16)
        result1 = _get_cached_datasets(cfg1)
        result2 = _get_cached_datasets(cfg2)
        assert result1 is not result2

    def test_cache_miss_on_different_seed(self):
        cfg1 = FedConfig(p=SMALL_P, num_clients=3, hidden_width=16, seed=1)
        cfg2 = FedConfig(p=SMALL_P, num_clients=3, hidden_width=16, seed=2)
        result1 = _get_cached_datasets(cfg1)
        result2 = _get_cached_datasets(cfg2)
        assert result1 is not result2

    def test_cache_miss_on_different_dirichlet_alpha(self):
        cfg1 = FedConfig(p=SMALL_P, num_clients=3, partition="dirichlet",
                         dirichlet_alpha=0.1, hidden_width=16)
        cfg2 = FedConfig(p=SMALL_P, num_clients=3, partition="dirichlet",
                         dirichlet_alpha=1.0, hidden_width=16)
        result1 = _get_cached_datasets(cfg1)
        result2 = _get_cached_datasets(cfg2)
        assert result1 is not result2

    def test_cache_key_ignores_training_params(self):
        """Cache key should NOT include lr, optimizer, etc. — only data params."""
        cfg1 = FedConfig(p=SMALL_P, num_clients=3, hidden_width=16, lr=50.0)
        cfg2 = FedConfig(p=SMALL_P, num_clients=3, hidden_width=16, lr=0.001)
        result1 = _get_cached_datasets(cfg1)
        result2 = _get_cached_datasets(cfg2)
        # Same data params, different training params → should be same object
        assert result1 is result2


# ── GrokClient ────────────────────────────────────────────────────────────────

class TestGrokClient:
    def _make_client_config(self, partition_id=0, partition="iid"):
        """Create a fit config dict mimicking what Flower sends."""
        cfg = FedConfig(
            p=SMALL_P, num_clients=3, local_epochs=2,
            hidden_width=16, partition=partition, seed=42, lr=STABLE_LR,
        )
        return cfg, _cfg_to_fit_config(cfg, server_round=1)

    def _make_initial_parameters(self, cfg):
        """Create initial model parameters as ndarrays."""
        model = _make_model(cfg)
        return _model_to_ndarrays(model)

    def test_fit_returns_correct_structure(self):
        _dataset_cache.clear()
        cfg, fit_config = self._make_client_config()
        params = self._make_initial_parameters(cfg)

        client = GrokClient(partition_id=0)
        result = client.fit(params, fit_config)

        assert len(result) == 3
        updated_params, num_samples, metrics = result
        assert isinstance(updated_params, list)
        assert len(updated_params) == 2  # W1, W2
        assert isinstance(num_samples, int)
        assert num_samples > 0
        assert "loss" in metrics
        assert "accuracy" in metrics

    def test_fit_modifies_weights(self):
        """After local training, weights should be different from initial."""
        _dataset_cache.clear()
        cfg, fit_config = self._make_client_config()
        params = self._make_initial_parameters(cfg)

        client = GrokClient(partition_id=0)
        updated_params, _, _ = client.fit(params, fit_config)

        # At least one weight array should have changed
        changed = any(
            not np.allclose(orig, updated, atol=1e-10)
            for orig, updated in zip(params, updated_params)
        )
        assert changed, "Weights should change after local training"

    def test_fit_preserves_shapes(self):
        _dataset_cache.clear()
        cfg, fit_config = self._make_client_config()
        params = self._make_initial_parameters(cfg)

        client = GrokClient(partition_id=0)
        updated_params, _, _ = client.fit(params, fit_config)

        for orig, updated in zip(params, updated_params):
            assert orig.shape == updated.shape

    def test_different_partitions_produce_different_updates(self):
        """Clients with different data should produce different weight updates."""
        _dataset_cache.clear()
        cfg, fit_config = self._make_client_config()
        params = self._make_initial_parameters(cfg)

        client0 = GrokClient(partition_id=0)
        client1 = GrokClient(partition_id=1)

        updated0, _, _ = client0.fit(params, fit_config)
        updated1, _, _ = client1.fit(params, fit_config)

        # Different partitions should produce different updates
        different = any(
            not np.allclose(a, b, atol=1e-10)
            for a, b in zip(updated0, updated1)
        )
        assert different, "Different clients should produce different weight updates"

    def test_fit_returns_correct_num_samples(self):
        _dataset_cache.clear()
        cfg = FedConfig(
            p=SMALL_P, num_clients=3, local_epochs=1,
            hidden_width=16, partition="iid", seed=42, lr=STABLE_LR,
        )
        fit_config = _cfg_to_fit_config(cfg, server_round=1)
        params = self._make_initial_parameters(cfg)

        # Get the actual partition size
        client_data, _, _, _, _ = make_federated_datasets(cfg)
        expected_size = len(client_data[0][1])

        client = GrokClient(partition_id=0)
        _, num_samples, _ = client.fit(params, fit_config)
        assert num_samples == expected_size

    def test_evaluate_returns_stub(self):
        client = GrokClient(partition_id=0)
        loss, num_samples, metrics = client.evaluate([], {})
        assert loss == 0.0
        assert num_samples == 0
        assert metrics == {}

    def test_fit_loss_is_finite(self):
        _dataset_cache.clear()
        cfg, fit_config = self._make_client_config()
        params = self._make_initial_parameters(cfg)

        client = GrokClient(partition_id=0)
        _, _, metrics = client.fit(params, fit_config)
        assert np.isfinite(metrics["loss"])
        assert 0 <= metrics["accuracy"] <= 100


# ── fed_train integration ─────────────────────────────────────────────────────

class TestFedTrain:
    def setup_method(self):
        _dataset_cache.clear()

    def test_returns_history_and_model(self):
        cfg = FedConfig(
            p=SMALL_P, num_clients=3, num_rounds=2, local_epochs=1,
            hidden_width=16, seed=42, lr=STABLE_LR,
            output_dir="/tmp/test_fed_results",
        )
        history, model = fed_train(cfg)
        assert isinstance(history, dict)
        assert isinstance(model, GrokNet)

    def test_history_has_correct_keys(self):
        cfg = FedConfig(
            p=SMALL_P, num_clients=3, num_rounds=2, local_epochs=1,
            hidden_width=16, seed=42, lr=STABLE_LR,
            output_dir="/tmp/test_fed_results",
        )
        history, _ = fed_train(cfg)
        expected_keys = {
            "round", "total_steps", "sequential_steps", "n_participating",
            "train_loss", "test_loss",
            "train_acc", "test_acc",
            "weight_norm_layer1", "weight_norm_layer2",
            "ipr",
            "mean_client_drift", "client_weight_divergence",
        }
        assert set(history.keys()) == expected_keys

    def test_history_length_matches_rounds(self):
        cfg = FedConfig(
            p=SMALL_P, num_clients=3, num_rounds=3, local_epochs=1,
            hidden_width=16, seed=42, lr=STABLE_LR,
            output_dir="/tmp/test_fed_results",
        )
        history, _ = fed_train(cfg)
        # Round 0 (initial eval) + rounds 1..3 = 4 entries
        assert len(history["round"]) == 4

    def test_total_steps_computation(self):
        local_epochs = 2
        cfg = FedConfig(
            p=SMALL_P, num_clients=3, num_rounds=3, local_epochs=local_epochs,
            hidden_width=16, seed=42, lr=STABLE_LR,
            output_dir="/tmp/test_fed_results",
        )
        history, _ = fed_train(cfg)
        for i, (rnd, steps) in enumerate(zip(history["round"], history["total_steps"])):
            assert steps == rnd * local_epochs

    def test_final_model_not_initial(self):
        """Final model should have different weights than random init."""
        torch.manual_seed(42)
        cfg = FedConfig(
            p=SMALL_P, num_clients=3, num_rounds=3, local_epochs=2,
            hidden_width=16, seed=42, lr=STABLE_LR,
            output_dir="/tmp/test_fed_results",
        )
        history, final_model = fed_train(cfg)

        # Create a fresh model with same init to compare
        torch.manual_seed(42)
        init_model = _make_model(cfg)

        # Weights should differ after training
        w1_changed = not torch.allclose(final_model.W1.data.cpu(), init_model.W1.data, atol=1e-8)
        w2_changed = not torch.allclose(final_model.W2.data.cpu(), init_model.W2.data, atol=1e-8)
        assert w1_changed or w2_changed, "Final model weights should differ from init"

    def test_losses_are_finite(self):
        cfg = FedConfig(
            p=SMALL_P, num_clients=3, num_rounds=2, local_epochs=1,
            hidden_width=16, seed=42, lr=STABLE_LR,
            output_dir="/tmp/test_fed_results",
        )
        history, _ = fed_train(cfg)
        for loss in history["train_loss"] + history["test_loss"]:
            assert np.isfinite(loss)

    def test_accuracies_are_bounded(self):
        cfg = FedConfig(
            p=SMALL_P, num_clients=3, num_rounds=2, local_epochs=1,
            hidden_width=16, seed=42, lr=STABLE_LR,
            output_dir="/tmp/test_fed_results",
        )
        history, _ = fed_train(cfg)
        for acc in history["train_acc"] + history["test_acc"]:
            assert 0 <= acc <= 100

    @pytest.mark.parametrize("partition", ["iid", "operand", "target", "dirichlet"])
    def test_all_partitions_run(self, partition):
        _dataset_cache.clear()
        cfg = FedConfig(
            p=SMALL_P, num_clients=3, num_rounds=2, local_epochs=1,
            hidden_width=16, partition=partition, seed=42, lr=STABLE_LR,
            output_dir="/tmp/test_fed_results",
        )
        history, model = fed_train(cfg)
        assert len(history["round"]) == 3  # round 0 + 2 rounds

    def test_saves_history_json(self, tmp_path):
        import json
        cfg = FedConfig(
            p=SMALL_P, num_clients=3, num_rounds=2, local_epochs=1,
            hidden_width=16, seed=42, lr=STABLE_LR,
            output_dir=str(tmp_path),
        )
        fed_train(cfg)

        # Find the saved JSON
        json_files = list(tmp_path.glob("history_*.json"))
        assert len(json_files) == 1

        with open(json_files[0]) as f:
            saved = json.load(f)
        assert "round" in saved
        assert "test_acc" in saved

    def test_saves_weights_when_requested(self, tmp_path):
        cfg = FedConfig(
            p=SMALL_P, num_clients=3, num_rounds=2, local_epochs=1,
            hidden_width=16, seed=42, lr=STABLE_LR,
            output_dir=str(tmp_path),
            save_weights=True,
        )
        fed_train(cfg)

        pt_files = list(tmp_path.glob("weights_*.pt"))
        assert len(pt_files) == 1

        # Verify saved weights can be loaded
        state_dict = torch.load(pt_files[0], weights_only=True)
        assert "W1" in state_dict
        assert "W2" in state_dict


# ── Step accounting ──────────────────────────────────────────────────────────


class TestStepAccounting:
    """total_steps must reflect actual gradient work, not rounds * E.

    The old accounting was `server_round * local_epochs`, which is correct only
    at full participation. Under partial participation it over-counted, since
    only a fraction of the clients (and hence of the data) trained each round.
    """

    def _run(self, fraction_train, tmp_path):
        from fedgrok.training.federated import fed_train, _dataset_cache
        _dataset_cache.clear()
        cfg = FedConfig(
            task="addition", p=7, alpha=0.5, seed=42, hidden_width=16,
            num_clients=5, num_rounds=5, local_epochs=3, lr=1.0,
            fraction_train=fraction_train, partition="iid",
            output_dir=str(tmp_path),
        )
        history, _ = fed_train(cfg)
        return history

    def test_full_participation_matches_rounds_times_epochs(self, tmp_path):
        """At C=1.0 the compute-matched axis coincides with rounds * E."""
        history = self._run(1.0, tmp_path)
        expected = [r * 3 for r in history["round"]]
        assert history["total_steps"] == pytest.approx(expected)
        assert history["sequential_steps"] == expected

    def test_partial_participation_counts_less_than_sequential(self, tmp_path):
        """At C<1.0 fewer samples are touched per round, so total_steps lags."""
        history = self._run(0.4, tmp_path)
        assert history["total_steps"][-1] < history["sequential_steps"][-1]
        # sequential_steps is participation-independent
        assert history["sequential_steps"] == [r * 3 for r in history["round"]]

    def test_participating_client_count_is_recorded(self, tmp_path):
        history = self._run(0.4, tmp_path)
        # Round 0 is the pre-training evaluation of the initial parameters.
        assert history["n_participating"][0] == 0
        assert all(n == 2 for n in history["n_participating"][1:])

    def test_total_steps_is_monotonic(self, tmp_path):
        history = self._run(0.4, tmp_path)
        steps = history["total_steps"]
        assert all(b >= a for a, b in zip(steps, steps[1:]))
