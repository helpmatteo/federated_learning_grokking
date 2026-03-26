"""Tests for core/ package: Config, GrokNet, dataset, metrics, utils."""

import math
import pytest
import torch
import numpy as np

from core.config import Config
from core.model import GrokNet
from core.dataset import TASKS, make_dataset
from core.metrics import weight_norms, gradient_norms, compute_ipr, compute_accuracy
from core.utils import get_device, make_optimizer, make_targets_onehot


# ── Config ────────────────────────────────────────────────────────────────────

class TestConfig:
    def test_defaults(self):
        cfg = Config()
        assert cfg.p == 97
        assert cfg.task == "addition"
        assert cfg.alpha == 0.5
        assert cfg.hidden_width == 100
        assert cfg.activation == "quadratic"
        assert cfg.optimizer == "gd"
        assert cfg.lr == 50.0
        assert cfg.weight_decay == 0.0
        assert cfg.momentum == 0.0
        assert cfg.output_dir == "results/baselines/centralized"

    def test_apply_adamw_defaults_overrides_when_not_set(self):
        cfg = Config(optimizer="adamw")
        cfg.apply_adamw_defaults()
        assert cfg.lr == 1e-4
        assert cfg.weight_decay == 1.0
        assert cfg.epochs == 5000

    def test_apply_adamw_defaults_preserves_user_values(self):
        cfg = Config(optimizer="adamw", lr=0.001, _lr_set=True,
                     weight_decay=0.5, _wd_set=True,
                     epochs=1000, _epochs_set=True)
        cfg.apply_adamw_defaults()
        assert cfg.lr == 0.001
        assert cfg.weight_decay == 0.5
        assert cfg.epochs == 1000

    def test_apply_adamw_defaults_noop_for_gd(self):
        cfg = Config(optimizer="gd", lr=50.0)
        cfg.apply_adamw_defaults()
        assert cfg.lr == 50.0
        assert cfg.weight_decay == 0.0


# ── GrokNet ───────────────────────────────────────────────────────────────────

class TestGrokNet:
    def test_output_shape(self, small_model, random_batch):
        out = small_model(random_batch)
        assert out.shape == (10, 7)

    def test_no_bias_parameters(self, small_model):
        param_names = [name for name, _ in small_model.named_parameters()]
        assert param_names == ["W1", "W2"]

    def test_weight_shapes(self, small_model):
        assert small_model.W1.shape == (16, 14)  # (N, 2p)
        assert small_model.W2.shape == (7, 16)   # (P, N)

    def test_mean_field_init_scale(self):
        """W1 ~ N(0, 1/D), W2 ~ N(0, 1/N^2)."""
        torch.manual_seed(0)
        p, N = 97, 200
        model = GrokNet(2 * p, N, p)
        # W1 std should be close to 1/sqrt(D) = 1/sqrt(194) ≈ 0.0718
        w1_std = model.W1.data.std().item()
        assert abs(w1_std - 1.0 / math.sqrt(2 * p)) < 0.01
        # W2 std should be close to 1/N = 0.005
        w2_std = model.W2.data.std().item()
        assert abs(w2_std - 1.0 / N) < 0.002

    @pytest.mark.parametrize("activation", ["quadratic", "relu", "gelu", "abs", "quartic"])
    def test_all_activations_produce_output(self, activation):
        model = GrokNet(14, 16, 7, activation=activation)
        x = torch.randn(5, 14)
        out = model(x)
        assert out.shape == (5, 7)
        assert torch.isfinite(out).all()

    def test_invalid_activation_raises(self):
        model = GrokNet(14, 16, 7, activation="invalid")
        with pytest.raises(ValueError, match="Unknown activation"):
            model(torch.randn(1, 14))

    def test_quadratic_activation_is_squaring(self):
        model = GrokNet(14, 16, 7, activation="quadratic")
        h = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        result = model._activate(h)
        expected = torch.tensor([4.0, 1.0, 0.0, 1.0, 4.0])
        assert torch.allclose(result, expected)

    def test_deterministic_with_seed(self):
        torch.manual_seed(42)
        m1 = GrokNet(14, 16, 7)
        w1 = m1.W1.data.clone()

        torch.manual_seed(42)
        m2 = GrokNet(14, 16, 7)
        assert torch.equal(m1.W1.data, m2.W1.data)
        assert torch.equal(m1.W2.data, m2.W2.data)


# ── Dataset ───────────────────────────────────────────────────────────────────

class TestDataset:
    def test_shapes(self, small_cfg):
        x_train, y_train, x_test, y_test = make_dataset(small_cfg)
        p = small_cfg.p
        n_total = p * p
        n_train = int(small_cfg.alpha * n_total)
        n_test = n_total - n_train

        assert x_train.shape == (n_train, 2 * p)
        assert y_train.shape == (n_train,)
        assert x_test.shape == (n_test, 2 * p)
        assert y_test.shape == (n_test,)

    def test_label_range(self, small_cfg):
        _, y_train, _, y_test = make_dataset(small_cfg)
        for y in [y_train, y_test]:
            assert y.min() >= 0
            assert y.max() < small_cfg.p
            assert y.dtype == torch.long

    def test_onehot_input_structure(self, small_cfg):
        """Each row of x should have exactly 2 ones (one-hot for n and m)."""
        x_train, _, _, _ = make_dataset(small_cfg)
        row_sums = x_train.sum(dim=1)
        assert torch.allclose(row_sums, torch.full_like(row_sums, 2.0))

    def test_no_train_test_overlap(self, small_cfg):
        """Train and test sets should partition the full p^2 dataset."""
        x_train, _, x_test, _ = make_dataset(small_cfg)
        n_total = small_cfg.p ** 2
        assert len(x_train) + len(x_test) == n_total

    def test_seed_reproducibility(self, small_cfg):
        d1 = make_dataset(small_cfg)
        d2 = make_dataset(small_cfg)
        for a, b in zip(d1, d2):
            assert torch.equal(a, b)

    def test_different_seed_gives_different_split(self, small_cfg):
        cfg2 = Config(p=small_cfg.p, seed=99, hidden_width=16)
        x1, _, _, _ = make_dataset(small_cfg)
        x2, _, _, _ = make_dataset(cfg2)
        # Different seeds should produce different train sets
        assert not torch.equal(x1, x2)

    @pytest.mark.parametrize("task_name", list(TASKS.keys()))
    def test_all_tasks_produce_valid_labels(self, task_name):
        cfg = Config(p=7, task=task_name, hidden_width=16)
        _, y_train, _, y_test = make_dataset(cfg)
        for y in [y_train, y_test]:
            assert y.min() >= 0
            assert y.max() < 7

    def test_addition_correctness(self):
        """Spot-check: for addition mod p, verify labels match (n+m) mod p."""
        cfg = Config(p=5, alpha=1.0, hidden_width=8, seed=0)
        x, y, _, _ = make_dataset(cfg)
        p = 5
        for i in range(len(x)):
            # Decode n and m from one-hot
            n = x[i, :p].argmax().item()
            m = x[i, p:].argmax().item()
            expected = (n + m) % p
            assert y[i].item() == expected, f"n={n}, m={m}: got {y[i].item()}, expected {expected}"


# ── Metrics ───────────────────────────────────────────────────────────────────

class TestMetrics:
    def test_weight_norms_returns_positive(self, small_model):
        wn = weight_norms(small_model)
        assert "weight_norm_layer1" in wn
        assert "weight_norm_layer2" in wn
        assert wn["weight_norm_layer1"] > 0
        assert wn["weight_norm_layer2"] > 0

    def test_weight_norms_matches_manual(self, small_model):
        wn = weight_norms(small_model)
        expected_w1 = small_model.W1.data.norm().item()
        expected_w2 = small_model.W2.data.norm().item()
        assert abs(wn["weight_norm_layer1"] - expected_w1) < 1e-6
        assert abs(wn["weight_norm_layer2"] - expected_w2) < 1e-6

    def test_gradient_norms_before_backward(self, small_model):
        """Before backward, gradients don't exist — should return empty."""
        gn = gradient_norms(small_model)
        assert gn == {}

    def test_gradient_norms_after_backward(self, trained_model_with_grads):
        gn = gradient_norms(trained_model_with_grads)
        assert "grad_norm_layer1" in gn
        assert "grad_norm_layer2" in gn
        assert gn["grad_norm_layer1"] > 0
        assert gn["grad_norm_layer2"] > 0

    def test_compute_ipr_returns_positive(self, small_model):
        ipr = compute_ipr(small_model)
        assert "ipr" in ipr
        assert ipr["ipr"] > 0

    def test_compute_ipr_bounded(self, small_model):
        """IPR should be between 0 and 1 (normalized)."""
        ipr_val = compute_ipr(small_model)["ipr"]
        assert 0 < ipr_val <= 1.0

    def test_compute_accuracy_perfect(self):
        logits = torch.tensor([[10.0, 0.0, 0.0],
                               [0.0, 10.0, 0.0],
                               [0.0, 0.0, 10.0]])
        targets = torch.tensor([0, 1, 2])
        assert compute_accuracy(logits, targets) == 100.0

    def test_compute_accuracy_zero(self):
        logits = torch.tensor([[10.0, 0.0, 0.0],
                               [10.0, 0.0, 0.0],
                               [10.0, 0.0, 0.0]])
        targets = torch.tensor([1, 2, 2])
        assert compute_accuracy(logits, targets) == 0.0

    def test_compute_accuracy_partial(self):
        logits = torch.tensor([[10.0, 0.0],
                               [0.0, 10.0],
                               [10.0, 0.0],
                               [0.0, 10.0]])
        targets = torch.tensor([0, 1, 1, 0])
        assert compute_accuracy(logits, targets) == 50.0

    def test_fourier_spectrum_shape(self, small_model):
        from core.metrics import fourier_spectrum
        spec = fourier_spectrum(small_model)
        assert "spectrum" in spec
        assert len(spec["spectrum"]) == small_model.N
        assert len(spec["spectrum"][0]) == small_model.P

    def test_fourier_spectrum_nonnegative(self, small_model):
        from core.metrics import fourier_spectrum
        spec = fourier_spectrum(small_model)
        for row in spec["spectrum"]:
            assert all(v >= 0 for v in row)


# ── Utils ─────────────────────────────────────────────────────────────────────

class TestUtils:
    def test_get_device_returns_device(self):
        device = get_device()
        assert isinstance(device, torch.device)
        assert device.type in ("cuda", "mps", "cpu")

    def test_make_optimizer_sgd(self, small_model, small_cfg):
        opt = make_optimizer(small_model, small_cfg)
        assert isinstance(opt, torch.optim.SGD)

    def test_make_optimizer_adamw(self, small_model):
        cfg = Config(optimizer="adamw", lr=1e-4, weight_decay=1.0, hidden_width=16)
        opt = make_optimizer(small_model, cfg)
        assert isinstance(opt, torch.optim.AdamW)

    def test_make_optimizer_invalid(self, small_model):
        cfg = Config(optimizer="invalid", hidden_width=16)
        with pytest.raises(ValueError, match="Unknown optimizer"):
            make_optimizer(small_model, cfg)

    def test_make_optimizer_respects_lr(self, small_model):
        cfg = Config(lr=123.0, hidden_width=16)
        opt = make_optimizer(small_model, cfg)
        assert opt.param_groups[0]["lr"] == 123.0

    def test_make_targets_onehot_shape(self):
        labels = torch.tensor([0, 3, 6])
        oh = make_targets_onehot(labels, 7)
        assert oh.shape == (3, 7)

    def test_make_targets_onehot_values(self):
        labels = torch.tensor([0, 2, 4])
        oh = make_targets_onehot(labels, 5)
        # Each row should be one-hot
        assert torch.equal(oh.sum(dim=1), torch.ones(3))
        assert oh[0, 0] == 1.0
        assert oh[1, 2] == 1.0
        assert oh[2, 4] == 1.0
        # All other entries should be 0
        assert oh[0, 1:].sum() == 0.0

    def test_make_targets_onehot_dtype(self):
        labels = torch.tensor([0, 1])
        oh = make_targets_onehot(labels, 3)
        assert oh.dtype == torch.float32
