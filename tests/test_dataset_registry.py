"""Tests for the dataset registry and the generic MLP (Omnigrok MNIST)."""

import pytest
import torch

from fedgrok.core.config import Config
from fedgrok.data.registry import build_dataset, dataset_dims, registered_datasets
from fedgrok.data.modular import make_dataset
from fedgrok.models.mlp import MLP
from fedgrok.core.registry import build_model
from fedgrok.metrics.fourier import fourier_applicable


class TestDatasetRegistry:
    def test_modular_and_mnist_registered(self):
        assert "modular" in registered_datasets()
        assert "mnist" in registered_datasets()

    def test_modular_dims_are_2p_and_p(self):
        assert dataset_dims(Config(dataset="modular", p=97)) == (194, 97)

    def test_mnist_dims_are_784_and_10(self):
        assert dataset_dims(Config(dataset="mnist")) == (784, 10)

    def test_modular_build_matches_make_dataset(self):
        """The modular path through the registry must be byte-identical."""
        cfg = Config(dataset="modular", task="addition", p=17, alpha=0.5, seed=42)
        a = build_dataset(cfg)
        b = make_dataset(cfg)
        for ta, tb in zip(a, b):
            assert torch.equal(ta, tb)

    def test_unknown_dataset_raises(self):
        with pytest.raises(ValueError, match="Unknown dataset"):
            build_dataset(Config(dataset="cifar"))


class TestMLP:
    def test_shape_and_depth(self):
        model = MLP(input_dim=784, hidden_width=200, output_dim=10, n_layers=3)
        assert len(model.layers) == 3
        assert model(torch.randn(8, 784)).shape == (8, 10)

    def test_P_is_output_dim(self):
        assert MLP(784, 200, 10).P == 10

    def test_not_fourier_applicable(self):
        assert fourier_applicable(MLP(784, 200, 10)) is False

    def test_init_scale_multiplies_norm(self):
        """init_scale > 1 must scale the initial weight norm by that factor."""
        torch.manual_seed(0)
        base = MLP(64, 32, 10, n_layers=2, init_scale=1.0)
        torch.manual_seed(0)
        scaled = MLP(64, 32, 10, n_layers=2, init_scale=5.0)
        base_norm = sum(p.norm().item() ** 2 for p in base.parameters()) ** 0.5
        scaled_norm = sum(p.norm().item() ** 2 for p in scaled.parameters()) ** 0.5
        assert scaled_norm == pytest.approx(5.0 * base_norm, rel=1e-4)

    def test_rejects_zero_layers(self):
        with pytest.raises(ValueError, match="n_layers"):
            MLP(10, 10, 10, n_layers=0)


class TestMLPRegistryWiring:
    def test_mlp_registered_and_sized_from_dataset(self):
        cfg = Config(dataset="mnist", model="mlp", hidden_width=200, n_layers=3)
        model = build_model(cfg)
        assert isinstance(model, MLP)
        # first layer takes 784-dim MNIST input, last emits 10 classes
        assert model.layers[0].in_features == 784
        assert model.layers[-1].out_features == 10
