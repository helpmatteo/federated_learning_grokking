"""Tests for the S_n group-composition dataset and its registry wiring."""

import itertools

import pytest
import torch

from fedgrok.core.config import Config
from fedgrok.data.groups import (
    group_order, build_sn_grid, make_sn_dataset, _elements, _compose,
)
from fedgrok.data.registry import build_dataset, dataset_dims
from fedgrok.core.registry import build_model
from fedgrok.metrics.fourier import fourier_applicable


class TestGroupOrder:
    @pytest.mark.parametrize("n,order", [(3, 6), (4, 24), (5, 120)])
    def test_factorial(self, n, order):
        assert group_order(n) == order


class TestComposition:
    def test_identity_composes_to_identity(self):
        _, labels, _, _ = build_sn_grid(3)
        assert labels[0] == 0                      # id ∘ id = id (index 0)

    def test_composition_is_associative_via_table(self):
        """(a∘b)∘c == a∘(b∘c) for a sample of S4 triples."""
        elements = _elements(4)
        index = {p: i for i, p in enumerate(elements)}
        for a, b, c in itertools.islice(itertools.product(elements, repeat=3), 200):
            left = _compose(_compose(a, b), c)
            right = _compose(a, _compose(b, c))
            assert index[left] == index[right]

    def test_non_abelian(self):
        """S_n (n>=3) is non-abelian: some pair does not commute."""
        elements = _elements(3)
        assert any(_compose(a, b) != _compose(b, a)
                   for a in elements for b in elements)

    def test_grid_shape_and_labels_in_range(self):
        x, labels, ia, ib = build_sn_grid(4)
        G = group_order(4)
        assert x.shape == (G * G, 2 * G)
        assert labels.min() >= 0 and labels.max() < G
        # each row is two one-hots
        assert (x.sum(axis=1) == 2).all()


class TestSnDataset:
    def test_split_sizes(self):
        cfg = Config(dataset="s5", group_n=4, alpha=0.5, seed=42)
        x_train, y_train, x_test, y_test = make_sn_dataset(cfg)
        G = group_order(4)
        assert len(x_train) == int(0.5 * G * G)
        assert len(x_train) + len(x_test) == G * G

    def test_registry_dims_and_build(self):
        cfg = Config(dataset="s5", group_n=5)
        assert dataset_dims(cfg) == (240, 120)
        x_train, _, _, _ = build_dataset(cfg)
        assert x_train.shape[1] == 240


class TestS5ModelSizing:
    def test_groknet_sizes_from_group_order(self):
        cfg = Config(dataset="s5", group_n=5, model="groknet", hidden_width=64)
        model = build_model(cfg)
        assert model.W1.shape == (64, 240)         # (N, 2|G|)
        assert model.P == 120

    def test_fourier_metrics_skip_s5(self):
        """GrokNet on S5 has W1/P but the DFT is meaningless — must skip."""
        cfg = Config(dataset="s5", group_n=5, model="groknet", hidden_width=64)
        model = build_model(cfg)
        assert fourier_applicable(model, cfg) is False
        # ... but the same model on a modular cfg is fine
        assert fourier_applicable(model, Config(dataset="modular", p=120)) is True
