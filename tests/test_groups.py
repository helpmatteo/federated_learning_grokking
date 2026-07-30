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


# ── Coset structure, partition, and attribution metric ────────────────────────


class TestCosets:
    def test_s4_subgroup_gives_5_cosets_of_24(self):
        from fedgrok.data.groups import coset_labels
        import numpy as np
        labels = coset_labels(5, "s_nm1")
        assert labels.max() + 1 == 5
        assert sorted(np.bincount(labels)) == [24, 24, 24, 24, 24]

    def test_a5_subgroup_gives_2_cosets_of_60(self):
        from fedgrok.data.groups import coset_labels
        import numpy as np
        labels = coset_labels(5, "a_n")
        assert labels.max() + 1 == 2
        assert sorted(np.bincount(labels)) == [60, 60]

    def test_identity_lies_in_the_subgroup_coset(self):
        """Element 0 (the identity) is in H, so its coset is the subgroup itself."""
        from fedgrok.data.groups import coset_labels, _elements, _in_subgroup
        labels = coset_labels(5, "s_nm1")
        # every element whose coset label equals the identity's must be in H
        id_label = labels[0]
        elements = _elements(5)
        for i, lab in enumerate(labels):
            if lab == id_label:
                assert _in_subgroup(elements[i], "s_nm1", 5)

    def test_unknown_subgroup_raises(self):
        from fedgrok.data.groups import coset_labels
        with pytest.raises(ValueError, match="Unknown subgroup"):
            coset_labels(5, "z5")


class TestCosetPartition:
    def test_five_balanced_non_empty_clients(self):
        from fedgrok.core.fed_config import FedConfig
        from fedgrok.data.partition import make_federated_datasets
        cfg = FedConfig(dataset="s5", group_n=5, alpha=0.5, seed=42,
                        num_clients=5, partition="coset", coset_subgroup="s_nm1")
        client_data, _, y_train, _, _ = make_federated_datasets(cfg)
        sizes = [len(y) for _, y in client_data]
        assert len(sizes) == 5
        assert all(s > 0 for s in sizes)
        assert sum(sizes) == len(y_train)

    def test_wrong_client_count_rejected(self):
        from fedgrok.core.fed_config import FedConfig
        from fedgrok.data.partition import make_federated_datasets
        cfg = FedConfig(dataset="s5", group_n=5, num_clients=3, partition="coset")
        with pytest.raises(ValueError, match="cosets"):
            make_federated_datasets(cfg)

    @pytest.mark.parametrize("partition", ["operand", "coset"])
    def test_mnist_rejects_structure_dependent_partitions(self, partition):
        """MNIST has no first operand and no subgroup, so these cannot apply.

        The rest of the modes do — MNIST federated training used to be refused
        outright, which cost the study its only non-algebraic setup.
        """
        from fedgrok.core.fed_config import FedConfig
        from fedgrok.data.partition import make_federated_datasets
        cfg = FedConfig(dataset="mnist", model="mlp", n_train=500,
                        num_clients=5, partition=partition)
        with pytest.raises(ValueError, match="algebraic structure"):
            make_federated_datasets(cfg)

    @pytest.mark.parametrize("partition", ["iid", "dirichlet", "label_block", "target"])
    def test_mnist_federated_partitions_cover_the_training_set(self, partition):
        """Federated MNIST must shard exactly the centralized training set.

        MNIST's split comes from n_train/n_test under a torch.Generator, not from
        cfg.alpha, so the FL path must take build_dataset's tensors rather than
        re-splitting -- otherwise federated and centralized MNIST train on
        different examples under the same config.
        """
        import torch
        from fedgrok.core.fed_config import FedConfig
        from fedgrok.data.partition import make_federated_datasets
        from fedgrok.data.registry import build_dataset

        cfg = FedConfig(dataset="mnist", model="mlp", n_train=500, n_test=200,
                        seed=42, num_clients=5, partition=partition,
                        dirichlet_alpha=0.5)
        client_data, x_train_full, y_train_full, _, _ = make_federated_datasets(cfg)
        x_cent, y_cent, _, _ = build_dataset(cfg)

        assert sum(len(y) for _, y in client_data) == len(y_train_full)
        assert all(len(y) > 0 for _, y in client_data)
        assert torch.equal(torch.sort(y_train_full).values,
                           torch.sort(y_cent).values)
        assert len(x_train_full) == len(x_cent)


class TestCosetAttribution:
    def test_chance_level_untrained(self):
        import torch
        from fedgrok.core.config import Config
        from fedgrok.core.registry import build_model
        from fedgrok.data.registry import build_dataset
        from fedgrok.metrics.nonabelian import coset_attribution, nonabelian_applicable
        cfg = Config(dataset="s5", group_n=5, model="groknet", hidden_width=32,
                     coset_subgroup="s_nm1")
        assert nonabelian_applicable(cfg) and not nonabelian_applicable(Config())
        x, y, _, _ = build_dataset(cfg)
        r = coset_attribution(build_model(cfg), x[:2000], y[:2000], cfg)
        # 5 equal cosets -> coset accuracy near 1/5 at init
        assert 0.1 < r["coset_accuracy"] < 0.3

    def test_perfect_predictions_saturate(self):
        import torch
        import torch.nn.functional as F
        from fedgrok.core.config import Config
        from fedgrok.data.registry import build_dataset
        from fedgrok.metrics.nonabelian import coset_attribution
        cfg = Config(dataset="s5", group_n=5, coset_subgroup="s_nm1")
        x, y, _, _ = build_dataset(cfg)

        class Perfect(torch.nn.Module):
            P = 120
            def forward(self, xb):
                return F.one_hot(y[:xb.shape[0]], 120).float()

        r = coset_attribution(Perfect(), x[:1000], y[:1000], cfg)
        assert r["coset_accuracy"] == pytest.approx(1.0)
        assert r["coset_purity"] == pytest.approx(1.0)
