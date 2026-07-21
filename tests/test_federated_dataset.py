"""Tests for federated/dataset.py: all 4 partition strategies."""

import pytest
import numpy as np
import torch

from fedgrok.core.fed_config import FedConfig
from fedgrok.data.partition import (
    make_federated_datasets,
    _partition_iid,
    _partition_by_operand,
    _partition_by_target,
    _partition_dirichlet,
)
from fedgrok.data.modular import make_dataset

SMALL_P = 7


# ── Partition helper tests ────────────────────────────────────────────────────

class TestPartitionIID:
    def test_returns_correct_number_of_shards(self):
        rng = np.random.RandomState(42)
        shards = _partition_iid(100, 5, rng)
        assert len(shards) == 5

    def test_covers_all_indices(self):
        rng = np.random.RandomState(42)
        shards = _partition_iid(100, 5, rng)
        all_idx = np.sort(np.concatenate(shards))
        np.testing.assert_array_equal(all_idx, np.arange(100))

    def test_no_duplicate_indices(self):
        rng = np.random.RandomState(42)
        shards = _partition_iid(100, 5, rng)
        all_idx = np.concatenate(shards)
        assert len(all_idx) == len(set(all_idx))

    def test_roughly_equal_sizes(self):
        rng = np.random.RandomState(42)
        shards = _partition_iid(100, 5, rng)
        sizes = [len(s) for s in shards]
        assert max(sizes) - min(sizes) <= 1

    def test_different_seed_gives_different_partition(self):
        s1 = _partition_iid(100, 5, np.random.RandomState(1))
        s2 = _partition_iid(100, 5, np.random.RandomState(2))
        assert not np.array_equal(s1[0], s2[0])


class TestPartitionByOperand:
    def test_covers_all_indices(self):
        nn_train = np.array([0, 1, 2, 3, 4, 5, 6] * 3)  # 21 samples
        shards = _partition_by_operand(nn_train, 7, 3)
        all_idx = np.sort(np.concatenate(shards))
        np.testing.assert_array_equal(all_idx, np.arange(21))

    def test_correct_assignment(self):
        """Client i should get samples where n % K == i."""
        nn_train = np.arange(20)  # n values 0..19
        shards = _partition_by_operand(nn_train, 20, 4)
        for i, shard in enumerate(shards):
            for idx in shard:
                assert nn_train[idx] % 4 == i

    def test_no_overlap(self):
        nn_train = np.arange(49)
        shards = _partition_by_operand(nn_train, 49, 7)
        all_idx = np.concatenate(shards)
        assert len(all_idx) == len(set(all_idx))


class TestPartitionByTarget:
    def test_covers_all_indices(self):
        y_train = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
        shards = _partition_by_target(y_train, 5, 3)
        all_idx = np.sort(np.concatenate(shards))
        np.testing.assert_array_equal(all_idx, np.arange(10))

    def test_correct_assignment(self):
        """Client i should get samples where target % K == i."""
        y_train = np.array([0, 1, 2, 3, 4, 5, 6])
        shards = _partition_by_target(y_train, 7, 3)
        for i, shard in enumerate(shards):
            for idx in shard:
                assert y_train[idx] % 3 == i

    def test_no_overlap(self):
        y_train = np.arange(20) % 7
        shards = _partition_by_target(y_train, 7, 5)
        all_idx = np.concatenate(shards)
        assert len(all_idx) == len(set(all_idx))


class TestPartitionDirichlet:
    def test_covers_all_indices(self):
        rng = np.random.RandomState(42)
        y_train = np.arange(100) % 7
        shards = _partition_dirichlet(y_train, 7, 5, 0.5, rng)
        all_idx = np.sort(np.concatenate(shards))
        np.testing.assert_array_equal(all_idx, np.arange(100))

    def test_no_overlap(self):
        rng = np.random.RandomState(42)
        y_train = np.arange(100) % 7
        shards = _partition_dirichlet(y_train, 7, 5, 0.5, rng)
        all_idx = np.concatenate(shards)
        assert len(all_idx) == len(set(all_idx))

    def test_returns_correct_number_of_shards(self):
        rng = np.random.RandomState(42)
        y_train = np.arange(100) % 7
        shards = _partition_dirichlet(y_train, 7, 5, 0.5, rng)
        assert len(shards) == 5

    def test_high_alpha_is_roughly_uniform(self):
        """Large alpha should produce near-IID splits."""
        rng = np.random.RandomState(42)
        y_train = np.arange(500) % 7
        shards = _partition_dirichlet(y_train, 7, 5, 1000.0, rng)
        sizes = [len(s) for s in shards]
        # With very high alpha, sizes should be nearly equal
        assert max(sizes) - min(sizes) < 20

    def test_low_alpha_is_skewed(self):
        """Very low alpha should produce highly non-uniform splits."""
        rng = np.random.RandomState(42)
        y_train = np.arange(500) % 7
        shards = _partition_dirichlet(y_train, 7, 5, 0.01, rng)
        sizes = [len(s) for s in shards]
        # With very low alpha, there should be large size differences
        assert max(sizes) - min(sizes) > 20

    def test_seed_reproducibility(self):
        y_train = np.arange(100) % 7
        s1 = _partition_dirichlet(y_train, 7, 5, 0.5, np.random.RandomState(42))
        s2 = _partition_dirichlet(y_train, 7, 5, 0.5, np.random.RandomState(42))
        for a, b in zip(s1, s2):
            np.testing.assert_array_equal(a, b)


# ── Full pipeline: make_federated_datasets ────────────────────────────────────

class TestMakeFederatedDatasets:
    @pytest.mark.parametrize("partition", ["iid", "operand", "target", "dirichlet"])
    def test_returns_correct_structure(self, partition):
        cfg = FedConfig(p=SMALL_P, num_clients=3, partition=partition, hidden_width=16)
        result = make_federated_datasets(cfg)
        assert len(result) == 5  # (client_data, x_train_full, y_train_full, x_test, y_test)

        client_data, x_train_full, y_train_full, x_test, y_test = result
        assert len(client_data) == 3

    @pytest.mark.parametrize("partition", ["iid", "operand", "target", "dirichlet"])
    def test_client_data_covers_full_training_set(self, partition):
        cfg = FedConfig(p=SMALL_P, num_clients=3, partition=partition, hidden_width=16)
        client_data, x_train_full, y_train_full, _, _ = make_federated_datasets(cfg)

        total_client_samples = sum(len(y) for _, y in client_data)
        assert total_client_samples == len(y_train_full)

    @pytest.mark.parametrize("partition", ["iid", "operand", "target", "dirichlet"])
    def test_train_test_split_matches_centralized(self, partition):
        """Federated and centralized should use the same train/test split."""
        from fedgrok.core.config import Config
        fed_cfg = FedConfig(p=SMALL_P, num_clients=3, partition=partition,
                            hidden_width=16, seed=42)
        cen_cfg = Config(p=SMALL_P, hidden_width=16, seed=42)

        _, _, _, fed_x_test, fed_y_test = make_federated_datasets(fed_cfg)
        _, _, cen_x_test, cen_y_test = make_dataset(cen_cfg)

        assert torch.equal(fed_x_test, cen_x_test)
        assert torch.equal(fed_y_test, cen_y_test)

    @pytest.mark.parametrize("partition", ["iid", "operand", "target", "dirichlet"])
    def test_input_shapes(self, partition):
        cfg = FedConfig(p=SMALL_P, num_clients=3, partition=partition, hidden_width=16)
        client_data, x_train_full, y_train_full, x_test, y_test = make_federated_datasets(cfg)

        assert x_train_full.shape[1] == 2 * SMALL_P
        assert x_test.shape[1] == 2 * SMALL_P
        for xi, yi in client_data:
            assert xi.shape[1] == 2 * SMALL_P
            assert len(xi) == len(yi)

    @pytest.mark.parametrize("partition", ["iid", "operand", "target", "dirichlet"])
    def test_label_ranges(self, partition):
        cfg = FedConfig(p=SMALL_P, num_clients=3, partition=partition, hidden_width=16)
        client_data, _, y_train_full, _, y_test = make_federated_datasets(cfg)

        for y in [y_train_full, y_test]:
            assert y.min() >= 0
            assert y.max() < SMALL_P

        for _, yi in client_data:
            assert yi.min() >= 0
            assert yi.max() < SMALL_P

    def test_iid_seed_reproducibility(self):
        cfg = FedConfig(p=SMALL_P, num_clients=3, partition="iid", hidden_width=16, seed=42)
        r1 = make_federated_datasets(cfg)
        r2 = make_federated_datasets(cfg)
        for (x1, y1), (x2, y2) in zip(r1[0], r2[0]):
            assert torch.equal(x1, x2)
            assert torch.equal(y1, y2)

    def test_different_num_clients(self):
        """Varying K should produce different numbers of partitions."""
        for K in [2, 5, 7]:
            cfg = FedConfig(p=SMALL_P, num_clients=K, hidden_width=16)
            client_data, _, _, _, _ = make_federated_datasets(cfg)
            assert len(client_data) == K

    def test_invalid_partition_raises(self):
        cfg = FedConfig(p=SMALL_P, num_clients=3, hidden_width=16)
        cfg.partition = "invalid"
        with pytest.raises(ValueError, match="Unknown partition"):
            make_federated_datasets(cfg)


# ── Split is shared with the centralized path ────────────────────────────────


class TestSplitSharedWithCentralized:
    """The grid and train/test split have one source of truth.

    partition.py previously reimplemented both, so a drift in either copy would
    have silently trained federated and centralized runs on different data.
    """

    @pytest.mark.parametrize("task", ["addition", "division", "x2_plus_y2"])
    def test_centralized_and_federated_see_identical_split(self, task):
        from fedgrok.core.config import Config

        cent_cfg = Config(task=task, p=SMALL_P, alpha=0.5, seed=42)
        fed_cfg = FedConfig(task=task, p=SMALL_P, alpha=0.5, seed=42,
                            num_clients=3, partition="iid")

        x_train, y_train, x_test, y_test = make_dataset(cent_cfg)
        _, x_train_f, y_train_f, x_test_f, y_test_f = make_federated_datasets(fed_cfg)

        assert torch.equal(x_train, x_train_f)
        assert torch.equal(y_train, y_train_f)
        assert torch.equal(x_test, x_test_f)
        assert torch.equal(y_test, y_test_f)

    @pytest.mark.parametrize("partition", ["iid", "operand", "target", "dirichlet"])
    def test_client_shards_union_to_the_training_set(self, partition):
        cfg = FedConfig(task="addition", p=SMALL_P, alpha=0.5, seed=42,
                        num_clients=3, partition=partition, dirichlet_alpha=1.0)
        client_data, _, y_train_full, _, _ = make_federated_datasets(cfg)
        assert sum(len(y) for _, y in client_data) == len(y_train_full)
