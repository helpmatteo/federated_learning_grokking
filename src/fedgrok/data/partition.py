"""Data partitioning strategies for federated grokking.

Partition modes, all operating on the training set:
  - iid:       random equal-sized shards
  - operand:   partition by first operand (fragments the input structure)
  - target:    partition by output class (fragments the output space)
  - dirichlet: Dirichlet(dirichlet_alpha) over target classes — continuously
               interpolates between IID (alpha→∞) and one class per client (alpha→0)
  - coset:     (S_n only) partition by the left coset of the first operand w.r.t.
               a subgroup — "heterogeneity over algebraic structure", where each
               client's local elements are algebraically closed in a meaningful
               way and lack the structure the global task needs.

The test set stays global for server-side evaluation.
"""

import numpy as np
import torch
from fedgrok.core.fed_config import FedConfig
from fedgrok.data.modular import split_indices
from fedgrok.data.registry import dataset_grid, has_grid


def make_federated_datasets(cfg: FedConfig):
    """Build partitioned training data and global test data.

    Returns:
        client_data: list of (x_train_i, y_train_i) tensors, one per client
        x_test, y_test: global test set tensors

    The grid and the split come from the dataset registry, so a federated run
    and a centralized run with the same config train on identical data. Grid
    datasets (modular, S_n) support every partition; non-grid datasets (MNIST)
    have no operand structure, so operand/coset partitions do not apply.
    """
    p = cfg.p
    K = cfg.num_clients

    if not has_grid(cfg):
        raise NotImplementedError(
            f"Federated training for dataset {cfg.dataset!r} is not wired yet "
            f"(no grid; only centralized). Grid datasets support FL."
        )

    # operand_a: first-operand index per sample (n for modular, element index for S_n).
    x, labels, operand_a = dataset_grid(cfg)

    # The partitioners below draw from this same RandomState *after* the split
    # has advanced it. Passing rng (not seed) preserves that exact consumption
    # order, so IID and Dirichlet assignments match previously published runs.
    rng = np.random.RandomState(cfg.seed)
    train_idx, test_idx = split_indices(len(labels), cfg.alpha, rng=rng)

    x_train_all = x[train_idx]
    y_train_all = labels[train_idx]
    operand_train = operand_a[train_idx]  # first operand for each training sample

    # Global test set
    x_test = torch.from_numpy(x[test_idx])
    y_test = torch.from_numpy(labels[test_idx]).long()

    # Partition training data
    if cfg.partition == "iid":
        client_indices = _partition_iid(len(train_idx), K, rng)
    elif cfg.partition == "operand":
        client_indices = _partition_by_operand(operand_train, p, K)
    elif cfg.partition == "target":
        client_indices = _partition_by_target(y_train_all, p, K)
    elif cfg.partition == "dirichlet":
        client_indices = _partition_dirichlet(y_train_all, p, K, cfg.dirichlet_alpha, rng)
    elif cfg.partition == "coset":
        client_indices = _partition_by_coset(operand_train, cfg, K)
    else:
        raise ValueError(f"Unknown partition: {cfg.partition}")

    # An empty shard is not survivable downstream: the client would take a
    # full-batch step over zero samples, MSELoss would return nan, and the nan
    # would propagate into the aggregate even though the shard carries weight 0
    # (numpy gives 0 * nan = nan). Fail loudly here instead.
    empty = [k for k, idx in enumerate(client_indices) if len(idx) == 0]
    if empty:
        sizes = [len(idx) for idx in client_indices]
        raise ValueError(
            f"Partition '{cfg.partition}' with K={K}, alpha={cfg.alpha}, p={p} "
            f"left client(s) {empty} with no samples (shard sizes: {sizes}). "
            f"{len(train_idx)} training samples cannot support {K} clients at "
            f"this heterogeneity; reduce num_clients, raise alpha, or raise "
            f"dirichlet_alpha."
        )

    client_data = []
    for indices in client_indices:
        xi = torch.from_numpy(x_train_all[indices])
        yi = torch.from_numpy(y_train_all[indices]).long()
        client_data.append((xi, yi))

    # Also return full training set for global evaluation
    x_train_full = torch.from_numpy(x_train_all)
    y_train_full = torch.from_numpy(y_train_all).long()

    return client_data, x_train_full, y_train_full, x_test, y_test


def _partition_iid(n_samples, num_clients, rng):
    """Random equal-sized shards."""
    indices = rng.permutation(n_samples)
    return np.array_split(indices, num_clients)


def _partition_by_operand(nn_train, p, num_clients):
    """Partition by first operand n: client i gets samples where n % K == i.

    This fragments the Fourier input structure across clients — each client
    only sees a subset of n-values, making it harder to learn the full
    periodic features.
    """
    client_indices = []
    for i in range(num_clients):
        mask = (nn_train % num_clients) == i
        client_indices.append(np.where(mask)[0])
    return client_indices


def _partition_by_coset(operand_train, cfg, num_clients):
    """Partition by the left coset of the first operand (S_n only).

    Client c gets every training sample whose first operand lies in coset c of
    the chosen subgroup. This is the algebraic-structure heterogeneity split:
    for S5 with the S4 subgroup there are 5 cosets of 24 elements, so 5 clients
    each see a coset's worth of first operands — a genuinely different local
    slice than a random or label-based split.

    Requires num_clients == number of cosets; otherwise some clients would be
    empty or some cosets dropped.
    """
    from fedgrok.data.groups import coset_labels

    labels = coset_labels(getattr(cfg, "group_n", 5), cfg.coset_subgroup)
    n_cosets = int(labels.max()) + 1
    if num_clients != n_cosets:
        raise ValueError(
            f"coset partition of subgroup {cfg.coset_subgroup!r} has {n_cosets} "
            f"cosets, but num_clients={num_clients}. Set num_clients={n_cosets}."
        )
    sample_coset = labels[operand_train]
    return [np.where(sample_coset == c)[0] for c in range(num_clients)]


def _partition_by_target(y_train, p, num_clients):
    """Partition by output class: client i gets samples where target % K == i.

    This fragments the output space — each client sees a biased distribution
    of target values.
    """
    client_indices = []
    for i in range(num_clients):
        mask = (y_train % num_clients) == i
        client_indices.append(np.where(mask)[0])
    return client_indices


def _partition_dirichlet(y_train, p, num_clients, dirichlet_alpha, rng):
    """Dirichlet-based non-IID partition over target classes.

    For each class c, sample a proportion vector q ~ Dir(dirichlet_alpha)
    of length K and allocate that fraction of class-c samples to each client.

    dirichlet_alpha → ∞ : approaches IID
    dirichlet_alpha → 0 : each client receives samples from ~1 class only
    Typical values: 0.1 (very non-IID), 0.5, 1.0, 10.0 (near-IID)
    """
    client_indices = [[] for _ in range(num_clients)]

    for c in range(p):
        class_idx = np.where(y_train == c)[0]
        if len(class_idx) == 0:
            continue
        rng.shuffle(class_idx)
        proportions = rng.dirichlet(np.full(num_clients, dirichlet_alpha))
        # Convert proportions to cumulative split points
        splits = (np.cumsum(proportions) * len(class_idx)).astype(int)[:-1]
        for k, chunk in enumerate(np.split(class_idx, splits)):
            client_indices[k].extend(chunk.tolist())

    # dtype=int matters: np.array([]) is float64 and cannot be used as an index,
    # so an empty shard would raise IndexError rather than produce an empty one.
    return [np.array(idx, dtype=int) for idx in client_indices]
