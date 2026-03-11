"""Data partitioning strategies for federated grokking.

Four partition modes, all operating on the training set:
  - iid:       random equal-sized shards
  - operand:   partition by first operand n (fragments Fourier input structure)
  - target:    partition by output class f(n,m) mod p (fragments output space)
  - dirichlet: Dirichlet(dirichlet_alpha) over target classes — continuously
               interpolates between IID (alpha→∞) and one class per client (alpha→0)

The test set stays global for server-side evaluation.
"""

import numpy as np
import torch
from federated.config import FedConfig
from core.dataset import TASKS


def make_federated_datasets(cfg: FedConfig):
    """Build partitioned training data and global test data.

    Returns:
        client_data: list of (x_train_i, y_train_i) tensors, one per client
        x_test, y_test: global test set tensors
    """
    p = cfg.p
    K = cfg.num_clients
    task_fn = TASKS[cfg.task]

    # Build full dataset
    ns = np.arange(p)
    ms = np.arange(p)
    nn, mm = np.meshgrid(ns, ms, indexing="ij")
    nn, mm = nn.flatten(), mm.flatten()
    labels = np.array([task_fn(int(n), int(m), p) for n, m in zip(nn, mm)])

    # One-hot encode
    x = np.zeros((p * p, 2 * p), dtype=np.float32)
    x[np.arange(p * p), nn] = 1.0
    x[np.arange(p * p), p + mm] = 1.0

    # Train/test split (same as centralized)
    rng = np.random.RandomState(cfg.seed)
    perm = rng.permutation(p * p)
    n_train = int(cfg.alpha * p * p)
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]

    x_train_all = x[train_idx]
    y_train_all = labels[train_idx]
    nn_train = nn[train_idx]  # first operand for each training sample

    # Global test set
    x_test = torch.from_numpy(x[test_idx])
    y_test = torch.from_numpy(labels[test_idx]).long()

    # Partition training data
    if cfg.partition == "iid":
        client_indices = _partition_iid(len(train_idx), K, rng)
    elif cfg.partition == "operand":
        client_indices = _partition_by_operand(nn_train, p, K)
    elif cfg.partition == "target":
        client_indices = _partition_by_target(y_train_all, p, K)
    elif cfg.partition == "dirichlet":
        client_indices = _partition_dirichlet(y_train_all, p, K, cfg.dirichlet_alpha, rng)
    else:
        raise ValueError(f"Unknown partition: {cfg.partition}")

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

    return [np.array(idx) for idx in client_indices]
