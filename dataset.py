"""Modular arithmetic dataset over Z_p.

Each sample is a pair (n, m) encoded as two concatenated one-hot vectors (size 2p).
The target is f(n, m) mod p, encoded as a one-hot vector (size p).
"""

import torch
import numpy as np
from config import Config


# ── Task functions ──────────────────────────────────────────────────────────

TASKS = {
    "addition":        lambda n, m, p: (n + m) % p,
    "subtraction":     lambda n, m, p: (n - m) % p,
    "division":        lambda n, m, p: (n * pow(int(m), int(p - 2), int(p))) % p if m != 0 else 0,
    "x2_plus_y2":      lambda n, m, p: (n**2 + m**2) % p,
    "x_plus_y_squared": lambda n, m, p: ((n + m)**2) % p,
    "multiplication":  lambda n, m, p: (n * m) % p,
    "x2_y2_xy":        lambda n, m, p: (n**2 + m**2 + n * m) % p,
    "x3_xy2_y":        lambda n, m, p: (n**3 + n * m**2 + m) % p,
}


def make_dataset(cfg: Config):
    """Build full dataset and split into train/test.

    Returns:
        x_train, y_train, x_test, y_test  (all torch tensors on CPU)
        x are float tensors of shape (num_samples, 2p)  — concatenated one-hots
        y are long  tensors of shape (num_samples,)      — class labels (0..p-1)
    """
    p = cfg.p
    task_fn = TASKS[cfg.task]

    # all p^2 pairs
    ns = np.arange(p)
    ms = np.arange(p)
    nn, mm = np.meshgrid(ns, ms, indexing="ij")
    nn, mm = nn.flatten(), mm.flatten()
    labels = np.array([task_fn(int(n), int(m), p) for n, m in zip(nn, mm)])

    # one-hot encode inputs: concatenate one_hot(n) and one_hot(m)
    x = np.zeros((p * p, 2 * p), dtype=np.float32)
    x[np.arange(p * p), nn] = 1.0
    x[np.arange(p * p), p + mm] = 1.0

    # random train/test split
    rng = np.random.RandomState(cfg.seed)
    perm = rng.permutation(p * p)
    n_train = int(cfg.alpha * p * p)

    train_idx = perm[:n_train]
    test_idx = perm[n_train:]

    x_train = torch.from_numpy(x[train_idx])
    y_train = torch.from_numpy(labels[train_idx]).long()
    x_test = torch.from_numpy(x[test_idx])
    y_test = torch.from_numpy(labels[test_idx]).long()

    return x_train, y_train, x_test, y_test
