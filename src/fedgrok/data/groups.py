"""Symmetric-group composition datasets (S_n) for grokking.

The non-abelian analogue of modular addition: given two permutations a, b in
S_n, predict their composition a∘b. For S5 (n=5) there are 120 elements and
120×120 = 14,400 ordered pairs, ~94% of which do not commute — so unlike Z_p
there is no cyclic-Fourier structure, which is exactly the point (it tests
whether the FL findings depend on the abelian Fourier circuit).

Encoding mirrors the modular dataset: each operand is one-hot over the |G|
group elements, concatenated to a 2|G| input; the target is the index of the
composed element. Group elements are enumerated in a fixed canonical order
(itertools.permutations), so element index ↔ permutation is stable.
"""

import itertools

import numpy as np
import torch


def _elements(n: int):
    """All permutations of range(n), in canonical order. Index i ↔ elements[i]."""
    return list(itertools.permutations(range(n)))


def _compose(a, b):
    """(a∘b)(i) = a(b(i)). a, b are tuples mapping position -> image."""
    return tuple(a[b[i]] for i in range(len(a)))


def group_order(n: int) -> int:
    """|S_n| = n!."""
    order = 1
    for k in range(2, n + 1):
        order *= k
    return order


def build_sn_grid(n: int):
    """Full one-hot dataset for S_n composition.

    Returns:
        x       float32 (n_pairs, 2G) — one-hot(a) ‖ one-hot(b), G = n!
        labels  int     (n_pairs,)    — index of a∘b
        ia, ib  int     (n_pairs,)    — the operand indices (for partitioning)
    """
    elements = _elements(n)
    G = len(elements)
    index = {perm: i for i, perm in enumerate(elements)}

    ia, ib = np.meshgrid(np.arange(G), np.arange(G), indexing="ij")
    ia, ib = ia.flatten(), ib.flatten()

    labels = np.array(
        [index[_compose(elements[i], elements[j])] for i, j in zip(ia, ib)],
        dtype=np.int64,
    )

    n_pairs = len(ia)
    x = np.zeros((n_pairs, 2 * G), dtype=np.float32)
    x[np.arange(n_pairs), ia] = 1.0
    x[np.arange(n_pairs), G + ib] = 1.0
    return x, labels, ia, ib


def make_sn_dataset(cfg):
    """Build S_n composition and split into train/test (mirrors make_dataset).

    n is taken from cfg.group_n (default 5). The train fraction is cfg.alpha.
    """
    from fedgrok.data.modular import split_indices

    n = getattr(cfg, "group_n", 5)
    x, labels, _, _ = build_sn_grid(n)
    train_idx, test_idx = split_indices(len(labels), cfg.alpha, seed=cfg.seed)

    x_train = torch.from_numpy(x[train_idx])
    y_train = torch.from_numpy(labels[train_idx]).long()
    x_test = torch.from_numpy(x[test_idx])
    y_test = torch.from_numpy(labels[test_idx]).long()
    return x_train, y_train, x_test, y_test
