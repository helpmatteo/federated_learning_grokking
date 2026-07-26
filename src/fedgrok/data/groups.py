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


def _parity(perm) -> int:
    """Sign of a permutation as 0 (even) / 1 (odd), via inversion count."""
    inversions = 0
    n = len(perm)
    for i in range(n):
        for j in range(i + 1, n):
            if perm[i] > perm[j]:
                inversions += 1
    return inversions % 2


def _in_subgroup(perm, subgroup: str, n: int) -> bool:
    """Membership test for the supported subgroups of S_n."""
    if subgroup == "a_n":
        return _parity(perm) == 0                 # even permutations (A_n)
    if subgroup == "s_nm1":
        return perm[n - 1] == n - 1               # stabiliser of the last point (S_{n-1})
    raise ValueError(f"Unknown subgroup {subgroup!r} (expected 'a_n' or 's_nm1')")


def coset_labels(n: int, subgroup: str):
    """Left-coset index of every element of S_n w.r.t. `subgroup`.

    Returns an int array of length |S_n|: elements in the same left coset gH
    share a label, labels are 0..(index-1). For S5, "s_nm1" gives 5 cosets (of
    S4, 24 each) and "a_n" gives 2 cosets (of A5, 60 each). This is the algebraic
    structure the coset FL partition splits clients along.
    """
    elements = _elements(n)
    index = {perm: i for i, perm in enumerate(elements)}
    H = [i for i, perm in enumerate(elements) if _in_subgroup(perm, subgroup, n)]

    # Left coset gH keyed by its smallest member index; distinct keys -> labels.
    label_of = {}
    key_to_label = {}
    for gi, g in enumerate(elements):
        members = [index[_compose(g, elements[h])] for h in H]
        key = min(members)
        if key not in key_to_label:
            key_to_label[key] = len(key_to_label)
        label_of[gi] = key_to_label[key]

    return np.array([label_of[i] for i in range(len(elements))], dtype=np.int64)


def num_cosets(n: int, subgroup: str) -> int:
    """|S_n| / |H| — the number of cosets (= client count for the coset split)."""
    return int(coset_labels(n, subgroup).max()) + 1


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
