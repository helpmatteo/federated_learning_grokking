"""Exact isotypic (irrep) decomposition of functions on S_n.

WHY THIS EXISTS. The modular study's mechanistic spine is the Fourier power
spectrum of W1's operand block: the generalising circuit is a handful of cyclic
frequencies, and watching their power rise IS the grokking story. On S_5 there
is no cyclic Fourier basis, so that instrument does not transfer, and the only
S_5 measure in the repo is `coset_attribution` -- a coarse proxy that reads the
model's OUTPUT, not its weights.

Representation theory supplies the exact replacement. Any function f: G -> R
decomposes uniquely into isotypic components, one per irreducible representation
of G, and the decomposition is orthogonal, so the energies partition ||f||^2
exactly (Parseval). For S_5 there are 7 irreps, indexed by partitions of 5:

    partition      dim   name
    [5]              1   trivial
    [4,1]            4   standard
    [3,2]            5
    [3,1,1]          6
    [2,2,1]          5
    [2,1,1,1]        4   standard (x) sign
    [1,1,1,1,1]      1   sign
                        sum d^2 = 1+16+25+36+25+16+1 = 120 = |S_5|

A hidden unit's operand weights U[k, :] are exactly such a function -- one real
value per group element -- so the per-irrep energies of U are the direct S_5
analogue of the Fourier power spectrum, and they are exact rather than a proxy.

HOW, WITHOUT BUILDING IRREP MATRICES. The usual route needs Young's orthogonal
representation, which is a few hundred fiddly lines. It is unnecessary: the
projection onto the rho-isotypic component of the regular representation is
convolution with the character,

    (P_rho f)(x) = (d_rho / |G|) * sum_y chi_rho(x y^-1) f(y),

so all that is required is the CHARACTER TABLE -- a 7x7 integer table -- plus the
group's multiplication and inverses. Each projector is a |G| x |G| real matrix
built once and cached; applying all 7 to a (N, |G|) weight block costs
7 * N * |G|^2, which is 26 MFLOP for setup D. Negligible per eval.

The characters of S_n are real, so the projectors are real symmetric and the
components are genuinely orthogonal -- `energies` therefore sums to ||f||^2, and
`test_irreps.py` asserts exactly that.

READING THE OUTPUT. Under a random weight matrix the expected energy fraction in
irrep rho is d_rho^2 / |G| -- the isotypic component's share of the regular
representation. `random_baseline_fractions` returns those numbers, and they are
the line to compare against: structure is energy ABOVE the random share, not
energy per se. For S_5 the baseline is

    [5] 0.008 | [4,1] 0.133 | [3,2] 0.208 | [3,1,1] 0.300
    | [2,2,1] 0.208 | [2,1,1,1] 0.133 | [1^5] 0.008
"""

import functools

import numpy as np
import torch

from fedgrok.data.groups import _compose, _elements

# Conjugacy classes of S_5 keyed by cycle type, in the column order of the
# character table below. Sizes 1, 10, 15, 20, 20, 30, 24 sum to 120.
S5_CLASSES = ((1, 1, 1, 1, 1), (2, 1, 1, 1), (2, 2, 1), (3, 1, 1),
              (3, 2), (4, 1), (5,))

# Character table of S_5. Rows are irreps (partitions of 5), columns are the
# conjugacy classes above. Standard and verifiable: rows are orthonormal under
# the class-size-weighted inner product, which `test_irreps.py` checks rather
# than taking on trust.
S5_CHARACTERS = {
    (5,):             (1,  1,  1,  1,  1,  1,  1),
    (4, 1):           (4,  2,  0,  1, -1,  0, -1),
    (3, 2):           (5,  1,  1, -1,  1, -1,  0),
    (3, 1, 1):        (6,  0, -2,  0,  0,  0,  1),
    (2, 2, 1):        (5, -1,  1, -1, -1,  1,  0),
    (2, 1, 1, 1):     (4, -2,  0,  1,  1,  0, -1),
    (1, 1, 1, 1, 1):  (1, -1,  1,  1, -1, -1,  1),
}

IRREP_NAMES = tuple("".join(str(part) for part in p) for p in S5_CHARACTERS)


def irrep_dims(n: int = 5):
    """Dimension of each irrep, in table order. dim = chi(identity)."""
    _require_s5(n)
    return tuple(chi[0] for chi in S5_CHARACTERS.values())


def applicable(n: int) -> bool:
    """True iff an irrep decomposition exists for this group.

    Callers that run over several group sizes must gate on this rather than
    assume S_5. The character table here is S_5-specific, and S_4 is a real
    configuration -- `make_sn_dataset` takes any `group_n`, and the test suite
    uses S_4 for speed -- so an ungated call turns every S_4 run into a crash.
    """
    return n == 5


def _require_s5(n):
    if not applicable(n):
        raise NotImplementedError(
            f"Irrep decomposition is implemented for S_5 only (got n={n}). "
            "The character table above is S_5-specific; another n needs its own."
        )


def _cycle_type(perm):
    """Cycle type of a permutation as a descending tuple, e.g. (3, 2)."""
    seen = [False] * len(perm)
    lengths = []
    for start in range(len(perm)):
        if seen[start]:
            continue
        length, cur = 0, start
        while not seen[cur]:
            seen[cur] = True
            cur = perm[cur]
            length += 1
        lengths.append(length)
    return tuple(sorted(lengths, reverse=True))


def _inverse(perm):
    inv = [0] * len(perm)
    for i, image in enumerate(perm):
        inv[image] = i
    return tuple(inv)


@functools.lru_cache(maxsize=4)
def isotypic_projectors(n: int = 5):
    """(names, dims, P) where P is (n_irreps, |G|, |G|) of projection matrices.

    P[r] @ f projects the function f (indexed by group element) onto the r-th
    isotypic component. Built once per n and cached -- the S_5 tensor is
    7 x 120 x 120 float64, about 800 KB.
    """
    _require_s5(n)
    elements = _elements(n)
    order = len(elements)
    index = {perm: i for i, perm in enumerate(elements)}

    # class_of[x, y] = column of the character table for x * y^-1.
    inverses = [_inverse(perm) for perm in elements]
    class_col = {ctype: col for col, ctype in enumerate(S5_CLASSES)}
    prod_class = np.empty((order, order), dtype=np.int64)
    for xi, x in enumerate(elements):
        for yi in range(order):
            prod = _compose(x, inverses[yi])
            prod_class[xi, yi] = class_col[_cycle_type(prod)]

    names, dims, mats = [], [], []
    for partition, chi in S5_CHARACTERS.items():
        chi_arr = np.asarray(chi, dtype=np.float64)
        dim = chi[0]
        mats.append((dim / order) * chi_arr[prod_class])
        dims.append(dim)
        names.append("".join(str(part) for part in partition))
    return tuple(names), tuple(dims), np.stack(mats)


def random_baseline_fractions(n: int = 5):
    """Expected energy fraction per irrep for an iid-random function: d^2/|G|.

    The comparison line for every measurement this module produces. A learned
    weight block only carries structure insofar as it departs from these.
    """
    _require_s5(n)
    order = 1
    for k in range(2, n + 1):
        order *= k
    return tuple(d * d / order for d in irrep_dims(n))


@functools.lru_cache(maxsize=8)
def _projector_tensor(n, device, dtype):
    """Projectors as a torch tensor on `device`, cached per (n, device, dtype).

    The projectors are built once in float64 (they are exact rational multiples
    of small integers) and converted here. Keeping them on the model's device
    matters: this runs inside the per-eval probe, and shipping a (256, 120)
    weight block to CPU 1,600 times a run costs more than the projection does.
    """
    _, _, proj = isotypic_projectors(n)
    return torch.as_tensor(proj, dtype=dtype, device=device)


def energies(weights, n: int = 5):
    """Per-irrep energy of a weight block whose columns are indexed by G.

    weights: (..., |G|) tensor -- e.g. GrokNet's U = W1[:, :|G|], one row per
             hidden unit, one column per group element.

    Runs on the input's own device and dtype (promoted to at least float32).
    Returns a dict {irrep_name: energy}. By Parseval the values sum to
    ||weights||_F^2 -- exactly in float64, to ~1e-6 relative in float32, which
    `test_irreps.py` pins at both precisions.
    """
    names, _, proj = isotypic_projectors(n)
    w = torch.as_tensor(weights)
    if not torch.is_floating_point(w):
        w = w.to(torch.float64)
    elif w.dtype not in (torch.float32, torch.float64):
        w = w.to(torch.float32)
    if w.shape[-1] != proj.shape[-1]:
        raise ValueError(
            f"Last dim must index the group: got {w.shape[-1]}, "
            f"expected {proj.shape[-1]} for S_{n}."
        )
    p = _projector_tensor(n, w.device, w.dtype)
    # (n_irreps, ..., |G|) — one projected copy of the block per irrep.
    projected = torch.einsum("rgh,...h->r...g", p, w)
    flat = (projected ** 2).flatten(1).sum(dim=1)
    return {name: float(flat[r]) for r, name in enumerate(names)}


def fractions(weights, n: int = 5):
    """`energies` normalised to sum to 1. NaN-free: returns zeros for a zero block."""
    raw = energies(weights, n)
    total = sum(raw.values())
    if total <= 0:
        return {k: 0.0 for k in raw}
    return {k: v / total for k, v in raw.items()}


def structure_score(weights, n: int = 5):
    """Total-variation distance between the energy profile and iid-random.

    One scalar for "how far from random is this weight block", in [0, 1]. Zero
    at initialisation by construction; it rises as any irrep's share departs
    from d^2/|G|. Useful as a single per-round series when 7 are too many.
    """
    frac = fractions(weights, n)
    base = dict(zip(frac.keys(), random_baseline_fractions(n)))
    return 0.5 * sum(abs(frac[k] - base[k]) for k in frac)
