"""Data partitioning strategies for federated grokking.

Partition modes, all operating on the training set:
  - iid:       random equal-sized shards
  - operand:   partition by first operand (fragments the input structure)
  - target:    partition by output class (fragments the output space)
  - dirichlet: Dirichlet(dirichlet_alpha) over target classes — continuously
               interpolates between IID (alpha→∞) and one class per client (alpha→0)
  - label_block: contiguous blocks of the label-sorted training set — the
               structured partition that works on EVERY dataset, including the
               ones with no algebraic structure to partition on
  - coset:     (S_n only) partition by the left coset of the first operand w.r.t.
               a subgroup — "heterogeneity over algebraic structure", where each
               client's local elements are algebraically closed in a meaningful
               way and lack the structure the global task needs.

Which modes a dataset supports depends on whether it has a *grid* — the full
unsplit input space plus a first-operand index per sample. Grid datasets (modular,
S_n) support all six. Non-grid datasets (MNIST) have no first operand and no
subgroup, so `operand` and `coset` do not apply; everything else does.

The test set stays global for server-side evaluation.
"""

import numpy as np
import torch
from fedgrok.core.fed_config import FedConfig
from fedgrok.data.modular import split_indices
from fedgrok.data.registry import build_dataset, dataset_dims, dataset_grid, has_grid


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

    # Two ways to reach the training set, because the two dataset families define
    # their split differently.
    if has_grid(cfg):
        # Grid datasets (modular, S_n): re-derive the split from the full grid so
        # a federated run and a centralized run with the same config see identical
        # data. The partitioners below draw from this same RandomState *after* the
        # split has advanced it. Passing rng (not seed) preserves that exact
        # consumption order, so IID and Dirichlet assignments match previously
        # published runs.
        x, labels, operand_a = dataset_grid(cfg)
        rng = np.random.RandomState(cfg.seed)
        train_idx, test_idx = split_indices(len(labels), cfg.alpha, rng=rng)

        x_train_all = x[train_idx]
        y_train_all = labels[train_idx]
        operand_train = operand_a[train_idx]

        x_test = torch.from_numpy(x[test_idx])
        y_test = torch.from_numpy(labels[test_idx]).long()
    else:
        # Non-grid datasets (MNIST): there is no grid to split and cfg.alpha is
        # meaningless — the split is n_train/n_test under a torch.Generator inside
        # load_mnist_subset. So we take the tensors build_dataset already produced
        # rather than re-splitting, which is what keeps federated MNIST training on
        # exactly the same examples as centralized MNIST. (Consequence: this path's
        # RNG stream is NOT advanced by a split before the partitioners draw, unlike
        # the grid path. That is fine — the two families were never comparable
        # shard-for-shard — but it is why the partitioners must not assume it.)
        if cfg.partition in ("operand", "coset"):
            raise ValueError(
                f"partition={cfg.partition!r} needs the algebraic structure of a "
                f"grid dataset (a first operand / a subgroup), which "
                f"{cfg.dataset!r} does not have. Use 'iid', 'dirichlet', "
                f"'target', or 'label_block' — the last is the structured "
                f"partition that works on every dataset."
            )
        x_tr, y_tr, x_te, y_te = build_dataset(cfg)
        x_train_all = x_tr.numpy()
        y_train_all = y_tr.numpy()
        operand_train = None
        rng = np.random.RandomState(cfg.seed)
        train_idx = np.arange(len(y_train_all))

        x_test, y_test = x_te, y_te.long()

    # The number of output classes. NOT cfg.p: that is the modular modulus, and it
    # coincides with the class count only for the modular dataset. S_5 has 120
    # classes against a default cfg.p of 97, and MNIST has 10 -- so a class loop
    # bounded by cfg.p silently discards every sample whose label is >= cfg.p.
    # Measured before this was fixed: S5 dirichlet at K=10 assigned 5846 of 7200
    # samples and reached only 97 of 120 classes, capping train accuracy at 81.2%
    # against an 85.0 grok bar. Every such cell would have censored, and read as a
    # federated null rather than a partitioning bug.
    n_classes = dataset_dims(cfg)[1]

    # Partition training data
    if cfg.partition == "iid":
        client_indices = _partition_iid(len(train_idx), K, rng)
    elif cfg.partition == "operand":
        client_indices = _partition_by_operand(operand_train, K)
    elif cfg.partition == "target":
        client_indices = _partition_by_target(y_train_all, K)
    elif cfg.partition == "dirichlet":
        client_indices = _partition_dirichlet(y_train_all, n_classes, K,
                                              cfg.dirichlet_alpha, rng)
    elif cfg.partition == "label_block":
        client_indices = _partition_label_block(y_train_all, K)
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
        # Report the size knob that actually applies: alpha/p for a grid dataset,
        # n_train for MNIST (where alpha is ignored entirely).
        size_desc = (f"alpha={cfg.alpha}, p={p}" if has_grid(cfg)
                     else f"n_train={cfg.n_train}")
        hint = ("reduce num_clients, raise alpha, or raise dirichlet_alpha"
                if has_grid(cfg) else
                "reduce num_clients, raise n_train, or raise dirichlet_alpha")
        if cfg.partition == "target" and K > n_classes:
            hint = (f"'target' assigns by label % K and this dataset has only "
                    f"{n_classes} classes, so K={K} leaves the trailing clients "
                    f"empty by construction; use 'label_block' for a structured "
                    f"partition that works at any K")
        raise ValueError(
            f"Partition '{cfg.partition}' with K={K}, {size_desc} "
            f"left client(s) {empty} with no samples (shard sizes: {sizes}). "
            f"{len(train_idx)} training samples cannot support {K} clients at "
            f"this heterogeneity; {hint}."
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


def _partition_by_operand(nn_train, num_clients):
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


def _partition_by_target(y_train, num_clients):
    """Partition by output class: client i gets samples where target % K == i.

    This fragments the output space — each client sees a biased distribution
    of target values.

    NOTE: needs num_clients <= n_classes, or the trailing clients get nothing
    and the empty-shard guard fires. On MNIST (10 classes) that caps K at 10;
    use "label_block" for a structured partition that survives any K.
    """
    client_indices = []
    for i in range(num_clients):
        mask = (y_train % num_clients) == i
        client_indices.append(np.where(mask)[0])
    return client_indices


def _partition_label_block(y_train, num_clients):
    """Contiguous blocks of the label-sorted training set, one per client.

    The structured-heterogeneity partition that works on EVERY dataset. The
    "operand" partition needs an algebraic first operand and "coset" needs a
    subgroup, so neither exists for MNIST; without an alternative, the study's
    most interesting axis — coherent shards vs random ones — would simply be
    unavailable there. Sorting by label and splitting into K contiguous chunks
    gives each client a coherent slice of the output space at any K, never
    empty, with identical semantics across modular / S_n / MNIST.
    """
    order = np.argsort(y_train, kind="stable")
    return list(np.array_split(order, num_clients))


def _partition_dirichlet(y_train, n_classes, num_clients, dirichlet_alpha, rng):
    """Dirichlet-based non-IID partition over target classes.

    For each class c, sample a proportion vector q ~ Dir(dirichlet_alpha)
    of length K and allocate that fraction of class-c samples to each client.

    dirichlet_alpha → ∞ : approaches IID
    dirichlet_alpha → 0 : each client receives samples from ~1 class only
    Typical values: 0.1 (very non-IID), 0.5, 1.0, 10.0 (near-IID)

    `n_classes` must be the dataset's true class count (dataset_dims(cfg)[1]),
    not cfg.p — see the note at the call site.
    """
    client_indices = [[] for _ in range(num_clients)]

    for c in range(n_classes):
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
