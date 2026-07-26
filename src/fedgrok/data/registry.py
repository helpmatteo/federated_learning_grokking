"""Dataset registry: cfg -> (x_train, y_train, x_test, y_test) and its dims.

`cfg.dataset` names a dataset family. `build_dataset(cfg)` returns the four
tensors; `dataset_dims(cfg)` returns (input_dim, output_dim) so the model
registry can size a generic model without loading the data. Modular arithmetic
is the default and is byte-identical to the pre-registry path (it just calls
`make_dataset`).
"""

from fedgrok.data.modular import make_dataset


_DATASET_BUILDERS = {}
_DATASET_DIMS = {}
_DATASET_GRIDS = {}


def register_dataset(name, dims_fn, grid_fn=None):
    """Register a dataset.

    build_fn(cfg) -> (xtr, ytr, xte, yte); dims_fn(cfg) -> (in, out).
    grid_fn(cfg) -> (x_np, labels_np, operand_a_np) is optional: only the
    "grid" datasets (modular, S_n composition) expose the full unsplit grid and
    the first-operand index per sample, which the operand/coset FL partitions
    need. Datasets without it (MNIST) support only iid/dirichlet partitions.
    """
    def _decorator(build_fn):
        if name in _DATASET_BUILDERS:
            raise ValueError(f"Dataset {name!r} already registered")
        _DATASET_BUILDERS[name] = build_fn
        _DATASET_DIMS[name] = dims_fn
        if grid_fn is not None:
            _DATASET_GRIDS[name] = grid_fn
        return build_fn
    return _decorator


def has_grid(cfg) -> bool:
    return getattr(cfg, "dataset", "modular") in _DATASET_GRIDS


def dataset_grid(cfg):
    """(x, labels, operand_a) for a grid dataset — the full unsplit grid.

    operand_a is the first-operand index per sample (group-element index for
    S_n, n-index for modular), used by the operand and coset FL partitions.
    """
    name = getattr(cfg, "dataset", "modular")
    if name not in _DATASET_GRIDS:
        raise ValueError(
            f"Dataset {name!r} has no grid; operand/coset partitions need one. "
            f"Grid datasets: {sorted(_DATASET_GRIDS)}"
        )
    return _DATASET_GRIDS[name](cfg)


def build_dataset(cfg):
    name = getattr(cfg, "dataset", "modular")
    if name not in _DATASET_BUILDERS:
        raise ValueError(f"Unknown dataset {name!r}. Registered: {sorted(_DATASET_BUILDERS)}")
    return _DATASET_BUILDERS[name](cfg)


def dataset_dims(cfg):
    """(input_dim, output_dim) for cfg's dataset — for sizing generic models."""
    name = getattr(cfg, "dataset", "modular")
    if name not in _DATASET_DIMS:
        raise ValueError(f"Unknown dataset {name!r}. Registered: {sorted(_DATASET_DIMS)}")
    return _DATASET_DIMS[name](cfg)


def registered_datasets():
    return sorted(_DATASET_BUILDERS)


# ── Built-in datasets ────────────────────────────────────────────────────────

def _modular_grid(cfg):
    from fedgrok.data.modular import build_encoded_grid
    x, labels, nn, _mm = build_encoded_grid(cfg.task, cfg.p)
    return x, labels, nn                       # operand_a = first operand n


@register_dataset("modular", dims_fn=lambda cfg: (2 * cfg.p, cfg.p),
                  grid_fn=_modular_grid)
def _build_modular(cfg):
    # Byte-identical to the previous direct make_dataset call.
    return make_dataset(cfg)


@register_dataset("mnist", dims_fn=lambda cfg: (784, 10))
def _build_mnist(cfg):
    # Imported lazily so torchvision is only required when MNIST is actually used.
    from fedgrok.data.mnist import load_mnist_subset
    return load_mnist_subset(cfg.n_train, cfg.n_test, cfg.seed)


def _sn_dims(cfg):
    from fedgrok.data.groups import group_order
    g = group_order(getattr(cfg, "group_n", 5))
    return (2 * g, g)


def _sn_grid(cfg):
    from fedgrok.data.groups import build_sn_grid
    x, labels, ia, _ib = build_sn_grid(getattr(cfg, "group_n", 5))
    return x, labels, ia                       # operand_a = first operand's element index


@register_dataset("s5", dims_fn=_sn_dims, grid_fn=_sn_grid)
def _build_s5(cfg):
    """Symmetric-group S_n composition (n = cfg.group_n; S5 by default)."""
    from fedgrok.data.groups import make_sn_dataset
    return make_sn_dataset(cfg)
