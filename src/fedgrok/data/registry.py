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


def register_dataset(name, dims_fn):
    """Register a `build(cfg) -> (xtr, ytr, xte, yte)` and a `dims(cfg) -> (in, out)`."""
    def _decorator(build_fn):
        if name in _DATASET_BUILDERS:
            raise ValueError(f"Dataset {name!r} already registered")
        _DATASET_BUILDERS[name] = build_fn
        _DATASET_DIMS[name] = dims_fn
        return build_fn
    return _decorator


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

@register_dataset("modular", dims_fn=lambda cfg: (2 * cfg.p, cfg.p))
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


@register_dataset("s5", dims_fn=_sn_dims)
def _build_s5(cfg):
    """Symmetric-group S_n composition (n = cfg.group_n; S5 by default)."""
    from fedgrok.data.groups import make_sn_dataset
    return make_sn_dataset(cfg)
