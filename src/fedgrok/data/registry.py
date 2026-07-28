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
_DATASET_THRESHOLDS = {}

DEFAULT_GROK_THRESHOLD = 95.0


def register_dataset(name, dims_fn, grid_fn=None,
                     grok_threshold=DEFAULT_GROK_THRESHOLD):
    """Register a dataset.

    build_fn(cfg) -> (xtr, ytr, xte, yte); dims_fn(cfg) -> (in, out).
    grid_fn(cfg) -> (x_np, labels_np, operand_a_np) is optional: only the
    "grid" datasets (modular, S_n composition) expose the full unsplit grid and
    the first-operand index per sample, which the operand/coset FL partitions
    need. Datasets without it (MNIST) support only iid/dirichlet partitions.

    grok_threshold is the test accuracy (%) that counts as "generalised" on
    this dataset — see `grok_threshold()` below for why it cannot be one
    global constant.
    """
    def _decorator(build_fn):
        if name in _DATASET_BUILDERS:
            raise ValueError(f"Dataset {name!r} already registered")
        _DATASET_BUILDERS[name] = build_fn
        _DATASET_DIMS[name] = dims_fn
        _DATASET_THRESHOLDS[name] = grok_threshold
        if grid_fn is not None:
            _DATASET_GRIDS[name] = grid_fn
        return build_fn
    return _decorator


def grok_threshold(cfg) -> float:
    """Test accuracy (%) that counts as generalisation on cfg's dataset.

    A single global 95% is WRONG once the study spans datasets with different
    achievable ceilings: MNIST-1k tops out near 93%, so a 95% bar records every
    run as `t_grok = inf` — "never grokked" — even when the history plainly
    shows memorisation at epoch ~600 and generalisation thousands of epochs
    later. That is a measurement artifact reported as a scientific null.

    Per-dataset values and the reasoning behind them:

      modular  95.0  chance 1/p (~1%), ceiling 100%. Wide margin either side;
                     this is the historical value and is unchanged, so every
                     modular result ever recorded is unaffected.
      s5       85.0  chance 1/120 (~0.8%), ceiling ~92%. Test accuracy jumps
                     from ~1% essentially discontinuously, so anything in
                     10-90 returns the same step; 85 is picked for headroom
                     under the ceiling rather than for sensitivity.
      mnist    90.0  chance 10%, ceiling ~93%. Here the threshold genuinely
                     matters — test accuracy climbs gradually through the 80s,
                     so a bar set too low fires during memorisation and erases
                     the delay. 90 is the level four of the five weight-decay
                     bands reach and the weakest (lr*wd=1e-5, peak 89.2%) does
                     not, which is the honest reading of that cell.
    """
    return _DATASET_THRESHOLDS.get(getattr(cfg, "dataset", "modular"),
                                   DEFAULT_GROK_THRESHOLD)


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


@register_dataset("mnist", dims_fn=lambda cfg: (784, 10), grok_threshold=90.0)
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


@register_dataset("s5", dims_fn=_sn_dims, grid_fn=_sn_grid, grok_threshold=85.0)
def _build_s5(cfg):
    """Symmetric-group S_n composition (n = cfg.group_n; S5 by default)."""
    from fedgrok.data.groups import make_sn_dataset
    return make_sn_dataset(cfg)
