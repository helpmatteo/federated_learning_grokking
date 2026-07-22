"""Registries mapping a config to the concrete model it selects.

`cfg.model` names an architecture family; `build_model(cfg)` returns a fresh
nn.Module for it. This is the seam that lets Phase 3 add the Nanda transformer,
the Omnigrok MNIST MLP, and the S5 model without touching the training loops —
each is one new `@register_model` entry that knows how to read its dimensions
off the config.

Loss and dataset registries live alongside this once the CE/one-hot target
handling and the non-modular datasets land (Phase 2/3); kept out for now so the
model seam can go in cleanly on its own.
"""

from fedgrok.models.groknet import GrokNet


_MODEL_BUILDERS = {}


def register_model(name):
    """Decorator: register a `build(cfg) -> nn.Module` under `name`."""
    def _decorator(fn):
        if name in _MODEL_BUILDERS:
            raise ValueError(f"Model {name!r} is already registered")
        _MODEL_BUILDERS[name] = fn
        return fn
    return _decorator


def build_model(cfg):
    """Construct a fresh model (on CPU) for `cfg.model`."""
    name = getattr(cfg, "model", "groknet")
    if name not in _MODEL_BUILDERS:
        raise ValueError(
            f"Unknown model {name!r}. Registered: {sorted(_MODEL_BUILDERS)}"
        )
    return _MODEL_BUILDERS[name](cfg)


def registered_models():
    return sorted(_MODEL_BUILDERS)


# ── Built-in models ──────────────────────────────────────────────────────────

@register_model("groknet")
def _build_groknet(cfg):
    """Gromov's 2-layer MLP for modular arithmetic.

    Input/output dimensions are the modular-arithmetic convention: two
    concatenated one-hot operands (2p in) and one-hot class logits (p out).
    """
    return GrokNet(
        input_dim=2 * cfg.p,
        hidden_width=cfg.hidden_width,
        output_dim=cfg.p,
        activation=cfg.activation,
    )
