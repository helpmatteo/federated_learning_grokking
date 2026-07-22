"""Registries mapping a config to the concrete pieces it selects.

`cfg.model` names an architecture family and `build_model(cfg)` returns a fresh
nn.Module; `cfg.loss` names a loss and `build_loss(cfg)` returns both the loss
function and the target-prep it needs (MSE trains against one-hot targets, CE
against class indices). These are the seams that let Phase 3 add the Nanda
transformer (CE), the Omnigrok MNIST MLP, and the S5 model without touching the
training loops — each is one new registry entry.

The dataset registry lands with the non-modular datasets in Phase 3.
"""

from dataclasses import dataclass

import torch.nn as nn

from fedgrok.models.groknet import GrokNet
from fedgrok.core.utils import make_targets_onehot


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


# ── Loss registry ────────────────────────────────────────────────────────────

@dataclass
class LossSpec:
    """A loss and the target it expects.

    loss_fn(logits, target) -> scalar; prepare_target(labels, p) -> target,
    where `labels` are int class indices on the compute device. Keeping the
    two together means a training loop never has to know whether it is feeding
    one-hot vectors (MSE) or class indices (CE).
    """
    loss_fn: object
    prepare_target: object


def build_loss(cfg):
    name = getattr(cfg, "loss", "mse")
    if name == "mse":
        # Gromov's setting: regress one-hot targets under mean-squared error.
        return LossSpec(
            loss_fn=nn.MSELoss(),
            prepare_target=lambda labels, p: make_targets_onehot(labels, p),
        )
    if name == "ce":
        # Power/Nanda setting: cross-entropy over class-index targets.
        return LossSpec(
            loss_fn=nn.CrossEntropyLoss(),
            prepare_target=lambda labels, p: labels.long(),
        )
    raise ValueError(f"Unknown loss {name!r} (expected 'mse' or 'ce')")
