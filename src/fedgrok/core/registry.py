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
from fedgrok.models.transformer import GrokFormer
from fedgrok.models.mlp import MLP
from fedgrok.data.registry import dataset_dims
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
    """Gromov's 2-layer MLP for token-composition tasks.

    Sizes from the dataset dims: two concatenated one-hot operands (input_dim)
    and one-hot class logits (output_dim). For modular arithmetic that is
    (2p, p); for S_n composition it is (2|G|, |G|).
    """
    input_dim, output_dim = dataset_dims(cfg)
    return GrokNet(
        input_dim=input_dim,
        hidden_width=cfg.hidden_width,
        output_dim=output_dim,
        activation=cfg.activation,
    )


@register_model("transformer")
def _build_transformer(cfg):
    """Nanda's 1-layer decoder-only transformer for token-composition tasks.

    Consumes the same concatenated one-hot input as GrokNet (see GrokFormer).
    The vocabulary / output size is the dataset's output_dim (p for modular,
    |G| for S_n), and d_model maps to cfg.hidden_width. Pair with loss="ce".
    """
    _, output_dim = dataset_dims(cfg)
    return GrokFormer(
        p=output_dim,
        d_model=cfg.hidden_width,
        n_heads=getattr(cfg, "n_heads", 4),
        d_mlp=getattr(cfg, "d_mlp", 512),
    )


@register_model("mlp")
def _build_mlp(cfg):
    """Generic ReLU MLP (Omnigrok MNIST). Sized from the dataset's dims, with
    the large-init scale from cfg.init_scale."""
    input_dim, output_dim = dataset_dims(cfg)
    return MLP(
        input_dim=input_dim,
        hidden_width=cfg.hidden_width,
        output_dim=output_dim,
        n_layers=cfg.n_layers,
        init_scale=cfg.init_scale,
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
