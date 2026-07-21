"""Shared training utilities: device selection, optimizer factory, one-hot encoding."""

import warnings

import torch
from fedgrok.core.config import Config


def get_device():
    """Select best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def check_decay_stability(lr: float, weight_decay: float):
    """Reject weight-decay settings that destroy the model.

    Both SGD (coupled L2) and AdamW (decoupled) apply a per-step multiplicative
    shrinkage of (1 - lr*wd) to the weights.  So `lr * wd` — not `wd` — is the
    meaningful quantity, and the decay timescale is 1 / (lr*wd) steps.

    Stability requires |1 - lr*wd| < 1, i.e. 0 < lr*wd < 2.  Published grokking
    configs all sit at lr*wd in [1e-4, 1e-3]:

        Nanda / Power / Varma   lr=1e-3, wd=1.0  -> lr*wd = 1e-3
        Omnigrok                lr=1e-4, wd=1.0  -> lr*wd = 1e-4

    This guard exists because exp5 originally ran lr=50 with wd in [0.01, 1.0],
    giving lr*wd in [0.5, 50] — a per-step shrinkage of 0.5 down to -49.  Every
    one of those runs was destroyed by the decay before it could learn anything,
    which is why they all reported T_grok=inf.
    """
    product = lr * weight_decay
    if product <= 0:
        return

    timescale = 1.0 / product
    hint = (f"At lr={lr}, the grokking band lr*wd in [1e-4, 1e-3] means "
            f"weight_decay in [{1e-4 / lr:.3g}, {1e-3 / lr:.3g}].")

    # Divergence: |1 - lr*wd| >= 1 with lr*wd >= 2 flips sign and grows.
    if product >= 2.0:
        raise ValueError(
            f"Divergent weight decay: lr*weight_decay = {product:.4g} "
            f"(lr={lr}, weight_decay={weight_decay}). Weights are multiplied by "
            f"(1 - lr*wd) = {1 - product:.4g} every step. {hint}"
        )

    # Not divergent, but still destructive: the weights are annihilated long
    # before the model can memorise, let alone generalise.  lr*wd = 0.5 does
    # not diverge, yet it halves every weight on every step.
    if product > 0.1:
        raise ValueError(
            f"Destructive weight decay: lr*weight_decay = {product:.4g} "
            f"(lr={lr}, weight_decay={weight_decay}) shrinks weights by a "
            f"factor {1 - product:.4g} per step — a decay timescale of "
            f"{timescale:.3g} steps, far shorter than the ~1e3-1e4 steps "
            f"needed to grok. {hint}"
        )

    if product > 1e-2:
        warnings.warn(
            f"Aggressive weight decay: lr*weight_decay = {product:.4g} gives a "
            f"decay timescale of {timescale:.0f} steps, which is likely to "
            f"suppress memorisation entirely. {hint}",
            RuntimeWarning,
            stacklevel=2,
        )


def make_optimizer(model, cfg: Config):
    if cfg.optimizer == "gd":
        check_decay_stability(cfg.lr, cfg.weight_decay)
        return torch.optim.SGD(
            model.parameters(),
            lr=cfg.lr,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
        )
    elif cfg.optimizer == "adamw":
        check_decay_stability(cfg.lr, cfg.weight_decay)
        return torch.optim.AdamW(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer: {cfg.optimizer}")


def make_targets_onehot(labels, p):
    """Convert class labels to one-hot for MSE loss.

    The one-hot is built on the labels' own device; building it on CPU while
    the labels live on GPU raises a device-mismatch error in scatter_.
    """
    onehot = torch.zeros(labels.size(0), p, device=labels.device)
    onehot.scatter_(1, labels.unsqueeze(1), 1.0)
    return onehot
