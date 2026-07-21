"""Shared training utilities: device selection, optimizer factory, one-hot encoding."""

import torch
from fedgrok.core.config import Config


def get_device():
    """Select best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_optimizer(model, cfg: Config):
    if cfg.optimizer == "gd":
        return torch.optim.SGD(
            model.parameters(),
            lr=cfg.lr,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
        )
    elif cfg.optimizer == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer: {cfg.optimizer}")


def make_targets_onehot(labels, p):
    """Convert class labels to one-hot for MSE loss."""
    onehot = torch.zeros(labels.size(0), p)
    onehot.scatter_(1, labels.unsqueeze(1), 1.0)
    return onehot
