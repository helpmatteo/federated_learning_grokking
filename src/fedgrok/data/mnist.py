"""MNIST for the Omnigrok grokking setup — a tiny subset, MSE on one-hot.

Omnigrok (Liu et al., 2210.01117) induces grokking on MNIST with three load-
bearing choices: a small subset (~1k of 60k), MSE regression on one-hot targets
(not cross-entropy), and a large initialisation scale. This module provides the
data; the large-init MLP and the MSE loss come from the model/loss registries.

train and test are disjoint random draws from MNIST's *training* split (the
Omnigrok convention for the small-data regime — the point is a tiny labelled
set, not the canonical 10k test set).
"""

import os

import torch
from torchvision import datasets

_CACHE = os.path.expanduser("~/.cache/fedgrok/mnist")

# Fixed dimensions, exposed so the model registry can size a generic MLP without
# loading the data first.
MNIST_INPUT_DIM = 784
MNIST_NUM_CLASSES = 10

_raw_cache = {}


def _raw_mnist():
    """Load (and cache in-process) MNIST's training images/labels as tensors."""
    if "data" not in _raw_cache:
        ds = datasets.MNIST(_CACHE, train=True, download=True)
        # images: (60000, 28, 28) uint8 -> (60000, 784) float in [0, 1]
        images = ds.data.reshape(len(ds), -1).float() / 255.0
        labels = ds.targets.long()
        _raw_cache["data"] = (images, labels)
    return _raw_cache["data"]


def load_mnist_subset(n_train: int, n_test: int, seed: int):
    """Return (x_train, y_train, x_test, y_test) for an Omnigrok-style subset.

    Images are flattened to 784 in [0, 1]; labels are int class indices 0..9.
    train and test are disjoint random draws from MNIST's train split.
    """
    images, labels = _raw_mnist()
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(images), generator=g)
    train_idx = perm[:n_train]
    test_idx = perm[n_train:n_train + n_test]
    return (images[train_idx], labels[train_idx],
            images[test_idx], labels[test_idx])
