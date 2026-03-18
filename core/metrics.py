"""Metrics from Gromov (2023): weight/gradient norms, IPR, phase analysis."""

import torch
import numpy as np


def weight_norms(model):
    """Frobenius norm of W1 and W2."""
    return {
        "weight_norm_layer1": model.W1.data.norm().item(),
        "weight_norm_layer2": model.W2.data.norm().item(),
    }


def gradient_norms(model):
    """Frobenius norm of gradients of W1 and W2 (call after loss.backward())."""
    norms = {}
    if model.W1.grad is not None:
        norms["grad_norm_layer1"] = model.W1.grad.norm().item()
    if model.W2.grad is not None:
        norms["grad_norm_layer2"] = model.W2.grad.norm().item()
    return norms


def compute_ipr(model, r=2):
    """Inverse Participation Ratio (Eq 20-21).

    Measures periodicity of W1 weights in Fourier space.
    Higher IPR → more periodic (structured) weights.

    We Fourier-transform each row of W1 w.r.t. the first p columns (the n-index),
    normalize, then compute IPR_r(k) = sum_nu |w_tilde_nu_k|^(2r).
    Average over all N neurons.
    """
    W1 = model.W1.data  # (N, 2p)
    p = model.P
    N = model.N

    # Take the first p columns (corresponding to the n input)
    W1_n = W1[:, :p]  # (N, p)

    # DFT along the p dimension (Fourier transform w.r.t. input index)
    W1_fft = torch.fft.fft(W1_n, dim=1)  # (N, p) complex

    # Normalize per neuron (Eq 20)
    magnitudes = W1_fft.abs()  # (N, p)
    norms = magnitudes.norm(dim=1, keepdim=True)  # (N, 1)
    norms = norms.clamp(min=1e-10)
    w_tilde = magnitudes / norms  # (N, p) — normalized

    # IPR_r(k) per neuron, then average (Eq 21)
    ipr_per_neuron = (w_tilde ** (2 * r)).sum(dim=1)  # (N,)
    ipr_avg = ipr_per_neuron.mean().item()

    return {"ipr": ipr_avg}


def compute_accuracy(logits, targets):
    """Classification accuracy (argmax prediction vs target label)."""
    preds = logits.argmax(dim=1)
    correct = (preds == targets).float().mean().item()
    return correct * 100.0


def fourier_spectrum(model):
    """Full Fourier power spectrum |W_tilde_1(nu)|^2 per neuron.

    Returns dict with 'spectrum': list of lists (N x p), each entry is
    the squared magnitude of the Fourier coefficient at that frequency.
    """
    W1 = model.W1.data
    p = model.P

    W1_n = W1[:, :p]  # (N, p) — first-operand weights
    W1_fft = torch.fft.fft(W1_n, dim=1)  # (N, p) complex
    power = (W1_fft.abs() ** 2)  # (N, p)

    return {"spectrum": power.cpu().tolist()}
