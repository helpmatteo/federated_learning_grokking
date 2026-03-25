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


# ---------------------------------------------------------------------------
# Mechanistic metrics
# ---------------------------------------------------------------------------


def gini_coefficient(x: torch.Tensor) -> float:
    """Gini coefficient of a 1-D tensor measuring inequality/sparsity.

    Returns 0 for perfectly equal distribution, ~1 for maximally unequal.
    """
    x = x.detach().float().abs()
    if x.sum() == 0:
        return 0.0
    sorted_x, _ = torch.sort(x)
    n = len(sorted_x)
    index = torch.arange(1, n + 1, dtype=torch.float32, device=x.device)
    return (2.0 * (index * sorted_x).sum() / (n * sorted_x.sum()) - (n + 1) / n).item()


def effective_rank(W: torch.Tensor) -> float:
    """Effective rank via Shannon entropy of normalized singular values.

    Roy & Bhattacharyya (2007).  Returns exp(entropy) where entropy is
    computed from the normalised singular value distribution.
    """
    S = torch.linalg.svdvals(W.detach().float())
    S = S[S > 1e-10]
    if len(S) == 0:
        return 0.0
    p = S / S.sum()
    entropy = -(p * p.log()).sum()
    return entropy.exp().item()


def neuron_frequency_assignment(model) -> list:
    """Assign each neuron its dominant Fourier frequency.

    For each row of W1 (first p columns), compute the DFT and return the
    argmax of the magnitude spectrum.  Returns a list of length N.
    """
    W1 = model.W1.data
    p = model.P
    W1_n = W1[:, :p]
    W1_fft = torch.fft.fft(W1_n, dim=1)
    magnitudes = W1_fft.abs()
    return magnitudes.argmax(dim=1).tolist()


def restricted_excluded_loss(model, x: torch.Tensor, y_onehot: torch.Tensor,
                             key_freqs: list) -> dict:
    """Nanda's progress measures: restricted and excluded loss.

    Computes logits, takes FFT along the output (class) dimension, then:
      - restricted_loss: keep only Fourier components at *key_freqs*, zero the rest
      - excluded_loss:  zero out Fourier components at *key_freqs*, keep the rest

    Both losses are MSE against *y_onehot*.
    """
    with torch.no_grad():
        logits = model(x)
        logits_fft = torch.fft.fft(logits, dim=1)

        # Restricted: keep only key_freqs
        restricted_fft = torch.zeros_like(logits_fft)
        for k in key_freqs:
            restricted_fft[:, k] = logits_fft[:, k]
        restricted_logits = torch.fft.ifft(restricted_fft, dim=1).real

        # Excluded: remove key_freqs
        excluded_fft = logits_fft.clone()
        for k in key_freqs:
            excluded_fft[:, k] = 0
        excluded_logits = torch.fft.ifft(excluded_fft, dim=1).real

        restricted_loss = torch.nn.functional.mse_loss(restricted_logits, y_onehot).item()
        excluded_loss = torch.nn.functional.mse_loss(excluded_logits, y_onehot).item()

    return {"restricted_loss": restricted_loss, "excluded_loss": excluded_loss}
