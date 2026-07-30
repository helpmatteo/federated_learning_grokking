"""Metrics from Gromov (2023): weight/gradient norms, IPR, phase analysis.

Capability note: the Fourier metrics here (compute_ipr, fourier_spectrum,
neuron_frequency_assignment, restricted_excluded_loss) are specific to GrokNet
— they read `model.W1`/`model.P` and assume the modular-arithmetic input
convention (first p columns are the one-hot of operand n). They do NOT apply to
a transformer, an MNIST MLP, or an S5 model; call `fourier_applicable(model)`
first and skip them when it is False. weight_norms / gradient_norms are likewise
GrokNet-specific (W1/W2). effective_rank and gini_coefficient are basis-free and
apply to any tensor.
"""

import torch
import numpy as np


def dft_applicable(model, cfg=None) -> bool:
    """True iff the DFT-based Fourier metrics are meaningful for this run.

    Two conditions:
      1. `model` exposes the GrokNet interface the metrics read — weight matrix
         `W1` and output size `P`, with the first columns the one-hot operand.
         A transformer / MNIST MLP lacks this.
      2. The task is over a CYCLIC group. The DFT presumes Z_p structure; on a
         non-abelian group like S_n it is meaningless, so even a GrokNet run on
         the "s5" dataset must skip it (use coset-attribution metrics instead).
         Only checked when `cfg` is supplied.
    """
    if not (hasattr(model, "W1") and hasattr(model, "P")):
        return False
    if cfg is not None and getattr(cfg, "dataset", "modular") != "modular":
        return False
    return True


# Historical name. `dft_applicable` says what it actually gates; this alias keeps
# existing call sites working.
fourier_applicable = dft_applicable


def weight_norms_applicable(model) -> bool:
    """True iff this model exposes GrokNet's named W1/W2 layers.

    Deliberately NOT the same test as `dft_applicable`. A Frobenius norm is
    basis-free: it is well defined on S_n, where the DFT is meaningless. Gating
    the two together meant every S5+GrokNet run recorded weight_norm_layer1/2 as
    NaN despite the code being present and correct — a measurement reported as
    "not applicable" when it applied.
    """
    return hasattr(model, "W1") and hasattr(model, "W2")


def weight_norms(model):
    """Frobenius norm of W1 and W2."""
    return {
        "weight_norm_layer1": model.W1.data.norm().item(),
        "weight_norm_layer2": model.W2.data.norm().item(),
    }


def weight_norm_report(model):
    """Architecture-agnostic weight norms, for models with no W1/W2 interface.

    Total parameter norm is the Omnigrok order parameter — the quantity whose
    drift into the "Goldilocks zone" produces delayed generalisation — so it
    must be logged on the MNIST MLP and the transformer, neither of which has a
    W1. Restricted to matrices (dim >= 2) so biases do not dominate the ends.
    """
    mats = [(n, p) for n, p in model.named_parameters() if p.dim() >= 2]
    total = float(sum(float(p.data.norm()) ** 2 for _, p in mats) ** 0.5)
    return {
        "weight_norm_total": total,
        "weight_norm_first": float(mats[0][1].data.norm()) if mats else float("nan"),
        "weight_norm_last": float(mats[-1][1].data.norm()) if mats else float("nan"),
    }


def gradient_norms(model):
    """Frobenius norm of gradients of W1 and W2 (call after loss.backward())."""
    norms = {}
    if model.W1.grad is not None:
        norms["grad_norm_layer1"] = model.W1.grad.norm().item()
    if model.W2.grad is not None:
        norms["grad_norm_layer2"] = model.W2.grad.norm().item()
    return norms


def spectral_ipr(mat, r=2):
    """Inverse Participation Ratio of a (rows, period) matrix's row-wise DFT.

    Factored out of `compute_ipr` so the same measure can be applied to any
    matrix whose rows are indexed by a cyclic variable — GrokNet's W1[:, :p]
    (neurons x operand) and the transformer's W_E.T (d_model x token) are the
    same object in this sense. Operation order is unchanged from the original,
    so GrokNet IPR values are bit-identical.
    """
    fft = torch.fft.fft(mat, dim=1)                      # (rows, period) complex
    magnitudes = fft.abs()
    norms = magnitudes.norm(dim=1, keepdim=True).clamp(min=1e-10)
    w_tilde = magnitudes / norms
    return (w_tilde ** (2 * r)).sum(dim=1).mean().item()


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

    # Take the first p columns (corresponding to the n input)
    return {"ipr": spectral_ipr(W1[:, :p], r=r)}


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
