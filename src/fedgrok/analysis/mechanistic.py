"""Post-hoc mechanistic analysis of checkpoint data.

Loads model checkpoints saved by exp7 and computes:
- Fourier power spectrum evolution (global and per-client)
- Gini coefficient of Fourier components over time
- Effective rank of W1 and W2 over time
- Neuron frequency clustering over time
- Restricted vs excluded loss (Nanda's progress measures)
- Per-client Fourier coverage analysis
"""

import os
import glob
import json
import re

import numpy as np
import torch

from fedgrok.models.groknet import GrokNet
from fedgrok.metrics.fourier import (
    fourier_spectrum, compute_ipr, gini_coefficient,
    effective_rank, neuron_frequency_assignment, restricted_excluded_loss,
)
from fedgrok.data.modular import make_dataset
from fedgrok.core.config import Config
from fedgrok.core.utils import make_targets_onehot


def load_checkpoints(ckpt_dir: str) -> list:
    """Load all checkpoints from a directory, sorted by step.

    Returns list of (step, state_dict) tuples.
    """
    pattern = os.path.join(ckpt_dir, "ckpt_*.pt")
    files = sorted(glob.glob(pattern))
    checkpoints = []
    for f in files:
        match = re.search(r'ckpt_(?:round|epoch)(\d+)\.pt', os.path.basename(f))
        if match:
            step = int(match.group(1))
            state_dict = torch.load(f, weights_only=True, map_location="cpu")
            checkpoints.append((step, state_dict))
    checkpoints.sort(key=lambda x: x[0])
    return checkpoints


def load_client_w1(ckpt_dir: str) -> list:
    """Load per-client W1 data from checkpoint directory.

    Returns list of (round, client_w1_list) tuples.
    """
    pattern = os.path.join(ckpt_dir, "client_w1_round*.pt")
    files = sorted(glob.glob(pattern))
    results = []
    for f in files:
        match = re.search(r'client_w1_round(\d+)\.pt', os.path.basename(f))
        if match:
            rnd = int(match.group(1))
            data = torch.load(f, weights_only=False, map_location="cpu")
            results.append((rnd, data))
    results.sort(key=lambda x: x[0])
    return results


def identify_key_frequencies(model, top_k: int = 5) -> list:
    """Identify the top-k dominant Fourier frequencies from a trained model.

    Uses the global Fourier power spectrum averaged over all neurons.
    """
    spec = fourier_spectrum(model)
    power = np.array(spec["spectrum"])  # (N, p)
    avg_power = power.mean(axis=0)     # (p,)
    p = len(avg_power)
    avg_power[0] = 0  # ignore DC
    top_indices = np.argsort(avg_power)[::-1][:top_k]
    return sorted(top_indices.tolist())


def load_run_config(config_dir: str):
    """Rebuild the exact Config a run used, from the spec.json it wrote.

    Returns None when the directory predates spec.json (see `analyze_config`).
    """
    spec_path = os.path.join(config_dir, "spec.json")
    if not os.path.exists(spec_path):
        return None
    from fedgrok.manifest import build_config
    with open(spec_path) as handle:
        return build_config(json.load(handle))


def analyze_setup_agnostic(config_dir: str) -> dict:
    """Mechanistic time-series that apply to ANY architecture.

    The Fourier path below needs GrokNet on a cyclic group. This one needs only a
    checkpointed state_dict, so it is the analysis available for the transformer,
    the MNIST MLP, and S_5 — which between them had none. Everything here is
    basis-free: total and per-matrix weight norms (the Omnigrok order parameter)
    and effective rank (Roy & Bhattacharyya) of each weight matrix.
    """
    ckpt_dir = os.path.join(config_dir, "checkpoints")
    checkpoints = load_checkpoints(ckpt_dir) if os.path.exists(ckpt_dir) else []
    if not checkpoints:
        return {}

    results = {"steps": [], "weight_norm_total": [], "per_matrix_rank": [],
               "per_matrix_norm": [], "gini_spectrum": []}
    for step, state in checkpoints:
        # Norms are defined for any tensor; the SVD-based measures need a matrix.
        # The transformer stores its attention projections per head (W_Q is
        # (n_heads, d_model, d_head)), and torch.linalg.svdvals would treat that
        # as a BATCH of SVDs and return a 2-D result -- so restrict rank and
        # spectral Gini to genuinely 2-D parameters rather than reporting a
        # batched quantity as if it were one spectrum. Those tensors still
        # contribute their norms.
        tensors = {k: v for k, v in state.items()
                   if hasattr(v, "dim") and v.dim() >= 2}
        mats = {k: v for k, v in tensors.items() if v.dim() == 2}
        results["steps"].append(step)
        results["weight_norm_total"].append(
            float(sum(float(v.norm()) ** 2 for v in tensors.values()) ** 0.5))
        results["per_matrix_norm"].append(
            {k: float(v.norm()) for k, v in tensors.items()})
        results["per_matrix_rank"].append(
            {k: effective_rank(v) for k, v in mats.items()})
        results["gini_spectrum"].append(
            {k: gini_coefficient(torch.linalg.svdvals(v.float()))
             for k, v in mats.items()})
    return results


def analyze_config(config_dir: str, p: int = None, hidden_width: int = None,
                   is_centralized: bool = False) -> dict:
    """Run full mechanistic analysis on a single config's checkpoints.

    The run's own config is read from the `spec.json` that `fedgrok.run` writes
    next to the history. This used to be hardcoded as
    `Config(p=p, alpha=0.5, seed=42)` with `task` defaulting to "addition", and a
    fresh `GrokNet(...)` — which meant (a) a transformer or MNIST checkpoint
    could not be loaded at all, and (b) for any run at a different alpha, seed or
    task, restricted/excluded loss was silently evaluated against the WRONG train
    split. Every v2 sweep runs alpha=0.3 or 0.25, so that second failure was live.

    `p` and `hidden_width` remain as overrides for directories that predate
    spec.json.

    Returns a dict of time-series. Non-GrokNet or non-cyclic runs get the
    setup-agnostic measures only; the Fourier block requires both.
    """
    ckpt_dir = os.path.join(config_dir, "checkpoints")
    if not os.path.exists(ckpt_dir):
        print(f"No checkpoints in {config_dir}")
        return {}

    checkpoints = load_checkpoints(ckpt_dir)
    if not checkpoints:
        print(f"No checkpoint files found in {ckpt_dir}")
        return {}

    cfg = load_run_config(config_dir)
    if cfg is None:
        if p is None:
            raise ValueError(
                f"{config_dir} has no spec.json (it predates run-config "
                f"provenance) and no explicit p= was given, so the architecture "
                f"and data split are unknown. Pass p=/hidden_width= to analyse "
                f"it as a GrokNet run."
            )
        cfg = Config(p=p, alpha=0.5, seed=42,
                     hidden_width=hidden_width or 256)
        print(f"WARNING: {config_dir} has no spec.json; assuming GrokNet "
              f"p={p}, alpha=0.5, seed=42, task=addition. Any restricted/"
              f"excluded loss is only valid if the run actually used those.")

    base = analyze_setup_agnostic(config_dir)

    from fedgrok.core.registry import build_model
    from fedgrok.data.registry import build_dataset
    from fedgrok.metrics.fourier import dft_applicable

    model = build_model(cfg)
    if not dft_applicable(model, cfg):
        # Transformer / MNIST MLP / S_5: the Z_p DFT is meaningless. Return what
        # is defined rather than fabricating a spectrum.
        base["setup"] = {"dataset": cfg.dataset, "model": cfg.model,
                         "loss": cfg.loss}
        base["fourier_applicable"] = False
        return base

    p = cfg.p
    hidden_width = cfg.hidden_width
    x_train, y_train, x_test, y_test = build_dataset(cfg)
    y_train_oh = make_targets_onehot(y_train, p)

    # Identify key frequencies from the final checkpoint
    final_step, final_sd = checkpoints[-1]
    model.load_state_dict(final_sd)
    key_freqs = identify_key_frequencies(model)
    print(f"Key frequencies (from final checkpoint): {key_freqs}")

    results = {
        **base,
        "setup": {"dataset": cfg.dataset, "model": cfg.model, "loss": cfg.loss},
        "fourier_applicable": True,
        "steps": [],
        "ipr": [],
        "gini_w1_fourier": [],
        "gini_w2_fourier": [],
        "effective_rank_w1": [],
        "effective_rank_w2": [],
        "freq_histogram": [],
        "restricted_loss": [],
        "excluded_loss": [],
        "key_freqs": key_freqs,
    }

    for step, sd in checkpoints:
        model.load_state_dict(sd)

        ipr_val = compute_ipr(model)["ipr"]

        # Gini of Fourier magnitudes (averaged over neurons)
        W1_n = model.W1.data[:, :p]
        W1_fft = torch.fft.fft(W1_n, dim=1)
        W1_mag = W1_fft.abs()
        gini_w1 = float(np.mean([gini_coefficient(W1_mag[k]) for k in range(W1_mag.shape[0])]))

        W2_fft = torch.fft.fft(model.W2.data, dim=1)
        W2_mag = W2_fft.abs()
        gini_w2 = float(np.mean([gini_coefficient(W2_mag[k]) for k in range(W2_mag.shape[0])]))

        erank_w1 = effective_rank(model.W1.data)
        erank_w2 = effective_rank(model.W2.data)

        freqs = neuron_frequency_assignment(model)

        re_loss = restricted_excluded_loss(model, x_train, y_train_oh, key_freqs)

        results["steps"].append(step)
        results["ipr"].append(ipr_val)
        results["gini_w1_fourier"].append(gini_w1)
        results["gini_w2_fourier"].append(gini_w2)
        results["effective_rank_w1"].append(erank_w1)
        results["effective_rank_w2"].append(erank_w2)
        results["freq_histogram"].append(freqs)
        results["restricted_loss"].append(re_loss["restricted_loss"])
        results["excluded_loss"].append(re_loss["excluded_loss"])

    # Per-client analysis (if available)
    client_data = load_client_w1(ckpt_dir)
    if client_data:
        results["client_fourier"] = analyze_client_fourier(client_data, p, key_freqs)

    return results


def analyze_client_fourier(client_data: list, p: int, key_freqs: list) -> dict:
    """Analyze per-client Fourier spectra over time."""
    results = {
        "rounds": [],
        "fourier_coverage": [],
        "per_client_key_power": [],
        "spectral_divergence": [],
    }

    for rnd, client_w1_list in client_data:
        n_clients = len(client_w1_list)
        if n_clients == 0:
            continue

        all_spectra = []
        client_key_powers = []

        for cw1 in client_w1_list:
            w1_tensor = torch.tensor(np.array(cw1), dtype=torch.float32)
            fft_result = torch.fft.fft(w1_tensor, dim=1)
            power = (fft_result.abs() ** 2).numpy()
            avg_power = power.mean(axis=0)
            all_spectra.append(avg_power)

            key_power = sum(avg_power[k] for k in key_freqs) / max(1, len(key_freqs))
            client_key_powers.append(float(key_power))

        all_spectra = np.array(all_spectra)

        # Fourier coverage: which key freqs have significant power in at least one client
        threshold = np.median(all_spectra) * 2
        covered = sum(1 for k in key_freqs if any(all_spectra[c, k] > threshold for c in range(n_clients)))
        coverage = covered / max(1, len(key_freqs))

        spectral_div = float(np.mean(np.std(all_spectra, axis=0)))

        results["rounds"].append(rnd)
        results["fourier_coverage"].append(float(coverage))
        results["per_client_key_power"].append(client_key_powers)
        results["spectral_divergence"].append(spectral_div)

    return results


def run_analysis(output_dir: str = "results/experiments/exp7_mechanistic"):
    """Run analysis on all exp7 configs and save results."""
    config_dirs = {
        "c1_centralized": True,
        "c2_fl_K5_iid": False,
        "c3_fl_K10_iid": False,
        "c4_fl_K10_boundary": False,
        "c5_fl_K10_noniid": False,
        "c6_fl_K97_iid": False,
    }

    all_results = {}
    for name, is_cent in config_dirs.items():
        config_path = os.path.join(output_dir, name)
        if not os.path.exists(config_path):
            print(f"Skipping {name} (not found)")
            continue
        print(f"\nAnalyzing {name}...")
        results = analyze_config(config_path, is_centralized=is_cent)
        all_results[name] = results

    analysis_path = os.path.join(output_dir, "mechanistic_analysis.json")

    def _convert(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_convert(v) for v in obj]
        return obj

    with open(analysis_path, "w") as f:
        json.dump(_convert(all_results), f, indent=2)
    print(f"\nAnalysis saved to {analysis_path}")
    return all_results


if __name__ == "__main__":
    run_analysis()
