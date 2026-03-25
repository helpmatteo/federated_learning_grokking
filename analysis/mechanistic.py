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

from core.model import GrokNet
from core.metrics import (
    fourier_spectrum, compute_ipr, gini_coefficient,
    effective_rank, neuron_frequency_assignment, restricted_excluded_loss,
)
from core.dataset import make_dataset
from core.config import Config
from core.utils import make_targets_onehot


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


def analyze_config(config_dir: str, p: int = 97, hidden_width: int = 256,
                   is_centralized: bool = False) -> dict:
    """Run full mechanistic analysis on a single config's checkpoints.

    Returns dict with time-series of all mechanistic metrics.
    """
    ckpt_dir = os.path.join(config_dir, "checkpoints")
    if not os.path.exists(ckpt_dir):
        print(f"No checkpoints in {config_dir}")
        return {}

    checkpoints = load_checkpoints(ckpt_dir)
    if not checkpoints:
        print(f"No checkpoint files found in {ckpt_dir}")
        return {}

    model = GrokNet(input_dim=2 * p, hidden_width=hidden_width, output_dim=p)

    # Prepare training data for restricted/excluded loss
    cfg = Config(p=p, alpha=0.5, seed=42, hidden_width=hidden_width)
    x_train, y_train, x_test, y_test = make_dataset(cfg)
    y_train_oh = make_targets_onehot(y_train, p)

    # Identify key frequencies from the final checkpoint
    final_step, final_sd = checkpoints[-1]
    model.load_state_dict(final_sd)
    key_freqs = identify_key_frequencies(model)
    print(f"Key frequencies (from final checkpoint): {key_freqs}")

    results = {
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


def run_analysis(output_dir: str = "results/exp7_mechanistic"):
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
