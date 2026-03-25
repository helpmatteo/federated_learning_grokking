"""Visualization for Experiment 7: Mechanistic checkpoint analysis.

Produces 5 figures:
  Fig 11: Fourier spectrum heatmaps (pre-grok, mid-grok, post-grok)
  Fig 12: Gini coefficient + effective rank evolution
  Fig 13: Restricted vs excluded loss (Nanda's progress measures)
  Fig 14: Neuron frequency clustering histograms
  Fig 15: Per-client Fourier coverage and spectral divergence
"""

import json
import os

import numpy as np
import matplotlib.pyplot as plt
import torch


OUTPUT_DIR = "results/exp7_mechanistic"
FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")


def load_analysis():
    path = os.path.join(OUTPUT_DIR, "mechanistic_analysis.json")
    with open(path) as f:
        return json.load(f)


def load_spectrum(config_name: str, step: int) -> np.ndarray:
    """Load a saved Fourier spectrum for a specific checkpoint."""
    for prefix in ["round", "epoch"]:
        path = os.path.join(OUTPUT_DIR, config_name, "checkpoints",
                           f"spectrum_{prefix}{step}.pt")
        if os.path.exists(path):
            return np.array(torch.load(path, weights_only=False, map_location="cpu"))
    return None


def fig11_fourier_heatmaps(data: dict):
    """Fig 11: Fourier spectrum heatmaps at 3 timepoints for each config."""
    configs = ["c1_centralized", "c3_fl_K10_iid", "c5_fl_K10_noniid"]
    labels = ["Centralized", "FL K=10 IID", "FL K=10 non-IID"]

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle("Fig 11: Fourier Spectrum Evolution (|FFT(W1)|² per neuron)",
                 fontsize=14, y=0.98)

    for row, (cfg_name, label) in enumerate(zip(configs, labels)):
        if cfg_name not in data or not data[cfg_name].get("steps"):
            continue
        steps = data[cfg_name]["steps"]
        n = len(steps)
        indices = [n // 10, n // 2, 9 * n // 10]
        timepoint_labels = ["Pre-grok", "Mid-grok", "Post-grok"]

        for col, (idx, tp_label) in enumerate(zip(indices, timepoint_labels)):
            step = steps[idx]
            spec = load_spectrum(cfg_name, step)
            if spec is None:
                continue

            ax = axes[row, col]
            p = spec.shape[1]
            half_p = p // 2 + 1
            spec_half = spec[:, :half_p]
            dominant = np.argmax(spec_half, axis=1)
            sort_idx = np.argsort(dominant)
            spec_sorted = spec_half[sort_idx]

            im = ax.imshow(spec_sorted, aspect="auto", cmap="hot",
                          interpolation="nearest")
            ax.set_xlabel("Fourier frequency")
            if col == 0:
                ax.set_ylabel(f"{label}\nNeuron (sorted)")
            ax.set_title(f"{tp_label} (step {step})")

    plt.tight_layout()
    os.makedirs(FIGURE_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIGURE_DIR, "fig11_fourier_heatmaps.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print("Saved fig11_fourier_heatmaps.png")


def fig12_gini_rank_evolution(data: dict):
    """Fig 12: Gini coefficient and effective rank over training."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Fig 12: Fourier Sparsification and Rank Collapse", fontsize=14)

    colors = {
        "c1_centralized": "black",
        "c2_fl_K5_iid": "tab:blue",
        "c3_fl_K10_iid": "tab:green",
        "c4_fl_K10_boundary": "tab:red",
        "c5_fl_K10_noniid": "tab:orange",
        "c6_fl_K97_iid": "tab:purple",
    }
    labels = {
        "c1_centralized": "Centralized",
        "c2_fl_K5_iid": "FL K=5 IID",
        "c3_fl_K10_iid": "FL K=10 IID",
        "c4_fl_K10_boundary": "FL K=10 a=0.25",
        "c5_fl_K10_noniid": "FL K=10 non-IID",
        "c6_fl_K97_iid": "FL K=97 IID",
    }

    for cfg_name, d in data.items():
        if not d.get("steps"):
            continue
        steps = d["steps"]
        c = colors.get(cfg_name, "gray")
        lbl = labels.get(cfg_name, cfg_name)

        axes[0, 0].plot(steps, d["gini_w1_fourier"], color=c, label=lbl, alpha=0.8)
        axes[0, 1].plot(steps, d["gini_w2_fourier"], color=c, label=lbl, alpha=0.8)
        axes[1, 0].plot(steps, d["effective_rank_w1"], color=c, label=lbl, alpha=0.8)
        axes[1, 1].plot(steps, d["effective_rank_w2"], color=c, label=lbl, alpha=0.8)

    axes[0, 0].set_title("Gini(W1 Fourier)")
    axes[0, 0].set_ylabel("Gini coefficient")
    axes[0, 1].set_title("Gini(W2 Fourier)")
    axes[1, 0].set_title("Effective Rank(W1)")
    axes[1, 0].set_ylabel("Effective rank")
    axes[1, 1].set_title("Effective Rank(W2)")

    for ax in axes.flat:
        ax.set_xlabel("Step / Round")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, "fig12_gini_rank_evolution.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print("Saved fig12_gini_rank_evolution.png")


def fig13_restricted_excluded_loss(data: dict):
    """Fig 13: Nanda's progress measures — restricted vs excluded loss."""
    configs = ["c1_centralized", "c3_fl_K10_iid", "c4_fl_K10_boundary", "c5_fl_K10_noniid"]
    labels = ["Centralized", "FL K=10 IID", "FL K=10 a=0.25", "FL K=10 non-IID"]

    present = [(c, l) for c, l in zip(configs, labels) if c in data and data[c].get("steps")]
    if not present:
        print("Skipping fig13: no data")
        return

    fig, axes = plt.subplots(1, len(present), figsize=(5 * len(present), 5))
    if len(present) == 1:
        axes = [axes]
    fig.suptitle("Fig 13: Restricted vs Excluded Loss (Nanda's Progress Measures)",
                 fontsize=14)

    for i, (cfg_name, label) in enumerate(present):
        d = data[cfg_name]
        steps = d["steps"]
        ax = axes[i]

        ax.semilogy(steps, d["restricted_loss"], "b-", label="Restricted (key freqs only)", alpha=0.8)
        ax.semilogy(steps, d["excluded_loss"], "r-", label="Excluded (non-key freqs)", alpha=0.8)
        ax.set_title(label)
        ax.set_xlabel("Step / Round")
        if i == 0:
            ax.set_ylabel("MSE Loss (log)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, "fig13_restricted_excluded_loss.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print("Saved fig13_restricted_excluded_loss.png")


def fig14_neuron_clustering(data: dict):
    """Fig 14: Neuron frequency clustering histograms at 3 timepoints."""
    configs = ["c1_centralized", "c3_fl_K10_iid", "c5_fl_K10_noniid"]
    labels = ["Centralized", "FL K=10 IID", "FL K=10 non-IID"]

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle("Fig 14: Neuron Frequency Clustering (dominant freq per neuron)",
                 fontsize=14, y=0.98)

    for row, (cfg_name, label) in enumerate(zip(configs, labels)):
        if cfg_name not in data or not data[cfg_name].get("steps"):
            continue
        d = data[cfg_name]
        steps = d["steps"]
        freq_hist = d["freq_histogram"]
        n = len(steps)
        indices = [n // 10, n // 2, 9 * n // 10]
        timepoint_labels = ["Pre-grok", "Mid-grok", "Post-grok"]
        p = 97

        for col, (idx, tp_label) in enumerate(zip(indices, timepoint_labels)):
            ax = axes[row, col]
            freqs = freq_hist[idx]
            freqs_folded = [min(f, p - f) for f in freqs]
            ax.hist(freqs_folded, bins=range(p // 2 + 2), color="steelblue",
                    edgecolor="white", linewidth=0.5)
            ax.set_title(f"{tp_label} (step {steps[idx]})")
            if col == 0:
                ax.set_ylabel(f"{label}\nCount")
            ax.set_xlabel("Fourier frequency")
            ax.set_xlim(0, p // 2 + 1)

            if d.get("key_freqs"):
                for kf in d["key_freqs"]:
                    kf_folded = min(kf, p - kf)
                    ax.axvline(kf_folded, color="red", linestyle="--", alpha=0.5, linewidth=1)

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, "fig14_neuron_clustering.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print("Saved fig14_neuron_clustering.png")


def fig15_client_fourier(data: dict):
    """Fig 15: Per-client Fourier coverage and spectral divergence."""
    configs = ["c2_fl_K5_iid", "c3_fl_K10_iid", "c5_fl_K10_noniid", "c6_fl_K97_iid"]
    labels = ["FL K=5 IID", "FL K=10 IID", "FL K=10 non-IID", "FL K=97 IID"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Fig 15: Per-Client Fourier Analysis", fontsize=14)

    for cfg_name, label in zip(configs, labels):
        if cfg_name not in data:
            continue
        d = data[cfg_name]
        if "client_fourier" not in d:
            continue
        cf = d["client_fourier"]
        rounds = cf["rounds"]

        axes[0, 0].plot(rounds, cf["fourier_coverage"], label=label, alpha=0.8)
        axes[0, 1].plot(rounds, cf["spectral_divergence"], label=label, alpha=0.8)

    axes[0, 0].set_title("Fourier Coverage (frac key freqs present)")
    axes[0, 0].set_ylabel("Coverage")
    axes[0, 0].set_ylim(0, 1.1)
    axes[0, 1].set_title("Spectral Divergence Across Clients")
    axes[0, 1].set_ylabel("Mean std of power spectrum")

    for col, (cfg_name, label) in enumerate([("c3_fl_K10_iid", "FL K=10 IID"),
                                              ("c5_fl_K10_noniid", "FL K=10 non-IID")]):
        if cfg_name not in data or "client_fourier" not in data[cfg_name]:
            continue
        cf = data[cfg_name]["client_fourier"]
        rounds = cf["rounds"]
        per_client = cf["per_client_key_power"]

        ax = axes[1, col]
        n_clients = len(per_client[0]) if per_client else 0
        for c_idx in range(n_clients):
            client_powers = [per_client[r][c_idx] for r in range(len(rounds))]
            ax.plot(rounds, client_powers, alpha=0.3, linewidth=0.8)
        ax.set_title(f"{label}: Per-Client Key Freq Power")
        ax.set_xlabel("Round")
        ax.set_ylabel("Mean power at key freqs")

    for ax in axes[0]:
        ax.set_xlabel("Round")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, "fig15_client_fourier.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print("Saved fig15_client_fourier.png")


def main():
    os.makedirs(FIGURE_DIR, exist_ok=True)
    data = load_analysis()

    fig11_fourier_heatmaps(data)
    fig12_gini_rank_evolution(data)
    fig13_restricted_excluded_loss(data)
    fig14_neuron_clustering(data)
    fig15_client_fourier(data)

    print(f"\nAll figures saved to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
