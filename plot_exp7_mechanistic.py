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

# Local epochs used in FL configs — needed to convert rounds → gradient steps
LOCAL_EPOCHS = 5

# Config metadata for consistent styling
CONFIG_META = {
    "c1_centralized":    {"label": "Centralized",       "color": "black",      "ls": "-",  "is_fl": False},
    "c2_fl_K5_iid":      {"label": "FL K=5 IID",        "color": "tab:blue",   "ls": "-",  "is_fl": True},
    "c3_fl_K10_iid":     {"label": "FL K=10 IID",       "color": "tab:green",  "ls": "-",  "is_fl": True},
    "c4_fl_K10_boundary":{"label": "FL K=10 α=0.25",    "color": "tab:red",    "ls": "--", "is_fl": True},
    "c5_fl_K10_noniid":  {"label": "FL K=10 non-IID",   "color": "tab:orange", "ls": "-.", "is_fl": True},
    "c6_fl_K97_iid":     {"label": "FL K=97 IID",       "color": "tab:purple", "ls": ":",  "is_fl": True},
}


def load_analysis():
    path = os.path.join(OUTPUT_DIR, "mechanistic_analysis.json")
    with open(path) as f:
        return json.load(f)


def to_grad_steps(steps, is_fl):
    """Convert rounds to gradient steps for FL configs."""
    if is_fl:
        return [s * LOCAL_EPOCHS for s in steps]
    return steps


def load_spectrum(config_name: str, step: int) -> np.ndarray:
    """Load a saved Fourier spectrum for a specific checkpoint."""
    for prefix in ["round", "epoch"]:
        path = os.path.join(OUTPUT_DIR, config_name, "checkpoints",
                           f"spectrum_{prefix}{step}.pt")
        if os.path.exists(path):
            return np.array(torch.load(path, weights_only=False, map_location="cpu"))
    return None


def fig11_fourier_heatmaps(data: dict):
    """Fig 11: Fourier spectrum heatmaps at 3 timepoints for 3 configs."""
    configs = ["c1_centralized", "c3_fl_K10_iid", "c5_fl_K10_noniid"]
    row_labels = ["Centralized", "FL K=10 IID", "FL K=10 non-IID"]

    fig, axes = plt.subplots(3, 3, figsize=(16, 13))
    fig.suptitle("Fourier Spectrum Evolution  ($|\\mathrm{FFT}(W_1)|^2$ per neuron)",
                 fontsize=16, fontweight="bold", y=0.99)

    for row, (cfg_name, label) in enumerate(zip(configs, row_labels)):
        if cfg_name not in data or not data[cfg_name].get("steps"):
            for col in range(3):
                axes[row, col].set_visible(False)
            continue

        meta = CONFIG_META[cfg_name]
        steps = data[cfg_name]["steps"]
        grad_steps = to_grad_steps(steps, meta["is_fl"])
        n = len(steps)
        # Pick timepoints: ~10%, ~50%, ~90% of training
        indices = [max(0, n // 10), n // 2, min(n - 1, 9 * n // 10)]
        tp_labels = ["Early", "Mid-training", "Late"]

        for col, (idx, tp_label) in enumerate(zip(indices, tp_labels)):
            step = steps[idx]
            gs = grad_steps[idx]
            spec = load_spectrum(cfg_name, step)
            if spec is None:
                axes[row, col].text(0.5, 0.5, "No data", transform=axes[row, col].transAxes,
                                    ha="center", va="center", fontsize=12, color="gray")
                continue

            ax = axes[row, col]
            p = spec.shape[1]
            half_p = p // 2 + 1
            spec_half = spec[:, :half_p]
            # Sort neurons by dominant frequency for clearer visualization
            dominant = np.argmax(spec_half, axis=1)
            sort_idx = np.argsort(dominant)
            spec_sorted = spec_half[sort_idx]

            im = ax.imshow(spec_sorted, aspect="auto", cmap="hot",
                          interpolation="nearest", vmin=0)
            ax.set_xlabel("Fourier frequency", fontsize=10)
            if col == 0:
                ax.set_ylabel(f"{label}\nNeuron (sorted)", fontsize=11, fontweight="bold")
            ax.set_title(f"{tp_label} (step {gs:,})", fontsize=11)

        # Add colorbar to last column
        cbar = fig.colorbar(im, ax=axes[row, 2], fraction=0.046, pad=0.04)
        cbar.set_label("Power", fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(FIGURE_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIGURE_DIR, "fig11_fourier_heatmaps.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print("Saved fig11_fourier_heatmaps.png")


def fig12_gini_rank_evolution(data: dict):
    """Fig 12: Gini coefficient and effective rank over training (gradient steps)."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Fourier Sparsification and Rank Collapse During Grokking",
                 fontsize=16, fontweight="bold")

    for cfg_name, d in data.items():
        if not d.get("steps"):
            continue
        meta = CONFIG_META.get(cfg_name)
        if not meta:
            continue
        grad_steps = to_grad_steps(d["steps"], meta["is_fl"])

        kw = dict(color=meta["color"], label=meta["label"], alpha=0.85,
                  linewidth=2, linestyle=meta["ls"])
        axes[0, 0].plot(grad_steps, d["gini_w1_fourier"], **kw)
        axes[0, 1].plot(grad_steps, d["gini_w2_fourier"], **kw)
        axes[1, 0].plot(grad_steps, d["effective_rank_w1"], **kw)
        axes[1, 1].plot(grad_steps, d["effective_rank_w2"], **kw)

    titles = [("Gini coefficient of $W_1$ (Fourier)", "Gini coefficient"),
              ("Gini coefficient of $W_2$ (Fourier)", "Gini coefficient"),
              ("Effective rank of $W_1$", "Effective rank"),
              ("Effective rank of $W_2$", "Effective rank")]

    for ax, (title, ylabel) in zip(axes.flat, titles):
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Gradient steps", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=10)

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, "fig12_gini_rank_evolution.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print("Saved fig12_gini_rank_evolution.png")


def fig13_restricted_excluded_loss(data: dict):
    """Fig 13: Nanda's progress measures — restricted vs excluded loss."""
    configs = ["c1_centralized", "c3_fl_K10_iid", "c4_fl_K10_boundary", "c5_fl_K10_noniid"]
    labels = ["Centralized", "FL K=10 IID", "FL K=10 α=0.25", "FL K=10 non-IID"]

    present = [(c, l) for c, l in zip(configs, labels) if c in data and data[c].get("steps")]
    if not present:
        print("Skipping fig13: no data")
        return

    ncols = len(present)
    fig, axes = plt.subplots(1, ncols, figsize=(5.5 * ncols, 5.5))
    if ncols == 1:
        axes = [axes]
    fig.suptitle("Restricted vs Excluded Loss (Nanda's Progress Measures)",
                 fontsize=16, fontweight="bold", y=1.02)

    for i, (cfg_name, label) in enumerate(present):
        d = data[cfg_name]
        meta = CONFIG_META[cfg_name]
        grad_steps = to_grad_steps(d["steps"], meta["is_fl"])
        ax = axes[i]

        ax.semilogy(grad_steps, d["restricted_loss"], "b-", linewidth=2,
                    label="Restricted (key freqs only)", alpha=0.85)
        ax.semilogy(grad_steps, d["excluded_loss"], "r-", linewidth=2,
                    label="Excluded (non-key freqs)", alpha=0.85)

        # Add key freq annotation
        key_freqs = d.get("key_freqs", [])
        freq_str = ", ".join(str(k) for k in key_freqs)
        ax.text(0.98, 0.02, f"Key freqs: [{freq_str}]", transform=ax.transAxes,
                fontsize=8, ha="right", va="bottom", color="gray",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        ax.set_title(label, fontsize=13, fontweight="bold")
        ax.set_xlabel("Gradient steps", fontsize=11)
        if i == 0:
            ax.set_ylabel("MSE Loss", fontsize=11)
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=10)

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, "fig13_restricted_excluded_loss.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print("Saved fig13_restricted_excluded_loss.png")


def fig14_neuron_clustering(data: dict):
    """Fig 14: Neuron frequency clustering histograms at 3 timepoints."""
    configs = ["c1_centralized", "c3_fl_K10_iid", "c5_fl_K10_noniid"]
    row_labels = ["Centralized", "FL K=10 IID", "FL K=10 non-IID"]
    p = 97

    fig, axes = plt.subplots(3, 3, figsize=(16, 13))
    fig.suptitle("Neuron Frequency Specialization (dominant Fourier freq per neuron)",
                 fontsize=16, fontweight="bold", y=0.99)

    for row, (cfg_name, label) in enumerate(zip(configs, row_labels)):
        if cfg_name not in data or not data[cfg_name].get("steps"):
            for col in range(3):
                axes[row, col].set_visible(False)
            continue

        d = data[cfg_name]
        meta = CONFIG_META[cfg_name]
        steps = d["steps"]
        grad_steps = to_grad_steps(steps, meta["is_fl"])
        freq_hist = d["freq_histogram"]
        n = len(steps)
        indices = [max(0, n // 10), n // 2, min(n - 1, 9 * n // 10)]
        tp_labels = ["Early", "Mid-training", "Late"]

        for col, (idx, tp_label) in enumerate(zip(indices, tp_labels)):
            ax = axes[row, col]
            gs = grad_steps[idx]
            freqs = freq_hist[idx]
            # Fold frequencies by symmetry: freq k and p-k are conjugates
            freqs_folded = [min(f, p - f) for f in freqs]

            ax.hist(freqs_folded, bins=np.arange(p // 2 + 2) - 0.5,
                    color="steelblue", edgecolor="white", linewidth=0.5, alpha=0.85)
            ax.set_title(f"{tp_label} (step {gs:,})", fontsize=11)
            if col == 0:
                ax.set_ylabel(f"{label}\nNeuron count", fontsize=11, fontweight="bold")
            ax.set_xlabel("Fourier frequency (folded)", fontsize=10)
            ax.set_xlim(-0.5, p // 2 + 1)

            # Mark key frequencies
            if d.get("key_freqs"):
                for kf in d["key_freqs"]:
                    kf_folded = min(kf, p - kf)
                    ax.axvline(kf_folded, color="red", linestyle="--", alpha=0.6,
                              linewidth=1.5, zorder=5)
                # Add legend entry for key freqs
                if col == 2:
                    ax.axvline(-10, color="red", linestyle="--", linewidth=1.5,
                              label="Key frequencies")
                    ax.legend(fontsize=8, loc="upper right")

            ax.grid(True, alpha=0.2, axis="y")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(FIGURE_DIR, "fig14_neuron_clustering.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print("Saved fig14_neuron_clustering.png")


def fig15_client_fourier(data: dict):
    """Fig 15: Per-client Fourier coverage and spectral divergence."""
    fl_configs = ["c2_fl_K5_iid", "c3_fl_K10_iid", "c5_fl_K10_noniid", "c6_fl_K97_iid"]

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle("Per-Client Fourier Analysis: Coverage, Divergence, and Key-Frequency Power",
                 fontsize=16, fontweight="bold")

    # Top row: aggregated metrics across configs
    for cfg_name in fl_configs:
        if cfg_name not in data or "client_fourier" not in data[cfg_name]:
            continue
        meta = CONFIG_META[cfg_name]
        cf = data[cfg_name]["client_fourier"]
        grad_steps = to_grad_steps(cf["rounds"], True)

        kw = dict(color=meta["color"], label=meta["label"], linewidth=2,
                  alpha=0.85, linestyle=meta["ls"])
        axes[0, 0].plot(grad_steps, cf["fourier_coverage"], **kw)
        axes[0, 1].plot(grad_steps, cf["spectral_divergence"], **kw)

    axes[0, 0].set_title("Fourier Coverage\n(fraction of key freqs with significant power)", fontsize=11)
    axes[0, 0].set_ylabel("Coverage", fontsize=11)
    axes[0, 0].set_ylim(-0.05, 1.15)
    axes[0, 0].axhline(1.0, color="gray", linestyle=":", alpha=0.5, linewidth=1)
    axes[0, 1].set_title("Spectral Divergence Across Clients\n(mean std of Fourier power)", fontsize=11)
    axes[0, 1].set_ylabel("Mean std of power spectrum", fontsize=11)

    for ax in axes[0]:
        ax.set_xlabel("Gradient steps", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=10)

    # Bottom row: per-client key frequency power for IID vs non-IID
    bottom_configs = [("c3_fl_K10_iid", "FL K=10 IID"), ("c5_fl_K10_noniid", "FL K=10 non-IID")]
    client_colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for col, (cfg_name, label) in enumerate(bottom_configs):
        ax = axes[1, col]
        if cfg_name not in data or "client_fourier" not in data[cfg_name]:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center", fontsize=12, color="gray")
            continue

        cf = data[cfg_name]["client_fourier"]
        grad_steps = to_grad_steps(cf["rounds"], True)
        per_client = cf["per_client_key_power"]
        n_clients = len(per_client[0]) if per_client else 0

        for c_idx in range(n_clients):
            client_powers = [per_client[r][c_idx] for r in range(len(grad_steps))]
            ax.plot(grad_steps, client_powers, color=client_colors[c_idx % 10],
                    alpha=0.5, linewidth=1.2, label=f"Client {c_idx}" if c_idx < 5 else None)

        # Add mean line
        mean_powers = [np.mean([per_client[r][c] for c in range(n_clients)])
                       for r in range(len(grad_steps))]
        ax.plot(grad_steps, mean_powers, color="black", linewidth=2.5,
                label="Mean", zorder=10)

        ax.set_title(f"{label}: Per-Client Key Freq Power", fontsize=12, fontweight="bold")
        ax.set_xlabel("Gradient steps", fontsize=11)
        ax.set_ylabel("Mean power at key freqs", fontsize=11)
        ax.legend(fontsize=8, loc="upper left", ncol=2)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=10)

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
