"""Plot Experiment 2 results from available per-cell JSON files."""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
})

RESULTS_DIR = "results/exp2_aggregation"
OUTPUT_DIR = "results/exp2_aggregation/figures"


def parse_inf(val):
    if val == "inf" or val == float("inf"):
        return float("inf")
    return float(val)


def load_all_cells():
    """Load all per-cell JSON files."""
    cells = []
    for fname in sorted(os.listdir(RESULTS_DIR)):
        if fname.startswith("exp2_a") and fname.endswith(".json"):
            with open(os.path.join(RESULTS_DIR, fname)) as f:
                cells.append(json.load(f))
    return cells


def plot_grokking_heatmap(cells):
    """Heatmap: did each (alpha, K, condition) grok?"""
    alphas = sorted(set(c["alpha"] for c in cells))
    ks = sorted(set(c["K"] for c in cells))

    conditions = ["cent_full", "cent_reduced", "fl_iid"]
    cond_labels = ["Centralized (full)", "Centralized (reduced)", "FL IID"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

    for ax, cond, label in zip(axes, conditions, cond_labels):
        grid = np.full((len(alphas), len(ks)), np.nan)
        for c in cells:
            ai = alphas.index(c["alpha"])
            ki = ks.index(c["K"])
            summary = c[cond]["summary"]
            n_grokked = summary["n_grokked"]
            n_seeds = summary["n_seeds"]
            grid[ai, ki] = n_grokked / n_seeds

        im = ax.imshow(grid, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1,
                       origin="lower")
        ax.set_xticks(range(len(ks)))
        ax.set_xticklabels(ks)
        ax.set_yticks(range(len(alphas)))
        ax.set_yticklabels([f"{a:.2f}" for a in alphas])
        ax.set_xlabel("K (clients)")
        ax.set_title(label)

        # Annotate cells
        for i in range(len(alphas)):
            for j in range(len(ks)):
                val = grid[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                            fontsize=9, color="black" if 0.3 < val < 0.7 else "white")
                else:
                    ax.text(j, i, "?", ha="center", va="center",
                            fontsize=9, color="gray")

    axes[0].set_ylabel(r"$\alpha$ (train fraction)")
    fig.colorbar(im, ax=axes, label="Fraction of seeds that grokked", shrink=0.8)
    fig.suptitle("Exp 2: Grokking Success Rate", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "exp2_grokking_heatmap.png"), bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/exp2_grokking_heatmap.png")


def plot_t_grok_vs_K(cells):
    """T_grok vs K for each alpha, comparing conditions."""
    alphas = sorted(set(c["alpha"] for c in cells))
    ks = sorted(set(c["K"] for c in cells))

    fig, axes = plt.subplots(1, len(alphas), figsize=(4 * len(alphas), 4.5), sharey=True)
    if len(alphas) == 1:
        axes = [axes]

    conditions = ["cent_full", "fl_iid", "cent_reduced"]
    colors = {"cent_full": "#2196F3", "fl_iid": "#FF5722", "cent_reduced": "#9E9E9E"}
    labels = {"cent_full": "Centralized (full)", "fl_iid": "FL IID", "cent_reduced": "Centralized (reduced)"}
    markers = {"cent_full": "o", "fl_iid": "s", "cent_reduced": "^"}

    for ax, alpha in zip(axes, alphas):
        for cond in conditions:
            k_vals, means, stds = [], [], []
            for K in ks:
                match = [c for c in cells if c["alpha"] == alpha and c["K"] == K]
                if not match:
                    continue
                summary = match[0][cond]["summary"]
                t_mean = parse_inf(summary["t_grok_mean"])
                t_std = parse_inf(summary["t_grok_std"])
                if t_mean < float("inf"):
                    k_vals.append(K)
                    means.append(t_mean)
                    stds.append(t_std if t_std < float("inf") else 0)

            if k_vals:
                ax.errorbar(k_vals, means, yerr=stds, marker=markers[cond],
                            color=colors[cond], label=labels[cond], capsize=3,
                            linewidth=1.5, markersize=5)

        ax.set_xlabel("K (clients)")
        ax.set_title(f"α = {alpha:.2f}")
        ax.set_xscale("log")
        ax.set_xticks(ks)
        ax.set_xticklabels(ks)

    axes[0].set_ylabel(r"$T_{grok}$ (steps)")
    axes[-1].legend(loc="upper left", fontsize=8)
    fig.suptitle("Exp 2: Grokking Time vs Number of Clients", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "exp2_t_grok_vs_K.png"), bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/exp2_t_grok_vs_K.png")


def plot_fl_vs_centralized(cells):
    """Scatter: FL T_grok vs Centralized T_grok for matched (alpha, K)."""
    fig, ax = plt.subplots(figsize=(7, 7))

    alphas = sorted(set(c["alpha"] for c in cells))
    alpha_colors = {a: plt.cm.viridis((a - min(alphas)) / (max(alphas) - min(alphas)))
                    for a in alphas}

    # Plot one scatter per alpha for clean legend
    for alpha in alphas:
        xs, ys, ks_list = [], [], []
        for c in cells:
            if c["alpha"] != alpha:
                continue
            t_cent = parse_inf(c["cent_full"]["summary"]["t_grok_mean"])
            t_fl = parse_inf(c["fl_iid"]["summary"]["t_grok_mean"])
            if t_cent == float("inf") or t_fl == float("inf"):
                continue
            xs.append(t_cent)
            ys.append(t_fl)
            ks_list.append(c["K"])

        if xs:
            ax.scatter(xs, ys, c=[alpha_colors[alpha]], s=50, zorder=3,
                       label=f"α={alpha:.2f}")
            for x, y, K in zip(xs, ys, ks_list):
                ax.annotate(f"K={K}", (x, y), fontsize=7,
                            textcoords="offset points", xytext=(5, 5))

    # Diagonal — use full data range
    all_vals = []
    for c in cells:
        for v in [parse_inf(c["cent_full"]["summary"]["t_grok_mean"]),
                  parse_inf(c["fl_iid"]["summary"]["t_grok_mean"])]:
            if v < float("inf"):
                all_vals.append(v)
    lo = min(all_vals) * 0.9
    hi = max(all_vals) * 1.05
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.3, label="FL = Centralized")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)

    ax.set_xlabel(r"Centralized $T_{grok}$")
    ax.set_ylabel(r"FL IID $T_{grok}$")
    ax.set_title("FL vs Centralized Grokking Time")
    ax.legend(fontsize=8)
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "exp2_fl_vs_centralized.png"), bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/exp2_fl_vs_centralized.png")


def plot_final_accuracy_heatmap(cells):
    """Heatmap of final test accuracy for FL IID condition."""
    alphas = sorted(set(c["alpha"] for c in cells))
    ks = sorted(set(c["K"] for c in cells))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    conditions = ["cent_full", "cent_reduced", "fl_iid"]
    cond_labels = ["Centralized (full)", "Centralized (reduced)", "FL IID"]

    for ax, cond, label in zip(axes, conditions, cond_labels):
        grid = np.full((len(alphas), len(ks)), np.nan)
        for c in cells:
            ai = alphas.index(c["alpha"])
            ki = ks.index(c["K"])
            grid[ai, ki] = c[cond]["summary"]["final_acc_mean"]

        im = ax.imshow(grid, aspect="auto", cmap="viridis", vmin=0, vmax=100,
                       origin="lower")
        ax.set_xticks(range(len(ks)))
        ax.set_xticklabels(ks)
        ax.set_yticks(range(len(alphas)))
        ax.set_yticklabels([f"{a:.2f}" for a in alphas])
        ax.set_xlabel("K (clients)")
        ax.set_title(label)

        for i in range(len(alphas)):
            for j in range(len(ks)):
                val = grid[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                            fontsize=8, color="white" if val < 50 else "black")
                else:
                    ax.text(j, i, "?", ha="center", va="center",
                            fontsize=9, color="gray")

    axes[0].set_ylabel(r"$\alpha$ (train fraction)")
    fig.colorbar(im, ax=axes, label="Final test accuracy (%)", shrink=0.8)
    fig.suptitle("Exp 2: Final Test Accuracy", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "exp2_final_accuracy_heatmap.png"), bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/exp2_final_accuracy_heatmap.png")


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cells = load_all_cells()
    print(f"Loaded {len(cells)} cells")

    plot_grokking_heatmap(cells)
    plot_t_grok_vs_K(cells)
    plot_fl_vs_centralized(cells)
    plot_final_accuracy_heatmap(cells)

    print("\nDone! All figures saved to", OUTPUT_DIR)