"""Plot Experiment 3 results."""

import json
import os
import glob
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

RESULTS_DIR = "results/exp3_heterogeneity"
OUTPUT_DIR = "results/exp3_heterogeneity/figures"


def parse_inf(val):
    if val == "inf" or val == float("inf"):
        return float("inf")
    return float(val)


def load_cells(pattern):
    cells = []
    for f in sorted(glob.glob(os.path.join(RESULTS_DIR, pattern))):
        with open(f) as fh:
            cells.append(json.load(fh))
    return cells


def plot_3a_phase_diagram(cells):
    """Heatmap: T_grok as function of alpha vs dirichlet_alpha."""
    alphas = sorted(set(c["alpha"] for c in cells))
    dir_alphas = sorted(set(c["dirichlet_alpha"] for c in cells))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Grokking time heatmap
    grid = np.full((len(alphas), len(dir_alphas)), np.nan)
    for c in cells:
        ai = alphas.index(c["alpha"])
        di = dir_alphas.index(c["dirichlet_alpha"])
        t = parse_inf(c["summary"]["t_grok_mean"])
        if t < float("inf"):
            grid[ai, di] = t

    im = axes[0].imshow(grid, aspect="auto", cmap="viridis_r", origin="lower")
    axes[0].set_xticks(range(len(dir_alphas)))
    axes[0].set_xticklabels([str(d) for d in dir_alphas], rotation=45)
    axes[0].set_yticks(range(len(alphas)))
    axes[0].set_yticklabels([f"{a:.2f}" for a in alphas])
    axes[0].set_xlabel(r"Dirichlet $\alpha_{dir}$ (→ IID)")
    axes[0].set_ylabel(r"Training fraction $\alpha$")
    axes[0].set_title(r"$T_{grok}$ (steps)")
    fig.colorbar(im, ax=axes[0], shrink=0.8)

    # Annotate
    for i in range(len(alphas)):
        for j in range(len(dir_alphas)):
            val = grid[i, j]
            if np.isnan(val):
                axes[0].text(j, i, "∞", ha="center", va="center",
                            fontsize=9, color="red")
            else:
                axes[0].text(j, i, f"{val:.0f}", ha="center", va="center",
                            fontsize=8, color="white" if val > np.nanmedian(grid) else "black")

    # Grokking success rate
    grid_grok = np.full((len(alphas), len(dir_alphas)), np.nan)
    for c in cells:
        ai = alphas.index(c["alpha"])
        di = dir_alphas.index(c["dirichlet_alpha"])
        grid_grok[ai, di] = c["summary"]["n_grokked"] / c["summary"]["n_seeds"]

    im2 = axes[1].imshow(grid_grok, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1,
                         origin="lower")
    axes[1].set_xticks(range(len(dir_alphas)))
    axes[1].set_xticklabels([str(d) for d in dir_alphas], rotation=45)
    axes[1].set_yticks(range(len(alphas)))
    axes[1].set_yticklabels([f"{a:.2f}" for a in alphas])
    axes[1].set_xlabel(r"Dirichlet $\alpha_{dir}$ (→ IID)")
    axes[1].set_title("Grokking success rate")
    fig.colorbar(im2, ax=axes[1], shrink=0.8)

    for i in range(len(alphas)):
        for j in range(len(dir_alphas)):
            val = grid_grok[i, j]
            if not np.isnan(val):
                axes[1].text(j, i, f"{val:.0%}", ha="center", va="center",
                            fontsize=9, color="black" if 0.3 < val < 0.7 else "white")

    fig.suptitle("Exp 3a: Dirichlet Heterogeneity Phase Diagram (K=10)", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "exp3a_phase_diagram.png"), bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/exp3a_phase_diagram.png")


def plot_3a_t_grok_vs_dir_alpha(cells):
    """Line plot: T_grok vs dirichlet alpha for each training fraction alpha."""
    alphas = sorted(set(c["alpha"] for c in cells))
    dir_alphas = sorted(set(c["dirichlet_alpha"] for c in cells))

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(alphas)))

    for alpha, color in zip(alphas, colors):
        xs, ys, errs = [], [], []
        for c in cells:
            if c["alpha"] != alpha:
                continue
            t = parse_inf(c["summary"]["t_grok_mean"])
            if t < float("inf"):
                xs.append(c["dirichlet_alpha"])
                ys.append(t)
                t_std = parse_inf(c["summary"]["t_grok_std"])
                errs.append(t_std if t_std < float("inf") else 0)
        if xs:
            ax.errorbar(xs, ys, yerr=errs, marker="o", color=color,
                        label=f"α={alpha:.2f}", capsize=3, linewidth=1.5)

    ax.set_xscale("log")
    ax.set_xlabel(r"Dirichlet $\alpha_{dir}$ (← non-IID | IID →)")
    ax.set_ylabel(r"$T_{grok}$ (steps)")
    ax.set_title("Exp 3a: Grokking Time vs Heterogeneity (K=10)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "exp3a_t_grok_vs_dir_alpha.png"), bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/exp3a_t_grok_vs_dir_alpha.png")


def plot_3a_k_validation(cells_k10, cells_k20):
    """Compare K=10 vs K=20 phase boundaries."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for cells, K, marker, ls in [(cells_k10, 10, "o", "-"), (cells_k20, 20, "s", "--")]:
        alphas = sorted(set(c["alpha"] for c in cells))
        dir_alphas = sorted(set(c["dirichlet_alpha"] for c in cells))
        colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(alphas)))

        for alpha, color in zip(alphas, colors):
            xs, ys = [], []
            for c in cells:
                if c["alpha"] != alpha:
                    continue
                t = parse_inf(c["summary"]["t_grok_mean"])
                if t < float("inf"):
                    xs.append(c["dirichlet_alpha"])
                    ys.append(t)
            if xs:
                ax.plot(xs, ys, marker=marker, linestyle=ls, color=color,
                        label=f"α={alpha:.2f} K={K}", linewidth=1.5, markersize=5)

    ax.set_xscale("log")
    ax.set_xlabel(r"Dirichlet $\alpha_{dir}$ (← non-IID | IID →)")
    ax.set_ylabel(r"$T_{grok}$ (steps)")
    ax.set_title("Exp 3a: K-validation (K=10 solid, K=20 dashed)")
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "exp3a_k_validation.png"), bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/exp3a_k_validation.png")


def plot_3b_structured(cells_3b, cells_3a=None):
    """Grouped bar chart: T_grok by partition type for each alpha."""
    alphas = sorted(set(c["alpha"] for c in cells_3b))
    partitions = sorted(set(c["partition"] for c in cells_3b))

    # Add IID from 3a (dir=1000) if available
    iid_data = {}
    if cells_3a:
        for c in cells_3a:
            if c["dirichlet_alpha"] == 1000.0:
                iid_data[c["alpha"]] = c

    all_parts = ["iid"] + [p for p in partitions if p != "iid"]
    colors = {"iid": "#2196F3", "operand": "#FF5722", "target": "#4CAF50"}
    labels = {"iid": "IID", "operand": "Operand", "target": "Target"}

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(alphas))
    width = 0.25

    for i, part in enumerate(all_parts):
        means, errs = [], []
        for alpha in alphas:
            if part == "iid" and alpha in iid_data:
                s = iid_data[alpha]["summary"]
            else:
                match = [c for c in cells_3b if c["alpha"] == alpha and c["partition"] == part]
                if not match:
                    means.append(0)
                    errs.append(0)
                    continue
                s = match[0]["summary"]

            t = parse_inf(s["t_grok_mean"])
            t_std = parse_inf(s["t_grok_std"])
            if t < float("inf"):
                means.append(t)
                errs.append(t_std if t_std < float("inf") else 0)
            else:
                means.append(0)
                errs.append(0)

        bars = ax.bar(x + i * width, means, width, yerr=errs, capsize=3,
                      color=colors.get(part, "gray"), label=labels.get(part, part))

        # Mark non-grokking bars
        for j, m in enumerate(means):
            if m == 0:
                ax.text(x[j] + i * width, 500, "∞", ha="center", va="bottom",
                        fontsize=10, color="red", fontweight="bold")

    ax.set_xticks(x + width)
    ax.set_xticklabels([f"α={a:.2f}" for a in alphas])
    ax.set_ylabel(r"$T_{grok}$ (steps)")
    ax.set_title("Exp 3b: Structured Partition Comparison (K=10)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "exp3b_structured_partitions.png"), bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/exp3b_structured_partitions.png")


def plot_3b_slowdown(cells_3b, cells_3a=None):
    """Relative slowdown vs IID for each partition type."""
    alphas = sorted(set(c["alpha"] for c in cells_3b))

    # Get IID baseline from 3a dir=1000
    iid_baseline = {}
    if cells_3a:
        for c in cells_3a:
            if c["dirichlet_alpha"] == 1000.0:
                t = parse_inf(c["summary"]["t_grok_mean"])
                if t < float("inf"):
                    iid_baseline[c["alpha"]] = t

    partitions = ["operand", "target"]
    colors = {"operand": "#FF5722", "target": "#4CAF50"}

    fig, ax = plt.subplots(figsize=(8, 5))

    for part in partitions:
        xs, ys = [], []
        for alpha in alphas:
            if alpha not in iid_baseline:
                continue
            match = [c for c in cells_3b if c["alpha"] == alpha and c["partition"] == part]
            if not match:
                continue
            t = parse_inf(match[0]["summary"]["t_grok_mean"])
            if t < float("inf"):
                xs.append(alpha)
                ys.append(t / iid_baseline[alpha])

        if xs:
            ax.plot(xs, ys, "o-", color=colors[part], label=part.capitalize(),
                    linewidth=2, markersize=8)

    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5, label="IID baseline")
    ax.set_xlabel(r"Training fraction $\alpha$")
    ax.set_ylabel(r"$T_{grok}$ / $T_{grok}^{IID}$ (slowdown ratio)")
    ax.set_title("Exp 3b: Partition Slowdown Relative to IID (K=10)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "exp3b_slowdown.png"), bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/exp3b_slowdown.png")


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    cells_3a = load_cells("exp3a_*_K10.json")
    cells_3a_kval = load_cells("exp3a_*_K20.json")
    cells_3b = load_cells("exp3b_*.json")

    print(f"Loaded: 3a={len(cells_3a)}, 3a_kval={len(cells_3a_kval)}, 3b={len(cells_3b)}")

    plot_3a_phase_diagram(cells_3a)
    plot_3a_t_grok_vs_dir_alpha(cells_3a)

    if cells_3a_kval:
        plot_3a_k_validation(cells_3a, cells_3a_kval)

    if cells_3b:
        plot_3b_structured(cells_3b, cells_3a)
        plot_3b_slowdown(cells_3b, cells_3a)

    print(f"\nDone! All figures saved to {OUTPUT_DIR}")
