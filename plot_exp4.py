"""Plot Experiment 4 results."""

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

RESULTS_DIR = "results/exp4_optimization"
OUTPUT_DIR = "results/exp4_optimization/figures"


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


def plot_4a_drift(cells):
    """T_grok vs E for IID and non-IID, one panel per alpha."""
    alphas = sorted(set(c["alpha"] for c in cells))

    fig, axes = plt.subplots(1, len(alphas), figsize=(5 * len(alphas), 5), sharey=True)
    if len(alphas) == 1:
        axes = [axes]

    colors = {"iid": "#2196F3", "noniid": "#FF5722"}
    markers = {"iid": "o", "noniid": "s"}
    labels = {"iid": "IID", "noniid": "Non-IID (Dir 0.1)"}

    for ax, alpha in zip(axes, alphas):
        for het in ["iid", "noniid"]:
            es, means, errs = [], [], []
            for c in cells:
                if c["alpha"] != alpha or c["heterogeneity"] != het:
                    continue
                t = parse_inf(c["summary"]["t_grok_mean"])
                t_std = parse_inf(c["summary"]["t_grok_std"])
                if t < float("inf"):
                    es.append(c["E"])
                    means.append(t)
                    errs.append(t_std if t_std < float("inf") else 0)
                else:
                    # Mark failed points at top of plot
                    es_fail = c["E"]
                    ax.scatter([es_fail], [52000], marker="x", color=colors[het],
                              s=80, zorder=5, linewidths=2)

            if es:
                ax.errorbar(es, means, yerr=errs, marker=markers[het],
                            color=colors[het], label=labels[het], capsize=4,
                            linewidth=2, markersize=7)

        ax.set_xlabel("Local epochs (E)")
        ax.set_title(f"α = {alpha:.2f}")
        ax.set_xticks([5, 10, 25, 50])

    axes[0].set_ylabel(r"$T_{grok}$ (steps)")
    axes[-1].legend(loc="upper left")
    fig.suptitle("Exp 4a: Drift Accumulation × Heterogeneity (K=10, S=50k)",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "exp4a_drift.png"), bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/exp4a_drift.png")


def plot_4a_slowdown(cells):
    """Slowdown ratio (non-IID / IID) vs E for each alpha."""
    alphas = sorted(set(c["alpha"] for c in cells))
    e_vals = sorted(set(c["E"] for c in cells))

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(alphas)))

    for alpha, color in zip(alphas, colors):
        es, ratios = [], []
        for E in e_vals:
            iid = [c for c in cells if c["alpha"] == alpha and c["E"] == E and c["heterogeneity"] == "iid"]
            noniid = [c for c in cells if c["alpha"] == alpha and c["E"] == E and c["heterogeneity"] == "noniid"]
            if not iid or not noniid:
                continue
            t_iid = parse_inf(iid[0]["summary"]["t_grok_mean"])
            t_noniid = parse_inf(noniid[0]["summary"]["t_grok_mean"])
            if t_iid < float("inf") and t_noniid < float("inf"):
                es.append(E)
                ratios.append(t_noniid / t_iid)

        if es:
            ax.plot(es, ratios, "o-", color=color, label=f"α={alpha:.2f}",
                    linewidth=2, markersize=7)

    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Local epochs (E)")
    ax.set_ylabel(r"$T_{grok}^{non-IID}$ / $T_{grok}^{IID}$")
    ax.set_title("Exp 4a: Non-IID Slowdown vs Drift (K=10)")
    ax.set_xticks([5, 10, 25, 50])
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "exp4a_slowdown.png"), bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/exp4a_slowdown.png")


def plot_4b_participation(cells):
    """T_grok vs participation fraction f, one panel per alpha."""
    alphas = sorted(set(c["alpha"] for c in cells))

    fig, axes = plt.subplots(1, len(alphas), figsize=(5 * len(alphas), 5), sharey=True)
    if len(alphas) == 1:
        axes = [axes]

    colors = {"iid": "#2196F3", "noniid": "#FF5722"}
    markers = {"iid": "o", "noniid": "s"}
    labels = {"iid": "IID", "noniid": "Non-IID (Dir 0.1)"}

    for ax, alpha in zip(axes, alphas):
        for het in ["iid", "noniid"]:
            fs, means, errs = [], [], []
            for c in cells:
                if c["alpha"] != alpha or c["heterogeneity"] != het:
                    continue
                t = parse_inf(c["summary"]["t_grok_mean"])
                t_std = parse_inf(c["summary"]["t_grok_std"])
                if t < float("inf"):
                    fs.append(c["f"])
                    means.append(t)
                    errs.append(t_std if t_std < float("inf") else 0)

            if fs:
                ax.errorbar(fs, means, yerr=errs, marker=markers[het],
                            color=colors[het], label=labels[het], capsize=4,
                            linewidth=2, markersize=7)

        ax.set_xlabel("Participation fraction (f)")
        ax.set_title(f"α = {alpha:.2f}")
        ax.set_xticks([0.2, 0.4, 0.6, 1.0])

    axes[0].set_ylabel(r"$T_{grok}$ (steps)")
    axes[-1].legend(loc="upper right")
    fig.suptitle("Exp 4b: Partial Participation × Heterogeneity (K=10, E=5)",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "exp4b_participation.png"), bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/exp4b_participation.png")


def plot_4c_compute_vs_comm(cells):
    """T_grok vs total compute S at fixed R=2000, one panel per alpha."""
    alphas = sorted(set(c["alpha"] for c in cells))

    fig, axes = plt.subplots(1, len(alphas), figsize=(5 * len(alphas), 5), sharey=True)
    if len(alphas) == 1:
        axes = [axes]

    for ax, alpha in zip(axes, alphas):
        es, ss, means, errs, grokked = [], [], [], [], []
        for c in cells:
            if c["alpha"] != alpha:
                continue
            t = parse_inf(c["summary"]["t_grok_mean"])
            t_std = parse_inf(c["summary"]["t_grok_std"])
            S = c["S"]
            E = c["E"]
            es.append(E)
            ss.append(S)
            if t < float("inf"):
                means.append(t)
                errs.append(t_std if t_std < float("inf") else 0)
                grokked.append(True)
            else:
                means.append(0)
                errs.append(0)
                grokked.append(False)

        # Sort by S
        order = np.argsort(ss)
        ss = [ss[i] for i in order]
        means = [means[i] for i in order]
        errs = [errs[i] for i in order]
        grokked = [grokked[i] for i in order]
        es = [es[i] for i in order]

        # Plot grokked points
        s_grok = [s for s, g in zip(ss, grokked) if g]
        m_grok = [m for m, g in zip(means, grokked) if g]
        e_grok = [e for e, g in zip(errs, grokked) if g]
        ax.errorbar(s_grok, m_grok, yerr=e_grok, marker="o", color="#2196F3",
                    capsize=4, linewidth=2, markersize=7)

        # Mark failed points
        s_fail = [s for s, g in zip(ss, grokked) if not g]
        for sf in s_fail:
            ax.scatter([sf], [52000], marker="x", color="red", s=80, zorder=5,
                      linewidths=2)

        # Annotate E values
        for s, m, g, E in zip(ss, means, grokked, es):
            y = m if g else 52000
            ax.annotate(f"E={E}", (s, y), fontsize=8,
                        textcoords="offset points", xytext=(5, 5))

        ax.set_xlabel("Total compute S = R × E")
        ax.set_title(f"α = {alpha:.2f}")

    axes[0].set_ylabel(r"$T_{grok}$ (steps)")
    fig.suptitle("Exp 4c: Compute vs Communication (K=10, R=2000, IID)",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "exp4c_compute_comm.png"), bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/exp4c_compute_comm.png")


def plot_4a_heatmap(cells):
    """Heatmap: grokking success across E × het for each alpha."""
    alphas = sorted(set(c["alpha"] for c in cells))
    e_vals = sorted(set(c["E"] for c in cells))
    hets = ["iid", "noniid"]
    het_labels = ["IID", "Non-IID"]

    fig, axes = plt.subplots(1, len(alphas), figsize=(4 * len(alphas), 3.5), sharey=True)
    if len(alphas) == 1:
        axes = [axes]

    for ax, alpha in zip(axes, alphas):
        grid = np.full((len(hets), len(e_vals)), np.nan)
        for c in cells:
            if c["alpha"] != alpha:
                continue
            hi = hets.index(c["heterogeneity"])
            ei = e_vals.index(c["E"])
            grid[hi, ei] = c["summary"]["n_grokked"] / c["summary"]["n_seeds"]

        im = ax.imshow(grid, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
        ax.set_xticks(range(len(e_vals)))
        ax.set_xticklabels(e_vals)
        ax.set_yticks(range(len(hets)))
        ax.set_yticklabels(het_labels)
        ax.set_xlabel("E (local epochs)")
        ax.set_title(f"α = {alpha:.2f}")

        for i in range(len(hets)):
            for j in range(len(e_vals)):
                val = grid[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                            fontsize=10, color="black" if 0.3 < val < 0.7 else "white")

    fig.colorbar(im, ax=axes, label="Grokking success rate", shrink=0.8)
    fig.suptitle("Exp 4a: Grokking Success (K=10, S=50k)", fontsize=14, y=1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "exp4a_heatmap.png"), bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/exp4a_heatmap.png")


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    cells_4a = load_cells("exp4a_*.json")
    cells_4b = load_cells("exp4b_*.json")
    cells_4c = load_cells("exp4c_*.json")

    print(f"Loaded: 4a={len(cells_4a)}, 4b={len(cells_4b)}, 4c={len(cells_4c)}")

    plot_4a_drift(cells_4a)
    plot_4a_slowdown(cells_4a)
    plot_4a_heatmap(cells_4a)
    plot_4b_participation(cells_4b)
    plot_4c_compute_vs_comm(cells_4c)

    print(f"\nDone! All figures saved to {OUTPUT_DIR}")
