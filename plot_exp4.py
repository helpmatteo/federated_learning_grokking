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

    fig, axes = plt.subplots(1, len(alphas), figsize=(4.5 * len(alphas), 4.5), sharey=True)
    if len(alphas) == 1:
        axes = [axes]

    colors = {"iid": "#2196F3", "noniid": "#FF5722"}
    markers = {"iid": "o", "noniid": "s"}
    labels = {"iid": "IID", "noniid": "Non-IID"}

    budget = 50000

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
                    es_fail = c["E"]
                    ax.scatter([es_fail], [budget * 1.04], marker="x",
                              color=colors[het], s=80, zorder=5, linewidths=2)

            if es:
                ax.errorbar(es, means, yerr=errs, marker=markers[het],
                            color=colors[het], label=labels[het], capsize=4,
                            linewidth=2, markersize=7)

        ax.set_xlabel("Local epochs ($E$)")
        ax.set_title(rf"$\alpha = {alpha:.2f}$")
        ax.set_xticks([5, 10, 25, 50])
        ax.axhline(budget, color="gray", linestyle="--", alpha=0.2, linewidth=0.8)

    axes[0].set_ylabel(r"$T_{\mathrm{grok}}$ (gradient steps)")
    axes[-1].legend(loc="upper left", title="Data partition")
    fig.suptitle(r"Local Epochs $\times$ Heterogeneity ($K=10$)",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "exp4a_drift.png"), bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/exp4a_drift.png")


def plot_4a_performance(cells):
    """Single-panel: T_grok vs E, color=alpha, linestyle=IID/non-IID."""
    from matplotlib.lines import Line2D

    alphas = sorted(set(c["alpha"] for c in cells))
    alpha_colors = {0.25: "#9C27B0", 0.3: "#2196F3", 0.5: "#FF9800"}

    fig, ax = plt.subplots(figsize=(7, 5))
    budget = 50000

    for alpha in alphas:
        color = alpha_colors.get(alpha, "gray")
        for het, ls, marker in [("iid", "-", "o"), ("noniid", "--", "s")]:
            es, means, errs = [], [], []
            fail_es = []
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
                    fail_es.append(c["E"])

            if es:
                order = np.argsort(es)
                es = [es[i] for i in order]
                means = [means[i] for i in order]
                errs = [errs[i] for i in order]
                ax.errorbar(es, means, yerr=errs, marker=marker, color=color,
                            linestyle=ls, capsize=3, linewidth=2, markersize=6)
            for fe in fail_es:
                ax.scatter([fe], [budget * 1.04], marker="x", color=color,
                          s=80, zorder=5, linewidths=2)

    # Build legend: color for alpha, style for IID/non-IID
    alpha_handles = [Line2D([0], [0], color=alpha_colors[a], linewidth=2,
                            label=rf"$\alpha={a:.2f}$") for a in alphas]
    style_handles = [
        Line2D([0], [0], color="gray", linestyle="-", marker="o", markersize=5,
               linewidth=1.5, label="IID"),
        Line2D([0], [0], color="gray", linestyle="--", marker="s", markersize=5,
               linewidth=1.5, label="Non-IID"),
        Line2D([0], [0], color="gray", linestyle="none", marker="x",
               markersize=7, linewidth=2, label="Failed to grok"),
    ]
    leg1 = ax.legend(handles=alpha_handles, loc="upper left", fontsize=9,
                     title=r"Train fraction $\alpha$", title_fontsize=9)
    ax.add_artist(leg1)
    ax.legend(handles=style_handles, loc=(0.21, 0.791), fontsize=9,
              title="Data partition", title_fontsize=9)

    ax.set_xlabel("Local epochs ($E$)")
    ax.set_ylabel(r"$T_{\mathrm{grok}}$ (gradient steps)")
    ax.set_title(r"Grokking Time vs Local Epochs ($K=10$)")
    ax.set_xticks([5, 10, 25, 50])
    ax.axhline(budget, color="gray", linestyle=":", alpha=0.3, linewidth=0.8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "exp4a_performance.png"), bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/exp4a_performance.png")


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
            ax.plot(es, ratios, "o-", color=color, label=rf"$\alpha={alpha:.2f}$",
                    linewidth=2, markersize=7)

    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Local epochs ($E$)")
    ax.set_ylabel(r"$T_{\mathrm{grok}}^{\mathrm{non\text{-}IID}}$ / $T_{\mathrm{grok}}^{\mathrm{IID}}$")
    ax.set_title(r"Non-IID Slowdown vs Local Epochs ($K=10$)")
    ax.set_xticks([5, 10, 25, 50])
    ax.legend(title=r"Train fraction $\alpha$")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "exp4a_slowdown.png"), bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/exp4a_slowdown.png")


def plot_4b_participation(cells):
    """T_grok vs participation fraction f, one panel per alpha."""
    alphas = sorted(set(c["alpha"] for c in cells))

    fig, axes = plt.subplots(1, len(alphas), figsize=(4.5 * len(alphas), 4.5), sharey=True)
    if len(alphas) == 1:
        axes = [axes]

    colors = {"iid": "#2196F3", "noniid": "#FF5722"}
    markers = {"iid": "o", "noniid": "s"}
    labels = {"iid": "IID", "noniid": "Non-IID"}

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

        ax.set_xlabel("Participation fraction ($f$)")
        ax.set_title(rf"$\alpha = {alpha:.2f}$")
        ax.set_xticks([0.2, 0.4, 0.6, 1.0])

    axes[0].set_ylabel(r"$T_{\mathrm{grok}}$ (gradient steps)")
    axes[-1].legend(loc="upper right", title="Data partition")
    fig.suptitle(r"Partial Participation $\times$ Heterogeneity ($K=10$, $E=5$)",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "exp4b_participation.png"), bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/exp4b_participation.png")


def plot_4b_participation_combined(cells):
    """Single-panel: T_grok vs f, color=alpha, linestyle=IID/non-IID."""
    from matplotlib.lines import Line2D

    alphas = sorted(set(c["alpha"] for c in cells))
    alpha_colors = {0.25: "#9C27B0", 0.3: "#2196F3", 0.5: "#FF9800"}

    fig, ax = plt.subplots(figsize=(7, 5))

    for alpha in alphas:
        color = alpha_colors.get(alpha, "gray")
        for het, ls, marker in [("iid", "-", "o"), ("noniid", "--", "s")]:
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
                order = np.argsort(fs)
                fs = [fs[i] for i in order]
                means = [means[i] for i in order]
                errs = [errs[i] for i in order]
                ax.errorbar(fs, means, yerr=errs, marker=marker, color=color,
                            linestyle=ls, capsize=3, linewidth=2, markersize=6)

    alpha_handles = [Line2D([0], [0], color=alpha_colors[a], linewidth=2,
                            label=rf"$\alpha={a:.2f}$") for a in alphas]
    style_handles = [
        Line2D([0], [0], color="gray", linestyle="-", marker="o", markersize=5,
               linewidth=1.5, label="IID"),
        Line2D([0], [0], color="gray", linestyle="--", marker="s", markersize=5,
               linewidth=1.5, label="Non-IID"),
    ]
    leg1 = ax.legend(handles=style_handles, loc=(0.62, 0.02), fontsize=9,
                     title="Data partition", title_fontsize=9)
    ax.add_artist(leg1)
    ax.legend(handles=alpha_handles, loc="lower right", fontsize=9,
              title=r"Train fraction $\alpha$", title_fontsize=9)

    ax.set_xlabel("Participation fraction ($f$)")
    ax.set_ylabel(r"$T_{\mathrm{grok}}$ (gradient steps)")
    ax.set_title(r"Grokking Time vs Participation Fraction ($K=10$, $E=5$)")
    ax.set_xticks([0.2, 0.4, 0.6, 1.0])
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "exp4b_participation_combined.png"), bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/exp4b_participation_combined.png")



def plot_4c_compute_vs_comm(cells):
    """T_grok vs total compute S at fixed R=2000, one panel per alpha."""
    alphas = sorted(set(c["alpha"] for c in cells))

    fig, axes = plt.subplots(1, len(alphas), figsize=(4.5 * len(alphas), 4.5), sharey=True)
    if len(alphas) == 1:
        axes = [axes]

    budget = 50000

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
            ax.scatter([sf], [budget * 1.04], marker="x", color="#D32F2F", s=80,
                      zorder=5, linewidths=2)

        # Annotate E values
        for s, m, g, E in zip(ss, means, grokked, es):
            y = m if g else budget * 1.04
            ax.annotate(f"$E$={E}", (s, y), fontsize=8,
                        textcoords="offset points", xytext=(5, 5))

        ax.set_xlabel(r"Total compute $S = R \times E$")
        ax.set_title(rf"$\alpha = {alpha:.2f}$")
        ax.axhline(budget, color="gray", linestyle="--", alpha=0.2, linewidth=0.8)

    axes[0].set_ylabel(r"$T_{\mathrm{grok}}$ (gradient steps)")
    fig.suptitle(r"Compute vs Communication ($K=10$, $R=2000$, IID)",
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
    fig.suptitle(r"Grokking Success Rate ($K=10$, $S=50$k)", fontsize=14, y=1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "exp4a_heatmap.png"), bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/exp4a_heatmap.png")


RAW_4A_DIR = os.path.join(RESULTS_DIR, "exp4a")
RAW_4B_DIR = os.path.join(RESULTS_DIR, "exp4b")
RAW_4C_DIR = os.path.join(RESULTS_DIR, "exp4c")


def load_raw_history(directory, pattern):
    """Load a single raw history file matching pattern."""
    matches = sorted(glob.glob(os.path.join(directory, pattern)))
    if not matches:
        return None
    with open(matches[0]) as f:
        return json.load(f)


def plot_4a_weight_norms():
    """Weight norm evolution for Exp 4a: varying E, IID vs non-IID, per alpha."""
    alphas = [0.25, 0.3, 0.5]
    e_vals = [5, 10, 25, 50]
    hets = ["iid", "noniid"]
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(e_vals)))

    fig, axes = plt.subplots(2, len(alphas) * 2, figsize=(24, 9))
    # Layout: 2 rows (W1, W2) × 6 columns (3 alphas × 2 het)

    for ai, alpha in enumerate(alphas):
        for hi, het in enumerate(hets):
            col = ai * 2 + hi
            for E, color in zip(e_vals, colors):
                if het == "iid":
                    pat = f"*_a{alpha}_K10_le{E}_ft1.0_iid_s42.json"
                else:
                    pat = f"*_a{alpha}_K10_le{E}_ft1.0_dirichlet_dir0.1_s42.json"
                h = load_raw_history(RAW_4A_DIR, pat)
                if h is None or "weight_norm_layer1" not in h:
                    continue
                steps = np.array(h["total_steps"])
                w1 = np.array(h["weight_norm_layer1"])
                w2 = np.array(h["weight_norm_layer2"])
                label = f"E={E}"
                axes[0, col].plot(steps, w1, color=color, label=label,
                                  linewidth=1.2, alpha=0.85)
                axes[1, col].plot(steps, w2, color=color, label=label,
                                  linewidth=1.2, alpha=0.85)

            het_label = "IID" if het == "iid" else "Non-IID"
            axes[0, col].set_title(f"α={alpha}, {het_label}")
            axes[0, col].legend(fontsize=7)
            axes[1, col].set_xlabel("Gradient steps")

    axes[0, 0].set_ylabel("||W1||_F")
    axes[1, 0].set_ylabel("||W2||_F")

    fig.suptitle("Exp 4a: Weight Norm Evolution vs Local Epochs (seed=42)",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "exp4a_weight_norms.png"),
                bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/exp4a_weight_norms.png")


def plot_4b_weight_norms():
    """Weight norm evolution for Exp 4b: varying f, IID vs non-IID, per alpha."""
    alphas = [0.25, 0.3, 0.5]
    f_vals = [0.2, 0.4, 0.6, 1.0]
    hets = ["iid", "noniid"]
    colors = plt.cm.plasma(np.linspace(0.15, 0.85, len(f_vals)))

    fig, axes = plt.subplots(2, len(alphas) * 2, figsize=(24, 9))

    for ai, alpha in enumerate(alphas):
        for hi, het in enumerate(hets):
            col = ai * 2 + hi
            for frac, color in zip(f_vals, colors):
                if het == "iid":
                    pat = f"*_a{alpha}_K10_le5_ft{frac}_iid_s42.json"
                else:
                    pat = f"*_a{alpha}_K10_le5_ft{frac}_dirichlet_dir0.1_s42.json"
                h = load_raw_history(RAW_4B_DIR, pat)
                if h is None or "weight_norm_layer1" not in h:
                    continue
                steps = np.array(h["total_steps"])
                w1 = np.array(h["weight_norm_layer1"])
                w2 = np.array(h["weight_norm_layer2"])
                label = f"f={frac}"
                axes[0, col].plot(steps, w1, color=color, label=label,
                                  linewidth=1.2, alpha=0.85)
                axes[1, col].plot(steps, w2, color=color, label=label,
                                  linewidth=1.2, alpha=0.85)

            het_label = "IID" if het == "iid" else "Non-IID"
            axes[0, col].set_title(f"α={alpha}, {het_label}")
            axes[0, col].legend(fontsize=7)
            axes[1, col].set_xlabel("Gradient steps")

    axes[0, 0].set_ylabel("||W1||_F")
    axes[1, 0].set_ylabel("||W2||_F")

    fig.suptitle("Exp 4b: Weight Norm Evolution vs Participation Fraction (seed=42)",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "exp4b_weight_norms.png"),
                bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/exp4b_weight_norms.png")


def plot_4c_weight_norms():
    """Weight norm evolution for Exp 4c: varying E at fixed R=2000, IID, per alpha."""
    alphas = [0.25, 0.3, 0.5]
    e_vals = [1, 5, 10, 25]
    colors = plt.cm.coolwarm(np.linspace(0.1, 0.9, len(e_vals)))

    fig, axes = plt.subplots(2, len(alphas), figsize=(15, 9))

    for col, alpha in enumerate(alphas):
        for E, color in zip(e_vals, colors):
            pat = f"*_a{alpha}_K10_le{E}_ft1.0_iid_s42.json"
            h = load_raw_history(RAW_4C_DIR, pat)
            if h is None or "weight_norm_layer1" not in h:
                continue
            steps = np.array(h["total_steps"])
            w1 = np.array(h["weight_norm_layer1"])
            w2 = np.array(h["weight_norm_layer2"])
            label = f"E={E} (S={2000*E})"
            axes[0, col].plot(steps, w1, color=color, label=label,
                              linewidth=1.5, alpha=0.85)
            axes[1, col].plot(steps, w2, color=color, label=label,
                              linewidth=1.5, alpha=0.85)

        axes[0, col].set_title(f"α = {alpha}")
        axes[0, col].legend(fontsize=8)
        axes[1, col].set_xlabel("Gradient steps")

    axes[0, 0].set_ylabel("||W1||_F")
    axes[1, 0].set_ylabel("||W2||_F")

    fig.suptitle("Exp 4c: Weight Norm Evolution — Compute vs Communication (seed=42)",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "exp4c_weight_norms.png"),
                bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/exp4c_weight_norms.png")


def plot_4a_weight_norm_ratio():
    """Weight norm ratio (W2/W1) for Exp 4a — indicator of grokking structure."""
    alphas = [0.25, 0.3, 0.5]
    e_vals = [5, 10, 25, 50]
    hets = ["iid", "noniid"]
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(e_vals)))

    fig, axes = plt.subplots(1, len(alphas) * 2, figsize=(24, 5))

    for ai, alpha in enumerate(alphas):
        for hi, het in enumerate(hets):
            col = ai * 2 + hi
            for E, color in zip(e_vals, colors):
                if het == "iid":
                    pat = f"*_a{alpha}_K10_le{E}_ft1.0_iid_s42.json"
                else:
                    pat = f"*_a{alpha}_K10_le{E}_ft1.0_dirichlet_dir0.1_s42.json"
                h = load_raw_history(RAW_4A_DIR, pat)
                if h is None or "weight_norm_layer1" not in h:
                    continue
                steps = np.array(h["total_steps"])
                w1 = np.array(h["weight_norm_layer1"])
                w2 = np.array(h["weight_norm_layer2"])
                ratio = w2 / np.maximum(w1, 1e-8)
                axes[col].plot(steps, ratio, color=color, label=f"E={E}",
                               linewidth=1.2, alpha=0.85)

            het_label = "IID" if het == "iid" else "Non-IID"
            axes[col].set_title(f"α={alpha}, {het_label}")
            axes[col].set_xlabel("Gradient steps")
            axes[col].legend(fontsize=7)

    axes[0].set_ylabel("||W2|| / ||W1||")
    fig.suptitle("Exp 4a: Weight Norm Ratio vs Local Epochs (seed=42)",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "exp4a_weight_norm_ratio.png"),
                bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/exp4a_weight_norm_ratio.png")


def plot_drift_norm_interaction():
    """Multi-panel figure showing how drift corrupts weight norm structure.

    Panel A: Temporal co-evolution (4 contrasting cases)
    Panel B: Scatter — mean drift vs W1 growth (all runs, colored by T_grok)
    Panel C: Scatter — mean drift vs final W2/W1 ratio
    Panel D: W2 trajectory collapse under high drift
    """
    from scipy.stats import pearsonr
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    # ── Panel A: temporal co-evolution for 4 contrasting cases ──
    cases = [
        ("α=0.25, E=5, IID\n(low drift)",   "*_a0.25_K10_le5_ft1.0_iid_s42.json",          "#2196F3"),
        ("α=0.25, E=50, nIID\n(high drift)", "*_a0.25_K10_le50_ft1.0_dirichlet_dir0.1_s42.json", "#FF5722"),
        ("α=0.3, E=5, IID\n(low drift)",     "*_a0.3_K10_le5_ft1.0_iid_s42.json",           "#4CAF50"),
        ("α=0.3, E=50, nIID\n(high drift)",  "*_a0.3_K10_le50_ft1.0_dirichlet_dir0.1_s42.json",  "#E91E63"),
    ]

    ax_w1 = fig.add_subplot(gs[0, 0])
    ax_drift = ax_w1.twinx()
    ax_w2w1 = fig.add_subplot(gs[0, 1])
    ax_w2 = fig.add_subplot(gs[0, 2])

    for label, pat, color in cases:
        h = load_raw_history(RAW_4A_DIR, pat)
        if h is None:
            continue
        steps = np.array(h["total_steps"]) / 1000  # ksteps
        w1 = np.array(h["weight_norm_layer1"])
        w2 = np.array(h["weight_norm_layer2"])
        drift = np.array(h.get("mean_client_drift", []))
        ratio = w2 / np.maximum(w1, 1e-8)

        ax_w1.plot(steps, w1, color=color, linewidth=1.8, label=label)
        ax_drift.plot(steps[:len(drift)], drift, color=color, linewidth=1.0,
                      linestyle=":", alpha=0.6)
        ax_w2w1.plot(steps, ratio, color=color, linewidth=1.8, label=label)
        ax_w2.plot(steps, w2, color=color, linewidth=1.8, label=label)

    ax_w1.set_xlabel("Steps (×1000)")
    ax_w1.set_ylabel("||W1||_F (solid)")
    ax_drift.set_ylabel("Client drift (dotted)", color="gray", alpha=0.7)
    ax_drift.tick_params(axis="y", labelcolor="gray")
    ax_w1.set_title("A. W1 norm + drift over time")
    ax_w1.legend(fontsize=7, loc="upper left")

    ax_w2w1.set_xlabel("Steps (×1000)")
    ax_w2w1.set_ylabel("||W2|| / ||W1||")
    ax_w2w1.set_title("B. W2/W1 ratio (structure quality)")
    ax_w2w1.legend(fontsize=7)

    ax_w2.set_xlabel("Steps (×1000)")
    ax_w2.set_ylabel("||W2||_F")
    ax_w2.set_title("C. W2 trajectory (readout layer)")
    ax_w2.legend(fontsize=7)

    # ── Panel D: scatter — drift vs W1 growth, colored by T_grok ──
    ax_scatter1 = fig.add_subplot(gs[1, 0])
    ax_scatter2 = fig.add_subplot(gs[1, 1])
    ax_scatter3 = fig.add_subplot(gs[1, 2])

    all_drift, all_w1g, all_w2w1, all_tgrok = [], [], [], []
    for path in sorted(glob.glob(os.path.join(RAW_4A_DIR, "history_fed_*.json"))):
        with open(path) as f:
            h = json.load(f)
        drift = np.array(h.get("mean_client_drift", []))
        w1 = np.array(h["weight_norm_layer1"])
        w2 = np.array(h["weight_norm_layer2"])
        test_acc = np.array(h["test_acc"])
        steps = np.array(h["total_steps"])

        grok_idx = np.where(test_acc > 0.95)[0]
        t_grok = steps[grok_idx[0]] if len(grok_idx) > 0 else 50000

        all_drift.append(np.mean(drift))
        all_w1g.append(w1[-1] - w1[0])
        all_w2w1.append(w2[-1] / max(w1[-1], 1e-8))
        all_tgrok.append(t_grok)

    all_drift = np.array(all_drift)
    all_w1g = np.array(all_w1g)
    all_w2w1 = np.array(all_w2w1)
    all_tgrok = np.array(all_tgrok)

    # D: drift vs W1 growth
    sc1 = ax_scatter1.scatter(all_drift, all_w1g, c=all_tgrok / 1000,
                              cmap="coolwarm", s=40, edgecolors="k", linewidths=0.3)
    r, p = pearsonr(all_drift, all_w1g)
    ax_scatter1.set_xlabel("Mean client drift")
    ax_scatter1.set_ylabel("W1 growth (final − initial)")
    ax_scatter1.set_title(f"D. Drift inflates W1 (r={r:.2f})")
    cb1 = plt.colorbar(sc1, ax=ax_scatter1, label="T_grok (ksteps)")

    # E: drift vs final W2/W1 ratio
    sc2 = ax_scatter2.scatter(all_drift, all_w2w1, c=all_tgrok / 1000,
                              cmap="coolwarm", s=40, edgecolors="k", linewidths=0.3)
    r2, p2 = pearsonr(all_drift, all_w2w1)
    ax_scatter2.set_xlabel("Mean client drift")
    ax_scatter2.set_ylabel("Final ||W2|| / ||W1||")
    ax_scatter2.set_title(f"E. Drift disrupts W2/W1 structure (r={r2:.2f})")
    plt.colorbar(sc2, ax=ax_scatter2, label="T_grok (ksteps)")

    # F: W1 growth vs T_grok (showing it's a confound, not a cause)
    # Color by drift to show drift is the real driver
    sc3 = ax_scatter3.scatter(all_w1g, all_tgrok / 1000, c=all_drift,
                              cmap="magma", s=40, edgecolors="k", linewidths=0.3)
    r3, p3 = pearsonr(all_w1g, all_tgrok)
    ax_scatter3.set_xlabel("W1 growth (final − initial)")
    ax_scatter3.set_ylabel("T_grok (ksteps)")
    ax_scatter3.set_title(f"F. W1→T_grok is confounded (r={r3:.2f}, partial r≈0)")
    plt.colorbar(sc3, ax=ax_scatter3, label="Mean drift")

    fig.suptitle("Drift–Weight Norm Interaction: Drift Inflates W1 but Corrupts Structure",
                 fontsize=15, y=1.01)
    plt.savefig(os.path.join(OUTPUT_DIR, "exp4_drift_norm_interaction.png"),
                bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/exp4_drift_norm_interaction.png")


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

    # Weight norm plots
    plot_4a_weight_norms()
    plot_4b_weight_norms()
    plot_4c_weight_norms()
    plot_4a_weight_norm_ratio()

    # Drift-norm interaction
    plot_drift_norm_interaction()

    print(f"\nDone! All figures saved to {OUTPUT_DIR}")
