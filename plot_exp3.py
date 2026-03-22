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


import re

RAW_3A_DIR = os.path.join(RESULTS_DIR, "exp3a")
RAW_3B_DIR = os.path.join(RESULTS_DIR, "exp3b")


def load_raw_history(subdir, pattern_parts):
    """Load a raw history JSON from a subdirectory matching pattern parts."""
    search = os.path.join(subdir, f"*{'*'.join(pattern_parts)}*.json")
    matches = sorted(glob.glob(search))
    if not matches:
        return None
    with open(matches[0]) as f:
        return json.load(f)


def load_raw_histories_3a(alpha, dir_alpha, seed=42):
    """Load a single exp3a raw history by parameters."""
    pattern = f"*_a{alpha}_K10_*_dir{dir_alpha}_s{seed}.json"
    matches = sorted(glob.glob(os.path.join(RAW_3A_DIR, pattern)))
    if not matches:
        return None
    with open(matches[0]) as f:
        return json.load(f)


def load_raw_histories_3b(alpha, partition, seed=42):
    """Load a single exp3b raw history by parameters."""
    pattern = f"*_a{alpha}_K10_*_{partition}_s{seed}.json"
    matches = sorted(glob.glob(os.path.join(RAW_3B_DIR, pattern)))
    if not matches:
        return None
    with open(matches[0]) as f:
        return json.load(f)


def plot_test_acc_trajectories():
    """Plot 1: Test accuracy curves overlaid by heterogeneity level at fixed alpha."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    focus_alphas = [0.25, 0.3, 0.5]
    dir_alphas = [0.01, 0.1, 1.0, 1000.0]
    colors = plt.cm.plasma(np.linspace(0.1, 0.85, len(dir_alphas)))

    for ax, alpha in zip(axes, focus_alphas):
        for dir_alpha, color in zip(dir_alphas, colors):
            h = load_raw_histories_3a(alpha, dir_alpha)
            if h is None:
                continue
            steps = np.array(h["total_steps"])
            acc = np.array(h["test_acc"])
            label_str = f"dir={dir_alpha}"
            if dir_alpha >= 1000:
                label_str = "IID (dir=1000)"
            ax.plot(steps, acc, color=color, label=label_str,
                    linewidth=1.2, alpha=0.9)

        ax.set_xlabel("Gradient steps")
        ax.set_ylabel("Test accuracy (%)")
        ax.set_title(f"α = {alpha}")
        ax.set_ylim(-5, 105)
        ax.legend(fontsize=8)
        ax.axhline(95, color="gray", linestyle=":", alpha=0.4, linewidth=0.8)

    fig.suptitle("Test Accuracy Trajectories by Heterogeneity Level (seed=42)",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "exp3_test_acc_trajectories.png"),
                bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/exp3_test_acc_trajectories.png")


def plot_client_drift_dynamics():
    """Plot 2: Client drift and weight divergence around grokking transition."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    focus_alphas = [0.25, 0.3, 0.5]
    dir_alphas = [0.01, 0.1, 1.0, 1000.0]
    colors = plt.cm.plasma(np.linspace(0.1, 0.85, len(dir_alphas)))

    for col, alpha in enumerate(focus_alphas):
        for dir_alpha, color in zip(dir_alphas, colors):
            h = load_raw_histories_3a(alpha, dir_alpha)
            if h is None:
                continue
            steps = np.array(h["total_steps"])
            label_str = f"dir={dir_alpha}" if dir_alpha < 1000 else "IID"

            # Top row: mean client drift
            drift = np.array(h["mean_client_drift"])
            axes[0, col].plot(steps, drift, color=color, label=label_str,
                              linewidth=1.0, alpha=0.85)

            # Bottom row: client weight divergence
            div = np.array(h["client_weight_divergence"])
            axes[1, col].plot(steps, div, color=color, label=label_str,
                              linewidth=1.0, alpha=0.85)

        # Mark approximate grokking region with a vertical band
        for dir_alpha_mark in [0.01, 1000.0]:
            cell_pattern = f"exp3a_a{alpha}_dir{dir_alpha_mark}_K10.json"
            matches = sorted(glob.glob(os.path.join(RESULTS_DIR, cell_pattern)))
            if matches:
                with open(matches[0]) as f:
                    cell = json.load(f)
                t = parse_inf(cell["summary"]["t_grok_mean"])
                if t < float("inf"):
                    for row in range(2):
                        axes[row, col].axvline(t, color="gray", linestyle="--",
                                               alpha=0.3, linewidth=0.8)

        axes[0, col].set_title(f"α = {alpha}")
        axes[0, col].set_ylabel("Mean client drift")
        axes[1, col].set_ylabel("Client weight divergence")
        axes[1, col].set_xlabel("Gradient steps")
        axes[0, col].legend(fontsize=7)

    fig.suptitle("Client Drift Dynamics Across Heterogeneity Levels (seed=42)",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "exp3_client_drift_dynamics.png"),
                bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/exp3_client_drift_dynamics.png")


def plot_ipr_trajectories():
    """Plot 3: IPR evolution across heterogeneity levels."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    focus_alphas = [0.25, 0.3, 0.5]
    dir_alphas = [0.01, 0.1, 1.0, 1000.0]
    colors = plt.cm.plasma(np.linspace(0.1, 0.85, len(dir_alphas)))

    for ax, alpha in zip(axes, focus_alphas):
        for dir_alpha, color in zip(dir_alphas, colors):
            h = load_raw_histories_3a(alpha, dir_alpha)
            if h is None:
                continue
            steps = np.array(h["total_steps"])
            ipr = np.array(h["ipr"])
            label_str = f"dir={dir_alpha}" if dir_alpha < 1000 else "IID"
            ax.plot(steps, ipr, color=color, label=label_str,
                    linewidth=1.2, alpha=0.9)

        ax.set_xlabel("Gradient steps")
        ax.set_ylabel("IPR")
        ax.set_title(f"α = {alpha}")
        ax.legend(fontsize=8)

    fig.suptitle("IPR Trajectories by Heterogeneity Level (seed=42)",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "exp3_ipr_trajectories.png"),
                bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/exp3_ipr_trajectories.png")


def plot_grokking_sharpness(cells):
    """Plot 4: Grokking transition sharpness (T_grok - T_50) vs heterogeneity."""
    alphas = sorted(set(c["alpha"] for c in cells))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(alphas)))

    # Left: T_grok - T_50 vs dir_alpha (line plot)
    for alpha, color in zip(alphas, colors):
        xs, ys, errs = [], [], []
        for c in cells:
            if c["alpha"] != alpha:
                continue
            t_grok = parse_inf(c["summary"]["t_grok_mean"])
            t_50 = parse_inf(c["summary"]["t_50_mean"])
            if t_grok < float("inf") and t_50 < float("inf"):
                gap = t_grok - t_50
                xs.append(c["dirichlet_alpha"])
                ys.append(gap)
                # Propagate std: approximate with t_grok_std
                t_std = parse_inf(c["summary"]["t_grok_std"])
                t50_std = parse_inf(c["summary"]["t_50_std"])
                if t_std < float("inf") and t50_std < float("inf"):
                    errs.append(np.sqrt(t_std**2 + t50_std**2))
                else:
                    errs.append(0)
        if xs:
            axes[0].errorbar(xs, ys, yerr=errs, marker="o", color=color,
                             label=f"α={alpha:.2f}", capsize=3, linewidth=1.5)

    axes[0].set_xscale("log")
    axes[0].set_xlabel(r"Dirichlet $\alpha_{dir}$ (← non-IID | IID →)")
    axes[0].set_ylabel(r"$T_{grok} - T_{50}$ (steps)")
    axes[0].set_title("Transition width vs heterogeneity")
    axes[0].legend(fontsize=8)

    # Right: T_grok - T_50 as fraction of T_grok (normalized sharpness)
    for alpha, color in zip(alphas, colors):
        xs, ys = [], []
        for c in cells:
            if c["alpha"] != alpha:
                continue
            t_grok = parse_inf(c["summary"]["t_grok_mean"])
            t_50 = parse_inf(c["summary"]["t_50_mean"])
            if t_grok < float("inf") and t_50 < float("inf") and t_grok > 0:
                xs.append(c["dirichlet_alpha"])
                ys.append((t_grok - t_50) / t_grok)
        if xs:
            axes[1].plot(xs, ys, "o-", color=color, label=f"α={alpha:.2f}",
                         linewidth=1.5)

    axes[1].set_xscale("log")
    axes[1].set_xlabel(r"Dirichlet $\alpha_{dir}$ (← non-IID | IID →)")
    axes[1].set_ylabel(r"$(T_{grok} - T_{50}) / T_{grok}$")
    axes[1].set_title("Normalized transition sharpness")
    axes[1].legend(fontsize=8)

    fig.suptitle("Exp 3a: Grokking Transition Sharpness (K=10)", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "exp3_grokking_sharpness.png"),
                bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/exp3_grokking_sharpness.png")


def plot_weight_norm_evolution():
    """Plot 5: Weight norm evolution across heterogeneity levels."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    focus_alphas = [0.25, 0.3, 0.5]
    dir_alphas = [0.01, 0.1, 1.0, 1000.0]
    colors = plt.cm.plasma(np.linspace(0.1, 0.85, len(dir_alphas)))

    for col, alpha in enumerate(focus_alphas):
        for dir_alpha, color in zip(dir_alphas, colors):
            h = load_raw_histories_3a(alpha, dir_alpha)
            if h is None:
                continue
            steps = np.array(h["total_steps"])
            label_str = f"dir={dir_alpha}" if dir_alpha < 1000 else "IID"

            # Top row: layer 1 weight norm
            w1 = np.array(h["weight_norm_layer1"])
            axes[0, col].plot(steps, w1, color=color, label=label_str,
                              linewidth=1.0, alpha=0.85)

            # Bottom row: layer 2 weight norm
            w2 = np.array(h["weight_norm_layer2"])
            axes[1, col].plot(steps, w2, color=color, label=label_str,
                              linewidth=1.0, alpha=0.85)

        axes[0, col].set_title(f"α = {alpha}")
        axes[0, col].set_ylabel("Layer 1 weight norm")
        axes[1, col].set_ylabel("Layer 2 weight norm")
        axes[1, col].set_xlabel("Gradient steps")
        axes[0, col].legend(fontsize=7)

    fig.suptitle("Weight Norm Evolution Across Heterogeneity Levels (seed=42)",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "exp3_weight_norm_evolution.png"),
                bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/exp3_weight_norm_evolution.png")


def plot_loss_and_generalization_gap():
    """Plot 6: Train/test loss + generalization gap (test_loss - train_loss)."""
    fig, axes = plt.subplots(3, 3, figsize=(18, 13))
    focus_alphas = [0.25, 0.3, 0.5]
    dir_alphas = [0.01, 0.1, 1.0, 1000.0]
    colors = plt.cm.plasma(np.linspace(0.1, 0.85, len(dir_alphas)))

    for col, alpha in enumerate(focus_alphas):
        for dir_alpha, color in zip(dir_alphas, colors):
            h = load_raw_histories_3a(alpha, dir_alpha)
            if h is None:
                continue
            steps = np.array(h["total_steps"])
            train_loss = np.array(h["train_loss"])
            test_loss = np.array(h["test_loss"])
            gap = test_loss - train_loss
            label_str = f"dir={dir_alpha}" if dir_alpha < 1000 else "IID"

            axes[0, col].plot(steps, train_loss, color=color, label=label_str,
                              linewidth=1.0, alpha=0.85)
            axes[1, col].plot(steps, test_loss, color=color, label=label_str,
                              linewidth=1.0, alpha=0.85)
            axes[2, col].plot(steps, gap, color=color, label=label_str,
                              linewidth=1.0, alpha=0.85)

        axes[0, col].set_title(f"α = {alpha}")
        axes[0, col].set_ylabel("Train loss")
        axes[0, col].set_yscale("log")
        axes[1, col].set_ylabel("Test loss")
        axes[1, col].set_yscale("log")
        axes[2, col].set_ylabel("Generalization gap\n(test - train loss)")
        axes[2, col].set_xlabel("Gradient steps")
        axes[0, col].legend(fontsize=7)
        axes[2, col].axhline(0, color="gray", linestyle=":", alpha=0.4)

    fig.suptitle("Loss Dynamics & Generalization Gap (seed=42)", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "exp3_loss_generalization_gap.png"),
                bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/exp3_loss_generalization_gap.png")


def plot_3b_trajectories():
    """Plot 7: Learning curves for operand vs target vs IID partitions."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    focus_alphas = [0.25, 0.3, 0.5]
    partitions = ["iid", "operand", "target"]
    part_colors = {"iid": "#2196F3", "operand": "#FF5722", "target": "#4CAF50"}

    for col, alpha in enumerate(focus_alphas):
        for part in partitions:
            if part == "iid":
                h = load_raw_histories_3a(alpha, 1000.0)
            else:
                h = load_raw_histories_3b(alpha, part)
            if h is None:
                continue
            steps = np.array(h["total_steps"])
            test_acc = np.array(h["test_acc"])
            ipr = np.array(h["ipr"])
            color = part_colors[part]
            label = part.capitalize()

            axes[0, col].plot(steps, test_acc, color=color, label=label,
                              linewidth=1.5, alpha=0.9)
            axes[1, col].plot(steps, ipr, color=color, label=label,
                              linewidth=1.5, alpha=0.9)

        axes[0, col].set_title(f"α = {alpha}")
        axes[0, col].set_ylabel("Test accuracy (%)")
        axes[0, col].set_ylim(-5, 105)
        axes[0, col].axhline(95, color="gray", linestyle=":", alpha=0.4)
        axes[0, col].legend(fontsize=9)
        axes[1, col].set_ylabel("IPR")
        axes[1, col].set_xlabel("Gradient steps")
        axes[1, col].legend(fontsize=9)

    fig.suptitle("Structured Partitions: Test Acc & IPR Trajectories (seed=42)",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "exp3b_partition_trajectories.png"),
                bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/exp3b_partition_trajectories.png")


def plot_drift_ipr_correlation():
    """Plot 8: Scatter — mean client drift at T_grok vs IPR at T_grok,
    across all exp3a conditions. Tests if drift interferes with feature learning."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    alphas_all = [0.2, 0.25, 0.3, 0.35, 0.5]
    dir_alphas = [0.01, 0.1, 0.5, 1.0, 10.0, 1000.0]
    seeds = [42, 123, 456]
    alpha_colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(alphas_all)))
    alpha_cmap = {a: c for a, c in zip(alphas_all, alpha_colors)}

    # Collect: for each run, get drift and IPR at the grokking step (or final)
    drift_at_grok = []
    ipr_at_grok = []
    div_at_grok = []
    colors_scatter = []
    sizes_scatter = []

    for alpha in alphas_all:
        for dir_alpha in dir_alphas:
            for seed in seeds:
                h = load_raw_histories_3a(alpha, dir_alpha, seed=seed)
                if h is None:
                    continue
                test_acc = np.array(h["test_acc"])
                drift = np.array(h["mean_client_drift"])
                ipr_arr = np.array(h["ipr"])
                div_arr = np.array(h["client_weight_divergence"])

                # Find grokking step (first step where test_acc >= 95 permanently)
                grok_idx = None
                for i in range(len(test_acc)):
                    if all(a >= 95 for a in test_acc[i:]):
                        grok_idx = i
                        break
                if grok_idx is None:
                    grok_idx = len(test_acc) - 1  # use final values

                drift_at_grok.append(drift[grok_idx])
                ipr_at_grok.append(ipr_arr[grok_idx])
                div_at_grok.append(div_arr[grok_idx])
                colors_scatter.append(alpha_cmap[alpha])
                sizes_scatter.append(20 + 80 * np.log10(dir_alpha + 0.001) / 6)

    drift_at_grok = np.array(drift_at_grok)
    ipr_at_grok = np.array(ipr_at_grok)
    div_at_grok = np.array(div_at_grok)

    # Left: drift vs IPR
    axes[0].scatter(drift_at_grok, ipr_at_grok, c=colors_scatter, s=50,
                    alpha=0.7, edgecolors="k", linewidths=0.3)
    axes[0].set_xlabel("Mean client drift at T_grok")
    axes[0].set_ylabel("IPR at T_grok")
    axes[0].set_title("Client drift vs feature quality at grokking")

    # Right: weight divergence vs IPR
    axes[1].scatter(div_at_grok, ipr_at_grok, c=colors_scatter, s=50,
                    alpha=0.7, edgecolors="k", linewidths=0.3)
    axes[1].set_xlabel("Client weight divergence at T_grok")
    axes[1].set_ylabel("IPR at T_grok")
    axes[1].set_title("Weight divergence vs feature quality at grokking")

    # Legend for alpha values
    for alpha, color in zip(alphas_all, alpha_colors):
        for ax in axes:
            ax.scatter([], [], c=[color], s=50, label=f"α={alpha:.2f}",
                       edgecolors="k", linewidths=0.3)
    axes[0].legend(fontsize=8, loc="lower right")

    fig.suptitle("Drift-IPR Correlation at Grokking Transition (all seeds)",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "exp3_drift_ipr_correlation.png"),
                bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/exp3_drift_ipr_correlation.png")


def plot_weight_norm_ratio():
    """Plot 9: Weight norm ratio (layer2/layer1) over time."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    focus_alphas = [0.25, 0.3, 0.5]
    dir_alphas = [0.01, 0.1, 1.0, 1000.0]
    colors = plt.cm.plasma(np.linspace(0.1, 0.85, len(dir_alphas)))

    for ax, alpha in zip(axes, focus_alphas):
        for dir_alpha, color in zip(dir_alphas, colors):
            h = load_raw_histories_3a(alpha, dir_alpha)
            if h is None:
                continue
            steps = np.array(h["total_steps"])
            w1 = np.array(h["weight_norm_layer1"])
            w2 = np.array(h["weight_norm_layer2"])
            ratio = w2 / np.maximum(w1, 1e-8)
            label_str = f"dir={dir_alpha}" if dir_alpha < 1000 else "IID"
            ax.plot(steps, ratio, color=color, label=label_str,
                    linewidth=1.2, alpha=0.9)

        ax.set_xlabel("Gradient steps")
        ax.set_ylabel("||W2|| / ||W1||")
        ax.set_title(f"α = {alpha}")
        ax.legend(fontsize=8)

    fig.suptitle("Weight Norm Ratio (Layer 2 / Layer 1) Evolution (seed=42)",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "exp3_weight_norm_ratio.png"),
                bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/exp3_weight_norm_ratio.png")


def plot_per_seed_variability():
    """Plot 10: Per-seed trajectories for a representative condition to show variance."""
    seeds = [42, 123, 456]
    seed_colors = ["#E91E63", "#3F51B5", "#FF9800"]

    # Show 4 conditions: IID mild alpha, non-IID mild alpha, IID boundary, non-IID boundary
    conditions = [
        (0.5, 1000.0, "α=0.5, IID"),
        (0.5, 0.01, "α=0.5, dir=0.01"),
        (0.25, 1000.0, "α=0.25, IID"),
        (0.25, 0.01, "α=0.25, dir=0.01"),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(20, 8))

    for col, (alpha, dir_alpha, title) in enumerate(conditions):
        for seed, scolor in zip(seeds, seed_colors):
            h = load_raw_histories_3a(alpha, dir_alpha, seed=seed)
            if h is None:
                continue
            steps = np.array(h["total_steps"])
            test_acc = np.array(h["test_acc"])
            ipr = np.array(h["ipr"])

            axes[0, col].plot(steps, test_acc, color=scolor, label=f"seed={seed}",
                              linewidth=1.0, alpha=0.8)
            axes[1, col].plot(steps, ipr, color=scolor, label=f"seed={seed}",
                              linewidth=1.0, alpha=0.8)

        axes[0, col].set_title(title)
        axes[0, col].set_ylim(-5, 105)
        axes[0, col].axhline(95, color="gray", linestyle=":", alpha=0.4)
        axes[0, col].legend(fontsize=7)
        axes[1, col].set_xlabel("Gradient steps")

    axes[0, 0].set_ylabel("Test accuracy (%)")
    axes[1, 0].set_ylabel("IPR")

    fig.suptitle("Per-Seed Variability: Test Accuracy & IPR", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "exp3_per_seed_variability.png"),
                bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/exp3_per_seed_variability.png")


def smooth(arr, window=50):
    """Simple moving average for noisy per-round metrics."""
    if len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="same")


def compute_t_grok(steps, test_accs, threshold=95.0):
    """Compute grokking step (smallest step where test_acc >= threshold permanently)."""
    n = len(steps)
    last_below = -1
    for i in range(n - 1, -1, -1):
        if test_accs[i] < threshold:
            last_below = i
            break
    if last_below == -1:
        return steps[0]
    if last_below == n - 1:
        return float("inf")
    return steps[last_below + 1]


def plot_grokking_anatomy():
    """Overlay 1: Grokking anatomy — 4 metrics on shared time axis for two
    representative conditions (IID vs extreme non-IID), showing temporal
    coincidence of the grokking transition across all observables."""
    conditions = [
        (0.3, 1000.0, "α=0.3, IID (dir=1000)"),
        (0.3, 0.01, "α=0.3, Extreme non-IID (dir=0.01)"),
        (0.25, 1000.0, "α=0.25, IID (dir=1000)"),
        (0.25, 0.01, "α=0.25, Extreme non-IID (dir=0.01)"),
    ]

    fig, axes = plt.subplots(4, len(conditions), figsize=(6 * len(conditions), 14),
                             sharex="col")

    for col, (alpha, dir_alpha, title) in enumerate(conditions):
        h = load_raw_histories_3a(alpha, dir_alpha)
        if h is None:
            continue
        steps = np.array(h["total_steps"])
        test_acc = np.array(h["test_acc"])
        test_loss = np.array(h["test_loss"])
        train_loss = np.array(h["train_loss"])
        gap = test_loss - train_loss
        drift = smooth(np.array(h["mean_client_drift"]), window=80)
        ipr = np.array(h["ipr"])
        w1 = np.array(h["weight_norm_layer1"])
        w2 = np.array(h["weight_norm_layer2"])
        ratio = w2 / np.maximum(w1, 1e-8)

        # Find T_grok and T_onset for vertical markers
        t_grok = compute_t_grok(steps.tolist(), test_acc.tolist())

        # T_onset: first step where test_acc > 5% (clearly above chance ~1/p ≈ 1%)
        onset_indices = np.where(test_acc > 5.0)[0]
        t_onset = steps[onset_indices[0]] if len(onset_indices) > 0 else float("inf")

        # Row 0: Test accuracy
        ax = axes[0, col]
        ax.plot(steps, test_acc, color="#2196F3", linewidth=1.5)
        ax.set_ylabel("Test accuracy (%)" if col == 0 else "")
        ax.set_ylim(-5, 105)
        ax.axhline(95, color="gray", linestyle=":", alpha=0.3)
        ax.set_title(title, fontsize=11)

        # Row 1: Generalization gap
        ax = axes[1, col]
        ax.plot(steps, gap, color="#FF5722", linewidth=1.2)
        ax.set_ylabel("Gen. gap (test−train loss)" if col == 0 else "")
        ax.axhline(0, color="gray", linestyle=":", alpha=0.3)

        # Row 2: Client drift
        ax = axes[2, col]
        ax.plot(steps, drift, color="#9C27B0", linewidth=1.2)
        ax.set_ylabel("Mean client drift" if col == 0 else "")

        # Row 3: IPR + weight norm ratio (twin y-axes)
        ax = axes[3, col]
        l1 = ax.plot(steps, ipr, color="#4CAF50", linewidth=1.5, label="IPR")
        ax.set_ylabel("IPR" if col == 0 else "", color="#4CAF50")
        ax.tick_params(axis="y", labelcolor="#4CAF50")
        ax2 = ax.twinx()
        l2 = ax2.plot(steps, ratio, color="#FF9800", linewidth=1.2, alpha=0.8,
                       label="W2/W1 ratio")
        ax2.set_ylabel("W2/W1" if col == len(conditions) - 1 else "",
                        color="#FF9800")
        ax2.tick_params(axis="y", labelcolor="#FF9800")
        if col == 0:
            lines = l1 + l2
            labels = [l.get_label() for l in lines]
            ax.legend(lines, labels, fontsize=8, loc="lower right")

        ax.set_xlabel("Gradient steps")

        # Vertical markers across all rows
        for row in range(4):
            if t_onset < float("inf"):
                axes[row, col].axvline(t_onset, color="#2E7D32", linestyle=":",
                                       alpha=0.8, linewidth=1.8)
            if t_grok < float("inf"):
                axes[row, col].axvline(t_grok, color="#C62828", linestyle="--",
                                       alpha=0.7, linewidth=1.8)

        # Labels only in top row to avoid clutter
        if t_onset < float("inf"):
            axes[0, col].text(t_onset - 200, 85,
                              f"T_onset\n{int(t_onset)}",
                              fontsize=7, color="#2E7D32", fontweight="bold",
                              ha="right", va="top")
        if t_grok < float("inf"):
            axes[0, col].text(t_grok + 200, 55,
                              f"T_grok\n{int(t_grok)}",
                              fontsize=7, color="#C62828", fontweight="bold",
                              ha="left", va="top")

    # Row labels on left
    row_labels = ["(a) Test Accuracy", "(b) Generalization Gap",
                  "(c) Client Drift", "(d) IPR + Weight Ratio"]
    for row, label in enumerate(row_labels):
        axes[row, 0].annotate(label, xy=(-0.25, 0.5),
                               xycoords="axes fraction", fontsize=10,
                               fontweight="bold", rotation=90,
                               va="center", ha="center")

    fig.suptitle("Grokking Anatomy: Co-evolution of All Metrics (seed=42)",
                 fontsize=15, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "exp3_grokking_anatomy.png"),
                bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/exp3_grokking_anatomy.png")


def plot_acc_drift_overlay():
    """Overlay 2: Test accuracy and client drift on twin y-axes for each
    heterogeneity level. Directly shows drift peak precedes grokking."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    focus_alphas = [0.25, 0.3, 0.5]
    dir_alphas = [0.01, 0.1, 1.0, 1000.0]
    colors = plt.cm.plasma(np.linspace(0.1, 0.85, len(dir_alphas)))

    for col, alpha in enumerate(focus_alphas):
        # Top row: all heterogeneity levels overlaid
        ax = axes[0, col]
        ax2 = ax.twinx()

        for dir_alpha, color in zip(dir_alphas, colors):
            h = load_raw_histories_3a(alpha, dir_alpha)
            if h is None:
                continue
            steps = np.array(h["total_steps"])
            test_acc = np.array(h["test_acc"])
            drift = smooth(np.array(h["mean_client_drift"]), window=80)
            label_str = f"dir={dir_alpha}" if dir_alpha < 1000 else "IID"

            ax.plot(steps, test_acc, color=color, linewidth=1.3, alpha=0.9,
                    label=label_str)
            ax2.plot(steps, drift, color=color, linewidth=0.8, alpha=0.5,
                     linestyle="--")

        ax.set_ylim(-5, 105)
        ax.axhline(95, color="gray", linestyle=":", alpha=0.3)
        ax.set_ylabel("Test accuracy (%) — solid")
        ax2.set_ylabel("Client drift — dashed")
        ax.set_title(f"α = {alpha}")
        ax.legend(fontsize=7, loc="center left")

        # Bottom row: zoomed single comparison — IID vs most extreme
        ax = axes[1, col]
        ax2 = ax.twinx()

        for dir_alpha, color, lw in [(1000.0, "#2196F3", 2.0), (0.01, "#F44336", 2.0)]:
            h = load_raw_histories_3a(alpha, dir_alpha)
            if h is None:
                continue
            steps = np.array(h["total_steps"])
            test_acc = np.array(h["test_acc"])
            drift = smooth(np.array(h["mean_client_drift"]), window=80)
            label_str = "IID" if dir_alpha >= 1000 else "dir=0.01"

            ax.plot(steps, test_acc, color=color, linewidth=lw, alpha=0.9,
                    label=f"{label_str} acc")
            ax2.plot(steps, drift, color=color, linewidth=lw * 0.6, alpha=0.6,
                     linestyle="--", label=f"{label_str} drift")

        ax.set_ylim(-5, 105)
        ax.axhline(95, color="gray", linestyle=":", alpha=0.3)
        ax.set_ylabel("Test accuracy (%) — solid")
        ax2.set_ylabel("Client drift — dashed")
        ax.set_xlabel("Gradient steps")
        ax.set_title(f"α = {alpha}: IID vs Extreme")
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="center left")

    fig.suptitle("Test Accuracy + Client Drift Overlay (seed=42)",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "exp3_acc_drift_overlay.png"),
                bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/exp3_acc_drift_overlay.png")


def plot_ipr_wratio_overlay():
    """Overlay 3: IPR and W2/W1 ratio on twin y-axes — dual readiness metrics.
    Shows whether feature formation (IPR) and layer balance (ratio) are coupled."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    focus_alphas = [0.25, 0.3, 0.5]
    dir_alphas = [0.01, 1.0, 1000.0]
    for ax, alpha in zip(axes, focus_alphas):
        for dir_alpha in dir_alphas:
            h = load_raw_histories_3a(alpha, dir_alpha)
            if h is None:
                continue
            steps = np.array(h["total_steps"])
            ipr = np.array(h["ipr"])
            w1 = np.array(h["weight_norm_layer1"])
            w2 = np.array(h["weight_norm_layer2"])
            ratio = w2 / np.maximum(w1, 1e-8)

            label_str = f"dir={dir_alpha}" if dir_alpha < 1000 else "IID"

            # Normalize both to [0, 1] for visual comparison
            ipr_norm = (ipr - ipr.min()) / (ipr.max() - ipr.min() + 1e-10)
            ratio_norm = (ratio - ratio.min()) / (ratio.max() - ratio.min() + 1e-10)

            color_idx = [0.01, 1.0, 1000.0].index(dir_alpha)
            color = ["#F44336", "#FF9800", "#2196F3"][color_idx]

            ax.plot(steps, ipr_norm, color=color, linewidth=1.5, alpha=0.9,
                    linestyle="-", label=f"{label_str} IPR")
            ax.plot(steps, ratio_norm, color=color, linewidth=1.0, alpha=0.6,
                    linestyle="--", label=f"{label_str} W2/W1")

        ax.set_xlabel("Gradient steps")
        ax.set_ylabel("Normalized value (0–1)")
        ax.set_title(f"α = {alpha}")
        ax.legend(fontsize=6, ncol=2, loc="lower right")
        ax.set_ylim(-0.05, 1.05)

    fig.suptitle("IPR (solid) vs W2/W1 Ratio (dashed) — Normalized Overlay (seed=42)",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "exp3_ipr_wratio_overlay.png"),
                bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/exp3_ipr_wratio_overlay.png")


def plot_phase_aligned_fingerprint():
    """Overlay 4: All grokking runs aligned to T_grok=0. Shows universal
    fingerprint: test acc jumps, gap closes, drift collapses, IPR saturates —
    all centered on the same moment regardless of heterogeneity."""
    alphas = [0.25, 0.3, 0.35, 0.5]
    dir_alphas = [0.01, 0.1, 0.5, 1.0, 10.0, 1000.0]
    seeds = [42, 123, 456]

    aligned = {"test_acc": [], "gap": [], "drift": [], "ipr": []}
    run_colors = []

    cmap = plt.cm.plasma
    dir_color_map = {d: cmap(i / (len(dir_alphas) - 1) * 0.85 + 0.1)
                     for i, d in enumerate(dir_alphas)}

    for alpha in alphas:
        for dir_alpha in dir_alphas:
            for seed in seeds:
                h = load_raw_histories_3a(alpha, dir_alpha, seed=seed)
                if h is None:
                    continue
                steps = np.array(h["total_steps"])
                test_acc = np.array(h["test_acc"])
                t_grok = compute_t_grok(steps.tolist(), test_acc.tolist())
                if t_grok == float("inf"):
                    continue

                rel_steps = steps - t_grok
                test_loss = np.array(h["test_loss"])
                train_loss = np.array(h["train_loss"])
                gap = test_loss - train_loss
                drift = smooth(np.array(h["mean_client_drift"]), window=80)
                ipr = np.array(h["ipr"])

                aligned["test_acc"].append((rel_steps, test_acc))
                aligned["gap"].append((rel_steps, gap))
                aligned["drift"].append((rel_steps, drift))
                aligned["ipr"].append((rel_steps, ipr))
                run_colors.append(dir_color_map[dir_alpha])

    n_runs = len(aligned["test_acc"])
    if n_runs == 0:
        print("No grokking runs found for phase-aligned plot")
        return

    fig, axes = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
    metrics = [
        ("test_acc", "Test accuracy (%)", "#2196F3"),
        ("gap", "Generalization gap", "#FF5722"),
        ("drift", "Client drift (smoothed)", "#9C27B0"),
        ("ipr", "IPR", "#4CAF50"),
    ]

    window = (-25000, 15000)

    for ax, (key, ylabel, median_color) in zip(axes, metrics):
        # Individual runs as thin lines
        for (s, v), c in zip(aligned[key], run_colors):
            mask = (s >= window[0]) & (s <= window[1])
            ax.plot(s[mask], v[mask], color=c, alpha=0.15, linewidth=0.5)

        # Compute median + IQR in bins
        bins = np.linspace(window[0], window[1], 300)
        bin_centers, bin_meds, bin_q25, bin_q75 = [], [], [], []

        for i in range(len(bins) - 1):
            vals_in_bin = []
            for s, v in aligned[key]:
                mask = (s >= bins[i]) & (s < bins[i + 1])
                if mask.any():
                    vals_in_bin.extend(v[mask].tolist())
            if len(vals_in_bin) >= 3:
                bin_centers.append((bins[i] + bins[i + 1]) / 2)
                bin_meds.append(np.median(vals_in_bin))
                bin_q25.append(np.percentile(vals_in_bin, 25))
                bin_q75.append(np.percentile(vals_in_bin, 75))

        if bin_centers:
            ax.plot(bin_centers, bin_meds, color="black", linewidth=2.5,
                    zorder=10, label="Median")
            ax.fill_between(bin_centers, bin_q25, bin_q75, color="black",
                            alpha=0.08, zorder=9)

        ax.axvline(0, color="red", linestyle="--", alpha=0.7, linewidth=1.5,
                   label=r"$T_{grok}$")
        ax.set_ylabel(ylabel)
        if key == "test_acc":
            ax.set_ylim(-5, 105)
            ax.axhline(95, color="gray", linestyle=":", alpha=0.3)
            ax.legend(fontsize=9)

    axes[-1].set_xlabel(r"Steps relative to $T_{grok}$")
    axes[-1].set_xlim(*window)

    # Color legend for heterogeneity
    for d in dir_alphas:
        label = f"dir={d}" if d < 1000 else "IID"
        axes[0].plot([], [], color=dir_color_map[d], linewidth=2,
                     alpha=0.7, label=label)
    axes[0].legend(fontsize=7, ncol=4, loc="lower left")

    labels = ["(a)", "(b)", "(c)", "(d)"]
    for ax, lbl in zip(axes, labels):
        ax.set_title(lbl, loc="left", fontweight="bold", fontsize=12)

    fig.suptitle(f"Phase-Aligned Universal Fingerprint ({n_runs} grokking runs)",
                 fontsize=15, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "exp3_phase_aligned_fingerprint.png"),
                bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/exp3_phase_aligned_fingerprint.png ({n_runs} runs)")


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

    # Diagnostic plots (round 1)
    print("\n--- Diagnostic plots (round 1) ---")
    plot_test_acc_trajectories()
    plot_client_drift_dynamics()
    plot_ipr_trajectories()
    plot_grokking_sharpness(cells_3a)
    plot_weight_norm_evolution()

    # Diagnostic plots (round 2)
    print("\n--- Diagnostic plots (round 2) ---")
    plot_loss_and_generalization_gap()
    plot_3b_trajectories()
    plot_drift_ipr_correlation()
    plot_weight_norm_ratio()
    plot_per_seed_variability()

    # Overlay plots (round 3)
    print("\n--- Overlay plots (round 3) ---")
    plot_grokking_anatomy()
    plot_acc_drift_overlay()
    plot_ipr_wratio_overlay()
    plot_phase_aligned_fingerprint()

    print(f"\nDone! All figures saved to {OUTPUT_DIR}")
