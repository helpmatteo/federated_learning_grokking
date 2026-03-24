"""Paper figure: Client drift as the mechanistic bottleneck for grokking.

Multi-panel figure showing that across all FL factors (local epochs, number
of clients, data heterogeneity, participation fraction), increased client
drift consistently delays grokking.

Each panel varies one factor while fixing the others, plotting
mean client drift (x) vs T_grok (y) with the factor value as color.
"""

import json
import glob
import re
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "legend.fontsize": 10,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 300,
})

OUTPUT_DIR = "results/paper_figures"
BUDGET = 50_000
SEEDS = [42, 123, 456]


# ── Helpers ───────────────────────────────────────────────────────────────

def _load(path):
    with open(path) as f:
        return json.load(f)


def _compute_t_grok(h, threshold=95.0):
    """Compute T_grok from history dict. test_acc is 0-100 scale."""
    test_acc = np.array(h["test_acc"])
    steps = np.array(h["total_steps"])
    for i in range(len(test_acc)):
        if test_acc[i] >= threshold:
            if all(a >= threshold for a in test_acc[i:]):
                return steps[i]
    return None


def _mean_drift(h):
    drift = h.get("mean_client_drift", [])
    if len(drift) == 0:
        return None
    return float(np.mean(drift))


def _collect_runs(file_pattern, param_extractor):
    """Collect (param_value, mean_drift, t_grok) across seeds from matching files."""
    points = []
    for path in sorted(glob.glob(file_pattern)):
        param_val = param_extractor(path)
        if param_val is None:
            continue
        h = _load(path)
        drift = _mean_drift(h)
        t_grok = _compute_t_grok(h)
        if drift is not None:
            points.append({
                "param": param_val,
                "drift": drift,
                "t_grok": t_grok,
                "grokked": t_grok is not None,
            })
    return points


# ── Data collection per panel ─────────────────────────────────────────────

def collect_local_epochs():
    """Exp 4a: vary E ∈ {5, 10, 25, 50}, fix α=0.3, K=10, IID."""
    def extract_E(path):
        m = re.search(r"_le(\d+)_", path)
        return int(m.group(1)) if m else None

    return _collect_runs(
        "results/exp4_optimization/exp4a/history_fed_*_a0.3_K10_le*_ft1.0_iid_s*.json",
        extract_E,
    )


def collect_num_clients():
    """Exp 2: vary K ∈ {2, 5, 10, 20, 50, 97}, fix α=0.3, E=5, IID."""
    def extract_K(path):
        m = re.search(r"_K(\d+)_", path)
        return int(m.group(1)) if m else None

    return _collect_runs(
        "results/exp2_aggregation/fl_iid/history_fed_*_a0.3_K*_le5_ft1.0_iid_s*.json",
        extract_K,
    )


def collect_heterogeneity():
    """Exp 3a: vary Dir(α) ∈ {0.01, 0.1, 0.5, 1.0, 10.0, 1000.0}, fix α=0.3, K=10."""
    def extract_dir(path):
        m = re.search(r"_dir([\d.]+)_", path)
        return float(m.group(1)) if m else None

    return _collect_runs(
        "results/exp3_heterogeneity/exp3a/history_fed_*_a0.3_K10_le5_ft1.0_dirichlet_dir*_s*.json",
        extract_dir,
    )


def collect_participation():
    """Exp 4b: vary f ∈ {0.2, 0.4, 0.6, 1.0}, fix α=0.3, K=10, IID."""
    def extract_f(path):
        m = re.search(r"_ft([\d.]+)_", path)
        return float(m.group(1)) if m else None

    return _collect_runs(
        "results/exp4_optimization/exp4b/history_fed_*_a0.3_K10_le5_ft*_iid_s*.json",
        extract_f,
    )


# ── Plotting ──────────────────────────────────────────────────────────────

def plot_drift_vs_grokking():
    """2×2 figure: each panel = one FL factor, x=mean drift, y=T_grok."""
    panels = [
        ("Local epochs $E$", collect_local_epochs(), "E"),
        ("Clients $K$", collect_num_clients(), "K"),
        ("Heterogeneity Dir($\\alpha_d$)", collect_heterogeneity(), "dir"),
        ("Participation $f$", collect_participation(), "f"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 7))
    axes = axes.flatten()

    for idx, (title, points, ptype) in enumerate(panels):
        ax = axes[idx]

        if not points:
            ax.set_title(title)
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
            continue

        # Get unique param values and assign colors
        param_vals = sorted(set(p["param"] for p in points))
        cmap = plt.cm.viridis(np.linspace(0.1, 0.9, len(param_vals)))
        color_map = dict(zip(param_vals, cmap))

        # Plot grokked points
        for p in points:
            if p["grokked"]:
                ax.scatter(p["drift"], p["t_grok"] / 1000,
                           c=[color_map[p["param"]]], s=60, alpha=0.8,
                           edgecolors="white", linewidth=0.5, zorder=3)

        # Plot failed points at budget line
        for p in points:
            if not p["grokked"]:
                ax.scatter(p["drift"], BUDGET / 1000,
                           c=[color_map[p["param"]]], s=60, alpha=0.2,
                           marker="x", linewidth=1.5, zorder=2)

        # Legend with param values
        handles = []
        for val, color in zip(param_vals, cmap):
            if ptype == "dir":
                label = f"{val}"
            elif ptype == "f":
                label = f"{val}"
            else:
                label = f"{val}"
            handles.append(Line2D([0], [0], marker="o", color="w",
                                  markerfacecolor=color, markersize=8,
                                  label=label))
        ax.legend(handles=handles, title=title, fontsize=9,
                  title_fontsize=10, loc="upper left")

        ax.set_xlabel("Mean client drift", fontsize=13)
        if idx % 2 == 0:
            ax.set_ylabel(r"$T_{\mathrm{grok}}$ ($\times$1000)", fontsize=13)
        ax.tick_params(labelsize=11)
        ax.text(-0.12, 1.08, chr(ord('a') + idx) + ")",
                transform=ax.transAxes, fontsize=15, fontweight="bold", va="top")

    plt.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "drift_vs_grokking.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    plot_drift_vs_grokking()
