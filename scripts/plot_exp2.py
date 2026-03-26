"""Plot Experiment 2 results — improved figures for the aggregation effect."""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.family": "sans-serif",
})

RESULTS_DIR = "results/experiments/exp2_aggregation"
OUTPUT_DIR = "results/experiments/exp2_aggregation/figures"


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


def load_history(path):
    """Load a raw training history JSON file."""
    with open(path) as f:
        return json.load(f)


# ── Figure 1: Phase diagram ──────────────────────────────────────────────────

def plot_phase_diagram(cells):
    """Single heatmap: outcome category in (alpha, K) space.

    Categories:
      0 = Below boundary (cent_full doesn't grok)
      1 = Neither FL nor reduced grok (fragmentation overwhelms)
      2 = FL groks, reduced doesn't (AGGREGATION RESCUES)
      3 = Both FL and reduced grok (aggregation not needed)
    """
    alphas = sorted(set(c["alpha"] for c in cells))
    ks = sorted(set(c["K"] for c in cells))

    grid = np.full((len(alphas), len(ks)), np.nan)
    for c in cells:
        ai = alphas.index(c["alpha"])
        ki = ks.index(c["K"])
        cf = c["cent_full"]["summary"]["n_grokked"] > 0
        cr = c["cent_reduced"]["summary"]["n_grokked"] > 0
        fl = c["fl_iid"]["summary"]["n_grokked"] > 0

        if not cf:
            grid[ai, ki] = 0  # below boundary
        elif fl and cr:
            grid[ai, ki] = 3  # both grok
        elif fl and not cr:
            grid[ai, ki] = 2  # aggregation rescues
        elif not fl and not cr:
            grid[ai, ki] = 1  # fragmentation overwhelms
        else:
            grid[ai, ki] = 1  # reduced groks but FL doesn't (shouldn't happen)

    cmap = ListedColormap(["#bdbdbd", "#e74c3c", "#2196F3", "#4CAF50"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(grid, aspect="auto", cmap=cmap, norm=norm, origin="lower")

    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels(ks)
    ax.set_yticks(range(len(alphas)))
    ax.set_yticklabels([f"{a:.2f}" for a in alphas])
    ax.set_xlabel("K (number of clients)")
    ax.set_ylabel(r"$\alpha$ (train fraction)")

    # Annotate with grokking fractions
    for c in cells:
        ai = alphas.index(c["alpha"])
        ki = ks.index(c["K"])
        fl_frac = c["fl_iid"]["summary"]["n_grokked"] / c["fl_iid"]["summary"]["n_seeds"]
        cat = int(grid[ai, ki])
        text_color = "white" if cat in [1, 2] else "black"
        if cat == 0:
            ax.text(ki, ai, "no grok", ha="center", va="center",
                    fontsize=8, color="#555555", style="italic")
        else:
            ax.text(ki, ai, f"FL: {fl_frac:.0%}", ha="center", va="center",
                    fontsize=9, fontweight="bold", color=text_color)

    # Legend — place below the plot to avoid overlapping cells
    legend_elements = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#bdbdbd",
               markersize=12, label="Below phase boundary"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#e74c3c",
               markersize=12, label="Fragmentation overwhelms"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#2196F3",
               markersize=12, label="Aggregation rescues grokking"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#4CAF50",
               markersize=12, label="Both grok (aggregation not needed)"),
    ]
    ax.legend(handles=legend_elements, loc="upper center",
              bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=8,
              framealpha=0.9)

    ax.set_title("Aggregation Effect: Phase Diagram", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "exp2_phase_diagram.png"),
                bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/exp2_phase_diagram.png")


# ── Figure 2: Slowdown ratio ─────────────────────────────────────────────────

def plot_slowdown_ratio(cells):
    """T_grok(FL) / T_grok(cent_full) vs K, one line per alpha."""
    alphas = sorted(set(c["alpha"] for c in cells))
    ks = sorted(set(c["K"] for c in cells))

    colors = {"0.25": "#9C27B0", "0.3": "#2196F3", "0.35": "#009688",
              "0.5": "#FF9800"}

    fig, ax = plt.subplots(figsize=(7, 5))

    for alpha in alphas:
        if alpha == 0.2:
            continue  # nothing groks
        k_vals, ratios, lo_bars, hi_bars = [], [], [], []
        for K in ks:
            match = [c for c in cells if c["alpha"] == alpha and c["K"] == K]
            if not match:
                continue
            c = match[0]
            t_cf = parse_inf(c["cent_full"]["summary"]["t_grok_mean"])
            t_fl = parse_inf(c["fl_iid"]["summary"]["t_grok_mean"])
            if t_cf == float("inf") or t_fl == float("inf"):
                continue
            ratio = t_fl / t_cf
            # Propagate uncertainty: ratio_std ≈ ratio * sqrt((s_fl/m_fl)^2 + (s_cf/m_cf)^2)
            s_cf = parse_inf(c["cent_full"]["summary"]["t_grok_std"])
            s_fl = parse_inf(c["fl_iid"]["summary"]["t_grok_std"])
            if s_cf < float("inf") and s_fl < float("inf") and t_cf > 0 and t_fl > 0:
                ratio_std = ratio * np.sqrt((s_fl / t_fl) ** 2 + (s_cf / t_cf) ** 2)
            else:
                ratio_std = 0
            k_vals.append(K)
            ratios.append(ratio)
            lo_bars.append(ratio_std)
            hi_bars.append(ratio_std)

        if k_vals:
            c_key = f"{alpha:.2g}"
            ax.errorbar(k_vals, ratios, yerr=[lo_bars, hi_bars],
                        marker="o", color=colors.get(c_key, "gray"),
                        label=f"α = {alpha:.2f}", capsize=3,
                        linewidth=1.8, markersize=6)

    ax.axhline(1.0, color="black", linestyle="--", alpha=0.4, linewidth=1,
               label="No slowdown")
    ax.set_xscale("log")
    ax.set_xticks(ks)
    ax.set_xticklabels(ks)
    ax.set_xlabel("K (number of clients)")
    ax.set_ylabel(r"$T_\mathrm{grok}^\mathrm{FL}\; /\; T_\mathrm{grok}^\mathrm{cent}$")
    ax.set_title("FL Slowdown Relative to Centralized", fontsize=14)
    ax.legend(fontsize=9)
    ax.set_ylim(0.9, None)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "exp2_slowdown_ratio.png"),
                bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/exp2_slowdown_ratio.png")

    # --- Consistent linear scale version ---
    # Collect global max across all finite T_grok values
    all_t = []
    for c in cells:
        for cond in conditions:
            match = [cc for cc in cells if cc["alpha"] == c["alpha"] and cc["K"] == c["K"]]
            if match:
                t = parse_inf(match[0][cond]["summary"]["t_grok_mean"])
                s = parse_inf(match[0][cond]["summary"]["t_grok_std"])
                if t < float("inf"):
                    all_t.append(t + (s if s < float("inf") else 0))

    y_max = max(all_t) * 1.1 if all_t else 50000

    fig2, axes2 = plt.subplots(1, len(alphas), figsize=(4 * len(alphas), 4.5), sharey=True)
    if len(alphas) == 1:
        axes2 = [axes2]

    for ax, alpha in zip(axes2, alphas):
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
        ax.set_xticks(ks)
        ax.set_xticklabels(ks)
        ax.set_ylim(0, y_max)

    axes2[0].set_ylabel(r"$T_{grok}$ (steps)")
    axes2[-1].legend(loc="upper left", fontsize=8)
    fig2.suptitle("Exp 2: Grokking Time vs Number of Clients (consistent scale)", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "exp2_t_grok_vs_K_linear.png"), bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/exp2_t_grok_vs_K_linear.png")


# ── Figure 3: FL vs Centralized scatter (improved) ───────────────────────────

def plot_fl_vs_centralized(cells):
    """Scatter: FL T_grok vs Centralized T_grok. K encoded as marker size."""
    alphas = sorted(set(c["alpha"] for c in cells))
    ks = sorted(set(c["K"] for c in cells))

    # Qualitative colormap for categorical alpha
    alpha_colors = {
        0.25: "#9C27B0",
        0.3: "#2196F3",
        0.35: "#009688",
        0.5: "#FF9800",
    }
    # Marker size proportional to K
    k_min, k_max = min(ks), max(ks)
    def k_to_size(k):
        return 40 + 200 * (np.log(k) - np.log(k_min)) / (np.log(k_max) - np.log(k_min))

    fig, ax = plt.subplots(figsize=(6, 6))

    for alpha in alphas:
        if alpha == 0.2:
            continue
        for c in cells:
            if c["alpha"] != alpha:
                continue
            t_cent = parse_inf(c["cent_full"]["summary"]["t_grok_mean"])
            t_fl = parse_inf(c["fl_iid"]["summary"]["t_grok_mean"])
            if t_cent == float("inf") or t_fl == float("inf"):
                continue
            K = c["K"]
            ax.scatter(t_cent, t_fl, c=alpha_colors.get(alpha, "gray"),
                       s=k_to_size(K), zorder=3, edgecolors="white",
                       linewidths=0.5)

    # Legend for alpha (color)
    alpha_handles = []
    for a in sorted(alpha_colors):
        if a not in set(c["alpha"] for c in cells):
            continue
        alpha_handles.append(
            Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=alpha_colors[a], markersize=10,
                   label=f"α = {a:.2f}"))
    leg1 = ax.legend(handles=alpha_handles, loc="lower right", fontsize=9,
                     title="Train fraction", title_fontsize=9)
    ax.add_artist(leg1)

    # Legend for K (size)
    size_ks = [2, 10, 50, 97]
    size_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#888",
               markersize=np.sqrt(k_to_size(k)) * 0.8,
               label=f"K = {k}")
        for k in size_ks
    ]
    ax.legend(handles=size_handles, loc="upper left", fontsize=9,
              title="Clients (K)", title_fontsize=9)

    # Diagonal
    all_vals = []
    for c in cells:
        for v in [parse_inf(c["cent_full"]["summary"]["t_grok_mean"]),
                  parse_inf(c["fl_iid"]["summary"]["t_grok_mean"])]:
            if v < float("inf"):
                all_vals.append(v)
    lo = min(all_vals) * 0.9
    hi = max(all_vals) * 1.05
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.3, linewidth=1)
    ax.fill_between([lo, hi], [lo, hi], [hi, hi], alpha=0.04, color="red")
    ax.fill_between([lo, hi], [lo, lo], [lo, hi], alpha=0.04, color="green")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")

    ax.set_xlabel(r"Centralized $T_\mathrm{grok}$ (steps)", fontsize=12)
    ax.set_ylabel(r"FL IID $T_\mathrm{grok}$ (steps)", fontsize=12)
    ax.set_title("FL vs Centralized Grokking Time", fontsize=14)
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "exp2_fl_vs_centralized.png"),
                bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/exp2_fl_vs_centralized.png")


# ── Figure 4: Representative training curves ─────────────────────────────────

def plot_training_curves(cells):
    """Test accuracy curves for 3 representative (alpha, K) cells.

    Panels:
      (a) α=0.5, K=2  — easy case, all three conditions grok
      (b) α=0.3, K=10 — aggregation rescues (FL groks, reduced doesn't)
      (c) α=0.25, K=97 — fragmentation overwhelms (FL fails too)
    """
    panels = [
        {"alpha": 0.5, "K": 2, "title": r"$\alpha=0.50,\ K=2$ — all grok",
         "xlim": 30000},
        {"alpha": 0.3, "K": 10, "title": r"$\alpha=0.30,\ K=10$ — aggregation rescues",
         "xlim": 30000},
        {"alpha": 0.25, "K": 97, "title": r"$\alpha=0.25,\ K=97$ — fragmentation wins",
         "xlim": 50000},
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)

    cond_styles = {
        "cent_full": {"color": "#2196F3", "linewidth": 2.0, "label": "Centralized (full data)"},
        "cent_reduced": {"color": "#e74c3c", "linewidth": 1.5, "label": "Centralized (1/K data)",
                         "linestyle": "--"},
        "fl_iid": {"color": "#4CAF50", "linewidth": 2.0, "label": "FL IID (K clients)"},
    }

    for ax, panel in zip(axes, panels):
        alpha, K = panel["alpha"], panel["K"]
        # Find the cell
        match = [c for c in cells if c["alpha"] == alpha and c["K"] == K]
        if not match:
            ax.set_title(panel["title"])
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center")
            continue

        cell = match[0]
        seed_idx = 0  # use first seed for curves

        # Centralized full
        cf_path = cell["cent_full"]["per_seed"][seed_idx]["history_path"]
        if os.path.exists(cf_path):
            h = load_history(cf_path)
            steps = np.array(h["epoch"])
            ax.plot(steps, h["test_acc"], **cond_styles["cent_full"])

        # Centralized reduced
        cr_path = cell["cent_reduced"]["per_seed"][seed_idx]["history_path"]
        if os.path.exists(cr_path):
            h = load_history(cr_path)
            steps = np.array(h["epoch"])
            ax.plot(steps, h["test_acc"], **cond_styles["cent_reduced"])

        # FL IID — need to find the file
        fl_seed = cell["fl_iid"]["per_seed"][seed_idx]
        # Construct FL path from naming convention
        seed_val = [42, 123, 456][seed_idx]
        fl_path = (f"results/experiments/exp2_aggregation/fl_iid/"
                   f"history_fed_addition_gd_p97_N256_a{alpha}_K{K}"
                   f"_le5_ft1.0_iid_s{seed_val}.json")
        if os.path.exists(fl_path):
            h = load_history(fl_path)
            steps = np.array(h["total_steps"])
            # Subsample for readability (FL has 10k+ points)
            stride = max(1, len(steps) // 1000)
            ax.plot(steps[::stride], np.array(h["test_acc"])[::stride],
                    **cond_styles["fl_iid"])

        ax.set_title(panel["title"], fontsize=11)
        ax.set_xlabel("Gradient steps")
        ax.axhline(95, color="gray", linestyle=":", alpha=0.4, linewidth=0.8)
        ax.set_ylim(-2, 105)
        ax.set_xlim(0, panel.get("xlim", 100000))

    axes[0].set_ylabel("Test accuracy (%)")
    # Single legend at top
    handles = [Line2D([0], [0], **{k: v for k, v in s.items() if k != "label"},
                      label=s["label"])
               for s in cond_styles.values()]
    fig.legend(handles=handles, loc="upper center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, 1.05))
    fig.suptitle("Representative Training Dynamics", fontsize=14, y=1.10)
    fig.subplots_adjust(wspace=0.08)
    plt.savefig(os.path.join(OUTPUT_DIR, "exp2_training_curves.png"),
                bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/exp2_training_curves.png")


# ── Figure 5: Grokking success — 2-panel (FL vs reduced only) ────────────────

def plot_grokking_comparison(cells):
    """Side-by-side heatmaps: FL IID vs Centralized-reduced grokking rate.
    Drops the cent_full panel (trivially 100% above α_crit).
    """
    alphas = sorted(set(c["alpha"] for c in cells))
    ks = sorted(set(c["K"] for c in cells))

    conditions = ["cent_reduced", "fl_iid"]
    cond_labels = ["Centralized (1/K data)", "FL IID (K clients)"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax, cond, label in zip(axes, conditions, cond_labels):
        grid = np.full((len(alphas), len(ks)), np.nan)
        for c in cells:
            ai = alphas.index(c["alpha"])
            ki = ks.index(c["K"])
            summary = c[cond]["summary"]
            grid[ai, ki] = summary["n_grokked"] / summary["n_seeds"]

        im = ax.imshow(grid, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1,
                       origin="lower")
        ax.set_xticks(range(len(ks)))
        ax.set_xticklabels(ks)
        ax.set_yticks(range(len(alphas)))
        ax.set_yticklabels([f"{a:.2f}" for a in alphas])
        ax.set_xlabel("K (number of clients)")
        ax.set_title(label, fontsize=12)

        for i in range(len(alphas)):
            for j in range(len(ks)):
                val = grid[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                            fontsize=10, fontweight="bold",
                            color="black" if 0.3 < val < 0.7 else "white")

    axes[0].set_ylabel(r"$\alpha$ (train fraction)")
    fig.colorbar(im, ax=axes, label="Fraction of seeds that grokked",
                 shrink=0.7, pad=0.04)
    fig.suptitle("Grokking Success: FL Aggregation vs Data Reduction",
                 fontsize=14, y=1.02)
    plt.savefig(os.path.join(OUTPUT_DIR, "exp2_grokking_comparison.png"),
                bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/exp2_grokking_comparison.png")


# ── Figure 6: IPR evolution comparing conditions ─────────────────────────────

def _get_fl_history_path(alpha, K, seed_val):
    """Construct FL history file path from parameters."""
    return (f"results/experiments/exp2_aggregation/fl_iid/"
            f"history_fed_addition_gd_p97_N256_a{alpha}_K{K}"
            f"_le5_ft1.0_iid_s{seed_val}.json")


def plot_ipr_evolution(cells):
    """IPR (Fourier structure) over time: cent_full vs FL vs cent_reduced.

    Panels show representative cells at different regimes.
    """
    panels = [
        {"alpha": 0.5, "K": 2, "title": r"$\alpha=0.50,\ K=2$", "xlim": 30000},
        {"alpha": 0.3, "K": 10, "title": r"$\alpha=0.30,\ K=10$", "xlim": 30000},
        {"alpha": 0.3, "K": 97, "title": r"$\alpha=0.30,\ K=97$", "xlim": 50000},
        {"alpha": 0.25, "K": 97, "title": r"$\alpha=0.25,\ K=97$", "xlim": 50000},
    ]

    cond_styles = {
        "cent_full": {"color": "#2196F3", "linewidth": 2.0,
                      "label": "Centralized (full)"},
        "cent_reduced": {"color": "#e74c3c", "linewidth": 1.5,
                         "label": "Centralized (1/K)", "linestyle": "--"},
        "fl_iid": {"color": "#4CAF50", "linewidth": 2.0,
                   "label": "FL IID"},
    }

    fig, axes = plt.subplots(1, len(panels), figsize=(4.5 * len(panels), 4.5),
                             sharey=True)
    seed_val = 42

    for ax, panel in zip(axes, panels):
        alpha, K = panel["alpha"], panel["K"]
        cell = next((c for c in cells
                     if c["alpha"] == alpha and c["K"] == K), None)
        if cell is None:
            ax.set_title(panel["title"])
            continue

        # Centralized full
        cf_path = cell["cent_full"]["per_seed"][0]["history_path"]
        if os.path.exists(cf_path):
            h = load_history(cf_path)
            ax.plot(h["epoch"], h["ipr"], **cond_styles["cent_full"])

        # Centralized reduced
        cr_path = cell["cent_reduced"]["per_seed"][0]["history_path"]
        if os.path.exists(cr_path):
            h = load_history(cr_path)
            ax.plot(h["epoch"], h["ipr"], **cond_styles["cent_reduced"])

        # FL IID
        fl_path = _get_fl_history_path(alpha, K, seed_val)
        if os.path.exists(fl_path):
            h = load_history(fl_path)
            steps = np.array(h["total_steps"])
            ipr = np.array(h["ipr"])
            stride = max(1, len(steps) // 1000)
            ax.plot(steps[::stride], ipr[::stride], **cond_styles["fl_iid"])

        ax.set_title(panel["title"], fontsize=11)
        ax.set_xlabel("Gradient steps")
        ax.set_xlim(0, panel["xlim"])

    axes[0].set_ylabel("IPR (Fourier structure)")
    handles = [Line2D([0], [0], **{k: v for k, v in s.items() if k != "label"},
                      label=s["label"])
               for s in cond_styles.values()]
    fig.legend(handles=handles, loc="upper center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, 1.05))
    fig.suptitle("Fourier Structure Emergence Across Conditions",
                 fontsize=14, y=1.10)
    fig.subplots_adjust(wspace=0.08)
    plt.savefig(os.path.join(OUTPUT_DIR, "exp2_ipr_evolution.png"),
                bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/exp2_ipr_evolution.png")


# ── Figure 7: Phase portrait (IPR vs test accuracy) ──────────────────────────

def plot_phase_portrait(cells):
    """Trajectories in (IPR, test_acc) space for cent_full vs FL IID.

    Shows whether FL follows the same mechanistic pathway as centralized.
    """
    # Pick a range of (alpha, K) pairs that grok in both conditions
    portrait_cells = [
        {"alpha": 0.5, "K": 5, "label": r"$\alpha$=0.5, K=5"},
        {"alpha": 0.35, "K": 10, "label": r"$\alpha$=0.35, K=10"},
        {"alpha": 0.3, "K": 10, "label": r"$\alpha$=0.3, K=10"},
        {"alpha": 0.3, "K": 97, "label": r"$\alpha$=0.3, K=97"},
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    seed_val = 42

    cmap = plt.cm.viridis
    n = len(portrait_cells)
    cell_colors = [cmap(i / max(n - 1, 1)) for i in range(n)]

    for ax, (cond, cond_label) in zip(axes, [("cent_full", "Centralized (full)"),
                                              ("fl_iid", "FL IID")]):
        for pc, color in zip(portrait_cells, cell_colors):
            alpha, K = pc["alpha"], pc["K"]
            cell = next((c for c in cells
                         if c["alpha"] == alpha and c["K"] == K), None)
            if cell is None:
                continue

            if cond == "cent_full":
                path = cell["cent_full"]["per_seed"][0]["history_path"]
                if not os.path.exists(path):
                    continue
                h = load_history(path)
                ipr = np.array(h["ipr"])
                acc = np.array(h["test_acc"])
            else:
                fl_path = _get_fl_history_path(alpha, K, seed_val)
                if not os.path.exists(fl_path):
                    continue
                h = load_history(fl_path)
                ipr = np.array(h["ipr"])
                acc = np.array(h["test_acc"])
                # Subsample
                stride = max(1, len(ipr) // 1000)
                ipr = ipr[::stride]
                acc = acc[::stride]

            ax.plot(ipr, acc, color=color, linewidth=1.5, alpha=0.8,
                    label=pc["label"])
            # Mark endpoint
            ax.scatter(ipr[-1], acc[-1], color=color, s=40, zorder=5,
                       edgecolors="white", linewidths=0.5)

        ax.set_xlabel("IPR (Fourier structure)")
        ax.set_ylabel("Test accuracy (%)")
        ax.set_title(cond_label, fontsize=12)
        ax.set_xlim(-0.01, 0.55)
        ax.set_ylim(-2, 105)
        ax.axhline(95, color="gray", linestyle=":", alpha=0.3)
        ax.legend(fontsize=8, loc="center right")

    fig.suptitle("Phase Portrait: Same Pathway to Grokking?",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "exp2_phase_portrait.png"),
                bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/exp2_phase_portrait.png")


# ── Figure 8: Client drift vs K ──────────────────────────────────────────────

def plot_client_drift(cells):
    """Final client drift and weight divergence vs K, one line per alpha.

    Uses mean_client_drift and client_weight_divergence from FL history files.
    """
    alphas = sorted(set(c["alpha"] for c in cells))
    ks = sorted(set(c["K"] for c in cells))
    seed_val = 42

    colors = {0.2: "#888888", 0.25: "#9C27B0", 0.3: "#2196F3",
              0.35: "#009688", 0.5: "#FF9800"}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for alpha in alphas:
        k_vals = []
        drifts = []
        divergences = []
        for K in ks:
            fl_path = _get_fl_history_path(alpha, K, seed_val)
            if not os.path.exists(fl_path):
                continue
            h = load_history(fl_path)
            k_vals.append(K)
            drifts.append(h["mean_client_drift"][-1])
            divergences.append(h["client_weight_divergence"][-1])

        if k_vals:
            axes[0].plot(k_vals, drifts, marker="o", color=colors.get(alpha, "gray"),
                         label=f"α = {alpha:.2f}", linewidth=1.8, markersize=5)
            axes[1].plot(k_vals, divergences, marker="o",
                         color=colors.get(alpha, "gray"),
                         label=f"α = {alpha:.2f}", linewidth=1.8, markersize=5)

    for ax, ylabel, title in zip(
        axes,
        ["Mean client drift (L2)", "Client weight divergence"],
        ["Client Drift vs K", "Weight Divergence vs K"],
    ):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(ks)
        ax.set_xticklabels(ks)
        ax.set_xlabel("K (number of clients)")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    # Mark the failure zone on drift plot
    # Find the drift value where FL starts failing (alpha=0.25, K=50)
    fail_path = _get_fl_history_path(0.25, 50, seed_val)
    if os.path.exists(fail_path):
        h = load_history(fail_path)
        fail_drift = h["mean_client_drift"][-1]
        axes[0].axhline(fail_drift, color="red", linestyle=":", alpha=0.5,
                        linewidth=1)
        axes[0].text(2.2, fail_drift * 1.15, "FL failure onset",
                     fontsize=8, color="red", alpha=0.7)

    fig.suptitle("Client Heterogeneity Under FedAvg", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "exp2_client_drift.png"),
                bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/exp2_client_drift.png")


SEEDS = [42, 123, 456]

# ── Shared helpers for multi-seed loading ─────────────────────────────────────

def _load_all_seeds(cell, alpha, K):
    """Load histories for all 3 seeds for a given cell.

    Returns dict: {cond: [list of history dicts per seed]}.
    Skips missing files.
    """
    result = {}
    for cond in ["cent_full", "cent_reduced", "fl_iid"]:
        hists = []
        for si, seed_val in enumerate(SEEDS):
            if cond in ("cent_full", "cent_reduced"):
                path = cell[cond]["per_seed"][si]["history_path"]
            else:
                path = _get_fl_history_path(alpha, K, seed_val)
            if os.path.exists(path):
                hists.append(load_history(path))
        result[cond] = hists
    return result


def _steps_for(h, cond):
    """Return the step array for a history dict."""
    return np.array(h["total_steps"] if cond == "fl_iid" else h["epoch"])


def _subsample(arr, n=1000):
    """Subsample an array to at most n points."""
    stride = max(1, len(arr) // n)
    return arr[::stride]


# ── Figure 9: Loss dynamics ──────────────────────────────────────────────────

def plot_loss_dynamics(cells):
    """Train and test loss for representative cells.

    Shows the classic grokking signature: train loss drops early,
    test loss follows much later.
    """
    panels = [
        {"alpha": 0.5, "K": 2, "title": r"$\alpha=0.50,\ K=2$", "xlim": 30000},
        {"alpha": 0.3, "K": 10, "title": r"$\alpha=0.30,\ K=10$", "xlim": 30000},
        {"alpha": 0.25, "K": 97, "title": r"$\alpha=0.25,\ K=97$", "xlim": 50000},
    ]

    cond_styles = {
        "cent_full": {"color": "#2196F3", "label": "Centralized (full)"},
        "fl_iid": {"color": "#4CAF50", "label": "FL IID"},
        "cent_reduced": {"color": "#e74c3c", "label": "Centralized (1/K)"},
    }

    fig, axes = plt.subplots(2, len(panels), figsize=(5 * len(panels), 8),
                             sharex="col")

    for col, panel in enumerate(panels):
        alpha, K = panel["alpha"], panel["K"]
        cell = next((c for c in cells
                     if c["alpha"] == alpha and c["K"] == K), None)
        if cell is None:
            continue

        for cond, style in cond_styles.items():
            if cond in ("cent_full", "cent_reduced"):
                path = cell[cond]["per_seed"][0]["history_path"]
                if not os.path.exists(path):
                    continue
                h = load_history(path)
                steps = np.array(h["epoch"])
            else:
                fl_path = _get_fl_history_path(alpha, K, SEEDS[0])
                if not os.path.exists(fl_path):
                    continue
                h = load_history(fl_path)
                steps = np.array(h["total_steps"])

            train_loss = np.array(h["train_loss"])
            test_loss = np.array(h["test_loss"])
            s = _subsample(steps)
            train_loss = _subsample(train_loss)
            test_loss = _subsample(test_loss)

            axes[0, col].plot(s, train_loss, color=style["color"],
                              linewidth=1.5, label=style["label"])
            axes[1, col].plot(s, test_loss, color=style["color"],
                              linewidth=1.5, linestyle="--",
                              label=style["label"])

        axes[0, col].set_title(panel["title"], fontsize=11)
        axes[1, col].set_xlabel("Gradient steps")
        axes[0, col].set_xlim(0, panel["xlim"])
        axes[1, col].set_xlim(0, panel["xlim"])
        for row in range(2):
            axes[row, col].set_yscale("log")
            axes[row, col].grid(alpha=0.2)

    axes[0, 0].set_ylabel("Train loss")
    axes[1, 0].set_ylabel("Test loss")

    handles = [Line2D([0], [0], color=s["color"], linewidth=1.5,
                      label=s["label"])
               for s in cond_styles.values()]
    fig.legend(handles=handles, loc="upper center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, 1.03))
    fig.suptitle("Loss Dynamics: Memorization Before Generalization",
                 fontsize=14, y=1.06)
    fig.subplots_adjust(hspace=0.15, wspace=0.25)
    plt.savefig(os.path.join(OUTPUT_DIR, "exp2_loss_dynamics.png"),
                bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/exp2_loss_dynamics.png")


# ── Figure 10: Weight norm evolution ─────────────────────────────────────────

def plot_weight_norm_evolution(cells):
    """Weight norms (layer 1 + layer 2) over time.

    Gromov's theory: weight decay compresses norms, driving the phase transition.
    """
    panels = [
        {"alpha": 0.5, "K": 2, "title": r"$\alpha=0.50,\ K=2$", "xlim": 30000},
        {"alpha": 0.3, "K": 10, "title": r"$\alpha=0.30,\ K=10$", "xlim": 30000},
        {"alpha": 0.3, "K": 97, "title": r"$\alpha=0.30,\ K=97$", "xlim": 50000},
        {"alpha": 0.25, "K": 97, "title": r"$\alpha=0.25,\ K=97$", "xlim": 50000},
    ]

    cond_styles = {
        "cent_full": {"color": "#2196F3", "linewidth": 2.0,
                      "label": "Centralized (full)"},
        "cent_reduced": {"color": "#e74c3c", "linewidth": 1.5,
                         "label": "Centralized (1/K)", "linestyle": "--"},
        "fl_iid": {"color": "#4CAF50", "linewidth": 2.0,
                   "label": "FL IID"},
    }

    fig, axes = plt.subplots(2, len(panels),
                             figsize=(4.5 * len(panels), 8),
                             sharex="col")

    for col, panel in enumerate(panels):
        alpha, K = panel["alpha"], panel["K"]
        cell = next((c for c in cells
                     if c["alpha"] == alpha and c["K"] == K), None)
        if cell is None:
            continue

        for cond, style in cond_styles.items():
            if cond in ("cent_full", "cent_reduced"):
                path = cell[cond]["per_seed"][0]["history_path"]
                if not os.path.exists(path):
                    continue
                h = load_history(path)
                steps = np.array(h["epoch"])
                w1 = np.array(h["weight_norm_layer1"])
                w2 = np.array(h["weight_norm_layer2"])
            else:
                fl_path = _get_fl_history_path(alpha, K, SEEDS[0])
                if not os.path.exists(fl_path):
                    continue
                h = load_history(fl_path)
                steps = _subsample(np.array(h["total_steps"]))
                w1 = _subsample(np.array(h["weight_norm_layer1"]))
                w2 = _subsample(np.array(h["weight_norm_layer2"]))

            plot_kw = {k: v for k, v in style.items() if k != "label"}
            axes[0, col].plot(steps, w1, **plot_kw, label=style["label"])
            axes[1, col].plot(steps, w2, **plot_kw, label=style["label"])

        axes[0, col].set_title(panel["title"], fontsize=11)
        axes[1, col].set_xlabel("Gradient steps")
        for row in range(2):
            axes[row, col].set_xlim(0, panel["xlim"])
            axes[row, col].grid(alpha=0.2)

    axes[0, 0].set_ylabel(r"$\|W_1\|_F$")
    axes[1, 0].set_ylabel(r"$\|W_2\|_F$")

    handles = [Line2D([0], [0], **{k: v for k, v in s.items() if k != "label"},
                      label=s["label"])
               for s in cond_styles.values()]
    fig.legend(handles=handles, loc="upper center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, 1.03))
    fig.suptitle("Weight Norm Compression Drives Grokking",
                 fontsize=14, y=1.06)
    fig.subplots_adjust(hspace=0.15, wspace=0.25)
    plt.savefig(os.path.join(OUTPUT_DIR, "exp2_weight_norms.png"),
                bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/exp2_weight_norms.png")


# ── Figure 11: Seed consistency (mean ± std bands) ───────────────────────────

def plot_seed_consistency(cells):
    """Test accuracy with mean ± std shaded bands across 3 seeds.

    Shows reproducibility and whether FL increases variance near the boundary.
    """
    panels = [
        {"alpha": 0.5, "K": 5, "title": r"$\alpha=0.50,\ K=5$", "xlim": 30000},
        {"alpha": 0.35, "K": 10, "title": r"$\alpha=0.35,\ K=10$", "xlim": 30000},
        {"alpha": 0.3, "K": 10, "title": r"$\alpha=0.30,\ K=10$", "xlim": 50000},
        {"alpha": 0.3, "K": 50, "title": r"$\alpha=0.30,\ K=50$", "xlim": 50000},
    ]

    cond_info = [
        ("cent_full", "Centralized (full)", "#2196F3"),
        ("fl_iid", "FL IID", "#4CAF50"),
    ]

    fig, axes = plt.subplots(1, len(panels),
                             figsize=(4.5 * len(panels), 4.5), sharey=True)

    for ax, panel in zip(axes, panels):
        alpha, K = panel["alpha"], panel["K"]
        cell = next((c for c in cells
                     if c["alpha"] == alpha and c["K"] == K), None)
        if cell is None:
            ax.set_title(panel["title"])
            continue

        all_seeds = _load_all_seeds(cell, alpha, K)

        for cond, label, color in cond_info:
            hists = all_seeds[cond]
            if len(hists) < 2:
                continue

            # Align to common step grid
            if cond == "fl_iid":
                # All FL runs have same round structure -> same total_steps
                steps = np.array(hists[0]["total_steps"])
                accs = np.array([h["test_acc"] for h in hists])
                stride = max(1, len(steps) // 500)
                steps = steps[::stride]
                accs = accs[:, ::stride]
            else:
                steps = np.array(hists[0]["epoch"])
                accs = np.array([h["test_acc"] for h in hists])

            mean = accs.mean(axis=0)
            std = accs.std(axis=0)
            ax.plot(steps, mean, color=color, linewidth=1.8, label=label)
            ax.fill_between(steps, mean - std, mean + std,
                            color=color, alpha=0.15)

        ax.set_title(panel["title"], fontsize=11)
        ax.set_xlabel("Gradient steps")
        ax.axhline(95, color="gray", linestyle=":", alpha=0.3)
        ax.set_ylim(-2, 105)
        ax.set_xlim(0, panel["xlim"])

    axes[0].set_ylabel("Test accuracy (%)")
    handles = [Line2D([0], [0], color=c, linewidth=1.8, label=l)
               for _, l, c in cond_info]
    fig.legend(handles=handles, loc="upper center", ncol=2, fontsize=9,
               bbox_to_anchor=(0.5, 1.05))
    fig.suptitle("Seed Consistency: Mean ± Std (3 seeds)",
                 fontsize=14, y=1.10)
    fig.subplots_adjust(wspace=0.08)
    plt.savefig(os.path.join(OUTPUT_DIR, "exp2_seed_consistency.png"),
                bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/exp2_seed_consistency.png")


# ── Figure 12: Client drift over time ────────────────────────────────────────

def plot_drift_over_time(cells):
    """Client drift trajectory overlaid with test accuracy.

    Shows whether drift spikes during the grokking transition.
    """
    panels = [
        {"alpha": 0.5, "K": 5, "title": r"$\alpha=0.50,\ K=5$", "xlim": 30000},
        {"alpha": 0.3, "K": 10, "title": r"$\alpha=0.30,\ K=10$", "xlim": 30000},
        {"alpha": 0.3, "K": 50, "title": r"$\alpha=0.30,\ K=50$", "xlim": 50000},
        {"alpha": 0.25, "K": 97, "title": r"$\alpha=0.25,\ K=97$", "xlim": 50000},
    ]

    fig, axes = plt.subplots(1, len(panels),
                             figsize=(4.5 * len(panels), 4.5))

    for ax, panel in zip(axes, panels):
        alpha, K = panel["alpha"], panel["K"]
        fl_path = _get_fl_history_path(alpha, K, SEEDS[0])
        if not os.path.exists(fl_path):
            ax.set_title(panel["title"])
            continue
        h = load_history(fl_path)

        steps = _subsample(np.array(h["total_steps"]))
        drift = _subsample(np.array(h["mean_client_drift"]))
        acc = _subsample(np.array(h["test_acc"]))

        ax.plot(steps, drift, color="#e74c3c", linewidth=1.5,
                label="Client drift")
        ax.set_ylabel("Mean client drift (L2)", color="#e74c3c")
        ax.tick_params(axis="y", labelcolor="#e74c3c")

        ax2 = ax.twinx()
        ax2.plot(steps, acc, color="#2196F3", linewidth=1.5, alpha=0.7,
                 label="Test accuracy")
        ax2.set_ylabel("Test accuracy (%)", color="#2196F3")
        ax2.tick_params(axis="y", labelcolor="#2196F3")
        ax2.set_ylim(-2, 105)

        ax.set_title(panel["title"], fontsize=11)
        ax.set_xlabel("Gradient steps")
        ax.set_xlim(0, panel["xlim"])

    # Combined legend
    handles = [
        Line2D([0], [0], color="#e74c3c", linewidth=1.5, label="Client drift"),
        Line2D([0], [0], color="#2196F3", linewidth=1.5, label="Test accuracy"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, fontsize=9,
               bbox_to_anchor=(0.5, 1.05))
    fig.suptitle("Client Drift Dynamics During Grokking",
                 fontsize=14, y=1.10)
    fig.subplots_adjust(wspace=0.55)
    plt.savefig(os.path.join(OUTPUT_DIR, "exp2_drift_over_time.png"),
                bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/exp2_drift_over_time.png")


# ── Figure 13: Aggregation benefit heatmap ───────────────────────────────────

def plot_aggregation_benefit(cells):
    """Heatmap of T_grok(cent_reduced) / T_grok(FL).

    Values > 1 mean FL is faster than training on the reduced dataset alone.
    Inf means reduced never grokked but FL did (maximum benefit).
    """
    alphas = sorted(set(c["alpha"] for c in cells))
    ks = sorted(set(c["K"] for c in cells))

    grid = np.full((len(alphas), len(ks)), np.nan)
    annotations = [[None] * len(ks) for _ in range(len(alphas))]

    for c in cells:
        ai = alphas.index(c["alpha"])
        ki = ks.index(c["K"])
        t_fl = parse_inf(c["fl_iid"]["summary"]["t_grok_mean"])
        t_cr = parse_inf(c["cent_reduced"]["summary"]["t_grok_mean"])
        cf_groks = c["cent_full"]["summary"]["n_grokked"] > 0

        if not cf_groks:
            annotations[ai][ki] = "below\nboundary"
            grid[ai, ki] = 0
        elif t_fl == float("inf") and t_cr == float("inf"):
            annotations[ai][ki] = "neither\ngroks"
            grid[ai, ki] = 0
        elif t_fl == float("inf"):
            annotations[ai][ki] = "FL\nfails"
            grid[ai, ki] = 0
        elif t_cr == float("inf"):
            annotations[ai][ki] = "∞"
            grid[ai, ki] = 10  # cap for colormap
        else:
            ratio = t_cr / t_fl
            grid[ai, ki] = ratio
            annotations[ai][ki] = f"{ratio:.1f}×"

    fig, ax = plt.subplots(figsize=(8, 5))

    im = ax.imshow(grid, aspect="auto", cmap="YlOrRd", vmin=0, vmax=10,
                   origin="lower")
    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels(ks)
    ax.set_yticks(range(len(alphas)))
    ax.set_yticklabels([f"{a:.2f}" for a in alphas])
    ax.set_xlabel("K (number of clients)")
    ax.set_ylabel(r"$\alpha$ (train fraction)")

    for i in range(len(alphas)):
        for j in range(len(ks)):
            txt = annotations[i][j]
            if txt is not None:
                val = grid[i, j]
                color = "white" if val > 5 else "black"
                if txt in ("below\nboundary", "neither\ngroks", "FL\nfails"):
                    color = "#555555"
                    fontsize = 8
                    style = "italic"
                else:
                    fontsize = 10
                    style = "normal"
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=fontsize, fontweight="bold",
                        color=color, style=style)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(r"$T_\mathrm{grok}^\mathrm{reduced}\; /\; T_\mathrm{grok}^\mathrm{FL}$")

    ax.set_title("Aggregation Benefit: How Much Faster Does FL Grok?",
                 fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "exp2_aggregation_benefit.png"),
                bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/exp2_aggregation_benefit.png")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cells = load_all_cells()
    print(f"Loaded {len(cells)} cells")

    plot_phase_diagram(cells)        # Fig 1: phase diagram
    plot_slowdown_ratio(cells)       # Fig 2: ratio vs K
    plot_fl_vs_centralized(cells)    # Fig 3: scatter (improved)
    plot_training_curves(cells)      # Fig 4: representative curves
    plot_grokking_comparison(cells)  # Fig 5: 2-panel heatmap
    plot_ipr_evolution(cells)        # Fig 6: IPR over time
    plot_phase_portrait(cells)       # Fig 7: IPR vs test acc
    plot_client_drift(cells)         # Fig 8: drift vs K
    plot_loss_dynamics(cells)        # Fig 9: train/test loss
    plot_weight_norm_evolution(cells) # Fig 10: weight norms
    plot_seed_consistency(cells)     # Fig 11: mean ± std bands
    plot_drift_over_time(cells)      # Fig 12: drift trajectory
    plot_aggregation_benefit(cells)  # Fig 13: benefit heatmap

    print("\nDone! All figures saved to", OUTPUT_DIR)
