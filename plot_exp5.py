"""Plot Experiment 5 results: Algorithm rescue."""

import json
import os
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

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

RESULTS_DIR = "results/exp5_algorithms"
OUTPUT_DIR = "results/exp5_algorithms/figures"

HARD_SETTINGS = {
    "H1": "Easy rescue\nα=0.25, E=25, IID",
    "H2": "Hard rescue\nα=0.25, E=25, non-IID",
    "H3": "Acceleration\nα=0.3, E=50, non-IID",
}


def parse_inf(v):
    if v == "inf" or v == float("inf"):
        return float("inf")
    return float(v)


def load_exp5_results():
    """Load all exp5 summary JSONs."""
    results = {}
    for label in ["H1", "H2", "H3"]:
        results[label] = {}
        for f in sorted(glob.glob(os.path.join(RESULTS_DIR, f"exp5_{label}_*.json"))):
            with open(f) as fh:
                d = json.load(fh)
            algo = os.path.basename(f).replace(f"exp5_{label}_", "").replace(".json", "")
            results[label][algo] = d
    return results


def plot_algorithm_rescue(results):
    """Main figure: horizontal bar chart of T_grok across algorithms per hard setting."""
    algo_order = [
        "FedAvg",
        "FedProx-0.001", "FedProx-0.01", "FedProx-0.1", "FedProx-1.0",
        "FedAdam-0.01", "FedAdam-0.1", "FedAdam-1.0",
        "FedAvg+WD-0.01", "FedAvg+WD-0.1", "FedAvg+WD-1.0",
    ]
    algo_labels = [
        "FedAvg (baseline)",
        r"FedProx $\mu$=0.001", r"FedProx $\mu$=0.01",
        r"FedProx $\mu$=0.1", r"FedProx $\mu$=1.0",
        r"FedAdam $\tau$=0.01", r"FedAdam $\tau$=0.1", r"FedAdam $\tau$=1.0",
        r"FedAvg+WD $\lambda$=0.01", r"FedAvg+WD $\lambda$=0.1",
        r"FedAvg+WD $\lambda$=1.0",
    ]
    algo_colors = [
        "#757575",
        "#BBDEFB", "#64B5F6", "#1E88E5", "#0D47A1",
        "#C8E6C9", "#4CAF50", "#1B5E20",
        "#FFCDD2", "#EF5350", "#B71C1C",
    ]

    settings = ["H1", "H2", "H3"]
    setting_titles = {
        "H1": r"H1: Easy rescue ($\alpha$=0.25, $E$=25, IID)",
        "H2": r"H2: Hard rescue ($\alpha$=0.25, $E$=25, non-IID)",
        "H3": r"H3: Acceleration ($\alpha$=0.3, $E$=50, non-IID)",
    }
    cent_tgrok = {"H1": 25133, "H2": 25133, "H3": 12833}
    budget = 80000

    fig, axes = plt.subplots(1, 3, figsize=(18, 7), sharey=True)

    y = np.arange(len(algo_order))[::-1]  # top-to-bottom

    for ax, setting in zip(axes, settings):
        if setting not in results:
            continue

        for i, (algo, color, label) in enumerate(zip(algo_order, algo_colors, algo_labels)):
            if algo not in results[setting]:
                continue

            s = results[setting][algo]["summary"]
            t = parse_inf(s["t_grok_mean"])
            t_std = parse_inf(s["t_grok_std"])
            n_grok = s["n_grokked"]
            n_seeds = s["n_seeds"]
            grokked = t < float("inf")

            if grokked:
                bar_val = t
                bar_err = t_std if t_std < float("inf") else 0
            else:
                bar_val = budget
                bar_err = 0

            bar = ax.barh(y[i], bar_val, 0.7, xerr=bar_err, capsize=2,
                          color=color, edgecolor="white", linewidth=0.5)

            if not grokked:
                bar[0].set_hatch("//")
                bar[0].set_edgecolor("#999999")

            # Annotate grok fraction
            x_text = min(bar_val, budget) + 800
            frac_str = f"{n_grok}/{n_seeds}"
            ax.text(x_text, y[i], frac_str, va="center", fontsize=8,
                    fontweight="bold" if n_grok == n_seeds and grokked else "normal",
                    color="#333333")

        # Centralized baseline
        if setting in cent_tgrok:
            ax.axvline(cent_tgrok[setting], color="#FF9800", linestyle="-",
                       alpha=0.7, linewidth=1.5, zorder=0)
            ax.text(cent_tgrok[setting] + 500, len(algo_order) - 0.3,
                    "centralized", va="top", fontsize=8,
                    color="#FF9800", fontstyle="italic")

        ax.set_xlim(0, budget * 1.15)
        ax.set_title(setting_titles[setting], fontsize=11)
        ax.set_xlabel(r"$T_{\mathrm{grok}}$ (gradient steps)")

        # Group separators
        for sep_y in [y[0] + 0.5, y[4] + 0.5, y[7] + 0.5]:
            ax.axhline(sep_y, color="#E0E0E0", linestyle="-", linewidth=0.5)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(algo_labels, fontsize=9)

    fig.suptitle("Algorithm Rescue: Can FL Algorithms Recover Grokking?",
                 fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "exp5_algorithm_rescue.png"),
                bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/exp5_algorithm_rescue.png")


def plot_algorithm_rescue_vertical(results):
    """Original vertical bar chart version."""
    algo_order = [
        "FedAvg",
        "FedProx-0.001", "FedProx-0.01", "FedProx-0.1", "FedProx-1.0",
        "FedAdam-0.01", "FedAdam-0.1", "FedAdam-1.0",
        "FedAvg+WD-0.01", "FedAvg+WD-0.1", "FedAvg+WD-1.0",
    ]
    algo_labels = [
        "FedAvg\n(baseline)",
        "FedProx\n" + r"$\mu$=0.001", "FedProx\n" + r"$\mu$=0.01",
        "FedProx\n" + r"$\mu$=0.1", "FedProx\n" + r"$\mu$=1.0",
        "FedAdam\n" + r"$\tau$=0.01", "FedAdam\n" + r"$\tau$=0.1",
        "FedAdam\n" + r"$\tau$=1.0",
        "FedAvg+WD\n" + r"$\lambda$=0.01", "FedAvg+WD\n" + r"$\lambda$=0.1",
        "FedAvg+WD\n" + r"$\lambda$=1.0",
    ]
    algo_colors = [
        "#757575",
        "#BBDEFB", "#64B5F6", "#1E88E5", "#0D47A1",
        "#C8E6C9", "#4CAF50", "#1B5E20",
        "#FFCDD2", "#EF5350", "#B71C1C",
    ]

    settings = ["H1", "H2", "H3"]
    setting_titles = {
        "H1": r"H1: Easy rescue ($\alpha$=0.25, $E$=25, IID)",
        "H2": r"H2: Hard rescue ($\alpha$=0.25, $E$=25, non-IID)",
        "H3": r"H3: Acceleration ($\alpha$=0.3, $E$=50, non-IID)",
    }
    cent_tgrok = {"H1": 25133, "H2": 25133, "H3": 12833}
    budget = 80000

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    for ax, setting in zip(axes, settings):
        if setting not in results:
            continue

        t_means, t_stds, colors, labels, grok_fracs = [], [], [], [], []

        for algo, color, label in zip(algo_order, algo_colors, algo_labels):
            if algo not in results[setting]:
                continue

            s = results[setting][algo]["summary"]
            t = parse_inf(s["t_grok_mean"])
            t_std = parse_inf(s["t_grok_std"])
            n_grok = s["n_grokked"]
            n_seeds = s["n_seeds"]

            if t < float("inf") and t < budget:
                t_means.append(t)
                t_stds.append(t_std if t_std < float("inf") else 0)
            else:
                t_means.append(budget)
                t_stds.append(0)

            colors.append(color)
            labels.append(label)
            grok_fracs.append(f"{n_grok}/{n_seeds}")

        x = np.arange(len(t_means))
        bars = ax.bar(x, t_means, 0.7, yerr=t_stds, capsize=3,
                      color=colors, edgecolor="white", linewidth=0.5)

        for i, (bar, t_m, gf) in enumerate(zip(bars, t_means, grok_fracs)):
            if t_m >= budget:
                bar.set_hatch("//")
                bar.set_edgecolor("#999999")
                bar.set_linewidth(0.5)
            y_pos = min(t_m, budget) + 1500
            ax.text(bar.get_x() + bar.get_width()/2, y_pos,
                    gf, ha="center", va="bottom", fontsize=7,
                    fontweight="bold" if gf.startswith("3") else "normal")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_title(setting_titles[setting], fontsize=11)
        ax.axhline(budget, color="gray", linestyle="--", alpha=0.3, linewidth=0.8)

        if setting in cent_tgrok:
            ax.axhline(cent_tgrok[setting], color="#FF9800", linestyle="-",
                       alpha=0.7, linewidth=1.5, zorder=0)
            ax.text(len(t_means) - 0.5, cent_tgrok[setting] + 1000,
                    "centralized", ha="right", va="bottom", fontsize=7,
                    color="#FF9800", fontstyle="italic")

        ax.set_ylim(0, budget * 1.15)
        for sep_x in [0.5, 4.5, 7.5]:
            ax.axvline(sep_x, color="#E0E0E0", linestyle="-", linewidth=0.5)

    axes[0].set_ylabel(r"$T_{\mathrm{grok}}$ (gradient steps)")
    fig.suptitle("Algorithm Rescue: Can FL Algorithms Recover Grokking?",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "exp5_algorithm_rescue_vertical.png"),
                bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/exp5_algorithm_rescue_vertical.png")


def plot_algorithm_speedup(results):
    """Speedup ratio relative to FedAvg baseline (where available)."""
    settings = ["H1", "H2", "H3"]
    algo_order = [
        "FedProx-0.001", "FedAdam-0.01", "FedAdam-0.1",
    ]
    algo_labels = ["FedProx\nμ=0.001", "FedAdam\nτ=0.01", "FedAdam\nτ=0.1"]
    algo_colors = ["#1E88E5", "#C8E6C9", "#4CAF50"]

    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(settings))
    n_algos = len(algo_order)
    width = 0.25

    for i, (algo, label, color) in enumerate(zip(algo_order, algo_labels, algo_colors)):
        speedups = []
        for setting in settings:
            # Get baseline
            if "FedAvg" in results[setting]:
                t_base = parse_inf(results[setting]["FedAvg"]["summary"]["t_grok_mean"])
            else:
                t_base = float("inf")

            if algo in results[setting]:
                t_algo = parse_inf(results[setting][algo]["summary"]["t_grok_mean"])
            else:
                t_algo = float("inf")

            if t_base < float("inf") and t_algo < float("inf"):
                speedups.append(t_base / t_algo)
            elif t_algo < float("inf"):
                speedups.append(float("nan"))  # rescued where baseline failed
            else:
                speedups.append(0)

        bars = ax.bar(x + i * width - width, speedups, width, label=label,
                      color=color, edgecolor="white")
        for bar, s in zip(bars, speedups):
            if s > 0 and not np.isnan(s):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                        f"{s:.1f}×", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([HARD_SETTINGS[s].replace("\n", " — ") for s in settings],
                       fontsize=9)
    ax.set_ylabel("Speedup (T_base / T_algo)")
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
    ax.legend()
    ax.set_title("Speedup Relative to FedAvg Baseline")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "exp5_speedup.png"), bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/exp5_speedup.png")


def plot_algorithm_trajectories(results):
    """Test accuracy trajectories for key algorithms on H1 (legacy)."""
    pass


def _find_histories(raw_dir, frag, seed):
    """Find history files matching a fragment and seed.

    frag=None -> FedAvg (no _mu, _adam_tau, _wd in name)
    frag="mu0.001" -> FedProx mu=0.001
    frag="adam_tau0.001_slr0.01" -> FedAdam tau=0.01
    frag="wd0.01" -> WD lambda=0.01
    """
    matches = []
    for f in sorted(glob.glob(os.path.join(raw_dir, f"*_s{seed}.json"))):
        fname = os.path.basename(f)
        has_mu = "_mu" in fname
        has_adam = "adam_tau" in fname
        has_wd = "_wd" in fname
        if frag is None:
            if not has_mu and not has_adam and not has_wd:
                matches.append(f)
        elif frag in fname:
            matches.append(f)
    return matches


def _normalize_acc(acc):
    """Normalize test/train accuracy to 0-1 scale (histories store 0-100 percentages)."""
    return np.array(acc) / 100.0


def plot_training_curves(results):
    """Test accuracy curves: 3 rows (H1/H2/H3) x 4 cols (FedAvg, FedProx, FedAdam, WD)."""
    settings = ["H1", "H2", "H3"]
    setting_titles = {
        "H1": r"H1: $\alpha$=0.25, $E$=25, IID",
        "H2": r"H2: $\alpha$=0.25, $E$=25, non-IID",
        "H3": r"H3: $\alpha$=0.3, $E$=50, non-IID",
    }

    algo_groups = {
        "FedAvg": [
            ("FedAvg", None, "#757575", "-", 2.0),
        ],
        "FedProx": [
            (r"$\mu$=0.001", "mu0.001", "#64B5F6", "-", 1.5),
            (r"$\mu$=0.01", "mu0.01", "#1E88E5", "-", 1.5),
            (r"$\mu$=0.1", "mu0.1", "#0D47A1", "--", 1.5),
            (r"$\mu$=1.0", "mu1.0", "#0A3069", "--", 1.0),
        ],
        "FedAdam": [
            (r"$\tau$=0.01", "adam_tau0.001_slr0.01", "#A5D6A7", "-", 1.5),
            (r"$\tau$=0.1", "adam_tau0.001_slr0.1", "#2E7D32", "-", 2.0),
            (r"$\tau$=1.0", "adam_tau0.001_slr1.0", "#004D40", "--", 1.5),
        ],
        "FedAvg + Weight Decay": [
            (r"$\lambda$=0.01", "wd0.01", "#EF9A9A", "-", 1.5),
            (r"$\lambda$=0.1", "wd0.1", "#EF5350", "-", 1.5),
            (r"$\lambda$=1.0", "wd1.0", "#B71C1C", "--", 1.5),
        ],
    }
    group_names = ["FedAvg", "FedProx", "FedAdam", "FedAvg + Weight Decay"]

    group_xlim = {
        "FedAvg": 80,
        "FedProx": 80,
        "FedAdam": 80,
        "FedAvg + Weight Decay": 80,
    }

    seeds = [42, 123, 456]
    fig, axes = plt.subplots(len(settings), len(group_names),
                             figsize=(16, 3.5 * len(settings)), sharey=True)

    for row, setting in enumerate(settings):
        raw_dir = os.path.join(RESULTS_DIR, setting)

        for col, gname in enumerate(group_names):
            ax = axes[row][col]
            algos = algo_groups[gname]
            xlim = group_xlim[gname]

            # Always plot FedAvg baseline in gray behind
            if gname != "FedAvg":
                for si, seed in enumerate(seeds):
                    files = _find_histories(raw_dir, None, seed)
                    for f in files[:1]:
                        h = json.load(open(f))
                        steps = np.array(h["total_steps"]) / 1000
                        label = "FedAvg" if si == 0 else None
                        ax.plot(steps, _normalize_acc(h["test_acc"]), color="#BDBDBD",
                                linewidth=1.0, alpha=0.5, label=label)

            for algo_label, frag, color, ls, lw in algos:
                for si, seed in enumerate(seeds):
                    files = _find_histories(raw_dir, frag, seed)
                    for f in files[:1]:
                        h = json.load(open(f))
                        steps = np.array(h["total_steps"]) / 1000
                        label = algo_label if si == 0 else None
                        alpha = 1.0 if si == 0 else 0.3
                        ax.plot(steps, _normalize_acc(h["test_acc"]), color=color,
                                linestyle=ls, linewidth=lw, alpha=alpha, label=label)

            ax.set_xlim(0, xlim)
            ax.set_ylim(-0.05, 1.05)
            ax.axhline(0.95, color="gray", linestyle=":", alpha=0.3, linewidth=0.8)
            if row == 0:
                ax.set_title(gname, fontsize=16, fontweight="bold")
            if col == 0:
                ax.set_ylabel(setting_titles[setting] + "\n\nTest accuracy", fontsize=13)
            if row == len(settings) - 1:
                ax.set_xlabel(r"Gradient steps ($\times$1000)", fontsize=13)
            ax.tick_params(labelsize=11)
            ax.legend(fontsize=11, loc="lower right")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "exp5_training_curves.png"),
                bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/exp5_training_curves.png")

    # Version without FedAvg column
    group_names_no_fa = ["FedProx", "FedAdam", "FedAvg + Weight Decay"]
    fig2, axes2 = plt.subplots(len(settings), len(group_names_no_fa),
                               figsize=(13, 3.5 * len(settings)), sharey=True)

    for row, setting in enumerate(settings):
        raw_dir = os.path.join(RESULTS_DIR, setting)

        for col, gname in enumerate(group_names_no_fa):
            ax = axes2[row][col]
            algos = algo_groups[gname]
            xlim = group_xlim[gname]

            # FedAvg baseline in gray behind
            for si, seed in enumerate(seeds):
                files = _find_histories(raw_dir, None, seed)
                for f in files[:1]:
                    h = json.load(open(f))
                    steps = np.array(h["total_steps"]) / 1000
                    label = "FedAvg" if si == 0 else None
                    ax.plot(steps, _normalize_acc(h["test_acc"]), color="#BDBDBD",
                            linewidth=1.0, alpha=0.5, label=label)

            for algo_label, frag, color, ls, lw in algos:
                for si, seed in enumerate(seeds):
                    files = _find_histories(raw_dir, frag, seed)
                    for f in files[:1]:
                        h = json.load(open(f))
                        steps = np.array(h["total_steps"]) / 1000
                        label = algo_label if si == 0 else None
                        alpha = 1.0 if si == 0 else 0.3
                        ax.plot(steps, _normalize_acc(h["test_acc"]), color=color,
                                linestyle=ls, linewidth=lw, alpha=alpha, label=label)

            ax.set_xlim(0, xlim)
            ax.set_ylim(-0.05, 1.05)
            ax.axhline(0.95, color="gray", linestyle=":", alpha=0.3, linewidth=0.8)
            if row == 0:
                ax.set_title(gname, fontsize=16, fontweight="bold")
            if col == 0:
                ax.set_ylabel(setting_titles[setting] + "\n\nTest accuracy", fontsize=13)
            if row == len(settings) - 1:
                ax.set_xlabel(r"Gradient steps ($\times$1000)", fontsize=13)
            ax.tick_params(labelsize=11)
            ax.legend(fontsize=11, loc="lower right")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "exp5_training_curves_no_fedavg.png"),
                bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/exp5_training_curves_no_fedavg.png")


def _load_seed0_history(raw_dir, frag):
    """Load seed=42 history for a given algorithm fragment."""
    files = _find_histories(raw_dir, frag, 42)
    if files:
        with open(files[0]) as fh:
            return json.load(fh)
    return None


# Key algorithms to compare in mechanistic plots
MECH_ALGOS = [
    ("FedAvg", None, "#757575", "-", 2.0),
    (r"FedProx $\mu$=0.001", "mu0.001", "#1E88E5", "--", 1.5),
    (r"FedAdam $\tau$=0.01", "adam_tau0.001_slr0.01", "#66BB6A", "-", 1.5),
    (r"FedAdam $\tau$=0.1", "adam_tau0.001_slr0.1", "#4CAF50", "-", 2.0),
    (r"WD $\lambda$=0.01", "wd0.01", "#EF5350", "--", 1.5),
]

SETTINGS = ["H1", "H2", "H3"]
SETTING_TITLES = {
    "H1": r"H1: $\alpha$=0.25, E=25, IID",
    "H2": r"H2: $\alpha$=0.25, E=25, non-IID",
    "H3": r"H3: $\alpha$=0.3, E=50, non-IID",
}

# Centralized baselines (matching alpha, seed=42)
CENT_PATHS = {
    "H1": "results/exp1_boundary/history_addition_gd_p97_N256_a0.25_s42.json",
    "H2": "results/exp1_boundary/history_addition_gd_p97_N256_a0.25_s42.json",
    "H3": "results/exp1_boundary/history_addition_gd_p97_N256_a0.3_s42.json",
}


def _load_centralized(setting):
    """Load centralized baseline, normalizing keys to match FL format."""
    path = CENT_PATHS.get(setting)
    if path is None or not os.path.exists(path):
        return None
    with open(path) as f:
        h = json.load(f)
    # Centralized uses 'epoch' for steps, and has no drift
    h["total_steps"] = h["epoch"]
    return h


def plot_mechanistic_grid(results):
    """6-panel grid: 3 settings × (drift, W2/W1 ratio) for key algorithms."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex=True)

    for col, setting in enumerate(SETTINGS):
        raw_dir = os.path.join(RESULTS_DIR, setting)
        ax_drift = axes[0][col]
        ax_ratio = axes[1][col]

        # Centralized baseline (W2/W1 only, no drift)
        ch = _load_centralized(setting)
        if ch is not None:
            cs = np.array(ch["total_steps"]) / 1000
            cw1 = np.array(ch.get("weight_norm_layer1", []))
            cw2 = np.array(ch.get("weight_norm_layer2", []))
            if len(cw1) > 0 and len(cw2) > 0:
                cr = cw2 / np.maximum(cw1, 1e-6)
                ax_ratio.plot(cs[:len(cr)], cr, color="#FF9800",
                              linestyle="-", linewidth=2.0, label="Centralized")

        for algo_label, frag, color, ls, lw in MECH_ALGOS:
            h = _load_seed0_history(raw_dir, frag)
            if h is None:
                continue
            steps = np.array(h["total_steps"]) / 1000
            drift = np.array(h.get("mean_client_drift", []))
            w1 = np.array(h.get("weight_norm_layer1", []))
            w2 = np.array(h.get("weight_norm_layer2", []))

            if len(drift) > 0:
                ax_drift.plot(steps[:len(drift)], drift, color=color,
                              linestyle=ls, linewidth=lw, label=algo_label)
            if len(w1) > 0 and len(w2) > 0:
                ratio = w2 / np.maximum(w1, 1e-6)
                ax_ratio.plot(steps[:len(ratio)], ratio, color=color,
                              linestyle=ls, linewidth=lw, label=algo_label)

        ax_drift.set_title(SETTING_TITLES[setting], fontsize=11)
        ax_drift.set_ylabel("Client drift" if col == 0 else "")
        ax_ratio.set_ylabel(r"$\|W_2\|/\|W_1\|$" if col == 0 else "")
        ax_ratio.set_xlabel(r"Gradient steps ($\times$1000)")
        if col == 2:
            ax_drift.legend(fontsize=7, loc="upper right")
            ax_ratio.legend(fontsize=7, loc="upper left")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "exp5_drift_ratio_grid.png"),
                bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/exp5_drift_ratio_grid.png")


def plot_weight_norms(results):
    """3×2 grid: W1 and W2 evolution per setting for key algorithms."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex=True)

    for col, setting in enumerate(SETTINGS):
        raw_dir = os.path.join(RESULTS_DIR, setting)
        ax_w1 = axes[0][col]
        ax_w2 = axes[1][col]

        # Centralized baseline
        ch = _load_centralized(setting)
        if ch is not None:
            cs = np.array(ch["total_steps"]) / 1000
            cw1 = np.array(ch.get("weight_norm_layer1", []))
            cw2 = np.array(ch.get("weight_norm_layer2", []))
            if len(cw1) > 0:
                ax_w1.plot(cs[:len(cw1)], cw1, color="#FF9800",
                           linestyle="-", linewidth=2.0, label="Centralized")
            if len(cw2) > 0:
                ax_w2.plot(cs[:len(cw2)], cw2, color="#FF9800",
                           linestyle="-", linewidth=2.0, label="Centralized")

        for algo_label, frag, color, ls, lw in MECH_ALGOS:
            h = _load_seed0_history(raw_dir, frag)
            if h is None:
                continue
            steps = np.array(h["total_steps"]) / 1000
            w1 = np.array(h.get("weight_norm_layer1", []))
            w2 = np.array(h.get("weight_norm_layer2", []))

            if len(w1) > 0:
                ax_w1.plot(steps[:len(w1)], w1, color=color,
                           linestyle=ls, linewidth=lw, label=algo_label)
            if len(w2) > 0:
                ax_w2.plot(steps[:len(w2)], w2, color=color,
                           linestyle=ls, linewidth=lw, label=algo_label)

        ax_w1.set_title(SETTING_TITLES[setting], fontsize=11)
        ax_w1.set_ylabel(r"$\|W_1\|_F$" if col == 0 else "")
        ax_w2.set_ylabel(r"$\|W_2\|_F$" if col == 0 else "")
        ax_w2.set_xlabel(r"Gradient steps ($\times$1000)")
        if col == 2:
            ax_w1.legend(fontsize=7, loc="upper left")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "exp5_weight_norms.png"),
                bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/exp5_weight_norms.png")


def plot_ipr_trajectories(results):
    """IPR evolution per setting for key algorithms."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True, sharey=True)

    for col, setting in enumerate(SETTINGS):
        ax = axes[col]
        raw_dir = os.path.join(RESULTS_DIR, setting)

        # Centralized baseline
        ch = _load_centralized(setting)
        if ch is not None:
            cs = np.array(ch["total_steps"]) / 1000
            cipr = np.array(ch.get("ipr", []))
            if len(cipr) > 0:
                ax.plot(cs[:len(cipr)], cipr, color="#FF9800",
                        linestyle="-", linewidth=2.0, label="Centralized")

        for algo_label, frag, color, ls, lw in MECH_ALGOS:
            h = _load_seed0_history(raw_dir, frag)
            if h is None:
                continue
            steps = np.array(h["total_steps"]) / 1000
            ipr = np.array(h.get("ipr", []))
            if len(ipr) > 0:
                ax.plot(steps[:len(ipr)], ipr, color=color,
                        linestyle=ls, linewidth=lw, label=algo_label)

        ax.set_title(SETTING_TITLES[setting], fontsize=11)
        ax.set_ylabel("IPR" if col == 0 else "")
        ax.set_xlabel(r"Gradient steps ($\times$1000)")
        if col == 2:
            ax.legend(fontsize=7, loc="upper left")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "exp5_ipr_trajectories.png"),
                bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/exp5_ipr_trajectories.png")


def plot_mechanistic_summary(results):
    """Single 4×3 figure: test_acc, drift, W2/W1, IPR across settings."""
    metrics = [
        ("test_acc", "Test accuracy", None, (0, 1.05)),
        ("mean_client_drift", "Client drift", None, None),
        ("w2_w1_ratio", r"$\|W_2\|/\|W_1\|$", None, None),
        ("ipr", "IPR", None, (0, 0.6)),
    ]

    fig, axes = plt.subplots(len(metrics), 3, figsize=(15, 12), sharex=True)

    for col, setting in enumerate(SETTINGS):
        raw_dir = os.path.join(RESULTS_DIR, setting)

        # Centralized baseline
        ch = _load_centralized(setting)

        for row, (key, ylabel, _, ylim) in enumerate(metrics):
            ax = axes[row][col]

            # Plot centralized first (behind FL curves)
            if ch is not None and key != "mean_client_drift":
                cs = np.array(ch["total_steps"]) / 1000
                if key == "w2_w1_ratio":
                    cw1 = np.array(ch.get("weight_norm_layer1", []))
                    cw2 = np.array(ch.get("weight_norm_layer2", []))
                    if len(cw1) > 0:
                        cy = cw2 / np.maximum(cw1, 1e-6)
                        ax.plot(cs[:len(cy)], cy, color="#FF9800",
                                linestyle="-", linewidth=2.0, label="Centralized")
                else:
                    cy = np.array(ch.get(key, []))
                    if key == "test_acc":
                        cy = _normalize_acc(cy)
                    if len(cy) > 0:
                        ax.plot(cs[:len(cy)], cy, color="#FF9800",
                                linestyle="-", linewidth=2.0, label="Centralized")

            for algo_label, frag, color, ls, lw in MECH_ALGOS:
                h = _load_seed0_history(raw_dir, frag)
                if h is None:
                    continue
                steps = np.array(h["total_steps"]) / 1000

                if key == "w2_w1_ratio":
                    w1 = np.array(h.get("weight_norm_layer1", []))
                    w2 = np.array(h.get("weight_norm_layer2", []))
                    if len(w1) == 0:
                        continue
                    y = w2 / np.maximum(w1, 1e-6)
                else:
                    y = np.array(h.get(key, []))
                    if key == "test_acc":
                        y = _normalize_acc(y)
                    if len(y) == 0:
                        continue

                ax.plot(steps[:len(y)], y, color=color,
                        linestyle=ls, linewidth=lw, label=algo_label)

            if row == 0:
                ax.set_title(SETTING_TITLES[setting], fontsize=11)
            if col == 0:
                ax.set_ylabel(ylabel, fontsize=10)
            if ylim:
                ax.set_ylim(ylim)
            if row == len(metrics) - 1:
                ax.set_xlabel(r"Gradient steps ($\times$1000)")
            if col == 2 and row == 0:
                ax.legend(fontsize=11, loc="lower right")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "exp5_mechanistic_summary.png"),
                bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/exp5_mechanistic_summary.png")


def plot_paper_mechanistic_fingerprint(results):
    """Paper figure: 4-row × 1-col temporal evolution for H2 setting.

    Shows how FedAdam finds a qualitatively different solution path.
    Rows: test accuracy, client drift, W2/W1 ratio, IPR.
    """
    import re

    setting = "H2"
    raw_dir = os.path.join(RESULTS_DIR, setting)

    paper_algos = [
        ("Centralized", None, "#FF9800", "-", 2.5),
        ("FedAvg", None, "#757575", "-", 2.0),
        (r"FedProx ($\mu$=0.001)", "mu0.001", "#1E88E5", "--", 1.8),
        (r"FedAdam ($\tau$=0.1)", "adam_tau0.001_slr0.1", "#2E7D32", "-", 2.5),
        (r"Weight Decay ($\lambda$=0.01)", "wd0.01", "#EF5350", "--", 1.8),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 6))
    axes = axes.flatten()
    panel_config = [
        ("test_acc", "Test accuracy", (0, 1.05)),
        ("mean_client_drift", "Client drift", (0, 5)),
        ("w2_w1_ratio", r"$\|W_2\| / \|W_1\|$", None),
        ("ipr", "IPR (Fourier concentration)", (0, 0.55)),
    ]

    for algo_label, frag, color, ls, lw in paper_algos:
        if algo_label == "Centralized":
            ch = _load_centralized(setting)
            if ch is None:
                continue
            steps = np.array(ch["total_steps"]) / 1000
            data = {
                "test_acc": _normalize_acc(ch.get("test_acc", [])),
                "mean_client_drift": np.array([]),
                "weight_norm_layer1": np.array(ch.get("weight_norm_layer1", [])),
                "weight_norm_layer2": np.array(ch.get("weight_norm_layer2", [])),
                "ipr": np.array(ch.get("ipr", [])),
            }
        else:
            h = _load_seed0_history(raw_dir, frag)
            if h is None:
                continue
            steps = np.array(h["total_steps"]) / 1000
            data = {
                "test_acc": _normalize_acc(h.get("test_acc", [])),
                "mean_client_drift": np.array(h.get("mean_client_drift", [])),
                "weight_norm_layer1": np.array(h.get("weight_norm_layer1", [])),
                "weight_norm_layer2": np.array(h.get("weight_norm_layer2", [])),
                "ipr": np.array(h.get("ipr", [])),
            }

        for idx, (key, ylabel, ylim) in enumerate(panel_config):
            ax = axes[idx]
            if key == "w2_w1_ratio":
                w1 = data["weight_norm_layer1"]
                w2 = data["weight_norm_layer2"]
                if len(w1) == 0:
                    continue
                y = w2 / np.maximum(w1, 1e-6)
            elif key == "mean_client_drift":
                y = data[key]
                if len(y) == 0:
                    continue
                if algo_label == "Centralized":
                    continue  # no drift for centralized
            else:
                y = data.get(key, np.array([]))
                if len(y) == 0:
                    continue

            ax.plot(steps[:len(y)], y, color=color, linestyle=ls,
                    linewidth=lw, label=algo_label)

    for idx, (key, ylabel, ylim) in enumerate(panel_config):
        ax = axes[idx]
        ax.set_ylabel(ylabel, fontsize=13)
        ax.set_xlim(0, 80)
        if ylim:
            ax.set_ylim(ylim)
        ax.tick_params(labelsize=11)
        ax.set_xlabel(r"Gradient steps ($\times$1000)", fontsize=12)
        ax.text(-0.08, 1.05, chr(ord('a') + idx) + ")", transform=ax.transAxes,
                fontsize=14, fontweight="bold", va="top")

    axes[1].legend(fontsize=9, loc="upper right")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "paper_mechanistic_fingerprint.png"),
                bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/paper_mechanistic_fingerprint.png")


def plot_paper_mechanistic_scatter(results):
    """Paper figure: cross-run scatter plots.

    2 panels: (a) mean drift vs T_grok, (b) W2/W1 ratio vs T_grok.
    Points from ALL exp5 runs, colored by algorithm family.
    Failed runs shown as × at T_grok = budget.
    """
    import re

    budget = 80000
    algo_families = {
        "FedAvg": ("#757575", "o"),
        "FedProx": ("#1E88E5", "s"),
        "FedAdam": ("#2E7D32", "D"),
        "WD": ("#EF5350", "X"),
    }

    def classify_family(fname):
        if "adam_tau" in fname:
            return "FedAdam"
        elif "_wd" in fname:
            return "WD"
        elif "_mu" in fname:
            return "FedProx"
        else:
            return "FedAvg"

    # Collect all data points
    points = []
    for setting in ["H1", "H2", "H3"]:
        raw_dir = os.path.join(RESULTS_DIR, setting)
        for f in sorted(glob.glob(os.path.join(raw_dir, "history_*.json"))):
            with open(f) as fh:
                h = json.load(fh)
            fname = os.path.basename(f)
            family = classify_family(fname)

            drift = np.array(h.get("mean_client_drift", []))
            w1 = np.array(h.get("weight_norm_layer1", []))
            w2 = np.array(h.get("weight_norm_layer2", []))
            ipr = np.array(h.get("ipr", []))
            test_acc = _normalize_acc(h.get("test_acc", []))

            if len(drift) == 0 or len(w1) == 0:
                continue

            mean_drift = float(np.mean(drift))
            final_w1 = float(w1[-1])
            final_w2 = float(w2[-1])
            ratio = final_w2 / max(final_w1, 1e-6)
            final_ipr = float(ipr[-1]) if len(ipr) > 0 else 0.0

            # Compute T_grok
            steps = h["total_steps"]
            t_grok = None
            for i, acc in enumerate(test_acc):
                if acc >= 0.95 and t_grok is None:
                    # Check it stays above
                    if all(a >= 0.95 for a in test_acc[i:]):
                        t_grok = steps[i]
                        break
            grokked = t_grok is not None

            points.append({
                "family": family, "setting": setting,
                "mean_drift": mean_drift, "w2_w1": ratio,
                "ipr": final_ipr, "t_grok": t_grok if grokked else budget,
                "grokked": grokked,
            })

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    scatter_configs = [
        ("mean_drift", r"Mean client drift", "a)"),
        ("w2_w1", r"Final $\|W_2\| / \|W_1\|$", "b)"),
        ("ipr", "Final IPR", "c)"),
    ]

    for ax, (xkey, xlabel, panel_label) in zip(axes, scatter_configs):
        for family, (color, marker) in algo_families.items():
            grokked_pts = [p for p in points if p["family"] == family and p["grokked"]]
            failed_pts = [p for p in points if p["family"] == family and not p["grokked"]]

            if grokked_pts:
                xs = [p[xkey] for p in grokked_pts]
                ys = [p["t_grok"] / 1000 for p in grokked_pts]
                ax.scatter(xs, ys, c=color, marker=marker, s=50, alpha=0.8,
                           edgecolors="white", linewidth=0.5, label=family, zorder=3)

            if failed_pts:
                xs = [p[xkey] for p in failed_pts]
                ys = [budget / 1000] * len(xs)
                ax.scatter(xs, ys, c=color, marker=marker, s=50, alpha=0.3,
                           edgecolors="white", linewidth=0.5, zorder=2)

        # Correlation for grokked points
        grokked_all = [p for p in points if p["grokked"]]
        if len(grokked_all) > 2:
            xs = np.array([p[xkey] for p in grokked_all])
            ys = np.array([p["t_grok"] for p in grokked_all])
            valid = np.isfinite(xs) & np.isfinite(ys)
            if valid.sum() > 2:
                r = np.corrcoef(xs[valid], ys[valid])[0, 1]
                ax.text(0.95, 0.05, f"r = {r:.2f}",
                        transform=ax.transAxes, fontsize=12,
                        ha="right", va="bottom",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                                  edgecolor="gray", alpha=0.8))

        ax.axhline(budget / 1000, color="gray", linestyle="--", alpha=0.3)
        ax.set_xlabel(xlabel, fontsize=13)
        ax.set_ylabel(r"$T_{\mathrm{grok}}$ ($\times$1000)" if ax == axes[0] else "",
                       fontsize=13)
        ax.tick_params(labelsize=11)
        ax.text(-0.05, 1.05, panel_label, transform=ax.transAxes,
                fontsize=14, fontweight="bold", va="top")

    axes[0].legend(fontsize=11, loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "paper_mechanistic_scatter.png"),
                bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/paper_mechanistic_scatter.png")


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results = load_exp5_results()

    n_total = sum(len(v) for v in results.values())
    for s in ["H1", "H2", "H3"]:
        print(f"  {s}: {len(results[s])} algorithms loaded")

    plot_algorithm_rescue(results)
    plot_algorithm_rescue_vertical(results)
    plot_algorithm_speedup(results)
    plot_training_curves(results)
    plot_mechanistic_grid(results)
    plot_weight_norms(results)
    plot_ipr_trajectories(results)
    plot_mechanistic_summary(results)
    plot_paper_mechanistic_fingerprint(results)
    plot_paper_mechanistic_scatter(results)

    print(f"\nDone! Figures saved to {OUTPUT_DIR}")
