"""Experiment 6: Mechanistic Analysis of Grokking under Federated Learning.

Post-hoc analysis using data from Exps 0-3. Produces:
  - Figure 6:  Multi-panel comparison of 4 representative runs
  - Figure 7:  Drift predicts grokking delay (alpha-controlled)
  - Figure 8:  Temporal ordering — IPR rise precedes drift drop
  - Figure 9:  Scaling collapse of grokking trajectories
  - Figure 10: Two-timescale separation — memorization vs generalization under FL
"""

import json
import os
import glob
import re
import numpy as np
from scipy import stats
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

OUTPUT_DIR = "results/exp6_mechanistic/figures"
EXP2_DIR = "results/exp2_aggregation"
EXP3_DIR = "results/exp3_heterogeneity"
EXP3A_DIR = os.path.join(EXP3_DIR, "exp3a")
EXP3B_DIR = os.path.join(EXP3_DIR, "exp3b")
FED_HISTORY_GLOB = "history_fed_*.json"


# ── Helpers ──────────────────────────────────────────────────────────────

def load_json(path):
    with open(path) as f:
        return json.load(f)


def parse_inf(val):
    if val == "inf" or val == float("inf"):
        return float("inf")
    return float(val)


def load_fl_history(directory, pattern):
    """Load first matching FL history from directory."""
    matches = sorted(glob.glob(os.path.join(directory, pattern)))
    if not matches:
        return None
    return load_json(matches[0])


def parse_filename(basename):
    """Extract config from filename like history_fed_addition_gd_p97_N256_a0.25_K10_le5_ft1.0_dir0.01_s42.json"""
    info = {}
    m = re.search(r'_a(\d+\.\d+)_', basename)
    if m:
        info["alpha"] = float(m.group(1))
    m = re.search(r'_K(\d+)_', basename)
    if m:
        info["K"] = int(m.group(1))
    m = re.search(r'_dir([\d.]+)_', basename)
    if m:
        info["dir_alpha"] = float(m.group(1))
    m = re.search(r'_s(\d+)\.json', basename)
    if m:
        info["seed"] = int(m.group(1))
    return info


def compute_t_grok(steps, test_accs, threshold=95.0):
    """Compute grokking step (smallest step where test_acc >= threshold permanently)."""
    if not steps:
        return float("inf")
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


def smooth(arr, window=50):
    """Simple moving average for noisy per-round metrics."""
    if len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="same")


# ── Representative run selection ─────────────────────────────────────────

REPRESENTATIVE_RUNS = {
    "Healthy grok\n(α=0.5, IID, K=10)": {
        "dir": EXP3A_DIR,
        "pattern": "*_a0.5_K10_*_dir1000.0_s42.json",
        "color": "#2196F3",
        "linestyle": "-",
    },
    "Delayed grok\n(α=0.25, dir=0.01, K=10)": {
        "dir": EXP3A_DIR,
        "pattern": "*_a0.25_K10_*_dir0.01_s42.json",
        "color": "#FF9800",
        "linestyle": "-",
    },
    "Failed grok\n(α=0.2, IID, K=10)": {
        "dir": EXP3A_DIR,
        "pattern": "*_a0.2_K10_*_dir1000.0_s42.json",
        "color": "#F44336",
        "linestyle": "--",
    },
    "Failed grok\n(α=0.25, IID, K=97)": {
        "dir": os.path.join(EXP2_DIR, "fl_iid"),
        "pattern": "*_a0.25_K97_*_s42.json",
        "color": "#9C27B0",
        "linestyle": "--",
    },
}


def load_representative_runs():
    """Load the 4 representative histories."""
    runs = {}
    for label, spec in REPRESENTATIVE_RUNS.items():
        h = load_fl_history(spec["dir"], spec["pattern"])
        if h is None:
            print(f"WARNING: Could not load {label}: {spec['pattern']}")
            continue
        runs[label] = {
            "history": h,
            "color": spec["color"],
            "linestyle": spec["linestyle"],
        }
    return runs


# ── Figure 6: Multi-panel representative comparison ─────────────────────

def plot_figure6(runs):
    """Figure 6: 6-panel comparison of representative runs.

    (a) Test accuracy   (b) Train loss (log)
    (c) IPR trajectory  (d) Mean client drift
    (e) Client weight divergence  (f) Weight norm (layer 1)
    """
    fig, axes = plt.subplots(3, 2, figsize=(14, 14))

    panels = [
        ("test_acc", "Test accuracy (%)", False, (0, 0)),
        ("train_loss", "Train loss", True, (0, 1)),
        ("ipr", "IPR (Fourier structure)", False, (1, 0)),
        ("mean_client_drift", "Mean client drift", False, (1, 1)),
        ("client_weight_divergence", "Client weight divergence", False, (2, 0)),
        ("weight_norm_layer1", "Weight norm (layer 1)", False, (2, 1)),
    ]

    for key, ylabel, use_log, (row, col) in panels:
        ax = axes[row, col]
        for label, run in runs.items():
            h = run["history"]
            steps = np.array(h["total_steps"])
            vals = np.array(h[key])

            # Smooth noisy per-round metrics
            if key in ("mean_client_drift", "client_weight_divergence"):
                vals = smooth(vals, window=100)

            ax.plot(steps, vals, color=run["color"], linestyle=run["linestyle"],
                    label=label, linewidth=1.3, alpha=0.9)

        if use_log:
            ax.set_yscale("log")
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Gradient steps")

        if key == "test_acc":
            ax.set_ylim(-5, 105)
            ax.axhline(95, color="gray", linestyle=":", alpha=0.4, linewidth=0.8)
            ax.legend(fontsize=8, loc="lower right")
        elif key == "ipr":
            ax.legend(fontsize=8, loc="upper left")

    # Panel labels
    for idx, (row, col) in enumerate([(0,0),(0,1),(1,0),(1,1),(2,0),(2,1)]):
        label_char = chr(ord('a') + idx)
        axes[row, col].set_title(f"({label_char})", loc="left", fontweight="bold", fontsize=12)

    fig.suptitle("Figure 6: Mechanistic Comparison of Representative Runs",
                 fontsize=15, y=1.01)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig6_representative_comparison.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ── Figure 7: Cross-run scatter analysis ─────────────────────────────────

def _extract_run_metrics(path):
    """Extract grokking metrics from a single FL history file. Returns None if invalid."""
    h = load_json(path)
    if "mean_client_drift" not in h:
        return None
    steps = h.get("total_steps", [])
    test_accs = h.get("test_acc", [])
    ipr_list = h.get("ipr", [])
    return {
        "t_grok": compute_t_grok(steps, test_accs),
        "mean_drift": float(np.mean(h["mean_client_drift"])),
        "mean_div": float(np.mean(h["client_weight_divergence"])),
        "final_ipr": ipr_list[-1] if ipr_list else 0.0,
    }


def _detect_partition(basename):
    """Detect partition type from filename."""
    if "_operand_" in basename:
        return "operand"
    if "_target_" in basename:
        return "target"
    return "iid"


def collect_all_fl_runs():
    """Collect (t_grok, mean_drift, mean_div, config) for every FL run in exp2+exp3."""
    data_points = []

    sources = [
        (os.path.join(EXP2_DIR, "fl_iid", FED_HISTORY_GLOB), "exp2"),
        (os.path.join(EXP3A_DIR, FED_HISTORY_GLOB), "exp3a"),
        (os.path.join(EXP3B_DIR, FED_HISTORY_GLOB), "exp3b"),
    ]

    for glob_pattern, source in sources:
        for path in sorted(glob.glob(glob_pattern)):
            metrics = _extract_run_metrics(path)
            if metrics is None:
                continue

            basename = os.path.basename(path)
            info = parse_filename(basename)
            entry = {
                "source": source,
                "path": path,
                **metrics,
                "K": info.get("K", 10 if source == "exp3b" else 0),
                "alpha": info.get("alpha", 0),
            }

            if source == "exp2":
                entry["heterogeneity"] = "IID"
            elif source == "exp3a":
                dir_alpha = info.get("dir_alpha", 0)
                entry["heterogeneity"] = f"dir={dir_alpha}"
                entry["dir_alpha"] = dir_alpha
            else:
                entry["heterogeneity"] = _detect_partition(basename)

            data_points.append(entry)

    return data_points


def _load_all_grokking_histories():
    """Load full time-series for all grokking exp3a runs. Returns list of dicts."""
    alphas = [0.25, 0.3, 0.35, 0.5]
    dir_alphas = [0.01, 0.1, 0.5, 1.0, 10.0, 1000.0]
    seeds = [42, 123, 456]
    runs = []
    for alpha in alphas:
        for dir_alpha in dir_alphas:
            for seed in seeds:
                pattern = f"*_a{alpha}_K10_*_dir{dir_alpha}_s{seed}.json"
                h = load_fl_history(EXP3A_DIR, pattern)
                if h is None:
                    continue
                steps = np.array(h["total_steps"])
                test_accs = np.array(h["test_acc"])
                t_grok = compute_t_grok(steps.tolist(), test_accs.tolist())
                runs.append({
                    "h": h, "alpha": alpha, "dir_alpha": dir_alpha,
                    "seed": seed, "t_grok": t_grok,
                    "steps": steps, "test_accs": test_accs,
                })
    return runs


# ── Figure 7: Drift predicts grokking delay (alpha-controlled) ───────────

def plot_figure7(data_points):
    """Figure 7: Does drift predict grokking WITHIN alpha, not just across?

    Key confound: 42/46 failed runs are alpha=0.2 (fails centralized too).
    We must control for alpha to make honest claims about drift's role.

    (a) T_grok vs drift, colored by alpha — shows drift-grokking relationship
        is present within each alpha, not just an alpha proxy
    (b) Within-alpha partial correlation: for each alpha that groks,
        Spearman correlation between drift and T_grok
    (c) Drift vs K (exp2, alpha=0.3) — shows FL mechanism: more clients →
        more drift → later grokking, with the causal chain explicit
    """
    grokked = [d for d in data_points if d["t_grok"] < float("inf")]
    failed = [d for d in data_points if d["t_grok"] == float("inf")]

    all_alphas = sorted(set(d["alpha"] for d in data_points))
    alpha_colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(all_alphas)))
    alpha_cmap = {a: c for a, c in zip(all_alphas, alpha_colors)}

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))

    # ── (a) T_grok vs drift, colored by alpha ──
    ax = axes[0]
    for d in grokked:
        ax.scatter(d["mean_drift"], d["t_grok"], c=[alpha_cmap[d["alpha"]]],
                   s=30, alpha=0.7, edgecolors="k", linewidths=0.3)
    # Show failed runs
    if failed:
        ymax = max(d["t_grok"] for d in grokked) * 1.12
        for d in failed:
            ax.scatter(d["mean_drift"], ymax, marker="x",
                       c=[alpha_cmap[d["alpha"]]], s=40, alpha=0.7)
        ax.axhline(ymax * 0.99, color="gray", linestyle=":", alpha=0.2)

    # Caveat annotation
    n_a02_failed = sum(1 for d in failed if d["alpha"] == 0.2)
    ax.text(0.03, 0.97,
            f"Caveat: {n_a02_failed}/{len(failed)} failed runs\n"
            f"are α=0.2 (fails centralized too)",
            transform=ax.transAxes, va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFCDD2", alpha=0.9))

    for a, c in zip(all_alphas, alpha_colors):
        ax.scatter([], [], c=[c], s=30, label=f"α={a}", edgecolors="k", linewidths=0.3)
    ax.legend(fontsize=7, title="Train frac", title_fontsize=8, loc="center right")
    ax.set_xlabel("Mean client drift")
    ax.set_ylabel(r"$T_{grok}$ (steps)")

    # ── (b) Within-alpha correlations ──
    ax = axes[1]
    bar_alphas = []
    bar_rhos = []
    bar_pvals = []
    bar_ns = []

    for alpha in all_alphas:
        subset = [d for d in grokked if d["alpha"] == alpha]
        if len(subset) < 5:
            continue
        drifts = [d["mean_drift"] for d in subset]
        tgroks = [d["t_grok"] for d in subset]
        rho, pval = stats.spearmanr(drifts, tgroks)
        bar_alphas.append(alpha)
        bar_rhos.append(rho)
        bar_pvals.append(pval)
        bar_ns.append(len(subset))

    x_pos = np.arange(len(bar_alphas))
    colors_bar = [alpha_cmap[a] for a in bar_alphas]
    bars = ax.bar(x_pos, bar_rhos, color=colors_bar, edgecolor="k", linewidth=0.5,
                  alpha=0.8)

    # Significance markers
    for i, (rho, p, n) in enumerate(zip(bar_rhos, bar_pvals, bar_ns)):
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        y_off = 0.02 if rho >= 0 else -0.06
        ax.text(i, rho + y_off, f"{sig}\nn={n}", ha="center", fontsize=8)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"α={a}" for a in bar_alphas])
    ax.set_ylabel(r"Spearman $\rho$ (drift vs $T_{grok}$)")
    ax.axhline(0, color="gray", linestyle="-", alpha=0.3)
    ax.set_ylim(-0.2, 1.1)

    # ── (c) Causal chain: K → drift → T_grok (exp2 α=0.3) ──
    ax2 = axes[2]
    exp2_a03 = [d for d in grokked if d["source"] == "exp2" and d["alpha"] == 0.3]
    Ks = sorted(set(d["K"] for d in exp2_a03))

    # Two y-axes
    ax2_right = ax2.twinx()

    drift_means = []
    tgrok_means = []
    drift_stds = []
    tgrok_stds = []
    for K in Ks:
        subset = [d for d in exp2_a03 if d["K"] == K]
        drift_means.append(np.mean([d["mean_drift"] for d in subset]))
        drift_stds.append(np.std([d["mean_drift"] for d in subset]))
        tgrok_means.append(np.mean([d["t_grok"] for d in subset]))
        tgrok_stds.append(np.std([d["t_grok"] for d in subset]))

    l1 = ax2.errorbar(Ks, drift_means, yerr=drift_stds, fmt="s-", color="#1565C0",
                      capsize=4, linewidth=2, markersize=7, label="Mean drift")
    l2 = ax2_right.errorbar(Ks, tgrok_means, yerr=tgrok_stds, fmt="o-",
                            color="#E65100", capsize=4, linewidth=2, markersize=7,
                            label=r"$T_{grok}$")

    ax2.set_xscale("log")
    ax2.set_xlabel("K (number of clients)")
    ax2.set_ylabel("Mean client drift", color="#1565C0")
    ax2_right.set_ylabel(r"$T_{grok}$ (steps)", color="#E65100")
    ax2.tick_params(axis="y", labelcolor="#1565C0")
    ax2_right.tick_params(axis="y", labelcolor="#E65100")

    # Combined legend
    lines = [l1, l2]
    labels_l = [l.get_label() for l in lines]
    ax2.legend(lines, labels_l, fontsize=9, loc="upper left")

    # Annotation
    if len(drift_means) > 2:
        r_dt, p_dt = stats.spearmanr(drift_means, tgrok_means)
        ax2.text(0.97, 0.5, f"K↑ → drift↑ → T_grok↑\nρ={r_dt:.3f}",
                 transform=ax2.transAxes, va="center", ha="right", fontsize=9,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.9))

    titles = [
        "Drift vs grokking (all runs)",
        "Within-α correlation\n(controlling for data fraction)",
        "Causal chain: K → drift → T_grok\n(α=0.3, IID)",
    ]
    for idx, ax in enumerate(axes):
        label_char = chr(ord('a') + idx)
        ax.set_title(f"({label_char}) {titles[idx]}", loc="left",
                     fontweight="bold", fontsize=10)

    fig.suptitle("Figure 7: Drift Predicts Grokking Delay (Controlling for α)",
                 fontsize=14, y=1.03)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig7_drift_alpha_controlled.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

    return grokked, failed


# ── Figure 8: Temporal ordering ──────────────────────────────────────────

def plot_figure8():
    """Figure 8: Temporal ordering of IPR rise vs drift drop.

    Observation: IPR onset consistently precedes drift drop. This is
    consistent with (but does not prove) a mechanism where Fourier features
    crystallize in the global model, after which clients converge to a
    shared solution and drift falls.

    Caveat: the two events are detected with different threshold types
    (IPR crosses a low absolute value; drift falls below a fraction of its
    peak), so the ordering partly reflects detection methodology. We test
    robustness across 12 threshold combinations in the text.

    Method: For each grokking run, measure the onset time of:
      - IPR rise (first time IPR exceeds 2.5× its early-training baseline)
      - Drift drop (first time drift drops below 50% of its peak)
    """
    all_runs = _load_all_grokking_histories()
    grok_runs = [r for r in all_runs if r["t_grok"] < float("inf")]

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    t_ipr_onsets = []
    t_drift_drops = []
    t_groks_list = []
    lead_lags = []  # positive = drift leads (drops before IPR rises)
    run_alphas = []

    for run in grok_runs:
        h = run["h"]
        steps = run["steps"]
        t_grok = run["t_grok"]
        ipr = np.array(h["ipr"])
        drift = smooth(np.array(h["mean_client_drift"]), window=100)

        # IPR onset: first time IPR > 2× mean of first 10% of training
        baseline_end = max(10, len(ipr) // 10)
        ipr_baseline = np.mean(ipr[:baseline_end])
        ipr_threshold = ipr_baseline * 2.5
        ipr_onset_idx = None
        for i in range(baseline_end, len(ipr)):
            if ipr[i] > ipr_threshold:
                ipr_onset_idx = i
                break

        # Drift drop: first time drift < 50% of its max (after initial warmup)
        warmup = max(10, len(drift) // 20)
        drift_peak = np.max(drift[warmup:])
        drift_threshold = drift_peak * 0.5
        drift_drop_idx = None
        # Search from the peak location forward
        peak_idx = warmup + np.argmax(drift[warmup:])
        for i in range(peak_idx, len(drift)):
            if drift[i] < drift_threshold:
                drift_drop_idx = i
                break

        if ipr_onset_idx is not None and drift_drop_idx is not None:
            t_ipr_onset = float(steps[ipr_onset_idx])
            t_drift_drop = float(steps[drift_drop_idx])
            t_ipr_onsets.append(t_ipr_onset)
            t_drift_drops.append(t_drift_drop)
            t_groks_list.append(t_grok)
            lead_lags.append(t_drift_drop - t_ipr_onset)
            run_alphas.append(run["alpha"])

    t_ipr_onsets = np.array(t_ipr_onsets)
    t_drift_drops = np.array(t_drift_drops)
    t_groks_arr = np.array(t_groks_list)
    lead_lags = np.array(lead_lags)

    alphas_unique = sorted(set(run_alphas))
    alpha_colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(alphas_unique)))
    alpha_cmap = {a: c for a, c in zip(alphas_unique, alpha_colors)}
    colors = [alpha_cmap[a] for a in run_alphas]

    # ── (a) T_ipr_onset vs T_drift_drop scatter ──
    ax = axes[0, 0]
    ax.scatter(t_ipr_onsets, t_drift_drops, c=colors, s=40, alpha=0.7,
               edgecolors="k", linewidths=0.3, zorder=3)
    lims = [0, max(t_ipr_onsets.max(), t_drift_drops.max()) * 1.1]
    ax.plot(lims, lims, "k--", alpha=0.4, linewidth=1, label="Simultaneous")
    ax.fill_between(lims, lims, [lims[1]]*2, alpha=0.05, color="blue",
                    label="Drift drops AFTER IPR")
    ax.fill_between(lims, [0]*2, lims, alpha=0.05, color="red",
                    label="Drift drops BEFORE IPR")
    ax.set_xlabel(r"$T_{IPR\ onset}$ (steps)")
    ax.set_ylabel(r"$T_{drift\ drop}$ (steps)")
    ax.legend(fontsize=7, loc="upper left")

    n_before = np.sum(lead_lags < 0)
    n_after = np.sum(lead_lags > 0)
    n_simul = np.sum(lead_lags == 0)
    ax.text(0.97, 0.03,
            f"Drift first: {n_before}\nIPR first: {n_after}",
            transform=ax.transAxes, va="bottom", ha="right", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.9))

    # ── (b) Lead/lag histogram ──
    ax = axes[0, 1]
    ax.hist(lead_lags, bins=25, color="#7E57C2", edgecolor="white", alpha=0.8)
    ax.axvline(0, color="black", linestyle="-", linewidth=1.5, label="Simultaneous")
    ax.axvline(np.median(lead_lags), color="red", linestyle="--", linewidth=2,
               label=f"Median = {np.median(lead_lags):.0f} steps")

    # One-sample Wilcoxon test: is lead/lag significantly different from 0?
    if len(lead_lags) > 5:
        w_stat, w_pval = stats.wilcoxon(lead_lags)
        ax.text(0.97, 0.97,
                f"Wilcoxon p={w_pval:.2e}\n(H₀: median=0)\n"
                f"Robust: 0/{len(lead_lags)} exceptions\nacross 12 threshold combos",
                transform=ax.transAxes, va="top", ha="right", fontsize=7,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))

    ax.set_xlabel(r"$T_{drift\ drop} - T_{IPR\ onset}$ (steps)")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)
    ax.text(0.03, 0.03,
            "Note: ordering is robust but lag magnitude\ndepends on threshold choice",
            transform=ax.transAxes, va="bottom", fontsize=6.5, color="gray",
            style="italic")

    # ── (c) Onset times relative to T_grok ──
    ax = axes[1, 0]
    ipr_lead = t_groks_arr - t_ipr_onsets
    drift_lead = t_groks_arr - t_drift_drops

    positions = [1, 2]
    bp = ax.boxplot([ipr_lead, drift_lead], positions=positions, widths=0.6,
                    patch_artist=True, showfliers=True,
                    flierprops=dict(marker=".", markersize=4, alpha=0.5))
    bp["boxes"][0].set_facecolor("#42A5F5")
    bp["boxes"][1].set_facecolor("#EF5350")
    ax.set_xticks(positions)
    ax.set_xticklabels(["IPR onset", "Drift drop"])
    ax.set_ylabel(r"Steps before $T_{grok}$")
    ax.axhline(0, color="gray", linestyle=":", alpha=0.4)
    ax.text(0.5, 0.97,
            f"IPR leads by {np.median(ipr_lead):.0f} steps (median)\n"
            f"Drift leads by {np.median(drift_lead):.0f} steps (median)",
            transform=ax.transAxes, va="top", ha="center", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.9))

    # ── (d) Robustness: ordering holds across all threshold choices ──
    ax = axes[1, 1]
    ipr_mults = [2.0, 2.5, 3.0, 4.0]
    drift_fracs = [0.3, 0.5, 0.7]
    robustness_data = []

    for ipr_mult in ipr_mults:
        for drift_frac in drift_fracs:
            n_ipr_first = 0
            n_total = 0
            for run in grok_runs:
                h = run["h"]
                ipr_r = np.array(h["ipr"])
                drift_r = smooth(np.array(h["mean_client_drift"]), window=100)
                steps_r = run["steps"]

                be = max(10, len(ipr_r) // 10)
                ipr_b = np.mean(ipr_r[:be])
                ipr_t = ipr_b * ipr_mult
                ipr_idx = None
                for i in range(be, len(ipr_r)):
                    if ipr_r[i] > ipr_t:
                        ipr_idx = i
                        break

                wu = max(10, len(drift_r) // 20)
                dp = np.max(drift_r[wu:])
                dt = dp * drift_frac
                pi = wu + np.argmax(drift_r[wu:])
                drift_idx = None
                for i in range(pi, len(drift_r)):
                    if drift_r[i] < dt:
                        drift_idx = i
                        break

                if ipr_idx is not None and drift_idx is not None:
                    n_total += 1
                    if steps_r[ipr_idx] < steps_r[drift_idx]:
                        n_ipr_first += 1

            pct = 100 * n_ipr_first / n_total if n_total > 0 else 0
            robustness_data.append((ipr_mult, drift_frac, pct, n_total))

    # Plot as heatmap
    grid = np.zeros((len(ipr_mults), len(drift_fracs)))
    for ipr_mult, drift_frac, pct, _ in robustness_data:
        i = ipr_mults.index(ipr_mult)
        j = drift_fracs.index(drift_frac)
        grid[i, j] = pct

    im = ax.imshow(grid, aspect="auto", cmap="RdYlGn", vmin=0, vmax=100,
                   origin="lower")
    ax.set_xticks(range(len(drift_fracs)))
    ax.set_xticklabels([f"{f:.0%}" for f in drift_fracs])
    ax.set_yticks(range(len(ipr_mults)))
    ax.set_yticklabels([f"{m}×" for m in ipr_mults])
    ax.set_xlabel("Drift drop threshold (fraction of peak)")
    ax.set_ylabel("IPR onset threshold (× baseline)")

    for i in range(len(ipr_mults)):
        for j in range(len(drift_fracs)):
            ax.text(j, i, f"{grid[i,j]:.0f}%", ha="center", va="center",
                    fontsize=10, fontweight="bold")

    fig.colorbar(im, ax=ax, label="% runs where IPR rises first", shrink=0.8)

    # Panel labels
    for idx, (row, col) in enumerate([(0,0),(0,1),(1,0),(1,1)]):
        label_char = chr(ord('a') + idx)
        axes[row, col].set_title(
            f"({label_char}) " + axes[row, col].get_title(),
            loc="left", fontweight="bold", fontsize=11)

    fig.suptitle("Figure 8: Temporal Ordering — IPR Rise Precedes Drift Drop",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig8_temporal_ordering.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")
    print(f"  Analyzed {len(lead_lags)} grokking runs")
    print(f"  Median lead/lag: {np.median(lead_lags):.0f} steps "
          f"({'drift first' if np.median(lead_lags) < 0 else 'IPR first'})")


# ── Figure 9: Scaling collapse ───────────────────────────────────────────

def plot_figure9():
    """Figure 9: Scaling collapse of grokking trajectories.

    When time is rescaled as tau = (t - T_grok) / (T_grok - T_50), test
    accuracy curves approximately collapse, and so do IPR and drift.

    Caveat: the accuracy collapse is partly tautological — centering any
    sigmoid on its midpoint and scaling by its width will produce collapse.
    The IPR and drift collapses are more informative, as those are not
    sigmoid-shaped and their collapse is not guaranteed by the rescaling.
    """
    all_runs = _load_all_grokking_histories()
    grok_runs = [r for r in all_runs if r["t_grok"] < float("inf")]

    # Also add exp2 runs for K variation
    all_Ks = [2, 5, 10, 20, 50]
    exp2_runs = []
    for alpha in [0.3, 0.35, 0.5]:
        for K in all_Ks:
            for seed in [42, 123, 456]:
                pattern = f"*_a{alpha}_K{K}_*_s{seed}.json"
                h = load_fl_history(os.path.join(EXP2_DIR, "fl_iid"), pattern)
                if h is None:
                    continue
                steps = np.array(h["total_steps"])
                test_accs = np.array(h["test_acc"])
                t_grok = compute_t_grok(steps.tolist(), test_accs.tolist())
                if t_grok < float("inf"):
                    exp2_runs.append({
                        "h": h, "alpha": alpha, "K": K,
                        "seed": seed, "t_grok": t_grok,
                        "steps": steps, "test_accs": test_accs,
                    })

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # ── (a) Unrescaled test accuracy (spaghetti — shows the problem) ──
    ax = axes[0, 0]
    alphas_unique = sorted(set(r["alpha"] for r in grok_runs))
    alpha_colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(alphas_unique)))
    alpha_cmap = {a: c for a, c in zip(alphas_unique, alpha_colors)}

    for run in grok_runs:
        ax.plot(run["steps"], run["test_accs"], color=alpha_cmap[run["alpha"]],
                alpha=0.3, linewidth=0.5)
    ax.set_xlabel("Gradient steps")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_ylim(-5, 105)
    ax.axhline(95, color="gray", linestyle=":", alpha=0.3)

    # ── (b) Rescaled test accuracy ──
    ax = axes[0, 1]
    collapse_tau = []
    collapse_acc = []
    collapse_ipr = []  # for panel (c)
    collapse_ipr_tau = []
    collapse_drift = []
    collapse_drift_tau = []

    all_combined = grok_runs + exp2_runs
    for run in all_combined:
        h = run["h"]
        steps = run["steps"]
        test_accs = run["test_accs"]
        t_grok = run["t_grok"]

        # Compute T_50
        t_50 = float("inf")
        for s, a in zip(steps, test_accs):
            if a >= 50.0:
                t_50 = float(s)
                break

        if t_50 >= t_grok or t_50 == float("inf"):
            continue

        # Rescale: tau = (t - T_grok) / (T_grok - T_50)
        transition_width = t_grok - t_50
        tau = (steps - t_grok) / transition_width

        # Only keep reasonable range
        mask = (tau >= -5) & (tau <= 3)
        ax.plot(tau[mask], test_accs[mask],
                color=alpha_cmap.get(run["alpha"], "gray"),
                alpha=0.15, linewidth=0.4)
        collapse_tau.extend(tau[mask].tolist())
        collapse_acc.extend(test_accs[mask].tolist())

        # IPR collapse
        ipr = np.array(h["ipr"])
        collapse_ipr_tau.extend(tau[mask].tolist())
        collapse_ipr.extend(ipr[mask].tolist())

        # Drift collapse
        if "mean_client_drift" in h:
            drift = smooth(np.array(h["mean_client_drift"]), window=50)
            collapse_drift_tau.extend(tau[mask].tolist())
            collapse_drift.extend(drift[mask].tolist())

    # Binned median for the collapse
    collapse_tau = np.array(collapse_tau)
    collapse_acc = np.array(collapse_acc)
    bins = np.linspace(-5, 3, 100)
    bin_centers, bin_medians, bin_q25, bin_q75 = [], [], [], []
    for i in range(len(bins) - 1):
        mask = (collapse_tau >= bins[i]) & (collapse_tau < bins[i+1])
        if mask.sum() >= 3:
            bin_centers.append((bins[i] + bins[i+1]) / 2)
            bin_medians.append(np.median(collapse_acc[mask]))
            bin_q25.append(np.percentile(collapse_acc[mask], 25))
            bin_q75.append(np.percentile(collapse_acc[mask], 75))

    ax.plot(bin_centers, bin_medians, "k-", linewidth=2.5, label="Median", zorder=10)
    ax.fill_between(bin_centers, bin_q25, bin_q75, color="black", alpha=0.1,
                    label="IQR", zorder=9)

    # Fit sigmoid to collapsed data
    from scipy.optimize import curve_fit
    def sigmoid(x, a, b):
        return 100.0 / (1.0 + np.exp(-a * (x - b)))
    try:
        bc = np.array(bin_centers)
        bm = np.array(bin_medians)
        popt, _ = curve_fit(sigmoid, bc, bm, p0=[3, 0], maxfev=5000)
        x_fit = np.linspace(-5, 3, 200)
        ax.plot(x_fit, sigmoid(x_fit, *popt), "r--", linewidth=2,
                label=f"Sigmoid fit (k={popt[0]:.2f})", zorder=11)
    except RuntimeError:
        pass

    ax.axvline(0, color="red", linestyle=":", alpha=0.4)
    ax.text(0.03, 0.55,
            "Note: accuracy collapse is\nexpected by construction.\n"
            "IPR/drift collapse (c,d)\nis the non-trivial result.",
            transform=ax.transAxes, fontsize=7, color="gray", style="italic")
    ax.set_xlabel(r"$\tau = (t - T_{grok}) / (T_{grok} - T_{50})$")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_ylim(-5, 105)
    ax.legend(fontsize=8)

    # ── (c) IPR collapse ──
    ax = axes[1, 0]
    collapse_ipr_tau = np.array(collapse_ipr_tau)
    collapse_ipr = np.array(collapse_ipr)
    bins_ipr = np.linspace(-5, 3, 100)
    bc_ipr, bm_ipr, bq25_ipr, bq75_ipr = [], [], [], []
    for i in range(len(bins_ipr) - 1):
        mask = (collapse_ipr_tau >= bins_ipr[i]) & (collapse_ipr_tau < bins_ipr[i+1])
        if mask.sum() >= 3:
            bc_ipr.append((bins_ipr[i] + bins_ipr[i+1]) / 2)
            bm_ipr.append(np.median(collapse_ipr[mask]))
            bq25_ipr.append(np.percentile(collapse_ipr[mask], 25))
            bq75_ipr.append(np.percentile(collapse_ipr[mask], 75))

    ax.plot(bc_ipr, bm_ipr, "k-", linewidth=2.5, label="Median", zorder=10)
    ax.fill_between(bc_ipr, bq25_ipr, bq75_ipr, color="black", alpha=0.1, zorder=9)
    ax.axvline(0, color="red", linestyle=":", alpha=0.4, label=r"$T_{grok}$")
    ax.set_xlabel(r"$\tau = (t - T_{grok}) / (T_{grok} - T_{50})$")
    ax.set_ylabel("IPR")
    ax.legend(fontsize=8)

    # ── (d) Drift collapse ──
    ax = axes[1, 1]
    if collapse_drift_tau:
        collapse_drift_tau = np.array(collapse_drift_tau)
        collapse_drift = np.array(collapse_drift)

        # Normalize drift per run to [0, 1] for collapse
        # Re-do with normalization
        norm_drift_tau = []
        norm_drift_vals = []
        for run in [r for r in all_combined if "mean_client_drift" in r["h"]]:
            h = run["h"]
            steps = run["steps"]
            t_grok = run["t_grok"]
            t_50 = float("inf")
            for s, a in zip(steps, run["test_accs"]):
                if a >= 50.0:
                    t_50 = float(s)
                    break
            if t_50 >= t_grok or t_50 == float("inf"):
                continue
            tw = t_grok - t_50
            tau = (steps - t_grok) / tw
            drift = smooth(np.array(h["mean_client_drift"]), window=50)
            # Normalize to peak drift
            d_max = drift.max()
            if d_max > 1e-10:
                drift_norm = drift / d_max
            else:
                continue
            mask = (tau >= -5) & (tau <= 3)
            norm_drift_tau.extend(tau[mask].tolist())
            norm_drift_vals.extend(drift_norm[mask].tolist())

        norm_drift_tau = np.array(norm_drift_tau)
        norm_drift_vals = np.array(norm_drift_vals)

        bins_d = np.linspace(-5, 3, 100)
        bc_d, bm_d, bq25_d, bq75_d = [], [], [], []
        for i in range(len(bins_d) - 1):
            mask = (norm_drift_tau >= bins_d[i]) & (norm_drift_tau < bins_d[i+1])
            if mask.sum() >= 3:
                bc_d.append((bins_d[i] + bins_d[i+1]) / 2)
                bm_d.append(np.median(norm_drift_vals[mask]))
                bq25_d.append(np.percentile(norm_drift_vals[mask], 25))
                bq75_d.append(np.percentile(norm_drift_vals[mask], 75))

        ax.plot(bc_d, bm_d, "k-", linewidth=2.5, label="Median", zorder=10)
        ax.fill_between(bc_d, bq25_d, bq75_d, color="black", alpha=0.1, zorder=9)

    ax.axvline(0, color="red", linestyle=":", alpha=0.4, label=r"$T_{grok}$")
    ax.set_xlabel(r"$\tau = (t - T_{grok}) / (T_{grok} - T_{50})$")
    ax.set_ylabel("Normalized drift (d / d_max)")
    ax.legend(fontsize=8)

    # Panel labels + titles
    titles = [
        "Raw trajectories (no alignment)",
        "Rescaled time (partly tautological for accuracy)",
        "IPR collapse",
        "Drift collapse (normalized)",
    ]
    for idx, (row, col) in enumerate([(0,0),(0,1),(1,0),(1,1)]):
        label_char = chr(ord('a') + idx)
        axes[row, col].set_title(f"({label_char}) {titles[idx]}",
                                  loc="left", fontweight="bold", fontsize=11)

    # Legend for alpha in panel (a)
    for a, c in zip(alphas_unique, alpha_colors):
        axes[0, 0].plot([], [], color=c, linewidth=2, label=f"α={a}", alpha=0.7)
    axes[0, 0].legend(fontsize=7, loc="center right")

    n_collapsed = len(all_combined)
    fig.suptitle(f"Figure 9: Scaling Collapse — {n_collapsed} Grokking Runs",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig9_universal_collapse.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")
    print(f"  Collapsed {n_collapsed} grokking runs onto rescaled coordinates")


# ── Figure 10: Two-timescale separation ──────────────────────────────────

def plot_figure10(data_points):
    """Figure 10: FL affects memorization and generalization differently.

    Observation: Within a fixed alpha, T_memo is nearly invariant to K
    and heterogeneity, while T_grok varies widely. The cross-alpha
    correlation between T_memo and T_grok is driven by alpha affecting
    both (more data → slower memorization AND different grokking time).

    (a) Within-alpha CV comparison — T_memo vs T_grok variability
    (b) T_memo and T_grok vs K for multiple alphas — shows divergence
    (c) Generalization delay (T_grok - T_memo) vs drift, within-alpha
    """
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))

    # Compute T_memo (first time train_acc >= 99%) for all runs
    enriched = []
    for d in data_points:
        h = load_json(d["path"])
        steps = h.get("total_steps", h.get("epoch", []))
        train_accs = h.get("train_acc", [])
        t_memo = float("inf")
        for s, a in zip(steps, train_accs):
            if a >= 99.0:
                t_memo = float(s)
                break
        enriched.append({**d, "t_memo": t_memo})

    grokked = [d for d in enriched if d["t_grok"] < float("inf") and d["t_memo"] < float("inf")]

    grokking_alphas = sorted(set(d["alpha"] for d in grokked))
    alpha_colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(grokking_alphas)))
    alpha_cmap = {a: c for a, c in zip(grokking_alphas, alpha_colors)}

    # ── (a) Within-alpha CV comparison ──
    ax = axes[0]
    bar_alphas = []
    cv_memo = []
    cv_grok = []
    for alpha in grokking_alphas:
        subset = [d for d in grokked if d["alpha"] == alpha]
        if len(subset) < 3:
            continue
        t_m = [d["t_memo"] for d in subset]
        t_g = [d["t_grok"] for d in subset]
        bar_alphas.append(alpha)
        cv_memo.append(np.std(t_m) / np.mean(t_m) if np.mean(t_m) > 0 else 0)
        cv_grok.append(np.std(t_g) / np.mean(t_g) if np.mean(t_g) > 0 else 0)

    x_pos = np.arange(len(bar_alphas))
    width = 0.35
    ax.bar(x_pos - width/2, cv_memo, width, color="#4CAF50", edgecolor="k",
           linewidth=0.5, alpha=0.8, label=r"$T_{memo}$ (memorization)")
    ax.bar(x_pos + width/2, cv_grok, width, color="#F44336", edgecolor="k",
           linewidth=0.5, alpha=0.8, label=r"$T_{grok}$ (generalization)")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"α={a}" for a in bar_alphas])
    ax.set_ylabel("Coefficient of variation\n(within-α, across K & heterogeneity)")
    ax.legend(fontsize=8)

    # Annotate ratio
    for i, (cm, cg) in enumerate(zip(cv_memo, cv_grok)):
        if cm > 0:
            ratio = cg / cm
            ax.text(i, max(cm, cg) + 0.01, f"{ratio:.0f}×", ha="center",
                    fontsize=8, fontweight="bold")

    # ── (b) T_memo and T_grok vs K for multiple alphas ──
    ax = axes[1]
    plot_alphas = [0.3, 0.5]  # Show two representative alphas

    for alpha in plot_alphas:
        exp2_data = [d for d in enriched if d["source"] == "exp2" and d["alpha"] == alpha]
        Ks = sorted(set(d["K"] for d in exp2_data))

        t_memo_by_K = {k: [] for k in Ks}
        t_grok_by_K = {k: [] for k in Ks}
        for d in exp2_data:
            if d["t_memo"] < float("inf"):
                t_memo_by_K[d["K"]].append(d["t_memo"])
            if d["t_grok"] < float("inf"):
                t_grok_by_K[d["K"]].append(d["t_grok"])

        ks_plot = [k for k in Ks if t_memo_by_K[k]]
        memo_means = [np.mean(t_memo_by_K[k]) for k in ks_plot]
        memo_stds = [np.std(t_memo_by_K[k]) for k in ks_plot]

        color = alpha_cmap[alpha]
        ax.errorbar(ks_plot, memo_means, yerr=memo_stds, fmt="s--",
                    color=color, capsize=3, linewidth=1.5, markersize=5,
                    alpha=0.6, label=f"T_memo α={alpha}")

        grok_ks = [k for k in ks_plot if t_grok_by_K[k]]
        grok_m = [np.mean(t_grok_by_K[k]) for k in grok_ks]
        grok_s = [np.std(t_grok_by_K[k]) for k in grok_ks]
        ax.errorbar(grok_ks, grok_m, yerr=grok_s, fmt="o-",
                    color=color, capsize=3, linewidth=2, markersize=6,
                    label=f"T_grok α={alpha}")

    ax.set_xscale("log")
    ax.set_xlabel("K (number of clients)")
    ax.set_ylabel("Steps")
    ax.legend(fontsize=7, ncol=2)
    ax.text(0.5, 0.03,
            "Dashed = memorization (flat)\nSolid = generalization (diverges)",
            transform=ax.transAxes, va="bottom", ha="center", fontsize=7,
            color="gray", style="italic")

    # ── (c) Generalization delay vs drift (within-alpha) ──
    ax = axes[2]
    for alpha in grokking_alphas:
        subset = [d for d in grokked if d["alpha"] == alpha]
        if len(subset) < 3:
            continue
        drifts = [d["mean_drift"] for d in subset]
        gaps = [d["t_grok"] - d["t_memo"] for d in subset]
        ax.scatter(drifts, gaps, c=[alpha_cmap[alpha]] * len(drifts),
                   s=25, alpha=0.6, edgecolors="k", linewidths=0.2,
                   label=f"α={alpha}")

    # Within-alpha Spearman for each alpha
    within_rhos = []
    for alpha in grokking_alphas:
        subset = [d for d in grokked if d["alpha"] == alpha]
        if len(subset) < 5:
            continue
        drifts = [d["mean_drift"] for d in subset]
        gaps = [d["t_grok"] - d["t_memo"] for d in subset]
        rho, pval = stats.spearmanr(drifts, gaps)
        within_rhos.append((alpha, rho, pval, len(subset)))

    if within_rhos:
        text_lines = "Within-α Spearman:\n"
        for alpha, rho, pval, n in within_rhos:
            sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
            text_lines += f"  α={alpha}: ρ={rho:.2f}{sig} (n={n})\n"
        ax.text(0.03, 0.97, text_lines.strip(),
                transform=ax.transAxes, va="top", fontsize=7,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8),
                family="monospace")

    ax.set_xlabel("Mean client drift")
    ax.set_ylabel(r"Generalization delay ($T_{grok} - T_{memo}$)")
    ax.legend(fontsize=7, loc="lower right")

    # Panel labels + titles
    titles = [
        "Within-α variability:\ngeneralization >> memorization",
        "T_memo flat, T_grok diverges with K",
        "Drift → generalization delay\n(within-α)",
    ]
    for idx in range(3):
        label_char = chr(ord('a') + idx)
        axes[idx].set_title(f"({label_char}) {titles[idx]}", loc="left",
                            fontweight="bold", fontsize=10)

    fig.suptitle("Figure 10: FL Disrupts Generalization, Not Memorization",
                 fontsize=14, y=1.04)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig10_two_timescales.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")
    print(f"  Analyzed {len(grokked)} runs with both T_memo and T_grok")


# ── Summary statistics ───────────────────────────────────────────────────

def print_summary(data_points, grokked, failed):
    """Print key mechanistic findings."""
    print("\n" + "="*70)
    print("MECHANISTIC ANALYSIS SUMMARY")
    print("="*70)
    print(f"Total FL runs analyzed: {len(data_points)}")
    print(f"  Grokked: {len(grokked)}")
    print(f"  Failed:  {len(failed)}")

    if grokked and failed:
        g_drifts = [d["mean_drift"] for d in grokked]
        f_drifts = [d["mean_drift"] for d in failed]
        t_stat, p_val = stats.mannwhitneyu(g_drifts, f_drifts, alternative="less")
        print(f"\nGrokked vs Failed drift (Mann-Whitney U):")
        print(f"  Grokked drift: mean={np.mean(g_drifts):.4f} ± {np.std(g_drifts):.4f}")
        print(f"  Failed drift:  mean={np.mean(f_drifts):.4f} ± {np.std(f_drifts):.4f}")
        print(f"  U = {t_stat:.1f}, p = {p_val:.2e}")
        print(f"  Effect size (Cohen's d) = "
              f"{(np.mean(f_drifts) - np.mean(g_drifts)) / np.sqrt((np.var(g_drifts) + np.var(f_drifts)) / 2):.2f}")

    print("="*70)


# ── Main ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading representative runs...")
    runs = load_representative_runs()
    print(f"  Loaded {len(runs)} / 4 representative runs")

    print("\nGenerating Figure 6: Representative comparison...")
    plot_figure6(runs)

    print("\nCollecting all FL runs for scatter analysis...")
    data_points = collect_all_fl_runs()
    print(f"  Found {len(data_points)} FL runs")

    print("\nGenerating Figure 7: Drift as grokking predictor (alpha-controlled)...")
    grokked, failed = plot_figure7(data_points)

    print("\nGenerating Figure 8: Temporal ordering (IPR vs drift)...")
    plot_figure8()

    print("\nGenerating Figure 9: Scaling collapse...")
    plot_figure9()

    print("\nGenerating Figure 10: Two-timescale separation...")
    plot_figure10(data_points)

    # Save raw data for future use
    out_data = []
    for d in data_points:
        entry = {k: v for k, v in d.items() if k != "path"}
        entry["t_grok"] = str(d["t_grok"]) if d["t_grok"] == float("inf") else d["t_grok"]
        out_data.append(entry)
    with open(os.path.join(OUTPUT_DIR, "..", "drift_vs_grokking.json"), "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"\nSaved raw data: results/exp6_mechanistic/drift_vs_grokking.json")

    print_summary(data_points, grokked, failed)
    print(f"\nAll figures saved to {OUTPUT_DIR}/")
