"""Publication-quality figures for all experiments."""

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


def _load(path):
    with open(path) as f:
        return json.load(f)


def _parse_inf(val):
    if val == "inf":
        return float("inf")
    return float(val)


def plot_exp1_phase_boundary(results_path: str, output_dir: str):
    """Plot T_grok vs alpha with error bars."""
    data = _load(results_path)
    alphas = []
    t_groks_mean = []
    t_groks_std = []

    for entry in data["per_alpha"]:
        alpha = entry["alpha"]
        summary = entry["summary"]
        t_mean = _parse_inf(summary["t_grok_mean"])
        t_std = _parse_inf(summary["t_grok_std"])
        alphas.append(alpha)
        t_groks_mean.append(t_mean if t_mean < float("inf") else None)
        t_groks_std.append(t_std if t_std < float("inf") else 0)

    fig, ax = plt.subplots(figsize=(7, 5))
    finite_mask = [t is not None for t in t_groks_mean]
    inf_mask = [t is None for t in t_groks_mean]

    finite_a = [a for a, m in zip(alphas, finite_mask) if m]
    finite_t = [t for t in t_groks_mean if t is not None]
    finite_s = [s for s, m in zip(t_groks_std, finite_mask) if m]

    ax.errorbar(finite_a, finite_t, yerr=finite_s, fmt="o-", capsize=4,
                color="tab:blue", label="T_grok (3 seeds)")

    inf_a = [a for a, m in zip(alphas, inf_mask) if m]
    if inf_a:
        ymax = max(finite_t) * 1.2 if finite_t else 100000
        ax.scatter(inf_a, [ymax] * len(inf_a), marker="x",
                   color="red", s=100, zorder=5, label="No grokking")

    ax.set_xlabel("Training fraction (alpha)")
    ax.set_ylabel("Grokking step (T_grok)")
    ax.set_title("Centralized Phase Boundary")
    ax.legend()
    ax.set_xscale("log")

    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, "fig_exp1_boundary.png"))
    plt.close(fig)


def plot_exp2_aggregation(results_path: str, output_dir: str):
    """For each alpha, plot T_grok vs K: centralized-full / reduced / FL."""
    data = _load(results_path)

    by_alpha = {}
    for entry in data:
        a = entry["alpha"]
        if a not in by_alpha:
            by_alpha[a] = []
        by_alpha[a].append(entry)

    n_alphas = len(by_alpha)
    fig, axes = plt.subplots(1, min(n_alphas, 4), figsize=(5 * min(n_alphas, 4), 5),
                              squeeze=False)

    for idx, (alpha, entries) in enumerate(sorted(by_alpha.items())):
        if idx >= 4:
            break
        ax = axes[0, idx]
        ks = sorted(set(e["K"] for e in entries))

        for condition, color, marker in [
            ("cent_full", "tab:green", "s"),
            ("cent_reduced", "tab:orange", "^"),
            ("fl_iid", "tab:blue", "o"),
        ]:
            t_vals = []
            for k in ks:
                match = [e for e in entries if e["K"] == k]
                if match:
                    t = _parse_inf(match[0][condition]["summary"]["t_grok_mean"])
                    t_vals.append(t if t < float("inf") else None)
                else:
                    t_vals.append(None)
            finite_k = [k for k, t in zip(ks, t_vals) if t is not None]
            finite_t = [t for t in t_vals if t is not None]
            ax.plot(finite_k, finite_t, f"{marker}-", label=condition.replace("_", " "),
                    color=color)

        ax.set_xlabel("K (clients)")
        ax.set_ylabel("T_grok")
        ax.set_title(f"alpha = {alpha}")
        ax.legend(fontsize=8)
        ax.set_xscale("log")

    fig.suptitle("Aggregation Effect: Centralized vs FL", fontsize=14)
    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, "fig_exp2_aggregation.png"))
    plt.close(fig)


def plot_exp3a_phase_diagram(results_path: str, output_dir: str):
    """Heatmap: x=Dirichlet alpha (log), y=training fraction, color=T_grok."""
    data = _load(results_path)

    alphas = sorted(set(r["alpha"] for r in data))
    dir_alphas = sorted(set(r["dirichlet_alpha"] for r in data))

    grid = np.full((len(alphas), len(dir_alphas)), np.nan)
    for r in data:
        i = alphas.index(r["alpha"])
        j = dir_alphas.index(r["dirichlet_alpha"])
        t = _parse_inf(r["summary"]["t_grok_mean"])
        grid[i, j] = t if t < float("inf") else np.nan

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(grid, aspect="auto", origin="lower",
                   extent=[0, len(dir_alphas), 0, len(alphas)],
                   cmap="viridis_r")
    ax.set_xticks(np.arange(len(dir_alphas)) + 0.5)
    ax.set_xticklabels([str(d) for d in dir_alphas])
    ax.set_yticks(np.arange(len(alphas)) + 0.5)
    ax.set_yticklabels([str(a) for a in alphas])
    ax.set_xlabel("Dirichlet alpha (heterogeneity)")
    ax.set_ylabel("Training fraction (alpha)")
    ax.set_title("Grokking Phase Diagram")
    fig.colorbar(im, ax=ax, label="T_grok")

    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, "fig_exp3a_phase_diagram.png"))
    plt.close(fig)


def plot_exp4a_drift(results_path: str, output_dir: str):
    """T_grok vs E, two lines (IID vs non-IID), per alpha panel."""
    data = _load(results_path)

    alphas = sorted(set(r["alpha"] for r in data))
    fig, axes = plt.subplots(1, len(alphas), figsize=(5 * len(alphas), 5), squeeze=False)

    for idx, alpha in enumerate(alphas):
        ax = axes[0, idx]
        for het, color in [("iid", "tab:blue"), ("noniid", "tab:red")]:
            entries = [r for r in data if r["alpha"] == alpha and r["heterogeneity"] == het]
            entries.sort(key=lambda r: r["E"])
            es = [r["E"] for r in entries]
            ts = [_parse_inf(r["summary"]["t_grok_mean"]) for r in entries]
            finite_e = [e for e, t in zip(es, ts) if t < float("inf")]
            finite_t = [t for t in ts if t < float("inf")]
            ax.plot(finite_e, finite_t, "o-", label=het, color=color)

        ax.set_xlabel("Local epochs (E)")
        ax.set_ylabel("T_grok")
        ax.set_title(f"alpha = {alpha}")
        ax.legend()

    fig.suptitle("Drift Accumulation x Heterogeneity", fontsize=14)
    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, "fig_exp4a_drift.png"))
    plt.close(fig)


def plot_exp5_algorithms(results_path: str, output_dir: str):
    """Bar chart of T_grok across algorithms for each hard setting."""
    data = _load(results_path)

    settings = sorted(set(r["setting"] for r in data))
    fig, axes = plt.subplots(1, len(settings), figsize=(6 * len(settings), 5), squeeze=False)

    for idx, setting in enumerate(settings):
        ax = axes[0, idx]
        entries = [r for r in data if r["setting"] == setting]
        labels = [r["algorithm"] for r in entries]
        ts = [_parse_inf(r["summary"]["t_grok_mean"]) for r in entries]

        colors = ["lightgray" if t == float("inf") else "tab:blue" for t in ts]
        max_finite = max((t for t in ts if t < float("inf")), default=0)
        plot_ts = [t if t < float("inf") else max_finite * 1.3 for t in ts]

        ax.bar(range(len(labels)), plot_ts, color=colors)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("T_grok")
        ax.set_title(f"Setting: {setting}")

    fig.suptitle("Algorithm Comparison", fontsize=14)
    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, "fig_exp5_algorithms.png"))
    plt.close(fig)


def plot_exp6_drift_scatter(drift_data_path: str, output_dir: str):
    """Scatter: mean client drift vs T_grok across all FL runs."""
    data = _load(drift_data_path)

    drifts = [d["mean_drift"] for d in data]
    t_groks = [_parse_inf(d["t_grok"]) for d in data]

    finite_mask = [t < float("inf") for t in t_groks]
    inf_mask = [not m for m in finite_mask]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(
        [d for d, m in zip(drifts, finite_mask) if m],
        [t for t, m in zip(t_groks, finite_mask) if m],
        alpha=0.6, label="Grokked", color="tab:blue"
    )
    if any(inf_mask):
        ymax = max((t for t in t_groks if t < float("inf")), default=100000) * 1.2
        ax.scatter(
            [d for d, m in zip(drifts, inf_mask) if m],
            [ymax] * sum(inf_mask),
            marker="x", color="red", s=60, label="No grokking"
        )

    ax.set_xlabel("Mean client drift")
    ax.set_ylabel("T_grok")
    ax.set_title("Client Drift vs Grokking Step")
    ax.legend()

    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, "fig_exp6_drift_scatter.png"))
    plt.close(fig)
