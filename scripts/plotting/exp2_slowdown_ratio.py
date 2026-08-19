"""main's exp2_slowdown_ratio.png, one figure per setup, one line per alpha.

    venv/bin/python scripts/plotting/exp2_slowdown_ratio.py            # -> paper/
    venv/bin/python scripts/plotting/exp2_slowdown_ratio.py --out DIR

main plotted T_grok(FL) / T_grok(cent_full) against K with one line per alpha on
its single setup (scripts/plot_exp2.py::plot_slowdown_ratio). v2 runs every setup
at its own working point, so the figure splits into one panel per setup; the
`aggregation_alpha2` campaign adds a second, easier alpha to each so the panels
recover main's multi-line form.

Each alpha carries its OWN centralized denominator. That is the part of this
script most able to fail quietly -- a ratio taken against the wrong baseline
still plots a perfectly plausible line -- so the two arms are matched on the data
axis and nothing is shared between series.

Departures from main's version, each because v2 measured something main could not:

  t_first_cross, not t_grok. The arms are compute-matched (setup A: 50,000
  centralized epochs against 10,000 rounds x E=5), so t_grok is admissible in
  principle -- but it requires the bar to hold for the REST of the run, which
  makes it depend on the logging rate on an unstable setup and on the budget
  (RESULTS 13.4, 14.4). t_first_cross is the statistic RESULTS 15.1 reports.

  Censored cells are DRAWN, not skipped. main's version dropped any cell whose
  T_grok was inf, so a setup that stopped grokking left no mark. Here each series
  gets its own censored row above the data, annotated with the grokked fraction.

  PARTLY censored cells are labelled with their fraction, because otherwise their
  median silently becomes a median over survivors.

  Each series shades its own baseline seed spread. The ratio divides by a 3-seed
  median, so part of any deviation from 1.0 is noise in the denominator. On setup
  C that band swallows the curve, which is why RESULTS 15.1 reports no number
  for C.
"""
import argparse
import csv
import glob
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from fedgrok.manifest import load_manifest, run_id            # noqa: E402
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Categorical slots 1 and 2 of the validated default palette. Verified with
# scripts/validate_palette.py (light, surface #fcfcfb, --pairs all): every check
# PASSes, CVD dE 24.7, normal-vision 33.6, contrast 4.30 / 3.12. A third alpha
# would need slot 3 (#1baf7a) and a re-run -- that slot is sub-3:1 on this
# surface and would oblige direct labels under the relief rule.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
CENSORED = "#d03b3b"
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#898781"
RULE, SURFACE = "#e1e0d9", "#fcfcfb"

# Rows are selected by MANIFEST RUN ID, not by the `group` column. A cell that
# dedupes against an earlier campaign was executed by that campaign, so its
# result JSON carries the ORIGINAL group tag -- C's alpha=0.50 arm lives under
# `setup_k_ladder` and `c_capacity`. Filtering on group silently drops 12 of the
# 15 points on C's second line; content hashes do not have that failure mode.
MANIFESTS = ("manifests/t2_aggregation.jsonl",
             "manifests/t2_aggregation_alpha2.jsonl")

SETUPS = [
    ("A",  "Quadratic MLP · mod-97 · GD"),
    ("A'", "Quadratic MLP · mod-97 · AdamW"),
    ("B",  "Nanda transformer · mod-113 · AdamW"),
    ("C",  "Transformer · S₅ · AdamW"),
    ("D",  "Quadratic MLP · S₅ · AdamW"),
    ("E",  "Omnigrok MLP · MNIST-1k · AdamW"),
]


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("inf")


def _tfc(row):
    """First crossing, falling back to a recorded t_grok (RESULTS 18 note)."""
    v = _f(row["t_first_cross"])
    if v != float("inf"):
        return v
    return _f(row["t_grok"]) if row["grokked"].lower() == "true" else float("inf")


def _axis_value(row):
    """The setup's data-fraction axis: n_train on MNIST, alpha everywhere else.

    alpha is a live Config field on MNIST but the dataset builder never reads it
    (data/registry.py dispatches MNIST to load_mnist_subset(n_train, n_test,
    seed)), so grouping E by alpha would collapse both of its series into one.
    """
    return row["n_train"] if row["dataset"] == "mnist" else row["alpha"]


# ── v1 fallback for setup A ──────────────────────────────────────────────────
# v1 was a single-setup study on A and swept alpha, so its runs.csv already holds
# complete K ladders at alpha 0.35 and 0.50 that v2 has never re-measured. They
# are drawn here rather than re-run.
#
# THE STATISTIC IS NOT THE SAME and the figure must not pretend otherwise. v1
# recorded `t_grok` only; the per-run histories it computed that from are
# gitignored and gone, and the surviving logs print one point per 1,000 rounds --
# too coarse to recover a first crossing (the whole transition on A/K=2 happens
# inside a single printed interval). RESULTS 14.4 shows t_grok is not comparable
# across budgets, so these series are dashed and labelled with their statistic.
V1_CSV = "results/data/runs.csv"
V1_ALPHAS = ("0.35", "0.5")


def _v1_num(row, key):
    try:
        return float(row[key])
    except (TypeError, ValueError, KeyError):
        return None


def load_v1_setup_a(csv_path=V1_CSV):
    """Series dicts for setup A from v1's runs.csv, shaped like load()'s."""
    if not os.path.exists(csv_path):
        return []
    rows = list(csv.DictReader(open(csv_path)))
    out = []
    for alpha in V1_ALPHAS:
        # fraction_train must be 1.0: Phase 0.6 fixed a step-axis bug that only
        # affects partial participation, and v1's alpha=0.50 ladder contains
        # ft in {0.2,0.4,0.6} rows whose x-axis is off by up to 2.5x.
        # experiment == "exp2" is load-bearing, not tidiness. Selecting on
        # (alpha, K, E, fraction_train, partition) alone pools v1's exp3a
        # (Dirichlet), exp3b (operand/target shards), exp4a/b/c (other E and
        # participation) and exp7 (other tasks and primes -- p=53, division,
        # multiplication) into the same cell: 53 rows where exp2 contributes 3.
        # That is the silent-pooling failure Phase 0.4 fixed for v2, arriving
        # here through the back door.
        fed = [r for r in rows
               if r.get("experiment") == "exp2"
               and r.get("alpha") == alpha
               and r.get("task") == "addition" and r.get("p") == "97"
               and (r.get("num_clients") or "") not in ("", "None")
               and r.get("local_epochs") == "5"
               and r.get("fraction_train") == "1.0"
               and (r.get("partition") or "iid") == "iid"]
        cent = [r for r in rows
                if r.get("experiment") == "exp2"
                and r.get("alpha") == alpha
                and r.get("task") == "addition" and r.get("p") == "97"
                and (r.get("num_clients") or "") in ("", "None")
                and (r.get("grokked") or "").lower() == "true"]
        base = [t for t in (_v1_num(r, "t_grok") for r in cent) if t]
        if not fed or not base:
            continue
        by_k = {}
        for r in fed:
            by_k.setdefault(int(float(r["num_clients"])), []).append(r)
        cells = []
        for k in sorted(by_k):
            v = by_k[k]
            fin = [t for t in (_v1_num(r, "t_grok") for r in v
                               if (r.get("grokked") or "").lower() == "true") if t]
            cells.append({"K": k, "n": len(v), "grokked": len(fin),
                          "median": float(np.median(fin)) if fin else None,
                          "lo": min(fin) if fin else None,
                          "hi": max(fin) if fin else None})
        out.append({"val": alpha, "base": float(np.median(base)),
                    "base_lo": min(base), "base_hi": max(base),
                    "base_n": len(base), "cells": cells, "legacy": True})
    return out


def load(csv_path="results/data/runs_v2.csv"):
    """{setup: [series]} — one series per data-axis value, each self-contained."""
    claimed = {}          # run id -> the manifest spec that claims it
    for m in MANIFESTS:
        if not os.path.exists(m):
            continue
        for spec in load_manifest(m):
            claimed.setdefault(run_id(spec), spec)
    banked = {r["id"]: r for r in csv.DictReader(open(csv_path))}

    rows = []
    for rid, spec in claimed.items():
        row = banked.get(rid)
        if row is None:
            continue          # not run yet
        # setup/arm come from the SPEC: a deduped row carries the other
        # campaign's tags, so the CSV's own arm column cannot be trusted here.
        rows.append({**row, "_setup": spec.get("setup"), "_arm": spec.get("arm")})

    out = {}
    for setup, _ in SETUPS:
        srows = [r for r in rows if r["_setup"] == setup]
        if not srows:
            continue
        series = []
        for val in sorted({_axis_value(r) for r in srows}, key=float):
            sub = [r for r in srows if _axis_value(r) == val]
            base = [_tfc(r) for r in sub if r["_arm"] == "cent_full"]
            base = [x for x in base if np.isfinite(x)]
            if not base:
                continue
            cells = []
            for k in sorted({int(r["num_clients"]) for r in sub if r["_arm"] == "fl"}):
                v = [_tfc(r) for r in sub
                     if r["_arm"] == "fl" and int(r["num_clients"]) == k]
                fin = [x for x in v if np.isfinite(x)]
                cells.append({"K": k, "n": len(v), "grokked": len(fin),
                              "median": float(np.median(fin)) if fin else None,
                              "lo": min(fin) if fin else None,
                              "hi": max(fin) if fin else None})
            if cells:
                series.append({"val": val, "base": float(np.median(base)),
                               "base_lo": min(base), "base_hi": max(base),
                               "base_n": len(base), "cells": cells})
        if setup == "A":
            series += load_v1_setup_a()
        if series:
            out[setup] = series
    return out


def draw(setup, label, series, out_dir):
    fig, ax = plt.subplots(figsize=(6.8, 4.7))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    is_mnist = setup == "E"
    axis_name = "n_train" if is_mnist else "α"

    highs, lows, all_ks = [1.0], [1.0], set()
    for s in series:
        b = s["base"]
        highs += [c["hi"] / b for c in s["cells"] if c["hi"]] + [s["base_hi"] / b]
        lows += [c["lo"] / b for c in s["cells"] if c["lo"]] + [s["base_lo"] / b]
        all_ks |= {c["K"] for c in s["cells"]}
    top, bottom = max(highs), min(lows)
    span = (top - bottom) or 1.0
    n_cens_rows = sum(1 for s in series if any(c["median"] is None for c in s["cells"]))
    partial = any(0 < c["grokked"] < c["n"] for s in series for c in s["cells"])
    ymax = top + span * (0.12 + 0.13 * n_cens_rows)
    # A ratio is a positive quantity, so the padding must not carry the axis
    # below zero -- on a wide-range panel like C, span*0.17 is larger than the
    # smallest ratio and would open a band of impossible values under the data.
    ymin = max(0.0, bottom - span * (0.17 if partial else 0.09))

    for i, s in enumerate(series):
        col, b = SERIES[i % len(SERIES)], s["base"]
        ax.axhspan(s["base_lo"] / b, s["base_hi"] / b, color=col, alpha=0.09, lw=0,
                   zorder=0)
    ax.axhline(1.0, color=INK3, ls="--", lw=1.1, zorder=1)

    handles, names = [], []
    cens_row = 0
    for i, s in enumerate(series):
        col, b = SERIES[i % len(SERIES)], s["base"]
        ok = [c for c in s["cells"] if c["median"] is not None]
        cens = [c for c in s["cells"] if c["median"] is None]
        ratios = [c["median"] / b for c in ok]
        if ok:
            yerr = [[r - c["lo"] / b for r, c in zip(ratios, ok)],
                    [c["hi"] / b - r for r, c in zip(ratios, ok)]]
            legacy = s.get("legacy", False)
            h = ax.errorbar([c["K"] for c in ok], ratios, yerr=yerr,
                            marker="s" if legacy else "o",
                            ls="--" if legacy else "-",
                            color=col, capsize=3, lw=1.8, markersize=6,
                            markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=3)
            handles.append(h)
            # Name the statistic on the legacy series. Dropping it would put a
            # t_grok line and a t_first_cross line on one axis unlabelled, which
            # RESULTS 14.4 says are not the same quantity.
            names.append(f"{axis_name} = {float(s['val']):g}"
                         f"   (baseline {b:,.0f})"
                         + ("   [v1, T_grok]" if legacy else ""))
            for j, (c, r) in enumerate(zip(ok, ratios)):
                last = j == len(ok) - 1
                # Stagger vertically by series: at easy alpha every line sits
                # on 1.00x for the low-K cells and the labels would overprint.
                dy = (i - (len(series) - 1) / 2) * 9
                ax.annotate(f"{r:.2f}×", (c["K"], r), textcoords="offset points",
                            xytext=(-9 if last else 9, dy),
                            ha="right" if last else "left", va="center",
                            fontsize=8, color=col, family="monospace", zorder=4)
                if c["grokked"] < c["n"]:
                    ax.annotate(f"{c['grokked']}/{c['n']}", (c["K"], c["lo"] / b),
                                textcoords="offset points", xytext=(0, -11),
                                ha="center", va="top", fontsize=7.5,
                                color=CENSORED, family="monospace", zorder=4)
        if cens:
            y = top + span * (0.10 + 0.13 * cens_row)
            cens_row += 1
            for c in cens:
                ax.plot([c["K"]], [y], marker="o", mfc="none", mec=CENSORED,
                        mew=1.9, markersize=8, ls="none", zorder=3)
                ax.annotate(f"{c['grokked']}/{c['n']}", (c["K"], y),
                            textcoords="offset points", xytext=(11, 0), ha="left",
                            va="center", fontsize=7.5, color=CENSORED,
                            family="monospace")
            ax.annotate(f"{axis_name}={float(s['val']):g} censored",
                        (min(all_ks), y), textcoords="offset points",
                        xytext=(-6, 0), ha="right", va="center", fontsize=7.5,
                        color=col, family="monospace")
    ax.set_ylim(ymin, ymax)

    ks = sorted(all_ks)
    ax.set_xscale("log")
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.minorticks_off()
    ax.set_xlim(min(ks) * 0.62, max(ks) * 1.42)
    ax.set_xlabel("K  (clients)", fontsize=10, color=INK2)
    # The v1 series are t_grok, not first crossing, so a panel carrying one
    # cannot label its axis "first crossing"; the legend names each line's
    # statistic and the axis stays neutral.
    ylab = ("federated / centralized\ngrokking time"
            if any(x.get("legacy") for x in series)
            else "federated / centralized\nfirst crossing")
    ax.set_ylabel(ylab, fontsize=10, color=INK2)
    ax.set_title(f"Setup {setup} — {label}", fontsize=12.5, color=INK, pad=18,
                 loc="left")
    ax.annotate("E=5 · FedAvg · iid · 3 seeds · shaded band = that line's "
                "centralized seed spread",
                xy=(0, 1.02), xycoords="axes fraction", fontsize=8, color=INK3)

    if handles:
        leg = ax.legend(handles, names, fontsize=8, loc="upper left", frameon=True,
                        framealpha=0.95, borderpad=0.6)
        leg.get_frame().set_edgecolor(RULE)
        leg.get_frame().set_facecolor(SURFACE)
        for t in leg.get_texts():
            t.set_color(INK2)

    ax.grid(axis="y", color=RULE, lw=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(RULE)
    ax.tick_params(colors=INK3, labelsize=9)

    os.makedirs(out_dir, exist_ok=True)
    name = f"exp2_slowdown_ratio_{setup.replace(chr(39), 'prime')}.png"
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return path, len(series)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="paper")
    ap.add_argument("--csv", default="results/data/runs_v2.csv")
    a = ap.parse_args()
    data = load(a.csv)
    for setup, label in SETUPS:
        if setup not in data:
            print(f"  setup {setup}: no rows, skipped")
            continue
        path, n = draw(setup, label, data[setup], a.out)
        vals = ", ".join(f"{float(s['val']):g}" for s in data[setup])
        print(f"  {path}   {n} line(s): {vals}")


if __name__ == "__main__":
    main()
