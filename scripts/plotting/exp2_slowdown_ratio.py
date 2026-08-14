"""main's exp2_slowdown_ratio.png, one figure per setup.

    venv/bin/python scripts/plotting/exp2_slowdown_ratio.py            # -> paper/
    venv/bin/python scripts/plotting/exp2_slowdown_ratio.py --out DIR

main plotted T_grok(FL) / T_grok(cent_full) against K with one line per alpha,
on a single setup (scripts/plot_exp2.py::plot_slowdown_ratio). v2's exp2 runs
every setup at its OWN working alpha, so alpha is no longer a free axis and the
figure splits: one panel per setup, one line each.

Three deliberate departures from main's version, each because v2 measured
something main could not:

  t_first_cross, not t_grok. The arms are compute-matched (setup A: 50,000
  centralized epochs against 10,000 rounds x E=5), so t_grok is admissible in
  principle -- but it requires the bar to hold for the REST of the run, which
  makes it depend on the logging rate on an unstable setup and on the budget
  (RESULTS 13.4, 14.4). t_first_cross is the statistic RESULTS 15.1 reports.

  Censored cells are DRAWN, not skipped. main's version dropped any cell whose
  T_grok was inf, so a setup that stopped grokking left no mark. Here it is an
  open marker at the top of the axis annotated with the grokked fraction --
  B and D at K=50 are 0/3, and that is the result rather than missing data.

  PARTLY censored cells are labelled with their grokked fraction. Only setup C
  at K=50 is affected (2/3), and its median is over the two survivors.

  The baseline's own seed spread is shaded. The ratio divides by a 3-seed
  median, so part of any deviation from 1.0 is noise in the denominator. The
  band is [min, max] of the centralized seeds over their median: if a point sits
  inside it, the ratio is not resolved. On setup C the band swallows the whole
  curve, which is why RESULTS 15.1 reports no number for C.
"""
import argparse
import csv
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Validated with scripts/validate_palette.py (light, surface #fcfcfb, --pairs all):
# all checks PASS, CVD dE 23.8, normal-vision 31.6, contrast 4.30 / 4.68.
SERIES, CENSORED = "#2a78d6", "#d03b3b"
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#898781"
RULE, SURFACE = "#e1e0d9", "#fcfcfb"

SETUPS = [
    ("A",  "Quadratic MLP · mod-97 · GD",              "α=0.30"),
    ("A'", "Quadratic MLP · mod-97 · AdamW",           "α=0.20"),
    ("B",  "Nanda transformer · mod-113 · AdamW",      "α=0.30"),
    ("C",  "Transformer · S₅ · AdamW",                 "α=0.40"),
    ("D",  "Quadratic MLP · S₅ · AdamW",               "α=0.30"),
    ("E",  "Omnigrok MLP · MNIST-1k · AdamW",          "n_train=2000"),
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


def load(csv_path="results/data/runs_v2.csv"):
    rows = [r for r in csv.DictReader(open(csv_path)) if r["group"] == "aggregation"]
    out = {}
    for setup, _, _ in SETUPS:
        cf = [_tfc(r) for r in rows if r["setup"] == setup and r["arm"] == "cent_full"]
        cf = [x for x in cf if np.isfinite(x)]
        if not cf:
            continue
        ks = sorted({int(r["num_clients"]) for r in rows
                     if r["setup"] == setup and r["arm"] == "fl"})
        cells = []
        for k in ks:
            v = [_tfc(r) for r in rows if r["setup"] == setup
                 and r["arm"] == "fl" and int(r["num_clients"]) == k]
            fin = [x for x in v if np.isfinite(x)]
            cells.append({"K": k, "n": len(v), "grokked": len(fin),
                          "median": float(np.median(fin)) if fin else None,
                          "lo": min(fin) if fin else None,
                          "hi": max(fin) if fin else None})
        out[setup] = {"base": float(np.median(cf)), "base_lo": min(cf),
                      "base_hi": max(cf), "base_n": len(cf), "cells": cells}
    return out


def draw(setup, label, wp, d, out_dir):
    base = d["base"]
    fig, ax = plt.subplots(figsize=(6.6, 4.5))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    ok = [c for c in d["cells"] if c["median"] is not None]
    cens = [c for c in d["cells"] if c["median"] is None]
    ratios = [c["median"] / base for c in ok]
    lo_b, hi_b = d["base_lo"] / base, d["base_hi"] / base

    # Extremes must include the ERROR BAR ends, not just the medians, or the
    # bars get clipped by the axis (they did, on A at K=50 and B at K=20).
    highs = [c["hi"] / base for c in ok] + [hi_b, 1.0]
    lows = [c["lo"] / base for c in ok] + [lo_b, 1.0]
    top, bottom = max(highs), min(lows)
    span = (top - bottom) or 1.0
    # Reserve headroom for the censored row so its marker never lands on data
    # or on the subtitle.
    head = 0.26 if cens else 0.09
    partial = any(0 < c["grokked"] < c["n"] for c in ok)
    ymax = top + span * head
    ymin = bottom - span * (0.17 if partial else 0.09)
    y_cens = top + span * 0.13

    band = ax.axhspan(lo_b, hi_b, color=SERIES, alpha=0.10, lw=0, zorder=0)
    ax.axhline(1.0, color=INK3, ls="--", lw=1.1, zorder=1)

    line = None
    if ok:
        yerr = [[r - c["lo"] / base for r, c in zip(ratios, ok)],
                [c["hi"] / base - r for r, c in zip(ratios, ok)]]
        line = ax.errorbar([c["K"] for c in ok], ratios, yerr=yerr, marker="o",
                           color=SERIES, capsize=3, lw=1.8, markersize=6,
                           markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=3)
        # Labels go HORIZONTALLY off the marker: the error bars are vertical, so
        # a sideways offset cannot collide with them or with the line.
        for i, (c, r) in enumerate(zip(ok, ratios)):
            last = i == len(ok) - 1
            ax.annotate(f"{r:.2f}×", (c["K"], r), textcoords="offset points",
                        xytext=(-9 if last else 9, 0),
                        ha="right" if last else "left", va="center",
                        fontsize=8.5, color=INK2, family="monospace", zorder=4)
            # A PARTLY censored cell still plots a marker, so without this the
            # median silently becomes a median over survivors only. Setup C at
            # K=50 is 2/3 and would otherwise read as complete.
            if c["grokked"] < c["n"]:
                ax.annotate(f"{c['grokked']}/{c['n']}", (c["K"], c["lo"] / base),
                            textcoords="offset points", xytext=(0, -11),
                            ha="center", va="top", fontsize=8, color=CENSORED,
                            family="monospace", zorder=4)

    cmark = None
    for c in cens:
        cmark, = ax.plot([c["K"]], [y_cens], marker="o", mfc="none", mec=CENSORED,
                         mew=1.9, markersize=8, ls="none", zorder=3)
        ax.annotate(f"{c['grokked']}/{c['n']}", (c["K"], y_cens),
                    textcoords="offset points", xytext=(0, -13), ha="center",
                    va="top", fontsize=8, color=CENSORED, family="monospace")
    ax.set_ylim(ymin, ymax)

    ks = [c["K"] for c in d["cells"]]
    ax.set_xscale("log")
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.minorticks_off()
    ax.set_xlim(min(ks) * 0.72, max(ks) * 1.38)
    ax.set_xlabel("K  (clients)", fontsize=10, color=INK2)
    ax.set_ylabel("federated / centralized\nfirst crossing", fontsize=10, color=INK2)
    ax.set_title(f"Setup {setup} — {label}", fontsize=12.5, color=INK, pad=18, loc="left")
    ax.annotate(f"{wp} · E=5 · FedAvg · 3 seeds", xy=(0, 1.02),
                xycoords="axes fraction", fontsize=8.5, color=INK3)

    handles = [h for h in (line, band, cmark) if h is not None]
    names = []
    if line is not None:
        names.append("median, bars = FL seed range")
    names.append("centralized seed spread")
    if cmark is not None:
        names.append("censored — never reached bar")
    leg = ax.legend(handles, names, fontsize=8, loc="upper left",
                    frameon=True, framealpha=0.95, borderpad=0.6)
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
    return path, len(ok), len(cens)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="paper")
    ap.add_argument("--csv", default="results/data/runs_v2.csv")
    a = ap.parse_args()
    data = load(a.csv)
    for setup, label, wp in SETUPS:
        if setup not in data:
            print(f"  setup {setup}: no aggregation rows, skipped")
            continue
        path, n_ok, n_cens = draw(setup, label, wp, data[setup], a.out)
        print(f"  {path}   {n_ok} plotted, {n_cens} censored")


if __name__ == "__main__":
    main()
