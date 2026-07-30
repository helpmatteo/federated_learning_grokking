"""Static PNG twin of grok_curves.py, for eyeballing the data before publishing.

The HTML page is the deliverable; this exists so the curves can be *looked at*
in an environment with no browser. Same data, same two series, same palette
slots, so anything wrong here is wrong there.

    python scripts/plotting/grok_curves_png.py --runs <id> ... --out curves.png
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from grok_curves import SETUP_NAMES, infer_setup, load_run, series_for

TRAIN, TEST = "#2a78d6", "#eb6834"          # validated categorical slots 1, 2


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--runs-root", default="results/data/runs")
    ap.add_argument("--hist-root", default="results/runs")
    args = ap.parse_args()

    panels = []
    for run_id in args.runs:
        try:
            row, history = load_run(run_id, args.runs_root, args.hist_root)
        except FileNotFoundError:
            print(f"  skip {run_id} (not finished)")
            continue
        xs, ys = series_for(history)
        if xs:
            panels.append((infer_setup(row), row, xs, ys))
    panels.sort(key=lambda t: (t[1]["mode"] != "centralized", t[0]))

    n = len(panels)
    cols = max(1, min(5, (n + 1) // 2))
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.5 * cols, 2.9 * rows),
                             squeeze=False)
    for ax in axes.flat:
        ax.set_visible(False)

    for i, (setup, row, xs, ys) in enumerate(panels):
        ax = axes[i // cols][i % cols]
        ax.set_visible(True)
        ax.plot(xs, ys.get("train_acc", []), color=TRAIN, lw=1.8, label="Train")
        ax.plot(xs, ys.get("test_acc", []), color=TEST, lw=1.8, label="Test")

        bar = row.get("grok_threshold")
        if bar:
            ax.axhline(bar, color="#82817b", lw=0.9, alpha=0.6)
        tg = row.get("t_grok")
        if isinstance(tg, (int, float)) and tg != float("inf"):
            ax.axvline(tg, color="#82817b", lw=0.9, alpha=0.75)
            ax.text(tg, 6, f" T={int(tg):,}", fontsize=7, color="#52514e")

        ax.set_xscale("log")
        ax.set_ylim(-3, 103)
        ax.set_title(f"{setup} · {SETUP_NAMES.get(setup,'?')}\n"
                     f"{row['mode']}  ·  final {row['final_train_acc']:.0f}/"
                     f"{row['final_acc']:.0f}%", fontsize=8.5)
        ax.tick_params(labelsize=7)
        ax.grid(True, which="major", axis="y", lw=0.5, color="#e8e7e2")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        if i == 0:
            ax.legend(fontsize=7, frameon=False, loc="upper left")

    fig.suptitle("Train (blue) and test (orange) accuracy vs gradient steps",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=125)
    print(f"Wrote {n} panels -> {args.out}")


if __name__ == "__main__":
    main()
