"""Censored-survival summary table from a runs CSV.

Groups runs into config cells (everything except seed), and reports, per cell,
the honest survival summary: n seeds, n grokked, fraction grokked, and the
Kaplan-Meier median T_grok with a bootstrap CI — right-censoring the seeds that
did not grok within budget, instead of the old inf-if-any-fail estimator.

Works on both results/data/runs.csv (the archived-log harvest) and
results/data/runs_v2.csv (the v2 collector output).

    python scripts/summarize_runs.py results/data/runs.csv
    python scripts/summarize_runs.py results/data/runs_v2.csv --group experiment,setting,algorithm
"""

import argparse
import csv
import collections
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fedgrok.analysis.survival import summarize_survival

# Columns that identify a config cell (a run differs from its cell-mates only by
# seed). Kept broad; missing columns in a given CSV are simply ignored.
# `num_rounds` and `grok_threshold` are cell keys, not incidental config: two
# runs at different budgets have different censoring times, and two runs judged
# at different accuracy bars are not measuring the same event. Pooling either
# silently mixes incomparable observations into one survival curve.
DEFAULT_CELL_KEYS = [
    "experiment", "setting", "algorithm", "mode", "dataset", "model", "loss",
    "task", "p", "alpha", "num_clients", "local_epochs", "num_rounds",
    "fraction_train", "partition", "dirichlet_alpha", "strategy",
    "weight_decay", "grok_threshold",
]


def _to_float(x):
    if x is None or x == "":
        return None
    if x == "inf":
        return float("inf")
    try:
        return float(x)
    except ValueError:
        return None


def summarize_csv(path, cell_keys, budget=None):
    rows = list(csv.DictReader(open(path)))
    present = [k for k in cell_keys if rows and k in rows[0]]

    cells = collections.defaultdict(list)
    for r in rows:
        key = tuple(r.get(k, "") for k in present)
        cells[key].append(r)

    out = []
    for key, cell_rows in sorted(cells.items()):
        durations, events = [], []
        for r in cell_rows:
            t = _to_float(r.get("t_grok"))
            if t is None:
                continue
            if t < float("inf"):
                durations.append(t)
                events.append(1)
            else:
                # censoring time: steps_run if present, else the CLI budget,
                # else the largest finite grok time in this cell.
                cens = _to_float(r.get("steps_run")) or budget
                durations.append(cens if cens else None)
                events.append(0)
        # fill any missing censor times with the cell's max finite grok time
        finite = [d for d, e in zip(durations, events) if e == 1]
        fallback = max(finite) if finite else float("inf")
        durations = [d if d is not None else fallback for d in durations]

        if durations:
            summary = summarize_survival(durations, events, n_boot=1000)
            summary["cell"] = dict(zip(present, key))
            out.append(summary)
    return present, out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", help="runs CSV (runs.csv or runs_v2.csv)")
    parser.add_argument("--group", default=None,
                        help="comma-separated cell keys (default: a broad set)")
    parser.add_argument("--budget", type=float, default=None,
                        help="censoring time for non-grokked seeds lacking steps_run")
    parser.add_argument("--out", default=None, help="write the table to a CSV")
    args = parser.parse_args()

    cell_keys = args.group.split(",") if args.group else DEFAULT_CELL_KEYS
    present, summaries = summarize_csv(args.csv, cell_keys, budget=args.budget)

    def fmt(x):
        return "inf" if x == float("inf") else (f"{x:.0f}" if x == x else "-")

    label_w = max((len(" ".join(str(v) for v in s["cell"].values()))
                   for s in summaries), default=10)
    print(f"{'cell':{label_w}s}  {'n':>2s} {'grok':>4s} {'frac':>5s} "
          f"{'KM median':>10s} {'CI low':>8s} {'CI high':>8s}")
    for s in summaries:
        label = " ".join(str(v) for v in s["cell"].values())
        print(f"{label:{label_w}s}  {s['n_seeds']:2d} {s['n_grokked']:4d} "
              f"{s['fraction_grokked']:5.2f} {fmt(s['t_grok_km_median']):>10s} "
              f"{fmt(s['t_grok_ci_low']):>8s} {fmt(s['t_grok_ci_high']):>8s}")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", newline="") as handle:
            cols = present + ["n_seeds", "n_grokked", "fraction_grokked",
                              "t_grok_km_median", "t_grok_ci_low", "t_grok_ci_high"]
            writer = csv.DictWriter(handle, fieldnames=cols, extrasaction="ignore")
            writer.writeheader()
            for s in summaries:
                writer.writerow({**s["cell"], **{k: s[k] for k in cols if k in s}})
        print(f"\nWrote {len(summaries)} cells -> {args.out}")


if __name__ == "__main__":
    main()
