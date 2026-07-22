"""Collect per-run result JSONs into one tidy CSV.

The launcher writes results/data/runs/<id>.json, one per completed run
(schema-compatible with the archived-log harvest in results/data/runs.csv).
This assembles them into results/data/runs_v2.csv, the committed evidentiary
base for the v2 experiments.

    python scripts/collect_runs.py
    python scripts/collect_runs.py --runs-dir results/data/runs --out results/data/runs_v2.csv
"""

import argparse
import csv
import glob
import json
import os

# Column order: shared columns first (aligned with runs.csv), then v2-only.
COLUMNS = [
    "id", "mode", "tier", "group", "experiment", "setting", "algorithm",
    "task", "optimizer", "p", "hidden_width", "alpha", "seed",
    "num_clients", "local_epochs", "num_rounds", "fraction_train", "partition",
    "dirichlet_alpha", "proximal_mu", "strategy", "server_lr", "tau", "eval_every",
    "lr", "weight_decay",
    "t_grok", "t_50", "final_acc", "final_train_acc", "final_ipr",
    "grokked", "censored", "steps_run", "wall_s",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", default="results/data/runs")
    parser.add_argument("--out", default="results/data/runs_v2.csv")
    args = parser.parse_args()

    paths = sorted(glob.glob(os.path.join(args.runs_dir, "*.json")))
    rows = []
    for path in paths:
        with open(path) as handle:
            rows.append(json.load(handle))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    n_grok = sum(1 for r in rows if r.get("grokked"))
    print(f"Collected {len(rows)} runs -> {args.out}")
    print(f"  grokked: {n_grok}   censored: {len(rows) - n_grok}")


if __name__ == "__main__":
    main()
