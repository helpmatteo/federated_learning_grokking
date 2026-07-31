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

# Preferred column order (shared columns first, aligned with runs.csv). This is an
# ORDERING, not a whitelist: any key present in a result row and absent here is
# appended rather than dropped.
#
# It used to be a whitelist, and that silently deleted the setup identity that
# `run.py` goes out of its way to record -- dataset, model, loss, group_n,
# coset_subgroup, and grok_threshold all vanished at collection time. In a
# single-setup study that was invisible. Across setups it is fatal: an S5 run
# whose spec omits task/p resolves to the defaults and lands as
# `task=addition, p=97`, indistinguishable from a mod-97 run, and S5-groknet vs
# S5-transformer differ ONLY in `model`. `summarize_runs.py` then pools them into
# one survival curve without complaint. Likewise a t_grok measured at the 85%
# (S5), 90% (MNIST) and 95% (modular) bars are not comparable quantities, which
# is precisely why `grok_threshold` is recorded per row.
PREFERRED_COLUMNS = [
    "id", "mode", "tier", "group", "experiment", "setting", "algorithm", "setup",
    "arm", "reduced_from_k",
    # setup identity -- what makes a multi-setup table interpretable
    "dataset", "model", "loss", "group_n", "coset_subgroup",
    "task", "optimizer", "p", "hidden_width", "n_layers", "activation",
    "init_scale", "batch_size", "n_train", "n_test", "alpha", "seed",
    "num_clients", "local_epochs", "num_rounds", "fraction_train", "partition",
    "dirichlet_alpha", "proximal_mu", "strategy", "server_lr", "server_momentum",
    "tau", "eval_every", "checkpoint_every", "epochs", "log_every", "momentum",
    "lr", "weight_decay",
    "grok_threshold",
    "t_grok", "t_50", "t_memo", "delay", "final_acc", "final_train_acc",
    "peak_train_acc", "final_ipr",
    "grokked", "censored", "steps_run", "wall_s",
]

# Rows written before run.py recorded setup identity (2026-07-28). Every such run
# predates the multi-setup work and is therefore the anchor setup; --backfill-legacy
# stamps that in rather than leaving the columns blank.
LEGACY_DEFAULTS = {"dataset": "modular", "model": "groknet", "loss": "mse"}


def resolve_columns(rows):
    """Preferred order first, then any other key that appears in the data."""
    seen = {k for row in rows for k in row}
    extra = sorted(seen - set(PREFERRED_COLUMNS))
    return [c for c in PREFERRED_COLUMNS if c in seen] + extra


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", default="results/data/runs")
    parser.add_argument("--out", default="results/data/runs_v2.csv")
    parser.add_argument("--backfill-legacy", action="store_true",
                        help="Stamp dataset/model/loss on pre-schema rows that "
                             "lack them (all such runs are the anchor setup).")
    args = parser.parse_args()

    paths = sorted(glob.glob(os.path.join(args.runs_dir, "*.json")))
    rows = []
    for path in paths:
        with open(path) as handle:
            rows.append(json.load(handle))

    legacy = [r for r in rows if "dataset" not in r]
    if legacy and args.backfill_legacy:
        for row in legacy:
            row.update({k: v for k, v in LEGACY_DEFAULTS.items() if k not in row})
            row["schema"] = "legacy-backfilled"

    columns = resolve_columns(rows)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, restval="",
                                extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    n_grok = sum(1 for r in rows if r.get("grokked"))
    print(f"Collected {len(rows)} runs -> {args.out}")
    print(f"  grokked: {n_grok}   censored: {len(rows) - n_grok}")
    print(f"  columns: {len(columns)}")
    if legacy and not args.backfill_legacy:
        print(f"  NOTE: {len(legacy)} row(s) predate the setup-identity schema and "
              f"have blank dataset/model/loss. Pass --backfill-legacy to stamp "
              f"them as the anchor setup ({LEGACY_DEFAULTS}).")


if __name__ == "__main__":
    main()
