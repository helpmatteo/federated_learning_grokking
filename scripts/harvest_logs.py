"""Harvest run results from logs/*.log into a single tidy CSV.

The per-run history JSONs under results/ are gitignored and no longer present on
disk, so these logs are the only surviving record of the ~276 runs behind the
committed figures.  This script reconstructs one row per (run, config, seed).

IMPORTANT — why the log filename is authoritative for the algorithm:
    fedgrok/training/federated.py builds its history filename from the config,
    but the
    `_adam_tau*_slr*` and `_wd*` suffixes were only added partway through the
    exp5 campaign (commit 3796754, 2026-03-23).  Runs from before that commit
    wrote *identical* paths for FedAvg, FedAdam and FedAvg+WD within the same
    (setting, seed) cell, so the history path does not uniquely identify a run.
    The log filename (e.g. exp5_H3_FedAdam-0.1.log) does, and is used instead.

Usage:
    python scripts/harvest_logs.py --logs logs --out results/data/runs.csv
"""

import argparse
import csv
import os
import re
import glob


# ── Line patterns ───────────────────────────────────────────────────────────

HISTORY_RE = re.compile(r"History saved to (?P<path>\S+\.json)")
# T_50 is optional: the exp7 task runs (experiments/exp_task_generality.py) emit a
# shorter summary line without it.
RESULT_RE = re.compile(
    r"->\s*T_grok=(?P<t_grok>inf|[\d.]+),\s*"
    r"(?:T_50=(?P<t_50>inf|[\d.]+),\s*)?"
    r"final_acc=(?P<final_acc>[\d.]+)%"
)

# history_fed_<task>_<opt>_p<p>_N<N>_a<alpha>_K<K>_le<E>_ft<frac>_<partition><extras>_s<seed>.json
FED_RE = re.compile(
    r"history_fed_(?P<task>.+?)_(?P<optimizer>gd|adamw)"
    r"_p(?P<p>\d+)_N(?P<hidden_width>\d+)_a(?P<alpha>[\d.]+)"
    r"_K(?P<num_clients>\d+)_le(?P<local_epochs>\d+)_ft(?P<fraction_train>[\d.]+)"
    r"_(?P<partition>iid|operand|target|dirichlet)(?P<extras>.*?)"
    r"_s(?P<seed>\d+)\.json$"
)

# history_<task>_<opt>_p<p>_N<N>_a<alpha>_s<seed>.json
CENT_RE = re.compile(
    r"history_(?P<task>.+?)_(?P<optimizer>gd|adamw)"
    r"_p(?P<p>\d+)_N(?P<hidden_width>\d+)_a(?P<alpha>[\d.]+)"
    r"_s(?P<seed>\d+)\.json$"
)

# Optional suffixes inside <extras>
EXTRA_RES = {
    "dirichlet_alpha": re.compile(r"_dir(?P<v>[\d.]+)"),
    "proximal_mu": re.compile(r"_mu(?P<v>[\d.]+)"),
    "tau": re.compile(r"_adam_tau(?P<v>[\d.e-]+)"),
    "server_lr": re.compile(r"_slr(?P<v>[\d.]+)"),
    "weight_decay": re.compile(r"_wd(?P<v>[\d.]+)"),
}

# exp5[_rerun]_<setting>_<algorithm>.log  -> setting + algorithm arm
EXP5_RE = re.compile(r"^exp5(?:_rerun)?_(?P<setting>H\d+)_(?P<algorithm>.+)$")
EXP_RE = re.compile(r"^(?P<experiment>exp\d+[ab]?)")


def _num(value, cast=float):
    """Parse a numeric field, mapping 'inf' to float('inf')."""
    if value is None or value == "":
        return None
    if value == "inf":
        return float("inf")
    return cast(value)


def parse_log_name(stem: str) -> dict:
    """Extract experiment / setting / algorithm from a log filename stem."""
    meta = {"experiment": None, "setting": None, "algorithm": None,
            "is_rerun": stem.startswith("exp5_rerun")}

    exp_match = EXP_RE.match(stem)
    if exp_match:
        meta["experiment"] = exp_match.group("experiment")

    exp5_match = EXP5_RE.match(stem)
    if exp5_match:
        meta["setting"] = exp5_match.group("setting")
        meta["algorithm"] = exp5_match.group("algorithm")

    return meta


def parse_history_path(path: str) -> dict:
    """Extract the run config from a history JSON path."""
    basename = os.path.basename(path)

    fed_match = FED_RE.search(basename)
    if fed_match:
        row = fed_match.groupdict()
        extras = row.pop("extras", "") or ""
        row["mode"] = "federated"
        for field, pattern in EXTRA_RES.items():
            match = pattern.search(extras)
            row[field] = _num(match.group("v")) if match else None
        return row

    cent_match = CENT_RE.search(basename)
    if cent_match:
        row = cent_match.groupdict()
        row["mode"] = "centralized"
        for field in ("num_clients", "local_epochs", "fraction_train",
                      "partition", "dirichlet_alpha", "proximal_mu",
                      "tau", "server_lr", "weight_decay"):
            row[field] = None
        return row

    return {}


def harvest_file(path: str) -> list:
    """Parse one log file into a list of run rows."""
    stem = os.path.splitext(os.path.basename(path))[0]
    log_meta = parse_log_name(stem)

    rows = []
    pending = None  # history path awaiting its result line

    with open(path, "r", errors="replace") as handle:
        for line in handle:
            history_match = HISTORY_RE.search(line)
            if history_match:
                pending = history_match.group("path")
                continue

            result_match = RESULT_RE.search(line)
            if result_match and pending is not None:
                config = parse_history_path(pending)
                if not config:
                    pending = None
                    continue

                t_grok = _num(result_match.group("t_grok"))
                row = {
                    "log_file": os.path.basename(path),
                    "experiment": log_meta["experiment"],
                    "setting": log_meta["setting"],
                    # Log filename is authoritative — see module docstring.
                    "algorithm": log_meta["algorithm"],
                    "is_rerun": log_meta["is_rerun"],
                    "history_path": pending,
                    "t_grok": t_grok,
                    "t_50": _num(result_match.group("t_50")),
                    "final_acc": _num(result_match.group("final_acc")),
                    "grokked": t_grok != float("inf"),
                    "censored": t_grok == float("inf"),
                }
                row.update(config)
                rows.append(row)
                pending = None

    return rows


FIELDS = [
    "log_file", "experiment", "setting", "algorithm", "is_rerun",
    "mode", "task", "optimizer", "p", "hidden_width", "alpha", "seed",
    "num_clients", "local_epochs", "fraction_train", "partition",
    "dirichlet_alpha", "proximal_mu", "server_lr", "tau", "weight_decay",
    "t_grok", "t_50", "final_acc", "grokked", "censored", "history_path",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs", default="logs", help="directory of .log files")
    parser.add_argument("--out", default="results/data/runs.csv", help="output CSV")
    args = parser.parse_args()

    log_paths = sorted(glob.glob(os.path.join(args.logs, "*.log")))
    if not log_paths:
        raise SystemExit(f"No .log files found in {args.logs}")

    all_rows = []
    skipped = []
    for path in log_paths:
        rows = harvest_file(path)
        if rows:
            all_rows.extend(rows)
        else:
            size = os.path.getsize(path)
            reason = "empty log" if size == 0 else "no run summary (crashed/incomplete)"
            skipped.append((os.path.basename(path), size, reason))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    n_grokked = sum(1 for r in all_rows if r["grokked"])
    n_files_with_rows = len({r["log_file"] for r in all_rows})
    print(f"Scanned {len(log_paths)} log files ({n_files_with_rows} yielded runs)")
    print(f"Harvested {len(all_rows)} runs -> {args.out}")
    print(f"  grokked: {n_grokked}   censored: {len(all_rows) - n_grokked}")

    # Logs that produced nothing are holes in the experiment grid, not noise.
    # Record them so missing cells stay visible.
    if skipped:
        skipped_path = os.path.splitext(args.out)[0] + "_skipped.csv"
        with open(skipped_path, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["log_file", "bytes", "reason"])
            writer.writerows(skipped)
        print(f"\n  {len(skipped)} logs yielded no runs -> {skipped_path}")
        for name, size, reason in skipped:
            print(f"    {name}  ({size} bytes, {reason})")

    # Collisions: distinct runs that wrote the same history path.
    by_path = {}
    for row in all_rows:
        by_path.setdefault(row["history_path"], set()).add(row["log_file"])
    collisions = {k: v for k, v in by_path.items() if len(v) > 1}
    if collisions:
        print(f"\n  WARNING: {len(collisions)} history paths were written by >1 "
              f"log file (pre-3796754 filename collisions).")
        print("  Any surviving JSONs at those paths hold only the last run written.")


if __name__ == "__main__":
    main()
