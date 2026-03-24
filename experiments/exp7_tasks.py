"""Experiment 7: Generalization across modular arithmetic tasks.

Tests whether FL grokking findings transfer beyond (a+b) mod 97.

Two tiers:
  Tier 1 (original):
    Tasks:  addition, multiplication, division, x2_plus_y2
    Primes: 53, 97   Alpha: 0.25, 0.3   Budget: 50k steps

  Tier 2 (easier — more data, bigger budget, larger prime):
    Tasks:  addition, multiplication, division, x2_plus_y2
    Primes: 53, 97, 113   Alpha: 0.4, 0.5   Budget: 100k steps
    Gives failing tasks more chance to grok.

Conditions: centralized + FL (FedAvg K=10 E=5 IID), seed=42.
Existing runs from exp1/exp2/tier1 are skipped.

Usage:
    # Single cell (for parallel launcher)
    python -m experiments.exp7_tasks --task multiplication --p 53 --alpha 0.3 --mode cent --budget 50000
    python -m experiments.exp7_tasks --task multiplication --p 113 --alpha 0.5 --mode fl --budget 100000

    # Full sweep (sequential)
    python -m experiments.exp7_tasks --all
"""

import argparse
import glob
import os

from core.config import Config
from federated.config import FedConfig
from experiments.runner import (
    run_single_centralized,
    run_single_federated,
    save_experiment_results,
)
from experiments.grokking_metrics import extract_grokking_results

# ── Sweep axes ────────────────────────────────────────────────────────────

TASKS = ["addition", "multiplication", "division", "x2_plus_y2"]
SEED = 42

# FL settings (matches exp2 defaults)
K = 10
LOCAL_EPOCHS = 5

OUTPUT_DIR = "results/exp7_tasks"

# Tier 1: original sweep (tight budget, near phase boundary)
TIER1_PRIMES = [53, 97]
TIER1_ALPHAS = [0.25, 0.3]
TIER1_BUDGET = 50_000

# Tier 2: easier settings (more data, longer budget, larger prime)
TIER2_PRIMES = [53, 97, 113]
TIER2_ALPHAS = [0.4, 0.5]
TIER2_BUDGET = 100_000


# ── Skip logic ────────────────────────────────────────────────────────────

def _cent_history_exists(task, p, alpha, seed):
    """Check if a centralized run already exists in exp1 or exp7."""
    exp1_pat = f"results/exp1_boundary/history_{task}_gd_p{p}_N256_a{alpha}_s{seed}.json"
    exp7_pat = f"{OUTPUT_DIR}/centralized/history_{task}_gd_p{p}_N256_a{alpha}_s{seed}.json"
    return glob.glob(exp1_pat) or glob.glob(exp7_pat)


def _fl_history_exists(task, p, alpha, seed):
    """Check if an FL run already exists in exp2 or exp7."""
    tag_frag = f"fed_{task}_gd_p{p}_N256_a{alpha}_K{K}_le{LOCAL_EPOCHS}_ft1.0_iid_s{seed}"
    exp2_pat = f"results/exp2_aggregation/fl_iid/history_{tag_frag}.json"
    exp7_pat = f"{OUTPUT_DIR}/fl_iid/history_{tag_frag}.json"
    return glob.glob(exp2_pat) or glob.glob(exp7_pat)


# ── Single-cell runners ──────────────────────────────────────────────────

def run_cent_cell(task, p, alpha, budget=TIER1_BUDGET):
    """Run a single centralized cell."""
    cfg = Config(
        task=task, p=p, optimizer="gd", lr=50.0,
        weight_decay=0.0, momentum=0.0, hidden_width=256,
        epochs=budget, log_every=100, alpha=alpha, seed=SEED,
        output_dir=f"{OUTPUT_DIR}/centralized",
    )
    metrics = run_single_centralized(cfg, label=f"cent {task} p={p} a={alpha}")
    print(f"  -> T_grok={metrics['t_grok']}, final_acc={metrics['final_test_acc']:.1f}%")
    return metrics


def run_fl_cell(task, p, alpha, budget=TIER1_BUDGET):
    """Run a single FL cell."""
    num_rounds = budget // LOCAL_EPOCHS
    fed_cfg = FedConfig(
        task=task, p=p, optimizer="gd", lr=50.0,
        weight_decay=0.0, momentum=0.0, hidden_width=256,
        alpha=alpha, num_clients=K, num_rounds=num_rounds,
        local_epochs=LOCAL_EPOCHS, fraction_train=1.0,
        partition="iid", strategy="fedavg", seed=SEED,
        output_dir=f"{OUTPUT_DIR}/fl_iid",
    )
    metrics = run_single_federated(fed_cfg, label=f"fl {task} p={p} a={alpha}")
    print(f"  -> T_grok={metrics['t_grok']}, final_acc={metrics['final_test_acc']:.1f}%")
    return metrics


# ── Full sweep ────────────────────────────────────────────────────────────

def _run_tier(tier_name, primes, alphas, budget):
    """Run one tier of the sweep, skipping existing runs."""
    results = []
    skipped = 0
    ran = 0

    for task in TASKS:
        for p in primes:
            for alpha in alphas:
                cell = {"task": task, "p": p, "alpha": alpha, "seed": SEED,
                        "tier": tier_name, "budget": budget}

                if _cent_history_exists(task, p, alpha, SEED):
                    print(f"SKIP centralized {task} p={p} α={alpha} (exists)")
                    skipped += 1
                else:
                    print(f"\nRUN  centralized {task} p={p} α={alpha}")
                    cell["cent"] = run_cent_cell(task, p, alpha, budget)
                    ran += 1

                if _fl_history_exists(task, p, alpha, SEED):
                    print(f"SKIP FL         {task} p={p} α={alpha} (exists)")
                    skipped += 1
                else:
                    print(f"\nRUN  FL         {task} p={p} α={alpha}")
                    cell["fl"] = run_fl_cell(task, p, alpha, budget)
                    ran += 1

                results.append(cell)

    return results, ran, skipped


def run_exp7():
    """Run the full task generalization sweep (both tiers), skipping existing runs."""
    print("=" * 60)
    print("  TIER 1: α ∈ {0.25, 0.3}, p ∈ {53, 97}, budget=50k")
    print("=" * 60)
    r1, ran1, skip1 = _run_tier("tier1", TIER1_PRIMES, TIER1_ALPHAS, TIER1_BUDGET)

    print("\n" + "=" * 60)
    print("  TIER 2: α ∈ {0.4, 0.5}, p ∈ {53, 97, 113}, budget=100k")
    print("=" * 60)
    r2, ran2, skip2 = _run_tier("tier2", TIER2_PRIMES, TIER2_ALPHAS, TIER2_BUDGET)

    results = r1 + r2
    ran = ran1 + ran2
    skipped = skip1 + skip2

    print(f"\n{'='*60}")
    print(f"EXP 7 COMPLETE: {ran} runs executed, {skipped} skipped")
    print(f"{'='*60}")

    save_experiment_results(results, f"{OUTPUT_DIR}/exp7_results.json")
    return results


# ── CLI ───────────────────────────────────────────────────────────────────

ALL_PRIMES = sorted(set(TIER1_PRIMES + TIER2_PRIMES))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp 7: Task generalization")
    parser.add_argument("--all", action="store_true", help="Run full sweep sequentially")
    parser.add_argument("--task", type=str, choices=TASKS)
    parser.add_argument("--p", type=int, choices=ALL_PRIMES)
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--mode", type=str, choices=["cent", "fl"])
    parser.add_argument("--budget", type=int, default=TIER1_BUDGET,
                        help="Max gradient steps (default: 50000)")
    args = parser.parse_args()

    if args.all:
        run_exp7()
    elif args.task and args.p and args.alpha and args.mode:
        if args.mode == "cent":
            run_cent_cell(args.task, args.p, args.alpha, args.budget)
        else:
            run_fl_cell(args.task, args.p, args.alpha, args.budget)
    else:
        parser.error("Provide --all or (--task, --p, --alpha, --mode)")
