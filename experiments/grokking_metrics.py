"""Grokking step detection and multi-seed result aggregation.

Definitions from experiment_plan.md Section 3.1:
- T_grok: smallest step t_j such that test_acc >= threshold for ALL subsequent steps
- T_50: smallest step t_j such that test_acc >= 50%
"""

import math
from typing import List
import numpy as np


def compute_t_grok(steps: list, test_accs: list, threshold: float = 95.0) -> float:
    """Compute grokking step T_grok.

    Returns the smallest step where test accuracy reaches `threshold`
    and never drops below it for the remainder of training.
    Returns float('inf') if no such step exists.
    """
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


def compute_t_50(steps: list, test_accs: list, threshold: float = 50.0) -> float:
    """Compute onset step T_50 — first step where test accuracy >= 50%."""
    for step, acc in zip(steps, test_accs):
        if acc >= threshold:
            return step
    return float("inf")


def extract_grokking_results(history: dict) -> dict:
    """Extract grokking metrics from a training history dict.

    Works with both centralized (key: 'epoch') and federated (key: 'total_steps').
    """
    steps = history.get("total_steps", history.get("epoch", []))
    test_accs = history.get("test_acc", [])
    train_accs = history.get("train_acc", [])

    t_grok = compute_t_grok(steps, test_accs)
    t_50 = compute_t_50(steps, test_accs)
    final_test_acc = test_accs[-1] if test_accs else 0.0
    final_train_acc = train_accs[-1] if train_accs else 0.0
    final_ipr = history.get("ipr", [0.0])[-1] if history.get("ipr") else 0.0

    return {
        "t_grok": t_grok,
        "t_50": t_50,
        "final_test_acc": final_test_acc,
        "final_train_acc": final_train_acc,
        "final_ipr": final_ipr,
    }


def summarize_seeds(results: List[dict]) -> dict:
    """Aggregate grokking metrics across seeds (mean +/- std).

    If not ALL seeds grokked, t_grok_mean = inf (conservative).
    t_50 is averaged only over seeds that achieved T_50.
    Uses ddof=1 (sample std) for std computation.
    """
    n = len(results)
    t_groks = [r["t_grok"] for r in results]
    t_50s = [r["t_50"] for r in results]
    final_accs = [r["final_test_acc"] for r in results]

    n_grokked = sum(1 for t in t_groks if t < float("inf"))
    finite_groks = [t for t in t_groks if t < float("inf")]
    finite_50s = [t for t in t_50s if t < float("inf")]

    return {
        "n_seeds": n,
        "n_grokked": n_grokked,
        "t_grok_mean": float(np.mean(finite_groks)) if n_grokked == n else float("inf"),
        "t_grok_std": float(np.std(finite_groks, ddof=1)) if n_grokked == n and n > 1 else 0.0 if n_grokked == n else float("inf"),
        "t_50_mean": float(np.mean(finite_50s)) if finite_50s else float("inf"),
        "t_50_std": float(np.std(finite_50s, ddof=1)) if len(finite_50s) > 1 else 0.0 if finite_50s else float("inf"),
        "final_acc_mean": float(np.mean(final_accs)),
        "final_acc_std": float(np.std(final_accs, ddof=1)) if n > 1 else 0.0,
    }
