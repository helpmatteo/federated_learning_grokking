"""Survival statistics for grokking times with right-censoring.

A seed that reaches its step budget without grokking is NOT evidence that
T_grok is infinite — it is a right-censored observation: we know only that
T_grok > budget. Treating it as `inf` (as the old summarize_seeds did) both
discards information and collapses "grokked 4/5, median ~8k" and "grokked 0/5"
to the same value. This module handles censoring properly:

  - fraction_grokked: the discrete "did it grok at all" axis;
  - Kaplan-Meier median: the central T_grok that incorporates censored seeds
    instead of dropping them;
  - bootstrap CIs over both.

All functions take `durations` (grok time for grokked seeds, censoring time =
budget for the rest) and `events` (1 grokked, 0 censored).
"""

import numpy as np


def kaplan_meier(durations, events):
    """Kaplan-Meier survival curve.

    Returns (times, survival) as step-function points: survival[i] = P(T > times[i]).
    """
    durations = np.asarray(durations, dtype=float)
    events = np.asarray(events, dtype=int)
    n = len(durations)
    if n == 0:
        return np.array([]), np.array([])

    times, surv, s = [], [], 1.0
    for t in np.unique(durations):
        at_risk = int(np.sum(durations >= t))            # still at risk just before t
        n_events = int(np.sum((durations == t) & (events == 1)))
        if at_risk > 0:
            s *= 1.0 - n_events / at_risk
        times.append(float(t))
        surv.append(s)
    return np.array(times), np.array(surv)


def km_median(durations, events):
    """KM median grok time — smallest t with survival <= 0.5.

    Returns inf when the survival curve never reaches 0.5 (a majority of seeds
    censored), which is the honest "most did not grok within budget".
    """
    times, surv = kaplan_meier(durations, events)
    for t, s in zip(times, surv):
        if s <= 0.5:
            return float(t)
    return float("inf")


def fraction_grokked(events) -> float:
    events = np.asarray(events, dtype=int)
    return float(events.mean()) if len(events) else 0.0


def bootstrap_ci(durations, events, statistic, n_boot=2000, alpha=0.05, seed=0):
    """Percentile bootstrap CI for a survival statistic over seeds.

    Resamples the (duration, event) pairs with replacement. With few seeds the
    interval is wide and discrete — that honesty is the point. Returns
    (low, high); either may be inf if the statistic is inf on a resample.
    """
    durations = np.asarray(durations, dtype=float)
    events = np.asarray(events, dtype=int)
    n = len(durations)
    if n == 0:
        return (float("nan"), float("nan"))

    rng = np.random.RandomState(seed)
    stats = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        stats.append(statistic(durations[idx], events[idx]))
    stats = np.array(stats, dtype=float)

    finite = stats[np.isfinite(stats)]
    frac_inf = float(np.mean(~np.isfinite(stats)))
    lo = float(np.quantile(finite, alpha / 2)) if len(finite) else float("inf")
    # If more than alpha/2 of resamples are inf, the upper tail is inf.
    hi = float("inf") if frac_inf > alpha / 2 else (
        float(np.quantile(finite, 1 - alpha / 2)) if len(finite) else float("inf"))
    return (lo, hi)


def summarize_survival(durations, events, n_boot=2000, seed=0) -> dict:
    """Full censored summary of grok times across seeds."""
    events = np.asarray(events, dtype=int)
    med_lo, med_hi = bootstrap_ci(durations, events, km_median,
                                  n_boot=n_boot, seed=seed)
    return {
        "n_seeds": int(len(events)),
        "n_grokked": int(events.sum()),
        "fraction_grokked": fraction_grokked(events),
        "t_grok_km_median": km_median(durations, events),
        "t_grok_ci_low": med_lo,
        "t_grok_ci_high": med_hi,
    }
