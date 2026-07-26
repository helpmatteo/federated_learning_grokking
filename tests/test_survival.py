"""Tests for the censored survival statistics."""

import numpy as np
import pytest

from fedgrok.analysis.survival import (
    kaplan_meier, km_median, fraction_grokked, bootstrap_ci, summarize_survival,
)


class TestKaplanMeier:
    def test_no_censoring_median_is_the_middle_event(self):
        # three uncensored grok times: KM median is where S(t) first <= 0.5
        durations = [500, 550, 600]
        events = [1, 1, 1]
        assert km_median(durations, events) == 550.0

    def test_survival_starts_high_and_decreases(self):
        times, surv = kaplan_meier([100, 200, 300], [1, 1, 1])
        assert list(times) == [100.0, 200.0, 300.0]
        assert surv[0] > surv[1] > surv[2]
        assert surv[-1] == pytest.approx(0.0)

    def test_censored_observations_extend_survival(self):
        """A censored seed keeps mass on 'not yet grokked' rather than dropping it."""
        # one grok at 500, one censored at 50000
        med = km_median([500, 50000], [1, 0])
        assert med == 500.0                        # half grokked -> median at the grok

    def test_all_censored_median_is_inf(self):
        """If nothing grokked within budget, the median is honestly inf."""
        assert km_median([50000, 50000, 50000], [0, 0, 0]) == float("inf")

    def test_majority_censored_median_is_inf(self):
        # 1 of 3 grokked: survival never reaches 0.5
        assert km_median([500, 50000, 50000], [1, 0, 0]) == float("inf")


class TestFractionGrokked:
    def test_fraction(self):
        assert fraction_grokked([1, 1, 0, 0]) == 0.5
        assert fraction_grokked([1, 1, 1]) == 1.0
        assert fraction_grokked([0, 0]) == 0.0


class TestBootstrap:
    def test_ci_brackets_the_point_estimate_when_all_grok(self):
        durations = [500, 520, 540, 560, 580]
        events = [1] * 5
        point = km_median(durations, events)
        lo, hi = bootstrap_ci(durations, events, km_median, n_boot=1000, seed=0)
        assert lo <= point <= hi

    def test_ci_upper_is_inf_when_grokking_is_marginal(self):
        """With half censored, some resamples are majority-censored -> hi = inf."""
        durations = [500, 550, 50000, 50000]
        events = [1, 1, 0, 0]
        _lo, hi = bootstrap_ci(durations, events, km_median, n_boot=1000, seed=0)
        assert hi == float("inf")


class TestSummarySurvival:
    def test_keys_and_values(self):
        s = summarize_survival([500, 600, 550], [1, 1, 1], n_boot=500)
        assert s["n_seeds"] == 3 and s["n_grokked"] == 3
        assert s["fraction_grokked"] == 1.0
        assert np.isfinite(s["t_grok_km_median"])
