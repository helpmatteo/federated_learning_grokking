import pytest
from experiments.grokking_metrics import compute_t_grok, compute_t_50, summarize_seeds


class TestTGrok:
    def test_clear_grokking(self):
        steps = list(range(0, 1000, 100))
        accs = [1.0, 1.0, 1.0, 10.0, 50.0, 96.0, 97.0, 98.0, 99.0, 99.5]
        assert compute_t_grok(steps, accs, threshold=95.0) == 500

    def test_no_grokking(self):
        steps = list(range(0, 1000, 100))
        accs = [1.0] * 10
        assert compute_t_grok(steps, accs, threshold=95.0) == float("inf")

    def test_transient_spike_rejected(self):
        steps = list(range(0, 800, 100))
        accs = [1.0, 1.0, 96.0, 80.0, 96.0, 97.0, 98.0, 99.0]
        assert compute_t_grok(steps, accs, threshold=95.0) == 400

    def test_single_point_at_end(self):
        steps = [0, 100, 200]
        accs = [1.0, 50.0, 96.0]
        assert compute_t_grok(steps, accs, threshold=95.0) == 200

    def test_empty_input(self):
        assert compute_t_grok([], [], threshold=95.0) == float("inf")


class TestT50:
    def test_onset_detected(self):
        steps = [0, 100, 200, 300]
        accs = [1.0, 30.0, 55.0, 90.0]
        assert compute_t_50(steps, accs) == 200

    def test_no_onset(self):
        steps = [0, 100, 200]
        accs = [1.0, 1.0, 1.0]
        assert compute_t_50(steps, accs) == float("inf")


class TestSummarizeSeeds:
    def test_aggregation(self):
        results = [
            {"t_grok": 500, "t_50": 300, "final_test_acc": 99.0},
            {"t_grok": 600, "t_50": 350, "final_test_acc": 98.5},
            {"t_grok": 550, "t_50": 320, "final_test_acc": 99.2},
        ]
        summary = summarize_seeds(results)
        assert summary["t_grok_mean"] == pytest.approx(550.0)
        assert summary["t_grok_std"] == pytest.approx(50.0, abs=0.1)
        assert summary["n_grokked"] == 3
        assert summary["n_seeds"] == 3

    def test_partial_grokking(self):
        results = [
            {"t_grok": 500, "t_50": 300, "final_test_acc": 99.0},
            {"t_grok": float("inf"), "t_50": float("inf"), "final_test_acc": 1.0},
        ]
        summary = summarize_seeds(results)
        assert summary["t_grok_mean"] == float("inf")
        assert summary["n_grokked"] == 1
        assert summary["n_seeds"] == 2
