"""Tests for the SCAFFOLD strategy (vendored/adapted from Flower's niid_bench).

The load-bearing correctness anchor is the round-1 reduction: at c = c_i = 0 the
gradient correction (c - c_i) is exactly zero, so a SCAFFOLD round must be
identical to a FedAvg round. If that holds, the correction timing, sign, and
control-variate plumbing are right; the E>1 behaviour is what the experiments
measure.
"""

import logging
import warnings

import numpy as np
import pytest

warnings.filterwarnings("ignore")
logging.getLogger("flwr").setLevel(logging.ERROR)

from fedgrok.core.fed_config import FedConfig
from fedgrok.training.federated import fed_train, _dataset_cache, _client_cache
from fedgrok.training import scaffold as sc


def _run(strategy, rounds, tmp_path, **kw):
    _dataset_cache.clear()
    _client_cache.clear()
    sc.reset_client_cv()
    cfg = FedConfig(
        task="addition", p=17, alpha=0.5, seed=42, hidden_width=32,
        num_clients=4, num_rounds=rounds, local_epochs=1, lr=1.0,
        partition="dirichlet", dirichlet_alpha=0.1, strategy=strategy,
        eval_every=1, output_dir=str(tmp_path), **kw,
    )
    history, _ = fed_train(cfg)
    return history


class TestScaffoldCorrectness:
    def test_round1_matches_fedavg(self, tmp_path):
        """At c=c_i=0 the correction vanishes -> SCAFFOLD round 1 == FedAvg."""
        avg = _run("fedavg", 2, tmp_path / "a")
        scf = _run("scaffold", 2, tmp_path / "s")
        assert scf["test_acc"][1] == pytest.approx(avg["test_acc"][1], abs=1e-4)
        assert scf["train_loss"][1] == pytest.approx(avg["train_loss"][1], abs=1e-6)

    def test_runs_finite_over_many_rounds(self, tmp_path):
        h = _run("scaffold", 15, tmp_path)
        assert all(x == x for x in h["train_loss"])          # no NaN
        assert all(np.isfinite(x) for x in h["test_acc"])

    def test_diverges_from_fedavg_after_round1(self, tmp_path):
        """The control variate must actually bite: identical at round 1, then
        different once c/c_i are non-zero under heterogeneity. (Clients run in
        Ray actor processes, so the control variates aren't inspectable from
        here — this checks the observable consequence instead.)"""
        avg = _run("fedavg", 8, tmp_path / "a")
        scf = _run("scaffold", 8, tmp_path / "s")
        assert scf["test_acc"][1] == pytest.approx(avg["test_acc"][1], abs=1e-4)
        # by the last round the corrected trajectory has departed from FedAvg
        assert abs(scf["train_loss"][-1] - avg["train_loss"][-1]) > 1e-6


class TestScaffoldHelpers:
    def test_cv_bytes_roundtrip(self):
        cv = [np.random.randn(3, 4).astype(np.float32),
              np.random.randn(5).astype(np.float32)]
        shapes = [a.shape for a in cv]
        back = sc._cv_from_bytes(sc._cv_bytes(cv), shapes)
        for a, b in zip(cv, back):
            assert np.allclose(a, b)

    def test_cv_update_reduces_to_gradient_at_e1(self):
        """At E=1 full-batch with c=c_i=0: c_i^+ = (x - y)/(lr*1) = grad."""
        x = [np.ones((2, 2), dtype=np.float32)]
        grad = np.array([[0.1, -0.2], [0.3, 0.4]], dtype=np.float32)
        lr = 0.5
        y = [x[0] - lr * grad]                     # one GD step
        c = [np.zeros((2, 2), dtype=np.float32)]
        ci = [np.zeros((2, 2), dtype=np.float32)]
        new_cv, delta = sc.client_cv_update(x, y, c, ci, lr=lr, n_steps=1)
        assert np.allclose(new_cv[0], grad, atol=1e-6)
        assert np.allclose(delta[0], grad, atol=1e-6)
