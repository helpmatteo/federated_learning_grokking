import pytest
import numpy as np
from federated.train import compute_drift


class TestComputeDrift:
    def test_zero_drift_for_identical_weights(self):
        w_before = [np.zeros((3, 4)), np.zeros((2, 3))]
        w_after = [np.zeros((3, 4)), np.zeros((2, 3))]
        assert compute_drift(w_before, w_after) == pytest.approx(0.0)

    def test_nonzero_drift(self):
        w_before = [np.zeros((2, 2))]
        w_after = [np.ones((2, 2))]
        assert compute_drift(w_before, w_after) == pytest.approx(2.0)

    def test_drift_is_frobenius_norm(self):
        w_before = [np.array([[1.0, 2.0], [3.0, 4.0]])]
        w_after = [np.array([[2.0, 3.0], [4.0, 5.0]])]
        assert compute_drift(w_before, w_after) == pytest.approx(2.0)

    def test_multi_layer_drift(self):
        w_before = [np.zeros((2, 2)), np.zeros((3, 3))]
        w_after = [np.ones((2, 2)), np.ones((3, 3))]
        # sqrt(4 + 9) = sqrt(13)
        assert compute_drift(w_before, w_after) == pytest.approx(np.sqrt(13.0))
