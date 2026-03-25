"""Tests for mechanistic metrics: Gini coefficient, effective rank,
neuron frequency assignment, and restricted/excluded loss."""

import math
import pytest
import torch

from core.metrics import (
    gini_coefficient,
    effective_rank,
    neuron_frequency_assignment,
    restricted_excluded_loss,
)
from core.model import GrokNet

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SMALL_P = 7


@pytest.fixture
def small_model():
    """GrokNet with p=7, N=16."""
    return GrokNet(input_dim=2 * SMALL_P, hidden_width=16, output_dim=SMALL_P)


# ===========================================================================
# gini_coefficient
# ===========================================================================

class TestGiniCoefficient:
    def test_uniform_near_zero(self):
        """Perfectly equal distribution should have Gini ~ 0."""
        x = torch.ones(100)
        g = gini_coefficient(x)
        assert abs(g) < 0.01, f"Expected near 0 for uniform, got {g}"

    def test_one_hot_near_one(self):
        """One-hot (max inequality) should have Gini near 1."""
        x = torch.zeros(100)
        x[0] = 1.0
        g = gini_coefficient(x)
        assert g > 0.95, f"Expected near 1 for one-hot, got {g}"

    def test_bounded_zero_one(self):
        """Gini should be in [0, 1]."""
        torch.manual_seed(42)
        x = torch.rand(50)
        g = gini_coefficient(x)
        assert 0.0 <= g <= 1.0, f"Gini out of bounds: {g}"

    def test_sparser_has_higher_gini(self):
        """A sparser vector should have a higher Gini than a denser one."""
        dense = torch.ones(20)
        sparse = torch.zeros(20)
        sparse[:3] = 1.0
        assert gini_coefficient(sparse) > gini_coefficient(dense)

    def test_all_zeros_returns_zero(self):
        """All-zero input should return 0."""
        x = torch.zeros(10)
        assert gini_coefficient(x) == 0.0

    def test_negative_values_use_abs(self):
        """Negative values should be handled via abs."""
        x = torch.tensor([-1.0, -1.0, -1.0, -1.0])
        g = gini_coefficient(x)
        assert abs(g) < 0.01, f"Expected near 0 for equal absolute values, got {g}"


# ===========================================================================
# effective_rank
# ===========================================================================

class TestEffectiveRank:
    def test_rank_one_matrix(self):
        """Rank-1 matrix should have effective rank ~1."""
        a = torch.randn(5, 1)
        b = torch.randn(1, 5)
        W = a @ b  # rank 1
        er = effective_rank(W)
        assert abs(er - 1.0) < 0.2, f"Expected erank ~1 for rank-1 matrix, got {er}"

    def test_identity_matrix(self):
        """Identity matrix should have effective rank ~n."""
        n = 5
        W = torch.eye(n)
        er = effective_rank(W)
        assert abs(er - n) < 0.2, f"Expected erank ~{n} for identity, got {er}"

    def test_positive(self):
        """Effective rank should be positive for non-zero matrix."""
        torch.manual_seed(0)
        W = torch.randn(4, 6)
        er = effective_rank(W)
        assert er > 0

    def test_bounded_by_min_dim(self):
        """Effective rank should be <= min(rows, cols)."""
        torch.manual_seed(1)
        W = torch.randn(3, 8)
        er = effective_rank(W)
        assert er <= min(3, 8) + 0.01, f"erank {er} exceeds min dim"

    def test_zero_matrix(self):
        """Zero matrix should have effective rank 0."""
        W = torch.zeros(4, 4)
        er = effective_rank(W)
        assert er == 0.0


# ===========================================================================
# neuron_frequency_assignment
# ===========================================================================

class TestNeuronFrequencyAssignment:
    def test_returns_correct_length(self, small_model):
        """Should return one frequency per neuron."""
        freqs = neuron_frequency_assignment(small_model)
        assert len(freqs) == small_model.N

    def test_frequencies_in_valid_range(self, small_model):
        """All frequencies should be in [0, p)."""
        freqs = neuron_frequency_assignment(small_model)
        for f in freqs:
            assert 0 <= f < small_model.P, f"Frequency {f} out of range [0, {small_model.P})"

    def test_pure_cosine_assigned_correct_freq(self):
        """A neuron with pure complex exponential at freq k should be assigned freq k.

        We use complex exponentials (not cosines) to avoid the conjugate
        symmetry ambiguity where cos(2*pi*k*t/p) produces equal magnitude
        at bins k and p-k.
        """
        p = 11  # larger prime to have more distinct bins
        model = GrokNet(input_dim=2 * p, hidden_width=4, output_dim=p)
        target_freqs = [1, 3, 2, 5]
        with torch.no_grad():
            for neuron_idx, target_freq in enumerate(target_freqs):
                t = torch.arange(p, dtype=torch.float32)
                # Use a skewed waveform so that DFT peak is unambiguously at target_freq
                # sin component breaks the conjugate symmetry
                wave = (torch.cos(2.0 * math.pi * target_freq * t / p)
                        + 0.5 * torch.sin(2.0 * math.pi * target_freq * t / p))
                model.W1.data[neuron_idx, :p] = wave
                model.W1.data[neuron_idx, p:] = 0.0
        freqs = neuron_frequency_assignment(model)
        assert freqs == target_freqs, f"Expected {target_freqs}, got {freqs}"

    def test_returns_list(self, small_model):
        """Should return a plain Python list."""
        freqs = neuron_frequency_assignment(small_model)
        assert isinstance(freqs, list)


# ===========================================================================
# restricted_excluded_loss
# ===========================================================================

class TestRestrictedExcludedLoss:
    def _make_batch(self, model):
        """Create a random input batch and one-hot targets."""
        torch.manual_seed(99)
        p = model.P
        batch_size = 20
        x = torch.randn(batch_size, model.D)
        targets = torch.randint(0, p, (batch_size,))
        y_onehot = torch.zeros(batch_size, p)
        y_onehot.scatter_(1, targets.unsqueeze(1), 1.0)
        return x, y_onehot

    def test_returns_both_keys(self, small_model):
        """Should return dict with 'restricted_loss' and 'excluded_loss'."""
        x, y_onehot = self._make_batch(small_model)
        result = restricted_excluded_loss(small_model, x, y_onehot, key_freqs=[1, 2])
        assert "restricted_loss" in result
        assert "excluded_loss" in result

    def test_values_are_finite(self, small_model):
        """Both losses should be finite numbers."""
        x, y_onehot = self._make_batch(small_model)
        result = restricted_excluded_loss(small_model, x, y_onehot, key_freqs=[1, 2])
        assert math.isfinite(result["restricted_loss"])
        assert math.isfinite(result["excluded_loss"])

    def test_values_are_nonnegative(self, small_model):
        """MSE losses should be non-negative."""
        x, y_onehot = self._make_batch(small_model)
        result = restricted_excluded_loss(small_model, x, y_onehot, key_freqs=[1, 2])
        assert result["restricted_loss"] >= 0.0
        assert result["excluded_loss"] >= 0.0

    def test_all_freqs_restricted_equals_full(self, small_model):
        """When all frequencies are key, restricted_loss should equal full loss."""
        x, y_onehot = self._make_batch(small_model)
        p = small_model.P
        all_freqs = list(range(p))
        result = restricted_excluded_loss(small_model, x, y_onehot, key_freqs=all_freqs)
        # Compute full loss for comparison
        with torch.no_grad():
            logits = small_model(x)
            full_loss = torch.nn.functional.mse_loss(logits, y_onehot).item()
        assert abs(result["restricted_loss"] - full_loss) < 1e-5, (
            f"restricted_loss {result['restricted_loss']} != full_loss {full_loss}"
        )

    def test_all_freqs_excluded_near_zero_target(self, small_model):
        """When all freqs are key, excluded logits are zero, so excluded_loss
        should equal MSE(0, y_onehot)."""
        x, y_onehot = self._make_batch(small_model)
        p = small_model.P
        all_freqs = list(range(p))
        result = restricted_excluded_loss(small_model, x, y_onehot, key_freqs=all_freqs)
        expected = torch.nn.functional.mse_loss(
            torch.zeros_like(y_onehot), y_onehot
        ).item()
        assert abs(result["excluded_loss"] - expected) < 1e-5

    def test_empty_key_freqs(self, small_model):
        """With no key frequencies, restricted should have zero logits."""
        x, y_onehot = self._make_batch(small_model)
        result = restricted_excluded_loss(small_model, x, y_onehot, key_freqs=[])
        # restricted logits are all zero -> loss = MSE(0, y_onehot)
        expected = torch.nn.functional.mse_loss(
            torch.zeros_like(y_onehot), y_onehot
        ).item()
        assert abs(result["restricted_loss"] - expected) < 1e-5
