"""exp2's floor arm: one model trained on one client's worth of data.

The floor is the arm that says whether a client could have done this alone, so
getting its construction wrong does not error -- it produces a floor that is
secretly the ceiling, and every "aggregation rescues grokking" claim built on it
is then vacuous.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from build_manifests import reduced_arm                      # noqa: E402
from fedgrok.core.config import Config                       # noqa: E402
from fedgrok.core.fed_config import FedConfig                # noqa: E402
from fedgrok.manifest import build_config                    # noqa: E402


def _fed(**overrides):
    spec = {"mode": "federated", "dataset": "modular", "task": "addition",
            "p": 97, "alpha": 0.25, "seed": 42, "num_clients": 10,
            "num_rounds": 1_000, "local_epochs": 5, "partition": "iid",
            "eval_every": 20, "strategy": "fedavg"}
    spec.update(overrides)
    return spec


class TestGridDatasets:
    def test_alpha_is_divided_by_client_count(self):
        [red] = reduced_arm([_fed(alpha=0.5, num_clients=10)])
        assert red["alpha"] == pytest.approx(0.05)

    def test_result_is_a_buildable_centralized_config(self):
        """Every federated-only field must be gone, or build_config raises."""
        [red] = reduced_arm([_fed()])
        cfg = build_config(red)
        assert isinstance(cfg, Config) and not isinstance(cfg, FedConfig)
        assert red["mode"] == "centralized"

    def test_setup_identity_survives_the_reduction(self):
        """The floor has to be the SAME setup as the arm it is a floor for."""
        [red] = reduced_arm([_fed(dataset="s5", group_n=5, model="groknet",
                                  loss="ce", optimizer="adamw", lr=1e-3)])
        assert red["dataset"] == "s5" and red["model"] == "groknet"
        assert red["loss"] == "ce" and red["optimizer"] == "adamw"


class TestMnist:
    """MNIST IGNORES alpha -- its data axis is n_train.

    Reducing MNIST by scaling alpha would leave the spec identical to the full
    arm in every field that MNIST actually reads, so the floor would train on the
    whole dataset and match the ceiling by construction. Nothing would error.
    """

    def _mnist(self, **overrides):
        base = {"dataset": "mnist", "model": "mlp", "n_train": 2_000,
                "batch_size": 100, "hidden_width": 200, "n_layers": 3,
                "init_scale": 9.0, "loss": "mse", "optimizer": "adamw",
                "lr": 1e-3}
        base.update(overrides)
        return _fed(**base)

    def test_n_train_is_reduced_not_alpha(self):
        [red] = reduced_arm([self._mnist(num_clients=10, alpha=0.25)])
        assert red["n_train"] == 200
        assert red["alpha"] == 0.25          # untouched, and unread by MNIST

    def test_batch_shrinks_with_the_shard(self):
        """Otherwise the floor differs from the ceiling by more than data volume.

        A batch larger than the shard silently becomes "one partial batch per
        epoch", changing the optimiser's effective batch size -- which is exactly
        the quantity MNIST's grokking delay depends on.
        """
        [red] = reduced_arm([self._mnist(num_clients=50, batch_size=100)])
        assert red["n_train"] == 40
        assert red["batch_size"] == 40

    def test_batch_left_alone_when_the_shard_still_fits_it(self):
        [red] = reduced_arm([self._mnist(num_clients=10, batch_size=100)])
        assert red["n_train"] == 200 and red["batch_size"] == 100


class TestBudget:
    def test_floor_gets_a_multiple_of_the_fl_arms_gradient_steps(self):
        """A floor whose failure could be "ran out of time" proves nothing."""
        [red] = reduced_arm([_fed(num_rounds=1_000, local_epochs=5)],
                            budget_multiple=2.0)
        assert red["epochs"] == 10_000

    def test_multiple_is_tunable(self):
        [red] = reduced_arm([_fed(num_rounds=1_000, local_epochs=5)],
                            budget_multiple=0.5)
        assert red["epochs"] == 2_500


class TestDedup:
    def test_cells_differing_only_in_federated_fields_collapse_to_one(self):
        """A floor does not depend on E or on the partition, so it runs once."""
        red = reduced_arm([
            _fed(local_epochs=5, partition="iid", num_rounds=1_000),
            _fed(local_epochs=5, partition="operand", num_rounds=1_000),
        ])
        assert len(red) == 1

    def test_cells_differing_in_seed_do_not_collapse(self):
        red = reduced_arm([_fed(seed=42), _fed(seed=123)])
        assert len(red) == 2

    def test_differing_local_epochs_change_the_budget_so_do_not_collapse(self):
        """Budget is derived from rounds x E, so E survives as an epoch count."""
        red = reduced_arm([_fed(local_epochs=5), _fed(local_epochs=50)])
        assert len(red) == 2


class TestGuards:
    def test_rejects_a_centralized_spec(self):
        with pytest.raises(ValueError, match="federated"):
            reduced_arm([{"mode": "centralized", "alpha": 0.5}])

    def test_arm_and_source_k_are_tags_not_config(self):
        """They must reach the results table without costing a run id."""
        [red] = reduced_arm([_fed(num_clients=97)])
        assert red["arm"] == "cent_reduced" and red["reduced_from_k"] == 97
        build_config(red)                    # tags are accepted, not rejected

    def test_tags_do_not_enter_the_content_hash(self):
        from fedgrok.manifest import run_id
        [red] = reduced_arm([_fed()])
        bare = {k: v for k, v in red.items()
                if k not in ("id", "arm", "reduced_from_k")}
        assert run_id(bare) == red["id"]
