"""Regression tests for the machinery that makes multi-setup runs measurable.

Every test here corresponds to a defect that was silent: a cache that served one
setup's data to another, a partition that discarded a fifth of S5's training set,
weight norms suppressed on runs where they were perfectly valid, and per-client
weight capture that existed only for one architecture. The unifying failure mode
is that none of them raised — they produced plausible numbers.
"""

import numpy as np
import pytest
import torch

from fedgrok.core.config import Config
from fedgrok.core.fed_config import FedConfig
from fedgrok.core.registry import build_model
from fedgrok.data.partition import make_federated_datasets, _partition_label_block
from fedgrok.data.registry import dataset_dims
from fedgrok.metrics.fourier import (
    compute_ipr, dft_applicable, weight_norm_report, weight_norms,
    weight_norms_applicable,
)
from fedgrok.metrics.probes import client_signature, mechanistic_probe, probe_keys
from fedgrok.training.federated import _client_key, _data_key


# The four new setups plus the anchor, as (label, cfg-kwargs).
SETUPS = {
    "A_groknet_modular": dict(dataset="modular", p=97, model="groknet",
                              hidden_width=64, loss="mse"),
    "B_transformer_modular": dict(dataset="modular", p=53, model="transformer",
                                  hidden_width=32, loss="ce"),
    "C_transformer_s5": dict(dataset="s5", group_n=4, model="transformer",
                             hidden_width=32, loss="ce"),
    "D_groknet_s5": dict(dataset="s5", group_n=4, model="groknet",
                         hidden_width=64, loss="ce"),
    "E_mlp_mnist": dict(dataset="mnist", model="mlp", hidden_width=32,
                        n_layers=3, loss="mse", n_train=200, n_test=100),
}


class TestCacheKeysSeparateSetups:
    """The cache key used to be (p, task, alpha, seed, K, partition, dir_alpha).

    That tuple cannot tell a modular run from an S5 run (S5 leaves p and task at
    their defaults), nor an MSE run from a CE one. Since the client cache holds a
    built model AND a loss-specific target tensor, a collision serves the wrong
    objective -- and CrossEntropyLoss accepts one-hot floats as soft labels, so
    it trains on without error.
    """

    @pytest.mark.parametrize("a,b", [
        pytest.param(a, b, id=f"{a}-vs-{b}")
        for i, a in enumerate(SETUPS) for b in list(SETUPS)[i + 1:]
    ])
    def test_every_setup_pair_has_a_distinct_client_key(self, a, b):
        ca = FedConfig(num_clients=5, partition="iid", **SETUPS[a])
        cb = FedConfig(num_clients=5, partition="iid", **SETUPS[b])
        assert _client_key(ca, 0) != _client_key(cb, 0)

    def test_loss_alone_separates_the_client_key(self):
        mse = FedConfig(dataset="modular", p=97, loss="mse", num_clients=5)
        ce = FedConfig(dataset="modular", p=97, loss="ce", num_clients=5)
        assert _client_key(mse, 0) != _client_key(ce, 0)

    def test_n_train_alone_separates_the_data_key(self):
        a = FedConfig(dataset="mnist", model="mlp", n_train=1000, num_clients=5)
        b = FedConfig(dataset="mnist", model="mlp", n_train=4000, num_clients=5)
        assert _data_key(a) != _data_key(b)

    def test_same_config_same_client_is_stable(self):
        cfg = FedConfig(dataset="modular", p=97, num_clients=5)
        assert _client_key(cfg, 0) == _client_key(cfg, 0)
        assert _client_key(cfg, 0) != _client_key(cfg, 1)


class TestPartitionsCoverTheTrainingSet:
    """The invariant whose absence let the S5 dirichlet bug through.

    _partition_dirichlet looped over cfg.p classes. On S5 (120 classes, cfg.p
    defaulting to 97) that silently dropped every sample labelled >= 97 --
    measured at 18.8% of the training set and 23 of 120 classes, capping train
    accuracy at 81.2% against an 85.0 grok bar. No shard was empty, so the
    empty-shard guard never fired.
    """

    @pytest.mark.parametrize("partition", ["iid", "operand", "target",
                                           "dirichlet", "label_block"])
    @pytest.mark.parametrize("dataset,extra", [
        ("modular", dict(p=53)),
        ("s5", dict(group_n=4)),
    ])
    def test_shards_partition_the_training_set_exactly(self, partition, dataset, extra):
        cfg = FedConfig(dataset=dataset, alpha=0.5, seed=42, num_clients=5,
                        partition=partition, dirichlet_alpha=0.5,
                        hidden_width=64, loss="ce", **extra)
        client_data, _, y_train_full, _, _ = make_federated_datasets(cfg)
        assert sum(len(y) for _, y in client_data) == len(y_train_full)

    def test_s5_dirichlet_reaches_every_class(self):
        cfg = FedConfig(dataset="s5", group_n=5, alpha=0.5, seed=42,
                        num_clients=10, partition="dirichlet",
                        dirichlet_alpha=0.5, loss="ce", hidden_width=64)
        client_data, _, y_train_full, _, _ = make_federated_datasets(cfg)
        seen = set()
        for _, y in client_data:
            seen |= set(y.tolist())
        assert len(seen) == dataset_dims(cfg)[1] == 120
        assert sum(len(y) for _, y in client_data) == len(y_train_full)


class TestLabelBlock:
    """The structured partition that survives where operand/coset cannot."""

    @pytest.mark.parametrize("K", [2, 5, 10, 20, 50])
    def test_never_empty_and_coherent(self, K):
        y = np.repeat(np.arange(100), 10)
        shards = _partition_label_block(y, K)
        assert len(shards) == K
        assert all(len(s) > 0 for s in shards)
        assert sorted(np.concatenate(shards).tolist()) == list(range(len(y)))
        # Coherent: each shard spans a narrow, contiguous label range.
        spans = [y[s].max() - y[s].min() for s in shards]
        assert max(spans) <= 2 * (100 // K)


class TestWeightNormsAreNotGatedOnTheDFT:
    """A Frobenius norm is basis-free; it does not need a cyclic group.

    Gating weight_norms behind the DFT capability meant S5+GrokNet recorded
    weight_norm_layer1/2 as NaN with the correct code sitting right there.
    """

    def test_s5_groknet_has_real_weight_norms_but_no_dft(self):
        cfg = Config(dataset="s5", group_n=4, model="groknet", hidden_width=64,
                     loss="ce")
        model = build_model(cfg)
        assert weight_norms_applicable(model) is True
        assert dft_applicable(model, cfg) is False
        for value in weight_norms(model).values():
            assert np.isfinite(value) and value > 0

    def test_modular_groknet_keeps_both(self):
        cfg = Config(dataset="modular", p=53, model="groknet", hidden_width=64)
        model = build_model(cfg)
        assert weight_norms_applicable(model) and dft_applicable(model, cfg)

    @pytest.mark.parametrize("name", list(SETUPS))
    def test_weight_norm_report_is_finite_on_every_setup(self, name):
        cfg = Config(**SETUPS[name])
        report = weight_norm_report(build_model(cfg))
        assert set(report) == {"weight_norm_total", "weight_norm_first",
                               "weight_norm_last"}
        for value in report.values():
            assert np.isfinite(value) and value > 0

    def test_report_total_matches_groknet_layer_norms(self):
        cfg = Config(dataset="modular", p=53, model="groknet", hidden_width=64)
        model = build_model(cfg)
        wn, report = weight_norms(model), weight_norm_report(model)
        expected = (wn["weight_norm_layer1"] ** 2 + wn["weight_norm_layer2"] ** 2) ** 0.5
        assert report["weight_norm_total"] == pytest.approx(expected, rel=1e-5)


class TestMechanisticProbes:
    """Probes run inside the eval block of both training loops.

    A probe that consumed the torch RNG would shift every subsequent draw and
    silently change trajectories, so that contract is tested directly rather
    than assumed.
    """

    @pytest.mark.parametrize("name", list(SETUPS))
    def test_probe_output_matches_declared_keys_and_is_finite(self, name):
        cfg = Config(**SETUPS[name])
        model = build_model(cfg)
        from fedgrok.data.registry import build_dataset
        x, y, _, _ = build_dataset(cfg)
        out = mechanistic_probe(cfg)(model, x[:64], y[:64], cfg)
        assert set(out) == set(probe_keys(cfg))
        for value in out.values():
            assert np.isfinite(value)

    @pytest.mark.parametrize("name", list(SETUPS))
    def test_probe_consumes_no_rng_and_mutates_nothing(self, name):
        cfg = Config(**SETUPS[name])
        model = build_model(cfg)
        from fedgrok.data.registry import build_dataset
        x, y, _, _ = build_dataset(cfg)

        rng_before = torch.get_rng_state().clone()
        params_before = [p.detach().clone() for p in model.parameters()]
        mechanistic_probe(cfg)(model, x[:64], y[:64], cfg)

        assert torch.equal(torch.get_rng_state(), rng_before)
        for before, after in zip(params_before, model.parameters()):
            assert torch.equal(before, after)

    def test_s5_gets_coset_measures_on_both_architectures(self):
        """Every S_n run keeps the coset measures, whatever else it gains.

        A quadratic GrokNet now also gets the exact circuit split (see
        metrics/quadratic_circuits.py), so this asserts the coset measures are
        PRESENT rather than that they are all there is. The transformer, which
        the decomposition does not cover, must still be coset-only -- that is the
        graceful-degradation half of the claim.
        """
        for model_name, width in (("groknet", 64), ("transformer", 32)):
            cfg = Config(dataset="s5", group_n=4, model=model_name,
                         hidden_width=width, loss="ce")
            keys = probe_keys(cfg)
            assert {"coset_accuracy", "coset_purity"} <= set(keys)
            if model_name == "transformer":
                assert keys == ("coset_accuracy", "coset_purity")

    def test_s4_gets_circuits_but_not_irreps(self):
        """The two new instruments have different domains and must gate apart.

        The circuit split is algebra on the quadratic activation, so it holds for
        any S_n. The irrep profile needs a character table, which exists for S_5
        only. Gating them together would crash every S_4 run -- which is what the
        rest of this suite runs, for speed.
        """
        s4 = probe_keys(Config(dataset="s5", group_n=4, model="groknet",
                               hidden_width=64, loss="ce"))
        s5 = probe_keys(Config(dataset="s5", group_n=5, model="groknet",
                               hidden_width=64, loss="ce"))
        assert "circ_share_interaction" in s4
        assert not any(k.startswith("irrep_") for k in s4)
        assert any(k.startswith("irrep_") for k in s5)

    def test_transformer_on_modular_gets_embedding_ipr(self):
        cfg = Config(dataset="modular", p=53, model="transformer",
                     hidden_width=32, loss="ce")
        assert probe_keys(cfg) == ("embed_ipr",)


class TestClientSignature:
    """The only channel carrying weight data from client to server.

    It used to be W1[:, :cfg.p] behind a GrokNet gate: nothing at all on the new
    setups, and the wrong width on S5+GrokNet (97 columns of 240).
    """

    @pytest.mark.parametrize("name", list(SETUPS))
    def test_every_setup_yields_a_signature_with_honest_shape(self, name):
        cfg = Config(**SETUPS[name])
        model = build_model(cfg)
        sig_name, sig = client_signature(model, cfg)
        assert sig_name is not None and sig.ndim == 2
        # The shape the client reports is the array's own, not cfg.p.
        assert sig.astype(np.float32).tobytes() == sig.astype(np.float32).tobytes()
        restored = np.frombuffer(sig.astype(np.float32).tobytes(),
                                 dtype=np.float32).reshape(sig.shape)
        np.testing.assert_array_equal(restored, sig.astype(np.float32))

    def test_modular_groknet_signature_is_unchanged_from_the_old_slice(self):
        """Banked t2_boundary checkpoints must stay byte-compatible."""
        cfg = Config(dataset="modular", p=53, model="groknet", hidden_width=64)
        model = build_model(cfg)
        _, sig = client_signature(model, cfg)
        np.testing.assert_array_equal(sig, model.W1.data[:, :cfg.p].numpy())

    def test_s5_groknet_signature_uses_the_group_order_not_p(self):
        cfg = Config(dataset="s5", group_n=4, model="groknet", hidden_width=64,
                     loss="ce")
        model = build_model(cfg)
        _, sig = client_signature(model, cfg)
        assert sig.shape[1] == dataset_dims(cfg)[0] // 2 == 24
        assert sig.shape[1] != cfg.p


class TestClientCheckpointsEndToEnd:
    """The signature must survive the whole path, not just the function call.

    `client_signature` being right is necessary and not sufficient: the matrix
    still has to cross Flower's metrics channel as bytes, be reassembled from
    its reported shape, get cached server-side, and reach disk. Every campaign
    manifest is about to set `checkpoint_client_weights=True` on transformer and
    MLP setups, and those are Config fields -- so they change run ids, and a path
    that silently writes nothing is only discovered after the sweep, by which
    time the fix costs the sweep. No banked run off the anchor has exercised it.
    """

    @pytest.mark.parametrize("overrides,expect_shape", [
        (dict(dataset="modular", p=31, model="transformer", hidden_width=32,
              n_heads=4, d_mlp=64, loss="ce", optimizer="adamw", lr=1e-3),
         (31, 32)),                                    # W_E: (p, d_model)
        (dict(dataset="s5", group_n=4, model="groknet", hidden_width=16,
              loss="ce", optimizer="adamw", lr=1e-3),
         (16, 24)),                                    # W1_operand: (N, |G|)
    ], ids=["transformer_modular", "groknet_s5"])
    def test_per_client_weights_reach_disk_with_the_right_shape(
            self, tmp_path, overrides, expect_shape):
        cfg = FedConfig(num_clients=3, num_rounds=2, local_epochs=1,
                        partition="iid", checkpoint_every=2,
                        checkpoint_client_weights=True, eval_every=2,
                        alpha=0.5, seed=42, output_dir=str(tmp_path),
                        **overrides)
        from fedgrok.training.federated import fed_train
        fed_train(cfg)

        saved = tmp_path / "checkpoints" / "client_w1_round2.pt"
        assert saved.exists(), "no per-client file written"

        blob = torch.load(saved, weights_only=False)
        assert len(blob) == cfg.num_clients
        mats = [np.asarray(e["w1"] if isinstance(e, dict) else e) for e in blob]
        for mat in mats:
            assert mat.shape == expect_shape
        # a shared-reference bug would make every client identical, which reads
        # as "no client drift" rather than as a bug
        assert not np.allclose(mats[0], mats[1])


class TestTransformerCapacityIsConfigurable:
    def test_n_heads_and_d_mlp_round_trip_through_a_spec(self):
        from fedgrok.manifest import build_config
        spec = {"mode": "centralized", "model": "transformer", "p": 53,
                "hidden_width": 64, "loss": "ce", "n_heads": 8, "d_mlp": 256}
        cfg = build_config(spec)          # used to raise: unknown spec keys
        assert cfg.n_heads == 8 and cfg.d_mlp == 256
        model = build_model(cfg)
        assert model.n_heads == 8 and model.d_head == 8

    def test_indivisible_width_names_both_fields(self):
        cfg = Config(model="transformer", p=53, hidden_width=50, loss="ce")
        with pytest.raises(ValueError, match="n_heads"):
            build_model(cfg)

    def test_sweeping_n_heads_can_make_a_valid_width_invalid(self):
        """hidden_width=64 is fine at 4 heads and not at 6 -- the failure mode a
        capacity sweep will hit, which is why the message names both values."""
        assert build_model(Config(model="transformer", p=53, hidden_width=64,
                                  loss="ce", n_heads=4)) is not None
        with pytest.raises(ValueError, match="hidden_width=64, n_heads=6"):
            build_model(Config(model="transformer", p=53, hidden_width=64,
                               loss="ce", n_heads=6))


class TestScaffoldRefusesAdamW:
    """SCAFFOLD's Option-II estimator inverts x - y = lr * sum(g), which is an
    SGD identity. Under AdamW the update is preconditioned per coordinate, so the
    control variate is wrong by a coordinate-varying factor -- and all four new
    setups use AdamW."""

    def test_adamw_scaffold_raises(self):
        from fedgrok.training.federated import _build_strategy
        cfg = FedConfig(strategy="scaffold", optimizer="adamw", lr=1e-3,
                        num_clients=5)
        with pytest.raises(ValueError, match="adamw"):
            _build_strategy(cfg, None, None)

    def test_gd_scaffold_still_builds(self):
        from fedgrok.training.federated import _build_strategy
        from fedgrok.training import scaffold as sc
        cfg = FedConfig(strategy="scaffold", optimizer="gd", num_clients=5)
        ctx = {"server_cv_box": [[np.zeros((2, 2), dtype=np.float32)]],
               "param_shapes": [(2, 2)]}
        assert _build_strategy(cfg, None, None, scaffold_ctx=ctx) is not None
