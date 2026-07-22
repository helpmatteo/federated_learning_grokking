"""Tests for the manifest / spec machinery (pure, no training)."""

import pytest

from fedgrok.core.config import Config
from fedgrok.core.fed_config import FedConfig
from fedgrok.manifest import (
    build_config, run_id, expand_grid, config_class,
    load_manifest, write_manifest,
)


class TestBuildConfig:
    def test_centralized_spec_builds_config(self):
        spec = {"mode": "centralized", "task": "addition", "p": 97,
                "alpha": 0.3, "seed": 42, "lr": 50.0}
        cfg = build_config(spec)
        assert isinstance(cfg, Config) and not isinstance(cfg, FedConfig)
        assert cfg.p == 97 and cfg.alpha == 0.3 and cfg.lr == 50.0

    def test_federated_spec_builds_fedconfig(self):
        spec = {"mode": "federated", "task": "addition", "p": 97,
                "num_clients": 10, "local_epochs": 5, "partition": "dirichlet",
                "dirichlet_alpha": 0.1}
        cfg = build_config(spec)
        assert isinstance(cfg, FedConfig)
        assert cfg.num_clients == 10 and cfg.dirichlet_alpha == 0.1

    def test_tags_are_ignored_not_passed_to_config(self):
        spec = {"mode": "centralized", "p": 7, "tier": "T0", "group": "wd",
                "experiment": "exp2", "label": "x", "id": "abc"}
        cfg = build_config(spec)  # must not raise on tag keys
        assert cfg.p == 7

    def test_unknown_field_raises(self):
        with pytest.raises(ValueError, match="neither"):
            build_config({"mode": "centralized", "p": 7, "not_a_field": 1})

    def test_bad_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown mode"):
            build_config({"mode": "distributed", "p": 7})


class TestRunId:
    def test_explicit_id_wins(self):
        assert run_id({"id": "myid", "mode": "centralized", "p": 7}) == "myid"

    def test_id_is_stable_across_key_order_and_tags(self):
        a = {"mode": "federated", "task": "addition", "p": 97, "seed": 42}
        b = {"seed": 42, "p": 97, "task": "addition", "mode": "federated",
             "tier": "T1", "group": "spine"}  # tags must not change the id
        assert run_id(a) == run_id(b)

    def test_id_changes_with_config(self):
        a = {"mode": "federated", "task": "addition", "p": 97, "seed": 42}
        b = {"mode": "federated", "task": "addition", "p": 97, "seed": 43}
        assert run_id(a) != run_id(b)


class TestExpandGrid:
    def test_cartesian_product_count(self):
        specs = expand_grid(
            {"mode": "federated", "task": "addition", "p": 97},
            {"seed": [42, 123, 456], "local_epochs": [1, 5]},
        )
        assert len(specs) == 6

    def test_every_spec_has_unique_stable_id(self):
        specs = expand_grid(
            {"mode": "federated", "task": "addition", "p": 97},
            {"seed": [42, 123], "local_epochs": [1, 5]},
        )
        ids = [s["id"] for s in specs]
        assert len(set(ids)) == len(ids)

    def test_tags_applied_to_all(self):
        specs = expand_grid(
            {"mode": "centralized", "p": 97},
            {"seed": [1, 2]},
            tags={"tier": "T0", "group": "pilot"},
        )
        assert all(s["tier"] == "T0" and s["group"] == "pilot" for s in specs)

    def test_expanded_specs_build(self):
        specs = expand_grid(
            {"mode": "federated", "task": "addition", "p": 7, "hidden_width": 16},
            {"seed": [1], "partition": ["iid", "dirichlet"]},
        )
        for spec in specs:
            build_config(spec)  # must not raise


class TestManifestIO:
    def test_roundtrip(self, tmp_path):
        specs = expand_grid(
            {"mode": "federated", "task": "addition", "p": 97},
            {"seed": [42, 123], "local_epochs": [1, 5]},
            tags={"tier": "T1"},
        )
        path = tmp_path / "m.jsonl"
        write_manifest(specs, str(path))
        loaded = load_manifest(str(path))
        assert len(loaded) == len(specs)
        assert {s["id"] for s in loaded} == {s["id"] for s in specs}

    def test_comments_and_blanks_skipped(self, tmp_path):
        path = tmp_path / "m.jsonl"
        path.write_text(
            '# a comment\n'
            '\n'
            '{"mode": "centralized", "p": 97, "seed": 1}\n'
        )
        loaded = load_manifest(str(path))
        assert len(loaded) == 1 and loaded[0]["p"] == 97
