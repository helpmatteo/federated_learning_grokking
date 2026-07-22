"""Tests for the model and loss registries."""

import pytest
import torch

from fedgrok.core.config import Config
from fedgrok.core.fed_config import FedConfig
from fedgrok.models.groknet import GrokNet
from fedgrok.core.registry import (
    build_model, registered_models, register_model, build_loss,
)


class TestBuildModel:
    def test_groknet_is_registered(self):
        assert "groknet" in registered_models()

    def test_builds_groknet_with_modular_dims(self):
        cfg = Config(p=97, hidden_width=256, activation="quadratic")
        model = build_model(cfg)
        assert isinstance(model, GrokNet)
        assert model.W1.shape == (256, 2 * 97)   # (N, 2p)
        assert model.W2.shape == (97, 256)       # (p, N)

    def test_respects_activation(self):
        cfg = Config(p=7, hidden_width=8, activation="relu")
        assert build_model(cfg).activation == "relu"

    def test_default_model_is_groknet(self):
        # Config.model defaults to "groknet"; build_model must honour it.
        assert Config().model == "groknet"
        assert isinstance(build_model(Config(p=7, hidden_width=8)), GrokNet)

    def test_fedconfig_also_dispatches(self):
        model = build_model(FedConfig(p=17, hidden_width=32))
        assert isinstance(model, GrokNet)
        assert model.W1.shape == (32, 34)

    def test_unknown_model_raises(self):
        cfg = Config(p=7, hidden_width=8)
        cfg.model = "does_not_exist"
        with pytest.raises(ValueError, match="Unknown model"):
            build_model(cfg)

    def test_double_registration_raises(self):
        with pytest.raises(ValueError, match="already registered"):
            register_model("groknet")(lambda cfg: None)


class TestBuildLoss:
    def test_default_loss_is_mse(self):
        assert Config().loss == "mse"

    def test_mse_target_is_onehot_float(self):
        spec = build_loss(Config(loss="mse"))
        labels = torch.tensor([0, 2, 4])
        target = spec.prepare_target(labels, 7)
        assert target.shape == (3, 7)
        assert target.dtype == torch.float32
        # one row per label, a single 1.0 in the labelled column
        assert torch.equal(target.sum(dim=1), torch.ones(3))
        assert target[1, 2] == 1.0

    def test_ce_target_is_class_indices(self):
        spec = build_loss(Config(loss="ce"))
        labels = torch.tensor([0, 2, 4])
        target = spec.prepare_target(labels, 7)
        assert target.shape == (3,)
        assert target.dtype == torch.int64
        assert torch.equal(target, labels)

    def test_mse_loss_runs_on_logits(self):
        spec = build_loss(Config(loss="mse"))
        logits = torch.randn(3, 7)
        target = spec.prepare_target(torch.tensor([0, 1, 2]), 7)
        assert torch.isfinite(spec.loss_fn(logits, target))

    def test_ce_loss_runs_on_logits(self):
        spec = build_loss(Config(loss="ce"))
        logits = torch.randn(3, 7)
        target = spec.prepare_target(torch.tensor([0, 1, 2]), 7)
        assert torch.isfinite(spec.loss_fn(logits, target))

    def test_unknown_loss_raises(self):
        with pytest.raises(ValueError, match="Unknown loss"):
            build_loss(Config(loss="hinge"))

    def test_target_follows_label_device(self):
        for name in ("mse", "ce"):
            spec = build_loss(Config(loss=name))
            labels = torch.tensor([0, 1, 2])
            assert spec.prepare_target(labels, 7).device == labels.device
