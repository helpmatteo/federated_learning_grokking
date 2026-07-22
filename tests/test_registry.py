"""Tests for the model registry."""

import pytest

from fedgrok.core.config import Config
from fedgrok.core.fed_config import FedConfig
from fedgrok.models.groknet import GrokNet
from fedgrok.core.registry import build_model, registered_models, register_model


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
