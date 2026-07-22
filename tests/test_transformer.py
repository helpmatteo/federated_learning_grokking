"""Tests for the Nanda transformer (GrokFormer) and its registry wiring."""

import pytest
import torch

from fedgrok.core.config import Config
from fedgrok.models.transformer import GrokFormer
from fedgrok.models.groknet import GrokNet
from fedgrok.core.registry import build_model, build_loss
from fedgrok.metrics.fourier import fourier_applicable
from fedgrok.data.modular import make_dataset


class TestGrokFormer:
    def test_output_shape_matches_p(self):
        model = GrokFormer(p=17, d_model=32, n_heads=4)
        x = torch.zeros(5, 34)               # (batch, 2p) one-hot
        x[range(5), 0] = 1.0                 # operand a = 0
        x[range(5), 17] = 1.0                # operand b = 0
        assert model(x).shape == (5, 17)

    def test_consumes_the_same_onehot_as_groknet(self):
        """The transformer must run on the modular dataset unchanged."""
        cfg = Config(task="addition", p=17, model="transformer", hidden_width=32)
        x_train, _, _, _ = make_dataset(cfg)
        model = build_model(cfg)
        out = model(x_train[:16])
        assert out.shape == (16, 17)
        assert torch.isfinite(out).all()

    def test_d_model_must_divide_n_heads(self):
        with pytest.raises(ValueError, match="not divisible"):
            GrokFormer(p=17, d_model=30, n_heads=4)

    def test_has_no_layernorm(self):
        """Nanda's setup omits LayerNorm; guard against it creeping in."""
        model = GrokFormer(p=17, d_model=32)
        assert not any(isinstance(m, torch.nn.LayerNorm) for m in model.modules())

    def test_exposes_P_but_not_W1(self):
        """P mirrors GrokNet for output dim; absence of W1 makes Fourier metrics
        correctly skip it."""
        model = GrokFormer(p=17, d_model=32)
        assert model.P == 17
        assert not hasattr(model, "W1")


class TestRegistryWiring:
    def test_transformer_is_registered(self):
        cfg = Config(p=17, model="transformer", hidden_width=32)
        assert isinstance(build_model(cfg), GrokFormer)

    def test_groknet_still_default(self):
        assert isinstance(build_model(Config(p=17, hidden_width=16)), GrokNet)

    def test_transformer_is_not_fourier_applicable(self):
        cfg = Config(p=17, model="transformer", hidden_width=32)
        assert fourier_applicable(build_model(cfg)) is False

    def test_transformer_trains_a_step_with_ce(self):
        """End-to-end: one CE gradient step on the transformer must be finite."""
        cfg = Config(task="addition", p=17, model="transformer", hidden_width=32,
                     loss="ce", optimizer="adamw", lr=1e-3)
        x_train, y_train, _, _ = make_dataset(cfg)
        model = build_model(cfg)
        spec = build_loss(cfg)
        target = spec.prepare_target(y_train, cfg.p)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        loss0 = spec.loss_fn(model(x_train), target)
        opt.zero_grad(); loss0.backward(); opt.step()
        loss1 = spec.loss_fn(model(x_train), target)
        assert torch.isfinite(loss0) and torch.isfinite(loss1)
