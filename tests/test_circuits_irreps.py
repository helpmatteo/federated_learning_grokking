"""Tests for the exact mechanistic instruments: S_5 irreps and the quadratic split.

Both modules claim EXACTNESS rather than approximation, so these tests are not
smoke tests -- they pin the algebraic identities the claims rest on:

  * the character table really is S_5's (orthonormality, not a citation)
  * the isotypic projectors resolve the identity and are mutually orthogonal
  * Parseval holds, so per-irrep energies partition ||W||^2
  * model(x) == A + 2T + B to float precision, on a TRAINED model as well as at
    init, because a decomposition that only holds at small weights is useless

Plus the probe contract that the training loops depend on: a probe must not
consume torch RNG (it runs inside the eval block, so a draw would shift every
subsequent weight update) and must not mutate parameters.
"""

import numpy as np
import pytest
import torch

from fedgrok.core.config import Config
from fedgrok.core.registry import build_model
from fedgrok.data.groups import _elements, _parity, coset_labels
from fedgrok.data.registry import build_dataset
from fedgrok.metrics import irreps
from fedgrok.metrics import quadratic_circuits as qc
from fedgrok.metrics.probes import mechanistic_probe, probe_keys

S5_ORDER = 120
CLASS_SIZES = (1, 10, 15, 20, 20, 30, 24)


def setup_d_cfg(**over):
    base = dict(dataset="s5", group_n=5, model="groknet", hidden_width=256,
                loss="ce", optimizer="adamw", lr=1e-3, weight_decay=1.0,
                alpha=0.45, seed=42)
    base.update(over)
    return Config(**base)


# ---------------------------------------------------------------- character table

def test_class_sizes_sum_to_group_order():
    assert sum(CLASS_SIZES) == S5_ORDER


def test_character_table_is_orthonormal():
    """The real content of the hardcoded table: rows orthonormal under the
    class-weighted inner product. A typo anywhere fails this."""
    chars = np.array(list(irreps.S5_CHARACTERS.values()), dtype=np.float64)
    weights = np.array(CLASS_SIZES, dtype=np.float64) / S5_ORDER
    gram = (chars * weights) @ chars.T
    assert np.allclose(gram, np.eye(len(chars)), atol=1e-12)


def test_irrep_dimensions_square_to_group_order():
    dims = irreps.irrep_dims(5)
    assert sorted(dims) == [1, 1, 4, 4, 5, 5, 6]
    assert sum(d * d for d in dims) == S5_ORDER


def test_non_s5_rejected_with_actionable_message():
    with pytest.raises(NotImplementedError, match="S_5 only"):
        irreps.isotypic_projectors(4)


# ------------------------------------------------------------------- projectors

def test_projectors_resolve_the_identity():
    _, _, proj = irreps.isotypic_projectors(5)
    assert np.allclose(proj.sum(axis=0), np.eye(S5_ORDER), atol=1e-9)


def test_projectors_are_orthogonal_idempotents():
    _, _, proj = irreps.isotypic_projectors(5)
    for r in range(len(proj)):
        assert np.allclose(proj[r] @ proj[r], proj[r], atol=1e-9)
        assert np.allclose(proj[r], proj[r].T, atol=1e-12)
        for s in range(len(proj)):
            if r != s:
                assert np.allclose(proj[r] @ proj[s], 0.0, atol=1e-9)


def test_projector_traces_equal_squared_dimensions():
    """trace P_rho = d_rho^2 — the isotypic component's dimension in the
    regular representation. Ties the projectors back to the dimensions."""
    _, dims, proj = irreps.isotypic_projectors(5)
    traces = [float(np.trace(proj[r])) for r in range(len(proj))]
    assert np.allclose(traces, [d * d for d in dims], atol=1e-9)


def test_projectors_are_cached_not_rebuilt():
    first = irreps.isotypic_projectors(5)[2]
    assert irreps.isotypic_projectors(5)[2] is first


# --------------------------------------------------------------------- Parseval

@pytest.mark.parametrize("dtype,tol", [(torch.float64, 1e-10), (torch.float32, 1e-5)])
def test_parseval_holds(dtype, tol):
    torch.manual_seed(0)
    block = (torch.randn(64, S5_ORDER) / 8).to(dtype)
    total = float((block.double() ** 2).sum())
    assert sum(irreps.energies(block).values()) == pytest.approx(total, rel=tol)


def test_fractions_sum_to_one():
    torch.manual_seed(1)
    block = torch.randn(32, S5_ORDER)
    assert sum(irreps.fractions(block).values()) == pytest.approx(1.0, rel=1e-6)


def test_zero_block_gives_zeros_not_nan():
    frac = irreps.fractions(torch.zeros(4, S5_ORDER))
    assert all(v == 0.0 for v in frac.values())


def test_wrong_group_dimension_rejected():
    with pytest.raises(ValueError, match="index the group"):
        irreps.energies(torch.zeros(4, 97))


# -------------------------------------------------------- known-answer decompositions

def test_constant_function_is_entirely_trivial_irrep():
    frac = irreps.fractions(torch.ones(8, S5_ORDER))
    assert frac["5"] == pytest.approx(1.0, abs=1e-6)


def test_sign_function_is_entirely_sign_irrep():
    sign = torch.tensor([[1.0 - 2 * _parity(p) for p in _elements(5)]])
    frac = irreps.fractions(sign)
    assert frac["11111"] == pytest.approx(1.0, abs=1e-6)


def test_coset_indicator_lives_in_low_dimensional_irreps():
    """An S_4-coset indicator is constant on cosets, so it lies in the
    permutation representation on 5 cosets = trivial + standard ([4,1]) only."""
    labels = coset_labels(5, "s_nm1")
    indicator = torch.tensor((labels == 0).astype(np.float64)).unsqueeze(0)
    frac = irreps.fractions(indicator)
    assert frac["5"] + frac["41"] == pytest.approx(1.0, abs=1e-9)


def test_random_block_matches_the_d_squared_baseline():
    torch.manual_seed(2)
    block = torch.randn(512, S5_ORDER, dtype=torch.float64)
    frac = irreps.fractions(block)
    for value, base in zip(frac.values(), irreps.random_baseline_fractions(5)):
        assert value == pytest.approx(base, abs=0.02)


def test_structure_score_bounds():
    torch.manual_seed(3)
    random_block = torch.randn(512, S5_ORDER, dtype=torch.float64)
    sign = torch.tensor([[1.0 - 2 * _parity(p) for p in _elements(5)]])
    assert irreps.structure_score(random_block) < 0.05
    assert irreps.structure_score(sign) > 0.95


# ------------------------------------------------------- quadratic decomposition

def test_decomposition_reproduces_the_model_at_init():
    cfg = setup_d_cfg()
    _, _, x_test, _ = build_dataset(cfg)
    model = build_model(cfg)
    with torch.no_grad():
        reference = model(x_test)
        rebuilt = qc.reconstruct(model, x_test)
    scale = reference.abs().max()
    assert (reference - rebuilt).abs().max() < 1e-5 * scale


def test_decomposition_reproduces_the_model_after_training():
    """The identity is algebraic, so it must survive weights that are no longer
    small and no longer random — the regime every measurement is taken in."""
    cfg = setup_d_cfg(alpha=0.55)
    x_train, y_train, x_test, _ = build_dataset(cfg)
    model = build_model(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1.0)
    loss_fn = torch.nn.CrossEntropyLoss()
    for _ in range(200):
        opt.zero_grad()
        loss_fn(model(x_train), y_train).backward()
        opt.step()
    with torch.no_grad():
        reference = model(x_test)
        rebuilt = qc.reconstruct(model, x_test)
    scale = reference.abs().max()
    assert scale > 1.0, "training did not move the weights; test is vacuous"
    assert (reference - rebuilt).abs().max() < 1e-4 * scale


def test_marginal_terms_are_blind_to_the_other_operand():
    """The whole claim rests on this: A cannot see b, B cannot see a. If either
    could, the interaction share would not measure composition."""
    cfg = setup_d_cfg()
    _, _, x_test, _ = build_dataset(cfg)
    model = build_model(cfg)
    a_term, _, b_term = qc.decompose(model, x_test)
    ia, ib = qc.operand_indices(x_test)
    for index, term in ((ia, a_term), (ib, b_term)):
        shared = index == index[0]
        assert shared.sum() > 1, "need repeated operands for this to bite"
        assert (term[shared] - term[shared][0]).abs().max() < 1e-6


def test_applicable_requires_quadratic_two_operand_mlp():
    assert qc.applicable(build_model(setup_d_cfg()))
    assert not qc.applicable(build_model(setup_d_cfg(activation="relu")))
    assert not qc.applicable(build_model(setup_d_cfg(
        model="transformer", hidden_width=128, n_heads=4, d_mlp=512)))


def test_interaction_units_spans_uniform_to_concentrated():
    model = build_model(setup_d_cfg())
    width = model.W1.shape[0]
    assert qc.interaction_units(model) > 0.5 * width      # random init: spread

    with torch.no_grad():                                  # one live unit
        model.W2.zero_()
        model.W2[:, 0] = 1.0
    assert qc.interaction_units(model) == pytest.approx(1.0, abs=1e-6)


def test_marginal_only_accuracy_is_a_real_ceiling():
    """A + B assign the same logits to every pair sharing an operand, so their
    accuracy cannot exceed what one operand can determine. On S_5 composition
    that is chance, since a alone fixes nothing about a*b."""
    cfg = setup_d_cfg()
    _, _, x_test, y_test = build_dataset(cfg)
    model = build_model(cfg)
    report = qc.circuit_report(model, x_test, y_test)
    assert report["circ_acc_marginal"] < 5.0


# ------------------------------------------------------------- probe integration

PROBE_CFGS = {
    "s5_groknet": setup_d_cfg(),
    "s5_transformer": setup_d_cfg(model="transformer", hidden_width=128,
                                  n_heads=4, d_mlp=512),
    "modular_groknet": Config(dataset="modular", task="addition", p=97,
                              model="groknet", hidden_width=256, loss="mse",
                              alpha=0.45, seed=42),
    "modular_transformer": Config(dataset="modular", task="addition", p=113,
                                  model="transformer", hidden_width=128,
                                  n_heads=4, d_mlp=512, loss="ce",
                                  alpha=0.45, seed=42),
}


@pytest.mark.parametrize("name", sorted(PROBE_CFGS))
def test_probe_keys_match_what_the_probe_emits(name):
    cfg = PROBE_CFGS[name]
    _, _, x_test, y_test = build_dataset(cfg)
    model = build_model(cfg)
    emitted = mechanistic_probe(cfg)(model, x_test, y_test, cfg)
    assert set(emitted) == set(probe_keys(cfg))


@pytest.mark.parametrize("name", sorted(PROBE_CFGS))
def test_probe_consumes_no_rng_and_mutates_nothing(name):
    cfg = PROBE_CFGS[name]
    _, _, x_test, y_test = build_dataset(cfg)
    model = build_model(cfg)
    before = [p.detach().clone() for p in model.parameters()]
    rng_state = torch.random.get_rng_state()

    mechanistic_probe(cfg)(model, x_test, y_test, cfg)

    assert torch.equal(rng_state, torch.random.get_rng_state()), \
        "probe drew from the RNG; every subsequent weight update would shift"
    for old, new in zip(before, model.parameters()):
        assert torch.equal(old, new)


def test_probe_emits_finite_values(name="s5_groknet"):
    cfg = PROBE_CFGS[name]
    _, _, x_test, y_test = build_dataset(cfg)
    emitted = mechanistic_probe(cfg)(build_model(cfg), x_test, y_test, cfg)
    for key, value in emitted.items():
        assert np.isfinite(value), f"{key} = {value}"


def test_s5_transformer_still_gets_coset_only():
    """The new probe is a superset, so it must degrade cleanly for a model the
    exact decomposition does not cover rather than raising or emitting NaN."""
    cfg = PROBE_CFGS["s5_transformer"]
    assert set(probe_keys(cfg)) == {"coset_accuracy", "coset_purity"}


def test_modular_probe_keys_unchanged():
    """Guards the banked modular history schema: adding S_5 instruments must not
    add a key to any modular run."""
    assert probe_keys(PROBE_CFGS["modular_groknet"]) == ()
    assert probe_keys(PROBE_CFGS["modular_transformer"]) == ("embed_ipr",)


# --------------------------------------------------------------------- caching

def test_coset_labels_returns_a_fresh_copy():
    """It is memoised now; a caller mutating the result must not poison others."""
    first = coset_labels(5, "s_nm1")
    first[0] = 99
    assert coset_labels(5, "s_nm1")[0] == 0
