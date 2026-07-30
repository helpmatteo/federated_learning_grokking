"""Per-round mechanistic progress measures, one per setup.

The modular study's mechanistic spine is a single scalar logged every eval round:
IPR, which rises as the Fourier circuit forms. Nothing analogous existed for any
other setup, so a transformer / S_5 / MNIST run produced accuracy curves and
nothing else — the "why" was unavailable at exactly the point the study started
caring about it.

`mechanistic_probe(cfg)` returns the right measure for a run:

    dataset s5 (any model)      coset_accuracy, coset_purity
                                Stander et al.'s coset structure is the S_n
                                analogue of the Fourier circuit. Model-agnostic
                                (it only needs argmax logits), so it covers both
                                GrokNet and the transformer on S_5.
    transformer + modular       embed_ipr — the same IPR formula applied to the
                                DFT of W_E over the token index. W_E's rows are
                                indexed by a cyclic variable exactly as GrokNet's
                                W1[:, :p] columns are, so this is the direct
                                analogue rather than a new measure.
    everything else             {} — GrokNet+modular already logs `ipr` in the
                                fixed series, and the MNIST MLP's order parameter
                                is the total weight norm, which
                                `weight_norm_report` logs for every setup.

Contract, enforced by tests: a probe must not consume torch RNG and must not
mutate parameters. Both training loops call it inside the eval block, and a probe
that drew from the RNG would shift every subsequent weight init and silently
change trajectories.
"""

import torch

from fedgrok.data.registry import dataset_dims
from fedgrok.metrics.fourier import spectral_ipr
from fedgrok.metrics.nonabelian import coset_attribution, nonabelian_applicable


def _no_probe(model, x, y, cfg):
    return {}


def _coset_probe(model, x, y, cfg):
    return coset_attribution(model, x, y, cfg)


def _embed_ipr_probe(model, x, y, cfg):
    """IPR of the transformer's token embedding in Fourier space.

    W_E is (p, d_model): row i is token i's embedding. Transposing gives
    (d_model, p) — one row per embedding dimension, indexed by the cyclic token
    variable — which is the layout `spectral_ipr` expects and the structural
    match to GrokNet's (N, p).
    """
    with torch.no_grad():
        return {"embed_ipr": spectral_ipr(model.W_E.data.T)}


def mechanistic_probe(cfg):
    """Return the probe callable for this run's (dataset, model) pair."""
    dataset = getattr(cfg, "dataset", "modular")
    model_name = getattr(cfg, "model", "groknet")

    if nonabelian_applicable(cfg):
        return _coset_probe
    if model_name == "transformer" and dataset == "modular":
        # The DFT presumes a cyclic group, so this is modular-only; on S_5 the
        # coset probe above takes over.
        return _embed_ipr_probe
    return _no_probe


def client_signature(model, cfg):
    """(name, ndarray) — the per-client weight matrix worth shipping at checkpoints.

    This is the ONLY channel carrying weight data from a client back to the
    server, and it used to be `model.W1[:, :cfg.p]` behind a GrokNet gate. So
    transformer / MNIST / S_5 runs captured nothing at all, and the "did
    different clients learn different things" analysis had no data source on any
    new setup. `cfg.p` was also the wrong width for S_5+GrokNet (97 columns of a
    240-column W1).

    Per architecture, the matrix whose structure the mechanism story is about:
      GrokNet      W1's first-operand block — dataset_dims[0]//2 columns, which
                   is exactly cfg.p on modular, so banked checkpoints stay
                   byte-identical
      transformer  W_E, the token embedding
      MLP          the first layer's weights

    Returns (None, None) for an architecture with none of these.
    """
    if hasattr(model, "W1"):
        operand_width = dataset_dims(cfg)[0] // 2
        return "W1_operand", model.W1.data[:, :operand_width].cpu().numpy()
    if hasattr(model, "W_E"):
        return "W_E", model.W_E.data.cpu().numpy()
    if hasattr(model, "layers") and len(model.layers):
        return "layer0", model.layers[0].weight.data.cpu().numpy()
    return None, None


def probe_keys(cfg):
    """Names the probe for `cfg` will emit — for schema checks without a model."""
    probe = mechanistic_probe(cfg)
    if probe is _coset_probe:
        return ("coset_accuracy", "coset_purity")
    if probe is _embed_ipr_probe:
        return ("embed_ipr",)
    return ()
