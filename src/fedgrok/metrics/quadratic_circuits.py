"""Exact circuit decomposition for a quadratic-activation two-operand MLP.

WHY THIS IS EXACT AND NOT A PROBE. Interpretability normally has to approximate:
fit a linear probe, ablate and hope, correlate an order parameter. For GrokNet's
quadratic activation on a concatenated one-hot pair, the split into "uses one
operand" and "composes both" is an algebraic identity with nothing to fit.

With input x = onehot(a) || onehot(b) over a group of order G, and

    U = W1[:, :G]      (N, G)   first-operand block
    V = W1[:, G:]      (N, G)   second-operand block

the preactivation of hidden unit k on pair (a, b) is exactly U[k,a] + V[k,b].
Squaring expands, and the output is

    logit_c(a,b) = A[c,a]  +  2 T[c,a,b]  +  B[c,b]

    A = W2 @ U**2                     (P, G)   depends on a ONLY
    B = W2 @ V**2                     (P, G)   depends on b ONLY
    T[c,a,b] = sum_k W2[c,k] U[k,a] V[k,b]     the cross term

A and B are marginals: whatever they encode, they cannot distinguish two pairs
that share an operand, so they CANNOT implement a group operation. Every bit of
composition the network performs lives in T. So "how much of this function is
real composition versus memorised marginal structure" is not an interpretation
here -- it is a projection, and `reconstruct` asserts it reproduces the model's
logits to float precision.

COST. The full T is (P, G, G) and is never formed. Every measure below needs
only T evaluated on the pairs actually being scored,

    T_pairs = (U[:, ia] * V[:, ib]).T @ W2.T        (n_pairs, P)

which for setup D's 7,920 test pairs is 243 MFLOP -- cheaper than the forward
pass it explains.

THE ONE HEURISTIC. `interaction_units` treats the N rank-1 CP components of T as
if their sizes partitioned its energy. They are not orthogonal, so that is a
size heuristic rather than an exact decomposition -- the same convention the
repo's existing IPR and Gini measures use. Everything else in this module is
exact.
"""

import torch

from fedgrok.metrics.fourier import compute_accuracy


def applicable(model, cfg=None) -> bool:
    """True iff the exact decomposition holds for this model.

    Requires the quadratic activation and a W1 whose columns split evenly into
    two one-hot operand blocks. A relu/gelu GrokNet is excluded: the expansion
    of phi(u + v) into single-operand and cross terms is specific to squaring.
    """
    if not (hasattr(model, "W1") and hasattr(model, "W2")):
        return False
    if getattr(model, "activation", None) != "quadratic":
        return False
    return model.W1.shape[1] % 2 == 0


def operand_blocks(model):
    """(U, V) -- the two one-hot operand blocks of W1, each (N, G)."""
    width = model.W1.shape[1] // 2
    W1 = model.W1.data
    return W1[:, :width], W1[:, width:]


def operand_indices(x):
    """Recover (ia, ib) from concatenated one-hot inputs, without touching cfg."""
    width = x.shape[1] // 2
    return x[:, :width].argmax(dim=1), x[:, width:].argmax(dim=1)


def decompose(model, x):
    """The three exact logit terms for each row of `x`.

    Returns (marginal_a, interaction, marginal_b), each (n_pairs, P). Their sum
    is the model's logits.
    """
    U, V = operand_blocks(model)
    W2 = model.W2.data
    ia, ib = operand_indices(x)

    # (P, G) tables, then gather the column each example needs.
    A = W2 @ (U ** 2)
    B = W2 @ (V ** 2)
    # Cross term on just the scored pairs -- the full (P, G, G) is never formed.
    interaction = 2.0 * ((U[:, ia] * V[:, ib]).t() @ W2.t())
    return A[:, ia].t(), interaction, B[:, ib].t()


def reconstruct(model, x):
    """Sum of the three terms -- must equal model(x). Used by the tests."""
    a_term, interaction, b_term = decompose(model, x)
    return a_term + interaction + b_term


def _centred_energy(term):
    """Mean squared deviation across classes, averaged over examples.

    Centring on the class axis is what makes this measure meaningful: a term
    that is constant across classes shifts every logit equally and cannot change
    an argmax, so it should count as zero contribution rather than as energy.
    """
    centred = term - term.mean(dim=1, keepdim=True)
    return float((centred ** 2).sum(dim=1).mean())


def interaction_units(model):
    """Effective number of hidden units carrying the compositional circuit.

    T is a sum of N rank-1 terms, one per hidden unit, whose size is
    w_k = ||W2[:,k]|| * ||U[k,:]|| * ||V[k,:]||. The participation ratio
    (sum w^2)^2 / sum w^4 reads as "how many units are doing the composing":
    N if every unit contributes equally, ~1 if a single unit dominates.
    """
    U, V = operand_blocks(model)
    W2 = model.W2.data
    w = W2.norm(dim=0) * U.norm(dim=1) * V.norm(dim=1)
    sq = (w ** 2).sum()
    if sq <= 0:
        return 0.0
    return float(sq ** 2 / (w ** 4).sum())


def circuit_report(model, x, y) -> dict:
    """Per-eval scalars describing the compositional / marginal split.

    share_interaction   fraction of class-varying logit energy in the cross term
    share_marginal      the same for A + B (the two do not sum to 1: the terms
                        are not orthogonal, so a cross-covariance remains, and
                        reporting both separately is more honest than forcing a
                        two-way split)
    acc_interaction     accuracy of the cross term alone -- composition with all
                        marginal preference stripped out
    acc_marginal        accuracy of A + B alone. This is the ceiling reachable
                        WITHOUT composing, so it is the number that says how much
                        of the test score is really group structure.
    units               participation ratio over hidden units (see above)
    """
    with torch.no_grad():
        a_term, interaction, b_term = decompose(model, x)
        marginal = a_term + b_term
        total = _centred_energy(marginal + interaction)
        if total <= 0:
            share_i = share_m = 0.0
        else:
            share_i = _centred_energy(interaction) / total
            share_m = _centred_energy(marginal) / total
        return {
            "circ_share_interaction": share_i,
            "circ_share_marginal": share_m,
            "circ_acc_interaction": compute_accuracy(interaction, y),
            "circ_acc_marginal": compute_accuracy(marginal, y),
            "circ_units": interaction_units(model),
        }


CIRCUIT_KEYS = ("circ_share_interaction", "circ_share_marginal",
                "circ_acc_interaction", "circ_acc_marginal", "circ_units")
