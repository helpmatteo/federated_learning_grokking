"""Generate the versioned run manifests under manifests/.

Each function builds one manifest (a JSONL of run specs) for a tier of the
plan. Regenerating is deterministic — run ids are content hashes — so
re-running never changes ids and the launcher's resume stays valid.

    python scripts/build_manifests.py            # write all
    python scripts/build_manifests.py t0_wd_grid # write one

Run ids are content hashes of the config, so a cell that appears in two
manifests (e.g. an E-spine point that is also a probe point) gets the SAME id
and the launcher executes it once — the second manifest's copy is skipped by
the normal resume check. Overlap between tiers is therefore free, not waste.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fedgrok.manifest import expand_grid, run_id, write_manifest

MANIFEST_DIR = os.path.join(os.path.dirname(__file__), "..", "manifests")
SEEDS5 = [42, 123, 456, 789, 1011]
SEEDS3 = [42, 123, 456]

# Every FL cell runs a FIXED number of communication rounds, so E is a clean
# *communication* axis and rounds — the expensive resource in FL — are matched
# across cells.
#
# NOTE (deliberate choice): we do NOT set num_rounds = budget // E to equalise
# gradient steps across E. That compute-matching is done at ANALYSIS time
# instead, using the `total_steps` axis the training loop already logs
# (alongside `sequential_steps` and `n_participating`) — see the Phase 0.6 step
# accounting. Keeping it out of the run config means:
#   - num_rounds stays an explicit, readable field in the manifest rather than
#     an implicit function of another field;
#   - a single run supports BOTH readings (per-round and per-step) rather than
#     baking one in;
#   - the consequence to keep in mind is that total gradient work scales with E
#     (steps = rounds x E), so raw wall-clock and total compute are NOT matched
#     across the E-spine, and low-E cells see proportionally fewer steps. When
#     comparing cells at equal compute, slice on `total_steps`, not on round.
#
# E RANGE (set by the t1_probe results, 2026-07-28): with rounds fixed, the
# extremes of the E axis are unusable in opposite directions, so the spine is
# restricted to E in {5, 10, 25, 50}:
#   - E = 1, 2 are UNDER-BUDGETED. At 10k rounds E=1 gets 10k gradient steps,
#     but this cell needs ~12.9k to grok (measured), so it censors for lack of
#     budget rather than for any federated reason — an uninformative cell that
#     invites exactly the wrong conclusion. (The E=1 identity is anyway proven
#     exactly in tests/test_fedavg_identity.py, so the spine does not need it.)
#   - E = 100, 250 are TOO EXPENSIVE. E=250 x 10k rounds is 2.5M gradient steps
#     per run (~1.5-2.5h); the probe's E=250 cells were cancelled for this.
# E in {5, 10, 25, 50} is a 10x span, all cells affordable and all informative.
FL_ROUNDS = 10_000
E_SPINE = [5, 10, 25, 50]
FL_EVAL_EVERY = 20          # ~500 curve points per run

# The one-shot (R=1) cell is the sole exception: with a single round there is no
# round axis, so its local-step count is stated outright.
ONE_SHOT_LOCAL_STEPS = 50_000


def t0_wd_grid():
    """Tier 0: loss x optimizer x weight-decay grid on setup A (mod-97 MLP).

    Fixes the coupled-WD defect and answers the lazy->rich / numerical-stability
    critiques. Weight-decay values are chosen so lr*wd (the comparable quantity)
    is log-spaced; see fedgrok.core.utils.check_decay_stability.

      GD    (lr=50)   : wd in {0, 2e-7, 2e-6, 2e-5, 2e-4} -> lr*wd {0,1e-5..1e-2}
      AdamW (lr=1e-3) : wd in {0, 0.01, 0.1, 1.0}         -> lr*wd {0,1e-5..1e-3}

    Centralized, 5 seeds. p=97, quadratic MLP, MSE (loss is fixed here; the CE
    arm needs the Phase 2/3 loss registry and is generated separately later).
    """
    specs = []
    specs += expand_grid(
        {"mode": "centralized", "task": "addition", "p": 97, "alpha": 0.5,
         "hidden_width": 256, "activation": "quadratic", "optimizer": "gd",
         "lr": 50.0, "epochs": 50000, "log_every": 100},
        {"weight_decay": [0.0, 2e-7, 2e-6, 2e-5, 2e-4], "seed": SEEDS5},
        tags={"tier": "T0", "group": "wd_grid", "experiment": "wd"},
    )
    specs += expand_grid(
        {"mode": "centralized", "task": "addition", "p": 97, "alpha": 0.5,
         "hidden_width": 256, "activation": "quadratic", "optimizer": "adamw",
         "lr": 1e-3, "epochs": 50000, "log_every": 100},
        {"weight_decay": [0.0, 0.01, 0.1, 1.0], "seed": SEEDS5},
        tags={"tier": "T0", "group": "wd_grid", "experiment": "wd"},
    )
    return specs


def t0_poly_pilot():
    """Tier 0 pilot: do x2+y2 and x2+xy+y2 grok centrally before FL compute?

    Doshi et al. (2406.03495) separate learnable from non-learnable modular
    polynomials for this architecture; a task that does not grok centrally can
    tell us nothing about FL, so gate the operation set on this.

    RESULT (run 2026-07-28, 3 seeds, GD lr=50, alpha=0.5, p=97):
        x2_plus_y2  3/3 grok, KM median 7000 [6900, 7000]  -> KEEP
        x2_y2_xy    0/3 grok (censored at 50k)             -> EXCLUDE
    The operation set in t1_replication therefore uses x2_plus_y2 and not
    x2_y2_xy, which is exactly what this gate is for.
    """
    return expand_grid(
        {"mode": "centralized", "p": 97, "alpha": 0.5, "hidden_width": 256,
         "activation": "quadratic", "optimizer": "gd", "lr": 50.0,
         "epochs": 50000, "log_every": 100},
        {"task": ["x2_plus_y2", "x2_y2_xy"], "seed": SEEDS3},
        tags={"tier": "T0", "group": "poly_pilot", "experiment": "pilot"},
    )


def t0_mnist_wd_band():
    """Locate the Omnigrok Goldilocks weight-decay band for MNIST-1k.

    The pipeline is verified (it reaches ~91% test), but a clean grok — fast
    memorisation then *delayed* generalisation — only appears in a narrow decay
    band: too weak and test never catches up within budget; too strong and there
    is no delay at all. This sweep over lr*wd (with large init) finds the band.
    Centralized, MSE, 3-layer MLP width 200, minibatch, 3 seeds.
    """
    return expand_grid(
        {"mode": "centralized", "dataset": "mnist", "model": "mlp",
         "hidden_width": 200, "n_layers": 3, "init_scale": 9.0,
         "loss": "mse", "optimizer": "adamw", "lr": 1e-3, "batch_size": 200,
         "n_train": 1000, "n_test": 5000, "epochs": 20000, "log_every": 100},
        # lr*wd in {1e-5, 3e-5, 1e-4, 3e-4, 1e-3}
        {"weight_decay": [0.01, 0.03, 0.1, 0.3, 1.0], "seed": SEEDS3},
        tags={"tier": "T0", "group": "mnist_wd_band", "experiment": "mnist"},
    )


def t3_server_lr_calibration():
    """Fair server-LR tuning for the server-optimiser strategies.

    The original exp5 tuned FedAdam's server LR but not the others', so its
    ~10x "speedup" was partly a tuning artifact. This sweeps server_lr for every
    tunable strategy on one representative heterogeneous cell, 3 seeds, so each
    method can be fixed at its own best LR before the main comparison. FedAvg /
    FedProx / SCAFFOLD have no server-LR knob (implicitly 1.0) and are the fixed
    references. FedAvgM additionally needs its momentum, swept lightly here.
    """
    cell = {"mode": "federated", "task": "addition", "p": 97, "alpha": 0.3,
            "hidden_width": 256, "num_clients": 10, "local_epochs": 5,
            "partition": "dirichlet", "dirichlet_alpha": 0.1,
            "num_rounds": FL_ROUNDS, "lr": 50.0, "eval_every": FL_EVAL_EVERY}
    specs = []
    for strat in ("fedadam", "fedyogi"):
        specs += expand_grid(
            {**cell, "strategy": strat},
            {"server_lr": [0.01, 0.1, 0.3, 1.0], "seed": SEEDS3},
            tags={"tier": "T3", "group": "server_lr_cal", "experiment": "cal",
                  "algorithm": strat},
        )
    specs += expand_grid(
        {**cell, "strategy": "fedavgm"},
        {"server_lr": [0.1, 0.3, 1.0], "server_momentum": [0.0, 0.9],
         "seed": SEEDS3},
        tags={"tier": "T3", "group": "server_lr_cal", "experiment": "cal",
              "algorithm": "fedavgm"},
    )
    return specs


# Setup A (the anchor): Gromov quadratic MLP, mod-97, MSE, full-batch GD.
SETUP_A = {"mode": "federated", "task": "addition", "p": 97, "alpha": 0.3,
           "model": "groknet", "hidden_width": 256, "loss": "mse",
           "optimizer": "gd", "lr": 50.0,
           "num_rounds": FL_ROUNDS, "eval_every": FL_EVAL_EVERY}


def t1_probe():
    """The early breakdown check — COMPLETED 2026-07-28, kept as a record.

    E values are left as they were actually run ({1,5,50,250}) so this manifest
    documents the executed sweep. Do NOT relaunch it: the E=250 cells were
    cancelled for cost and the E=1 cells proved under-budgeted, which is exactly
    why the going-forward spine is E_SPINE = {5,10,25,50} (see the E RANGE note
    at the top of this module). The T2 e-spine supersedes this.

    RESULT (18/24 completed, K=10, alpha=0.3, p=97, 3 seeds, KM median T_grok):
        E=1   iid/operand  0/3 grok  — censored by BUDGET, not by federation
                                       (10k steps available, ~12.9k needed).
                                       iid and operand were identical seed-for-
                                       seed: the E=1 FedAvg identity in the wild.
        E=5   iid 12900 / operand 12700   3/3
        E=50  iid 23000 / operand 17000   3/3
    Verdict: DELAY, not breakdown, at K=10 — grokking slows ~1.8x from E=5 to
    E=50 on the compute-matched step axis, but never fails. The breakdown search
    moves to higher K and stronger heterogeneity (the T2 K-sweep).
    """
    return expand_grid(
        {**SETUP_A, "num_clients": 10},
        {"local_epochs": [1, 5, 50, 250], "partition": ["iid", "operand"],
         "seed": SEEDS3},
        tags={"tier": "T1", "group": "probe", "experiment": "probe"},
    )


def t1_replication():
    """Tier 1: does the effect replicate across setups, tasks and moduli?

    Four blocks (MNIST is centralized-only — federated MNIST has no operand
    structure, so make_federated_datasets rejects it by design):
      - transformer @ mod-113 (Nanda config, CE)
      - S5 composition on both architectures, incl. the coset partition
      - prime ladder p in {53, 97, 113, 151} (finite-size scaling)
      - operation set (addition/subtraction/division/x2+y2) at p=97
    """
    specs = []

    # Transformer, Nanda config (CE + AdamW), mod-113.
    specs += expand_grid(
        {"mode": "federated", "task": "addition", "p": 113, "alpha": 0.3,
         "model": "transformer", "hidden_width": 128, "loss": "ce",
         "optimizer": "adamw", "lr": 1e-3, "weight_decay": 1.0,
         "num_clients": 10,
         "num_rounds": FL_ROUNDS, "eval_every": FL_EVAL_EVERY},
        {"local_epochs": [5, 25],
         "partition": ["iid", "operand", "dirichlet"], "seed": SEEDS3},
        tags={"tier": "T1", "group": "transformer", "experiment": "replication"},
    )

    # S5 composition. The coset partition needs exactly 5 clients (5 cosets of
    # S4), so it is a separate block from the K=10 iid/dirichlet cells.
    s5_base = {"mode": "federated", "dataset": "s5", "group_n": 5, "alpha": 0.5,
               "loss": "ce", "optimizer": "adamw", "lr": 1e-3, "weight_decay": 1.0,
               "num_rounds": FL_ROUNDS, "eval_every": FL_EVAL_EVERY}
    for model, width in (("groknet", 256), ("transformer", 128)):
        specs += expand_grid(
            {**s5_base, "model": model, "hidden_width": width, "num_clients": 10},
            {"local_epochs": [5, 25], "partition": ["iid", "dirichlet"],
             "seed": SEEDS3},
            tags={"tier": "T1", "group": "s5", "experiment": "replication"},
        )
        specs += expand_grid(
            {**s5_base, "model": model, "hidden_width": width,
             "num_clients": 5, "partition": "coset", "coset_subgroup": "s_nm1"},
            {"local_epochs": [5, 25], "seed": SEEDS3},
            tags={"tier": "T1", "group": "s5_coset", "experiment": "replication"},
        )

    # Prime ladder (finite-size scaling of the breakdown boundary).
    specs += expand_grid(
        {**SETUP_A, "num_clients": 10},
        {"p": [53, 97, 113, 151], "local_epochs": [5, 50],
         "partition": ["iid", "operand"], "seed": SEEDS3},
        tags={"tier": "T1", "group": "prime_ladder", "experiment": "replication"},
    )

    # Operation set at p=97. Multiplication is deliberately excluded (it is
    # cyclic Z_{p-1} in disguise — see DEGENERATE_TASKS).
    specs += expand_grid(
        {**SETUP_A, "num_clients": 10},
        {"task": ["addition", "subtraction", "division", "x2_plus_y2"],
         "local_epochs": [5, 50], "partition": ["iid", "operand"],
         "seed": SEEDS3},
        tags={"tier": "T1", "group": "operations", "experiment": "replication"},
    )
    return specs


def t2_phase_diagram():
    """Tier 2: the phase diagram on setup A — the paper's central grid.

    E-spine (the load-bearing axis), K-sweep in both disentangled forms,
    participation (the one FL knob with no centralized analogue), and the
    one-shot E->inf endpoint.
    """
    specs = []

    # E-spine: E from the exact centralized identity (E=1) out to near-independent.
    specs += expand_grid(
        {**SETUP_A, "num_clients": 10},
        {"local_epochs": E_SPINE,
         "partition": ["iid", "dirichlet", "operand"], "seed": SEEDS5},
        tags={"tier": "T2", "group": "e_spine", "experiment": "phase"},
    )

    # K-sweep, fixed TOTAL data (per-client shards shrink as K grows).
    specs += expand_grid(
        {**SETUP_A},
        {"num_clients": [5, 10, 20, 50], "local_epochs": [5, 50],
         "partition": ["iid", "dirichlet", "operand"], "seed": SEEDS5},
        tags={"tier": "T2", "group": "k_fixed_total", "experiment": "phase"},
    )

    # K-sweep, fixed PER-CLIENT data (alpha scales with K, so shard size is
    # constant). This disentangles "more clients" from "less data each" — the
    # confound flagged in the review.
    for k, alpha in ((5, 0.15), (10, 0.30), (20, 0.60), (50, 0.90)):
        specs += expand_grid(
            {**SETUP_A, "num_clients": k, "alpha": alpha},
            {"local_epochs": [5, 50], "partition": ["iid", "dirichlet", "operand"],
             "seed": SEEDS5},
            tags={"tier": "T2", "group": "k_fixed_per_client", "experiment": "phase"},
        )

    # Partial participation — no centralized analogue.
    specs += expand_grid(
        {**SETUP_A, "num_clients": 20},
        {"fraction_train": [0.2, 0.4, 0.6, 0.8, 1.0], "local_epochs": [5, 50],
         "partition": ["iid", "operand"], "seed": SEEDS5},
        tags={"tier": "T2", "group": "participation", "experiment": "phase"},
    )

    # One-shot FL: E -> infinity, R = 1. Each client trains to convergence
    # independently, then a single merge. The far endpoint of the E-spine, and
    # where frequency collision is most likely. This is the ONE cell where a
    # fixed round count makes no sense (R is 1 by definition), so E carries the
    # whole budget explicitly.
    specs += expand_grid(
        {**SETUP_A, "num_clients": 10, "num_rounds": 1,
         "local_epochs": ONE_SHOT_LOCAL_STEPS, "eval_every": 1},
        {"partition": ["iid", "dirichlet", "operand"], "seed": SEEDS5},
        tags={"tier": "T2", "group": "one_shot", "experiment": "phase"},
    )
    return specs


def t3_algorithm_comparison():
    """Tier 3: the FL algorithm comparison on the hard cells.

    Server-LR-tunable methods should be fixed at their calibrated LR (see
    t3_server_lr_calibration) before this is treated as final; the values here
    are placeholders that keep the grid runnable end to end.
    """
    hard_cells = [
        {"label": "H1", "alpha": 0.25, "num_clients": 10, "local_epochs": 25,
         "partition": "iid"},
        {"label": "H2", "alpha": 0.25, "num_clients": 10, "local_epochs": 25,
         "partition": "dirichlet", "dirichlet_alpha": 0.1},
        {"label": "H3", "alpha": 0.30, "num_clients": 10, "local_epochs": 50,
         "partition": "dirichlet", "dirichlet_alpha": 0.1},
    ]
    algos = [
        ("fedavg", {}),
        ("fedprox", {"proximal_mu": 0.01}),
        ("scaffold", {}),
        ("fedavgm", {"server_lr": 1.0, "server_momentum": 0.9}),
        ("fedadam", {"server_lr": 0.1}),
        ("fedyogi", {"server_lr": 0.1}),
    ]
    specs = []
    for cell in hard_cells:
        label = cell.pop("label")
        for strategy, kw in algos:
            specs += expand_grid(
                {**SETUP_A, **cell, "strategy": strategy, **kw},
                {"seed": SEEDS5},
                tags={"tier": "T3", "group": "algorithms", "experiment": "algo",
                      "setting": label, "algorithm": strategy},
            )
        cell["label"] = label
    return specs


def t2_k_breakdown():
    """The t1_probe follow-up: push K up, where breakdown is most likely.

    The probe found DELAY but no breakdown at K=10. Fragmenting the data across
    more clients is the axis most likely to actually break the circuit, so this
    is the K-sweep alone at E=5 (the cheap, known-good E), 3 partitions x 5
    seeds. It is a strict subset of t2_phase_diagram's k_fixed_total group, so
    ids match and running it means those cells are already done when the full
    T2 sweep is launched.
    """
    return expand_grid(
        {**SETUP_A, "local_epochs": 5},
        {"num_clients": [5, 10, 20, 50],
         "partition": ["iid", "dirichlet", "operand"], "seed": SEEDS5},
        tags={"tier": "T2", "group": "k_fixed_total", "experiment": "phase"},
    )


def t2_boundary():
    """The campaign that settles whether federation ever BREAKS grokking.

    Everything measured so far is delay, not breakdown: the alpha=0.3 plane
    groks 60/60 across K in {5..50}, and the probe found delay at K=10. The one
    standing breakdown claim comes from v1 exp2 at alpha=0.25, and it does not
    survive inspection.

    WHY THE OLD RESULT IS NOT EVIDENCE. exp2 ran `t_max=50000`. Its alpha=0.25,
    E=5, iid ladder (3 seeds, T_grok):

        K=2   23285 25315 27190      K=20  27215 29720 33025
        K=5   23625 25845 27880      K=50  40600 43975 CENSORED
        K=10  24450 26780 28790      K=97  CENSORED x3

    K=50 groks at 41-44k against a 50k budget. K=97's 0/3 is exactly what
    continued monotone delay looks like when it runs past the budget — the same
    trap as the E=1 probe cells, in the one place where censoring is supposed to
    BE the signal. Nothing there separates "federation broke the circuit" from
    "K=97 needs 60-90k steps".

    THE BUDGET. 20k rounds x E=5 = 100k gradient steps:
      - 2.3x the largest observed K=50 grok time (44k), so a censored K=97 sits
        well clear of its neighbour rather than 14% above it;
      - 1.33x the largest T_grok ever recorded at alpha<=0.25 (75425).
    Uniform across cells so every run shares one censoring time — a survival
    comparison across K is meaningless otherwise.

    THE CELLS.
      K=97 iid      the question.
      K=50 iid      proves the budget sufficed. If K=97 AND K=50 both censor,
                    the budget is still short and the sweep says nothing.
      K=20 iid      reproduces v1's 27-33k, confirming the v2 harness matches
                    exp2 before anything is concluded from a difference.
      K=97 operand  the rescue test. t2_k_breakdown found operand significantly
                    FASTER than iid at K=50 (13700 [13000,14000] vs 15200
                    [14600,16000]), and dirichlet tracking iid exactly — so it
                    is structure, not heterogeneity. Whether that advantage
                    survives to K=97 is a sharper question than the breakdown.
      Nearly free in wall-clock: the ten K=97 runs occupy slots the 1.2h K=20
      runs were never going to fill.

    5 seeds, not 1: grokking is stochastic (v1's K=50 cell was 2/3). Not 20
    either — seeds only buy CI separation where the grok fraction is partial,
    and which cell that is, is what this wave finds out. Deepening is wave 2.

    CHECKPOINTS ON. `checkpoint_every` is a config field and so feeds the run-id
    hash: enabling it later re-runs everything. The per-client W1 snapshots are
    what the frequency-consensus mechanism analysis needs, and this is the wave
    we least want to repeat. ~9.6 MB per checkpoint at K=97, 20 checkpoints per
    run, ~2.5 GB total.

    Verified before writing: all three partitions shard cleanly at alpha=0.25
    up to K=97 (smallest shard 7 samples, dirichlet; no empty shards).
    Cost, from the fitted model (9.8 + 1.291*K + 0.418*E) min at 10k rounds:
    K=20 1.2h, K=50 2.5h, K=97 4.5h per run -> ~7.5h wall-clock on 12 slots.
    """
    boundary = {**SETUP_A, "alpha": 0.25, "local_epochs": 5,
                "num_rounds": 20_000,
                "checkpoint_every": 1_000,
                "checkpoint_client_weights": True}
    specs = expand_grid(
        {**boundary, "partition": "iid"},
        {"num_clients": [20, 50, 97], "seed": SEEDS5},
        tags={"tier": "T2", "group": "boundary", "experiment": "boundary"},
    )
    specs += expand_grid(
        {**boundary, "partition": "operand", "num_clients": 97},
        {"seed": SEEDS5},
        tags={"tier": "T2", "group": "boundary", "experiment": "boundary"},
    )
    return specs


# ── The four new setups ──────────────────────────────────────────────────────
# Every federated result so far is SETUP_A. These are the setups built during the
# v2 rewrite and verified to grok centrally, but never run federated. Each is
# defined once here and referenced by every s5_* builder, so a setup's identity
# lives in exactly one place.
#
# `setup` is a TAG, not a config field, so adding it costs no run ids.

SETUP_B = {"dataset": "modular", "task": "addition", "p": 113,
           "model": "transformer", "hidden_width": 128, "n_heads": 4,
           "d_mlp": 512, "loss": "ce", "optimizer": "adamw", "lr": 1e-3,
           "weight_decay": 1.0}

SETUP_C = {"dataset": "s5", "group_n": 5, "model": "transformer",
           "hidden_width": 128, "n_heads": 4, "d_mlp": 512, "loss": "ce",
           "optimizer": "adamw", "lr": 1e-3, "weight_decay": 1.0}

SETUP_D = {"dataset": "s5", "group_n": 5, "model": "groknet",
           "hidden_width": 256, "loss": "ce", "optimizer": "adamw",
           "lr": 1e-3, "weight_decay": 1.0}

# MNIST: lr*wd = 1e-4 is the measured best band (t0_mnist_wd_band -- highest test
# accuracy, clean 3300-epoch delay). alpha is IGNORED for MNIST; its data-fraction
# axis is n_train.
SETUP_E = {"dataset": "mnist", "model": "mlp", "hidden_width": 200,
           "n_layers": 3, "init_scale": 9.0, "loss": "mse", "optimizer": "adamw",
           "lr": 1e-3, "weight_decay": 0.1, "batch_size": 200}

NEW_SETUPS = {"B": SETUP_B, "C": SETUP_C, "D": SETUP_D, "E": SETUP_E}


def s5_central_anchor():
    """STAGE 1: locate each new setup's own data cliff, centrally. Gate A.

    This is the cheapest and highest-value stage in the campaign, and it is a
    prerequisite rather than a formality, for two reasons.

    FIRST, THERE IS NO ANCHOR. No manifest has ever produced a centralized
    transformer or S5 run: the published grok times (transformer 6200, S5 ~14000)
    came from ad-hoc runs with no result JSON, and `results/data/runs/` contains
    zero of either. Federated delay is a RATIO, so without a pipeline-produced
    centralized T_grok there is no denominator.

    SECOND, alpha=0.25/0.3 IS SETUP-A'S BOUNDARY, NOT EVERYONE'S. Setup A's cliff
    is at alpha_c ~= 0.198 and its FL cells sit just above it. Where the cliff is
    for a transformer, for S5, or for MNIST is unknown. Placing FL cells before
    measuring it is how this project has twice produced a censored cell that meant
    nothing -- the E=1 probe cells and v1's K=97 breakdown claim, both budget
    artifacts (see the E RANGE note above and t2_boundary's docstring).

    Budgets are generous on purpose: a censored cell here must mean "past the
    cliff", not "past the clock".
    """
    specs = []

    # Setups B/C/D: the alpha ladder. Spans setup A's known boundary region so
    # the cliffs are directly comparable.
    for label, setup, epochs in (("B", SETUP_B, 30_000),
                                 ("C", SETUP_C, 40_000),
                                 ("D", SETUP_D, 40_000)):
        specs += expand_grid(
            {"mode": "centralized", **setup, "epochs": epochs, "log_every": 100},
            {"alpha": [0.15, 0.20, 0.25, 0.30, 0.40, 0.50], "seed": SEEDS5},
            tags={"tier": "S1", "group": "central_anchor",
                  "experiment": "anchor", "setup": label},
        )

    # Setup A on the same ladder, for a like-for-like reference measured by the
    # same harness rather than quoted from v1.
    specs += expand_grid(
        {"mode": "centralized", "task": "addition", "p": 97, "model": "groknet",
         "hidden_width": 256, "loss": "mse", "optimizer": "gd", "lr": 50.0,
         "epochs": 30_000, "log_every": 100},
        {"alpha": [0.15, 0.20, 0.25, 0.30, 0.40, 0.50], "seed": SEEDS5},
        tags={"tier": "S1", "group": "central_anchor", "experiment": "anchor",
              "setup": "A"},
    )

    # Setup E: alpha does not apply, so the data axis is n_train. The batch-size
    # block is NOT optional -- see s5_mnist_working_point below.
    specs += expand_grid(
        {"mode": "centralized", **SETUP_E, "n_test": 5000, "epochs": 20_000,
         "log_every": 100},
        {"n_train": [500, 1000, 2000, 4000], "seed": SEEDS5},
        tags={"tier": "S1", "group": "central_anchor", "experiment": "anchor",
              "setup": "E"},
    )
    return specs


def s5_mnist_working_point():
    """STAGE 1b: find an (n_train, batch_size) for MNIST that supports a K-sweep.

    At the Omnigrok grok point (n_train=1000) federated MNIST is degenerate. Every
    shard is n_train/K samples, so at batch_size=200:

        n_train  K    per client   batches per local epoch
           1000  10          100   1   <- batch_size is inert
           1000  50           20   1
           4000  10          400   2
           4000   5          800   4

    With one batch per local epoch, `local_epochs` stops meaning what it means on
    every other setup and the E axis is not comparable. So MNIST needs a working
    point that still groks centrally AND leaves >= 2 batches per local epoch at
    the campaign's largest K. That is what this measures.
    """
    return expand_grid(
        {"mode": "centralized", **{k: v for k, v in SETUP_E.items()
                                   if k != "batch_size"},
         "n_test": 5000, "epochs": 20_000, "log_every": 100},
        {"n_train": [1000, 2000, 4000], "batch_size": [25, 50, 100, 200],
         "seed": SEEDS3},
        tags={"tier": "S1", "group": "mnist_working_point",
              "experiment": "anchor", "setup": "E"},
    )


def s5_fl_probe():
    """STAGE 2: does FL run at all on each new setup, and what does it cost? Gate B.

    The cost model (9.8 + 1.291*K + 0.418*E) min/10k rounds is fitted on GrokNet
    alone. Wall-clock here is ~99% Flower/Ray orchestration, and orchestration is
    weight-shipping, so what matters is payload per client per round:

        GrokNet + modular   74,496 params    291 KB
        Transformer        225,792 params    882 KB   <- 3x
        GrokNet + S5        92,160 params    360 KB
        MLP + MNIST        199,210 params    778 KB

    Compute is ~6x for the transformer but is only ~1% of wall-clock, so the
    payload is the term to worry about. Rather than extrapolate, measure: this
    stage refits the coefficients per setup before anything expensive is planned.

    Also the first FL run of each new setup at a realistic K -- worth knowing
    before committing to a 200-run stage.
    """
    specs = []
    for label, setup in NEW_SETUPS.items():
        cell = {"mode": "federated", **setup, "num_rounds": 2_000,
                "local_epochs": 5, "eval_every": FL_EVAL_EVERY}
        if label == "E":
            # MNIST: no alpha, and operand/coset do not apply.
            cell.update({"n_train": 4000, "n_test": 5000})
            partitions = ["iid", "label_block"]
        elif label in ("C", "D"):
            cell["alpha"] = 0.5
            partitions = ["iid", "operand"]
        else:
            cell["alpha"] = 0.3
            partitions = ["iid", "operand"]
        specs += expand_grid(
            cell,
            {"num_clients": [10, 50], "partition": partitions, "seed": SEEDS3},
            tags={"tier": "S2", "group": "fl_probe", "experiment": "probe",
                  "setup": label},
        )

    # The AdamW confound, quantified. Rebuilding the optimizer every round is a
    # genuine no-op for GD at momentum=0 -- which is why setup A's E axis is clean
    # -- but under AdamW it makes every round E bias-corrected COLD-START Adam
    # steps. All four new setups use AdamW, so without this the new E results and
    # setup A's are not measuring the same quantity. 12 runs to find out.
    specs += expand_grid(
        {"mode": "federated", **SETUP_B, "alpha": 0.3, "num_clients": 10,
         "partition": "iid", "num_rounds": 2_000, "eval_every": FL_EVAL_EVERY},
        {"persist_local_opt_state": [False, True], "local_epochs": [5, 50],
         "seed": SEEDS3},
        tags={"tier": "S2", "group": "adam_restart", "experiment": "probe",
              "setup": "B"},
    )
    return specs


def x_d_alpha_fine():
    """TANGENT (not part of the staged campaign): a fine alpha ladder on setup D.

    Interstitial points between the Gate A ladder's rungs for the quadratic MLP
    on S5, which currently reads:

        a=0.5  7,200 | 0.4  12,600 | 0.3  21,300 | 0.25  36,200 | 0.2  0/5

    Adding 0.325/0.35/0.375, 0.425/0.45/0.475 and 0.525/0.55 takes that from 6
    rungs to 14, which is what a power-law fit of T_grok ~ (alpha - alpha_c)^-gamma
    needs to be worth quoting -- setup A's exponent (gamma ~ 0.99, alpha_c ~ 0.198,
    R^2 = 0.978) was fitted on a comparable density.

    Identical to the s5_central_anchor block for setup D in every other respect:
    same SETUP_D, same 40,000-epoch budget, same 5 seeds. Every new alpha is at or
    above 0.325, where the ladder already shows T_grok <= 21,300, so 40k leaves
    at least 2x headroom and nothing here should censor on the clock.

    Tagged tier "X" so it stays out of the Gate A / Stage 3 analysis by default.
    """
    return expand_grid(
        {"mode": "centralized", **SETUP_D, "epochs": 40_000, "log_every": 100},
        {"alpha": [0.325, 0.35, 0.375, 0.425, 0.45, 0.475, 0.525, 0.55],
         "seed": SEEDS5},
        tags={"tier": "X", "group": "d_alpha_fine", "experiment": "tangent",
              "setup": "D"},
    )


def x_d_alpha_high():
    """TANGENT: setup D's ladder extended up to alpha = 1.00 in 0.025 steps.

    Same cell as x_d_alpha_fine and the Gate A setup-D block in every respect --
    same SETUP_D, same 40,000-epoch budget, same 5 seeds -- so all three compose
    into one ladder. At these alphas T_grok is well under 5,000, so the budget is
    ample.

    NOTE on alpha = 1.00: alpha is the fraction of the grid used for TRAINING, so
    1.00 leaves the test set empty. It does not error -- compute_accuracy over
    zero samples returns NaN -- so the run completes and records t_grok = inf with
    a NaN test curve. Kept because it was asked for and because the train curve is
    still meaningful, but its test series is undefined by construction, not a
    measurement of failure. alpha = 0.975 already leaves only 360 test samples.
    """
    return expand_grid(
        {"mode": "centralized", **SETUP_D, "epochs": 40_000, "log_every": 100},
        {"alpha": [0.575, 0.6, 0.625, 0.65, 0.675, 0.7, 0.725, 0.75, 0.775,
                   0.8, 0.825, 0.85, 0.875, 0.9, 0.925, 0.95, 0.975, 1.0],
         "seed": SEEDS5},
        tags={"tier": "X", "group": "d_alpha_high", "experiment": "tangent",
              "setup": "D"},
    )


def x_d_alpha_cliff():
    """TANGENT: is setup D's cliff real, or is it the 40,000-epoch clock?

    The 14-rung ladder (x_d_alpha_fine + the Gate A block) is fitted far better
    by an exponential, T_grok ~ 10^(-2.78 alpha), than by the power law setup A
    follows -- R^2 0.983 on two parameters against 0.971 on three, and the power
    law's alpha_c runs to the bottom of its search range.

    Extrapolating that exponential to the two censored rungs:

        alpha = 0.20  ->  ~45,000 epochs
        alpha = 0.15  ->  ~62,000 epochs

    Both are ABOVE the 40,000-epoch budget they were run at. So their 0/5 is
    exactly what a smooth exponential looks like when it runs past the clock, and
    the apparent cliff between 0.20 and 0.25 may not exist at all. Reading it as
    an abrupt failure would repeat the mistake that cost v1 its headline claim.

    150,000 epochs -- 2.4x the exponential's prediction at the hardest rung -- and
    two intermediate rungs to see whether the curve simply continues.
    """
    return expand_grid(
        {"mode": "centralized", **SETUP_D, "epochs": 150_000, "log_every": 200},
        {"alpha": [0.15, 0.175, 0.2, 0.225], "seed": SEEDS5},
        tags={"tier": "X", "group": "d_alpha_cliff", "experiment": "tangent",
              "setup": "D"},
    )


def x_d_internals():
    """TANGENT: what is happening inside setup D across the alpha range 0.2-0.6?

    THE QUESTION. Read at fixed step counts rather than threshold crossings, the
    32-rung alpha ladder is two phases, not a family of different phenomena:

        alpha   0.30  0.375  0.40  0.45  0.50  0.55  0.60
        @1800    3.1   19.1  32.6  57.1  77.4  88.2  94.0    <- phase-1 plateau
        @3600    2.4   20.2  32.6  55.9  73.4  84.1  91.5    <- the dip
        T@95   27067  19400 17200 13960 11380  8920  6700    <- phase-2

    Phase 1 finishes by ~1,800 steps at EVERY alpha and lands on a plateau
    P(alpha) that is a sigmoid with midpoint ~0.47. The dip bottoms at ~3,600
    steps at every alpha. Phase 2's rate is smooth, T@95 ~ exp(-n/3190) over
    alpha in [0.3, 0.6] to 4.7% residual. So alpha's dramatic effect is almost
    entirely P(alpha); the apparent cliff is where P crosses the 85% grok bar and
    it moves when the bar moves (T@25 collapses at 0.40, T@50 at 0.45, T@85 at
    0.55). None of that says WHY.

    WHAT THIS RUN ADDS. The accuracy histories cannot answer it, because nothing
    logged separates a memorising circuit from a compositional one. Setup D's
    quadratic activation makes that separation exact rather than fitted -- see
    metrics/quadratic_circuits.py: logit = A[c,a] + 2T[c,a,b] + B[c,b], where A
    and B cannot compose by construction, so T IS the compositional circuit. And
    metrics/irreps.py gives S_5's exact analogue of the modular study's Fourier
    spectrum via the isotypic decomposition. Both are new, both log per eval, and
    NO banked run has them -- nor any saved weights, so this cannot be done
    post-hoc on the 160 ladder runs.

    A single pilot at alpha=0.55 already shows the measures move with the dip:
    the compositional circuit's participation ratio bottoms at 54 units at step
    1,800 (the plateau) and rises to 75 at step 3,600 (the trough), exactly
    anti-correlated with test accuracy. That is one seed at one alpha, which is
    what this manifest is for.

    BUDGET. Two blocks, because the low rungs are slow and the high ones are not:
    alpha <= 0.275 gets 150,000 epochs (alpha=0.25's T@85 is 35,140 and 0.225 is
    censored at 40k), the rest 40,000. log_every=25 throughout -- the dip is
    ~1,800 steps wide and the ladder's 100-step resolution smears its edges.
    checkpoint_every=500 so the post-hoc questions the time series cannot answer
    (CKA to the final representation, cumulative unit ablation) stay open without
    a re-run; at 369 KB a checkpoint that is ~4 GB, against 32 TB free.
    """
    slow = expand_grid(
        {"mode": "centralized", **SETUP_D, "epochs": 150_000,
         "log_every": 25, "checkpoint_every": 500},
        {"alpha": [0.2, 0.225, 0.25, 0.275], "seed": SEEDS5},
        tags={"tier": "X", "group": "d_internals", "experiment": "tangent",
              "setup": "D"},
    )
    fast = expand_grid(
        {"mode": "centralized", **SETUP_D, "epochs": 40_000,
         "log_every": 25, "checkpoint_every": 500},
        {"alpha": [0.3, 0.325, 0.35, 0.375, 0.4, 0.425, 0.45, 0.475, 0.5,
                   0.525, 0.55, 0.575, 0.6],
         "seed": SEEDS5},
        tags={"tier": "X", "group": "d_internals", "experiment": "tangent",
              "setup": "D"},
    )
    return slow + fast


def x_d_wd_ladder():
    """TANGENT: is setup D's fixed-epoch dip a weight-decay transient?

    THE OBSERVATION. Read at fixed step counts rather than at threshold
    crossings, every rung of the 32-point alpha ladder has the same shape: a
    fast rise that plateaus by ~1,800 steps, a dip bottoming at ~3,600, then a
    slow climb. The dip sits at the SAME step for every alpha, while T_grok over
    that ladder spans 300 to 35,000 steps -- two orders of magnitude. Whatever
    sets the dip's clock is therefore not the data.

    Test acc at fixed steps (5-seed means), showing plateau height P(alpha):

        alpha  0.30  0.375  0.40  0.45  0.50  0.55  0.60
        @1800   3.1   19.1  32.6  57.1  77.4  88.2  94.0
        @3600   2.4   20.2  32.6  55.9  73.4  84.1  91.5

    P(alpha) is a sigmoid with midpoint ~0.47, and it -- not any transition in
    the dynamics -- is what makes the ladder's curves look like different
    phenomena. The apparent alpha cliff moves with the grok threshold (T@25
    collapses at alpha~0.40, T@50 at ~0.45, T@70 at ~0.50, T@85 at ~0.55),
    i.e. it is just where P(alpha) crosses the chosen bar.

    THE HYPOTHESIS. The one clock in this setup that ignores alpha is decoupled
    weight decay: SETUP_D runs lr=1e-3, wd=1.0, so weights shrink by (1 - lr*wd)
    per step and the decay timescale is tau = 1/(lr*wd) = 1,000 steps. The
    observed plateau is at ~1.8 tau and the trough at ~3.6 tau.

    THE TEST. Move tau and see whether the dip moves with it.

        wd    lr*wd    tau      predicted trough (3.6 tau)
        0.0   0        none     no dip at all
        0.1   1e-4     10,000   ~36,000
        0.3   3e-4      3,333   ~12,000
        1.0   1e-3      1,000    ~3,600   (the banked ladder -- the control)
        3.0   3e-3        333    ~1,200

    Proportional movement confirms decay. No movement falsifies it cheaply and
    sends the search to the task instead. All five sit under the lr*wd <= 0.1
    band that check_decay_stability enforces.

    Three alphas spanning the sigmoid -- 0.30 (below the midpoint, dip masked by
    growth), 0.45 (mid-rise, deepest dip), 0.55 (near ceiling) -- so the same
    runs also show whether P(alpha) itself depends on decay, and whether phase
    2's rate constant (T@95 ~ exp(-n/3190) over alpha in [0.3, 0.6]) rescales.

    log_every=25 rather than the ladder's 100: at wd=3.0 the whole event is
    predicted to be over by step ~1,200, which 100-step resolution would smear.
    """
    return expand_grid(
        {"mode": "centralized", **SETUP_D, "epochs": 40_000, "log_every": 25},
        {"weight_decay": [0.0, 0.1, 0.3, 1.0, 3.0],
         "alpha": [0.30, 0.45, 0.55],
         "seed": SEEDS3},
        tags={"tier": "X", "group": "d_wd_ladder", "experiment": "tangent",
              "setup": "D"},
    )


def x_d_lr_control():
    """TANGENT: the confound control for x_d_wd_ladder -- decay clock or step size?

    x_d_wd_ladder varies wd at fixed lr, so it varies lr*wd. But lr also sets
    the step size, and a dip that moves with 1/lr would look identical to one
    that moves with 1/(lr*wd) in that ladder alone. The two are separated by
    holding lr*wd FIXED and varying lr:

        lr       wd     lr*wd    tau        step size vs control
        2.5e-4   4.0    1e-3     1,000      0.25x
        1.0e-3   1.0    1e-3     1,000      1x        (control)
        4.0e-3   0.25   1e-3     1,000      4x

    Products are exactly 1e-3 in all three, so tau is identical by construction.
    Predictions:
      dip at the same STEP in all three  -> the decay timescale sets the clock
      dip epoch scaling as 1/lr          -> it is the step size, and the wd
                                            ladder's result is coincidental

    Two alphas (0.45, 0.55) rather than three: this arm only has to localise the
    dip, which the ladder shows is deepest and cleanest above the P(alpha)
    midpoint.

    60,000 epochs rather than 40,000. At lr=2.5e-4 every timescale that is
    gradient-driven stretches 4x, and alpha=0.45's T@95 of ~13,960 at the control
    lr would land near 55,000. The dip is early either way, but the extra budget
    keeps phase 2 measurable in the slow arm so the same runs report whether the
    rate constant follows lr or lr*wd.

    lr=4e-3 is 4x the published grokking band and may be unstable on a quadratic
    activation. That is a real risk and an informative outcome rather than a
    failure -- divergence is visible in the loss curve, and the arm is 6 runs.
    """
    specs = []
    for lr, weight_decay in [(2.5e-4, 4.0), (1.0e-3, 1.0), (4.0e-3, 0.25)]:
        specs += expand_grid(
            {"mode": "centralized", **SETUP_D, "epochs": 60_000,
             "log_every": 25, "lr": lr, "weight_decay": weight_decay},
            {"alpha": [0.45, 0.55], "seed": SEEDS3},
            tags={"tier": "X", "group": "d_lr_control", "experiment": "tangent",
                  "setup": "D"},
        )
    return specs


def s5_setup_c_capacity():
    """GATE A follow-up: does setup C fail because of alpha, or because of capacity?

    C (transformer on S5) never groks reliably: 4/5 at alpha=0.5, 3/5 at 0.4, 0/5
    below. Two explanations, and they lead to opposite decisions.

    (a) THE CLIFF IS SIMPLY HIGHER. S5 is a harder function than modular addition
        -- 120 classes, non-abelian -- so it may just need more data. Tested by
        extending the ladder to alpha 0.6/0.7.

    (b) THE MODEL IS TOO SMALL. d_model=128 against 120 output classes is thin,
        and the suspicious part is that the *quadratic MLP* at width 256 handles
        the same task cleanly (setup D: 5/5 down to alpha=0.25). A transformer
        losing to a 2-layer MLP on the canonical grokking benchmark is more
        likely a capacity or tuning artifact than a fact about transformers.
        Tested by sweeping n_heads / d_mlp / hidden_width -- newly possible, since
        those only became Config fields in this session; a manifest setting them
        used to raise in build_config.

    100k epochs throughout: C's slowest observed grok is 33,100, so a censored
    cell here means "past the cliff", not "past the clock".

    Decision rule: C stays in the campaign if some configuration reaches 5/5 at a
    workable alpha with T_grok below a third of budget. Otherwise it is dropped
    and the campaign runs A/B/D/E -- which still separates architecture (B vs A)
    from task (D vs A); C is the interpolation cell, not a load-bearing one.
    """
    specs = expand_grid(
        {"mode": "centralized", **SETUP_C, "epochs": 100_000, "log_every": 200},
        {"alpha": [0.6, 0.7], "seed": SEEDS5},
        tags={"tier": "S1b", "group": "c_alpha", "experiment": "gate_a", "setup": "C"},
    )
    for width in (128, 256):
        specs += expand_grid(
            {"mode": "centralized", **SETUP_C, "hidden_width": width,
             "alpha": 0.5, "epochs": 100_000, "log_every": 200},
            {"n_heads": [4, 8], "d_mlp": [512, 1024], "seed": SEEDS3},
            tags={"tier": "S1b", "group": "c_capacity", "experiment": "gate_a",
                  "setup": "C"},
        )
    return specs


def s5_mnist_fl():
    """GATE A follow-up: federated MNIST at a config that actually groks.

    The first probe ran MNIST at (n_train=4000, batch=200) -- which the
    working-point sweep then showed has NO DELAY at all centrally: T_grok 500
    against memorisation at 500. So the probe measured federation on a setup that
    was not grokking to begin with, and its censoring says nothing about
    federation.

    The working-point sweep's finding is that delay and shardability oppose each
    other. A large memorise->generalise gap needs a large batch; a large batch on
    a shard of n_train/K samples degenerates to a single full-batch step, at which
    point `local_epochs` stops meaning what it means on every other setup. The
    two cells that have both a real delay and >= 2 batches per local epoch:

        (2000, 50)   delay 200,  8/4/2 batches at K = 5/10/20
        (2000, 100)  delay 500,  4/2/1 batches at K = 5/10/20

    Both are swept. batch_size is held FIXED across the K sweep within each arm --
    varying it with K to keep shards viable would confound the K axis with a
    change in the optimiser's effective batch, which is the very thing the delay
    depends on. The (100, K=20) cell is therefore degenerate by construction and
    is kept deliberately, as the control that shows what degeneracy looks like.

    4,000 rounds x E=5 = 20,000 steps, ~25x the centralized requirement.
    """
    return expand_grid(
        {"mode": "federated", **{k: v for k, v in SETUP_E.items() if k != "batch_size"},
         "n_train": 2000, "n_test": 5000, "num_rounds": 4_000, "local_epochs": 5,
         "eval_every": FL_EVAL_EVERY},
        {"batch_size": [50, 100], "num_clients": [5, 10, 20],
         "partition": ["iid", "label_block"], "seed": SEEDS3},
        tags={"tier": "S2b", "group": "mnist_fl", "experiment": "gate_a", "setup": "E"},
    )


def s5_probe_rerun():
    """GATE A follow-up: the probe cells that genuinely ran out of budget.

    Deliberately narrow. Of the probe's censored cells, only these three had
    MEMORISED and were waiting to generalise -- the rest sit at 1-5% train
    accuracy, which no amount of extra budget fixes (see s5_k50_diagnosis).

        D quad/S5  K=50 iid      99% train, 77-82% test against an 85 bar
        D quad/S5  K=10 operand  95% train, 13% test
        B tfmr/mod K=10 operand  50-96% train, 8-91% test -- one seed nearly made it

    10,000 rounds x E=5 = 50,000 steps, 5x the original probe and >2x each
    setup's centralized T_grok, so a cell that still censors here is reporting
    federation rather than the clock.
    """
    specs = expand_grid(
        {"mode": "federated", **SETUP_D, "alpha": 0.5, "num_clients": 50,
         "partition": "iid", "num_rounds": 10_000, "local_epochs": 5,
         "eval_every": FL_EVAL_EVERY},
        {"seed": SEEDS3},
        tags={"tier": "S2b", "group": "probe_rerun", "experiment": "gate_a", "setup": "D"},
    )
    specs += expand_grid(
        {"mode": "federated", **SETUP_D, "alpha": 0.5, "num_clients": 10,
         "partition": "operand", "num_rounds": 10_000, "local_epochs": 5,
         "eval_every": FL_EVAL_EVERY},
        {"seed": SEEDS3},
        tags={"tier": "S2b", "group": "probe_rerun", "experiment": "gate_a", "setup": "D"},
    )
    specs += expand_grid(
        {"mode": "federated", **SETUP_B, "alpha": 0.3, "num_clients": 10,
         "partition": "operand", "num_rounds": 10_000, "local_epochs": 5,
         "eval_every": FL_EVAL_EVERY},
        {"seed": SEEDS3},
        tags={"tier": "S2b", "group": "probe_rerun", "experiment": "gate_a", "setup": "B"},
    )
    return specs


def s5_k50_diagnosis():
    """GATE A follow-up: why does every AdamW setup fail to TRAIN at K=50?

    B, C and D all sit at 1-5% train accuracy at K=50 -- not memorised-but-not-
    generalising, but barely learning -- while setup A under plain GD groks 5/5 at
    the same K. So it is not client count by itself, and it is not budget: a model
    at 4% train after 10,000 steps, whose centralized twin reaches 100% train by
    epoch 200, is not short of steps.

    Two mechanisms already ruled out from the logged history:
      * weight-norm collapse (AdamW's decay is data-independent, so it survives
        averaging intact while gradient terms partially cancel) -- norms are flat
        or GROWING; D K=50 operand reaches a norm comparable to the grokking K=10
        run while sitting at 2% train;
      * client drift -- elevated but not decisive, and the failing D K=50 operand
        drifts LESS than the working D K=50 iid.

    So this sweep asks the two cheap remaining questions:

      LOCAL STEP SIZE. lr and wd are tuned for full-batch centralized training. A
      K=50 client holds ~76 samples and takes 5 full-batch steps on them; the same
      lr may simply be far too large at that shard size, so 50 clients overfit in
      opposite directions and the average is noise.

      WHERE IT BREAKS. K=10 works and K=50 does not. A ladder between them says
      whether this is a sharp transition (interesting) or a smooth degradation
      (a tuning gradient).

    Decision rule: if a lower local lr recovers training, this is a
    hyperparameter mismatch and Stage 3 must tune per-K or move adaptivity
    server-side (FedAdam/FedYogi are already implemented). If nothing recovers it,
    it is a candidate breakdown mechanism and becomes a headline rather than a
    nuisance -- with the per-client signature checkpoints as the evidence base.
    """
    cell = {"mode": "federated", **SETUP_B, "alpha": 0.3, "partition": "iid",
            "num_rounds": 2_000, "local_epochs": 5, "eval_every": FL_EVAL_EVERY}
    specs = expand_grid(
        {**cell, "num_clients": 50},
        {"lr": [1e-4, 3e-4, 1e-3], "weight_decay": [0.1, 1.0], "seed": [42]},
        tags={"tier": "S2b", "group": "k50_hparam", "experiment": "gate_a", "setup": "B"},
    )
    specs += expand_grid(
        cell,
        {"num_clients": [20, 30, 40], "seed": [42]},
        tags={"tier": "S2b", "group": "k50_ladder", "experiment": "gate_a", "setup": "B"},
    )
    return specs


def estimate_minutes(spec):
    """Rough wall-clock for one spec, used only to order a manifest.

    `launch_sweep.py` fills slots strictly FIFO in manifest order, so a long run
    that happens to sit last starts last and tails the whole sweep behind it.
    Emitting longest-first fills the short runs in behind the long ones instead;
    on t2_boundary that was worth ~18% (7.2h -> 5.9h).

    These constants are deliberately crude -- ordering only needs the ranking to
    be right, not the magnitude. Measured on an L4, 2026-07-29. Stage 2 replaces
    them with a per-setup fit; until then a transformer is charged ~3x a GrokNet
    because wall-clock here is ~99% weight-shipping and it ships ~3x the payload.
    """
    # The two modes are bottlenecked by DIFFERENT things, so they need different
    # model multipliers. Federated wall-clock is ~99% Flower/Ray weight-shipping,
    # which scales with parameter count: the transformer's 226k params against
    # GrokNet's 74k is the ~3x. Centralized wall-clock is per-step compute, where
    # the transformer's attention and MLP make it ~9x GrokNet -- measured 8.4
    # ms/step at p=113 against 1.3 ms/step. Using one multiplier for both
    # under-costs the centralized transformer threefold.
    if spec.get("mode") == "federated":
        payload_cost = {"transformer": 3.0, "mlp": 2.7}.get(spec.get("model"), 1.0)
        K = spec.get("num_clients", 5)
        E = spec.get("local_epochs", 5)
        rounds = spec.get("num_rounds", 10_000)
        return (9.8 + 1.291 * K + 0.418 * E) * rounds / 10_000 * payload_cost
    model_cost = {"transformer": 9.0, "mlp": 1.0}.get(spec.get("model"), 1.4)

    # Centralized cost tracks GRADIENT STEPS, not epochs. Under minibatching an
    # epoch is n_train/batch_size steps, so a batch-size sweep spans an order of
    # magnitude at a fixed epoch budget: MNIST at (n_train 4000, batch 25) is
    # 3.2M steps against 100k at (1000, 200) -- 32x. Costing by epochs alone
    # scored all of those equal and put the most expensive cell two-thirds of the
    # way down the file, which is precisely the tail-blocking this ordering
    # exists to avoid.
    epochs = spec.get("epochs", 10_000)
    batch = spec.get("batch_size", 0)
    n_train = spec.get("n_train", 1000)
    steps = epochs * max(1, -(-n_train // batch)) if batch else epochs
    return steps * 0.0009 / 60 * model_cost      # ~0.9 ms/step on an L4


def _longest_first(specs):
    return sorted(specs, key=estimate_minutes, reverse=True)


BUILDERS = {
    "x_d_alpha_fine": x_d_alpha_fine,
    "x_d_alpha_cliff": x_d_alpha_cliff,
    "x_d_alpha_high": x_d_alpha_high,
    "x_d_internals": x_d_internals,
    "x_d_wd_ladder": x_d_wd_ladder,
    "x_d_lr_control": x_d_lr_control,
    "s5_setup_c_capacity": s5_setup_c_capacity,
    "s5_mnist_fl": s5_mnist_fl,
    "s5_probe_rerun": s5_probe_rerun,
    "s5_k50_diagnosis": s5_k50_diagnosis,
    "s5_central_anchor": s5_central_anchor,
    "s5_mnist_working_point": s5_mnist_working_point,
    "s5_fl_probe": s5_fl_probe,
    "t0_wd_grid": t0_wd_grid,
    "t0_poly_pilot": t0_poly_pilot,
    "t0_mnist_wd_band": t0_mnist_wd_band,
    "t1_probe": t1_probe,
    "t1_replication": t1_replication,
    "t2_phase_diagram": t2_phase_diagram,
    "t2_k_breakdown": t2_k_breakdown,
    "t2_boundary": t2_boundary,
    "t3_server_lr_calibration": t3_server_lr_calibration,
    "t3_algorithm_comparison": t3_algorithm_comparison,
}


def main():
    os.makedirs(MANIFEST_DIR, exist_ok=True)
    which = sys.argv[1:] or list(BUILDERS)
    for name in which:
        if name not in BUILDERS:
            raise SystemExit(f"Unknown manifest {name}; options: {list(BUILDERS)}")
        specs = BUILDERS[name]()
        # Only the new staged manifests are reordered. The completed ones are
        # left byte-for-byte as they ran, so their record stays exactly as
        # executed -- and write_manifest would refuse to orphan their ids anyway.
        if name.startswith(("s5_", "x_")):
            specs = _longest_first(specs)
        path = os.path.join(MANIFEST_DIR, name + ".jsonl")
        write_manifest(specs, path)
        est = sum(estimate_minutes(s) for s in specs)
        print(f"{name}: {len(specs)} runs -> {os.path.relpath(path)} "
              f"(~{est / 60:.1f} slot-hours)")


if __name__ == "__main__":
    main()
