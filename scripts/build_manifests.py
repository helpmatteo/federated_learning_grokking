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


BUILDERS = {
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
        path = os.path.join(MANIFEST_DIR, name + ".jsonl")
        write_manifest(specs, path)
        print(f"{name}: {len(specs)} runs -> {os.path.relpath(path)}")


if __name__ == "__main__":
    main()
