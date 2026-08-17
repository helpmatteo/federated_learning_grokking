# Plan: the boundary campaign

*Supersedes the v2 build plan. Phases 0a–5 of that plan are **code-complete** — 364 tests
green, four setups verified to grok, 9 manifests written, 123 runs executed. The per-phase
done-list lives in `PROGRESS.md`; it is not repeated here.*

## In plain terms

**The question:** grokking is when a model memorises its training data, looks like a failure
for a long time, then suddenly generalises. Does that still happen when training is split
across many machines that average their weights together?

**What we've found so far:** federation doesn't break grokking, it slows it down. Today's
sweep ran 5 → 50 machines, 60 runs, and every one grokked — more machines cost ~16% more
time and nothing worse. Side-finding: splitting data *by structure* groks faster than
splitting it randomly, and the gap grows with machine count.

**The problem:** the project was aiming at the claim *"at 97 machines, grokking breaks."*
That came from an earlier experiment given a budget of 50,000 steps. But at 50 machines
grokking takes 41,000–44,000 steps. So if 97 machines simply needs 60,000–70,000, the old
experiment would look exactly the way it does. **The clock ran out — that isn't the same as
failing.** The central claim has no evidence behind it either way.

**The plan:** re-run 97 machines with double the budget (100,000 steps), 20 runs, ~7.5 hours.
Either it groks — the old breakdown was an under-powered experiment, itself a reportable
finding — or it doesn't, and we have a real result to build the rest of the project on. The
supporting work is small: switch on weight-saving so the *why* can be analysed later (the
code exists, no experiment ever enabled it), close two places where results from different
experiments could be silently pooled, and run cheap checks before spending the 7.5 hours.

## Context

The project needs one thing to be publishable: **a breakdown boundary with a mechanism.**
"FedAvg preserves grokking" is forced by the E=1 identity and a reviewer kills it in a line.

Two results from the completed sweeps changed what the campaign has to do.

**1. The α=0.3 plane is dead, and cleanly so.** `t2_k_breakdown` finished 60/60 with zero
censoring: K=5→50 across three partitions, every cell 5/5. A 10× increase in client count
buys ~16% delay and nothing else. This is a good control arm, and it means the boundary
search must move to α=0.25.

**2. v1's headline breakdown is a budget artifact, not evidence.** exp2 ran `t_max=50000`.
The α=0.25, E=5, iid ladder:

| K | 2 | 5 | 10 | 20 | 50 | 97 |
|---|---|---|---|---|---|---|
| T_grok | 23–27k | 24–28k | 25–29k | 27–33k | **41k, 44k, cens** | **all cens** |

K=50 groks at 41–44k against a 50k budget. K=97's 0/3 is exactly what continued monotone
delay looks like when it runs past the budget. Nothing in this data distinguishes "federation
broke the circuit" from "K=97 needs 60–90k steps" — the same trap as the E=1 probe cells,
in the one place where censoring is supposed to *be* the signal.

**Intended outcome:** a budget large enough that a censored K=97 means something, enough
seeds to call a transition, and the checkpoints needed to explain it — decided before the
runs rather than after.

## Two constraints discovered while planning

- **The fixed-per-client-n control is geometrically impossible at the boundary.** Holding
  shard size at the α=0.25/K=10 value (235/client) needs α=1.25 at K=50 and α=2.42 at K=97.
  The control the old plan leaned on to rule out "shards are just too small" cannot be run
  at the K values that matter. Replacement: an **α-ladder at fixed K=97**, deferred behind
  the decision gate (below).
- **Cost, measured from the 60 completed runs:** `minutes = 9.8 + 1.291·K` at 10k rounds.
  At the corrected 20k-round budget: K=20 → 1.2 h, K=50 → 2.5 h, **K=97 → 4.5 h/run.**

## Decisions taken

| Decision | Choice |
|---|---|
| Budget | **20k rounds = 100k steps.** 2.3× K=50's grok time, 1.33× the largest T_grok ever seen at α≤0.25 (75.4k) |
| Seeding | **Staged** — 5 seeds to find the transition, then 15 more only where the grok fraction is partial |
| α-ladder control | **Deferred** behind the gate; only run if K=97 actually censors |
| E | **5**, matching the v1 ladder this extends |
| Checkpoints | **On in wave 1** — `checkpoint_every` is a config field, so it changes run ids; adding it later re-runs everything |

---

## Phase A — Fixes

All small, all in code that already exists. **Checkpointing itself is already implemented**
(`training/federated.py:451-520` saves global weights, Fourier spectrum, and per-client W1;
`core/fed_config.py:35-36` has the flags) — no manifest has ever set it. That is a manifest
fix, not a code fix.

| # | Fix | File |
|---|---|---|
| A.1 | **`num_rounds` into the survival cell keys.** `DEFAULT_CELL_KEYS` lacks it, so a 10k-round cell and a 20k-round cell with the same (α,K,E,partition) pool into one cell with two different censoring times. This campaign creates exactly that collision. | `scripts/summarize_runs.py:27-31` |
| A.2 | **Result-row schema gap.** The row omits `dataset`, `model`, `loss`, `batch_size`, `init_scale`, `checkpoint_every`, `group_n`, `coset_subgroup`, `server_momentum`. T1 replication runs groknet *and* transformer on S5 under the same `group` tag — currently separable only by `hidden_width`, by accident. Six lines, and far cheaper now than after 150 runs. | `src/fedgrok/run.py:71-104` |
| A.3 | **Wire `coset_attribution` into per-round history**, behind `nonabelian_applicable(cfg)`, mirroring the existing `fourier_applicable` guard at `federated.py:478`. NaN elsewhere. Not needed for wave 1 (modular), needed before T1 replication — do it during wave 1. | `training/federated.py`, `training/centralized.py` |
| A.4 | **New `t2_boundary()` manifest builder** (below). Record the exp2-budget finding in the E RANGE note so the reasoning is not re-derived. | `scripts/build_manifests.py` |

### The wave-1 manifest

```python
BOUNDARY = {**SETUP_A, "alpha": 0.25, "local_epochs": 5,
            "num_rounds": 20_000,          # 100k steps
            "checkpoint_every": 1_000,     # 20 checkpoints/run
            "checkpoint_client_weights": True}

# the ladder — extends v1's exp2 ladder at a budget where censoring means something
expand_grid({**BOUNDARY, "partition": "iid"},
            {"num_clients": [20, 50, 97], "seed": SEEDS5})          # 15 runs

# the rescue test — does structured heterogeneity survive where iid does not?
expand_grid({**BOUNDARY, "partition": "operand", "num_clients": 97},
            {"seed": SEEDS5})                                        # 5 runs
```

**20 runs.** The operand block is nearly free in wall-clock: the ten K=97 runs fit inside the
12 slots together, so it costs slots the short runs were not going to use. It also tests the
most interesting hypothesis the project has produced — the completed K-sweep found operand
significantly *faster* than iid at K=50 (13,700 [13,000, 14,000] vs 15,200 [14,600, 16,000]),
and whether that advantage rescues K=97 is a sharper question than the breakdown itself.

---

## Phase B — Smoke tests (before committing 4.5 h/run)

| # | Check | Why |
|---|---|---|
| B.1 | `venv/bin/python -m pytest tests/ -q` → **364 pass** (~10.5 min) | baseline |
| B.2 | **Partition smoke, CPU only, no training.** `make_federated_datasets` for α=0.25 × K∈{20,50,97} × {iid, operand, dirichlet}. At K=97 that is 2,352 samples over 97 clients ≈ 24 each — `partition.py:80-85` **raises** on an empty shard | catches a hard failure in seconds instead of 4.5 h in |
| B.3 | **Checkpoint smoke.** One K=97 run, 200 rounds, `checkpoint_every=50`. Assert files land in `results/runs/<id>/checkpoints/`, per-client W1 has shape (97, 256, 97), and `analysis/mechanistic.py::load_checkpoints` reads them back | the checkpoint path has never been exercised at K=97 or through the launcher |
| B.4 | **Timing check** — extrapolate B.3's wall time; confirm the 4.5 h/run projection before launching | the cost model is fitted on K≤50 and extrapolates 2× beyond its data |
| B.5 | **Trajectory equivalence** after A.3 touches `evaluate_fn` — `traj.py` vs `ref_run1.json`, worst \|Δ\| < 1e-5 | this harness has already caught two real bugs |
| B.6 | **Storage + dry run.** Projected ~2.5 GB of checkpoints (9.6 MB/ckpt at K=97 × 20 × 10 runs); confirm `results/runs/` is gitignored. `launch_sweep.py manifests/t2_boundary.jsonl --dry-run` → 20 runs, no id collisions | |

---

## Phase C — Wave 1

```bash
venv/bin/python scripts/build_manifests.py t2_boundary
venv/bin/python scripts/launch_sweep.py manifests/t2_boundary.jsonl --gpus 0,2,4,5,6,7 --per-gpu 2
venv/bin/python scripts/collect_runs.py
venv/bin/python scripts/summarize_runs.py results/data/runs_v2.csv \
    --group alpha,num_clients,local_epochs,partition,num_rounds
```

**~7.5 h wall-clock** on 12 slots (the ten K=97 runs are the tail; the K=20/50 runs fill in
behind them).

---

## Phase D — Decision gate

Wave 1 has three possible outcomes and they lead to different papers. Read the gate before
spending anything else.

| Outcome | Reading | Wave 2 |
|---|---|---|
| **K=97 groks** (any seed ≤100k) | v1's breakdown was a budget artifact. Say so plainly — it is a finding, not a failure | **No boundary campaign.** Contribution becomes the delay law + the structured-heterogeneity acceleration. Skip the α-ladder. Go to T1 replication |
| **K=97 censors 0/5, K=50 groks** | Real breakdown candidate, with a 2.3× separation instead of v1's 1.14× | Run the **α-ladder control**: K=97, α∈{0.30, 0.40, 0.50}, 5 seeds (~9 h). Without it, censoring is not attributable to federation rather than to data-per-client. Then deepen seeds on the transition cell |
| **K=97 partial (1–4 of 5)** | That cell *is* the transition | Deepen to **20 seeds on that cell only** (15 runs, ~7 h) → CI width 0.40. Plus the α-ladder |

The operand-at-K=97 block reads across all three: if operand groks where iid censors, the
mechanism story is "structured heterogeneity preserves a shared frequency basis," and the
per-client W1 checkpoints from wave 1 are exactly what tests it.

---

## Phase E — Downstream (unchanged, gated on D)

Order and contents as previously agreed; all manifests are already written.

| manifest | runs | note |
|---|---|---|
| `t0_mnist_wd_band` | 15 | the one setup not yet confirmed to grok — cheap, run it alongside wave 1 |
| `t1_replication` | 150 | needs A.2 and A.3 landed first, or S5/transformer rows are ambiguous |
| `t2_phase_diagram` | 415 | `t2_k_breakdown`'s 60 cells already skip |
| `t3_server_lr_calibration` → `t3_algorithm_comparison` | 42 → 90 | **calibration first** — running the comparison uncalibrated reproduces the exp5 unfairness defect |
| T4 mechanism | ~60 | frequency consensus from wave-1 checkpoints; drift intervention; norm-clamp |

---

## Verification

1. **Unit:** 364 tests stay green. Add: `t2_boundary` specs carry `num_rounds=20000` and
   `checkpoint_every=1000`; `summarize_runs` puts different-`num_rounds` cells in different
   groups (A.1 regression); the result row contains the A.2 fields.
2. **Numerical:** `traj.py` equivalence after A.3 — worst \|Δ\| < 1e-5 against `ref_run1.json`.
3. **Pipeline:** B.3's checkpoints load through `analysis/mechanistic.py` and yield a
   per-client Fourier spectrum — i.e. the mechanism analysis is actually executable on what
   wave 1 saves, verified before wave 1 rather than after.
4. **Sanity on the science:** wave 1's K=20 cell should reproduce v1's 27–33k at 5 seeds. If
   it does not, the harness diverged from exp2 and nothing else in the campaign is
   interpretable.

## Out of scope (state in the paper)

DP-FedAvg, communication compression, Byzantine-robust aggregation — deferred follow-up.
Also: personalization (optimises per-client models; this studies one global model),
async/stragglers, FedBN (no BatchNorm here), MOON/FedDecorr (need a representation layer),
LEAF/FLamby/FedScale (no known grokking regime).
