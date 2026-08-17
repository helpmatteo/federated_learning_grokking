# Plan: the multi-setup campaign

*Supersedes this file's earlier contents (the setup/check phase), which is **complete** —
see the closeout at the end. Companion plans, both closed: `plan-all-that-needs-nested-seal.md`
(boundary campaign), `gate-a-closeout.md` (Gate A).*

## Context

The goal is to run v1's exp0→exp7 chain — the work on `main` — across the six v2 setups.
v1's chain was **870 runs on one setup**, and v2 cells run 2–10× longer, so replicating
all of it everywhere is ~7,300 slot-hours: 25 days of continuous 12-slot compute. That is
not the plan. The scope below covers the two experiments that carry the project's headline
claims, on every setup, plus repairing the anchor's two defective experiments.

**Decisions taken:**

| | |
|---|---|
| Scope | Re-scoped **exp2** + **exp3b** on all setups, plus the anchor redo |
| exp2 grid | Working α × K ∈ {5,10,20,50} × 3 arms × 3 seeds = **36 cells/setup** |
| Seeds | **3** (v1 parity) |
| Anchor redo | **Both** exp4b and exp5 |
| exp4a / exp4c / exp5 on new setups | **Out of scope**, stated as a limitation |

**The one correction that changed the costing.** Setup C was believed to need α≥0.5 and
≥200k-step budgets. That rested on a 0/5 at α=0.3 measured at width **128** with 40,000
epochs — while at that same width α=0.5 has a KM median of 39,800. The ladder was cut off
*at the median for an easier α*, so the cell was censored, not cliffed. C has **never** been
run below α=0.5 at width 256, where its first-crossing median is ~16,100. Part 0 measures
this; the costs below assume α≈0.3 works and drop C from ~5.6 to ~2.2 h/cell.

---

## Part 0 — Prerequisites

Nothing in Parts 1–3 can be budgeted or specified without these.

| # | Item | Cost |
|---|---|---|
| 0.1 | **Wire `reduced_arm()` into exp2.** It is written, tested, and `arm`/`reduced_from_k` are already in `TAG_KEYS` and `PREFERRED_COLUMNS` — the plumbing is complete end to end and **no manifest calls it**, so exp2 currently has two arms, not three. `scripts/build_manifests.py:940` | code only |
| 0.2 | **C's α ladder at width 256** — α ∈ {0.25, 0.3, 0.4, 0.5, 0.6} × 3 seeds, 40,000 epochs, wd=1.0. Settles C's actual working point and whether it can be matched to everyone else's 0.30 | ~2 h |
| 0.3 | **exp0 capacity for A′/B/D/E** — hidden_width ∈ {½, 1, 2}× the setup's default × 3 seeds, centralized. Only C has ever had one | ~3 h |
| 0.4 | **`t1_setup_k_ladder`** (written, 39 runs) — supplies `t_memo(K)` and `delay(K)` per setup, which is where every exp2/exp3b budget comes from. Also the first per-client checkpoints on B/C/D/E, closing exp7's data gap | ~88 h |
| 0.5 | **`t3_server_lr_calibration`** (written, 42 runs, unrun) — must precede exp5 or it reproduces v1's unfairness defect | ~17 h |

> **Gate.** 0.2 decides whether C joins the matched-α comparison or stays an existence
> proof at its own α. 0.4 decides every federated budget below. Do not write Parts 1–3's
> manifests until both have landed.

---

## Part 1 — exp2, re-scoped · ~112 slot-h

The aggregation experiment: **does averaging compensate for fragmenting the data?** Three
arms per cell, which is the whole point and which no v2 manifest has ever had.

Per setup: **working α** (its own, from Gate A / §13) × **K ∈ {5,10,20,50}** × 3 seeds.

| arm | what it is | how |
|---|---|---|
| ceiling | centralized on the full training set | mostly **already banked** from Gate A; dedupes by content hash |
| **floor** | one model on one client's shard | `reduced_arm(fed_specs)` — dataset-aware (α/K for grid datasets, `n_train`/K for MNIST), 2× budget so its failure is attributable to data and not to the clock |
| FL | FedAvg, iid, E=5 | 12 runs/setup, the only expensive arm |

### Why the floor arm — and what it is *not* for

**It was condition (b) of the original experiment.** `experiments/exp2_aggregation.py` runs
exactly three conditions per (α, K): *(a) Centralized-full, (b) Centralized-reduced with
n_train/K, (c) FL (IID)*. Drop it and this is not exp2, it is a two-arm subset — and the
comparison against v1's numbers stops being like-for-like.

**But v1's headline use of it was misleading, and this plan does not repeat it.** That
headline was "FL groks in 23 of 30 cells where the floor groks in 1." Read carefully, most
of that gap is just *FL sees K times more data than one shard* — it is not evidence about
aggregation. So the going-forward headline compares **FL against the ceiling on the
compute-matched step axis**, which is the comparison that actually isolates what averaging
does.

**What the floor is genuinely for:** it is the only arm that answers *could a client have
done this alone?* — which is the question a federated result is practically answering, and
the one a reviewer asks first. It is also the cheapest arm by a wide margin: centralized
runs on 1/K of the data, ~6 of Part 1's 112 slot-hours. Keeping it costs ~5% and closes an
obvious objection; dropping it saves almost nothing and invites that objection.

This reasoning is already recorded at `scripts/build_manifests.py:940`.

Dropped from v1's grid: the α sweep. exp1 covers α centrally, `t2_k_breakdown` showed the
α=0.3 plane uniformly safe, and K's cost is now known to be set by `t_memo(K) + delay`
rather than by proximity to the cliff.

**Budgets are `t_memo(K) + delay` with headroom above the estimate, never a multiple of the
centralized T_grok** — and which of the two terms carries the K dependence is
**setup-dependent** (on A memorisation is flat and the delay grows; on B memorisation
explodes). Take both from 0.4 per setup.

New builder `t2_aggregation` in `scripts/build_manifests.py`, one block per setup,
`checkpoint_every` + `checkpoint_client_weights=True` throughout.

---

## Part 2 — exp3b, structured partitions · ~148 slot-h

The project's strongest claim is that **coherent shards beat random ones, and the gap grows
with K** — established on the anchor alone. This tests whether it is a property of grokking
or of one architecture.

Per setup: working α × **K ∈ {10, 50}** (the effect scales with K, so one K cannot see it)
× 3 seeds × the partitions that setup supports:

| setup | partitions |
|---|---|
| A, A′, B | iid · operand · dirichlet |
| C, D (S₅) | iid · operand · dirichlet · **coset** (K must equal the coset count exactly — 5 for S₄, 2 for A₅, so coset is its own block) |
| E (MNIST) | iid · dirichlet · label_block — `operand`/`coset` correctly raise, MNIST has no operand structure |

Verified: all six partition modes construct cleanly on A/B/C/D at K=10; MNIST rejects
exactly the two it should.

---

## Part 3 — the anchor redo · ~275 slot-h

**"Anchor redo" = re-running two of v1's own experiments on setup A**, the anchor. Nothing
to do with the new setups. Both were run on `main`, both are in `results/data/runs.csv`, and
both are **defective for reasons unrelated to the v2 rewrite** — so they cannot be cited as
they stand and cannot serve as the baseline the new setups are compared against.

Verified against the banked v1 data:

### exp4b, partial participation — 72 cells, ~88 h. The x-axis is wrong.

Before Phase 0.6 (`094a291`) the step counter was `total_steps = server_round *
local_epochs`. That is the **depth of the update chain**, and it equals gradient work only
at full participation. Under `fraction_train < 1.0` just a fraction of the clients — and so
of the data — trains each round, but the old formula charged every round as if all K had.

At `fraction_train=0.4` that over-counts by ~2.5×. **54 of exp4b's 72 runs use
`fraction_train` ∈ {0.2, 0.4, 0.6}**, so their T_grok values sit that far to the right of
where they belong. The models trained fine; the axis they are plotted on is not the axis
every current run uses, so the two cannot appear on one chart.

The fix split the quantity in three: `total_steps` (centralized-equivalent gradient work),
`sequential_steps` (rounds × E), and `n_participating` per round. **A retrospective rescale
is not safe**: it would need the realised per-round participation, and `n_participating` is
exactly what was not logged before the fix — so the correction factor cannot be verified
against the runs it would be applied to. Re-running is the honest path.

### exp5, algorithms — 153 cells, ~187 h. Three independent defects.

**1. The weight-decay arm was arithmetically dead before training started.** Both SGD
(coupled) and AdamW (decoupled) shrink weights by `(1 − lr·λ)` per step, so `lr·λ` is the
meaningful quantity and `1/(lr·λ)` the decay timescale. exp5 ran **lr=50 × wd ∈ {0.01, 0.1,
1.0}** → `lr·λ` ∈ {0.5, 5, 50} → per-step multipliers of **0.5, −4, −49**. At 0.5 every
weight halves on every step: a decay timescale of 2 steps against the ~10⁴ needed to grok.
At 5 and 50 the sign flips and the magnitude explodes. All nine cells reporting `T_grok =
inf` was forced by arithmetic, not measured. `core/utils.py::check_decay_stability` now
raises on precisely these values.

**2. Only one method was tuned.** `server_lr` was swept {0.01, 0.1, 1.0} for **FedAdam
alone**; FedProx and FedAvg have no server-LR knob and ran at defaults. So the reported ~10×
FedAdam speedup is partly "the tuned method beat the untuned ones". This is what
`t3_server_lr_calibration` (0.5) exists to remove, and why it must run first.

**3. The per-round histories for the compared arms were overwritten.** The
`_adam_tau*_slr*` and `_wd*` filename suffixes landed *partway through* the campaign
(`3796754`, 2026-03-23). Before that, FedAvg, FedAdam-0.1, FedAdam-0.01 and FedAvg+WD-* all
wrote **identical history paths** within the same (setting, seed) cell — 61 paths were
written by more than one run, and each surviving JSON holds only whichever finished last.
`harvest_logs.py` recovers scalar outcomes by taking algorithm identity from the *log*
filename, which is why `runs.csv` rather than the JSONs is the v1 record. But the
trajectories are gone, and the arms that collided are exactly the algorithm comparison exp5
exists to make.

Defect 3 is the decisive one: 1 and 2 affect which numbers are trustworthy, but 3 means the
underlying curves cannot be re-analysed at all.

Both run on setup A at v1's grid, so results drop straight onto the existing comparison.

**If compute gets tight, this is the block to cut.** It is 43% of the campaign and buys
repair rather than new ground; Parts 1–2 are what extend coverage to the new setups.

---

## Cost

| | A | A′ | B | C | D | E | total |
|---|---|---|---|---|---|---|---|
| Part 0 prerequisites | | | | | | | **110** |
| Part 1 exp2 | 15 | 15 | 33 | 26 | 15 | 3 | **112** |
| Part 2 exp3b | — | 22 | 50 | 46 | 26 | 4 | **148** |
| Part 3 anchor redo | 275 | | | | | | **275** |
| | | | | | | | **~645 slot-h** |

≈ **2.5 days** continuous on 12 slots. Against 7,333 slot-h for the full matrix.

Rates are measured this session, not the built-in estimator (which is fitted on setup A and
over-costs transformers ~2.6×): transformer ~0.50 s/round, GrokNet ~0.22, MLP ~0.20.

---

## Verification

1. `venv/bin/python -m pytest tests/ -q` → 537 pass. Add: `reduced_arm` is reachable from a
   real manifest and its cells carry `arm="cent_reduced"` through to the CSV.
2. `scripts/validate_manifest.py` on every new manifest — it already catches empty shards,
   degenerate batch sizes, `hidden_width % n_heads`, invalid partitions, duplicate ids.
3. `--dry-run` each manifest: confirm ceiling-arm cells **dedupe** against banked Gate A
   runs rather than re-running them.
4. **Sanity on the science:** exp2's setup-A K=20 cell must reproduce ~29,800; if it does
   not, the harness has drifted and nothing else is interpretable.
5. Launch detached (`setsid nohup … > logs/sweeps/<name>.log`) — a sweep dies with its shell
   otherwise, which cost this session 11 runs at 0/11.

---

## Future work — deferred, not abandoned

These are real experiments with manifests that could be written; they are sequenced after
the campaign rather than cut. Each is costed from the matrix.

| | scope | cost |
|---|---|---|
| **exp4a — drift × heterogeneity on the new setups** | A′/B/C/D/E | ~550 slot-h |
| **exp4c — compute vs communication on the new setups** | A′/B/C/D/E | ~275 slot-h |
| **exp5 — algorithm comparison on the new setups** | A′/B/C/D/E | ~1,170 slot-h. Narrower there than on the anchor: **SCAFFOLD raises under AdamW by design**, so drift correction is FedProx-only on B/C/D/E |
| **exp7 — mechanistic checkpoints on the new setups** | A′/B/C/D/E | ~565 slot-h, but Part 0.4 already banks the first per-client weights on B/C/D/E, so the analysis can start without new runs |
| **exp3a — the Dirichlet sweep** | new setups | ~764 slot-h. Lowest priority: `t2_k_breakdown` showed Dirichlet tracks iid at every K on the anchor, and Part 2 carries a Dirichlet arm on every setup as its control |
| **The mechanism analysis** | no new compute | 400 per-client W1 snapshots from `t2_boundary` already on disk test the frequency-consensus explanation for the partition-structure result |
| **K=97 IID at 200k steps** | setup A | ~45 slot-h. The one cell that could still be a real breakdown; both successes landed within 5% of the ceiling |
| **The setup-D "dip"** | tier X | ~1 slot-h. `x_d_wd_ladder` (33 runs) + `x_d_lr_control` (18) are written and test whether a fixed-step dip at every α is a decay transient |

## Out of scope — state in the paper

DP-FedAvg, communication compression, Byzantine-robust aggregation, personalization,
async/stragglers, FedBN, MOON/FedDecorr, LEAF/FLamby/FedScale. **D′** is dropped — the K
gate opened without it, so A-vs-D stays confounded and is recorded as a limitation rather
than fixed with a seventh setup.

---

## Closeout: the setup/check phase (complete)

Delivered: Phase 1's four decision rules read (RESULTS §13); `p1_b_decay_band` added and it
caught a censored cell being read as a breakdown; `p1_k_collapse_budget` run — **there was
no K≈30 collapse**, and the K axis is open; `t_first_cross`/`post_grok_dips` added because
`t_grok` measures the logging rate on an unstable setup; the per-client checkpoint path
verified end-to-end off GrokNet and made a permanent test; `t1_setup_k_ladder` written;
README rewritten; 537 tests green; 842 runs banked.

Two rules came out of it and govern everything above:

> **Budget as `t_memo(K) + delay`, with headroom above the estimate.** Six — now seven —
> boundaries in this project have been manufactured by a budget set from the centralized
> number instead.
>
> **Which term carries the K dependence is setup-dependent.** Measure it per setup; a table
> of `T_grok` shows one number for both mechanisms and hides the difference.
