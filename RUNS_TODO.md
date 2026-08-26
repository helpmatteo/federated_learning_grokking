# RUNS_TODO

What still needs to be run, decided from scratch.

This supersedes every "Next" / "Outstanding" / "Stale" list in `PROGRESS.md`,
`RESULTS.md`, `plans/` and the `*_STATUS.md` files. Nothing carries over by
default — an item appears here only after it has been looked at explicitly and
kept.

Ground truth for what is already banked: `results/data/runs_v2.csv` and
`results/data/runs/*.json`.

---

## To run

### 1. Setup B at wd=0, federated — is the memorisation collapse about DECAY or about ADAMW?

> **SUPERSEDED IN ONE FIELD by entry 2 (2026-08-23): move α from 0.30 to 0.70.**
> At α=0.30, wd=0 cannot grok at any budget — B's decay band fits
> `t_first_cross ≈ 4,500/wd`, which diverges as wd→0, and the banked centralized
> control confirms it (2.5–5.9% test after 100,000 epochs). Entry 2 measured the
> α ladder: wd=0 groks **3/3 at α=0.70** with a **~28,850-step delay** intact.
> At α=0.70 the arm therefore answers the `t_memo(K)` question below *and*
> carries a `delay(K)` reading, instead of being a guaranteed null.
>
> The cost of moving: the banked wd=1.0 and wd=0.1 federated cells at K=10/20
> are at α=0.30, so the matched wd ladder has to be re-run at α=0.70 (~12 runs,
> cheap — everything is fast at that α) or the comparison is unmatched. The
> manifest `manifests/x_b_wd_zero_fl.jsonl` below is still at α=0.30 and has NOT
> been regenerated. Decide the α before launching it.

**Manifest** `manifests/x_b_wd_zero_fl.jsonl` — 9 specs, **6 to run** (the 3
centralized cells hash to runs already banked in `x_controls` and are skipped).

| | |
|---|---|
| setup | B (Nanda transformer, mod-113, CE+AdamW, lr 1e-3, width 128) |
| **weight_decay** | **0.0** |
| alpha | 0.30 |
| arms | centralized (banked) · federated **K=10** · federated **K=20** |
| E / partition / strategy | 5 · iid · FedAvg, full participation |
| seeds | 42, 123, 456 |
| budget | 20,000 rounds = **100,000 steps** (centralized arm: 100,000 epochs) |
| cost | **~8.6 slot-hours** — ~71 min/run at K=10, ~102 min/run at K=20, from the median `wall_s` of the matched banked cells. (`build_manifests` prints 19.4 h; its fitted model over-costs non-anchor setups ~2.6×.) |

**Why.** RESULTS §14.3's decay clock says memorisation blows up with K where
decay is decoupled and stays flat where it is absent — AdamW's decay is applied
per local step and does not scale with shard size, the learning signal from a
1/K shard does. The evidence is B/C/D/E (AdamW, wd>0) degrading against A (GD,
wd=0). **But A differs from B in the optimiser as well as the decay**, so
"AdamW" and "decoupled decay" are confounded in the one comparison the mechanism
rests on. B at wd=0 is AdamW with no decay clock, everything else fixed.

**Decision rule.** Read `t_memo(K)`, not `t_grok`.

- `t_memo` **flat in K** (≈ the centralized 200) while the wd=0.1 and wd=1.0
  ladders climb → the decay clock is about decay; §14.3 stands as written.
- `t_memo` **climbs with K anyway** → the collapse is a property of AdamW under
  fragmentation, and the paper has to say "adaptive optimiser", not "decay
  clock".

**These runs will NOT grok in budget, and that is the design, not a censoring
risk.** The banked centralized wd=0 curves end at 3.21 / 5.86 / 2.49% test
(chance 0.88, bar 95) with final-quarter slopes of +0.021 / +0.058 / +0.001
points per 1,000 steps. Extrapolated, the fastest seed needs **~1.55 M further
steps** to reach the bar and the slowest ~90 M. Centralized is the upper bound
for the federated arms on this setup, so a federated null at 100,000 steps
reproduces §15.4's centralized result rather than reporting a federated effect.
`t_memo` is the reading, and it cannot be censored here: at wd=0 it is bounded
above by the wd=1.0 value of 12,300 steps at K=20 (less decay memorises faster),
which 100,000 steps clears 8×.

**Do not trim the budget to the `t_memo` requirement.** 8,000 rounds would read
`t_memo` at ~3.5 slot-hours, but `num_rounds` is inside the content hash, so a
longer budget is a different run id — a short guess is re-run from scratch, not
resumed. It would also foreclose the one hypothesis under which these runs
return something other than a null: setup C's delay *collapses* with K and at
K=50 its first crossing precedes memorisation, so if averaging acts as an
implicit regulariser, B at wd=0 could generalise federated where it does not
centrally.

**Budget is set from the arm it must out-live, not from this arm's expectation.**
Banked `t_first_cross` at wd=0.1 reaches 77,200 (K=10) and 98,200 (K=20).
Anything shorter converts an honest null into the ninth censored boundary in
this project. 100,000 steps also matches the wd=1.0 aggregation arm and the
centralized wd=0 arm, so all three decay levels are read at one budget.

**What it completes** — setup B, α=0.30, iid, E=5, 3 seeds, banked `t_memo` range
and grokked fraction:

| wd | centralized `t_memo` / `t_fc` | K=10 | K=20 |
|---|---|---|---|
| 1.0 | 150 / 45,050 | 4,800–8,000 · 3/3 | 6,100–12,300 · 3/3 |
| 0.1 | 150 / 4,350 | 1,300–1,400 · 3/3 | 3,500–4,100 · 3/3 |
| **0.0** | **200 / never** | **← this** | **← this** |

```bash
setsid nohup venv/bin/python -u scripts/launch_sweep.py \
    manifests/x_b_wd_zero_fl.jsonl --gpus 0 --per-gpu 4 \
    > logs/sweeps/b_wd_zero_$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null &
```

Not a working-point change: B's published config stays wd=1.0, and new
transformer work stays at the 2026-08-20 wd=0.1 decision. This is a tier-X
mechanism control.

---

### 2. Setup B at wd=0 — a centralized α ladder. **DONE**

`manifests/x_b_wd_zero_alpha.jsonl` — **15/15, 0 failures, 289 min wall**,
2026-08-23. B centralized, wd=0.0, 100,000 epochs, `log_every=50`, 3 seeds.

**ANSWER: yes, α makes B grok with no weight decay. The working point is α=0.70.**

| α | grokked | median `t_first_cross` | delay over `t_memo` | vs wd=0.1 | final test per seed |
|---|---|---|---|---|---|
| 0.4 | 0/3 | — | — | — | 13.8 · 28.3 · 14.1% |
| 0.5 | 0/3 | — | — | — | 43.1 · 69.6 · 74.6% |
| 0.6 | 1/3 | 95,300 | 95,150 | 30× | 95.5 · 84.5 · 93.9% |
| **0.7** | **3/3** | **29,000** | **28,850** | 19× | 99.0 · 96.5 · 97.4% |
| 0.8 | 3/3 | 600 | 450 | — | 98.0 · 99.3 · 98.2% |

`t_memo` = **150 at every α and every seed**, so the whole α effect is on the
delay and none of it is a training difference. Per-seed `t_first_cross` at
α=0.7 is 3,900 / 29,000 / 31,200 and at α=0.8 is 450 / 600 / 1,650.

**Why 0.70 and not 0.80.** The decision rule was the *lowest* α that groks 3/3,
because a federated K effect needs a delay to act on. α=0.8 collapses the delay
to **450 steps** — §13.3's A′ trap, "no delay left for federation to disrupt".
α=0.7 keeps ~28,850, which is 19× what wd=0.1 has at the same α precisely
*because* nothing is driving the transition.

**The sub-threshold rungs are censored, not decided.** Final-quarter slopes and
the steps still needed to reach the bar: α=0.5 seed 123 at +0.604 pts/1k (~42k
short), α=0.6 seed 123 at +0.223 (~47k short), α=0.6 seed 456 at +0.126 (~8.5k
short). Every cell at α≥0.5 was still climbing at the ceiling, so α=0.6 would
likely be 3/3 at ~150,000 epochs. The α boundary is a smooth transition the
budget cuts across, not a cliff — **do not quote α=0.7 as "the threshold"**,
quote it as the lowest rung that groks 3/3 *within 100,000 epochs*.

**Read `t_first_cross`, never `t_grok`, on this ladder.** α=0.8 seed 42 crosses
at 600 and records `t_grok` = 91,200; α=0.7 seed 42 crosses at 3,900 and records
65,300. Post-crossing dips dominate `t_grok` here exactly as §14.4 describes.

**Side finding, still unacted.** B's decay band fits `t_first_cross ≈ 4,500/wd`
at α=0.30, which puts `p1_b_decay_band`'s wd=0.03 and wd=0.01 cells at ~150,000
and ~450,000 steps required against the **50,000** they were given. They are
censored, not decided, and RESULTS §13.5's "sharp threshold below wd=0.1" is a
clock artifact.

---

### 3. Setup B at wd=0, α=0.40 to 1,000,000 steps — the low-α anchor. **DONE**

`manifests/x_b_wd_zero_a04_long.jsonl` — **3/3 completed, 0 failures, 386.7 min
wall** (6.44 h/run, against a 8.8 h estimate). 2026-08-24.

**RESULT: 0/3, and the pre-registered prediction was WRONG.** The prediction on
record was 1/3–2/3 — seed 123 "safe" at ~283,000 from a linear extrapolation of
its accelerating slope. What happened instead:

| seed | test @100k | @300k | @500k | @1M | plateau |
|---|---|---|---|---|---|
| 42 | 13.8% | 39.8% | 70.1% | **70.2%** | flat from ~500k |
| 123 | 28.3% | 56.1% | 63.4% | **63.7%** | flat from ~400k |
| 456 | 14.1% | 36.3% | 60.7% | **70.2%** | flat, one late step at ~850k |

**Why: the gradient dies, and with wd=0 nothing else moves the weights.** Train
accuracy is 100% from step 150 and stays there, so `train_loss` decays toward
zero and reaches **exactly 0.0 in float32** at step **233,600 / 212,450 /
257,450**. After that there is no gradient, and at wd=0 there is no decay term
either, so the model is at a fixed point:

| seed | weight-norm range after 500k | test range after 500k |
|---|---|---|
| 42 | **0.0000** | 70.1–70.2% |
| 123 | 0.0007 | 63.3–63.7% |
| 456 | 2.2198 | 53.9–70.3% |

Two seeds froze completely. Seed 456 kept a little residual drift — the loss
oscillates between 0 and the ~2.33e-11 float32 floor and Adam normalises by
`sqrt(v)`, so denormal gradients can still produce steps — and it gained ~10
points late. That is the exception that shows the rule.

> **This is the first negative result in this project that is NOT a clock
> running out.** Eight apparent boundaries have dissolved into censoring on
> re-measurement. This one does not: at wd=0 the run does not slow down, it
> *stops*, and no budget reaches the bar. The lesson for the method is that
> extrapolating a wd=0 curve from a pre-saturation window overestimates badly —
> the process has a hard stop the early window cannot see. Every slope-based
> forecast in this file inherits that caveat.

**What it means for the science.** Three things fall out of one mechanism:

- **It explains `t_first_cross ≈ 4,500/wd`.** Once the training loss dies, decay
  is the only remaining force, so the rate of post-memorisation travel is
  proportional to wd and the requirement diverges as wd→0. The law and this
  plateau are the same fact seen twice.
- **It reframes what α buys.** Grokking without decay is a **race between
  generalisation and gradient death**. α sets who wins: at α=0.4 the model
  reaches 60–70% before the loss underflows around ~230,000 steps and is then
  frozen there; at α≥0.6 it crosses the bar first.
- **The plateau is a measurable quantity** — the generalisation available "for
  free" from the memorising solution at a given α. ~70 / 64 / 70% at α=0.40.

**Caveat to state if this is published.** The hard stop is partly a *precision*
fact: CE on a perfectly memorised training set decays toward zero and float32
makes it exactly zero. In float64, or with label smoothing or a loss floor, the
gradient would persist and the plateau might not be exact. The mechanism —
nothing drives the weights once the loss dies and decay is off — is real either
way, but "no budget suffices" is a claim about this numerical setup.

**Consequence for entry 2's follow-up, and it survives.** The α=0.6 rung's two
uncrossed seeds need ~8,700 and ~47,000 more steps (to ~109k and ~147k total),
while gradient death arrives around 200,000–260,000. The curve wins that race,
so **α=0.6 → 150,000 epochs is still a good bet for 3/3** — and now for a
mechanistic reason rather than an extrapolated slope. At 100k every ladder run
still had `train_loss` ~1e-8, three orders above the floor, with weight norms
drifting 2–4.8 units per 25k steps, so none of them had stalled.

---

## Decided against

**α=0.30 and α=0.40 to 250,000 / 200,000 steps** (10.7 slot-h). Every one of the
six cells still censors at those budgets: α=0.30 needs 1.65M–88M steps and
α=0.40's best seed needs ~283,000 against the 200,000 proposed. Cost without a
result.

**α=0.30 to 1,000,000 steps** (21.6 slot-h). 0/3 by extrapolation, and no seed
is accelerating. Would only be worth buying as a deliberate negative result at
B's own campaign working point — "wd=0 is dead at α=0.30 even at 1M" as a stated
limitation. Not currently needed.
