# exp3a's Dirichlet ladder on the other five setups

## Context

main's `exp3a_t_grok_vs_dir_alpha.png` (`main:scripts/plot_exp3.py:108`) plots
T_grok against `dirichlet_alpha` over the ladder {0.01, 0.1, 0.5, 1.0, 10.0,
1000.0} at K=10, one line per training fraction. v2 has that reading for
**setup A only** — `t3a_dirichlet_band` (24 runs, α=0.25, K ∈ {20, 50}), written
up as RESULTS §18.1–18.2, plus its `dirichlet_sizes` control in
`t3a_size_control`.

This plan extends the ladder to A′, B, C, D and E so the figure becomes six
per-setup panels, matching the shape `paper/exp2_slowdown_ratio_*.png` already
uses.

**Setup A needs nothing run.** Its ladder is banked and analysed; it is the panel
the other five are being measured against, and §18.2's starvation finding is the
reason this plan chooses K=10 rather than reusing A's K ∈ {20, 50}.

## The decision that shapes the grid: K=10, not K=20 or K=50

§18.2 established that A's ladder below dir_α ≈ 0.1 was measuring **client
starvation**, not heterogeneity: failure tracked the smallest shard seed-for-seed
and survived randomising the labels. `dirichlet` shards over target classes, so
concentrating labels on a dataset with few samples per class necessarily empties
someone's shard.

Partitions built directly (not assumed), at each setup's own working point:

| setup | classes | n_train | min shard @ dir_α=0.01, K=10 | classes/client | verdict |
|---|---|---|---|---|---|
| A | 97 | 2,823 | 116 | 15 | full ladder |
| A′ | 97 | 1,882 | 120 | 16 | full ladder |
| B | 113 | 3,831 | 215 | 19 | full ladder |
| C | 120 | 5,760 | 291 | 20 | full ladder — best conditioned |
| D | 120 | 4,320 | 176 | 18 | full ladder |
| E | **10** | 2,000 | **empty shard — raises** | — | 0.01 infeasible |

At K=10 the smallest shard on the five algebraic setups is ≥116 samples, an order
of magnitude clear of the ≤2 that killed cells in §18.2. **The confound that
required `dirichlet_sizes` beside every low rung is not in play at this K**, so
this plan does not carry a size-control arm for A′/B/C/D. K=10 is also v1's own
`k_primary`, and it is the one K at which all six setups have a banked iid
control that groks.

E is the exception on both counts and is handled below.

## Verified dedup — the `dir_α=0.5` rung is free, but only if written bare

`t3b_partitions` already ran a `dirichlet` cell on every setup at its working
point and exp2 budget. That cell **is** the 0.5 rung: `FedConfig.dirichlet_alpha`
defaults to 0.5 and the banked specs **omit the field entirely**.

Run ids are content hashes of the spec *as written*, so adding `dirichlet_alpha:
0.5` to reproduce that cell changes its id and re-runs banked work — the exact
hazard `manifest.orphaned_ids` documents. Checked by construction: writing the
rung explicitly deduplicates **0 of 210**; writing it bare deduplicates **3/3 at
every (setup, K) that `t3b_partitions` ran**.

> **The builder must therefore emit the 0.5 rung as a spec with no
> `dirichlet_alpha` key**, and say so in its docstring. Verify before launching:
> the manifest should report **87 specs, 72 to run, 15 banked** — the 15 being the
> 0.5 rung on each of the five setups.

Each setup consequently needs **5 new rungs × 3 seeds = 15 runs**, not 18.

## What to run

K=10, `local_epochs=5`, `partition="dirichlet"`, 3 seeds (`SEEDS3`), each setup at
**its own exp2 working point and budget** — the same values `t3b_partitions` used,
which is what makes the 0.5 rung and the iid control dedup.

| setup | working point | `num_rounds` | new rungs | runs | h/run | slot-hours |
|---|---|---|---|---|---|---|
| **A** | α=0.30 | 10,000 | — | **0** | — | **0** — banked, §18.1 |
| A′ | α=0.20 | 20,000 | 0.01, 0.1, 1.0, 10, 1000 | 15 | 0.96 | 14.4 |
| B | α=0.30 | 20,000 | same 5 | 15 | 1.56 | 23.4 |
| C | α=0.40, w256 | 40,000 | same 5 | 15 | 3.40 | 51.0 |
| D | α=0.30 | 50,000 | same 5 | 15 | 2.87 | 43.1 |
| E | n_train=2000 | 8,000 | 0.1, 1.0, 10, 1000 | 12 | 0.56 | 6.7 |
| | | | | **72** | | **~139** |

Hours are measured `wall_s` from each setup's `t3b_partitions` K=10 `dirichlet`
run — the identical config, differing only in `dirichlet_alpha`. Banked on the L4;
they transfer per the README's hardware note.

**Wall-clock on this machine: ~35 h** at `--gpus 0 --per-gpu 4`. K=10 is well
inside the range where four concurrent runs cost nothing per-run, so no
`FEDGROK_GPU_CLIENT_CAP` is needed.

If A's panel is wanted at the same working point as the other five rather than at
α=0.25, that is a further **15 runs / 9.9 slot-h** — but it is a second reading of
a question already answered, and the α=0.25 ladder is the harder and more
informative one.

### Optional second rung — each setup's exp3b partner K

`t3b_partitions` ran a second K per setup (A′ 20, B 20, C 50, D 20, E 20), so the
0.5 rung and the matched operand/target arms dedup there too. **72 runs, ~220
slot-h, ~55 h wall**, again with A excluded — its second rung is the K ∈ {20, 50}
ladder already banked at α=0.25.

Not recommended as part of this plan. C at K=50 is already 0/3 at dir_α=0.5 and
would return a censored panel for 109 of those slot-hours. And the high-K rungs
run into the same instrument limit A hit: at K=50 the empty-shard guard rejects
dir_α=1000 outright on A and A′, and 10.0 leaves a 2-sample client on A — which is
§18.2's starvation regime, not a heterogeneity measurement. K=10 avoids all of it,
which is the point.

## Two readings to state up front

**D is the panel most likely to come back censored, and that is a result.** D at
K=10 is **0/3 at dir_α=0.5** while its iid control at the same K groks 3/3 at
78,900 steps. Its budget is not the constraint — 3.2× headroom on the iid cell —
so D is genuinely heterogeneity-sensitive where A is not. Expect its panel to be a
fraction-grokked plot rather than a T_grok curve, and report it as the order
parameter per the project's standing convention.

**E's ladder moves two variables at once.** With 10 classes and `batch_size=100`,
concentration changes the effective batch size per client as well as the label
mix: shards run 5–400 at K=10, and the training loop's `range(0, n_local, bs)`
silently gives short final batches rather than failing. Same species of instrument
problem as §18.2 (partitioner moves size and labels together) and §16.1
(instrument = treatment). Either state it in the docstring, or pair every E rung
with a `dirichlet_sizes` control — **+12 runs, ~7 slot-h**, which is cheap enough
that it is probably worth taking.

## Implementation

1. **New builder `t3a_dirichlet_ladder()` in `scripts/build_manifests.py`.** A new
   manifest, not an edit to `t3a_dirichlet_band` — that one is complete and
   `write_manifest` will refuse to orphan its ids. Reuse `t3b_partitions`'s
   `BLOCKS` shape verbatim so the working points and budgets cannot drift; tag
   `group: "dirichlet_ladder"`, `experiment: "exp3a"`. Emit the 0.5 rung bare, per
   the dedup section, and carry the K=10 decision rule in the docstring.
2. **Plotting.** A per-setup panel consumer under `scripts/plotting/`, x =
   `dirichlet_alpha` on a log axis, y = `t_first_cross` (not `t_grok` — §14.4:
   budgets and logging rates differ across setups), censored cells marked and
   partial censoring labelled `n/3` as the exp2 panels already do.
3. **Execution.**
   ```bash
   venv/bin/python scripts/build_manifests.py t3a_dirichlet_ladder
   venv/bin/python scripts/validate_manifest.py manifests/t3a_dirichlet_ladder.jsonl
   setsid nohup venv/bin/python -u scripts/launch_sweep.py \
       manifests/t3a_dirichlet_ladder.jsonl --gpus 0 --per-gpu 4 \
       > logs/sweeps/dir_ladder_$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null &
   ```
   Order longest-first (C and D dominate). The alpha2 sweep has the card until it
   finishes.

## Verification

1. **Dedup before spending anything** — 87 specs, **72 to run, 15 banked**, and
   the 0.5 rung 0 of 15. If the 0.5 rung does not dedup, the builder is writing
   `dirichlet_alpha` explicitly; fix that rather than launching.
2. **Partition feasibility at build time**, as was done here: every (setup, K,
   rung) must construct without raising, and the min shard should be recorded
   alongside the result so a censored cell can be attributed to starvation or not.
3. **After the sweep:** `collect_runs.py`, then check that each setup's 0.5 rung
   reproduces the value §17.2 already reports for it — if it moves, the ladder is
   not sitting on the same config as `t3b_partitions` and the dedup was cosmetic.
4. Write up as a RESULTS §18.1 extension, stating for each setup whether the knob
   is flat down to 0.1 as it is on A, and whether any setup's failure tracks its
   smallest shard the way §18.2's did.
