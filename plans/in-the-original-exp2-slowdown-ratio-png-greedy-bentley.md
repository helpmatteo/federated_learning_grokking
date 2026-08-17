# Multi-α slowdown-ratio figures: what needs to be run

## Context

`paper/exp2_slowdown_ratio_*.png` (committed `274d496`) ports main's
`exp2_slowdown_ratio.png` to six per-setup panels. main's original carried **four
lines per panel**, one per α ∈ {0.25, 0.30, 0.35, 0.50}; v2's panels carry one,
because `t2_aggregation` deliberately dropped the α sweep — each setup runs at
its own working point instead (builder docstring: *"DROPPED FROM v1's GRID: the
alpha sweep"*).

This plan adds a **second α per setup** so every panel gets two lines. Decisions
taken by the user: two α per setup across all six, the second α **easier** (more
data), stepping **one rung** on each setup's own measured ladder rather than
jumping to 0.50.

## What to run

Every value is on a measured centralized ladder, so none is a guess. Baselines
are median `t_first_cross` from the banked rows:

| setup | banked α | **new α** | centralized baseline at new α | note |
|---|---|---|---|---|
| A | 0.30 | **0.40** | 8,800 (5/5) | |
| A′ | 0.20 | **0.30** | 300 (5/5) | see caveat |
| B | 0.30 | **0.40** | 1,400 (5/5) | |
| C | 0.40 | **0.50** | 15,200 (15/15) | forced — C is 1/3 at 0.30, 0/3 at 0.25 |
| D | 0.30 | **0.40** | 12,600 (10/10) | |
| E | n_train 2000 | **n_train 4000** | ladder in `central_anchor` | |

**K per setup** mirrors the existing block: {2,5,10,20,50}, and {2,5,10,20} for E
(at n_train=4000 the K=50 shard is 80 against batch=100 — below one batch, the
same degeneracy that caps E today). 3 seeds (`SEEDS3`), matching exp2.

### One caveat, scoped to A′

Stepping one rung keeps a workable baseline on A (8,800), B (1,400), C (15,200)
and D (12,600). **A′ is the exception**: at α=0.30 it crosses at 300 steps, so
with E=5 federation gets 60 rounds to act. §13.3 made this argument directly —
*"there is no delay left to disrupt"* — and A′ already reads 13.3× at K=2 at its
working point from optimiser restart alone (§15.3). A′'s second line will
therefore be dominated by that fixed cost divided by a short baseline, not by α.
There is no better option: A′ is 0/5 at α=0.175, so it has no harder rung, and
every easier rung is faster still (0.25 → 500, 0.40 → 200). Recording it so the
panel is not over-read.

### Budgets

**Federated: reuse each setup's existing exp2 `num_rounds` unchanged** —
A 10,000 · A′ 20,000 · B 20,000 · C 40,000 · D 50,000 · E 8,000. They were sized
for the harder working α so they are generous here, which is the safe direction,
and holding C's fixed is what lets its cells dedup. They are over-provisioned at
the easier α — D's 50,000 rounds against a 12,600-step baseline is the extreme —
so if cost binds, D is the one to trim, but not before seeing first crossings.

**Centralized:** use exp2's `cent_epochs` per setup, except where matching a
banked budget buys a dedup at ample headroom:

| setup | cent epochs | why |
|---|---|---|
| A | 50,000 | exp2 value |
| **A′** | **5,000** | matches `aprime_alpha`; 16× headroom over 300 → **3 free** |
| B | 100,000 | exp2 value |
| **C** | **100,000** | matches `c_capacity`; 6.6× over 15,200 → **3 free**, and halves C's 1.1 h/run |
| D | 250,000 | exp2 value |
| E | 40,000 | exp2 value |

A, B, D and E also have banked cells at the new α, but at `log_every` values the
builder's `max(10, epochs//500)` does not reproduce. Overriding `log_every` to
force a match would save ~1.7 slot-hours total and add a special case per setup —
not worth it, and `log_every` is not inert (§13.4).

**Arms:** `fl` + `cent_full` only. The `cent_reduced` floor is not needed — the
ratio is FL / cent_full, and main's figure never used the floor.

### Verified dedup — 18 of 105 specs already banked

Checked by exact config comparison, not assumed:

- **C at α=0.50, K ∈ {5,10,20,50}** — `setup_k_ladder` holds these at R=40,000,
  E=5, w256, wd=1.0, `eval_every`=20, `checkpoint_every`=4,000,
  `checkpoint_client_weights`=True, seeds {42,123,456}. exp2's C block builds
  `checkpoint_every = rounds//10` = 4,000 with the same flags, so mirroring it
  reproduces those ids. **12 fl runs free; only K=2 is new.**
- **A′ centralized at α=0.30** (`aprime_alpha`, 5,000 ep, `log_every`=10) and
  **C centralized at α=0.50** (`c_capacity`, 100,000 ep, `log_every`=200) —
  **6 cent runs free.**

No federated dedup exists for A, A′, B, D or E at the new α: their easier-α
federated cells (`fl_probe`, `probe_rerun`, `grok_confirm_fl`) all sit at
R=2,000–10,000, which does not match.

### Cost, from measured `wall_s`

Per-round rates measured per setup and K from banked federated runs — not the §8
fitted model, which over-costs non-anchor setups ~2.6×.

| setup | new fl runs | new cent runs | slot-hours |
|---|---|---|---|
| A | 15 | 3 | 9.1 |
| A′ | 15 | 0 *(dedup)* | 18.3 |
| B | 15 | 3 | 24.9 |
| C | **3** *(dedup)* | 0 *(dedup)* | 6.9 |
| D | 15 | 3 | 48.2 |
| E | 12 | 3 | 6.2 |
| **total** | **75** | **12** | **~114** |

D dominates, as it does in exp2 — its 50,000-round budget at K=50 is 6.2 h/run.

## Implementation

### 1. New manifest builder — `scripts/build_manifests.py`

Add `t2_aggregation_alpha2()`. **A new manifest, not an edit to
`t2_aggregation`**: `write_manifest` refuses to drop ids an existing manifest
claims, and `build_manifests.main()` deliberately leaves completed manifests
byte-identical ("left byte-for-byte as they ran"). Adding cells to the existing
file would rewrite a completed sweep's record for no reason.

Reuse the existing machinery rather than re-implementing it:

- copy `t2_aggregation`'s `BLOCKS` shape — `(label, base, wp, rounds,
  cent_epochs, Ks)` — changing only the working-point override and `cent_epochs`
- `expand_grid`, `SEEDS3`, `FL_EVAL_EVERY`, `SETUP_*`, `_FED_ONLY` as-is
- same tags, with `group: "aggregation_alpha2"` so the cells are selectable
  separately, and `arm` tags `fl` / `cent_full` so the plotting code's arm split
  keeps working
- register in `BUILDERS`

The docstring must carry a decision rule per the standing convention, and state
the A′ short-baseline caveat.

### 2. Plotting — `scripts/plotting/exp2_slowdown_ratio.py`

Currently one line per panel. Changes:

- `load()` groups by (setup, α) rather than setup, and resolves a **separate
  `cent_full` denominator per α** — the part most likely to fail silently, since
  a wrong denominator still plots a plausible line
- read from both `aggregation` and `aggregation_alpha2`
- one line per α from validated palette slots 1 and 2 (`#2a78d6`, `#eb6834`),
  already validated as a pair in both modes; re-run
  `scripts/validate_palette.py` if a third α is ever added
- legend gains the α values; keep the existing censored-marker,
  partial-censoring (`2/3`) and baseline-band behaviour, computed per α
- E's panel labels its axis `n_train`, not α

### 3. Execution

```bash
venv/bin/python scripts/build_manifests.py t2_aggregation_alpha2
venv/bin/python scripts/validate_manifest.py manifests/t2_aggregation_alpha2.jsonl
setsid nohup venv/bin/python -u scripts/launch_sweep.py \
    manifests/t2_aggregation_alpha2.jsonl --gpus 0 --per-gpu 1 \
    > logs/sweeps/agg_alpha2_$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null &
```

> **Hardware changed under this plan (2026-08-17).** It was written for the shared
> 8× L4 box (23 GB/card, 64 cores), where the instruction was to omit `--gpus` and
> let the launcher autodetect around three other users. This machine has **one
> RTX 3080 Laptop (8 GB) and 16 cores**, so:
>
> - `--gpus 0 --per-gpu 1`. Autodetect still works but there is only one card, and
>   a second concurrent run on it will not fit.
> - **Set `FEDGROK_GPU_CLIENT_CAP=8`.** Measured on B's K=50 cell, 300 rounds:
>   the default co-schedules all 50 clients and takes **723 s at 5,787 MiB**; a cap
>   of 8 takes **253 s at 3,086 MiB**. Capping is 2.9× faster *and* uses half the
>   memory, because 50 client processes on 16 cores spend their time contending
>   rather than computing. `FEDGROK_CLIENT_CPU=1` is the fallback: 308 s at
>   **389 MiB**, i.e. no GPU ceiling at all for 22% more wall-clock.
> - **B's K=50 cells do fit**, contrary to what this note first said. The ~12 GB
>   figure in `run_campaign_part0.sh` is from the 64-core L4 box, where all 50
>   clients really could co-schedule; here `num_cpus: 1` against 16 cores already
>   throttles them to 5,787 MiB of the 8,192 available. It fits with ~2.4 GB spare
>   at the default, and comfortably under the cap.
> - The two aborted launches of 2026-08-14 (`logs/sweeps/agg_alpha2_*.log`) left 16
>   spec-only run dirs under `results/runs/`. They carry no result JSON, so resume
>   re-runs them correctly — nothing to clean up.

## Verification

1. **Dedup is real before spending anything.** The manifest should report
   **87 of 105** specs to run:
   ```bash
   venv/bin/python - <<'EOF'
   import sys, os, glob; sys.path.insert(0,'src')
   from fedgrok.manifest import load_manifest, run_id
   banked = {os.path.basename(p)[:-5] for p in glob.glob('results/data/runs/*.json')}
   specs = load_manifest('manifests/t2_aggregation_alpha2.jsonl')
   print(len(specs), 'specs,', sum(run_id(s) not in banked for s in specs), 'to run')
   EOF
   ```
   If C's K ∈ {5,10,20,50} or A′/C's centralized do **not** dedup, the block does
   not match the banked config — stop and diff the specs rather than launching.

2. **Partition feasibility before launch:** E at n_train=4000 must give shards
   ≥ batch_size at every K it runs. Build the partitions directly, as was done
   for the Dirichlet ladder, rather than discovering it mid-sweep.

3. **After the sweep:** `collect_runs.py`, then confirm every new cell reached
   the bar and re-derive the ratio table. Cross-check that the existing α column
   still reproduces RESULTS §15.1 exactly (A 1.00→1.17, B 1.25→2.07, D 1.01→4.39,
   E 1.53→15.56) — if it moves, the loader change broke the existing line.

4. **Render and look at every figure.** The first pass of these panels shipped
   three layout bugs that only inspection caught — clipped error bars, a censored
   label colliding with the subtitle, and a partly-censored cell drawn as
   complete. Two lines per panel doubles the collision surface.

5. Write up as a RESULTS §15.1 extension, and state whether A′'s second line
   behaves as the caveat predicts.
