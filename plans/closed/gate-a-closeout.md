# Plan: close Gate A

## Context

The three Gate A sweeps finished — 211 runs, 0 failures, ~3 h. They settled two of
four setups and raised two questions that block Stage 3:

| Setup | Verdict |
|---|---|
| **B** transformer / mod-113 | **Ready.** Cliff α≈0.20, working α=0.30 (T_grok 6,600). Caveat: bimodal seed variance, ~5× spread |
| **D** quad-MLP / S₅ | **Ready.** Cliff α≈0.20, working α=0.30 (21,300) or 0.50 (7,200) |
| **C** transformer / S₅ | **Failed.** Never groks reliably — 4/5 at α=0.5, 3/5 at 0.4, 0/5 below |
| **E** Omnigrok MNIST | **Unresolved.** Delay and shardability oppose each other; the FL probe used a config with no delay at all |

Plus one unexplained result: **every AdamW setup fails to train at K=50** (1–5% train
accuracy), while setup A under plain GD groks 5/5 at K=50. Weight-norm collapse and
client drift were both checked and neither explains it.

Nothing here needs a schema change — every axis below uses existing `Config` fields,
so no banked run ids move.

---

## Step 0 — Commit (no compute) — NEEDS YOUR GO-AHEAD

31 files uncommitted, including two fixes that silently corrupt results (the S₅
dirichlet class-count bug, the dataset cache-key collision), federated MNIST, the
probe registry, 86 tests, and 211 runs of results. I'd split it roughly:

1. `fix:` partition class count + cache-key collisions — the two silent-corruption bugs
2. `feat:` federated MNIST + `label_block` partition
3. `feat:` per-setup mechanistic probes, generalised client signature, setup-aware analyzer
4. `feat:` manifest validation, orphan guard, collector/summariser setup identity
5. `results:` Gate A — cliffs, MNIST working point, AdamW A/B, refitted costs

I have not committed anything so far and won't without a word from you.

---

## Step 1 — Settle setup C  ·  34 runs, ~11 slot-h, ~1 h

Two competing explanations for C's failure, and they need separating before C either
stays or is dropped.

**1a. The cliff is simply higher than the ladder reached.** α ∈ {0.6, 0.7} × 5 seeds,
100k epoch budget. **10 runs.**

**1b. The model is underparameterised.** d_model=128 against 120 output classes is
thin, and the *quadratic MLP* at width 256 handles the same task cleanly — which is
the suspicious part, since the transformer is supposed to be the canonical grokking
architecture. `n_heads` ∈ {4,8} × `d_mlp` ∈ {512,1024} × `hidden_width` ∈ {128,256},
α=0.5, 100k budget, 3 seeds. **24 runs.** These are newly sweepable because this
session made them dataclass fields; before, a manifest setting them raised.

New builder: `s5_setup_c_capacity`.

> **Decision rule.** C stays if some configuration reaches 5/5 at a workable α with
> T_grok below a third of budget. Otherwise C is dropped and the campaign runs A/B/D/E
> — which still separates architecture (B vs A) from task (D vs A); C is the
> interpolation cell, not a load-bearing one.

---

## Step 2 — Federated MNIST at a working point that actually groks  ·  18 runs, ~4 slot-h, ~1 h

The probe ran MNIST at `(n_train=4000, batch=200)` — which the working-point sweep
then showed has **no delay at all** centrally (T_grok 500 = memorise 500). So the
probe was measuring federation on a setup that wasn't grokking to begin with, and its
censoring says nothing.

Re-run at `(n_train=2000, batch=100)`, the only cell with both a real delay (500) and
≥2 batches per local epoch at K=10. K ∈ {5, 10, 20} × {iid, label_block} × 3 seeds,
4,000 rounds (20,000 steps = 25× the centralized requirement).

New builder: `s5_mnist_fl`.

> **Decision rule.** If it groks at K ≤ 20, MNIST joins Stage 3 with a documented K
> ceiling and a documented caveat: its delay shrinks as K rises purely from the batch
> constraint, which is a property of the setup, not a finding about federation. If it
> does not grok, MNIST is centralized-only and that gets stated rather than patched.

---

## Step 3 — Targeted budget re-run  ·  9 runs, ~8 slot-h, ~40 min

Only the cells that genuinely memorised and are waiting to generalise. Not a blanket
re-run — the K=50 failures below are a different problem and more budget will not fix
a model sitting at 4% train.

| Cell | current | why re-run |
|---|---|---|
| D quad/S₅ K=50 iid | 99% train, 77–82% test *(bar 85)* | grokking pending, just short |
| D quad/S₅ K=10 operand | 95% train, 13% test | memorised, generalisation not started |
| B tfmr/mod K=10 operand | 50–96% train, 8–91% test | mid-transition, one seed nearly made it |

10,000 rounds (5× the probe). New builder: `s5_probe_rerun`.

MNIST's censored probe cells are deliberately *not* in this list — Step 2 supersedes
them at a different working point.

---

## Step 4 — Diagnose the K=50 AdamW failure  ·  9 runs, ~3 slot-h, ~15 min

The most interesting open item, and the cheapest. Every AdamW setup sits at 1–5% train
at K=50 while setup A under GD groks 5/5 at the same K — so it is not K by itself.
Ruled out already: weight-norm collapse (norms are flat or growing — D K=50 operand
reaches a norm comparable to the grokking K=10 run at 2% train), and client drift
(the failing D K=50 operand has *lower* drift than the working D K=50 iid).

- **Local step size.** `lr` ∈ {1e-4, 3e-4, 1e-3} × `weight_decay` ∈ {0.1, 1.0} on
  B K=50 iid, 2,000 rounds, 1 seed. **6 runs.**
- **Where it breaks.** K ∈ {20, 30, 40} at default hyperparameters, 1 seed. **3 runs.**

New builder: `s5_k50_diagnosis`.

> **Decision rule.** If a lower local lr recovers training, this is a
> local-hyperparameter mismatch — hyperparameters tuned for full-batch centralized
> training being wrong for 76-sample shards — and Stage 3 must tune per-K or move the
> adaptivity server-side (FedAdam/FedYogi, already implemented). If nothing recovers
> it, it is a candidate breakdown mechanism and becomes a headline rather than a
> nuisance, with the per-client checkpoints as the evidence base.

---

## Step 5 — Close Gate A and write Stage 3

Record the cliffs, working points, refitted cost constants and the four decision-rule
outcomes in `RESULTS.md` and `PROGRESS.md`, then write the Stage 3 manifests against
confirmed numbers.

One finding to carry into Stage 3 sizing rather than fix: **setup B's T_grok variance
is intrinsic and bimodal** — seeds 4,400 / 6,100 / 6,600 / 19,500 / 20,400 at α=0.30,
against setup A's 12,600–13,400. Two clusters ~3× apart, so more seeds narrow the
interval by ~√2 and do not make it unimodal. Stage 3 on B therefore cannot resolve
effects below roughly 2–3× at 5 seeds; either B carries more seeds per cell, or it is
not the setup that carries the headline claim.

---

## Totals

**~70 runs, ~26 slot-hours, ~2.5–3 h wall-clock** on 12 slots. All four steps are
independent and can run on separate GPU pools concurrently.

Steps 1 and 4 are the two that can change the shape of the campaign: one may remove a
setup, the other may promote a nuisance into the mechanism the project has been
looking for.

## Verification

- `scripts/validate_manifest.py` on each new manifest before launching — it already
  catches empty shards, degenerate batch sizes, and `hidden_width % n_heads`.
- 450 tests stay green after any builder change.
- Budgets stated as a multiple of the measured centralized T_grok for that setup, so
  no cell is censored by the clock. This is the failure mode that cost v1 its headline
  claim, cost the E=1 probe cells, and cost me the FL probe.
