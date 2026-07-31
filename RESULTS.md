# Results — federated grokking

Everything measured, as of 2026-07-31. Branch `v2-multisetup`.

Companion documents: `PROGRESS.md` (what is built and what remains),
`~/.claude/plans/plan-all-that-needs-nested-seal.md` (the campaign design).

**Data behind every number here:**
`results/data/runs_v2.csv` (720 v2 runs) and `results/data/runs.csv` (870 v1 runs,
recovered from logs). Both are committed. Regenerate any table with:

```bash
venv/bin/python scripts/summarize_runs.py results/data/runs_v2.csv --group <cols>
```

All T_grok figures are Kaplan–Meier medians with bootstrap 95% CIs over seeds,
right-censoring runs that did not grok within budget. `frac` is the fraction of
seeds that grokked — the order parameter, and the honest headline when a cell is
partially censored.

---

## The short version

1. **Federation does not break grokking; it delays it** — modestly. Ten times the
   clients costs ~16% more training time at α=0.3.
2. **v1's one breakdown claim was a budget artifact.** At α=0.25, K=97, two of five
   seeds grok at 95.6k/98.5k steps. The original experiment stopped at 50k.
3. **Partition structure is the variable that matters.** At K=97 a structured split
   groks 5/5 while a random split manages 2/5. Unstructured non-IID (Dirichlet)
   behaves exactly like random. *How* you partition matters more than *how far* you
   fragment.
4. **All four setups grok**, so the phenomenon is not an artifact of one architecture.
5. **Weight decay's effect depends on the optimizer, not the loss** — AdamW's
   decoupled decay accelerates grokking, plain GD's coupled decay does not.

6. **Every setup has its own cliff, and they nearly coincide** — A, B and D all
   sit at α≈0.20 (§11). The data threshold looks like a property of the task
   family, not of the architecture.
7. **Federation breaks the new setups at K≈30, and it is not a grokking
   failure** — those models never memorise (§12). Every affected setup uses
   AdamW; the anchor, under plain GD, is fine at K=97.

Open: whether K=97 IID *fails* or is merely *slow* — both successes landed within
5% of the budget ceiling, so that cell is not yet resolved. And whether the K≈30
collapse is an inherited-hyperparameter defect or a real breakdown mechanism.

---

## 11. Gate A — each setup's own cliff

α=0.25 is the *anchor's* working point; nothing said the other setups shared it.
Centralized α ladders, 5 seeds, KM median:

| setup | 0.5 | 0.4 | 0.3 | 0.25 | 0.2 | 0.15 |
|---|---|---|---|---|---|---|
| **A** quad-MLP mod-97 | 7,600 | 8,800 | 13,100 | 25,300 | 0/5 | 0/5 |
| **B** transformer mod-113 | 800 | 1,400 | 6,600 | 12,400 | 2/5 | 0/5 |
| **D** quad-MLP S₅ | 7,200 | 12,600 | 21,300 | 36,200 | 0/5 | 0/5 |
| **C** transformer S₅ | 4/5 | 3/5 | 0/5 | 0/5 | 0/5 | 0/5 |

A reproduces v1's α=0.25 value (25,300 vs 25,133) — independent harness
agreement. B and D cliff at ~0.20 with clean monotone divergence; working point
α=0.30.

**Setup C's failure was censoring, not capacity.** The ladder gave C 40,000
epochs. The capacity sweep gave it 100,000 and got **12/12 even at the baseline
width 128**, KM median 51,200 — above the budget the ladder allowed. Width 256
then halves it to 21,600, so capacity is a real accelerant, but C should not have
been written off. At α=0.6 → 20,200 and α=0.7 → 4,800, both 5/5. C stays, at
α≥0.5 with ≥100k epochs, with the caveat that its working point sits far above
the others' 0.30 so matched-α comparison is unavailable.

That is the fourth time a fixed budget has manufactured a boundary in this
project — after v1's headline claim, the E=1 probe cells, and the first FL probe.

**Setup B's seed variance is intrinsic and bimodal**: 4,400 / 6,100 / 6,600 /
19,500 / 20,400 at α=0.30, against A's 12,600–13,400. Two clusters ~3× apart, so
more seeds narrow the interval by ~√2 and do not make it unimodal. B cannot
resolve effects below ~2–3× at 5 seeds.

**MNIST: delay and shardability are in opposition.** Every config with a large
memorise→generalise delay needs a large batch, which makes shards degenerate;
every shardable config has a small delay or none. Best compromise
(n_train=2000, batch=100): delay 500.

---

## 12. The K≈30 collapse on AdamW setups

Setup B, one seed per cell, default hyperparameters — **peak** train accuracy:

| K | 10 | 20 | 30 | 40 | 50 |
|---|---|---|---|---|---|
| peak train | 100.0 | 98.2 | 42.8 | 5.9 | 5.0 |
| final train | 99.9 | 98.2 | 41.3 | 5.3 | 3.6 |

A smooth degradation, not a cliff — and a **training** failure, not a grokking
one. `peak ≈ final` throughout, so nothing memorises and then collapses; it never
learns. No budget fixes a model at 5% train accuracy. Setup A under plain GD at
wd=0 groks 5/5 at K=50 and is the only setup with no decay clock at all.

Ruled out: weight-norm collapse (norms flat or growing), client drift (the
failing D K=50 operand drifts *less* than the working D K=50 iid), and local step
size — lower lr is **worse**, 4.6% train at lr=1e-4.

The one knob that moves it is weight decay. At K=50 on setup B:

| lr | wd | lr·λ | train acc |
|---|---|---|---|
| 1e-3 | 1.0 | 1e-3 | ~3.6% ← inherited default |
| 1e-4 | 1.0 | 1e-4 | 4.6% |
| 3e-4 | 1.0 | 3e-4 | 3.4% |
| **1e-3** | **0.1** | **1e-4** | **70.2%** |

One seed per cell, so this is a lead rather than a result;
`manifests/p1_k_collapse_wd.jsonl` settles it at 3 seeds across K ∈ {20,30,50}.

**Do not lean on the numerical coincidence with §6.2.** lr·λ=1e-3 is also where
the anchor's corrected WD sweep found decay outrunning learning, but that sweep
was **GD**, where decay is coupled — and §6.4 is precisely the finding that the
optimiser is what discriminates. The federated weight-decay evidence stands on
its own; the matching number does not transfer.

### 12.1 Why this was invisible until now

`t_grok` is one number and grokking has two timescales, so "memorised but never
generalised" and "never trained" both recorded `inf`. `t_memo`, `delay` and
`peak_train_acc` are now recorded per run (and backfilled across all 720 banked
rows by `scripts/backfill_runs.py`), which is what makes the table above readable
as a training failure rather than a grokking one.

### 12.2 Related: D's operand partition fails, which inverts the headline

At K=10 on setup D, 50,000 steps: 99.8% train, **63% test**, 3/3 censored against
an 85 bar. Memorised, not generalising — while D at K=50 **iid** groks 3/3 at
~11,000. That is the opposite of §5.4, where the operand partition *rescues* K=97
on the anchor. Whether it is slow or stuck is unresolved and needs 5× budget
before "structure hurts D" is claimable.

Note also that on S₅ the operand partition shards by first-operand *element*,
which is not algebraically coherent the way a mod-p operand shard is. The
**coset** partition is S₅'s coherent one, so coset-vs-operand-vs-iid on D is what
actually separates "coherent shard" from "merely structured shard".

---

## 1. What groks, and when

Four setups, all confirmed. Delay = gap between memorisation and generalisation.

| setup | task | groks | memorise | generalise | delay |
|---|---|---|---|---|---|
| Quadratic MLP (Gromov) | mod-97 addition | yes | — | 7,600 | — |
| Nanda transformer | mod-113 addition | yes | 200 | 6,200 | 6,000 |
| Transformer | S₅ composition | yes | 200 | ~14,000 | ~14,000 |
| 3-layer MLP (Omnigrok) | MNIST-1k | yes | ~600 | 3,800 | 3,300 |

The quadratic MLP on modular addition is the anchor; every federated result below
uses it.

### Grok thresholds are dataset-dependent

`compute_t_grok` originally hardcoded a 95% test-accuracy bar. MNIST-1k peaks near
93%, so all 15 MNIST runs recorded `t_grok = inf` — "never grokked" — while their
histories plainly showed grokking. A measurement artifact reported as a scientific
null.

The bar is now a dataset property (`fedgrok.data.registry.grok_threshold`):

| dataset | bar | chance | ceiling |
|---|---|---|---|
| modular | 95.0 | ~1% | 100% |
| s5 | 85.0 | ~0.8% | ~92% |
| mnist | 90.0 | 10% | ~93% |

Modular is unchanged, and every prior modular cell was verified bit-identical after
the change. **S₅ was heading for the same silent failure**: at a 95% bar all 72
planned S₅ replication runs would have recorded `inf`.

`grok_threshold` is now stored per result row and is a survival cell key — a T_grok
is only interpretable next to the bar it was measured at.

---

## 2. The data threshold dominates everything

`α` is the fraction of the p²=9,409 possible pairs used for training. Centralized,
no federation involved (v1, modular addition):

| α | train examples | grokked | T_grok |
|---|---|---|---|
| ≤ 0.20 | ≤1,882 | **0 / 24** | never |
| 0.25 | 2,352 | 21/22 | **25,300** |
| 0.30 | 2,823 | 30/31 | 12,600 |
| 0.35 | 3,293 | 18/18 | 9,900 |
| 0.50 | 4,704 | 32/32 | 7,600 |

There is a **cliff between α=0.20 and α=0.25**, and T_grok diverges as it is
approached: 7,600 → 12,600 → 25,300 → never. (α=0.40 reads 12,100 at n=3 — noise.)

This is the most important fact about the setup. Federated effects are small next to
it, and the interesting regime is α=0.25, just above the cliff, where the problem is
already marginal and small penalties get amplified.

---

## 3. What federation costs

Like-for-like at α=0.3, E=5 (v2 `t2_k_breakdown`, 5 seeds/cell):

| | T_grok | vs centralized |
|---|---|---|
| Centralized | 12,600 | — |
| K=5 | 13,200 | +5% |
| K=10 | 13,400 | +6% |
| K=20 | 13,700 | +9% |
| K=50 | 15,200 | +21% |

**At E=1 the cost is exactly zero** — FedAvg with n_k/n weighting and full-batch
local GD is an algebraic identity with centralized GD, proven in
`tests/test_fedavg_identity.py` and observed in the wild: the T1 probe's IID and
operand runs returned identical test accuracies seed-for-seed (33.5 / 34.1 / 17.2).

**Local steps cost more than clients.** T1 probe, K=10, α=0.3:

| E | iid | operand |
|---|---|---|
| 1 | censored (budget, not federation) | censored |
| 5 | 12,900 [12,900, 13,700] | 12,700 [12,600, 13,400] |
| 50 | 23,000 [22,000, 25,000] | 17,000 [16,000, 18,000] |

E=5 → E=50 is +78% on the compute-matched step axis. The E=1 cells are **not**
breakdown evidence: they received 10,000 steps against the ~12,900 needed.

---

## 4. `t2_k_breakdown` — the α=0.3 control (60/60, complete)

K ∈ {5,10,20,50} × {iid, operand, dirichlet} × 5 seeds, α=0.3, E=5, 10k rounds.
KM median [95% CI]:

| K | iid | operand | dirichlet |
|---|---|---|---|
| 5 | 13200 [12700, 13500] | 13100 [12600, 13400] | 13400 [12700, 13600] |
| 10 | 13400 [13400, 13700] | 13200 [13200, 13500] | 13600 [13000, 13900] |
| 20 | 13700 [13200, 14100] | 13300 [12700, 13500] | 13900 [13200, 14200] |
| 50 | 15200 [14600, 16000] | **13700 [13000, 14000]** | 15400 [14700, 16100] |

**Every cell 5/5. Zero censoring.** The α=0.3 plane is uniformly safe.

**First sighting of the structure effect**: at K=50, operand is significantly faster
than iid (non-overlapping CIs). The gap scales with K — ~0 at K=5, ~400 steps at
K=20, ~1,500 at K=50. Dirichlet tracks iid exactly at every K, so this is
**structure, not heterogeneity**.

---

## 5. `t2_boundary` — the campaign that settled it (20/20, complete)

α=0.25, E=5, 20k rounds = 100k steps, 5 seeds. ~4.1 h/run at K=97.

| K | partition | grok | frac | KM median | 95% CI |
|---|---|---|---|---|---|
| 20 | iid | 5/5 | 1.00 | 29,800 | [27,300, 33,100] |
| 50 | iid | 5/5 | 1.00 | 46,800 | [40,700, 51,400] |
| 97 | iid | **2/5** | **0.40** | inf | [95,600, inf] |
| 97 | **operand** | **5/5** | 1.00 | **76,500** | [57,800, 77,500] |

Per seed at K=97 —
**iid:** 95,600 · 98,500 · and three censored at 100k ending 100% train with
**53% / 61% / 71% test**.
**operand:** 57,800 · 71,000 · 76,500 · 77,200 · 77,500, all reaching 100/100.

### 5.1 The harness sanity check passed

K=20 gives 29,800 [27,300, 33,100] against v1 exp2's 27,215 / 29,720 / 33,025. The
v2 rewrite reproduces exp2, so K=97 differences are attributable to federation
rather than to the refactor. K=50 grokking 5/5 confirms the 100k budget sufficed —
the other way this sweep could have returned nothing.

### 5.2 v1's breakdown claim does not survive

exp2 ran `t_max=50000`. Its α=0.25, E=5, iid ladder (3 seeds):

| K | 2 | 5 | 10 | 20 | 50 | 97 |
|---|---|---|---|---|---|---|
| T_grok | 23–27k | 24–28k | 25–29k | 27–33k | 41k, 44k, cens | all cens |

K=50 grokked at 41–44k against a 50k budget, so K=97's 0/3 was what continued
monotone delay looks like when it runs past the clock — the same trap as the E=1
probe cells, in the one place where censoring was supposed to *be* the signal.
**Two of five K=97 seeds do grok, at 95.6k and 98.5k.** Under a 50k budget all five
censor.

### 5.3 A partial transition is real but unresolved

2/5 at K=97 iid, with failures at 100% train and 53–71% test — memorised, not
generalised. **Caveat: both successes land within 5% of the 100k ceiling**, so the
budget is still marginal and those three may be mid-transition rather than stuck.

Separating "slower" from "broken" needs **more budget, not more seeds**. This
corrects the wave-2 plan, which called for deepening to 20 seeds — that would
sharpen a fraction whose denominator is itself budget-limited.

### 5.4 The headline: structure rescues K=97

operand 5/5 at 76,500 [57,800, 77,500] versus iid 2/5 at ~97,000 — non-overlapping,
and the difference between reliable grokking and marginal failure.

Combined with §4 (Dirichlet ≡ iid at every K), the claim is:

> **How you partition matters more than how far you fragment. Coherent shards are
> strictly better than random ones.**

Unlike the breakdown claim, this one is insensitive to where the budget was set.
The mechanism hypothesis — coherent shards let clients select a *shared* Fourier
basis that averaging reinforces, where iid clients pick conflicting sets that
partially cancel — is testable on **400 per-client weight snapshots already on
disk** (20 checkpoints × 20 runs, `results/runs/*/checkpoints/`).

---

## 6. Weight decay

### 6.1 The v1 result was a values bug

exp5 ran `lr=50 × wd ∈ {0.01, 0.1, 1.0}`, i.e. `lr·λ ∈ {0.5, 5, 50}`. Both coupled
(SGD) and decoupled (AdamW) decay shrink weights by `(1 − lr·λ)` per step, so `lr·λ`
is the comparable quantity and `1/(lr·λ)` the decay timescale. Those settings halve
or invert every weight every step. All nine cells returning `T_grok = inf` was
arithmetically forced.

`core.utils.check_decay_stability` now rejects divergence (`lr·λ ≥ 2`) and the
subtler destructive case (`lr·λ > 0.1`).

### 6.2 Corrected sweep — `t0_wd_grid` (45 runs, α=0.5, p=97, 5 seeds)

| lr·λ | λ (GD, lr=50) | grokked | T_grok |
|---|---|---|---|
| 0 | 0 | 5/5 | 7,600 [7,500, 7,800] |
| 1e-5 | 2e-7 | 5/5 | 7,700 [7,500, 7,800] |
| 1e-4 | 2e-6 | 5/5 | 8,300 [8,200, 8,600] |
| 1e-3 | 2e-5 | **0/5** | censored |
| 1e-2 | 2e-4 | **0/5** | censored |

Grokking survives to `lr·λ = 1e-4` and dies at 1e-3, where decay outruns learning —
train accuracy never exceeds 8% and decays back to 1%, so memorisation never happens
and there is nothing to grok from. Within the usable band, decay is
**neutral-to-mildly-slowing**.

The AdamW arm at α=0.5 reaches 100/100 by epoch 200 at every λ (no delay at all), so
it cannot inform the question at this α.

### 6.3 MNIST — `t0_mnist_wd_band` (15 runs, 3 seeds)

| lr·λ | memorise | generalise | delay | grokked | peak test |
|---|---|---|---|---|---|
| 1e-5 | 600 | never | — | 0/3 | 89.2% |
| 3e-5 | 600 | 11,100 | ~10,500 | 3/3 | 91.1% |
| **1e-4** | 500 | **3,800** | **3,300** | 3/3 | **92.7%** |
| 3e-4 | 500 | 3,000 | 2,500 | 3/3 | 92.4% |
| 1e-3 | 800 | 3,200 | 2,400 | 3/3 | 91.5% |

**Weight decay accelerates grokking here, monotonically** — the published Omnigrok
result reproduced. Best band is `lr·λ = 1e-4`.

### 6.4 The discriminator is the optimizer, not the loss

The modular result (§6.2, neutral-to-slowing) and the MNIST result (§6.3,
accelerating) were initially reconciled by attributing acceleration to the
"AdamW/cross-entropy transformer regime". **That was wrong.** MNIST is MSE + AdamW
and accelerates; modular is MSE + GD and does not. The distinguishing variable is
**AdamW's decoupled decay vs GD's coupled decay**, not the loss function.

---

## 7. Task coverage

`t0_poly_pilot` (3 seeds, centralized, α=0.5, p=97) gated the operation set:

| task | grokked | T_grok | verdict |
|---|---|---|---|
| x² + y² | 3/3 | 7,000 [6,900, 7,000] | kept |
| x² + xy + y² | 0/3 | censored | **excluded** |

Consistent with Doshi et al. (2406.03495) separating learnable from non-learnable
modular polynomials for this architecture. Multiplication is excluded by design: on
nonzero residues it is cyclic Z_{p−1} under discrete log, testing the same Fourier
structure over a composite-order group — a confound, not a control.

---

## 8. Cost model

Fitted on 78 completed federated runs, and validated out-of-sample at K=97
(predicted 165 s for 200 rounds, measured 174 s):

```
minutes = (9.8 + 1.291·K + 0.418·E) × rounds / 10000
```

The K and E effects are cleanly additive. Reference points at 10k rounds, E=5:
K=5 → 22 min, K=20 → 33 min, K=50 → 75 min, K=97 → 2.25 h.

**Wall-clock is ~99% orchestration, not compute.** 50,000 centralized gradient steps
take **48 s**; the identical arithmetic federated across 5 clients takes **22 min** —
27× — all of it Flower/Ray shipping weights between client processes each round.
Cost therefore scales with *client count*, not with training length.

Practical consequence: order manifests **longest-job-first**. On `t2_boundary` that
cut wall-clock from ~7.2 h to ~5.9 h for identical work.

---

## 9. Run inventory

| | runs | grokked | censored |
|---|---|---|---|
| **v1** (`results/data/runs.csv`, log-recovered) | 870 | 554 | 316 |
| **v2** (`results/data/runs_v2.csv`) | 158 | 133 | 25 |

v1 by experiment: exp2 333 · exp5 153 · exp3a 100 · exp7 74 · exp4a 72 · exp4b 72 ·
exp4 36 · exp3b 30.

v2 by campaign:

| campaign | runs | status |
|---|---|---|
| `t0_wd_grid` | 45 | done — §6.2 |
| `t0_poly_pilot` | 6 | done — §7 |
| `t0_mnist_wd_band` | 15 | done — §6.3 |
| `t1_probe` | 18 | done (6 E=250 cells cancelled for cost) — §3 |
| `t2_k_breakdown` | 54 | done — §4 |
| `t2_boundary` | 20 | done — §5 |

**v2 compute to date: 104.4 machine-hours.** Checkpoints on disk: 3.9 GB
(gitignored).

---

## 10. Caveats and known limitations

- **K=97 IID is unresolved** (§5.3). The budget is marginal; the 2/5 fraction has a
  budget-limited denominator.
- **Every federated result uses one setup** — the quadratic MLP on modular addition.
  Replication across transformer / S₅ / other primes is written but unrun, so
  generality of the structure effect is currently untested.
- **exp4b partial-participation T_grok values in `runs.csv` sit on the old inflated
  step axis** (~2.5× too high on `total_steps`); the Phase 0.6 fix changed that axis
  under `fraction_train < 1.0`. They will be corrected when exp4b is re-run.
- **v1 history filenames are not unique run keys.** Algorithm suffixes landed
  mid-campaign, so 61 paths were written by more than one run and any surviving JSON
  holds only the last. `harvest_logs.py` therefore takes algorithm identity from the
  log filename, and `runs.csv` — not the JSONs — is the v1 record.
- **Grid holes:** 6 empty exp3a α=0.20 logs; `exp3a_a0.30_dir10.0` crashed at round
  6,491 on a Ray actor failure. Tracked in `results/data/runs_skipped.csv`.
- **α=0.40 centralized reads 12,100** against 9,900 at α=0.35 and 7,600 at α=0.50 —
  n=3, almost certainly noise, but it is the one non-monotonicity in §2.
- **Out of scope, stated deliberately:** DP-FedAvg, communication compression,
  Byzantine-robust aggregation, personalization, async/stragglers, FedBN,
  MOON/FedDecorr, and the LEAF/FLamby/FedScale benchmarks.
