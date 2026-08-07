# Results — federated grokking

Everything measured, as of 2026-08-07. Branch `v2-multisetup`.

Companion documents: `PROGRESS.md` (what is built and what remains),
`~/.claude/plans/plan-all-that-needs-valiant-hamster.md` (the current plan),
`~/.claude/plans/plan-all-that-needs-nested-seal.md` (the boundary campaign, closed).

**Data behind every number here:**
`results/data/runs_v2.csv` (814 v2 runs) and `results/data/runs.csv` (870 v1 runs,
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

8. **The optimiser is worth ~45× on the anchor's own task** — the identical
   architecture on the identical data groks in 500 steps under AdamW where it
   takes 25,300 under Gromov's GD, at every α they share (§13.3).
9. **Two of the three inherited hyperparameters survived being checked** — C's
   and D's weight decay were adopted from Nanda's mod-113 transformer without
   justification, and measuring them for their own setups returns the same value
   (§13.2). The third, B's, is still being measured.

Open: whether K=97 IID *fails* or is merely *slow* — both successes landed within
5% of the budget ceiling, so that cell is not yet resolved. And whether the K≈30
collapse is an inherited-hyperparameter defect or a real breakdown mechanism.

---

## 13. Phase 1 — the setups, measured instead of inherited

Four sweeps, 96 runs, run to answer questions whose answers every downstream
manifest depends on. Each manifest's docstring in `scripts/build_manifests.py`
carries the decision rule it was written against; the verdicts below are those
rules applied.

### 13.1 The quadratic MLP does grok S₅ under plain GD (`p1_d_gd_probe`, 9/9)

Setup D is Gromov's architecture running Nanda's optimiser — inherited from the
S₅ side rather than the architecture side, and never checked: of 370 banked S₅
runs, **zero** used GD. That makes A vs D, which is meant to isolate the *task*,
move task and optimiser and loss together.

α=0.5, wd=0, MSE, 50,000 epochs, 3 seeds:

| lr | grokked | KM median T_grok |
|---|---|---|
| 5 | 0/3 | censored |
| 10 | 0/3 | censored |
| **50** | **3/3** | **22,600 [22,500, 23,200]** |

So the confound is closable — Gromov's exact configuration transfers to S₅ at
Gromov's exact learning rate. It costs ~3× the time AdamW needs at the same α
(22,600 against 7,200), which is the price of the clean contrast.

**Acted on by adding D′, not by moving D.** Run ids are content hashes, so
editing `SETUP_D` in place would orphan ~250 banked D runs — the Gate A ladder,
the 0.025-resolution tier-X ladder, and every D federated cell. This is the same
reasoning that produced A′ rather than a change to A. D′ is **gated**: its real
value is that wd=0 gives it no decay clock, which should make it immune to the
K≈30 collapse the way setup A is, and that only matters if §12 turns out to be a
genuine breakdown rather than a hyperparameter defect.

### 13.2 C's and D's decay: neither moves (`p1_cd_decay_band`, 30/30)

Both inherited wd=1.0 from B, which has a reason to carry it (B *is* the Nanda
replication) where they do not. Same log-spaced lr·λ ladder as `t0_wd_grid` and
`t0_mnist_wd_band`, 3 seeds, 100,000 epochs.

**D** (quad-MLP, S₅, α=0.30) — one band works and it is the inherited one:

| lr·λ | 1e-5 | 3e-5 | 1e-4 | 3e-4 | **1e-3** |
|---|---|---|---|---|---|
| grokked | 0/3 | 0/3 | 0/3 | 0/3 | **3/3** |
| T_grok | — | — | — | — | **21,300** |

**C** (transformer, S₅, α=0.50, width 256) — three bands work, and the tie-break
decides. Both statistics pick wd=1.0, by a wider margin on the sampling-robust one:

| lr·λ | grokked | KM median T_grok | median first crossing | dips after crossing |
|---|---|---|---|---|
| 1e-5 | 0/3 | censored | — | — |
| 3e-5 | 1/3 | censored | 97,100 | 0 |
| 1e-4 | 3/3 | 83,800 | 70,200 | 0, 1, 0 |
| 3e-4 | 3/3 | 96,750 | 42,650 | **13, 28, 2** |
| **1e-3** | **3/3** | **59,350** | **15,150** | 2, 1, 1 |

**Consequence: the re-ladder that Phase 2 was budgeted for does not happen.** The
plan reserved ~60 runs to re-measure C's and D's α ladders at a moved decay,
because the ladder sets the working point and every downstream budget. Neither
moved, so both Gate A ladders stand as written. This is Phase 1's largest saving.

**But C is unstable, and that is a new setup property.** Read the last two columns
together: higher decay reaches the bar much sooner and then oscillates, lower
decay is slow and steady. At wd=0.3 one seed drops below the bar 28 separate
times after first crossing it. C stays at wd=1.0 with `t_first_cross` as its
primary statistic — see §13.4 for why `t_grok` alone cannot be trusted here.

### 13.3 A′ cliffs *below* A, and groks ~45× faster (`p1_aprime_alpha`, 45/45)

A′ is A's architecture and task under AdamW (MSE, not CE, so A vs A′ moves one
variable). 5 seeds:

| α | 0.15 | 0.175 | **0.2** | 0.25 | 0.3 | 0.4 | 0.5 |
|---|---|---|---|---|---|---|---|
| **A′** (AdamW) | 0/5 | 0/5 | **5/5 · 5,500** | 5/5 · 500 | 5/5 · 300 | 5/5 · 200 | 5/5 · 170 |
| **A** (GD) | 0/5 | — | **0/5** | 25,300 | 13,100 | 8,800 | 7,600 |
| ratio | | | A′ only | **51×** | 44× | 44× | 45× |

Two things, both new:

**The optimiser is worth a factor of ~45, flat across α.** Same architecture,
same data, same loss, same seeds — only GD → AdamW. The ratio is remarkably
constant, which says AdamW is rescaling the clock rather than changing the
transition.

**The cliff moves down.** A′ groks 5/5 at α=0.20 where A manages 0/5. So the data
threshold is *not* purely a property of the task family, as §11 suggested from
four setups that happened to agree — the optimiser shifts it too.

**A′'s only usable federated working point is α=0.20.** Above it there is no
delay left to disrupt: 500 steps at α=0.25 means E=5 federation gets 100 rounds
to act. This is the trap setup E already fell into, where the FL probe ran MNIST
at a working point with no delay at all and its censoring meant nothing. Only the
α=0.20 cell has a real gap (5,500, with a wide seed spread of 3,000–8,500).

### 13.4 `t_grok` measures the logging rate on an unstable setup

Found while reading §13.2. `compute_t_grok` requires the bar to hold for the rest
of the run — correct for a phase transition, but on a non-monotone curve every
extra sample point is another chance to observe a dip and push the answer later.

Setup C, one seed, **one trajectory**: `t_grok` = 15,200 at `log_every=200` and
59,350 at `log_every=50`. A 4× difference from the instrument alone, because the
coarse run never sampled a collapse to 20.2% at epoch 59,300. C's decay band was
about to be chosen by tie-breaking on exactly that number.

Two outcomes are now recorded per run alongside it, at the run's own
`grok_threshold`, and backfilled across all 814 banked rows:

- **`t_first_cross`** — first time the bar is reached, sustained or not. Not
  sampling-invariant either, but stable to within one logging interval rather
  than to within the instability's duration.
- **`post_grok_dips`** — logged points below the bar after that crossing. Zero
  means the transition held. The magnitude scales with `log_every`, so only the
  zero/non-zero distinction transfers between runs.

Together they separate *when did it generalise* from *when did it stop falling
over*, which on an unstable setup are different questions. This is the fifth
measurement artifact this project has caught reading as a scientific result.

### 13.5 The K≈30 collapse — in progress (`p1_k_collapse_wd`, 16/18)

Weight decay does move the failure, decisively, but not in the direction the
one-seed lead suggested. At 3 seeds, α=0.30, E=5, 10,000 steps:

| K | wd | peak train | final test | t_memo | grokked |
|---|---|---|---|---|---|
| 20 | **0.1** | **100.0** | **0.2–0.5** | 3,500–3,900 | 0/3 |
| 20 | 1.0 | 44.9 / 100.0 | 7.3 / 99.8 | inf / 7,300 | 1/2 |
| 30 | **0.1** | **100.0** | **0.2–0.4** | 5,600–6,600 | 0/3 |
| 30 | 1.0 | 93.1 / 51.9 | 82.2 / 24.2 | inf | 0/2 |

**wd=0.1 restores training completely and generalisation not at all.** Every
wd=0.1 cell memorises — 100% train, 3/3, by epoch ~3,600 — and then sits at
0.2–0.5% test, which against chance (1/113 ≈ 0.88%) is not slow generalisation
but none. That is the two-timescale split §12.1 predicted would become visible
once `t_memo` was recorded.

**This is not yet interpretable, for two reasons, and both are being fixed.**
The K=50 cells — the ones the decision rule is actually evaluated on — are the
two runs missing from the sweep. And every centralized B run in the corpus is
wd=1.0, so there is no reference for what wd=0.1 does with *one* client;
`p1_b_decay_band` measures it. Until that lands, "memorised but never
generalised" cannot be attributed to federation rather than to wd=0.1 simply
having a longer delay than the 10,000-step budget allowed — which is precisely
the mistake that cost v1 its headline claim.

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
| **v2** (`results/data/runs_v2.csv`) | 814 | 558 | 256 |

v1 by experiment: exp2 333 · exp5 153 · exp3a 100 · exp7 74 · exp4a 72 · exp4b 72 ·
exp4 36 · exp3b 30.

v2 by campaign — regenerate with
`summarize_runs.py results/data/runs_v2.csv --group group`:

| campaign | runs | status |
|---|---|---|
| `central_anchor` | 131 | done — Gate A ladders, §11 |
| `d_alpha_high` / `d_alpha_fine` / `d_alpha_cliff` | 150 | done — tier X, setup D's α ladder at 0.025 resolution |
| `d_internals` | 85 | done — tier X, with checkpoints + the exact circuit instruments. **Unanalysed** |
| `k_fixed_total` | 54 | done — §4 |
| `fl_probe` | 48 | done — under-budgeted by construction; its censoring is not a federated effect |
| `aprime_alpha` | 45 | done — §13.3 |
| `wd_grid` | 45 | done — §6.2 |
| `mnist_fl` | 36 | done — E groks iid at K=10/20; `label_block` 0/12 |
| `mnist_working_point` | 33 | done — delay vs shardability, §11 |
| `cd_decay_band` | 30 | done — §13.2 |
| `c_capacity` | 24 | done — C's failure was censoring, not capacity, §11 |
| `boundary` | 20 | done — §5 |
| `probe` | 18 | done (6 E=250 cells cancelled for cost) — §3 |
| `mnist_wd_band` | 15 | done — §6.3 |
| `d_wd_ladder` | 12 | **partial** (12/45) — tier X, the dip's decay hypothesis |
| `k_collapse_wd` | 10 | **in flight** — §13.5 |
| `d_gd_probe` | 9 | done — §13.1 |
| `probe_rerun` | 9 | done — B K=10 operand and D K=50 iid recover at 5× budget |
| `poly_pilot` | 6 | done — §7 |
| plus 8 smaller diagnosis groups | 34 | `adam_restart`, `c_alpha`, `k50_hparam`, `k50_ladder`, `grok_confirm_fl` |

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
