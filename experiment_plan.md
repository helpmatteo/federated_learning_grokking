# Experimental Plan: Grokking in Federated Learning

> **Goal**: Produce a NeurIPS-quality empirical study characterising exactly when and why
> grokking survives — or fails — under federated learning.

---

## 1  Motivation

Prior work (Gromov 2023) shows that two-layer MLPs with quadratic activations exhibit
*grokking* on modular arithmetic: the network memorises the training set quickly, then
generalises long after memorisation, driven by the emergence of periodic Fourier features
in the first-layer weights.

Our preliminary experiments show that **grokking is remarkably robust to federated
training at α = 0.5** (50 % of all p² pairs used for training). Even K = 97 clients
(48 samples each), partial participation (f = 0.2), extreme local epochs (E = 200),
and non-IID target partitions all still grok to ≥ 94 % test accuracy.

This means the interesting regime is **not** the comfortable interior (α = 0.5) but the
**phase boundary** — the critical training fraction α_crit below which grokking fails.
The central question becomes:

> **Does federated learning shift the grokking phase boundary, and if so, why?**

---

## 2  Setup (fixed across all experiments unless stated otherwise)

| Symbol | Parameter | Value | Rationale |
|--------|-----------|-------|-----------|
| p | prime modulus | 97 | Gromov 2023 standard |
| — | task | (a + b) mod p | default; generality tested in appendix |
| — | activation | quadratic (x²) | theory-supported default |
| N | hidden width | 256 (validated in Exp 0) | well above Gromov Fig 4b plateau; 74 496 params |
| — | optimizer | SGD (no momentum) | no optimizer state → clean FL comparison |
| lr | learning rate | 50.0 | mean-field parameterisation default |
| λ | weight decay | 0.0 | isolates implicit regularisation |
| — | loss | MSE on one-hot targets | Gromov 2023 standard |
| — | training | full-batch | each "epoch" = 1 gradient step; no minibatch stochasticity |
| — | seeds | {42, 123, 456} | 3 seeds; report mean ± std |

### Derived quantities

| Quantity | Formula | Example (α = 0.5) |
|----------|---------|---------------------|
| total pairs | p² | 9 409 |
| training set | n_train = ⌊α · p²⌋ | 4 704 |
| test set | n_test = p² − n_train | 4 705 |
| per-client (IID) | n_client ≈ n_train / K | 941 (K = 5) |
| parameters | 3Np | 74 496 |
| param : data ratio | 3Np / n_train | 15.8 (α = 0.5) |

### Notation

| Symbol | Meaning |
|--------|---------|
| α | training fraction (fraction of p² used for training) |
| K | number of federated clients |
| E | local epochs per round (each = 1 full-batch gradient step on client data) |
| R | number of communication rounds |
| S = R × E | total gradient steps per client |
| f | participation fraction (fraction of K clients sampled per round) |
| μ | FedProx proximal coefficient (μ = 0 ⇒ FedAvg) |
| α_dir | Dirichlet concentration for non-IID partition |
| α_crit | critical training fraction below which grokking fails |

---

## 3  Formal Definitions

### 3.1  Grokking step

Let {t_i, a_i} be the sequence of (step, test accuracy) pairs logged during training.

**Definition.** The *grokking step* T_grok is the smallest logged step t_j such that
a_i ≥ 95 % for all i ≥ j (i.e., test accuracy reaches 95 % and never drops below it
for the remainder of training). If no such t_j exists within S_max steps, we record
T_grok = ∞ (labelled "no grokking").

**Rationale.** The 95 % threshold is conservative (grokking typically reaches > 99 %).
The "never drops below" condition avoids false positives from transient spikes. Using
all subsequent log points (rather than a fixed window) is stricter and eliminates an
arbitrary window-length parameter.

### 3.2  Total-step budget

Two step budgets, chosen by experiment type:

| Context | S_max | Rationale |
|---------|-------|-----------|
| Centralized (Exp 1) | 50 000 | Fast (~2 min per run); need headroom to find α_crit |
| Federated (Exps 2–5) | 30 000 | ~3.6× baseline grokking step; balances headroom vs Flower overhead |

- Centralized: 50 000 epochs, log every 50 epochs (1 000 log points).
- Federated: R = ⌈30 000 / E⌉ rounds, log every round.

If a federated run shows grokking *in-progress* at S = 30 000 (train acc = 100 %
but test acc climbing through 70–90 %), extend that specific run to S = 50 000.

**Runtime note.** Flower/Ray simulation has per-round overhead (~0.3–0.5 s).
At E = 5 → R = 6 000 → ~35–50 min per FL run. At E = 1 → R = 30 000 → ~3–5 hr
(impractical for sweeps). Therefore, for experiments where E = 1 is tested, we cap
at R = 10 000 (S = 10 000 steps) and note any grokking that is in-progress but
incomplete.

### 3.3  Centralized-reduced baseline

To test the effect of federated **aggregation** (vs simply having less data), we
define the *centralized-reduced* baseline:

> Train a single centralized model on n_train / K data points for S_max steps.

Implementation: set α_eff = α / K in the centralized config with the same seed.
Because the dataset is constructed from a seeded permutation of all p² pairs, and
the training set is the first ⌊α · p²⌋ elements, setting α_eff = α / K yields the
first ⌊α_eff · p²⌋ elements — a strict subset of the original training set.

This is statistically equivalent to one FL client's data (same size, random subset of
the training pairs), enabling a direct comparison:

| Condition | Data | Aggregation? |
|-----------|------|-------------|
| Centralized-full | n_train | — |
| Centralized-reduced | n_train / K | — |
| FL (IID) | n_train / K per client | Yes (K clients averaged) |

---

## 4  Experiments

### Experiment 0 — Width Validation (prerequisite)

**Problem.** The Fourier solution for (a + b) mod p requires (p−1)/2 = 48 frequency
modes, but redundant neurons improve accuracy via destructive interference of
unwanted terms (Gromov 2023, §3.1, Eq. 15). Gromov's Figure 4b shows that for GD
at α = 0.5 and p = 97, test accuracy reaches only ~80 % at N = 100 and requires
**N ≈ 120–140 for near-100 % accuracy**. Our default N = 128 sits right at this
edge. We must verify that N = 128 is safely above the critical width *across the α
range we will test*, so that grokking failures in later experiments are attributable
to FL — not to insufficient model capacity.

**Note.** The prior width sweep (N ∈ {20, 40, 60, 80, 100, 120, 140}) was invalid:
most runs completed only 100 epochs (vs ~8 000 needed for grokking). This must be
rerun properly.

**Design.**

| Parameter | Values |
|-----------|--------|
| N | {100, 128, 200, 256} |
| α | {0.1, 0.3, 0.5} |
| seeds | {42} (single seed; extend to 3 if results are noisy) |
| S_max | 50 000 epochs |

**Runs.** 4N × 3α = **12 runs** (fast: centralized, ~2 min each).

**Expected outcome.** N = 256 is our primary default (well above Gromov Fig 4b
plateau). This experiment confirms it works at low α, and documents the critical
width N_crit(α) for the paper. The comparison across widths also produces a
publishable figure (extending Gromov's Fig 4b to lower α values).

**Why this matters.** Without this step, any grokking failure at low α could be
confounded by width being too close to the theoretical minimum. This 30-minute
validation eliminates that confound.

---

### Experiment 1 — Centralized Grokking Phase Boundary

**Question.** At what training fraction α does centralized grokking fail?

**Hypothesis.** There exists a critical α_crit ∈ (0, 0.5) below which the model
memorises but never generalises. This boundary depends on the model capacity (N)
and the implicit regularisation from full-batch GD.

**Design.** Uses N* from Experiment 0.

| Parameter | Values |
|-----------|--------|
| α | {0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5} |
| seeds | {42, 123, 456} |
| S_max | 50 000 epochs |

**Controls.** α = 0.5 is the known-good baseline. Each α value gives a different
n_train, test set, and param-to-data ratio (shown for N = 256):

| α | n_train | n_test | params/data |
|---|---------|--------|-------------|
| 0.03 | 282 | 9 127 | 264.2 |
| 0.05 | 470 | 8 939 | 158.5 |
| 0.075 | 705 | 8 704 | 105.7 |
| 0.1 | 940 | 8 469 | 79.3 |
| 0.15 | 1 411 | 7 998 | 52.8 |
| 0.2 | 1 881 | 7 528 | 39.6 |
| 0.3 | 2 822 | 6 587 | 26.4 |
| 0.5 | 4 704 | 4 705 | 15.8 |

**Runs.** 8 × 3 = **24 runs.**

**Outputs.**
- α_crit (the phase boundary)
- Plot: grokking step vs α (with error bars). Sharp transition expected.
- For α near the boundary: full training dynamics (loss, acc, IPR, weight norms).

**This experiment is a prerequisite for all subsequent FL experiments.** It tells us
where to focus: the interesting FL experiments operate near α_crit.

---

### Experiment 2 — Aggregation Effect & FL Phase Boundary

**Question.** Does FL aggregation compensate for data fragmentation? Does FL shift
α_crit?

**Hypothesis.** Three possible outcomes:
1. FL α_crit ≈ centralized α_crit → aggregation perfectly compensates for
   fragmentation; FL is benign.
2. FL α_crit > centralized α_crit → FL needs more total data to grok;
   fragmentation hurts even with aggregation.
3. FL α_crit < centralized α_crit → aggregation acts as implicit regularisation;
   FL *helps* grokking (the surprising result).

**Design.** For each (α, K) pair, run three conditions:

| Condition | What | Data per model | Aggregation |
|-----------|------|----------------|-------------|
| **(a) Centralized-full** | Standard centralized | n_train = ⌊α · p²⌋ | — |
| **(b) Centralized-reduced** | One client's data size | n_reduced = ⌊(α/K) · p²⌋ | — |
| **(c) FL (IID)** | K clients, IID, f = 1.0 | n_train / K per client | FedAvg |

Sweep:

| Parameter | Values |
|-----------|--------|
| α | {α_crit − δ, α_crit, α_crit + δ, α_crit + 2δ, 0.5} where δ is chosen after Exp 1 to span the transition; fallback grid: {0.05, 0.1, 0.15, 0.2, 0.3, 0.5} |
| K | {5, 10, 20, 50} |
| E | 5 (default) |
| f | 1.0 (full participation) |
| R | ⌈30 000 / E⌉ = 6 000 |
| seeds | {42, 123, 456} |

**Runs.** 6 α × 4 K × 3 conditions × 3 seeds = **216 runs.**
(Condition (a) doesn't depend on K, so reuse: 6α × 1 × 3 seeds = 18 unique.
Condition (b): 6α × 4K × 3 seeds = 72. Condition (c): same 72. Total unique: **162 runs.**)

**Key per-client data sizes (α = 0.1, n_train = 940, N = 256):**

| K | n_client | params/data (client) |
|---|----------|----------------------|
| 5 | 188 | 396.3 |
| 10 | 94 | 792.5 |
| 20 | 47 | 1 585.0 |
| 50 | 18 | 4 138.7 |

**Note on extreme regimes.** The centralized-reduced condition at high K and low α
produces very small training sets. For example, α = 0.1 / K = 50 gives
n_reduced = ⌊0.002 × 9409⌋ = 18 samples — fewer than the 97 output classes.
These settings will almost certainly fail to grok, which is informative: if FL
*does* grok with the same per-client data, it proves aggregation is essential.

**Outputs.**
- **Figure 2 (main result):** For each α, plot grokking step vs K with three
  curves: centralized-full (flat), centralized-reduced (increasing),
  FL (the interesting one). Multiple panels for different α values.
- Alternatively: 2D heatmap of grokking step over (α, K) grid.
- The gap between centralized-reduced and FL curves quantifies the
  *aggregation benefit*. The gap between centralized-full and FL quantifies
  the *fragmentation cost*.

**Decision point.** If FL α_crit = centralized α_crit at all K values tested,
the conclusion is "FL does not affect the grokking phase boundary." The rest of
the experiments then focus on understanding *why* (mechanistic robustness) rather
than *where it breaks*.

---

### Experiment 3 — Heterogeneity at the Phase Boundary

**Question.** Does data heterogeneity shift the grokking phase boundary, and does
the *type* of heterogeneity matter?

**Hypothesis.** Operand-based partition disrupts grokking more than Dirichlet
(quantity skew) at matched imbalance levels, because it fragments the Fourier input
structure that the model must learn. Target-based partition disrupts the output space
but preserves input structure and may be less harmful.

**Design.** Fix K from Exp 2 (choose the K value that showed the most interesting
behaviour; default: K = 10). Use α values near α_crit from Exp 1.

**Sub-experiment 3a: Dirichlet sweep (continuous heterogeneity knob).**

| Parameter | Values |
|-----------|--------|
| α | {α_crit − δ, α_crit, α_crit + δ, 0.3, 0.5} (~5 values) |
| α_dir | {0.01, 0.1, 0.5, 1.0, 10.0, 1000.0} (6 values, 1000 ≈ IID) |
| K | fixed (from Exp 2) |
| E | 5, f = 1.0 |
| seeds | {42, 123, 456} |

**Runs.** 5 × 6 × 3 = **90 runs.**

**Sub-experiment 3b: Structured partition (operand vs target vs IID).**

| Parameter | Values |
|-----------|--------|
| α | same 5 values as 3a |
| partition | {iid, operand, target} |
| K | same as 3a |
| E | 5, f = 1.0 |
| seeds | {42, 123, 456} |

**Runs.** 5 × 3 × 3 = 45. IID runs overlap with 3a (α_dir = 1000), so
5 × 3 = 15 are reused. **30 new runs.**

**Combined Exp 3 unique runs:** 90 + 30 = **~120 runs.**

**Outputs.**
- **Figure 3a:** Phase diagram heatmap — x-axis: Dirichlet α_dir (log scale),
  y-axis: training fraction α. Colour: grokking step (∞ = white/hatched).
  The phase boundary is a curve in this space.
- **Figure 3b:** Structured partition comparison — for each α, grouped bars
  showing grokking step for IID / operand / target. If operand partition is
  worse than Dirichlet-matched heterogeneity, this supports the Fourier
  fragmentation hypothesis.

---

### Experiment 4 — Communication Efficiency at the Phase Boundary

**Question.** How much can communication (rounds) be reduced before grokking breaks?
Is the communication budget or the total computation budget the binding constraint?

**Hypothesis.** Grokking requires a minimum number of aggregation events (rounds) to
align client representations. In the easy regime (α = 0.5), this minimum is very low
(our breaking-point results show E = 200 / R = 100 still works). Near the phase
boundary, more frequent aggregation (lower E or higher f) should be necessary.

**Design.** Run two matched sub-experiments to disentangle communication from
computation.

**Sub-experiment 4a: Fixed total steps (S = 30 000), vary communication frequency.**
Measures the effect of communication at constant total computation.

| E | R = S / E | Communication events | Notes |
|---|-----------|---------------------|-------|
| 5 | 6 000 | 6 000 | baseline |
| 10 | 3 000 | 3 000 | |
| 25 | 1 200 | 1 200 | |
| 50 | 600 | 600 | |

E = 1 is excluded from the fixed-S design because R = 30 000 is impractical
(~4 hr per run). It is tested in Sub-experiment 4b instead, where R is capped.

| Parameter | Values |
|-----------|--------|
| α | {α_crit + δ, 0.3, 0.5} (one near-boundary, two comfortable) |
| f | {0.4, 1.0} |
| K | same as Exp 3 |
| seeds | {42, 123, 456} |

**Runs.** 4E × 3α × 2f × 3 seeds = **72 runs.** (All federated.)

**Sub-experiment 4b: Fixed total rounds (R = 2 000), vary local computation.**
Measures the practical tradeoff: more local work per round vs fixed comm budget.
Also the only place E = 1 is tested (R = 2 000 is practical).

| E | S = R × E | Notes |
|---|-----------|-------|
| 1 | 2 000 | low-S; may not grok — that is informative |
| 5 | 10 000 | baseline |
| 10 | 20 000 | |
| 25 | 50 000 | |

| Parameter | Values |
|-----------|--------|
| α | same 3 values |
| f | 1.0 |
| K | same as Exp 3 |
| seeds | {42, 123, 456} |

**Runs.** 4E × 3α × 3 seeds = **36 runs.**

**Outputs.**
- **Figure 4a:** Grokking step vs local epochs (log scale), one line per (α, f) combo.
  Steeper rise near α_crit means communication matters more at the boundary.
- **Figure 4b:** Same, but x-axis is communication rounds (= R), showing the practical
  cost. Mark: "communication rounds to grokking" (R_grok = T_grok / E) — the actual
  number of server-client exchanges needed.

---

### Experiment 5 — Algorithm Comparison & Regularisation

**Question.** When FedAvg fails (or is delayed), can better algorithms or explicit
regularisation recover grokking?

**Hypotheses.**
- **FedProx** constrains client drift → preserves Fourier feature alignment → recovers
  grokking. Effective μ should increase with heterogeneity and local epochs.
- **FedAdam** (adaptive server-side optimiser) compensates for heterogeneous client
  updates and may converge faster.
- **Weight decay** provides explicit regularisation. In centralized training, wd > 0
  promotes grokking (Gromov 2023). In FL, it might serve the same role while also
  implicitly controlling client drift (penalising large weight deviations).

**Design.** Select 2–3 "hard" settings from Exps 2–4 where FedAvg either fails to grok
or is significantly delayed compared to centralized. Candidate settings (contingent on
Exps 1–4 results):
- **(H1)** α near α_crit, K = 20, IID, E = 5 (data scarcity + fragmentation)
- **(H2)** α near α_crit, K = 10, Dirichlet α_dir = 0.1 (data scarcity + heterogeneity)
- **(H3)** α = 0.3, K = 10, E = 25, f = 0.4 (communication-limited)

If no FedAvg failure is found in Exps 2–4, use the *most delayed* setting instead
and test whether algorithms can *accelerate* grokking.

For each hard setting, sweep:

| Algorithm | Parameters |
|-----------|-----------|
| FedAvg (baseline) | — |
| FedProx | μ ∈ {0.001, 0.01, 0.1, 1.0} |
| FedAdam | server_lr ∈ {0.01, 0.1, 1.0}, τ = 1e-3 |
| FedAvg + weight decay | λ ∈ {0.01, 0.1, 1.0} |

**Runs.** 3 settings × (1 + 4 + 3 + 3) algorithms × 3 seeds = **99 runs.**

**Outputs.**
- **Figure 5:** For each hard setting, bar chart of grokking step across algorithms.
  Include IPR at the grokking step to show that successful algorithms produce
  higher Fourier structure.
- Table with final test accuracy, grokking step, and IPR for all configurations.

**Implementation note.** FedAdam requires adding Flower's `FedAdam` strategy.
This is straightforward — Flower provides it natively; only the strategy construction
in `fed_train.py` needs to be parameterised.

---

### Experiment 6 — Mechanistic Analysis

**Question.** *Why* does grokking survive (or fail) under FL? What is the mechanism?

**Hypothesis.** Grokking requires all clients to converge on the same Fourier
features in W1. When client drift is small relative to the basin of attraction of
the Fourier solution, aggregation produces a coherent global model and grokking
proceeds. When drift is large enough that clients learn *incompatible*
representations, aggregation destroys structure and grokking fails.

**Design.** No additional runs — this is post-hoc analysis of runs from Exps 1–5.
However, it requires **new metrics** collected during training:

#### 6.1  New metrics to implement

| Metric | Definition | Where to compute |
|--------|-----------|-----------------|
| **Client drift** | d_k = ‖w_k^{after} − w_global^{before}‖_F averaged over all clients | Client fit() return value |
| **Client weight divergence** | σ_w = std({‖w_k‖_F}) across clients per round | Server-side after receiving updates |
| **Per-client IPR** | IPR computed on each client's local model before aggregation | Client fit() or server-side |
| **Fourier spectrum** | Full |W̃1(ν)|² spectrum (not just scalar IPR), saved at checkpoints | Server-side evaluate_fn |

These metrics have minimal computational overhead but require modifications to
`fed_train.py` to expose per-client information.

#### 6.2  Analysis plan

Select 4 representative runs:
1. **Healthy grok** — α = 0.5, IID, K = 5 (baseline)
2. **Boundary grok** — α ≈ α_crit, IID, K = 10 (barely works)
3. **Failed grok** — α < α_crit, or hard non-IID setting (doesn't grok)
4. **Rescued grok** — failed setting recovered by FedProx/FedAdam (from Exp 5)

For each, produce:

**Figure 6 (multi-panel):**
- **(a)** Test accuracy and train loss curves (4 runs overlaid)
- **(b)** IPR trajectories — show Fourier structure emergence
- **(c)** Mean client drift d̄ per round — show whether drift magnitude predicts failure
- **(d)** Scatter plot across ALL runs from Exps 2–5: mean client drift vs grokking step.
  Hypothesis: strong positive correlation (higher drift → later/no grokking).
- **(e)** Fourier spectra of W1 at three timepoints (before grok, during transition,
  after grok) for the healthy vs failed cases.

---

## 5  Summary of Run Counts

| Experiment | Focus | Runs | FL runs | Centralized runs | Runtime est. |
|------------|-------|------|---------|-----------------|-------------|
| 1. Centralized phase boundary | α sweep | 24 | 0 | 24 | ~1 hr |
| 2. Aggregation & FL boundary | α × K × {full, reduced, FL} | 162 | 72 | 90 | ~55 hr |
| 3. Heterogeneity | (α × Dirichlet) + structured | ~120 | ~105 | ~15 (overlap w/ Exp 2) | ~65 hr |
| 4. Communication | (E × f) fixed-S + fixed-R | 108 | 108 | 0 | ~50 hr |
| 5. Algorithm rescue | 3 settings × algorithm sweep | 99 | 99 | 0 | ~60 hr |
| 6. Mechanistic | post-hoc analysis | 0 | 0 | 0 | — |
| **Total** | | **~513** | **~384** | **~129** | **~231 hr** |

**Wall-clock estimate.** With 4-way parallelism: ~58 hr ≈ **2.5 days.**

*Runtime assumptions: ~35–50 min per FL run (R = 6 000, E = 5), scaling with K.
~2 min per centralized run (50k epochs). Flower/Ray overhead dominates FL runtime.
Runs within each experiment are fully independent and parallelisable.*

---

## 6  Implementation Requirements

### 6.1  Code changes needed

| Change | File | Effort |
|--------|------|--------|
| Add `FedAdam` strategy option | `federated/train.py`, `federated/config.py` | Small — Flower provides `FedAdam` natively |
| Track client drift metric | `federated/train.py` (client fit + server callback) | Medium — return ‖Δw‖ from client, aggregate server-side |
| Track client weight divergence | `federated/train.py` (server aggregation callback) | Medium |
| Save Fourier spectrum at checkpoints | `core/metrics.py`, `federated/train.py` | Small |
| New sweep modes for Exps 1–5 | `main.py`, `fed_main.py` | Medium — parameterise from experiment tables above |
| Phase diagram visualisation | `federated/visualize.py` | Medium |

### 6.2  New config parameters

```python
# federated/config.py additions
strategy: str = "fedavg"        # "fedavg", "fedprox", "fedadam"
server_lr: float = 1.0          # server-side learning rate (FedAdam)
tau: float = 1e-3               # FedAdam adaptivity parameter
track_client_drift: bool = True # enables per-round drift logging
```

---

## 7  Execution Order & Dependencies

```
Exp 0 (width validation)
  │
  ├── determines N* (128 or 200)
  │
  ▼
Exp 1 (centralized boundary)          ← uses N*
  │
  ├── determines α_crit
  │
  ▼
Exp 2 (FL boundary + aggregation)     ← uses α values around α_crit
  │
  ├── determines best K for Exps 3–4
  ├── identifies whether FL shifts boundary
  │
  ▼
Exp 3 (heterogeneity)                 ← uses K from Exp 2, α near boundary
Exp 4 (communication)                 ← can run in parallel with Exp 3
  │
  ├── Exps 3–4 identify "hard" settings
  │
  ▼
Exp 5 (algorithm rescue)              ← uses hard settings from Exps 2–4
  │
  ▼
Exp 6 (mechanistic analysis)          ← post-hoc on all runs
```

**Parallelism opportunities:**
- Exp 1 runs are fast (centralized, small model) → run all 24 first
- Within Exp 2: all (α, K, condition) triples are independent → fully parallel
- Exps 3 and 4 are independent of each other → parallel
- Within Exp 5: all algorithm × setting triples are independent → parallel

---

## 8  Contingency Plans

### 8.1  If no centralized phase boundary exists (grokking at all α)

This would mean modular addition groks even with ~30 training pairs (α = 0.03).
If observed:
- Extend sweep to α ∈ {0.01, 0.02} (9–18 training samples)
- Switch to a harder task (e.g., `x2_plus_y2`) which may have a higher α_crit
- The finding "grokking is universal for this architecture" is itself publishable

### 8.2  If FL never breaks grokking (even at α_crit)

This would mean federated aggregation is always sufficient to preserve grokking.
- The Exp 2 aggregation isolation becomes the key result: FL matches centralized-full
  even though each client has centralized-reduced data → aggregation is the mechanism
- Focus the paper on the *robustness* story rather than the *fragility* story
- The mechanistic analysis (Exp 6) explains WHY: the Fourier solution is a global
  attractor robust to noisy gradient estimates

### 8.3  If FL helps grokking (FL α_crit < centralized α_crit)

This is the most surprising and publishable finding:
- Aggregation acts as implicit regularisation (averaging noisy client models
  suppresses memorisation, promotes generalisation)
- Add experiment: vary K at α < centralized α_crit to find the *optimal* K
- Connect to existing theory on "implicit regularisation from distributed training"

---

## 9  Expected Paper Figures

| # | Content | Experiment | Role |
|---|---------|-----------|------|
| 1 | Training dynamics: centralized vs FedAvg-IID | Baseline (existing) | Reproduce + validate |
| 2 | Grokking step vs K: centralized-full / centralized-reduced / FL | Exp 2 | **Main result**: aggregation effect |
| 3 | Phase diagram: training fraction × heterogeneity | Exp 3 | Non-IID effects |
| 4 | Communication efficiency: grokking step vs local epochs | Exp 4 | Practical FL tradeoff |
| 5 | Algorithm rescue: bar chart across algorithms | Exp 5 | Practical contribution |
| 6 | Mechanistic: IPR + client drift + Fourier spectra | Exp 6 | Explains the "why" |

**Appendix figures:**
- Centralized phase boundary (α sweep) — Exp 1
- Structured partition comparison (operand vs target) — Exp 3b
- Full 2D heatmaps for all Exp 2 conditions
- Activation function generality (quadratic vs ReLU vs GELU)
- Task generality (addition vs other operations)

---

## 10  Reproducibility Checklist

- [ ] All random seeds recorded and fixed
- [ ] Exact software versions pinned (Python 3.12, PyTorch, Flower, Ray)
- [ ] All history JSON files saved with full metric trajectories
- [ ] Code versioned with git tag per experiment phase
- [ ] Hyperparameter tables in appendix match code defaults
- [ ] 3 seeds per configuration with mean ± std reported
- [ ] Grokking step definition applied uniformly across all experiments
- [ ] S_max = 50 000 sufficient for all observed grokking events (verify post-hoc)
- [ ] Centralized-reduced comparison uses strict subset of original training set (same seed permutation → verified by construction)

---

## 11  Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Flower simulation too slow at R > 5 000 | Medium | Default R = 6 000 (E = 5); E = 1 only in Exp 4b (R = 2 000); profile early and optimise if needed |
| α_crit is extremely low (< 0.03) | Low | Switch to harder tasks; adjust p |
| FedAdam diverges with lr = 50 client-side | Medium | FedAdam uses server-side adaptive LR; client LR stays at 50. If unstable, sweep client LR too |
| Optimizer state restart (SGD → no issue; AdamW → known limitation) | Noted | Primary experiments use SGD; note limitation for AdamW runs |
| Per-round metric logging slows simulation | Low | Drift metrics are cheap (one norm computation per client); Fourier spectra saved only at checkpoints (every 100 rounds) |
| 546 runs × infrastructure failure | Medium | Checkpoint partial results; idempotent run scripts; track completion in results directory |
