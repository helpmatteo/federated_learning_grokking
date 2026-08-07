# v2-multisetup — progress and resume notes

Working branch for the multi-setup rewrite. Current plan:
`~/.claude/plans/plan-all-that-needs-valiant-hamster.md` (the setup/check phase).
Closed plans: `plan-all-that-needs-nested-seal.md` (boundary campaign),
`gate-a-closeout.md` (Gate A).

**Current as of 2026-08-07.** v2 code complete across all plan phases (0–5); Gate A
closed; **Phase 1 run, and its four decision rules answered** (RESULTS.md §13). The
**The setup/check phase is complete.** The K ceiling — the last thing gating it —
resolved: there was no collapse, and the K axis is open at a measurable cost
(below). The work in front is now writing the per-setup FL manifests and running
v1's exp0–exp7 chain on them.

> **Read the data, not just this file.** Sweeps get banked faster than these notes
> get rewritten. Ground truth is `results/data/runs_v2.csv` and
> `results/data/runs/*.json`; the "what is missing" recipe is in the Resume section.

## State

| | |
|---|---|
| Branch | `v2-multisetup` (branched from `main` @ `41c3fa8`; `main` has nothing this lacks) |
| Frozen reference | tag `v1-single-setup` — the state that produced the 32 figures in `results/figures/` |
| Tests | **537/537 pass** (`venv/bin/python -m pytest tests/ -q`, ~9 min incl. FL integration) |
| Runs banked | **842** in `results/data/runs_v2.csv` (572 grokked, 270 censored) + 870 v1 runs in `runs.csv`. 237 machine-hours |
| Setups | A quad-MLP/mod-97 · A′ quad-MLP/mod-97/AdamW (**measured**, §13.3) · B transformer/mod-113 · C transformer/S₅ · D quad-MLP/S₅ · E MLP/MNIST-1k. **D′ dropped** — the gate that would have required it opened (§13.7) |
| FL algorithms | FedAvg, FedProx, FedAvgM, FedYogi, FedAdam (native) + SCAFFOLD (adapted, **raises under AdamW by design**) |
| Statistics | censored survival (KM median + fraction-grokked + bootstrap CI); `scripts/summarize_runs.py` |
| deps | torch 2.10 + torchvision 0.25 (pinned pair); flwr 1.27 |
| Env | `venv/` (py3.10, torch 2.10, flwr 1.27); package installed via `pip install -e . --no-deps` |
| Hardware | 8× L4, **shared with other users** — check `nvidia-smi` before sizing; the launcher auto-skips busy GPUs |

## RESOLVED: there was never a K≈30 collapse

**The blocker is cleared and the K axis is open** (RESULTS.md §13.7). The whole
thing was a censored measurement, three times over.

The table that defined the blocker — setup B, peak train 100 / 98.2 / 42.8 / 5.9 /
5.0 at K = 10 / 20 / 30 / 40 / 50 — was read off runs given **10,000 steps**. At
K=30, memorisation alone takes **12,900**. Given 100,000 steps the same
configuration memorises and groks at 13,200. "42.8%" is where that run had got to
when the clock stopped.

**What federation actually does**, once `t_memo` is recorded separately from
`t_grok` (setup B, iid, E=5, α=0.30, wd=0.1):

| | t_memo | T_grok | delay |
|---|---|---|---|
| centralized | 150 | 45,050 (3/3) | 44,900 |
| K=20 | 3,600 | 93,700 (**3/3**) | 90,100 |
| K=30 | 5,900 | 94,300 (1/3) | 88,400 |
| K=50 | 53,300 | censored (0/3) | — |

Two separable effects, and only one depends on K:

1. **Memorisation slows steeply with K** — 24×, 39×, then 355× the centralized
   value at K = 20, 30, 50.
2. **The delay is large and roughly doubles under federation** — ~90,000 against
   the centralized 44,900, established at K=20 across 3 seeds.

**Whether the delay grows with K is not established.** It rests on two K values —
K=20 (3 seeds: 62,800 / 90,100 / 94,300) and K=30 (**one** seed: 88,700) — and the
K=30 point falls inside K=20's spread, which is itself wider than any K-effect it
could detect. Flat is consistent with the data; so is a mild increase.

So `T_grok(K) ≈ t_memo(K) + delay` accounts for the censored cells: K=30 needs
~95,900 against the 100,000 it got (hence 1/3, at 94,300), and K=50 needs ≳143,000
and got 100,000. Treat 143,000 as a **lower bound** — if the delay grows with K,
budgeting exactly to it recreates the censoring it came from.

**Consequences for the campaign:**

- **D′ is not needed.** The decision rule's first branch fired, so the A-vs-D
  optimiser confound is recorded as a stated limitation rather than fixed with a
  seventh setup. `p1_dprime_alpha` is not written and should not be.
- **Budgets come from `t_memo(K) + delay`**, never from a multiple of the
  centralized number. That is the rule every per-setup FL manifest must follow.
- **B's K ladder is bounded by cost, not capability.** K=97 extrapolates well past
  200,000 steps, so either B stops at K=50 or the campaign accepts one expensive
  high-K cell per setup.
- **The additive model is worth confirming before it is budgeted from**: 3 runs at
  K=50, 30,000 rounds, predicted to grok at ~143,000 steps.

The only cell that still looks like a genuine wall is **K=50 at the inherited
wd=1.0**, which fails to memorise even at 100,000 steps — though at peak train
37.8% against the 5.0% recorded at 10,000, so it too is slow rather than stuck.

**Ruled out along the way:** weight-norm collapse, client drift, and local step
size (lower lr is *worse* — 4.6% train at lr=1e-4).

Careful with the number: lr·λ=1e-3 is *also* where the anchor's WD sweep found
decay outrunning learning, but that sweep was **GD** (coupled decay) and
RESULTS.md §6.4 makes the point that the optimiser is exactly what discriminates.
The numerical coincidence is not the argument; the federated wd evidence is.

## The optimiser is a control variable, and it is not currently controlled

The campaign is a 2×2 (architecture × task). A is GD+MSE (Gromov, inherited from
v1); B, C, D are AdamW+CE. So:

| comparison | isolates | verdict |
|---|---|---|
| B vs C | task, on the transformer | **clean** |
| C vs D | architecture, on S₅ | **clean** |
| A vs B | architecture, on modular | confounded (optimiser + loss) |
| A vs D | task, on the quad-MLP | confounded (optimiser + loss) |

B/C/D are internally consistent; A is the odd one out and **cannot move** — it is
Gromov's config, the anchor to 870 v1 runs, and every banked federated result. So
the fix is the missing cell, **A′** (`SETUP_A_PRIME`): A's architecture and task
under AdamW, MSE not CE so that A vs A′ is a single-variable optimiser contrast.

Provenance of the AdamW choices, since only two are inherited:

- **B** — Nanda's published config verbatim (lr=1e-3, wd=1.0, CE). Its value is
  fidelity; do not tune it or it stops being a replication.
- **E** — Omnigrok's family, but wd=0.1 was **measured here** (`t0_mnist_wd_band`).
  The only AdamW setup whose decay was chosen for the setup it runs in — and the
  only one that does not collapse at K≥20.
- **C** — inherited from B, and since **measured** (`p1_cd_decay_band`): wd=1.0
  is optimal for C too, so the inheritance was lucky rather than justified.
- **D** — **inherited from nothing.** Gromov's architecture running Nanda's
  optimiser. Now measured twice over: wd=1.0 is the only band D groks in at all
  (`p1_cd_decay_band`), and the quad-MLP *does* grok S₅ under GD+MSE
  (`p1_d_gd_probe`). The confound is therefore *closable* via D′ — but the K gate
  opened without needing it (§13.7), so **A vs D stays confounded and that is
  recorded as a stated limitation** rather than fixed with a seventh setup.

## Phase 1 — RUN. The pilots that define the setups.

All four executed; full readings and tables in **RESULTS.md §13**. Each manifest's
docstring carries the decision rule it was written against — read it before the
results, not after.

| manifest | runs | verdict |
|---|---|---|
| `p1_d_gd_probe` | 9/9 | **The quad-MLP does grok S₅ under GD+MSE** — lr=50 → 3/3 at 22,600 (α=0.5); lr∈{5,10} → 0/3. D′ was the gated response; the gate opened without it, so **D′ is dropped** and A vs D stays a stated limitation |
| `p1_b_decay_band` | 15/15 | **B's decay does not move either** — wd=1.0 is optimal, and every cell memorises at epoch 150, so decay acts purely on the generalisation timescale. Supplies the control the K-collapse arm was missing |
| `p1_cd_decay_band` | 30/30 | **Neither C's nor D's decay moves.** D groks only at wd=1.0 (0/3 everywhere else); C ties 3/3 at wd∈{0.1,0.3,1.0} and wd=1.0 wins on both statistics. New: **C is unstable** — it dips back below the bar after crossing it, up to 28 times at wd=0.3 |
| `p1_aprime_alpha` | 45/45 | A′'s cliff sits between α=0.175 and 0.20, **below A's**, and it groks **~45× faster at every shared α**. Its only usable federated working point is **α=0.20** — above that there is no delay left for federation to disrupt |
| `p1_k_collapse_budget` | 11/11 | **THE GATE, and it opened.** wd=0.1 groks **3/3 at K=20** given 200,000 steps. `T_grok ≈ t_memo(K) + 90,000` predicts every previously-censored cell; §12's headline table was itself measured at 10,000 steps and is superseded |
| `p1_k_collapse_wd` | 18/18 | **Graded recovery.** wd=0.1 restores memorisation fully at K=20/30, partially at K=50 (~78% peak). Its zero-generalisation cells turned out to be **censored at 10,000 steps against a measured 45,050 requirement** — `p1_k_collapse_budget` re-runs them |

**Phase 2's hidden cost did not materialise.** The plan reserved ~60 runs to
re-measure C's and D's α ladders in case their decay moved. Neither moved, so both
Gate A ladders stand as written. That is Phase 1's largest saving.

**One gap Phase 1 missed, now closed:** `p1_cd_decay_band` measured C and D but
not **B** — the setup carrying the K-collapse diagnosis. `p1_b_decay_band` supplied
the missing centralized reference, and it immediately paid for itself by showing
the federated arm had been reading a censored cell as a result.

## How to run a sweep now

```bash
venv/bin/python scripts/build_manifests.py               # regenerate manifests/
venv/bin/python scripts/launch_sweep.py manifests/t0_wd_grid.jsonl --dry-run
venv/bin/python scripts/launch_sweep.py manifests/t0_wd_grid.jsonl --per-gpu 2
venv/bin/python scripts/collect_runs.py                  # per-run JSONs -> results/data/runs_v2.csv
```
Resume is automatic (skips runs whose result JSON exists). One run = one subprocess with `CUDA_VISIBLE_DEVICES` pinned.

## Done — Phase 0a + Phase 0 (all committed)

- **0a Branch + reorganise** (`6891ebd`, `d93babc`) — library at `src/fedgrok/`
  (`core/ models/ data/ metrics/ training/ analysis/`), plotting at `scripts/plotting/`,
  moves via `git mv`, imports rewritten, `pyproject.toml` with pinned deps. Two colliding
  `exp7_*` modules → `exp_task_generality.py` / `exp_mechanistic_checkpoints.py`.
- **0.7 Log harvest** (`7af94e4`) — `scripts/harvest_logs.py` → `results/data/runs.csv`,
  870 runs (554 grokked, 316 censored). `runs_skipped.csv` records 7 empty/crashed logs.
- **0.1/0.2** (`099622e`) — `check_decay_stability` guard; one-hot on label device.
- **0.3/0.4** (`3f518be`) — division on `p(p-1)` grid; single source of truth for the
  grid + split (`build_encoded_grid`, `split_indices`), verified byte-identical to old.
- **0.5/0.6** (`094a291`) — removed `should_abort`; `total_steps` now counts real
  gradient work (compute-matched), with `sequential_steps` and `n_participating` alongside.
- **Verification** (`c7c997f`) — `tests/test_fedavg_identity.py`: FedAvg E=1 ≡ centralized
  GD to 1e-4 across all partitions and K∈{2,4,12}, with an E=5-diverges anti-test. Surfaced
  and fixed two bugs (empty Dirichlet shard → float64 index error / nan-poisoned aggregate;
  now raises).

## Done — Phase 1.1 (`639efe7`)

- Per-client warm cache (`_client_cache`) + reused eval model + `eval_every` field +
  Ray dashboard/metrics-exporter disabled, in `src/fedgrok/training/federated.py`.
- **Verified numerically equivalent** to pre-change code (`c7c997f`) via a worktree diff:
  worst |Δ| = 9.5e-7 across all series (= reference's own fp32 aggregation noise). Caught
  and fixed an RNG-order bug (eval-model creation was stealing the init-weight draw).
- Speed: ref times out >10s/round (per-round rebuild + fractional-GPU contention); cached
  is ~80 ms/round at K=10. `eval_every` barely moves wall-clock — the Ray round-trip
  dominates — so it's for history size, not speed.
- Guard test `TestEvalEvery` added.

## Done — Phase 1.2 (`9ef8797`, + main/fed_main removal)

- `fedgrok/manifest.py` (spec→config, content-hash ids, grid expansion, JSONL I/O),
  `fedgrok/run.py` (single-run entry point, atomic result JSON — supersedes the deleted
  `main.py`/`fed_main.py`), `scripts/launch_sweep.py` (GPU-pinned subprocess pool,
  free-GPU autodetect, idempotent resume), `scripts/collect_runs.py`, `scripts/build_manifests.py`.
- `manifests/t0_wd_grid.jsonl` (45) + `t0_poly_pilot.jsonl` (6) generated and committed.
- Verified: two Flower/Ray sims coexist on one GPU (per-process local Ray); resume skips
  completed runs. +18 tests.

## Done — Phase 2 core (`b50ab87`, `5e31b72`, `35ec12f`)

- **Model registry** — `fedgrok/core/registry.py::build_model(cfg)` on `cfg.model`
  (default `"groknet"`). Wired into both training loops.
- **Loss registry** — `build_loss(cfg) -> LossSpec(loss_fn, prepare_target)`; `cfg.loss`
  ∈ {mse, ce}. MSE = one-hot targets, CE = class indices. Wired through every target site.
  **Critical fix**: `_cfg_to_fit_config`/`_fit_config_to_cfg` now carry `loss` and `model`
  (they didn't — a CE federated run would have silently trained clients with MSE).
  CE verified to train end-to-end (mod-97 MLP, CE+AdamW → 100/99.9%).
- **Metric capability protocol** — `fourier_applicable(model)`; both loops guard IPR /
  weight-norms / Fourier-spectrum / per-client-W1 and log NaN off-GrokNet, so Phase 3's
  transformer/MNIST/S₅ won't crash on them.
- All three verified MSE/GrokNet-equivalent to the archived baseline (4.8e-7). +21 tests.

## Done — Phase 2 complete + Phase 3a (`2d62601`, `6fd667e`)

- **Minibatching** (`2d62601`) — `Config.batch_size` (0 = full-batch default). Both loops
  branch on it; `randperm` only under batch_size>0 so full-batch is bit-identical (centralized
  0.0 vs pre-commit, FL 9.5e-7 vs baseline). Threaded through the fit-config.
- **Nanda transformer** (`6fd667e`) — `fedgrok/models/transformer.py::GrokFormer`, registered
  as `model="transformer"`, pair with `loss="ce"`. Consumes the same one-hot 2p input (one-hot
  × embedding = lookup), so no dataset change. **Verified it groks**: mod-113, 30% train, AdamW
  lr 1e-3 wd 1.0, CE → train≥99% at epoch 200, test≥95% at epoch 6200 (6000-epoch gap),
  ends 100/100. Fourier metrics NaN throughout (capability protocol working).

## Done — Phase 3b/3c-core (`a36072d`, `365a7c1`)

- **MNIST-1k** (`a36072d`) — dataset registry (`build_dataset`/`dataset_dims`, dispatching on
  `cfg.dataset`; modular byte-identical), `data/mnist.py` (torchvision 0.25, pinned to torch
  2.10), `models/mlp.py` (generic ReLU MLP + `init_scale` large-init knob). Pipeline reaches
  ~91% test. **Grok not yet located** — Omnigrok's clean grok is in a narrow weight-decay band
  (weak decay → gap stalls; wd=1.0 → 91% with no delay). Sweep queued: `manifests/t0_mnist_wd_band.jsonl`.
- **S5 core** (`365a7c1`) — `data/groups.py` (S_n composition, "s5" dataset). Model builders
  size from `dataset_dims` (2|G|/|G|), not `cfg.p`. `fourier_applicable(model, cfg)` now
  dataset-aware (False on S5). **Robust fit-config**: whole dataclass round-trips via
  `dataclasses.asdict` — no more hand-listed fields dropping loss/model/batch_size; guarded by
  a completeness test. **S5 groks** on the transformer (train≥99% @200, test ~1% for ~14k
  epochs then → 92%).

## Done — Phase 3c complete (`6dd4a4b`)

- **S5 coset partition** — `data/groups.py::coset_labels` (S4 → 5 cosets, A5 → 2);
  `data/partition.py` "coset" mode shards clients by the coset of each sample's first operand.
  `make_federated_datasets` is now dataset-aware (`dataset_grid`/`has_grid`), so S_n FL works;
  MNIST FL raises a clear "not wired" error. Modular FL bit-identical (4.8e-7).
- **Coset-attribution metric** — `metrics/nonabelian.py::coset_attribution` (coset_accuracy +
  coset_purity), the S5 analogue of IPR. Verified at chance (0.197≈1/5) untrained, 1.0 perfect.

## Done — Phase 4: FL algorithms (`7293081`, `3fa8f27`, `72f01e5`)

- **Native strategies** (`7293081`) — FedAvgM (server momentum = DiLoCo outer) + FedYogi added
  to `_build_strategy`; FedAdam/FedProx already there. FedAvgM(lr=1,mom=0) verified == FedAvg.
- **SCAFFOLD** (`3fa8f27`) — the load-bearing drift correction. flwr_baselines is gone and Flower
  has no standalone scaffold baseline, so this adapts Flower's niid_bench Option-II via the metrics
  channel: `fedgrok/training/scaffold.py` (per-client c_i module-dict, `apply_correction`,
  `client_cv_update`, `ScaffoldStrategy`), plus a scaffold branch in `GrokClient.fit` and the
  server_cv wiring in `fed_train`. **Round-1 reduces to FedAvg exactly** (c=c_i=0 → zero correction,
  verified diff=0); FedAvg itself stays bit-identical (9.5e-7). **FedDyn dropped** (no Flower
  reference; FedProx + SCAFFOLD span the two drift-correction mechanisms).
- **Server-LR calibration** (`72f01e5`) — `manifests/t3_server_lr_calibration.jsonl` (42 runs)
  sweeps server_lr for FedAdam/FedYogi/FedAvgM so each is tuned fairly (fixes the exp5 defect).

## Done — Phase 5 part 1: censored survival statistics (`3563d17`)

`fedgrok/analysis/survival.py` (kaplan_meier, km_median, fraction_grokked, bootstrap_ci);
`summarize_seeds` now reports fraction-grokked + KM median + CI instead of inf-if-any-fail;
`scripts/summarize_runs.py` builds the survival table from any runs CSV. Demonstrated on the
real exp5 H2 data (FedProx-0.001: old `inf` → fraction 0.33 surfaced; FedAdam-0.1 KM median
2875 [2450,3100]).

## Results so far (v2 pipeline, `results/data/runs_v2.csv`)

**T0 weight-decay grid — the review's blocker #1, resolved** (45 runs, GD lr=50, p=97,
α=0.5, MSE, 5 seeds, KM median [95% CI]):

| lr·λ | λ | grokked | T_grok |
|---|---|---|---|
| 0 | 0 | 5/5 | 7600 [7500, 7800] |
| 1e-5 | 2e-7 | 5/5 | 7700 [7500, 7800] |
| 1e-4 | 2e-6 | 5/5 | 8300 [8200, 8600] |
| 1e-3 | 2e-5 | 0/5 | censored |
| 1e-2 | 2e-4 | 0/5 | censored |

The v1 "WD prevents grokking at every strength" was an artifact — but **not** a sign flip
(which is what I predicted when first diagnosing it). Grokking is *preserved* up to
lr·λ = 1e-4; the old values (lr·λ 0.5–50) sat far past the usable band. The real boundary is
lr·λ ≈ 1e-3, where train accuracy never exceeds 8% and decays back to 1% — decay outruns
learning, so memorisation never happens and there is nothing to grok from. Within the band WD
is neutral-to-mildly-slowing, consistent with Gromov (this setup groks unregularised); the
"WD accelerates grokking" result is specific to the AdamW/CE transformer regime.
**Caveat:** the AdamW arm hits 100/100 by epoch 200 at α=0.5 regardless of λ (no delayed
generalisation at all), so it can't inform the WD question here — rerun nearer the boundary.

**T0 polynomial pilot** (3 seeds): `x2_plus_y2` groks 3/3 (KM median 7000 [6900, 7000]) →
kept in the operation set; `x2_y2_xy` censored 0/3 → correctly excluded. The gate worked.

**T1 probe — the breakdown gate** (18/24; the E=250 cells were cancelled for cost). K=10,
α=0.3, p=97, 3 seeds, fixed 10k rounds:

| E | partition | steps | grokked | KM median T_grok |
|---|---|---|---|---|
| 1 | iid / operand | 10,000 | 0/3 | censored — **by budget, not federation** |
| 5 | iid | 50,000 | 3/3 | 12900 [12900, 13700] |
| 5 | operand | 50,000 | 3/3 | 12700 [12600, 13400] |
| 50 | iid | 500,000 | 3/3 | 23000 [22000, 25000] |
| 50 | operand | 500,000 | 3/3 | 17000 [16000, 18000] |

- **The E=1 identity reproduced in the wild.** iid and operand gave *identical* test
  accuracies seed-for-seed (33.5 / 34.1 / 17.2) — at E=1 FedAvg is exactly centralized GD
  regardless of partition, which is the `test_fedavg_identity.py` property in a real sweep.
- **Local steps delay grokking**: 12,900 → 23,000 from E=5 to E=50 (iid), ~1.8× on the
  compute-matched `total_steps` axis. Real, not a budget artifact.
- **The E=1 cells are NOT breakdown evidence** — 100% train with ~30% test looks like the
  gate condition, but they only got 10,000 steps against the ~12,900 needed.
- Curiosity for T2 to confirm at 5 seeds: operand groks *faster* than iid at E=50
  (17,000 vs 23,000).

**GATE VERDICT: not passed — the story at K=10 is delay, not breakdown.** No cell fails to
grok for a federated reason. Per the plan this means don't treat "FL breaks grokking" as
established; the breakdown search moves to higher K / stronger heterogeneity
(`manifests/t2_k_breakdown.jsonl`, launched).

**T2 K-breakdown — COMPLETE** (60/60, 2026-07-28). K∈{5,10,20,50} × {iid, operand,
dirichlet} × 5 seeds, α=0.3, E=5, p=97, 10k rounds. KM median [95% CI]:

| K | iid | operand | dirichlet |
|---|---|---|---|
| 5 | 13200 [12700, 13500] | 13100 [12600, 13400] | 13400 [12700, 13600] |
| 10 | 13400 [13400, 13700] | 13200 [13200, 13500] | 13600 [13000, 13900] |
| 20 | 13700 [13200, 14100] | 13300 [12700, 13500] | 13900 [13200, 14200] |
| 50 | 15200 [14600, 16000] | **13700 [13000, 14000]** | 15400 [14700, 16100] |

- **Every cell 5/5, zero censoring.** The α=0.3 plane is dead as a breakdown regime: a 10×
  increase in K buys ~16% delay and nothing else. This is the control arm, and it is clean.
- **Structured heterogeneity ACCELERATES grokking, and the effect scales with K.** At K=50
  the operand partition is significantly faster than iid (non-overlapping CIs); the gap is
  ~0 at K=5, ~400 steps at K=20, ~1500 steps at K=50. Operand at K=50 is no slower than
  iid at K=5.
- **It is structure, not heterogeneity.** Dirichlet (unstructured non-IID) tracks iid
  exactly at every K. So this is not "non-IID helps" — only the *coherent* partition helps.
- **Mechanism hypothesis for T4:** an operand shard is a coherent slice of the group rather
  than a random sample, so clients plausibly select a *shared* frequency basis and averaging
  reinforces it, where iid clients find arbitrary frequency sets that partially cancel.
  Testable with the per-client checkpoints the readiness fixes add.
- **Cost datum:** K=50 runs take ~75 min vs ~22 min at low K — Flower's per-round cost scales
  with actor count. Budget ~2–2.5 h/run at K=97; this reprices the 20-seed boundary cells.

## T2 BOUNDARY — COMPLETE (2026-07-29). The question is answered.

20/20, α=0.25, E=5, 20k rounds = 100k steps, 5 seeds, ~4.1 h/run at K=97.

| K | partition | grok | frac | KM median | 95% CI |
|---|---|---|---|---|---|
| 20 | iid | 5/5 | 1.00 | 29800 | [27300, 33100] |
| 50 | iid | 5/5 | 1.00 | 46800 | [40700, 51400] |
| 97 | iid | **2/5** | **0.40** | inf | [95600, inf] |
| 97 | **operand** | **5/5** | 1.00 | **76500** | [57800, 77500] |

K=97 per seed — iid: 95600, 98500, and three censored at 100k ending
**100% train / 53%, 61%, 71% test**. operand: 57800, 71000, 76500, 77200, 77500,
all reaching 100/100.

**1. The harness sanity check PASSED.** K=20 gives 29800 [27300, 33100] against
v1 exp2's 27215/29720/33025. The v2 rewrite reproduces exp2, so differences at
K=97 are attributable to federation and not to the rewrite. K=50 grokking 5/5 at
46800 confirms the 100k budget was genuinely sufficient — the other way this
sweep could have been worthless.

**2. v1's "K=97 breaks grokking" was a BUDGET ARTIFACT, as suspected.** Two of
five seeds grok, at 95600 and 98500. Under exp2's 50k budget every one of those
censors — which is exactly the 0/3 that was reported. The claim as stated does
not survive.

**3. But a real partial transition exists at K=97 iid.** 2/5, with the three
failures at 100% train and 53–71% test: memorised, not generalised.
**CAVEAT — the budget is still marginal here.** Both successes land within 5% of
the 100k ceiling, so the three failures may be mid-transition rather than stuck.
Separating "slower" from "broken" at K=97 needs MORE BUDGET, not more seeds.
That is the correction to the wave-2 plan: deepening seeds at 100k would sharpen
a fraction whose denominator is still budget-limited.

**4. THE HEADLINE: structured partitioning rescues K=97 completely.** operand is
5/5 at 76500 [57800, 77500] vs iid 2/5 at ~97000 — non-overlapping, and the
difference between reliable grokking and marginal failure. With dirichlet
tracking iid exactly at every K in `t2_k_breakdown`, the claim is:
**how you partition matters more than how far you fragment, and coherent shards
are strictly better than random ones.** This is a stronger and more useful result
than the breakdown originally being chased, and the per-client W1 checkpoints
(20/run, on disk) are what tests the frequency-consensus explanation for it.

**Next:** re-run K=97 iid at a larger budget (200k steps / 40k rounds, ~9 h/run)
to settle whether the 3/5 failures are censored or genuinely stuck. Everything
downstream should be re-scoped around the partition-structure result.

## MNIST-1k groks — all four setups now confirmed (2026-07-28)

`t0_mnist_wd_band` complete, 15/15, ~90 s per run (whole sweep under 4 min). Centralized,
3-layer MLP width 200, init_scale 9, MSE + AdamW, 1k subset, minibatch 200, 20k epochs.

| lr·λ | memorise (train≥99%) | generalise (test≥90%) | DELAY | grokked | peak test |
|---|---|---|---|---|---|
| 1e-5 | 600 | never | — | 0/3 | 89.2% |
| 3e-5 | 600 | 11100 | ~10500 | 3/3 | 91.1% |
| **1e-4** | 500 | **3800** | **3300** | 3/3 | **92.7%** |
| 3e-4 | 500 | 3000 | 2500 | 3/3 | 92.4% |
| 1e-3 | 800 | 3200 | 2400 | 3/3 | 91.5% |

- **Clean delayed generalisation.** Train hits 100% by epoch ~600 in every cell; test lags by
  2,400–10,500 epochs. Best band is **lr·λ = 1e-4** (highest test accuracy, clean 3,300-epoch
  delay) — use it for any follow-up. MNIST stays centralized-only by design
  (`make_federated_datasets` rejects it: no operand structure to partition on).
- **Weight decay ACCELERATES grokking here**, monotonically, and at lr·λ=1e-5 generalisation
  never completes. That is the published Omnigrok result reproduced.
- **This corrects an earlier explanation.** The modular sweep found WD neutral-to-slowing, and
  that was attributed to the "AdamW/CE transformer regime". Wrong: MNIST here is **MSE + AdamW**
  and accelerates, modular is **MSE + GD** and does not. The discriminator is the **optimizer**
  (AdamW's decoupled decay vs GD's coupled decay), **not the loss**.

**Threshold fix that this exposed** (`analysis/grokking_metrics.py`, `data/registry.py`):
`compute_t_grok` had a hardcoded 95% bar. MNIST-1k peaks near 93%, so all 15 runs were
recorded `t_grok=inf` — "never grokked" — while the histories plainly showed grokking. A
measurement artifact reported as a scientific null. The bar is now a **dataset property**
(`registry.grok_threshold(cfg)`): modular 95.0 (unchanged — every prior modular result is
unaffected, verified bit-identical), **s5 85.0**, **mnist 90.0**, with the reasoning documented
at the registration site. S5 was heading for the same silent failure: its ceiling is ~92%, so
all 72 planned S5 replication runs would have recorded `inf` at a 95% bar.

`grok_threshold` is now recorded in every result row and is a survival cell key — a `t_grok`
is only interpretable next to the bar it was measured at.

## Readiness blockers before the full campaign (identified 2026-07-28)

Five fixes must land before committing to a full campaign. Two of them silently corrupt
results and cost a re-run to discover late.

| # | Fix | Evidence |
|---|---|---|
| 1 | **Budget: 20k rounds at E=5** (=100k steps) | v1 at α=0.25: median T_grok 27,505, max 75,425. At 10k rounds (50k steps) **5.6% of genuine groks are censored** — repeating the E=1 mistake exactly where censoring *is* the signal |
| 2 | **`checkpoint_every` on boundary cells** | No manifest sets it. Frequency-consensus analysis is post-hoc impossible without per-client weights (~100 MB/config from v1 exp7) |
| 3 | **20 seeds at boundary cells** | `fraction_grokked` is the order parameter. Wilson 95% CI width at a true 50% transition: 3 seeds→0.73, 5→0.65, 10→0.53, 20→0.40. Cannot separate K=50 from K=97 at 5 |
| 4 | **Wire mechanistic metrics into history** | `coset_attribution` + transformer embedding measures compute standalone, never log per-epoch. Same trap as #2 |
| 5 | **Add α as a manifest axis** | Every FL manifest inherits α=0.3 — now proven safe. The boundary is at **α=0.25, K≥50** (v1: 3/3 at K≤20, 2/3 at K=50, 0/3 at K=97) |

**Boundary campaign spec** (after the fixes): α ∈ {0.22, 0.25, 0.28} × K ∈ {20, 50, 97},
E=5, 20k rounds, 20 seeds at α=0.25 / 5 on the bracketing α, checkpoints + per-client W1 on,
plus the fixed-per-client-n control arm (separates "federation broke it" from "shards too
small" — without it the result collapses to the obvious objection).

**Decision taken from this evidence:** rounds stay fixed, but the E axis is trimmed to
`E_SPINE = {5, 10, 25, 50}` — E∈{1,2} are under-budgeted at 10k rounds and E∈{100,250} are
too expensive (2.5M steps/run). See the E RANGE note in `scripts/build_manifests.py`.

## Remaining work is EXECUTION

**All manifests are written** (829 unique runs across 8 files in `manifests/`). The loop is:

```bash
venv/bin/python scripts/build_manifests.py                       # (re)generate manifests/
venv/bin/python scripts/launch_sweep.py manifests/<name>.jsonl --gpus 0,2,4,5,6,7 --per-gpu 2
venv/bin/python scripts/collect_runs.py                          # -> results/data/runs_v2.csv
venv/bin/python scripts/summarize_runs.py results/data/runs_v2.csv --group <cols>
```
Resume is automatic; re-running a manifest only executes what's missing. Note `nohup` buffers
the launcher's stdout — use `-u`, or just watch `ls results/data/runs/*.json | wc -l`.

| manifest | runs | status |
|---|---|---|
| `t0_wd_grid` | 45 | **done** — see results above |
| `t0_poly_pilot` | 6 | **done** — see results above |
| `t1_probe` | 24 | **done** (18 ok, 6 E=250 cancelled) — gate verdict above; do not relaunch |
| `t2_k_breakdown` | 60 | **done** (2026-07-28, ~3h wall-clock on 12 slots) — 60/60, zero censored. Results + the structured-heterogeneity finding above. 6 cells auto-skipped as already done by the probe (content-hash dedup working across manifests). |
| `t0_mnist_wd_band` | 15 | **done** (2026-07-28, <4 min) — MNIST groks; results + the threshold fix above |
| `t2_boundary` | 20 | **done** (2026-07-29, ~5h) — see the boundary section above |
| `s5_central_anchor` | 140 | **done** — Gate A ladders, RESULTS §11 |
| `s5_mnist_working_point` | 36 | **done** — delay vs shardability; (n_train=2000, batch=100) |
| `s5_fl_probe` | 60 | **done** — first FL run of each setup. **Under-budgeted by construction** (10,000 total steps, below C's centralized requirement outright); its censored cells are not federated effects |
| `s5_setup_c_capacity` | 34 | **done** — C's failure was censoring, not capacity (RESULTS §11) |
| `s5_mnist_fl` | 36 | **done** — E groks iid at K=10/20 (batch=100); `label_block` 0/12 |
| `s5_probe_rerun` | 9 | **done** — B K=10 operand and D K=50 iid recover at 5× budget; **D K=10 operand does not** |
| `s5_k50_diagnosis` | 9 | **done** — lr does not rescue; wd does (RESULTS §12) |
| `p1_d_gd_probe` | 9 | **done** — RESULTS §13.1 |
| `p1_cd_decay_band` | 30 | **done** — RESULTS §13.2 |
| `p1_aprime_alpha` | 45 | **done** — RESULTS §13.3 |
| `p1_k_collapse_wd` | 18 | **done** — RESULTS §13.6 |
| `p1_b_decay_band` | 15 | **done** — RESULTS §13.5 |
| `p1_k_collapse_budget` | 11 | **done** (11/11, 4.9 h) — RESULTS §13.7, the gate |
| `x_d_alpha_{cliff,fine,high}` · `x_d_internals` | 235 | **done** — tier X, setup D's α ladder at 0.025 resolution + the internals run. `d_internals` is **unanalysed** |
| `x_d_wd_ladder` | 45 | 12/45 — tier X, tests whether the dip is a decay transient |
| `x_d_lr_control` | 18 | 0/18 — tier X, the step-size control for the above |
| `t1_replication` | 150 | **rewrite, do not launch** — 24/150 banked, but its setup definitions predate Gate A (wrong working points, budgets not keyed to measured T_grok) |
| `t2_phase_diagram` | 415 | **re-scope first** — 108/415 banked; the α=0.3 plane it grids is now known uniformly safe, and `k_fixed_per_client` is geometrically impossible at the boundary |
| `t3_server_lr_calibration` | 42 | pending (run before `t3_algorithm_comparison`) |
| `t3_algorithm_comparison` | 90 | pending — fix each method at its calibrated server LR first. Note **SCAFFOLD is unavailable on B/C/D/E** (raises under AdamW by design), so drift correction is FedProx-only there |

**The `t1_probe` gate:** if no cell reaches ~100% train accuracy while failing to grok, the
"FL breaks grokking" framing has no support, and the plan says stop and re-frame *before*
spending T2/T3. Check with
`summarize_runs.py results/data/runs_v2.csv --group local_epochs,partition`.

2. **Phase 5 part 2 — provenance-tracked figures.** BLOCKED on (1): there is nothing meaningful to
   plot until the sweeps produce `runs_v2.csv`. When ready, regenerate figures from that CSV, each
   stamped with its manifest ID + row IDs; extract shared helpers from `scripts/plotting/` then.

## Deferred follow-ups (lower priority, not blocking)

- **Native robust aggregators** — FedMedian / FedTrimmedAvg / Krum / Bulyan are all in stock Flower
  (zero custom code); add to `_build_strategy` if the robustness axis is wanted.
- **The `metrics/fourier.py` split** (fourier/spectral/basic/nonabelian) — cosmetic.
- **The deferred DP / compression / Byzantine follow-up paper** — out of scope for v2 (see plan).
- **`paper/` is empty.** (The README was v1-era and has been rewritten against the
  actual v2 layout.)

**Done, previously listed here as deferred:** wiring mechanistic metrics into per-epoch
history. `metrics/probes.py::mechanistic_probe(cfg)` dispatches per (dataset, model) —
coset attribution + the exact quadratic-circuit and irrep decompositions on S₅, `embed_ipr`
on the modular transformer, `ipr` on the anchor — and **both** training loops call it inside
the eval block. `client_signature(model, cfg)` likewise generalises per-client checkpointing
to every architecture (`W1_operand` / `W_E` / `layer0`).

**Equivalence harness** (reuse for every training-loop change): `traj.py` in the scratchpad
runs one config and dumps history JSON; diff against `ref_run1.json` (archived c7c997f
trajectory). Worst |Δ| < 1e-5 = fp32 noise = OK. It has now caught two real bugs (the 1.1
RNG-order bug and the stale `ipr` reference in 2b).

## Open tuning note (carried from 1.1)

Client `num_gpus=1/K`, `num_cpus=1` reservations are unchanged. Under heavy concurrency
(12 runs × large K × 1 CPU each on 64 cores) Ray will queue/serialise rather than deadlock;
if a big-K sweep is slow, lower `--per-gpu` or make `num_cpus` fractional in
`fedgrok/training/federated.py`. Not a blocker — the smoke tests ran clean.

## Findings worth not re-deriving

- **Weight decay was a values bug, not a mechanism bug.** Both SGD (coupled) and AdamW
  (decoupled) shrink weights by `(1 - lr*wd)` per step, so `lr*wd` is the comparable
  quantity and `1/(lr*wd)` the decay timescale. exp5 ran lr=50 × wd∈{0.01,0.1,1.0} →
  `lr*wd` ∈ {0.5, 5, 50}. All nine cells returned `T_grok=inf` for numerical reasons.
  Guard in `core/utils.check_decay_stability` rejects divergence (`lr*wd >= 2`) *and*
  the subtler destructive case (`lr*wd > 0.1`; 0.5 never diverges but halves every
  weight every step). Replacement sweep λ ∈ {2e-7, 2e-6, 2e-5, 2e-4} at lr=50.
- **History filenames are not unique run keys.** The `_adam_tau*_slr*` / `_wd*` suffixes
  landed mid-campaign (`3796754`, 2026-03-23), so earlier exp5 arms wrote *identical*
  paths — FedAvg, FedAdam-0.1, FedAdam-0.01 and FedAvg+WD-* collided per (setting, seed).
  61 paths were written by >1 run. Any surviving JSON holds only the last run written.
  `harvest_logs.py` therefore takes algorithm identity from the **log filename**.
- **Wall-clock is ~99% orchestration.** Measured on an L4: 50k full-batch GD steps at
  p=97 take **23 s** (0.458 ms/step; p=151 is 1.077 ms/step). The README budgeted ~26 min
  for the same run. Flower/Ray costs ~156 ms/round. Phase 1 targets ~40 ms/round.
- **The E-sweep is the expensive part**, because rounds = total_steps / E. E=1 means
  50,000 rounds. Those cells dominate the manifest's cost.
- **Grid holes:** 6 empty `exp3a` α=0.20 logs, and `exp3a_a0.30_dir10.0` crashed at
  round 6491 on a Ray actor failure. Tracked in `results/data/runs_skipped.csv`.
- **exp4b partial-participation T_grok values sit on the OLD (inflated) step axis.** The
  0.6 fix changes the x-axis under `fraction_train < 1.0`, so those rows in `runs.csv` are
  ~2.5× too high on `total_steps`. They will be corrected when exp4b is re-run.
- **Known benign warning:** `test_drift_metrics_in_history` emits `invalid value in
  subtract` — a pre-existing test-hygiene issue (default lr=50 diverges on p=7, weights
  overflow to inf, drift-std hits inf−inf). Library bug, not; test still passes. Out of
  Phase 0 scope.

## Resume

```bash
cd /home/jse44/modules/ToDL/federated_learning_grokking
git checkout v2-multisetup
venv/bin/python -m pytest tests/ -q          # ~530 passed, ~10.5 min
nvidia-smi                                   # SHARED box -- check what is free first
```

**First, find the real state.** These notes lag the banked data; the CSV and the
result JSONs do not:

```bash
venv/bin/python scripts/collect_runs.py                  # per-run JSONs -> runs_v2.csv
venv/bin/python scripts/summarize_runs.py results/data/runs_v2.csv --group group,setup,alpha
# what each manifest still owes -- run ids are content hashes, so this is exact
venv/bin/python - <<'EOF'
import sys, os, glob; sys.path.insert(0, 'src')
from fedgrok.manifest import load_manifest, run_id
banked = {os.path.basename(p)[:-5] for p in glob.glob('results/data/runs/*.json')}
for m in sorted(glob.glob('manifests/*.jsonl')):
    ids = [run_id(s) for s in load_manifest(m)]
    print(f"{os.path.basename(m):<32} {len(ids):>4} specs, "
          f"{sum(i not in banked for i in ids):>4} missing")
EOF
```

**The setup/check phase is done.** Next is Part 3 of the plan — writing the
per-setup FL manifests. The one rule that came out of it:

> **Budget every federated cell as `t_memo(K) + delay`, never as a multiple of the
> centralized T_grok.** Federation slows memorisation steeply with K while leaving
> the delay roughly flat, so a centralized-anchored budget under-provisions
> exactly the high-K cells the campaign cares about. Six boundaries in this
> project have been manufactured by getting this wrong.

Budget B's K=50 campaign cells with headroom **above** 143,000 (that figure is a
lower bound, not a prediction) and they double as the test of the additive model —
no separate confirmation run is needed, since run ids are content hashes.

**Still to write before exp2 can run:**

- Per-setup FL manifests, budgeted as `t_memo(K) + delay` per the rule above —
  NOT as a multiple of the centralized T_grok, which is what under-provisioned
  every high-K cell in the K-collapse diagnosis. Working points per setup are in
  RESULTS §11 and §13.
- `checkpoint_every` + `checkpoint_client_weights` on the cells feeding the
  circuit analysis — **no B/C/D/E run has per-client weights**, so exp7 has no
  data on any new setup. These are config fields, so they change run ids: setting
  them after a sweep re-runs it. The 20 anchor runs that do have them are
  findable by query (`checkpoint_client_weights == True`).
- The three-arm wiring for exp2: `reduced_arm()` builds the floor condition
  (dataset-aware — α/K for grid datasets, `n_train`/K for MNIST) and is tested,
  but **no manifest calls it yet**.
- Per-setup capacity (exp0) checks: done for C only.
