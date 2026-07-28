# v2-multisetup — progress and resume notes

Working branch for the multi-setup rewrite. Full plan lives at
`~/.claude/plans/plan-all-that-needs-nested-seal.md`.

**Paused:** 2026-07-21. **v2 code complete across all plan phases (0–5).** What remains is EXECUTION (run the manifests) + the figures that depend on it, plus deferred follow-ups.

## State

| | |
|---|---|
| Branch | `v2-multisetup` (branched from `main` @ `41c3fa8`) |
| Frozen reference | tag `v1-single-setup` — the state that produced the 32 figures in `results/figures/` |
| Tests | **364/364 pass** (`venv/bin/python -m pytest tests/ -q`, ~10.5 min incl. FL integration) |
| Groks verified | quad-MLP+modular (original); transformer+modular (T_grok 6200); transformer+S5 (T_grok ~20000, non-abelian) |
| FL algorithms | FedAvg, FedProx, FedAvgM, FedYogi, FedAdam (native) + SCAFFOLD (adapted); server-LR calibration manifest queued |
| Statistics | censored survival (KM median + fraction-grokked + bootstrap CI); `scripts/summarize_runs.py` |
| deps | torch 2.10 + torchvision 0.25 (pinned pair); flwr 1.27 |
| Env | `venv/` (py3.10, torch 2.10, flwr 1.27); package installed via `pip install -e . --no-deps` |
| Hardware | 8× L4; **use 6, ~2 runs/GPU = 12 slots**. Indices 1 and 3 had other work on them — the launcher auto-skips busy GPUs. |

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
| `t0_mnist_wd_band` | 15 | pending |
| `t1_replication` | 150 | pending |
| `t2_phase_diagram` | 415 | pending (`t2_k_breakdown` is a subset — those cells will skip) |
| `t3_server_lr_calibration` | 42 | pending (run before `t3_algorithm_comparison`) |
| `t3_algorithm_comparison` | 90 | pending — fix each method at its calibrated server LR first |

**The `t1_probe` gate:** if no cell reaches ~100% train accuracy while failing to grok, the
"FL breaks grokking" framing has no support, and the plan says stop and re-frame *before*
spending T2/T3. Check with
`summarize_runs.py results/data/runs_v2.csv --group local_epochs,partition`.

2. **Phase 5 part 2 — provenance-tracked figures.** BLOCKED on (1): there is nothing meaningful to
   plot until the sweeps produce `runs_v2.csv`. When ready, regenerate figures from that CSV, each
   stamped with its manifest ID + row IDs; extract shared helpers from `scripts/plotting/` then.

## Deferred follow-ups (lower priority, not blocking)

- **Wire mechanistic metrics into per-epoch history** — `coset_attribution` (S5) and embedding-
  space measures (transformer W_E restricted/excluded loss). Fold the `metrics/fourier.py` split
  (fourier/spectral/basic/nonabelian) in here.
- **Native robust aggregators** — FedMedian / FedTrimmedAvg / Krum / Bulyan are all in stock Flower
  (zero custom code); add to `_build_strategy` if the robustness axis is wanted.
- **The deferred DP / compression / Byzantine follow-up paper** — out of scope for v2 (see plan).

## Deferred follow-ups (lower priority than Phase 4)

- **Wire mechanistic metrics into the training-loop history** — `coset_attribution` for S5 and
  embedding-space measures (Nanda restricted/excluded loss over the transformer's W_E) for the
  transformer. Both currently compute fine as standalone functions but aren't logged per-epoch.
  Fold the deferred `metrics/fourier.py` split (fourier/spectral/basic/nonabelian) in here.
- **Locate the MNIST grok band** — run `manifests/t0_mnist_wd_band.jsonl` (queued).

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
venv/bin/python -m pytest tests/ -q          # expect 364 passed
```
Then continue at **writing the Tier-1/2 sweep manifests** + running them, above.
