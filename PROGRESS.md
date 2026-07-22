# v2-multisetup — progress and resume notes

Working branch for the multi-setup rewrite. Full plan lives at
`~/.claude/plans/plan-all-that-needs-nested-seal.md`.

**Paused:** 2026-07-21, mid Phase 2 (model registry done; loss/dataset/minibatch next).

## State

| | |
|---|---|
| Branch | `v2-multisetup` (branched from `main` @ `41c3fa8`) |
| Frozen reference | tag `v1-single-setup` — the state that produced the 32 figures in `results/figures/` |
| Tests | **285/285 pass** (`venv/bin/python -m pytest tests/ -q`, ~8.5 min incl. FL integration) |
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

## Done — Phase 2 part 1: model registry (`b50ab87`)

`fedgrok/core/registry.py` — `build_model(cfg)` dispatches on new `cfg.model` field
(default `"groknet"`). Wired into federated `_make_model` and centralized `train`; the
modular 2p/p dim convention now lives in the groknet builder. Verified equivalent to the
archived baseline (4.8e-7). +7 tests.

## Next up — Phase 2 part 2: loss + dataset registries, minibatching, metric capability

1. **Loss registry** (the intricate part — touches target handling). Add `loss: str = "mse"`
   to Config. `build_loss(cfg)` returns the loss fn **and a target-prep** (MSE needs one-hot
   via `make_targets_onehot`; CE needs raw class indices). Wire into all target sites: the 3
   `nn.MSELoss()` + `make_targets_onehot` uses in centralized `train`, federated `GrokClient.fit`,
   and federated `evaluate_fn`. **Re-verify equivalence** with the `traj.py` harness (scratchpad)
   — MSE path must stay bit-identical. This unblocks the CE transformer.
2. **Dataset registry** — `build_dataset(cfg)` / `build_federated_dataset(cfg)` dispatching on a
   dataset selector (modular now; MNIST/S₅ in Phase 3). Keep modular exactly as-is.
3. **Metric capability protocol** — `compute_ipr`/`fourier_spectrum`/`neuron_frequency_assignment`/
   `restricted_excluded_loss` assume `model.W1`, `model.P`, one-hot 2p input. Make each declare
   applicability and be **skipped (not crash)** on transformer/MNIST/S₅. `weight_norms`,
   `gini_coefficient`, `effective_rank` are basis-free. **Split `metrics/fourier.py`** into
   fourier / spectral / basic here (deferred from 0a).
4. **Minibatching** — everything is full-batch whole-dataset tensors; needed for MNIST and the
   transformer at batch 512. Keep full-batch as the default so setup A is unchanged.

**Equivalence harness** (reuse for every training-loop change): `traj.py` in the scratchpad
runs one config and dumps history JSON; diff against `ref_run1.json` (archived c7c997f
trajectory). Worst |Δ| < 1e-5 = fp32 noise = OK. This caught the RNG-order bug in 1.1.

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
venv/bin/python -m pytest tests/ -q          # expect 285 passed
```
Then continue at **Phase 2 part 2** (loss + dataset registries, minibatching) above.
