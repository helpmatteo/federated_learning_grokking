# v2-multisetup — progress and resume notes

Working branch for the multi-setup rewrite. Full plan lives at
`~/.claude/plans/plan-all-that-needs-nested-seal.md`.

**Paused:** 2026-07-21, Phase 2 core done (model + loss registries, metric capability). Next: minibatching, then Phase 3.

## State

| | |
|---|---|
| Branch | `v2-multisetup` (branched from `main` @ `41c3fa8`) |
| Frozen reference | tag `v1-single-setup` — the state that produced the 32 figures in `results/figures/` |
| Tests | **297/297 pass** (`venv/bin/python -m pytest tests/ -q`, ~8.5 min incl. FL integration) |
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

## Next up — finish Phase 2, then Phase 3

1. **Minibatching** (last Phase 2 item). Add `batch_size: int = 0` to Config (0 = full-batch,
   the default, so setup A and the transformer are unchanged). In centralized `train` and the
   federated client `fit`, iterate minibatches when `batch_size > 0`. Only MNIST (Omnigrok)
   needs it. **Re-verify** the full-batch path stays bit-identical with `traj.py`.
2. **Dataset registry** — do it *with* the Phase 3 datasets: `build_dataset(cfg)` dispatching
   on a dataset selector (modular now; add MNIST + S₅). Keep modular byte-identical.
3. **Phase 3 setups** — Nanda transformer (`@register_model("transformer")`, CE, mod-113),
   Omnigrok MNIST-1k MLP (large init), S₅ composition + coset partition + coset-attribution
   metric. Each new arch = one registry entry + (if needed) its own progress measure.

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
venv/bin/python -m pytest tests/ -q          # expect 297 passed
```
Then continue at **Minibatching** (finish Phase 2), then Phase 3, above.
