# v2-multisetup — progress and resume notes

Working branch for the multi-setup rewrite. Full plan lives at
`~/.claude/plans/plan-all-that-needs-nested-seal.md`.

**Paused:** 2026-07-21, after Phase 0 complete (all correctness fixes + verification). Next: Phase 1.

## State

| | |
|---|---|
| Branch | `v2-multisetup` (branched from `main` @ `41c3fa8`) |
| Frozen reference | tag `v1-single-setup` — the state that produced the 32 figures in `results/figures/` |
| Tests | **258/258 pass** (`venv/bin/python -m pytest tests/ -q`, ~7.5 min incl. FL integration) |
| Env | `venv/` (py3.10, torch 2.10, flwr 1.27); package installed via `pip install -e . --no-deps` |
| Hardware | 8× L4; **use 6, ~2 runs/GPU = 12 slots**. Indices 1 and 3 had other work on them — check before launching. |

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

## Next up — Phase 1 (harness)

1. **Per-round optimisation** in `src/fedgrok/training/federated.py`: cache model +
   on-device tensors per `partition_id` (pattern: `_dataset_cache`); reuse model in
   `evaluate_fn`; add `eval_every` to `FedConfig`; disable Ray dashboard/metrics exporter
   via `backend_config`. Target ~156 ms/round → ~40 ms/round. Guard with the existing
   equivalence check (see verification step 2 in the plan).
2. **`scripts/launch_sweep.py`** — JSONL manifest runner, one run per GPU via
   `CUDA_VISIBLE_DEVICES`, 12 slots, idempotent resume, writes rows to the tidy CSV. Fold
   `main.py` + `fed_main.py` into it (deferred from 0a — see below).

## Deferred items now due

- **Entry-point consolidation** (`main.py`/`fed_main.py` → unified runner): do in Phase 1
  with `launch_sweep.py`, as planned.
- **`metrics/fourier.py` split** into fourier/spectral/basic: do in Phase 2 with the
  capability protocol.

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
venv/bin/python -m pytest tests/ -q          # expect 258 passed
```
Then continue at **Phase 1, item 1** (per-round Flower optimisation) above.
