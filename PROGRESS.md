# v2-multisetup — progress and resume notes

Working branch for the multi-setup rewrite. Full plan lives at
`~/.claude/plans/plan-all-that-needs-nested-seal.md`.

**Paused:** 2026-07-21, after Phase 0.2.

## State

| | |
|---|---|
| Branch | `v2-multisetup` (branched from `main` @ `41c3fa8`) |
| Frozen reference | tag `v1-single-setup` — the state that produced the 32 figures in `results/figures/` |
| Tests | **230/230 pass** (`venv/bin/python -m pytest tests/ -q`, ~6.5 min incl. FL integration) |
| Env | `venv/` (py3.10, torch 2.10, flwr 1.27); package installed via `pip install -e . --no-deps` |
| Hardware | 8× L4; **use 6, ~2 runs/GPU = 12 slots**. Indices 1 and 3 had other work on them — check before launching. |

## Done

- **0a Branch + reorganise** (`6891ebd`, `d93babc`) — library moved to `src/fedgrok/`
  (`core/ models/ data/ metrics/ training/ analysis/`), plotting to `scripts/plotting/`,
  all moves via `git mv`, 32 files' imports rewritten, `pyproject.toml` added with pinned deps.
  The two colliding `exp7_*` modules renamed to `exp_task_generality.py` /
  `exp_mechanistic_checkpoints.py`.
- **0.7 Log harvest** (`7af94e4`) — `scripts/harvest_logs.py` → `results/data/runs.csv`,
  **870 runs, 554 grokked, 316 censored**, 195 KB, committed. `runs_skipped.csv` records
  the 7 logs that yielded nothing.
- **0.1 Weight decay** + **0.2 one-hot device** (`099622e`) — see below.

## Next up (in order)

1. **0.3 Task degeneracy** — `src/fedgrok/data/modular.py`. `division` maps `m=0` to a
   fabricated label 0 (p samples); `multiplication` includes all zero-products, over-
   representing class 0 ~2×. Define division on the `p(p-1)` grid per Power et al.
   Plan already drops multiplication from the operation set (it is Z_{p-1} in disguise).
2. **0.4 Dedup split** — `data/partition.py` re-implements the grid + train/test split
   from `data/modular.py` instead of calling it. Two sources of truth for the split.
3. **0.5 Dead code** — `should_abort` in `training/runner.py` is unit-tested but called
   from nowhere; README describes an early-abort protocol that never ran. Delete both.
4. **0.6 Step accounting** — `total_steps = server_round * local_epochs` in
   `training/federated.py` is wrong when `fraction_train < 1.0`; the participation sweep
   depends on it.
5. **Phase 0 verification** — assert FedAvg at E=1, full participation, `n_k/n` weighting
   reproduces centralized GD to float tolerance, under **both** IID and Dirichlet. This
   identity is also a stated result in the paper.
6. **Phase 1** — harness optimisation then `scripts/launch_sweep.py`.

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

## Deviations from the approved plan

- **Entry-point consolidation deferred to Phase 1.** The plan folds `main.py` and
  `fed_main.py` into `run_experiment.py` during 0a, but they carry ~40 argparse options
  between them and `scripts/launch_sweep.py` supersedes both. Doing it now is throwaway
  work. They still function post-migration.
- **`metrics/fourier.py` not yet split** into fourier/spectral/basic. That split belongs
  with the Phase 2 capability protocol; 0a moved whole files only to keep the diff
  reviewable.

## Resume

```bash
cd /home/jse44/modules/ToDL/federated_learning_grokking
git checkout v2-multisetup
venv/bin/python -m pytest tests/ -q          # expect 230 passed
```
Then continue at item 1 above (`0.3 Task degeneracy`).
