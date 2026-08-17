#!/usr/bin/env bash
# The control that decides whether exp3a's breakdown is heterogeneity or
# starved clients. Autodetects free GPUs -- this box is shared and three other
# users are on it, so pinning a fixed pool would fight them.
#
#   setsid nohup bash scripts/run_size_control.sh > logs/sweeps/size_control.log 2>&1 < /dev/null &
#
# COMPUTE ONLY: no commits, no result-dependent decisions.
set -u
cd "$(dirname "$(readlink -f "$0")")/.."   # repo root, wherever the tree lives
PY=venv/bin/python
step () { echo; echo "=============================================="
          echo "== $*"; echo "== $(date -Is)"; echo "=============================================="; }

step "t3a_size_control -- 12 runs, ~23 slot-h"
$PY -u scripts/launch_sweep.py manifests/t3a_size_control.jsonl --per-gpu 2
$PY scripts/collect_runs.py

step "verdict"
{
  echo "# size control finished $(date -Is)"
  echo
  echo "## the control (dirichlet_sizes: same shard sizes, IID labels)"
  $PY scripts/summarize_runs.py results/data/runs_v2.csv --group group,num_clients,dirichlet_alpha 2>/dev/null | grep -E "^(group|size_control)" || true
  echo
  echo "## against the concentrated arm it controls for"
  $PY scripts/summarize_runs.py results/data/runs_v2.csv --group group,num_clients,dirichlet_alpha 2>/dev/null | grep -E "^(group|dirichlet_band)" || true
  echo
  echo "## and the matched iid reference"
  $PY scripts/summarize_runs.py results/data/runs_v2.csv --group group,num_clients,partition 2>/dev/null | grep -E "^(group|boundary)" || true
} > SIZE_CONTROL_STATUS.md 2>&1
cat SIZE_CONTROL_STATUS.md
step "DONE"
