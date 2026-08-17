#!/usr/bin/env bash
# main's two unported experiments, asked at v2's boundary rather than v1's
# safe plane. Two pools so they run concurrently; GPU 7 is left to the other
# user on this shared box.
#
#   setsid nohup bash scripts/run_exp3a_exp4b.sh > logs/sweeps/exp3a_exp4b.log 2>&1 < /dev/null &
#
# COMPUTE ONLY: no commits, no result-dependent decisions.
set -u
cd "$(dirname "$(readlink -f "$0")")/.."   # repo root, wherever the tree lives
PY=venv/bin/python

step () { echo; echo "=============================================="
          echo "== $*"; echo "== $(date -Is)"; echo "=============================================="; }

step "exp3a  t3a_dirichlet_band -- 24 runs, ~40 slot-h  (GPUs 0-3)"
$PY -u scripts/launch_sweep.py manifests/t3a_dirichlet_band.jsonl --gpus 0,1,2,3 --per-gpu 2 &
P1=$!

step "exp4b  t4b_participation -- 9 runs, ~31 slot-h  (GPUs 4-6)"
$PY -u scripts/launch_sweep.py manifests/t4b_participation.jsonl --gpus 4,5,6 --per-gpu 2 &
P2=$!

wait $P1 $P2
$PY scripts/collect_runs.py

step "summary"
{
  echo "# exp3a + exp4b finished $(date -Is)"
  echo
  echo "## exp4b -- partial participation at alpha=0.25, K=50 (control f=1.0 is t2_boundary)"
  $PY scripts/summarize_runs.py results/data/runs_v2.csv --group group,fraction_train 2>/dev/null | grep -E "^(group|participation|boundary)" || true
  echo
  echo "## exp3a -- Dirichlet concentration at the boundary"
  $PY scripts/summarize_runs.py results/data/runs_v2.csv --group group,num_clients,dirichlet_alpha 2>/dev/null | grep -E "^(group|dirichlet_band)" || true
  echo
  echo "## the banked iid controls at the same cells"
  $PY scripts/summarize_runs.py results/data/runs_v2.csv --group group,num_clients,partition 2>/dev/null | grep -E "^(group|boundary)" || true
} > EXP3A_4B_STATUS.md 2>&1
cat EXP3A_4B_STATUS.md
step "DONE"
