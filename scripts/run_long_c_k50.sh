#!/usr/bin/env bash
# LONG chain: setup C's K=50 exp3b block, the campaign's longest jobs.
#
#   setsid nohup bash scripts/run_long_c_k50.sh > logs/sweeps/long_c_k50.log 2>&1 < /dev/null &
#
# Runs on its OWN GPU pool (0,1,2) at --per-gpu 1 so it can proceed in parallel
# with scripts/run_main_chain.sh on pool 3-7. Two reasons for per-gpu 1:
#   - precedent: run_campaign_part0.sh measured a K=50 S_5 transformer run at
#     ~12 GB, and two will not fit on a 23 GB L4;
#   - these are ~6.7 h/run (40,000 rounds x 0.605 s/round, measured), so an OOM
#     retry is the most expensive failure available here.
#
# 9 runs, ~60 slot-hours, 3 waves on 3 slots.
#
# COMPUTE ONLY: no commits, no result-dependent decisions.
set -u
cd "$(dirname "$(readlink -f "$0")")/.."   # repo root, wherever the tree lives
PY=venv/bin/python

step () { echo; echo "=============================================="
          echo "== $*"; echo "== $(date -Is)"; echo "=============================================="; }

step "t3b_bigk -- setup C, K=50, {dirichlet, operand, target} x 3 seeds (9 runs, ~60 slot-h)"
$PY -u scripts/launch_sweep.py logs/sweeps/split/t3b_bigk.jsonl --gpus 0,1,2 --per-gpu 1
$PY scripts/collect_runs.py

step "DONE"
