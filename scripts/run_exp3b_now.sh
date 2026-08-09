#!/usr/bin/env bash
# exp3b, restricted to the setups that can actually resolve the effect.
#   setsid nohup bash scripts/run_exp3b_now.sh > logs/sweeps/exp3b.log 2>&1 < /dev/null &
#
# A, A' and E have ~1.0-1.1x within-cell seed spread, so the ~10% structure
# effect is visible. B, C and D have ~1.9x -- the effect is buried at 3 seeds --
# and are deferred to logs/sweeps/split/t3b_later.jsonl.
#
# D's iid and operand cells ARE included: an earlier probe hinted D's operand
# split is much WORSE than iid, and an effect that large is not hidden by 1.9x
# noise. If it holds it inverts the project's headline, so it is worth 6 runs.
#
# Split files carry ids identical to manifests/t3b_partitions.jsonl (content
# hashes), so resume, dedup and collect_runs behave as one sweep.
# COMPUTE ONLY: no commits, no result-dependent decisions.
set -u
cd /home/jse44/modules/ToDL/federated_learning_grokking
PY=venv/bin/python

step () { echo; echo "=============================================="
          echo "== $*"; echo "== $(date -Is)"; echo "=============================================="; }

step "waiting for the x_controls diagnostics to finish"
while pgrep -f "launch[_]sweep" > /dev/null; do sleep 60; done
$PY scripts/collect_runs.py

step "exp3b (A, A', E + D's operand probe) -- 54 runs, ~71 slot-h"
$PY -u scripts/launch_sweep.py logs/sweeps/split/t3b_now.jsonl --gpus 0,1,2,3,4,5,6,7 --per-gpu 2
$PY scripts/collect_runs.py

step "summary"
{
  echo "# exp3b (resolvable setups) finished $(date -Is)"
  echo
  echo "## the diagnostics that ran first"
  echo "### A' at E=1, K=2 -- must reproduce the centralized ceiling (~4,000)"
  $PY scripts/summarize_runs.py results/data/runs_v2.csv --group group,setup,local_epochs,num_clients 2>/dev/null | grep -E "^(group|e1_identity)" || true
  echo "### wd=0 on B, C, E -- do they grok without decay?"
  $PY scripts/summarize_runs.py results/data/runs_v2.csv --group group,setup,weight_decay 2>/dev/null | grep -E "^(group|wd_zero)" || true
  echo
  echo "## exp3b: structured vs random shards"
  $PY scripts/summarize_runs.py results/data/runs_v2.csv --group group,setup,num_clients,partition 2>/dev/null | grep -E "^(group|partitions)" || true
  echo
  echo "## still owed on exp3b (the noisy setups, deferred)"
  echo "   logs/sweeps/split/t3b_later.jsonl -- B, C, D: 54 runs, ~190 slot-h"
} > EXP3B_STATUS.md 2>&1
cat EXP3B_STATUS.md
step "DONE"
