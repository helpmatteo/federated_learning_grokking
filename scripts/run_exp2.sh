#!/usr/bin/env bash
# Detached runner for exp2 (t2_aggregation). Survives the session that starts it:
#   setsid nohup bash scripts/run_exp2.sh > logs/sweeps/exp2.log 2>&1 < /dev/null &
#
# Two passes, because memory scales with CLIENT COUNT, not with run length. A
# K=50 cell holds ~50 Ray actors each carrying its own CUDA context and measures
# ~12 GB; two of those on a 23 GB L4 is an OOM, not a near miss. But only 15 of
# 192 runs are K=50, so forcing --per-gpu 1 on the whole sweep would idle half
# the box for the other 177.
#
# The split files carry IDENTICAL run ids to manifests/t2_aggregation.jsonl (ids
# are content hashes), so resume, dedup and collect_runs all behave as if this
# were one sweep. They live outside manifests/ so the "what is still owed" audit
# does not double-count them.
#
# COMPUTE ONLY. No commits, no decisions that depend on a result.
set -u
cd "$(dirname "$(readlink -f "$0")")/.."   # repo root, wherever the tree lives
PY=venv/bin/python
GPUS=0,1,2,3,4,5,6,7

step () { echo; echo "=================================================="
          echo "== $* "; echo "== $(date -Is)"
          echo "=================================================="; }

step "pass 1/2  K=50 cells, --per-gpu 1  (15 runs, ~12 GB each)"
$PY -u scripts/launch_sweep.py logs/sweeps/split/t2_agg_bigK.jsonl --gpus $GPUS --per-gpu 1
$PY scripts/collect_runs.py

step "pass 2/2  everything else, --per-gpu 2  (177 runs)"
$PY -u scripts/launch_sweep.py logs/sweeps/split/t2_agg_rest.jsonl --gpus $GPUS --per-gpu 2
$PY scripts/collect_runs.py

step "verify the whole manifest is actually complete"
$PY -u scripts/launch_sweep.py manifests/t2_aggregation.jsonl --dry-run

step "summary"
{
  echo "# exp2 (t2_aggregation) finished $(date -Is)"
  echo
  $PY -c "
import csv
r=list(csv.DictReader(open('results/data/runs_v2.csv')))
print(f'banked {len(r)}  grokked {sum(1 for x in r if x[\"grokked\"].lower()==\"true\")}  '
      f'machine-hours {sum(float(x[\"wall_s\"] or 0) for x in r)/3600:.0f}')"
  echo
  echo "## the three arms, per setup and K"
  $PY scripts/summarize_runs.py results/data/runs_v2.csv \
      --group group,setup,arm,num_clients 2>/dev/null | grep -E "^(group|aggregation)" || true
  echo
  echo "## K=2 control -- cent_full and fl should AGREE here"
  $PY scripts/summarize_runs.py results/data/runs_v2.csv \
      --group group,setup,arm,num_clients 2>/dev/null | grep -E "aggregation.* 2$" || true
} > EXP2_STATUS.md 2>&1
cat EXP2_STATUS.md
step "DONE"
