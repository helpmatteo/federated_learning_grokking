#!/usr/bin/env bash
# Detached Part 0 runner. Chains the remaining prerequisite compute so it
# survives the session that launched it.
#
#   setsid nohup bash scripts/run_campaign_part0.sh > logs/sweeps/part0_chain.log 2>&1 < /dev/null &
#
# It runs COMPUTE ONLY. It does not commit, and it does not write manifests or
# take any decision that depends on a result -- the C working-point verdict
# (0.2) and the per-setup budgets (0.4) are judgement calls for a human.
set -u
cd "$(dirname "$(readlink -f "$0")")/.."   # repo root, wherever the tree lives
PY=venv/bin/python
GPUS=0,1,2,3,4,5,6,7

step () { echo; echo "=================================================="
          echo "== $* "; echo "== $(date -Is)"
          echo "=================================================="; }

# 0. Wait out anything already in flight (the 0.2/0.3 sweep).
step "waiting for in-flight runs"
while pgrep -f "launch[_]sweep" > /dev/null; do sleep 60; done
$PY scripts/collect_runs.py

# 1. Full suite against the committed centralized-loop fix.
step "full test suite"
$PY -m pytest tests/ -q 2>&1 | tail -5

# 2. The long pole: per-setup K ladder. Supplies t_memo(K)/delay(K) for every
#    campaign budget, and the first per-client checkpoints on B/C/D/E.
#    --per-gpu 1: a K=50 transformer run measures ~12 GB, two would OOM on a 23 GB L4.
step "0.4  t1_setup_k_ladder  (39 runs, ~88 slot-h)"
$PY -u scripts/launch_sweep.py manifests/t1_setup_k_ladder.jsonl --gpus $GPUS --per-gpu 1
$PY scripts/collect_runs.py

# 3. Server-LR calibration. Must precede any exp5 work.
step "0.5  t3_server_lr_calibration  (42 runs, ~17 slot-h)"
$PY -u scripts/launch_sweep.py manifests/t3_server_lr_calibration.jsonl --gpus $GPUS --per-gpu 2
$PY scripts/collect_runs.py

# 4. Leave a readable verdict for whoever picks this up.
step "summary"
{
  echo "# Part 0 chain finished $(date -Is)"
  echo
  echo "## Banked"
  $PY -c "
import csv
r=list(csv.DictReader(open('results/data/runs_v2.csv')))
print(f'total {len(r)}  grokked {sum(1 for x in r if x[\"grokked\"].lower()==\"true\")}  '
      f'censored {sum(1 for x in r if x[\"censored\"].lower()==\"true\")}')
print(f'machine-hours {sum(float(x[\"wall_s\"] or 0) for x in r)/3600:.1f}')"
  echo
  echo "## 0.2 THE GATE -- does C work at alpha=0.30 at width 256?"
  $PY scripts/summarize_runs.py results/data/runs_v2.csv \
      --group group,setup,hidden_width,alpha 2>/dev/null | grep -E "^(group|c_alpha_w256)" || true
  echo
  echo "## 0.3 capacity -- is the inherited width binding?"
  $PY scripts/summarize_runs.py results/data/runs_v2.csv \
      --group group,setup,hidden_width 2>/dev/null | grep -E "^(group|capacity)" || true
  echo
  echo "## 0.4 K ladder -- t_memo(K) and delay(K) per setup"
  $PY scripts/summarize_runs.py results/data/runs_v2.csv \
      --group group,setup,num_clients 2>/dev/null | grep -E "^(group|setup_k_ladder)" || true
  echo
  echo "## still owed"
  $PY - <<'PYEOF'
import sys, os, glob
sys.path.insert(0,'src')
from fedgrok.manifest import load_manifest, run_id
banked = {os.path.basename(p)[:-5] for p in glob.glob('results/data/runs/*.json')}
for m in sorted(glob.glob('manifests/*.jsonl')):
    ids=[run_id(s) for s in load_manifest(m)]
    miss=sum(i not in banked for i in ids)
    if miss: print(f'  {os.path.basename(m)[:-6]:<28} {miss:>4} of {len(ids)}')
PYEOF
} > PART0_STATUS.md 2>&1
cat PART0_STATUS.md
step "DONE"
