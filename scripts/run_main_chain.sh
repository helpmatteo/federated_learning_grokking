#!/usr/bin/env bash
# MAIN chain: everything outstanding except setup C's K=50 block, which runs in
# parallel on its own pool via scripts/run_long_c_k50.sh.
#
#   setsid nohup bash scripts/run_main_chain.sh > logs/sweeps/main_chain.log 2>&1 < /dev/null &
#
# Pool 3-7 at --per-gpu 2 (10 slots). Ordered by decisiveness first, then cost:
#
#   1. t3b_coset   6 runs  ~13 slot-h   THE decisive cell. D's operand partition
#                                       fails 0/3 while iid groks, which inverts
#                                       the project's headline. coset is S_5's
#                                       algebraically coherent split where
#                                       operand is not, so this separates
#                                       "structure helps" from "coherence helps".
#   2. tier X     51 runs  ~1.4 slot-h  the two partially-run manifests; both
#                                       centralized S_5/groknet at 2.09 ms/epoch,
#                                       so they are ~25 min of the whole campaign.
#   3. exp5       90 runs  ~38 slot-h   t3_algorithm_comparison, unblocked by the
#                                       server-LR calibration (RESULTS 14.5). Its
#                                       "placeholder" server LRs already equal the
#                                       calibrated values (FedAdam 0.1, FedYogi
#                                       0.1, FedAvgM 1.0/0.9), so ids are unchanged.
#   4. t3b_small  39 runs  ~85 slot-h   the rest of exp3b on B, C(K<=20) and D.
#
# COMPUTE ONLY: no commits, no result-dependent decisions.
set -u
cd /home/jse44/modules/ToDL/federated_learning_grokking
PY=venv/bin/python
GPUS=3,4,5,6,7

step () { echo; echo "=============================================="
          echo "== $*"; echo "== $(date -Is)"; echo "=============================================="; }

status () {
  {
    echo "# all-5 campaign -- last block finished: $1"
    echo "# $(date -Is)"
    echo
    echo "## banked"
    $PY -c "
import csv
r=list(csv.DictReader(open('results/data/runs_v2.csv')))
print(f'total {len(r)}  grokked {sum(1 for x in r if x[\"grokked\"].lower()==\"true\")}  '
      f'censored {sum(1 for x in r if x[\"censored\"].lower()==\"true\")}')
print(f'machine-hours {sum(float(x[\"wall_s\"] or 0) for x in r)/3600:.1f}')"
    echo
    echo "## exp3b -- structured vs random shards (incl. coset)"
    $PY scripts/summarize_runs.py results/data/runs_v2.csv \
        --group group,setup,num_clients,partition 2>/dev/null | grep -E "^(group|partitions)" || true
    echo
    echo "## exp5 -- algorithms at calibrated server LRs"
    $PY scripts/summarize_runs.py results/data/runs_v2.csv \
        --group group,setting,algorithm 2>/dev/null | grep -E "^(group|algorithms)" || true
    echo
    echo "## tier X -- setup D wd ladder / lr control"
    $PY scripts/summarize_runs.py results/data/runs_v2.csv \
        --group group,weight_decay,lr 2>/dev/null | grep -E "^(group|d_wd_ladder|d_lr_control)" || true
    echo
    echo "## still owed"
    $PY - <<'PYEOF'
import sys, os, glob
sys.path.insert(0,'src')
from fedgrok.manifest import load_manifest, run_id
banked = {os.path.basename(p)[:-5] for p in glob.glob('results/data/runs/*.json')}
for m in sorted(glob.glob('manifests/*.jsonl')) + sorted(glob.glob('logs/sweeps/split/*.jsonl')):
    ids=[run_id(s) for s in load_manifest(m)]
    miss=sum(i not in banked for i in ids)
    if miss: print(f'  {m:<45} {miss:>4} of {len(ids)}')
PYEOF
  } > ALL5_STATUS.md 2>&1
}

step "1/4  t3b_coset -- D and C coset shards (6 runs, ~13 slot-h) -- THE DECISIVE CELL"
$PY -u scripts/launch_sweep.py logs/sweeps/split/t3b_coset.jsonl --gpus $GPUS --per-gpu 2
$PY scripts/collect_runs.py; status "t3b_coset"

step "2/4  tier X -- x_d_wd_ladder (33) + x_d_lr_control (18), ~1.4 slot-h"
$PY -u scripts/launch_sweep.py manifests/x_d_wd_ladder.jsonl  --gpus $GPUS --per-gpu 2
$PY -u scripts/launch_sweep.py manifests/x_d_lr_control.jsonl --gpus $GPUS --per-gpu 2
$PY scripts/collect_runs.py; status "tier X"

step "3/4  exp5 -- t3_algorithm_comparison (90 runs, ~38 slot-h)"
$PY -u scripts/launch_sweep.py manifests/t3_algorithm_comparison.jsonl --gpus $GPUS --per-gpu 2
$PY scripts/collect_runs.py; status "exp5"

step "4/4  t3b_small -- exp3b on B, C(K<=20), D (39 runs, ~85 slot-h)"
$PY -u scripts/launch_sweep.py logs/sweeps/split/t3b_small.jsonl --gpus $GPUS --per-gpu 2
$PY scripts/collect_runs.py; status "t3b_small"

cat ALL5_STATUS.md
step "DONE"
