# exp3b (resolvable setups) finished 2026-08-10T01:10:07+01:00

## the diagnostics that ran first
### A' at E=1, K=2 -- must reproduce the centralized ceiling (~4,000)
e1_identity A' 1 2         3    0  0.00        inf      inf      inf
### wd=0 on B, C, E -- do they grok without decay?
wd_zero B 0.0               3    0  0.00        inf      inf      inf
wd_zero C 0.0               3    0  0.00        inf      inf      inf
wd_zero E 0.0               3    0  0.00        inf      inf      inf

## exp3b: structured vs random shards
partitions A 10 dirichlet     3    3  1.00      13000    13000    13800
partitions A 10 operand       3    3  1.00      12700    12600    13400
partitions A 10 target        3    3  1.00      14100    13397    19565
partitions A 50 dirichlet     3    3  1.00      15000    14700    16000
partitions A 50 operand       3    3  1.00      13100    13000    14000
partitions A 50 target        3    3  1.00      28200    27500    30300
partitions A' 10 dirichlet    3    3  1.00      38200    34300    39000
partitions A' 10 operand      3    2  0.67      98400    98400      inf
partitions A' 10 target       3    3  1.00      18100    17900    19100
partitions A' 20 dirichlet    3    3  1.00      49300    47700    59100
partitions A' 20 operand      3    3  1.00      53300    53000    59200
partitions A' 20 target       3    3  1.00      33400    30500    35100
partitions D 10 operand       3    0  0.00        inf      inf      inf
partitions D 20 operand       3    0  0.00        inf      inf      inf
partitions E 10 dirichlet     3    1  0.33        inf     3500      inf
partitions E 10 label_block   3    0  0.00        inf      inf      inf
partitions E 20 dirichlet     3    3  1.00       6200     5800     6300
partitions E 20 label_block   3    0  0.00        inf      inf      inf

## still owed on exp3b (the noisy setups, deferred)
   logs/sweeps/split/t3b_later.jsonl -- B, C, D: 54 runs, ~190 slot-h
