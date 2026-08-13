# all-5 campaign -- last block finished: t3b_small
# 2026-08-11T09:30:22+01:00

## banked
total 1418  grokked 944  censored 474
machine-hours 820.2

## exp3b -- structured vs random shards (incl. coset)
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
partitions B 10 dirichlet     3    3  1.00       5300     4800     7600
partitions B 10 operand       3    2  0.67      96700    88600      inf
partitions B 10 target        3    0  0.00        inf      inf      inf
partitions B 20 dirichlet     3    2  0.67      25100     8200      inf
partitions B 20 operand       3    2  0.67      25700    18900      inf
partitions B 20 target        3    0  0.00        inf      inf      inf
partitions C 10 dirichlet     3    3  1.00      15600    12600    19600
partitions C 10 operand       3    3  1.00      11400     9200    14100
partitions C 10 target        3    0  0.00        inf      inf      inf
partitions C 5 coset          3    3  1.00       4300     4100     4500
partitions C 50 operand       3    0  0.00        inf      inf      inf
partitions C 50 target        3    0  0.00        inf      inf      inf
partitions D 10 dirichlet     3    0  0.00        inf      inf      inf
partitions D 10 operand       3    0  0.00        inf      inf      inf
partitions D 10 target        3    0  0.00        inf      inf      inf
partitions D 20 dirichlet     3    2  0.67     223100   182400      inf
partitions D 20 operand       3    0  0.00        inf      inf      inf
partitions D 20 target        3    0  0.00        inf      inf      inf
partitions D 5 coset          3    0  0.00        inf      inf      inf
partitions E 10 dirichlet     3    1  0.33        inf     3500      inf
partitions E 10 label_block   3    0  0.00        inf      inf      inf
partitions E 20 dirichlet     3    3  1.00       6200     5800     6300
partitions E 20 label_block   3    0  0.00        inf      inf      inf

## exp5 -- algorithms at calibrated server LRs
algorithms H1 fedadam    5    5  1.00       3000     2500     4000
algorithms H1 fedavg     5    5  1.00      45500    39000    50000
algorithms H1 fedavgm    5    5  1.00       7500     6500     8000
algorithms H1 fedprox    5    0  0.00        inf      inf      inf
algorithms H1 fedyogi    5    5  1.00       5000     3500    17000
algorithms H1 scaffold   5    5  1.00       4500     4000     5000
algorithms H2 fedadam    5    5  1.00       3000     2500     3000
algorithms H2 fedavg     5    5  1.00      61000    47500    75500
algorithms H2 fedavgm    5    5  1.00      10000     7500    10000
algorithms H2 fedprox    5    0  0.00        inf      inf      inf
algorithms H2 fedyogi    5    5  1.00       2500     2500     8000
algorithms H2 scaffold   5    5  1.00       5000     5000     5500
algorithms H3 fedadam    5    5  1.00       3000     3000     3000
algorithms H3 fedavg     5    5  1.00      31000    29000    35000
algorithms H3 fedavgm    5    5  1.00       6000     6000     7000
algorithms H3 fedprox    5    5  1.00     345000   333000   363000
algorithms H3 fedyogi    5    5  1.00       2000     2000     3000
algorithms H3 scaffold   5    5  1.00       4000     4000     4000

## tier X -- setup D wd ladder / lr control
d_lr_control 0.25 0.004         6    6  1.00      13600    13100    16700
d_lr_control 1.0 0.001          6    6  1.00       4275     1275     9650
d_lr_control 4.0 0.00025        6    6  1.00       2025     1950     3650
d_wd_ladder 0.0 0.001           9    0  0.00        inf      inf      inf
d_wd_ladder 0.1 0.001           9    6  0.67      23325    10100      inf
d_wd_ladder 0.3 0.001           9    6  0.67      19500    15250      inf
d_wd_ladder 1.0 0.001           9    9  1.00       9650     4150    21300
d_wd_ladder 3.0 0.001           9    6  0.67       2000      625      inf

## still owed
  manifests/t1_probe.jsonl                         6 of 24
  manifests/t1_replication.jsonl                 126 of 150
  manifests/t2_phase_diagram.jsonl               307 of 415
  manifests/t3b_partitions.jsonl                   3 of 144
  logs/sweeps/split/t3b_bigk.jsonl                 3 of 9
  logs/sweeps/split/t3b_later.jsonl                3 of 66
