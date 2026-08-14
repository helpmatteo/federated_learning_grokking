# size control finished 2026-08-14T15:52:16+01:00

## the control (dirichlet_sizes: same shard sizes, IID labels)
size_control 20 0.01       3    2  0.67      32100    29800      inf
size_control 20 0.1        3    3  1.00      29900    27300    33500
size_control 50 0.01       3    1  0.33        inf    47800      inf
size_control 50 0.1        3    3  1.00      45900    42100    48000

## against the concentrated arm it controls for
dirichlet_band 20 0.01     3    2  0.67      69900    49800      inf
dirichlet_band 20 0.1      3    3  1.00      34300    31000    39800
dirichlet_band 20 1.0      3    3  1.00      29900    28100    32300
dirichlet_band 20 10.0     3    3  1.00      28900    27300    31500
dirichlet_band 20 1000.0   3    3  1.00      28800    26500    31700
dirichlet_band 50 0.01     3    0  0.00        inf      inf      inf
dirichlet_band 50 0.1      3    3  1.00      53400    49700    55800
dirichlet_band 50 1.0      3    3  1.00      43800    38200    46100

## and the matched iid reference
boundary 20 iid                   5    5  1.00      29800    27300    33100
boundary 50 iid                   5    5  1.00      46800    40700    51400
boundary 97 iid                   5    2  0.40        inf    95600      inf
boundary 97 operand               5    5  1.00      76500    57800    77500
