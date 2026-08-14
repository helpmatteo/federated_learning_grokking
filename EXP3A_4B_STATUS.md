# exp3a + exp4b finished 2026-08-14T02:13:48+01:00

## exp4b -- partial participation at alpha=0.25, K=50 (control f=1.0 is t2_boundary)
boundary 1.0           20   17  0.85      51400    40700    77200
participation 0.2       3    3  1.00       8260     7700     9000
participation 0.4       3    3  1.00      16680    15680    18680
participation 0.6       3    3  1.00      25920    24000    31920

## exp3a -- Dirichlet concentration at the boundary
dirichlet_band 20 0.01     3    2  0.67      69900    49800      inf
dirichlet_band 20 0.1      3    3  1.00      34300    31000    39800
dirichlet_band 20 1.0      3    3  1.00      29900    28100    32300
dirichlet_band 20 10.0     3    3  1.00      28900    27300    31500
dirichlet_band 20 1000.0   3    3  1.00      28800    26500    31700
dirichlet_band 50 0.01     3    0  0.00        inf      inf      inf
dirichlet_band 50 0.1      3    3  1.00      53400    49700    55800
dirichlet_band 50 1.0      3    3  1.00      43800    38200    46100

## the banked iid controls at the same cells
boundary 20 iid               5    5  1.00      29800    27300    33100
boundary 50 iid               5    5  1.00      46800    40700    51400
boundary 97 iid               5    2  0.40        inf    95600      inf
boundary 97 operand           5    5  1.00      76500    57800    77500
