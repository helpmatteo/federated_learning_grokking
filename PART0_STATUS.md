# Part 0 chain finished 2026-08-08T13:34:06+01:00

## Banked
total 968  grokked 664  censored 304
machine-hours 338.3

## 0.2 THE GATE -- does C work at alpha=0.30 at width 256?
c_alpha_w256 C 256 0.25         3    0  0.00        inf      inf      inf
c_alpha_w256 C 256 0.3          3    1  0.33        inf    55500      inf
c_alpha_w256 C 256 0.4          3    3  1.00      63950    10900    88550
c_alpha_w256 C 256 0.6          3    3  1.00      52850     2550    77000

## 0.3 capacity -- is the inherited width binding?
capacity A' 128             3    3  1.00       1050      950     1050
capacity A' 256             3    3  1.00       3900     3000     8500
capacity A' 512             3    0  0.00        inf      inf      inf
capacity B 128              3    3  1.00      61950     6100    95750
capacity B 256              3    3  1.00      88350    52150    89450
capacity B 64               3    3  1.00       4650     3650    81700
capacity D 128              3    0  0.00        inf      inf      inf
capacity D 512              3    3  1.00      18550    17650    18750
capacity E 100              3    3  1.00       5600     2175     5625
capacity E 200              3    3  1.00        725      700      725
capacity E 400              3    3  1.00        600      575      625

## 0.4 K ladder -- t_memo(K) and delay(K) per setup
setup_k_ladder B 10      3    3  1.00      55900    55200    77200
setup_k_ladder B 5       3    3  1.00      51200    44500    65900
setup_k_ladder C 10      3    3  1.00       7800     5000    13900
setup_k_ladder C 20      3    3  1.00       7900     5600     9100
setup_k_ladder C 5       3    3  1.00       4700     4100     7400
setup_k_ladder C 50      3    3  1.00      36500    31100    56900
setup_k_ladder D 10      3    2  0.67      99200    69200      inf
setup_k_ladder D 20      3    1  0.33        inf    97500      inf
setup_k_ladder D 5       3    3  1.00      27700    25500    29200
setup_k_ladder D 50      3    0  0.00        inf      inf      inf
setup_k_ladder E 10      3    3  1.00       5200     5100     5500
setup_k_ladder E 20      3    3  1.00      11700    11100    11900
setup_k_ladder E 5       3    1  0.33        inf     2700      inf

## still owed
  t1_probe                        6 of 24
  t1_replication                126 of 150
  t2_phase_diagram              307 of 415
  t3_algorithm_comparison        90 of 90
  x_d_lr_control                 18 of 18
  x_d_wd_ladder                  33 of 45
