# Grokking in Federated Learning — Experiment Suite

Empirical study characterising when and why grokking survives — or fails — under federated learning. Reproduces Gromov (2023) "Grokking modular arithmetic" and extends it to federated settings with Flower.

## Setup

```bash
# Create venv (requires Python 3.12 for Ray/Flower compatibility)
python3.12 -m venv .venv
source .venv/bin/activate
pip install torch numpy matplotlib "flwr[simulation]>=1.27" pytest
```

## Quick Start

```bash
# Run a single experiment
python run_experiment.py exp0

# Run with custom parameters
python run_experiment.py exp1 --hidden_width 256

# Generate plots from saved results
python run_experiment.py plot --exp exp1 --results results/exp1_boundary/exp1_results.json
```

## Experiment Pipeline

The experiments must be run **in order** — each depends on results from the previous:

```
Exp 0  (width validation)
  |
  v
Exp 1  (centralized phase boundary)  --> determines alpha_crit, T_max
  |
  v
Exp 2  (aggregation effect + FL boundary)
  |
  +---> Exp 3  (heterogeneity)          \
  |                                       |-- can run in parallel
  +---> Exp 4  (optimization fragmentation) /
         |
         v
       Exp 5  (algorithm rescue)
         |
         v
       Exp 6  (mechanistic analysis, post-hoc)
```

### Step-by-step

#### 1. Experiment 0 — Width Validation (~6 min)

Verifies that N=256 is sufficient across the alpha range. Determines T_base.

```bash
python run_experiment.py exp0
# Results: results/exp0_width/exp0_results.json
```

**What to check:** All widths should grok at alpha=0.5. Record the smallest alpha where N=256 groks — this hints at where the phase boundary lies.

#### 2. Experiment 1 — Centralized Phase Boundary (~96 min)

Sweeps alpha to find alpha_crit (where grokking fails). Determines T_max for all FL budgets.

```bash
python run_experiment.py exp1 --hidden_width 256
# Results: results/exp1_boundary/exp1_results.json
```

**What to check:** The output prints `alpha_crit` and `T_max`. Record these — they parameterise all subsequent experiments.

```bash
# Generate the phase boundary figure
python run_experiment.py plot --exp exp1 --results results/exp1_boundary/exp1_results.json
```

#### 3. Experiment 2 — Aggregation Effect (~102 hr sequential)

The main result: does FL aggregation compensate for data fragmentation? Three conditions per (alpha, K): centralized-full, centralized-reduced, FL-IID.

```bash
# Use alpha values near the boundary found in Exp 1
python run_experiment.py exp2 --alphas 0.05,0.1,0.15,0.2,0.3,0.5 --t_max <T_MAX_FROM_EXP1>

# Generate aggregation effect figure
python run_experiment.py plot --exp exp2 --results results/exp2_aggregation/exp2_results.json
```

#### 4. Experiments 3 & 4 — Heterogeneity & Optimization (parallel, ~86+95 hr)

These can run in parallel on separate machines.

```bash
# Exp 3a: Dirichlet heterogeneity sweep
python run_experiment.py exp3a --alphas 0.1,0.15,0.2,0.3,0.5 --k 10 --t_max <T_MAX>

# Exp 3b: Structured partition comparison
python run_experiment.py exp3b --alphas 0.1,0.15,0.2,0.3,0.5 --k 10 --t_max <T_MAX>

# Exp 4a: Drift accumulation x heterogeneity
python run_experiment.py exp4a --alphas 0.2,0.3,0.5 --k 10 --t_max <T_MAX>

# Exp 4b: Partial participation x heterogeneity
python run_experiment.py exp4b --alphas 0.2,0.3,0.5 --k 10 --t_max <T_MAX>

# Exp 4c: Compute vs communication budget
python run_experiment.py exp4c --alphas 0.2,0.3,0.5 --k 10
```

#### 5. Experiment 5 — Algorithm Rescue (~110 hr)

Uses "hard" settings from Exps 2-4 where FedAvg fails. Create a JSON file with the hard settings:

```bash
cat > hard_settings.json << 'EOF'
[
  {"label": "H1", "alpha": 0.15, "K": 20, "partition": "iid", "local_epochs": 5, "fraction_train": 1.0},
  {"label": "H2", "alpha": 0.15, "K": 10, "partition": "dirichlet", "local_epochs": 5, "fraction_train": 1.0, "dirichlet_alpha": 0.1},
  {"label": "H3", "alpha": 0.3, "K": 10, "partition": "iid", "local_epochs": 25, "fraction_train": 0.4}
]
EOF

python run_experiment.py exp5 --hard_settings hard_settings.json --t_max <T_MAX>
```

#### 6. Experiment 6 — Mechanistic Analysis (fast, post-hoc)

No new training. Analyses drift vs grokking across all prior FL runs.

```bash
python run_experiment.py exp6
# Results: results/exp6_mechanistic/drift_vs_grokking.json

python run_experiment.py plot --exp exp6 --results results/exp6_mechanistic/drift_vs_grokking.json
```

## Architecture

```
core/                          # Shared components
  config.py                    # Config dataclass (p, alpha, N, optimizer, lr, ...)
  model.py                     # GrokNet: 2-layer MLP, quadratic activation, mean-field init
  dataset.py                   # Modular arithmetic dataset (one-hot encoded)
  metrics.py                   # Weight norms, IPR, Fourier spectrum, accuracy
  utils.py                     # Device selection, optimizer factory

centralized/                   # Baseline training
  train.py                     # Full-batch GD loop

federated/                     # Flower-based FL
  config.py                    # FedConfig (K, rounds, E, partition, strategy, ...)
  dataset.py                   # Data partitioning (IID, operand, target, Dirichlet)
  train.py                     # FedAvg/FedProx/FedAdam via Flower simulation

experiments/                   # Experiment orchestration
  grokking_metrics.py          # T_grok, T_50 detection + multi-seed aggregation
  runner.py                    # Multi-seed runner with adaptive step budgets
  exp0_width.py                # Width validation
  exp1_boundary.py             # Centralized phase boundary
  exp2_aggregation.py          # Aggregation effect (centralized-full/reduced/FL)
  exp3_heterogeneity.py        # Dirichlet + structured partition sweeps
  exp4_optimization.py         # Drift, participation, compute vs communication
  exp5_algorithms.py           # Algorithm comparison (FedAvg/FedProx/FedAdam/WD)
  exp6_mechanistic.py          # Post-hoc drift vs grokking analysis
  visualization.py             # Publication-quality figures

run_experiment.py              # Unified CLI entry point
```

## Key Definitions

| Symbol | Meaning |
|--------|---------|
| **T_grok** | Smallest step where test_acc >= 95% and never drops below |
| **T_50** | First step where test_acc >= 50% (onset of generalisation) |
| **alpha_crit** | Critical training fraction below which grokking fails |
| **S_FL** | Adaptive FL step budget: min(50000, ceil(1.5 * T_max / 1000) * 1000) |
| **S_rescue** | Extended budget for Exp 5: min(80000, 2 * T_max) |

## FL Strategies

| Strategy | CLI flag | Description |
|----------|----------|-------------|
| FedAvg | `--strategy fedavg` | Default; weighted averaging of client models |
| FedProx | `--strategy fedprox --proximal_mu 0.1` | FedAvg + proximal regularisation on client |
| FedAdam | `--strategy fedadam --server_lr 0.1` | Server-side adaptive optimiser |

## Metrics Tracked

Every federated run logs per-round:
- Train/test loss and accuracy (global model on full data)
- Weight norms (W1, W2 Frobenius)
- IPR (Inverse Participation Ratio — Fourier structure indicator)
- Mean client drift (||w_after - w_before||_F averaged over clients)
- Client weight divergence (std of client weight norms)

## Tests

```bash
source .venv/bin/activate
python -m pytest tests/ -v          # Full suite (~3 min, includes FL integration)
python -m pytest tests/ -v -k "not Fed"  # Fast unit tests only (~2 sec)
```

## Estimated Compute Budget

| Experiment | Runs | Type | Sequential runtime |
|-----------|------|------|-------------------|
| Exp 0 | 12 | Centralized | ~6 min |
| Exp 1 | 24 | Centralized | ~96 min |
| Exp 2 | 234 | Mixed | ~102 hr |
| Exp 3 | ~147 | FL | ~86 hr |
| Exp 4 | ~160 | FL | ~95 hr |
| Exp 5 | 99 | FL | ~110 hr |
| Exp 6 | 0 | Post-hoc | minutes |

All runs within each experiment are independent and can be parallelised across machines by running subsets of the parameter grid.

## References

- Gromov, A. (2023). "Grokking modular arithmetic." arXiv:2301.02679
