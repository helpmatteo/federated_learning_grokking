"""Experiment 7: Re-run representative configs with full checkpoint saving.

6 configs spanning the mechanistic spectrum:
  1. Centralized baseline (alpha=0.5)
  2. FL K=5, alpha=0.5, IID (healthy grok)
  3. FL K=10, alpha=0.5, IID (standard FL)
  4. FL K=10, alpha=0.25, IID (near boundary — long run)
  5. FL K=10, alpha=0.5, non-IID dir=0.01 (extreme heterogeneity)
  6. FL K=97, alpha=0.3, IID (many clients, delayed grok)

Each saves model checkpoints every 100 rounds (or epochs) for post-hoc
Fourier analysis, effective rank tracking, and per-client weight decomposition.
"""

import os
import sys
import time

from core.config import Config
from federated.config import FedConfig
from centralized.train import train as centralized_train
from federated.train import fed_train, _dataset_cache


OUTPUT_DIR = "results/experiments/exp7_mechanistic"
SEED = 42
CHECKPOINT_EVERY = 100


def run_config_1():
    """Centralized baseline: alpha=0.5, 10k epochs."""
    print("\n" + "=" * 70)
    print("Config 1: Centralized baseline (alpha=0.5)")
    print("=" * 70)
    cfg = Config(
        task="addition", p=97, optimizer="gd", lr=50.0,
        weight_decay=0.0, hidden_width=256, activation="quadratic",
        alpha=0.5, seed=SEED, epochs=10_000, log_every=100,
        output_dir=os.path.join(OUTPUT_DIR, "c1_centralized"),
        checkpoint_every=CHECKPOINT_EVERY,
    )
    t0 = time.time()
    history, model = centralized_train(cfg)
    print(f"Config 1 done in {time.time() - t0:.1f}s")
    return history


def run_config_2():
    """FL K=5, alpha=0.5, IID — healthy grok."""
    print("\n" + "=" * 70)
    print("Config 2: FL K=5, alpha=0.5, IID")
    print("=" * 70)
    _dataset_cache.clear()
    cfg = FedConfig(
        task="addition", p=97, optimizer="gd", lr=50.0,
        weight_decay=0.0, hidden_width=256, activation="quadratic",
        alpha=0.5, seed=SEED, num_clients=5, num_rounds=2000,
        local_epochs=5, fraction_train=1.0, partition="iid",
        output_dir=os.path.join(OUTPUT_DIR, "c2_fl_K5_iid"),
        checkpoint_every=CHECKPOINT_EVERY,
        checkpoint_client_weights=True,
    )
    t0 = time.time()
    history, model = fed_train(cfg)
    print(f"Config 2 done in {time.time() - t0:.1f}s")
    return history


def run_config_3():
    """FL K=10, alpha=0.5, IID — standard FL."""
    print("\n" + "=" * 70)
    print("Config 3: FL K=10, alpha=0.5, IID")
    print("=" * 70)
    _dataset_cache.clear()
    cfg = FedConfig(
        task="addition", p=97, optimizer="gd", lr=50.0,
        weight_decay=0.0, hidden_width=256, activation="quadratic",
        alpha=0.5, seed=SEED, num_clients=10, num_rounds=2000,
        local_epochs=5, fraction_train=1.0, partition="iid",
        output_dir=os.path.join(OUTPUT_DIR, "c3_fl_K10_iid"),
        checkpoint_every=CHECKPOINT_EVERY,
        checkpoint_client_weights=True,
    )
    t0 = time.time()
    history, model = fed_train(cfg)
    print(f"Config 3 done in {time.time() - t0:.1f}s")
    return history


def run_config_4():
    """FL K=10, alpha=0.25, IID — near boundary, long run."""
    print("\n" + "=" * 70)
    print("Config 4: FL K=10, alpha=0.25, IID (near boundary)")
    print("=" * 70)
    _dataset_cache.clear()
    cfg = FedConfig(
        task="addition", p=97, optimizer="gd", lr=50.0,
        weight_decay=0.0, hidden_width=256, activation="quadratic",
        alpha=0.25, seed=SEED, num_clients=10, num_rounds=10000,
        local_epochs=5, fraction_train=1.0, partition="iid",
        output_dir=os.path.join(OUTPUT_DIR, "c4_fl_K10_boundary"),
        checkpoint_every=CHECKPOINT_EVERY,
        checkpoint_client_weights=True,
    )
    t0 = time.time()
    history, model = fed_train(cfg)
    print(f"Config 4 done in {time.time() - t0:.1f}s")
    return history


def run_config_5():
    """FL K=10, alpha=0.5, non-IID dir=0.01 — extreme heterogeneity."""
    print("\n" + "=" * 70)
    print("Config 5: FL K=10, alpha=0.5, Dirichlet(0.01)")
    print("=" * 70)
    _dataset_cache.clear()
    cfg = FedConfig(
        task="addition", p=97, optimizer="gd", lr=50.0,
        weight_decay=0.0, hidden_width=256, activation="quadratic",
        alpha=0.5, seed=SEED, num_clients=10, num_rounds=2000,
        local_epochs=5, fraction_train=1.0,
        partition="dirichlet", dirichlet_alpha=0.01,
        output_dir=os.path.join(OUTPUT_DIR, "c5_fl_K10_noniid"),
        checkpoint_every=CHECKPOINT_EVERY,
        checkpoint_client_weights=True,
    )
    t0 = time.time()
    history, model = fed_train(cfg)
    print(f"Config 5 done in {time.time() - t0:.1f}s")
    return history


def run_config_6():
    """FL K=97, alpha=0.3, IID — many clients, delayed grok."""
    print("\n" + "=" * 70)
    print("Config 6: FL K=97, alpha=0.3, IID")
    print("=" * 70)
    _dataset_cache.clear()
    cfg = FedConfig(
        task="addition", p=97, optimizer="gd", lr=50.0,
        weight_decay=0.0, hidden_width=256, activation="quadratic",
        alpha=0.3, seed=SEED, num_clients=97, num_rounds=2000,
        local_epochs=5, fraction_train=1.0, partition="iid",
        output_dir=os.path.join(OUTPUT_DIR, "c6_fl_K97_iid"),
        checkpoint_every=CHECKPOINT_EVERY,
        checkpoint_client_weights=True,
    )
    t0 = time.time()
    history, model = fed_train(cfg)
    print(f"Config 6 done in {time.time() - t0:.1f}s")
    return history


CONFIGS = {
    1: run_config_1,
    2: run_config_2,
    3: run_config_3,
    4: run_config_4,
    5: run_config_5,
    6: run_config_6,
}


def run_exp7(configs=None):
    """Run all (or specified) mechanistic checkpoint configs."""
    if configs is None:
        configs = sorted(CONFIGS.keys())

    print(f"Running Exp 7 configs: {configs}")
    total_t0 = time.time()

    for c in configs:
        CONFIGS[c]()

    total = time.time() - total_t0
    print(f"\nExp 7 complete. Total time: {total:.0f}s ({total/60:.1f}min)")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        configs = [int(c) for c in sys.argv[1:]]
        run_exp7(configs)
    else:
        run_exp7()
