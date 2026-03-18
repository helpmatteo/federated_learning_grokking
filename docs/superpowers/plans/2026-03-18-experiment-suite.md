# Experiment Suite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the full experiment infrastructure for a NeurIPS-quality empirical study of grokking in federated learning, covering 6 experiments (~650 runs) with multi-seed support, early abort, adaptive budgets, new FL strategies (FedAdam), client drift metrics, and publication-quality visualization.

**Architecture:** The implementation extends the existing `core/`, `centralized/`, `federated/` package structure with a new `experiments/` package for experiment orchestration. Each experiment (0-6) gets its own module. A shared runner handles multi-seed execution, early abort, and result aggregation. New metrics (client drift, Fourier spectrum) are added to `core/metrics.py` and `federated/train.py`. FedAdam is added via Flower's native strategy. All experiments are runnable via a unified `run_experiment.py` CLI.

**Tech Stack:** Python 3.12, PyTorch, Flower (flwr[simulation]), Ray, matplotlib, numpy

**Spec document:** `experiment_plan.md` (root of repo)

---

## File Structure

### New files to create:
```
experiments/
  __init__.py                  # Package init
  runner.py                    # Multi-seed experiment runner with early abort
  grokking_metrics.py          # T_grok, T_50 computation and result aggregation
  exp0_width.py                # Exp 0: Width validation
  exp1_boundary.py             # Exp 1: Centralized phase boundary
  exp2_aggregation.py          # Exp 2: Aggregation effect & FL boundary
  exp3_heterogeneity.py        # Exp 3: Heterogeneity at phase boundary
  exp4_optimization.py         # Exp 4a/b/c: Optimization fragmentation
  exp5_algorithms.py           # Exp 5: Algorithm comparison & rescue
  exp6_mechanistic.py          # Exp 6: Mechanistic analysis (post-hoc)
  visualization.py             # Publication-quality figures for all experiments
run_experiment.py              # Unified CLI entry point
```

### Existing files to modify:
```
core/metrics.py                # Add: fourier_spectrum(), per-client IPR
federated/config.py            # Add: strategy, server_lr, tau, track_client_drift
federated/train.py             # Add: FedAdam strategy, client drift tracking, weight divergence
centralized/train.py           # Add: seed in history tag for multi-seed uniqueness
tests/conftest.py              # Add: fixtures for new config fields
tests/test_grokking_metrics.py # New: tests for T_grok, T_50
tests/test_runner.py           # New: tests for experiment runner
tests/test_fedadam.py          # New: tests for FedAdam integration
tests/test_client_drift.py     # New: tests for drift metrics
```

---

## Phase 1: Core Analysis Infrastructure

*No dependencies. Can run in parallel with Phase 2.*

### Task 1: Grokking Metrics Module

**Files:**
- Create: `experiments/__init__.py`
- Create: `experiments/grokking_metrics.py`
- Create: `tests/test_grokking_metrics.py`

- [ ] **Step 1: Create package init**

Create `experiments/__init__.py` (empty file).

- [ ] **Step 2: Write failing tests for T_grok computation**

```python
# tests/test_grokking_metrics.py
import pytest
from experiments.grokking_metrics import compute_t_grok, compute_t_50, summarize_seeds


class TestTGrok:
    def test_clear_grokking(self):
        """Test accuracy stays above 95% from step 500 onward."""
        steps = list(range(0, 1000, 100))
        accs = [1.0, 1.0, 1.0, 10.0, 50.0, 96.0, 97.0, 98.0, 99.0, 99.5]
        assert compute_t_grok(steps, accs, threshold=95.0) == 500

    def test_no_grokking(self):
        """Test accuracy never reaches 95%."""
        steps = list(range(0, 1000, 100))
        accs = [1.0] * 10
        assert compute_t_grok(steps, accs, threshold=95.0) == float("inf")

    def test_transient_spike_rejected(self):
        """Spike above 95% then drops — should not count."""
        steps = list(range(0, 800, 100))
        accs = [1.0, 1.0, 96.0, 80.0, 96.0, 97.0, 98.0, 99.0]
        # First sustained crossing is at step 400
        assert compute_t_grok(steps, accs, threshold=95.0) == 400

    def test_single_point_at_end(self):
        """Only the last point is above 95% — still counts (no subsequent drop)."""
        steps = [0, 100, 200]
        accs = [1.0, 50.0, 96.0]
        assert compute_t_grok(steps, accs, threshold=95.0) == 200

    def test_empty_input(self):
        assert compute_t_grok([], [], threshold=95.0) == float("inf")


class TestT50:
    def test_onset_detected(self):
        steps = [0, 100, 200, 300]
        accs = [1.0, 30.0, 55.0, 90.0]
        assert compute_t_50(steps, accs) == 200

    def test_no_onset(self):
        steps = [0, 100, 200]
        accs = [1.0, 1.0, 1.0]
        assert compute_t_50(steps, accs) == float("inf")


class TestSummarizeSeeds:
    def test_aggregation(self):
        results = [
            {"t_grok": 500, "t_50": 300, "final_test_acc": 99.0},
            {"t_grok": 600, "t_50": 350, "final_test_acc": 98.5},
            {"t_grok": 550, "t_50": 320, "final_test_acc": 99.2},
        ]
        summary = summarize_seeds(results)
        assert summary["t_grok_mean"] == pytest.approx(550.0)
        # np.std with ddof=1 (sample std): sqrt(((500-550)^2+(600-550)^2+(550-550)^2)/2)=50
        assert summary["t_grok_std"] == pytest.approx(50.0, abs=0.1)
        assert summary["n_grokked"] == 3
        assert summary["n_seeds"] == 3

    def test_partial_grokking(self):
        results = [
            {"t_grok": 500, "t_50": 300, "final_test_acc": 99.0},
            {"t_grok": float("inf"), "t_50": float("inf"), "final_test_acc": 1.0},
        ]
        summary = summarize_seeds(results)
        assert summary["t_grok_mean"] == float("inf")  # not all grokked
        assert summary["n_grokked"] == 1
        assert summary["n_seeds"] == 2
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd /Users/m/Desktop/federated_learning_grokking && source .venv/bin/activate && python -m pytest tests/test_grokking_metrics.py -v`
Expected: FAIL (module not found)

- [ ] **Step 4: Implement grokking_metrics.py**

```python
# experiments/grokking_metrics.py
"""Grokking step detection and multi-seed result aggregation.

Definitions from experiment_plan.md Section 3.1:
- T_grok: smallest step t_j such that test_acc >= threshold for ALL subsequent steps
- T_50: smallest step t_j such that test_acc >= 50%
"""

import math
from typing import List


def compute_t_grok(steps: list, test_accs: list, threshold: float = 95.0) -> float:
    """Compute grokking step T_grok.

    Returns the smallest step where test accuracy reaches `threshold`
    and never drops below it for the remainder of training.
    Returns float('inf') if no such step exists.
    """
    if not steps:
        return float("inf")

    n = len(steps)
    # Scan from the end to find the latest point where acc < threshold
    # Everything after that point is sustained above threshold
    last_below = -1
    for i in range(n - 1, -1, -1):
        if test_accs[i] < threshold:
            last_below = i
            break

    if last_below == -1:
        # All points are above threshold — grokking at step 0
        return steps[0]
    if last_below == n - 1:
        # Last point is below threshold — no grokking
        return float("inf")

    # The first sustained crossing is the point after last_below
    return steps[last_below + 1]


def compute_t_50(steps: list, test_accs: list, threshold: float = 50.0) -> float:
    """Compute onset step T_50 — first step where test accuracy >= 50%."""
    for step, acc in zip(steps, test_accs):
        if acc >= threshold:
            return step
    return float("inf")


def extract_grokking_results(history: dict) -> dict:
    """Extract grokking metrics from a training history dict.

    Works with both centralized (key: 'epoch') and federated (key: 'total_steps').
    """
    steps = history.get("total_steps", history.get("epoch", []))
    test_accs = history.get("test_acc", [])
    train_accs = history.get("train_acc", [])

    t_grok = compute_t_grok(steps, test_accs)
    t_50 = compute_t_50(steps, test_accs)
    final_test_acc = test_accs[-1] if test_accs else 0.0
    final_train_acc = train_accs[-1] if train_accs else 0.0
    final_ipr = history.get("ipr", [0.0])[-1] if history.get("ipr") else 0.0

    return {
        "t_grok": t_grok,
        "t_50": t_50,
        "final_test_acc": final_test_acc,
        "final_train_acc": final_train_acc,
        "final_ipr": final_ipr,
    }


def summarize_seeds(results: List[dict]) -> dict:
    """Aggregate grokking metrics across seeds (mean +/- std).

    If not ALL seeds grokked, t_grok_mean = inf (conservative).
    t_50 is averaged only over seeds that achieved T_50.
    """
    import numpy as np

    n = len(results)
    t_groks = [r["t_grok"] for r in results]
    t_50s = [r["t_50"] for r in results]
    final_accs = [r["final_test_acc"] for r in results]

    n_grokked = sum(1 for t in t_groks if t < float("inf"))
    finite_groks = [t for t in t_groks if t < float("inf")]
    finite_50s = [t for t in t_50s if t < float("inf")]

    return {
        "n_seeds": n,
        "n_grokked": n_grokked,
        "t_grok_mean": float(np.mean(finite_groks)) if n_grokked == n else float("inf"),
        "t_grok_std": float(np.std(finite_groks, ddof=1)) if n_grokked == n and n > 1 else 0.0 if n_grokked == n else float("inf"),
        "t_50_mean": float(np.mean(finite_50s)) if finite_50s else float("inf"),
        "t_50_std": float(np.std(finite_50s, ddof=1)) if len(finite_50s) > 1 else 0.0 if finite_50s else float("inf"),
        "final_acc_mean": float(np.mean(final_accs)),
        "final_acc_std": float(np.std(final_accs)),
    }
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /Users/m/Desktop/federated_learning_grokking && source .venv/bin/activate && python -m pytest tests/test_grokking_metrics.py -v`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add experiments/__init__.py experiments/grokking_metrics.py tests/test_grokking_metrics.py
git commit -m "feat: add grokking metrics module (T_grok, T_50, seed aggregation)"
```

---

### Task 2: Experiment Runner with Early Abort

**Files:**
- Create: `experiments/runner.py`
- Create: `tests/test_runner.py`

- [ ] **Step 1: Write failing tests for early abort logic**

```python
# tests/test_runner.py
import pytest
from experiments.runner import should_abort, RunConfig


class TestShouldAbort:
    def test_no_abort_normal_training(self):
        """Normal training — no abort."""
        assert not should_abort(
            step=5000, train_acc=100.0, test_acc=1.0,
            t_base=8000, t_max=30000
        )

    def test_abort_memorization_failure(self):
        """Train acc < 50% at 2*T_base — abort."""
        assert should_abort(
            step=16000, train_acc=30.0, test_acc=1.0,
            t_base=8000, t_max=30000
        )

    def test_no_abort_memorization_before_deadline(self):
        """Train acc < 50% but before 2*T_base — don't abort yet."""
        assert not should_abort(
            step=10000, train_acc=30.0, test_acc=1.0,
            t_base=8000, t_max=30000
        )

    def test_abort_generalization_hopeless(self):
        """Train 100%, test < 5% at T_max — abort."""
        assert should_abort(
            step=30000, train_acc=100.0, test_acc=2.0,
            t_base=8000, t_max=30000
        )

    def test_no_abort_generalization_progressing(self):
        """Train 100%, test > 5% — don't abort."""
        assert not should_abort(
            step=30000, train_acc=100.0, test_acc=10.0,
            t_base=8000, t_max=30000
        )


class TestRunConfig:
    def test_s_fl_formula(self):
        """S_FL = min(50000, ceil(1.5 * T_max / 1000) * 1000)"""
        rc = RunConfig(t_base=8000, t_max=30000)
        assert rc.s_fl == 45000

    def test_s_rescue_formula(self):
        """S_rescue = min(80000, 2 * T_max)"""
        rc = RunConfig(t_base=8000, t_max=30000)
        assert rc.s_rescue == 60000

    def test_s_fl_cap(self):
        """S_FL capped at 50000."""
        rc = RunConfig(t_base=8000, t_max=50000)
        assert rc.s_fl == 50000

    def test_s_rescue_cap(self):
        """S_rescue capped at 80000."""
        rc = RunConfig(t_base=8000, t_max=50000)
        assert rc.s_rescue == 80000
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_runner.py -v`
Expected: FAIL

- [ ] **Step 3: Implement runner.py**

```python
# experiments/runner.py
"""Multi-seed experiment runner with early abort and adaptive budgets.

Usage:
    runner = ExperimentRunner(seeds=[42, 123, 456], t_base=8000, t_max=30000)
    results = runner.run_centralized(cfg_template, s_max=100000)
    results = runner.run_federated(cfg_template, s_max=None)  # uses S_FL
"""

import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Optional

from experiments.grokking_metrics import extract_grokking_results, summarize_seeds


@dataclass
class RunConfig:
    """Adaptive step budgets derived from T_base and T_max (Section 3.2)."""
    t_base: int = 8000
    t_max: int = 30000

    @property
    def s_fl(self) -> int:
        """S_FL = min(50_000, ceil(1.5 * T_max / 1000) * 1000)"""
        raw = math.ceil(1.5 * self.t_max / 1000) * 1000
        return min(50_000, raw)

    @property
    def s_rescue(self) -> int:
        """S_rescue = min(80_000, 2 * T_max)"""
        return min(80_000, 2 * self.t_max)


def should_abort(step: int, train_acc: float, test_acc: float,
                 t_base: int, t_max: int) -> bool:
    """Check early abort conditions (Section 3.2).

    Rule 1: Memorisation failure — train_acc < 50% by min(2*T_base, 15000)
    Rule 2: Generalisation hopeless — train_acc ~100% and test_acc < 5% by T_max
    """
    mem_deadline = min(2 * t_base, 15000)
    if step >= mem_deadline and train_acc < 50.0:
        return True

    # 99.9% threshold approximates "100%" from spec (float equality impractical)
    if step >= t_max and train_acc >= 99.9 and test_acc < 5.0:
        return True

    return False


def run_single_centralized(cfg, label: str = "") -> dict:
    """Run one centralized experiment, return history + grokking metrics."""
    from centralized.train import train
    print(f"\n--- Centralized run: {label} seed={cfg.seed} ---")
    history, model = train(cfg)
    metrics = extract_grokking_results(history)
    metrics["history_path"] = os.path.join(
        cfg.output_dir,
        f"history_{cfg.task}_{cfg.optimizer}_p{cfg.p}_N{cfg.hidden_width}_a{cfg.alpha}_s{cfg.seed}.json"
    )
    return metrics


def run_single_federated(cfg, label: str = "") -> dict:
    """Run one federated experiment, return history + grokking metrics."""
    from federated.train import fed_train
    print(f"\n--- Federated run: {label} seed={cfg.seed} ---")
    history, model = fed_train(cfg)
    metrics = extract_grokking_results(history)
    return metrics


def run_multi_seed(run_fn, cfg_template, seeds: list, cfg_overrides: dict = None,
                   label: str = "") -> dict:
    """Run an experiment across multiple seeds and aggregate results.

    Args:
        run_fn: Either run_single_centralized or run_single_federated
        cfg_template: Base config (seed will be overridden)
        seeds: List of random seeds
        cfg_overrides: Additional config fields to override
        label: Human-readable experiment label

    Returns:
        Dict with per-seed results and aggregated summary.
    """
    from dataclasses import replace
    per_seed = []
    for seed in seeds:
        overrides = {"seed": seed}
        if cfg_overrides:
            overrides.update(cfg_overrides)
        cfg = replace(cfg_template, **overrides)
        result = run_fn(cfg, label=f"{label} seed={seed}")
        per_seed.append(result)
        print(f"  -> T_grok={result['t_grok']}, T_50={result['t_50']}, "
              f"final_acc={result['final_test_acc']:.1f}%")

    summary = summarize_seeds(per_seed)
    return {
        "label": label,
        "per_seed": per_seed,
        "summary": summary,
    }


def save_experiment_results(results: list, output_path: str):
    """Save experiment results to JSON, handling inf values."""
    def _sanitize(obj):
        if isinstance(obj, float) and math.isinf(obj):
            return "inf"
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitize(v) for v in obj]
        return obj

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(_sanitize(results), f, indent=2)
    print(f"Results saved to {output_path}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_runner.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add experiments/runner.py tests/test_runner.py
git commit -m "feat: add experiment runner with early abort and adaptive budgets"
```

---

## Phase 2: FL Strategy & Config Extensions

*No dependencies on Phase 1. Can run in parallel.*

### Task 3: Extend FedConfig with Strategy Parameters

**Files:**
- Modify: `federated/config.py`
- Modify: `tests/conftest.py`
- Create: `tests/test_strategy_config.py`

- [ ] **Step 1: Write failing tests for new config fields**

```python
# tests/test_strategy_config.py
import pytest
from federated.config import FedConfig


class TestFedConfigStrategy:
    def test_default_strategy_is_fedavg(self):
        cfg = FedConfig()
        assert cfg.strategy == "fedavg"

    def test_fedadam_params(self):
        cfg = FedConfig(strategy="fedadam", server_lr=0.1, tau=1e-3)
        assert cfg.strategy == "fedadam"
        assert cfg.server_lr == 0.1
        assert cfg.tau == 1e-3

    def test_track_client_drift_default_true(self):
        cfg = FedConfig()
        assert cfg.track_client_drift is True

    def test_weight_decay_in_fl(self):
        """FL experiments may use weight decay as explicit regularisation."""
        cfg = FedConfig(weight_decay=0.1)
        assert cfg.weight_decay == 0.1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_strategy_config.py -v`
Expected: FAIL (missing fields)

- [ ] **Step 3: Add new fields to FedConfig**

In `federated/config.py`, add after `proximal_mu`:

```python
    strategy: Literal[
        "fedavg", "fedprox", "fedadam"
    ] = "fedavg"
    server_lr: float = 1.0               # server-side learning rate (FedAdam)
    tau: float = 1e-3                     # FedAdam adaptivity parameter
    track_client_drift: bool = True       # enable per-round drift logging
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_strategy_config.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add federated/config.py tests/test_strategy_config.py
git commit -m "feat: add strategy, server_lr, tau, track_client_drift to FedConfig"
```

---

### Task 4: Add FedAdam Strategy to fed_train.py

**Files:**
- Modify: `federated/train.py`
- Create: `tests/test_fedadam.py`

- [ ] **Step 1: Write failing test for FedAdam strategy selection**

```python
# tests/test_fedadam.py
"""Test FedAdam integration — verifies strategy construction, not full training."""
import pytest
from unittest.mock import patch, MagicMock
from federated.config import FedConfig
from federated.train import _build_strategy


class TestBuildStrategy:
    def test_fedavg_default(self):
        cfg = FedConfig(p=7, hidden_width=16, num_clients=3, strategy="fedavg")
        strategy = _build_strategy(cfg, init_params=MagicMock(), evaluate_fn=lambda *a: (0, {}))
        from flwr.server.strategy import FedAvg
        assert isinstance(strategy, FedAvg)

    def test_fedadam_strategy(self):
        cfg = FedConfig(
            p=7, hidden_width=16, num_clients=3,
            strategy="fedadam", server_lr=0.1, tau=1e-3
        )
        # This also validates that Flower's FedAdam accepts `eta` parameter
        strategy = _build_strategy(cfg, init_params=MagicMock(), evaluate_fn=lambda *a: (0, {}))
        from flwr.server.strategy import FedAdam
        assert isinstance(strategy, FedAdam)

    def test_fedprox_uses_fedavg_strategy(self):
        """FedProx is FedAvg + proximal term on client side; strategy is still FedAvg."""
        cfg = FedConfig(
            p=7, hidden_width=16, num_clients=3,
            strategy="fedprox", proximal_mu=0.1
        )
        strategy = _build_strategy(cfg, init_params=MagicMock(), evaluate_fn=lambda *a: (0, {}))
        from flwr.server.strategy import FedAvg
        assert isinstance(strategy, FedAvg)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_fedadam.py -v`
Expected: FAIL (no _build_strategy function)

- [ ] **Step 3: Refactor fed_train.py — extract _build_strategy function and update serialization**

In `federated/train.py`, add this import at the top:

```python
from flwr.server.strategy import FedAdam
```

Also update `_cfg_to_fit_config` to include the new fields:

```python
        "strategy": cfg.strategy,
        "server_lr": cfg.server_lr,
        "tau": cfg.tau,
        "track_client_drift": cfg.track_client_drift,
```

And update `_fit_config_to_cfg` to reconstruct them:

```python
        strategy=config.get("strategy", "fedavg"),
        server_lr=float(config.get("server_lr", 1.0)),
        tau=float(config.get("tau", 1e-3)),
        track_client_drift=bool(config.get("track_client_drift", True)),
```

Then extract and replace the strategy construction in `server_fn` with a new top-level function:

```python
def _build_strategy(cfg: FedConfig, init_params, evaluate_fn,
                    fit_metrics_aggregation_fn=None):
    """Build Flower strategy based on config.strategy field."""
    common_kwargs = dict(
        fraction_fit=cfg.fraction_train,
        fraction_evaluate=0.0,
        min_fit_clients=max(1, int(cfg.num_clients * cfg.fraction_train)),
        min_available_clients=cfg.num_clients,
        initial_parameters=init_params,
        evaluate_fn=evaluate_fn,
        on_fit_config_fn=lambda rnd: _cfg_to_fit_config(cfg, rnd),
    )
    if fit_metrics_aggregation_fn is not None:
        common_kwargs["fit_metrics_aggregation_fn"] = fit_metrics_aggregation_fn

    if cfg.strategy == "fedadam":
        # NOTE: Flower's FedAdam uses `eta` for server LR. Verify against
        # installed version — some versions may use `server_learning_rate`.
        return FedAdam(
            **common_kwargs,
            eta=cfg.server_lr,
            tau=cfg.tau,
        )
    else:
        # Both "fedavg" and "fedprox" use FedAvg strategy;
        # FedProx proximal term is applied client-side in GrokClient.fit()
        return FedAvg(**common_kwargs)
```

Update `server_fn` inside `fed_train()` to call `_build_strategy`:

```python
    def server_fn(context: Context):
        strategy = _build_strategy(fed_cfg, init_params, evaluate_fn,
                                   fit_metrics_aggregation_fn=_aggregate_fit_metrics)
        return ServerAppComponents(
            strategy=strategy,
            config=ServerConfig(num_rounds=fed_cfg.num_rounds),
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_fedadam.py -v`
Expected: ALL PASS

- [ ] **Step 5: Run existing federated tests to ensure no regression**

Run: `python -m pytest tests/test_federated_train.py tests/test_fedavg_correctness.py -v`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add federated/train.py tests/test_fedadam.py
git commit -m "feat: add FedAdam strategy support via _build_strategy refactor"
```

---

## Phase 3: Client Drift Metrics

*No dependencies on Phases 1-2. Can run in parallel.*

### Task 5: Add Fourier Spectrum to core/metrics.py

**Files:**
- Modify: `core/metrics.py`
- Modify: `tests/test_core.py`

- [ ] **Step 1: Write failing test for fourier_spectrum**

Add to `tests/test_core.py` in the `TestMetrics` class:

```python
    def test_fourier_spectrum_shape(self, small_model):
        spec = fourier_spectrum(small_model)
        assert "spectrum" in spec
        # spectrum is (N, p) magnitudes
        assert len(spec["spectrum"]) == small_model.N
        assert len(spec["spectrum"][0]) == small_model.P

    def test_fourier_spectrum_nonnegative(self, small_model):
        spec = fourier_spectrum(small_model)
        for row in spec["spectrum"]:
            assert all(v >= 0 for v in row)
```

Update import at top of test file:
```python
from core.metrics import weight_norms, gradient_norms, compute_ipr, compute_accuracy, fourier_spectrum
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_core.py::TestMetrics::test_fourier_spectrum_shape -v`
Expected: FAIL

- [ ] **Step 3: Implement fourier_spectrum in core/metrics.py**

Add to `core/metrics.py`:

```python
def fourier_spectrum(model):
    """Full Fourier power spectrum |W_tilde_1(nu)|^2 per neuron.

    Returns dict with 'spectrum': list of lists (N x p), each entry is
    the squared magnitude of the Fourier coefficient at that frequency.
    """
    W1 = model.W1.data
    p = model.P

    W1_n = W1[:, :p]  # (N, p) — first-operand weights
    W1_fft = torch.fft.fft(W1_n, dim=1)  # (N, p) complex
    power = (W1_fft.abs() ** 2)  # (N, p)

    return {"spectrum": power.cpu().tolist()}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_core.py::TestMetrics -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add core/metrics.py tests/test_core.py
git commit -m "feat: add fourier_spectrum to core/metrics.py"
```

---

### Task 6: Add Client Drift Tracking to fed_train.py

**Files:**
- Modify: `federated/train.py`
- Create: `tests/test_client_drift.py`

- [ ] **Step 1: Write failing tests for drift metrics**

```python
# tests/test_client_drift.py
"""Test client drift metric collection in federated training."""
import pytest
import torch
import numpy as np
from federated.config import FedConfig
from federated.train import _model_to_ndarrays, compute_drift


class TestComputeDrift:
    def test_zero_drift_for_identical_weights(self):
        """No update = zero drift."""
        w_before = [np.zeros((3, 4)), np.zeros((2, 3))]
        w_after = [np.zeros((3, 4)), np.zeros((2, 3))]
        assert compute_drift(w_before, w_after) == pytest.approx(0.0)

    def test_nonzero_drift(self):
        w_before = [np.zeros((2, 2))]
        w_after = [np.ones((2, 2))]
        # Frobenius norm of 2x2 all-ones = 2.0
        assert compute_drift(w_before, w_after) == pytest.approx(2.0)

    def test_drift_is_frobenius_norm(self):
        w_before = [np.array([[1.0, 2.0], [3.0, 4.0]])]
        w_after = [np.array([[2.0, 3.0], [4.0, 5.0]])]
        # Diff is all ones 2x2, frobenius = 2.0
        assert compute_drift(w_before, w_after) == pytest.approx(2.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_client_drift.py -v`
Expected: FAIL

- [ ] **Step 3: Add compute_drift function and modify GrokClient.fit()**

Add to `federated/train.py`:

```python
def compute_drift(w_before: list, w_after: list) -> float:
    """Compute Frobenius norm of weight difference: ||w_after - w_before||_F."""
    total = 0.0
    for wb, wa in zip(w_before, w_after):
        diff = wa - wb
        total += float(np.sum(diff ** 2))
    return float(np.sqrt(total))
```

Modify `GrokClient.fit()` to return drift and per-client IPR in metrics. After the line `model.eval()`, before the return, add:

```python
        # Compute client drift if tracking enabled
        updated_weights = _model_to_ndarrays(model)
        drift = compute_drift(parameters, updated_weights)

        # Per-client weight norm (for cross-client divergence)
        weight_norm = float(sum(np.sum(w**2) for w in updated_weights) ** 0.5)

        # Per-client IPR
        local_ipr = compute_ipr(model)["ipr"]
```

And update the return to include all metrics:

```python
        return (
            updated_weights,
            len(y_local),
            {"loss": local_loss, "accuracy": local_acc, "drift": drift,
             "weight_norm": weight_norm, "ipr": local_ipr},
        )
```

- [ ] **Step 4: Add drift/divergence aggregation to fed_train()**

In `fed_train()`, extend the `history` dict:

```python
    history = {
        "round": [], "total_steps": [],
        "train_loss": [], "test_loss": [],
        "train_acc": [], "test_acc": [],
        "weight_norm_layer1": [], "weight_norm_layer2": [],
        "ipr": [],
        "mean_client_drift": [],
        "client_weight_divergence": [],
    }
```

Define `_aggregate_fit_metrics` and `_round_metrics` **inside `fed_train()`** (closure scope), NOT in `_build_strategy`:

```python
    # Mutable containers for inter-callback communication (closure-shared)
    _round_metrics = {"mean_drift": 0.0, "weight_divergence": 0.0}

    def _aggregate_fit_metrics(metrics_list):
        """Aggregate per-client fit metrics from GrokClient.fit()."""
        drifts = [m.get("drift", 0.0) for _, m in metrics_list]
        w_norms = [m.get("weight_norm", 0.0) for _, m in metrics_list]

        _round_metrics["mean_drift"] = float(np.mean(drifts)) if drifts else 0.0
        # Client weight divergence: std of weight norms across clients
        _round_metrics["weight_divergence"] = float(np.std(w_norms)) if len(w_norms) > 1 else 0.0

        return {
            "mean_drift": _round_metrics["mean_drift"],
            "weight_divergence": _round_metrics["weight_divergence"],
        }
```

Pass `_aggregate_fit_metrics` to `_build_strategy` (which now accepts it as a parameter — see Task 4):

```python
    def server_fn(context: Context):
        strategy = _build_strategy(fed_cfg, init_params, evaluate_fn,
                                   fit_metrics_aggregation_fn=_aggregate_fit_metrics)
        ...
```

Then in `evaluate_fn`, log both drift metrics:

```python
        history["mean_client_drift"].append(_round_metrics["mean_drift"])
        history["client_weight_divergence"].append(_round_metrics["weight_divergence"])
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_client_drift.py -v`
Expected: ALL PASS

- [ ] **Step 6: Run full federated test suite for regression**

Run: `python -m pytest tests/test_federated_train.py tests/test_fedavg_correctness.py -v`
Expected: ALL PASS

- [ ] **Step 7: Commit**

```bash
git add federated/train.py tests/test_client_drift.py
git commit -m "feat: add client drift tracking and weight divergence metrics"
```

---

## Phase 4: Centralized Train Enhancements

*Quick changes to support multi-seed and alpha-reduced baselines.*

### Task 7: Add Seed to Centralized History Tag & Early Abort Hook

**Files:**
- Modify: `centralized/train.py`

- [ ] **Step 1: Add seed to the history filename tag**

In `centralized/train.py`, modify the tag on line ~103:

Old:
```python
    tag = f"{cfg.task}_{cfg.optimizer}_p{cfg.p}_N{cfg.hidden_width}_a{cfg.alpha}"
```

New:
```python
    tag = f"{cfg.task}_{cfg.optimizer}_p{cfg.p}_N{cfg.hidden_width}_a{cfg.alpha}_s{cfg.seed}"
```

- [ ] **Step 2: Add seed to federated history tag**

In `federated/train.py`, modify the tag on line ~335-337 to include seed:

Old:
```python
    tag = (f"fed_{cfg.task}_{cfg.optimizer}_p{cfg.p}_N{cfg.hidden_width}"
           f"_a{cfg.alpha}_K{cfg.num_clients}_le{cfg.local_epochs}"
           f"_ft{cfg.fraction_train}_{cfg.partition}{dirichlet_suffix}{prox_suffix}")
```

New:
```python
    tag = (f"fed_{cfg.task}_{cfg.optimizer}_p{cfg.p}_N{cfg.hidden_width}"
           f"_a{cfg.alpha}_K{cfg.num_clients}_le{cfg.local_epochs}"
           f"_ft{cfg.fraction_train}_{cfg.partition}{dirichlet_suffix}"
           f"{prox_suffix}_s{cfg.seed}")
```

- [ ] **Step 3: Run existing tests to verify no regression**

Run: `python -m pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 4: Commit**

```bash
git add centralized/train.py federated/train.py
git commit -m "feat: include seed in history filenames for multi-seed experiments"
```

---

## Phase 5: Experiment Implementations

*Depends on Phases 1-4. Each experiment module is independent of the others.*

### Task 8: Experiment 0 — Width Validation

**Files:**
- Create: `experiments/exp0_width.py`

- [ ] **Step 1: Implement Exp 0**

```python
# experiments/exp0_width.py
"""Experiment 0: Width validation — verify N=256 works across alpha range.

Design (from experiment_plan.md):
  N in {100, 128, 200, 256}, alpha in {0.1, 0.3, 0.5}, seeds={42}
  S_max = 50,000 steps, centralized only.
  12 runs total, ~24 min.
"""

from dataclasses import replace
from core.config import Config
from experiments.runner import run_single_centralized, save_experiment_results


WIDTHS = [100, 128, 200, 256]
ALPHAS = [0.1, 0.3, 0.5]
SEEDS = [42]  # Single seed; extend to [42, 123, 456] if noisy
S_MAX = 50_000


def run_exp0(output_dir: str = "results/exp0_width"):
    """Run width validation experiment."""
    base_cfg = Config(
        task="addition",
        p=97,
        optimizer="gd",
        lr=50.0,
        weight_decay=0.0,
        momentum=0.0,
        log_every=100,
        output_dir=output_dir,
    )

    all_results = []

    for alpha in ALPHAS:
        for width in WIDTHS:
            for seed in SEEDS:
                cfg = replace(
                    base_cfg,
                    alpha=alpha,
                    hidden_width=width,
                    epochs=S_MAX,
                    seed=seed,
                )
                label = f"N={width}_a={alpha}_s={seed}"
                result = run_single_centralized(cfg, label=label)
                result["alpha"] = alpha
                result["width"] = width
                result["seed"] = seed
                all_results.append(result)
                print(f"  N={width}, alpha={alpha}: T_grok={result['t_grok']}, "
                      f"T_50={result['t_50']}, final_acc={result['final_test_acc']:.1f}%")

    save_experiment_results(all_results, f"{output_dir}/exp0_results.json")

    # Determine recommended N*
    # Pick the smallest width that groks at all tested alphas
    for width in WIDTHS:
        width_results = [r for r in all_results if r["width"] == width]
        all_grok = all(r["t_grok"] < float("inf") for r in width_results)
        if all_grok:
            print(f"\nRecommended N* = {width} (groks at all tested alphas)")
            break

    return all_results


if __name__ == "__main__":
    run_exp0()
```

- [ ] **Step 2: Commit**

```bash
git add experiments/exp0_width.py
git commit -m "feat: implement Experiment 0 — width validation"
```

---

### Task 9: Experiment 1 — Centralized Phase Boundary

**Files:**
- Create: `experiments/exp1_boundary.py`

- [ ] **Step 1: Implement Exp 1**

```python
# experiments/exp1_boundary.py
"""Experiment 1: Centralized grokking phase boundary.

Design (from experiment_plan.md):
  alpha in {0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5}
  seeds = {42, 123, 456}, S_max = 100,000 steps
  24 runs, ~96 min total.

  Outputs: alpha_crit, T_max, plot of T_grok vs alpha.
"""

from dataclasses import replace
from core.config import Config
from experiments.runner import run_multi_seed, save_experiment_results
from experiments.grokking_metrics import summarize_seeds


ALPHAS = [0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5]
SEEDS = [42, 123, 456]
S_MAX = 100_000


def run_exp1(hidden_width: int = 256,
             output_dir: str = "results/exp1_boundary"):
    """Run centralized phase boundary experiment.

    Args:
        hidden_width: N* from Exp 0 (default 256).

    Returns:
        dict with alpha_crit, t_max, and per-alpha results.
    """
    from experiments.runner import run_single_centralized

    base_cfg = Config(
        task="addition",
        p=97,
        optimizer="gd",
        lr=50.0,
        weight_decay=0.0,
        momentum=0.0,
        hidden_width=hidden_width,
        epochs=S_MAX,
        log_every=100,
        output_dir=output_dir,
    )

    all_results = []

    for alpha in ALPHAS:
        result = run_multi_seed(
            run_fn=run_single_centralized,
            cfg_template=replace(base_cfg, alpha=alpha),
            seeds=SEEDS,
            label=f"alpha={alpha}",
        )
        result["alpha"] = alpha
        all_results.append(result)

    # Determine alpha_crit: smallest alpha where all seeds grok
    alpha_crit = None
    for r in sorted(all_results, key=lambda x: x["alpha"]):
        if r["summary"]["n_grokked"] == len(SEEDS):
            alpha_crit = r["alpha"]
            break

    # T_max: max T_grok across all alpha that grok (finite values only)
    t_max = 0
    for r in all_results:
        for seed_result in r["per_seed"]:
            if seed_result["t_grok"] < float("inf"):
                t_max = max(t_max, int(seed_result["t_grok"]))

    print(f"\n{'='*60}")
    print(f"EXPERIMENT 1 RESULTS")
    print(f"  alpha_crit = {alpha_crit}")
    print(f"  T_max = {t_max}")
    print(f"{'='*60}")

    output = {
        "alpha_crit": alpha_crit,
        "t_max": t_max,
        "per_alpha": all_results,
    }
    save_experiment_results(output, f"{output_dir}/exp1_results.json")
    return output


if __name__ == "__main__":
    run_exp1()
```

- [ ] **Step 2: Commit**

```bash
git add experiments/exp1_boundary.py
git commit -m "feat: implement Experiment 1 — centralized phase boundary"
```

---

### Task 10: Experiment 2 — Aggregation Effect & FL Boundary

**Files:**
- Create: `experiments/exp2_aggregation.py`

- [ ] **Step 1: Implement Exp 2**

```python
# experiments/exp2_aggregation.py
"""Experiment 2: Aggregation effect & FL phase boundary.

Three conditions per (alpha, K):
  (a) Centralized-full: standard centralized with n_train
  (b) Centralized-reduced: centralized with n_train/K (one client's worth)
  (c) FL (IID): K clients, FedAvg, full participation

Design from experiment_plan.md Section 4, Experiment 2.
"""

from dataclasses import replace
from core.config import Config
from federated.config import FedConfig
from experiments.runner import (
    run_single_centralized, run_single_federated,
    run_multi_seed, save_experiment_results, RunConfig,
)

# Default grids — override with values from Exp 1
DEFAULT_ALPHAS = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
K_VALUES = [2, 5, 10, 20, 50, 97]
SEEDS = [42, 123, 456]
LOCAL_EPOCHS = 5


def run_exp2(alphas: list = None, hidden_width: int = 256,
             t_max: int = 30000, output_dir: str = "results/exp2_aggregation"):
    """Run aggregation effect experiment.

    Args:
        alphas: Training fractions to test (from Exp 1 boundary analysis).
        hidden_width: N* from Exp 0.
        t_max: T_max from Exp 1 (for computing S_FL).
    """
    if alphas is None:
        alphas = DEFAULT_ALPHAS

    rc = RunConfig(t_base=8000, t_max=t_max)
    s_fl = rc.s_fl
    num_rounds = s_fl // LOCAL_EPOCHS

    all_results = []

    # (a) Centralized-full: K-independent — run once per alpha, cache results
    cent_full_cache = {}
    for alpha in alphas:
        cent_full_cfg = Config(
            task="addition", p=97, optimizer="gd", lr=50.0,
            weight_decay=0.0, momentum=0.0, hidden_width=hidden_width,
            epochs=100_000, log_every=100,
            alpha=alpha, output_dir=f"{output_dir}/cent_full",
        )
        cent_full_cache[alpha] = run_multi_seed(
            run_fn=run_single_centralized,
            cfg_template=cent_full_cfg,
            seeds=SEEDS,
            label=f"cent_full a={alpha}",
        )

    for alpha in alphas:
        for K in K_VALUES:
            print(f"\n{'='*70}")
            print(f"Exp 2: alpha={alpha}, K={K}")
            print(f"{'='*70}")

            result_a = cent_full_cache[alpha]  # reuse cached centralized-full

            # (b) Centralized-reduced: alpha_eff = alpha / K
            alpha_eff = alpha / K
            cent_red_cfg = Config(
                task="addition", p=97, optimizer="gd", lr=50.0,
                weight_decay=0.0, momentum=0.0, hidden_width=hidden_width,
                epochs=100_000, log_every=100,
                alpha=alpha_eff, output_dir=f"{output_dir}/cent_reduced",
            )
            result_b = run_multi_seed(
                run_fn=run_single_centralized,
                cfg_template=cent_red_cfg,
                seeds=SEEDS,
                label=f"cent_reduced a={alpha}/K={K} (eff={alpha_eff:.4f})",
            )

            # (c) FL (IID)
            fed_cfg = FedConfig(
                task="addition", p=97, optimizer="gd", lr=50.0,
                weight_decay=0.0, momentum=0.0, hidden_width=hidden_width,
                alpha=alpha, num_clients=K, num_rounds=num_rounds,
                local_epochs=LOCAL_EPOCHS, fraction_train=1.0,
                partition="iid", strategy="fedavg",
                output_dir=f"{output_dir}/fl_iid",
            )
            result_c = run_multi_seed(
                run_fn=run_single_federated,
                cfg_template=fed_cfg,
                seeds=SEEDS,
                label=f"fl_iid a={alpha} K={K}",
            )

            all_results.append({
                "alpha": alpha,
                "K": K,
                "cent_full": result_a,
                "cent_reduced": result_b,
                "fl_iid": result_c,
            })

    save_experiment_results(all_results, f"{output_dir}/exp2_results.json")
    return all_results


if __name__ == "__main__":
    run_exp2()
```

- [ ] **Step 2: Commit**

```bash
git add experiments/exp2_aggregation.py
git commit -m "feat: implement Experiment 2 — aggregation effect and FL boundary"
```

---

### Task 11: Experiment 3 — Heterogeneity

**Files:**
- Create: `experiments/exp3_heterogeneity.py`

- [ ] **Step 1: Implement Exp 3**

```python
# experiments/exp3_heterogeneity.py
"""Experiment 3: Heterogeneity at the phase boundary.

Sub-experiment 3a: Dirichlet sweep (continuous heterogeneity knob)
Sub-experiment 3b: Structured partition (operand vs target vs IID)

Design from experiment_plan.md Section 4, Experiment 3.
"""

from dataclasses import replace
from federated.config import FedConfig
from experiments.runner import (
    run_single_federated, run_multi_seed,
    save_experiment_results, RunConfig,
)

SEEDS = [42, 123, 456]
DIRICHLET_ALPHAS = [0.01, 0.1, 0.5, 1.0, 10.0, 1000.0]
STRUCTURED_PARTITIONS = ["iid", "operand", "target"]
LOCAL_EPOCHS = 5


def run_exp3a(alphas: list, k_primary: int = 10,
              hidden_width: int = 256, t_max: int = 30000,
              output_dir: str = "results/exp3_heterogeneity"):
    """3a: Dirichlet sweep at K_primary."""
    rc = RunConfig(t_base=8000, t_max=t_max)
    s_fl = rc.s_fl
    num_rounds = s_fl // LOCAL_EPOCHS
    results = []

    for alpha in alphas:
        for dir_alpha in DIRICHLET_ALPHAS:
            cfg = FedConfig(
                task="addition", p=97, optimizer="gd", lr=50.0,
                weight_decay=0.0, momentum=0.0, hidden_width=hidden_width,
                alpha=alpha, num_clients=k_primary, num_rounds=num_rounds,
                local_epochs=LOCAL_EPOCHS, fraction_train=1.0,
                partition="dirichlet", dirichlet_alpha=dir_alpha,
                strategy="fedavg",
                output_dir=f"{output_dir}/exp3a",
            )
            result = run_multi_seed(
                run_fn=run_single_federated,
                cfg_template=cfg,
                seeds=SEEDS,
                label=f"3a a={alpha} dir={dir_alpha} K={k_primary}",
            )
            result["alpha"] = alpha
            result["dirichlet_alpha"] = dir_alpha
            result["K"] = k_primary
            results.append(result)

    save_experiment_results(results, f"{output_dir}/exp3a_results.json")
    return results


def run_exp3a_k_validation(alphas: list, k_secondary: int = 50,
                           hidden_width: int = 256, t_max: int = 30000,
                           output_dir: str = "results/exp3_heterogeneity"):
    """3a K-validation: reduced sweep at K_secondary."""
    rc = RunConfig(t_base=8000, t_max=t_max)
    reduced_alphas = alphas[:3]  # boundary + comfortable
    reduced_dir = [0.1, 1.0, 1000.0]
    num_rounds = rc.s_fl // LOCAL_EPOCHS
    results = []

    for alpha in reduced_alphas:
        for dir_alpha in reduced_dir:
            cfg = FedConfig(
                task="addition", p=97, optimizer="gd", lr=50.0,
                weight_decay=0.0, momentum=0.0, hidden_width=hidden_width,
                alpha=alpha, num_clients=k_secondary, num_rounds=num_rounds,
                local_epochs=LOCAL_EPOCHS, fraction_train=1.0,
                partition="dirichlet", dirichlet_alpha=dir_alpha,
                strategy="fedavg",
                output_dir=f"{output_dir}/exp3a_kval",
            )
            result = run_multi_seed(
                run_fn=run_single_federated,
                cfg_template=cfg,
                seeds=SEEDS,
                label=f"3a-kval a={alpha} dir={dir_alpha} K={k_secondary}",
            )
            result["alpha"] = alpha
            result["dirichlet_alpha"] = dir_alpha
            result["K"] = k_secondary
            results.append(result)

    save_experiment_results(results, f"{output_dir}/exp3a_kval_results.json")
    return results


def run_exp3b(alphas: list, k_primary: int = 10,
              hidden_width: int = 256, t_max: int = 30000,
              output_dir: str = "results/exp3_heterogeneity"):
    """3b: Structured partition comparison (IID vs operand vs target)."""
    rc = RunConfig(t_base=8000, t_max=t_max)
    num_rounds = rc.s_fl // LOCAL_EPOCHS
    results = []

    for alpha in alphas:
        for partition in STRUCTURED_PARTITIONS:
            cfg = FedConfig(
                task="addition", p=97, optimizer="gd", lr=50.0,
                weight_decay=0.0, momentum=0.0, hidden_width=hidden_width,
                alpha=alpha, num_clients=k_primary, num_rounds=num_rounds,
                local_epochs=LOCAL_EPOCHS, fraction_train=1.0,
                partition=partition,
                strategy="fedavg",
                output_dir=f"{output_dir}/exp3b",
            )
            result = run_multi_seed(
                run_fn=run_single_federated,
                cfg_template=cfg,
                seeds=SEEDS,
                label=f"3b a={alpha} part={partition} K={k_primary}",
            )
            result["alpha"] = alpha
            result["partition"] = partition
            result["K"] = k_primary
            results.append(result)

    save_experiment_results(results, f"{output_dir}/exp3b_results.json")
    return results


if __name__ == "__main__":
    # Use default alpha grid; replace with Exp 1 results
    alphas = [0.1, 0.15, 0.2, 0.3, 0.5]
    run_exp3a(alphas)
    run_exp3a_k_validation(alphas)
    run_exp3b(alphas)
```

- [ ] **Step 2: Commit**

```bash
git add experiments/exp3_heterogeneity.py
git commit -m "feat: implement Experiment 3 — heterogeneity at phase boundary"
```

---

### Task 12: Experiment 4 — Optimization Fragmentation

**Files:**
- Create: `experiments/exp4_optimization.py`

- [ ] **Step 1: Implement Exp 4a/b/c**

```python
# experiments/exp4_optimization.py
"""Experiment 4: Optimization fragmentation.

4a: Drift accumulation x heterogeneity (vary E, fixed S)
4b: Partial participation x heterogeneity (vary f, fixed E)
4c: Compute vs communication budget (fixed R, vary E)

Design from experiment_plan.md Section 4, Experiment 4.
"""

from dataclasses import replace
from federated.config import FedConfig
from experiments.runner import (
    run_single_federated, run_multi_seed,
    save_experiment_results, RunConfig,
)

SEEDS = [42, 123, 456]


def run_exp4a(alphas: list, k: int = 10,
              noniid_setting: dict = None,
              hidden_width: int = 256, t_max: int = 30000,
              output_dir: str = "results/exp4_optimization"):
    """4a: Drift accumulation x heterogeneity. Fixed total S, vary E."""
    rc = RunConfig(t_base=8000, t_max=t_max)
    s_fl = rc.s_fl
    e_values = [5, 10, 25, 50]

    # Default non-IID: mildest setting that showed delay in Exp 3
    if noniid_setting is None:
        noniid_setting = {"partition": "dirichlet", "dirichlet_alpha": 1.0}

    results = []

    for alpha in alphas:
        for E in e_values:
            num_rounds = s_fl // E
            for het_label, het_cfg in [("iid", {"partition": "iid"}),
                                        ("noniid", noniid_setting)]:
                cfg = FedConfig(
                    task="addition", p=97, optimizer="gd", lr=50.0,
                    weight_decay=0.0, momentum=0.0, hidden_width=hidden_width,
                    alpha=alpha, num_clients=k, num_rounds=num_rounds,
                    local_epochs=E, fraction_train=1.0,
                    strategy="fedavg",
                    output_dir=f"{output_dir}/exp4a",
                    **het_cfg,
                )
                result = run_multi_seed(
                    run_fn=run_single_federated,
                    cfg_template=cfg,
                    seeds=SEEDS,
                    label=f"4a a={alpha} E={E} {het_label}",
                )
                result["alpha"] = alpha
                result["E"] = E
                result["heterogeneity"] = het_label
                results.append(result)

    save_experiment_results(results, f"{output_dir}/exp4a_results.json")
    return results


def run_exp4b(alphas: list, k: int = 10,
              noniid_setting: dict = None,
              hidden_width: int = 256, t_max: int = 30000,
              output_dir: str = "results/exp4_optimization"):
    """4b: Partial participation x heterogeneity. Fixed E=5, vary f."""
    rc = RunConfig(t_base=8000, t_max=t_max)
    s_fl = rc.s_fl
    E = 5
    num_rounds = s_fl // E
    f_values = [0.2, 0.4, 0.6, 1.0]

    if noniid_setting is None:
        noniid_setting = {"partition": "dirichlet", "dirichlet_alpha": 1.0}

    results = []

    for alpha in alphas:
        for f in f_values:
            for het_label, het_cfg in [("iid", {"partition": "iid"}),
                                        ("noniid", noniid_setting)]:
                cfg = FedConfig(
                    task="addition", p=97, optimizer="gd", lr=50.0,
                    weight_decay=0.0, momentum=0.0, hidden_width=hidden_width,
                    alpha=alpha, num_clients=k, num_rounds=num_rounds,
                    local_epochs=E, fraction_train=f,
                    strategy="fedavg",
                    output_dir=f"{output_dir}/exp4b",
                    **het_cfg,
                )
                result = run_multi_seed(
                    run_fn=run_single_federated,
                    cfg_template=cfg,
                    seeds=SEEDS,
                    label=f"4b a={alpha} f={f} {het_label}",
                )
                result["alpha"] = alpha
                result["f"] = f
                result["heterogeneity"] = het_label
                results.append(result)

    save_experiment_results(results, f"{output_dir}/exp4b_results.json")
    return results


def run_exp4c(alphas: list, k: int = 10,
              hidden_width: int = 256,
              output_dir: str = "results/exp4_optimization"):
    """4c: Compute vs communication. Fixed R=2000, IID, vary E."""
    R = 2000
    e_values = [1, 5, 10, 25]
    results = []

    for alpha in alphas:
        for E in e_values:
            cfg = FedConfig(
                task="addition", p=97, optimizer="gd", lr=50.0,
                weight_decay=0.0, momentum=0.0, hidden_width=hidden_width,
                alpha=alpha, num_clients=k, num_rounds=R,
                local_epochs=E, fraction_train=1.0,
                partition="iid", strategy="fedavg",
                output_dir=f"{output_dir}/exp4c",
            )
            result = run_multi_seed(
                run_fn=run_single_federated,
                cfg_template=cfg,
                seeds=SEEDS,
                label=f"4c a={alpha} E={E} R={R} S={R*E}",
            )
            result["alpha"] = alpha
            result["E"] = E
            result["R"] = R
            result["S"] = R * E
            results.append(result)

    save_experiment_results(results, f"{output_dir}/exp4c_results.json")
    return results


if __name__ == "__main__":
    alphas = [0.2, 0.3, 0.5]  # Replace with Exp 1 boundary values
    run_exp4a(alphas)
    run_exp4b(alphas)
    run_exp4c(alphas)
```

- [ ] **Step 2: Commit**

```bash
git add experiments/exp4_optimization.py
git commit -m "feat: implement Experiment 4 — optimization fragmentation (4a/b/c)"
```

---

### Task 13: Experiment 5 — Algorithm Comparison & Rescue

**Files:**
- Create: `experiments/exp5_algorithms.py`

- [ ] **Step 1: Implement Exp 5**

```python
# experiments/exp5_algorithms.py
"""Experiment 5: Algorithm comparison & rescue.

For 2-3 'hard' settings from Exps 2-4 where FedAvg fails or is delayed,
test: FedAvg, FedProx (mu sweep), FedAdam (server_lr sweep), FedAvg+WD.

Design from experiment_plan.md Section 4, Experiment 5.
"""

from dataclasses import replace
from federated.config import FedConfig
from experiments.runner import (
    run_single_federated, run_multi_seed,
    save_experiment_results, RunConfig,
)

SEEDS = [42, 123, 456]

# Algorithm configurations to test
ALGORITHMS = [
    # (label, strategy, extra_kwargs)
    ("FedAvg", "fedavg", {}),
    ("FedProx-0.001", "fedprox", {"proximal_mu": 0.001}),
    ("FedProx-0.01", "fedprox", {"proximal_mu": 0.01}),
    ("FedProx-0.1", "fedprox", {"proximal_mu": 0.1}),
    ("FedProx-1.0", "fedprox", {"proximal_mu": 1.0}),
    ("FedAdam-0.01", "fedadam", {"server_lr": 0.01, "tau": 1e-3}),
    ("FedAdam-0.1", "fedadam", {"server_lr": 0.1, "tau": 1e-3}),
    ("FedAdam-1.0", "fedadam", {"server_lr": 1.0, "tau": 1e-3}),
    ("FedAvg+WD-0.01", "fedavg", {"weight_decay": 0.01}),
    ("FedAvg+WD-0.1", "fedavg", {"weight_decay": 0.1}),
    ("FedAvg+WD-1.0", "fedavg", {"weight_decay": 1.0}),
]


def run_exp5(hard_settings: list, hidden_width: int = 256,
             t_max: int = 30000,
             output_dir: str = "results/exp5_algorithms"):
    """Run algorithm rescue experiment.

    Args:
        hard_settings: List of dicts, each describing a hard FL setting:
            {"label": "H1", "alpha": 0.15, "K": 20, "partition": "iid",
             "local_epochs": 5, "fraction_train": 1.0,
             "dirichlet_alpha": 0.5}
    """
    rc = RunConfig(t_base=8000, t_max=t_max)
    s_rescue = rc.s_rescue
    results = []

    for setting in hard_settings:
        for algo_label, strategy, algo_kwargs in ALGORITHMS:
            E = setting.get("local_epochs", 5)
            num_rounds = s_rescue // E

            cfg = FedConfig(
                task="addition", p=97, optimizer="gd", lr=50.0,
                weight_decay=algo_kwargs.get("weight_decay", 0.0),
                momentum=0.0, hidden_width=hidden_width,
                alpha=setting["alpha"],
                num_clients=setting["K"],
                num_rounds=num_rounds,
                local_epochs=E,
                fraction_train=setting.get("fraction_train", 1.0),
                partition=setting.get("partition", "iid"),
                dirichlet_alpha=setting.get("dirichlet_alpha", 0.5),
                strategy=strategy,
                proximal_mu=algo_kwargs.get("proximal_mu", 0.0),
                server_lr=algo_kwargs.get("server_lr", 1.0),
                tau=algo_kwargs.get("tau", 1e-3),
                output_dir=f"{output_dir}/{setting['label']}",
            )
            result = run_multi_seed(
                run_fn=run_single_federated,
                cfg_template=cfg,
                seeds=SEEDS,
                label=f"5 {setting['label']} {algo_label}",
            )
            result["setting"] = setting["label"]
            result["algorithm"] = algo_label
            results.append(result)

    save_experiment_results(results, f"{output_dir}/exp5_results.json")
    return results


if __name__ == "__main__":
    # Placeholder hard settings — replace with actual failures from Exps 2-4
    hard_settings = [
        {"label": "H1", "alpha": 0.15, "K": 20, "partition": "iid",
         "local_epochs": 5, "fraction_train": 1.0},
        {"label": "H2", "alpha": 0.15, "K": 10, "partition": "dirichlet",
         "local_epochs": 5, "fraction_train": 1.0, "dirichlet_alpha": 0.1},
        {"label": "H3", "alpha": 0.3, "K": 10, "partition": "iid",
         "local_epochs": 25, "fraction_train": 0.4},
    ]
    run_exp5(hard_settings)
```

- [ ] **Step 2: Commit**

```bash
git add experiments/exp5_algorithms.py
git commit -m "feat: implement Experiment 5 — algorithm comparison and rescue"
```

---

### Task 14: Experiment 6 — Mechanistic Analysis (Post-hoc)

**Files:**
- Create: `experiments/exp6_mechanistic.py`

- [ ] **Step 1: Implement Exp 6**

```python
# experiments/exp6_mechanistic.py
"""Experiment 6: Mechanistic analysis (post-hoc on runs from Exps 1-5).

No new training runs. Selects representative runs and produces:
- IPR trajectories
- Client drift trajectories
- Fourier spectra at key timepoints
- Cross-run scatter: mean drift vs grokking step

Design from experiment_plan.md Section 4, Experiment 6.
"""

import json
import os
import numpy as np


def load_history(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def select_representative_runs(exp_results_dir: str) -> dict:
    """Select 4 representative runs for mechanistic analysis.

    Returns dict mapping category to history file path:
      - healthy_grok: alpha=0.5, IID, K=5
      - boundary_grok: alpha near alpha_crit, IID, K=10
      - failed_grok: alpha < alpha_crit or hard non-IID
      - rescued_grok: failed setting recovered by algorithm (from Exp 5)
    """
    # These paths will be populated from actual experiment results
    return {
        "healthy_grok": None,
        "boundary_grok": None,
        "failed_grok": None,
        "rescued_grok": None,
    }


def analyze_drift_vs_grokking(results_dirs: list, output_dir: str):
    """Cross-run scatter plot data: mean client drift vs grokking step.

    Scans all FL history files for mean_client_drift and computes T_grok.
    """
    from experiments.grokking_metrics import compute_t_grok
    import glob

    data_points = []
    for d in results_dirs:
        for path in glob.glob(os.path.join(d, "**", "history_fed_*.json"), recursive=True):
            h = load_history(path)
            if "mean_client_drift" not in h or not h["mean_client_drift"]:
                continue
            steps = h.get("total_steps", [])
            test_accs = h.get("test_acc", [])
            t_grok = compute_t_grok(steps, test_accs)
            mean_drift = float(np.mean(h["mean_client_drift"]))
            data_points.append({
                "path": path,
                "t_grok": t_grok,
                "mean_drift": mean_drift,
            })

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "drift_vs_grokking.json"), "w") as f:
        json.dump(data_points, f, indent=2, default=str)

    print(f"Collected {len(data_points)} data points for drift vs grokking analysis")
    return data_points


if __name__ == "__main__":
    analyze_drift_vs_grokking(
        results_dirs=["results/exp2_aggregation", "results/exp3_heterogeneity",
                       "results/exp4_optimization", "results/exp5_algorithms"],
        output_dir="results/exp6_mechanistic",
    )
```

- [ ] **Step 2: Commit**

```bash
git add experiments/exp6_mechanistic.py
git commit -m "feat: implement Experiment 6 — mechanistic analysis framework"
```

---

## Phase 6: Publication Visualization

*Depends on experiment output format being stable (Phases 1-5).*

### Task 15: Publication-Quality Figures

**Files:**
- Create: `experiments/visualization.py`

- [ ] **Step 1: Implement visualization module**

```python
# experiments/visualization.py
"""Publication-quality figures for all experiments.

Produces Figures 1-6 as described in experiment_plan.md Section 9.
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# Publication style
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox_inches": "tight",
})


def _load(path):
    with open(path) as f:
        return json.load(f)


def _parse_inf(val):
    """Convert 'inf' strings back to float('inf')."""
    if val == "inf":
        return float("inf")
    return float(val)


# ── Figure 1: Grokking step vs alpha (Exp 1) ─────────────────────────────

def plot_exp1_phase_boundary(results_path: str, output_dir: str):
    """Plot T_grok vs alpha with error bars. Sharp transition expected."""
    data = _load(results_path)
    alphas = []
    t_groks_mean = []
    t_groks_std = []

    for entry in data["per_alpha"]:
        alpha = entry["alpha"]
        summary = entry["summary"]
        t_mean = _parse_inf(summary["t_grok_mean"])
        t_std = _parse_inf(summary["t_grok_std"])
        alphas.append(alpha)
        t_groks_mean.append(t_mean if t_mean < float("inf") else None)
        t_groks_std.append(t_std if t_std < float("inf") else 0)

    fig, ax = plt.subplots(figsize=(7, 5))
    finite_mask = [t is not None for t in t_groks_mean]
    inf_mask = [t is None for t in t_groks_mean]

    finite_a = [a for a, m in zip(alphas, finite_mask) if m]
    finite_t = [t for t in t_groks_mean if t is not None]
    finite_s = [s for s, m in zip(t_groks_std, finite_mask) if m]

    ax.errorbar(finite_a, finite_t, yerr=finite_s, fmt="o-", capsize=4,
                color="tab:blue", label="T_grok (3 seeds)")

    # Mark non-grokking alphas
    inf_a = [a for a, m in zip(alphas, inf_mask) if m]
    if inf_a:
        ax.scatter(inf_a, [max(finite_t) * 1.2] * len(inf_a), marker="x",
                   color="red", s=100, zorder=5, label="No grokking")

    ax.set_xlabel("Training fraction (alpha)")
    ax.set_ylabel("Grokking step (T_grok)")
    ax.set_title("Centralized Phase Boundary")
    ax.legend()
    ax.set_xscale("log")

    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, "fig_exp1_boundary.png"))
    plt.close(fig)


# ── Figure 2: Aggregation effect (Exp 2) ────────────────────────────────

def plot_exp2_aggregation(results_path: str, output_dir: str):
    """For each alpha, plot T_grok vs K: centralized-full / reduced / FL."""
    data = _load(results_path)

    # Group by alpha
    by_alpha = {}
    for entry in data:
        a = entry["alpha"]
        if a not in by_alpha:
            by_alpha[a] = []
        by_alpha[a].append(entry)

    n_alphas = len(by_alpha)
    fig, axes = plt.subplots(1, min(n_alphas, 4), figsize=(5 * min(n_alphas, 4), 5),
                              squeeze=False)

    for idx, (alpha, entries) in enumerate(sorted(by_alpha.items())):
        if idx >= 4:
            break
        ax = axes[0, idx]
        ks = sorted(set(e["K"] for e in entries))

        for condition, color, marker in [
            ("cent_full", "tab:green", "s"),
            ("cent_reduced", "tab:orange", "^"),
            ("fl_iid", "tab:blue", "o"),
        ]:
            t_vals = []
            for k in ks:
                match = [e for e in entries if e["K"] == k]
                if match:
                    t = _parse_inf(match[0][condition]["summary"]["t_grok_mean"])
                    t_vals.append(t if t < float("inf") else None)
                else:
                    t_vals.append(None)
            finite_k = [k for k, t in zip(ks, t_vals) if t is not None]
            finite_t = [t for t in t_vals if t is not None]
            ax.plot(finite_k, finite_t, f"{marker}-", label=condition.replace("_", " "),
                    color=color)

        ax.set_xlabel("K (clients)")
        ax.set_ylabel("T_grok")
        ax.set_title(f"alpha = {alpha}")
        ax.legend(fontsize=8)
        ax.set_xscale("log")

    fig.suptitle("Aggregation Effect: Centralized vs FL", fontsize=14)
    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, "fig_exp2_aggregation.png"))
    plt.close(fig)


# ── Figure 3a: Phase diagram heatmap (Exp 3) ────────────────────────────

def plot_exp3a_phase_diagram(results_path: str, output_dir: str):
    """Heatmap: x=Dirichlet alpha (log), y=training fraction, color=T_grok."""
    data = _load(results_path)

    alphas = sorted(set(r["alpha"] for r in data))
    dir_alphas = sorted(set(r["dirichlet_alpha"] for r in data))

    grid = np.full((len(alphas), len(dir_alphas)), np.nan)
    for r in data:
        i = alphas.index(r["alpha"])
        j = dir_alphas.index(r["dirichlet_alpha"])
        t = _parse_inf(r["summary"]["t_grok_mean"])
        grid[i, j] = t if t < float("inf") else np.nan

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(grid, aspect="auto", origin="lower",
                   extent=[0, len(dir_alphas), 0, len(alphas)],
                   cmap="viridis_r")
    ax.set_xticks(np.arange(len(dir_alphas)) + 0.5)
    ax.set_xticklabels([str(d) for d in dir_alphas])
    ax.set_yticks(np.arange(len(alphas)) + 0.5)
    ax.set_yticklabels([str(a) for a in alphas])
    ax.set_xlabel("Dirichlet alpha (heterogeneity)")
    ax.set_ylabel("Training fraction (alpha)")
    ax.set_title("Grokking Phase Diagram")
    fig.colorbar(im, ax=ax, label="T_grok")

    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, "fig_exp3a_phase_diagram.png"))
    plt.close(fig)


# ── Figure 4: Communication efficiency (Exp 4) ──────────────────────────

def plot_exp4a_drift(results_path: str, output_dir: str):
    """T_grok vs E, two lines (IID vs non-IID), per alpha panel."""
    data = _load(results_path)

    alphas = sorted(set(r["alpha"] for r in data))
    fig, axes = plt.subplots(1, len(alphas), figsize=(5 * len(alphas), 5), squeeze=False)

    for idx, alpha in enumerate(alphas):
        ax = axes[0, idx]
        for het, color in [("iid", "tab:blue"), ("noniid", "tab:red")]:
            entries = [r for r in data if r["alpha"] == alpha and r["heterogeneity"] == het]
            entries.sort(key=lambda r: r["E"])
            es = [r["E"] for r in entries]
            ts = [_parse_inf(r["summary"]["t_grok_mean"]) for r in entries]
            finite_e = [e for e, t in zip(es, ts) if t < float("inf")]
            finite_t = [t for t in ts if t < float("inf")]
            ax.plot(finite_e, finite_t, "o-", label=het, color=color)

        ax.set_xlabel("Local epochs (E)")
        ax.set_ylabel("T_grok")
        ax.set_title(f"alpha = {alpha}")
        ax.legend()

    fig.suptitle("Drift Accumulation x Heterogeneity", fontsize=14)
    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, "fig_exp4a_drift.png"))
    plt.close(fig)


# ── Figure 5: Algorithm rescue (Exp 5) ──────────────────────────────────

def plot_exp5_algorithms(results_path: str, output_dir: str):
    """Bar chart of T_grok across algorithms for each hard setting."""
    data = _load(results_path)

    settings = sorted(set(r["setting"] for r in data))
    fig, axes = plt.subplots(1, len(settings), figsize=(6 * len(settings), 5), squeeze=False)

    for idx, setting in enumerate(settings):
        ax = axes[0, idx]
        entries = [r for r in data if r["setting"] == setting]
        labels = [r["algorithm"] for r in entries]
        ts = [_parse_inf(r["summary"]["t_grok_mean"]) for r in entries]

        colors = []
        for t in ts:
            if t == float("inf"):
                colors.append("lightgray")
            else:
                colors.append("tab:blue")

        # Replace inf with a visible cap for plotting
        max_finite = max((t for t in ts if t < float("inf")), default=0)
        plot_ts = [t if t < float("inf") else max_finite * 1.3 for t in ts]

        bars = ax.bar(range(len(labels)), plot_ts, color=colors)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("T_grok")
        ax.set_title(f"Setting: {setting}")

    fig.suptitle("Algorithm Comparison", fontsize=14)
    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, "fig_exp5_algorithms.png"))
    plt.close(fig)


# ── Figure 6: Mechanistic (Exp 6) ───────────────────────────────────────

def plot_exp6_drift_scatter(drift_data_path: str, output_dir: str):
    """Scatter: mean client drift vs T_grok across all FL runs."""
    data = _load(drift_data_path)

    drifts = [d["mean_drift"] for d in data]
    t_groks = [_parse_inf(d["t_grok"]) for d in data]

    finite_mask = [t < float("inf") for t in t_groks]
    inf_mask = [not m for m in finite_mask]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(
        [d for d, m in zip(drifts, finite_mask) if m],
        [t for t, m in zip(t_groks, finite_mask) if m],
        alpha=0.6, label="Grokked", color="tab:blue"
    )
    if any(inf_mask):
        ax.scatter(
            [d for d, m in zip(drifts, inf_mask) if m],
            [max(t for t in t_groks if t < float("inf")) * 1.2] * sum(inf_mask),
            marker="x", color="red", s=60, label="No grokking"
        )

    ax.set_xlabel("Mean client drift")
    ax.set_ylabel("T_grok")
    ax.set_title("Client Drift vs Grokking Step")
    ax.legend()

    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, "fig_exp6_drift_scatter.png"))
    plt.close(fig)
```

- [ ] **Step 2: Commit**

```bash
git add experiments/visualization.py
git commit -m "feat: add publication-quality visualization for all experiments"
```

---

## Phase 7: Unified CLI Entry Point

### Task 16: run_experiment.py CLI

**Files:**
- Create: `run_experiment.py`

- [ ] **Step 1: Implement unified CLI**

```python
# run_experiment.py
"""Unified CLI for running all experiments.

Usage:
    python run_experiment.py exp0
    python run_experiment.py exp1 --hidden_width 256
    python run_experiment.py exp2 --t_max 30000
    python run_experiment.py exp3a --alphas 0.1,0.15,0.2,0.3,0.5
    python run_experiment.py exp4a
    python run_experiment.py exp5 --hard_settings hard_settings.json
    python run_experiment.py exp6
    python run_experiment.py plot --exp exp1 --results results/exp1_boundary/exp1_results.json
"""

import argparse
import json
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="Run grokking experiments")
    subparsers = parser.add_subparsers(dest="experiment", help="Experiment to run")

    # Exp 0
    p0 = subparsers.add_parser("exp0", help="Width validation")
    p0.add_argument("--output_dir", default="results/exp0_width")

    # Exp 1
    p1 = subparsers.add_parser("exp1", help="Centralized phase boundary")
    p1.add_argument("--hidden_width", type=int, default=256)
    p1.add_argument("--output_dir", default="results/exp1_boundary")

    # Exp 2
    p2 = subparsers.add_parser("exp2", help="Aggregation effect")
    p2.add_argument("--alphas", type=str, default=None, help="Comma-separated alphas")
    p2.add_argument("--hidden_width", type=int, default=256)
    p2.add_argument("--t_max", type=int, default=30000)
    p2.add_argument("--output_dir", default="results/exp2_aggregation")

    # Exp 3
    for sub in ["exp3a", "exp3a_kval", "exp3b"]:
        p = subparsers.add_parser(sub, help=f"Heterogeneity: {sub}")
        p.add_argument("--alphas", type=str, default="0.1,0.15,0.2,0.3,0.5")
        p.add_argument("--k", type=int, default=10)
        p.add_argument("--hidden_width", type=int, default=256)
        p.add_argument("--t_max", type=int, default=30000)
        p.add_argument("--output_dir", default="results/exp3_heterogeneity")

    # Exp 4
    for sub in ["exp4a", "exp4b", "exp4c"]:
        p = subparsers.add_parser(sub, help=f"Optimization: {sub}")
        p.add_argument("--alphas", type=str, default="0.2,0.3,0.5")
        p.add_argument("--k", type=int, default=10)
        p.add_argument("--hidden_width", type=int, default=256)
        p.add_argument("--t_max", type=int, default=30000)
        p.add_argument("--output_dir", default="results/exp4_optimization")

    # Exp 5
    p5 = subparsers.add_parser("exp5", help="Algorithm rescue")
    p5.add_argument("--hard_settings", type=str, required=True,
                    help="JSON file with hard settings list")
    p5.add_argument("--hidden_width", type=int, default=256)
    p5.add_argument("--t_max", type=int, default=30000)
    p5.add_argument("--output_dir", default="results/exp5_algorithms")

    # Exp 6
    p6 = subparsers.add_parser("exp6", help="Mechanistic analysis")
    p6.add_argument("--output_dir", default="results/exp6_mechanistic")

    # Plotting
    pp = subparsers.add_parser("plot", help="Generate figures")
    pp.add_argument("--exp", required=True,
                    choices=["exp1", "exp2", "exp3a", "exp4a", "exp5", "exp6"])
    pp.add_argument("--results", required=True, help="Path to results JSON")
    pp.add_argument("--output_dir", default="results/figures")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.experiment == "exp0":
        from experiments.exp0_width import run_exp0
        run_exp0(output_dir=args.output_dir)

    elif args.experiment == "exp1":
        from experiments.exp1_boundary import run_exp1
        run_exp1(hidden_width=args.hidden_width, output_dir=args.output_dir)

    elif args.experiment == "exp2":
        from experiments.exp2_aggregation import run_exp2
        alphas = [float(a) for a in args.alphas.split(",")] if args.alphas else None
        run_exp2(alphas=alphas, hidden_width=args.hidden_width,
                 t_max=args.t_max, output_dir=args.output_dir)

    elif args.experiment in ("exp3a", "exp3a_kval", "exp3b"):
        alphas = [float(a) for a in args.alphas.split(",")]
        if args.experiment == "exp3a":
            from experiments.exp3_heterogeneity import run_exp3a
            run_exp3a(alphas=alphas, k_primary=args.k,
                      hidden_width=args.hidden_width, t_max=args.t_max,
                      output_dir=args.output_dir)
        elif args.experiment == "exp3a_kval":
            from experiments.exp3_heterogeneity import run_exp3a_k_validation
            run_exp3a_k_validation(alphas=alphas, k_secondary=args.k,
                                   hidden_width=args.hidden_width, t_max=args.t_max,
                                   output_dir=args.output_dir)
        elif args.experiment == "exp3b":
            from experiments.exp3_heterogeneity import run_exp3b
            run_exp3b(alphas=alphas, k_primary=args.k,
                      hidden_width=args.hidden_width, t_max=args.t_max,
                      output_dir=args.output_dir)

    elif args.experiment in ("exp4a", "exp4b", "exp4c"):
        alphas = [float(a) for a in args.alphas.split(",")]
        if args.experiment == "exp4a":
            from experiments.exp4_optimization import run_exp4a
            run_exp4a(alphas=alphas, k=args.k,
                      hidden_width=args.hidden_width, t_max=args.t_max,
                      output_dir=args.output_dir)
        elif args.experiment == "exp4b":
            from experiments.exp4_optimization import run_exp4b
            run_exp4b(alphas=alphas, k=args.k,
                      hidden_width=args.hidden_width, t_max=args.t_max,
                      output_dir=args.output_dir)
        elif args.experiment == "exp4c":
            from experiments.exp4_optimization import run_exp4c
            run_exp4c(alphas=alphas, k=args.k,
                      hidden_width=args.hidden_width,
                      output_dir=args.output_dir)

    elif args.experiment == "exp5":
        from experiments.exp5_algorithms import run_exp5
        with open(args.hard_settings) as f:
            hard_settings = json.load(f)
        run_exp5(hard_settings=hard_settings, hidden_width=args.hidden_width,
                 t_max=args.t_max, output_dir=args.output_dir)

    elif args.experiment == "exp6":
        from experiments.exp6_mechanistic import analyze_drift_vs_grokking
        analyze_drift_vs_grokking(
            results_dirs=["results/exp2_aggregation", "results/exp3_heterogeneity",
                          "results/exp4_optimization", "results/exp5_algorithms"],
            output_dir=args.output_dir,
        )

    elif args.experiment == "plot":
        from experiments import visualization as viz
        if args.exp == "exp1":
            viz.plot_exp1_phase_boundary(args.results, args.output_dir)
        elif args.exp == "exp2":
            viz.plot_exp2_aggregation(args.results, args.output_dir)
        elif args.exp == "exp3a":
            viz.plot_exp3a_phase_diagram(args.results, args.output_dir)
        elif args.exp == "exp4a":
            viz.plot_exp4a_drift(args.results, args.output_dir)
        elif args.exp == "exp5":
            viz.plot_exp5_algorithms(args.results, args.output_dir)
        elif args.exp == "exp6":
            viz.plot_exp6_drift_scatter(args.results, args.output_dir)

    else:
        print("No experiment specified. Use --help for options.")
        sys.exit(1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add run_experiment.py
git commit -m "feat: add unified run_experiment.py CLI for all experiments"
```

---

## Phase 8: Integration Testing

### Task 17: Smoke Test — End-to-End Validation

**Files:**
- Create: `tests/test_integration.py`

- [ ] **Step 1: Write integration tests (tiny p=7 runs)**

```python
# tests/test_integration.py
"""Integration tests: verify experiment pipeline runs end-to-end with tiny configs."""
import pytest
import os
import json
import tempfile

from core.config import Config
from federated.config import FedConfig
from experiments.runner import run_single_centralized, run_single_federated, run_multi_seed
from experiments.grokking_metrics import extract_grokking_results


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


class TestCentralizedPipeline:
    def test_single_run_produces_metrics(self, tmp_dir):
        cfg = Config(p=7, hidden_width=16, epochs=50, log_every=10,
                     seed=42, output_dir=tmp_dir)
        result = run_single_centralized(cfg, label="test")
        assert "t_grok" in result
        assert "t_50" in result
        assert "final_test_acc" in result

    def test_multi_seed_produces_summary(self, tmp_dir):
        cfg = Config(p=7, hidden_width=16, epochs=50, log_every=10,
                     output_dir=tmp_dir)
        result = run_multi_seed(
            run_fn=run_single_centralized,
            cfg_template=cfg,
            seeds=[42, 123],
            label="test",
        )
        assert result["summary"]["n_seeds"] == 2
        assert "t_grok_mean" in result["summary"]


class TestFederatedPipeline:
    def test_fedavg_run(self, tmp_dir):
        cfg = FedConfig(p=7, hidden_width=16, num_clients=2,
                        num_rounds=3, local_epochs=2,
                        seed=42, output_dir=tmp_dir, strategy="fedavg")
        result = run_single_federated(cfg, label="test")
        assert "t_grok" in result

    def test_fedadam_run(self, tmp_dir):
        cfg = FedConfig(p=7, hidden_width=16, num_clients=2,
                        num_rounds=3, local_epochs=2,
                        seed=42, output_dir=tmp_dir,
                        strategy="fedadam", server_lr=0.1, tau=1e-3)
        result = run_single_federated(cfg, label="test")
        assert "t_grok" in result

    def test_fedprox_run(self, tmp_dir):
        cfg = FedConfig(p=7, hidden_width=16, num_clients=2,
                        num_rounds=3, local_epochs=2,
                        seed=42, output_dir=tmp_dir,
                        strategy="fedprox", proximal_mu=0.1)
        result = run_single_federated(cfg, label="test")
        assert "t_grok" in result
```

- [ ] **Step 2: Run integration tests**

Run: `python -m pytest tests/test_integration.py -v --timeout=120`
Expected: ALL PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: add integration tests for experiment pipeline"
```

---

## Summary: Parallelization Strategy for Agent Teams

The plan has natural parallelism at two levels:

### Level 1: Independent Phases (can be dispatched to parallel agents)

```
Agent A: Phase 1 (Tasks 1-2)  — Grokking metrics + Runner
Agent B: Phase 2 (Tasks 3-4)  — FedConfig + FedAdam strategy
Agent C: Phase 3 (Tasks 5-6)  — Fourier spectrum + Client drift
```

These three agents work on completely independent code paths and can run simultaneously.

### Level 2: After Phase 1-3 merge (sequential dependency)

```
Agent D: Phase 4 (Task 7)     — Centralized enhancements (quick, 5 min)
Agent E: Phase 5 (Tasks 8-14) — Experiment implementations (can be further parallelized: one agent per experiment)
Agent F: Phase 6 (Task 15)    — Visualization
Agent G: Phase 7 (Task 16)    — CLI entry point
```

### Level 3: Integration

```
Agent H: Phase 8 (Task 17)    — Integration testing after all phases merge
```

### Dependency Graph

```
Phase 1 ──┐
Phase 2 ──┼── Phase 4 ── Phase 5 ── Phase 7 ── Phase 8
Phase 3 ──┘              Phase 6 ────────────┘
```

### Estimated Time (with agent parallelism)

| Phase | Tasks | Sequential est. | With agents |
|-------|-------|-----------------|-------------|
| 1-3 (parallel) | 6 | ~60 min | ~20 min |
| 4 | 1 | ~5 min | ~5 min |
| 5 (parallel across exps) | 7 | ~45 min | ~15 min |
| 6-7 (parallel) | 2 | ~20 min | ~10 min |
| 8 | 1 | ~10 min | ~10 min |
| **Total** | **17** | **~140 min** | **~60 min** |
