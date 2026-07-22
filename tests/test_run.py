"""Tests for the single-run entry point (fedgrok.run).

Kept to cheap centralized runs (no Ray) so the suite stays fast; the federated
path shares the same code below extract_grokking_results and is covered by the
integration tests.
"""

import json
import os

import pytest

from fedgrok.run import run_spec, result_path
from fedgrok.manifest import run_id


CHEAP = {
    "mode": "centralized", "task": "addition", "p": 7, "alpha": 0.5,
    "seed": 42, "hidden_width": 16, "lr": 1.0, "epochs": 100, "log_every": 50,
}


def test_run_spec_writes_result_and_returns_row(tmp_path):
    row = run_spec(CHEAP, results_root=str(tmp_path / "runs"),
                   histories_root=str(tmp_path / "hist"))

    # Result JSON exists at the resolved path.
    assert os.path.exists(result_path(CHEAP, str(tmp_path / "runs")))

    # Row carries id, outcomes, and config, schema-aligned with runs.csv.
    for key in ("id", "mode", "task", "p", "alpha", "seed", "t_grok",
                "final_acc", "grokked", "censored", "steps_run"):
        assert key in row
    assert row["mode"] == "centralized"
    assert row["id"] == run_id(CHEAP)
    assert row["steps_run"] == 100.0


def test_result_json_is_valid_and_inf_serialised(tmp_path):
    run_spec(CHEAP, results_root=str(tmp_path / "runs"),
             histories_root=str(tmp_path / "hist"))
    with open(result_path(CHEAP, str(tmp_path / "runs"))) as handle:
        data = json.load(handle)
    # A tiny non-grokking run: t_grok must round-trip as the string "inf".
    assert data["t_grok"] == "inf"
    assert data["censored"] is True


def test_result_path_is_stable_without_explicit_id():
    # No "id" in the spec -> result_path must still resolve deterministically.
    p1 = result_path(CHEAP, "results/data/runs")
    p2 = result_path(dict(CHEAP), "results/data/runs")
    assert p1 == p2 and p1.endswith(".json")


def test_tags_do_not_change_the_run_id():
    tagged = dict(CHEAP, tier="T0", group="pilot", experiment="exp2")
    assert run_id(tagged) == run_id(CHEAP)
