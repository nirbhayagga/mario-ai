"""
Tests for the pure helpers in src/train.py (run config, LR schedule).
Skipped if stable-baselines3 is not installed (the module imports it at load).
"""

import argparse
import json
import sys

import pytest

train = pytest.importorskip("train")


def test_build_run_config_records_flags_and_provenance(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["src/train.py", "--seed", "3"])
    args = argparse.Namespace(env_id="SuperMarioBros-1-1-v0", seed=3, n_envs=4, learning_rate=2.5e-4)
    config = train.build_run_config(args)
    assert config["env_id"] == "SuperMarioBros-1-1-v0"
    assert config["seed"] == 3
    assert config["learning_rate"] == 2.5e-4
    assert config["command"] == "src/train.py --seed 3"
    assert "created" in config
    assert "git_commit" in config


def test_write_run_config_round_trips(tmp_path):
    path = train.write_run_config(str(tmp_path), {"env_id": "x", "seed": None})
    with open(path) as f:
        assert json.load(f) == {"env_id": "x", "seed": None}


def test_git_commit_is_short_hash_or_none():
    commit = train.git_commit()
    assert commit is None or (len(commit) >= 7 and all(c in "0123456789abcdef" for c in commit))


def test_linear_schedule_decays_to_floor():
    schedule = train.linear_schedule(1e-3)
    assert schedule(1.0) == 1e-3
    assert schedule(0.5) == 5e-4
    assert schedule(0.0) == 1e-7
