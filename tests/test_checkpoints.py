"""
Tests for checkpoint selection in src/visualize_progress.py. Skipped if the
RL stack is not installed (the module imports stable-baselines3 at load).
"""

import pytest

vp = pytest.importorskip("visualize_progress")


def ckpt(steps):
    return f"runs/x/checkpoints/mario_ppo_agent_{steps}_steps.zip"


def test_checkpoint_step_parses_step_count():
    assert vp.checkpoint_step(ckpt(250000)) == 250000
    assert vp.checkpoint_step("runs/x/best_model.zip") == -1


def test_selection_sorts_numerically_not_lexicographically():
    files = [ckpt(1000000), ckpt(200000), ckpt(50000)]
    assert vp.select_checkpoints(files, max_models=4) == [ckpt(50000), ckpt(200000), ckpt(1000000)]


def test_selection_spans_first_to_last_evenly():
    files = [ckpt(n * 100000) for n in range(10, 0, -1)]
    assert vp.select_checkpoints(files, max_models=4) == [
        ckpt(100000),
        ckpt(400000),
        ckpt(700000),
        ckpt(1000000),
    ]


def test_selection_with_fewer_than_max_returns_all_sorted():
    files = [ckpt(300000), ckpt(100000)]
    assert vp.select_checkpoints(files, max_models=4) == [ckpt(100000), ckpt(300000)]


def test_selection_single_file():
    assert vp.select_checkpoints([ckpt(5)], max_models=4) == [ckpt(5)]
