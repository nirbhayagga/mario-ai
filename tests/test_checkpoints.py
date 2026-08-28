"""
Tests for checkpoint selection in src/visualize_progress.py. Skipped if the
RL stack is not installed (the module imports stable-baselines3 at load).
"""

import numpy as np
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


def test_gif_frame_halves_resolution_and_converts_to_rgb():
    bgr = np.zeros((480, 512, 3), np.uint8)
    bgr[..., 0] = 255  # pure blue in BGR
    rgb = vp.gif_frame(bgr, scale=0.5)
    assert rgb.shape == (240, 256, 3)
    assert tuple(rgb[0, 0]) == (0, 0, 255)  # blue lands in the R,G,B blue channel


def test_gif_frame_never_collapses_to_zero_size():
    assert vp.gif_frame(np.zeros((2, 2, 3), np.uint8), scale=0.1).shape == (1, 1, 3)
