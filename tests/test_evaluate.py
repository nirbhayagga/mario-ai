"""
Tests for the episode loop and summary in src/evaluate.py, using a scripted
vec-env and model. Skipped if the RL stack is not installed (the module
imports stable-baselines3 and cv2 at load).
"""

import numpy as np
import pytest

evaluate = pytest.importorskip("evaluate")


class ScriptedModel:
    def __init__(self, action=1):
        self.action = action

    def predict(self, obs, deterministic=True):
        return np.array([self.action]), None


class ScriptedVecEnv:
    """Mario advances 10 px per step; raises the flag at flag_step or dies at death_step."""

    def __init__(self, flag_step=None, death_step=None):
        self.flag_step = flag_step
        self.death_step = death_step
        self.reset()

    def reset(self):
        self.t = 0
        self.x = 0
        return np.zeros((1, 84, 84, 4), np.uint8)

    def step(self, action):
        self.t += 1
        self.x += 10
        flag = self.t == self.flag_step
        done = flag or self.t == self.death_step
        info = [{"x_pos": self.x, "flag_get": flag, "score": 0}]
        return np.zeros((1, 84, 84, 4), np.uint8), np.array([1.0]), np.array([done]), info


def test_run_episode_detects_level_clear():
    result = evaluate.run_episode(ScriptedVecEnv(flag_step=5), ScriptedModel(), max_steps=100)
    assert result == {"cleared": True, "reward": 5.0, "steps": 5, "max_x": 50}


def test_run_episode_stops_on_first_death():
    result = evaluate.run_episode(ScriptedVecEnv(death_step=3), ScriptedModel(), max_steps=100)
    assert result["cleared"] is False
    assert result["steps"] == 3
    assert result["max_x"] == 30


def test_run_episode_respects_max_steps():
    result = evaluate.run_episode(ScriptedVecEnv(), ScriptedModel(), max_steps=7)
    assert result["steps"] == 7
    assert result["cleared"] is False


def test_on_step_callback_sees_every_step():
    seen = []
    evaluate.run_episode(
        ScriptedVecEnv(flag_step=4), ScriptedModel(action=2), max_steps=100,
        on_step=lambda info, action, total: seen.append((info["x_pos"], action, total)),
    )
    assert seen == [(10, 2, 1.0), (20, 2, 2.0), (30, 2, 3.0), (40, 2, 4.0)]


def test_summarize_episodes():
    results = [
        {"cleared": True, "reward": 3000.0, "steps": 500, "max_x": 3200},
        {"cleared": False, "reward": 1000.0, "steps": 300, "max_x": 1200},
    ]
    summary = evaluate.summarize_episodes(results)
    assert summary == {
        "episodes": 2,
        "clear_rate": 0.5,
        "mean_reward": 2000.0,
        "mean_max_x": 2200.0,
        "best_max_x": 3200,
        "mean_steps": 400.0,
    }


def test_level_tag_extracts_world_and_stage():
    assert evaluate.level_tag("SuperMarioBros-1-2-v0") == "1-2"
    assert evaluate.level_tag("SuperMarioBros-8-4-v3") == "8-4"
    assert evaluate.level_tag("SuperMarioBros-v0") == "SuperMarioBros-v0"


def test_output_paths_single_level_keeps_plain_names():
    assert evaluate.output_paths("runs/x/evaluation_run.mp4", "SuperMarioBros-1-1-v0", multi_level=False) == (
        "runs/x/evaluation_run.mp4", "runs/x/evaluation_run.gif",
    )


def test_output_paths_multi_level_adds_level_suffix():
    assert evaluate.output_paths("runs/x/evaluation_run.mp4", "SuperMarioBros-1-2-v0", multi_level=True) == (
        "runs/x/evaluation_run_1-2.mp4", "runs/x/evaluation_run_1-2.gif",
    )


def test_format_summary_table_one_row_per_level():
    table = evaluate.format_summary_table({
        "SuperMarioBros-1-1-v0": {"clear_rate": 0.9, "mean_reward": 2950.4, "mean_max_x": 3100.2, "best_max_x": 3161},
        "SuperMarioBros-1-2-v0": {"clear_rate": 0.0, "mean_reward": 800.0, "mean_max_x": 900.0, "best_max_x": 1200},
    })
    lines = table.splitlines()
    assert len(lines) == 3
    assert lines[1].startswith("1-1") and "90%" in lines[1] and lines[1].endswith("3161")
    assert lines[2].startswith("1-2") and "0%" in lines[2]


def test_format_markdown_table_counts_clears():
    table = evaluate.format_markdown_table({
        "SuperMarioBros-1-1-v0": {"episodes": 10, "clear_rate": 0.9, "mean_reward": 2950.4,
                                  "mean_max_x": 3100.2, "best_max_x": 3161},
        "SuperMarioBros-1-4-v0": {"episodes": 10, "clear_rate": 0.0, "mean_reward": 800.0,
                                  "mean_max_x": 900.0, "best_max_x": 1200},
    })
    lines = table.splitlines()
    assert len(lines) == 4
    assert lines[1].startswith("| :---")
    assert lines[2] == "| 1-1 | 90% (9/10) | 2950 | 3100 | 3161 |"
    assert lines[3] == "| 1-4 | 0% (0/10) | 800 | 900 | 1200 |"


def test_video_fps_scales_from_real_time():
    assert evaluate.video_fps(1.0) == 15.0
    assert evaluate.video_fps(2.0) == 30.0
    assert evaluate.video_fps(0.5) == 7.5


def test_gif_stride_keeps_gif_near_15_fps():
    assert evaluate.gif_stride(1.0) == 1
    assert evaluate.gif_stride(2.0) == 2
    assert evaluate.gif_stride(4.0) == 4
    assert evaluate.gif_stride(0.5) == 1


def test_best_episode_prefers_a_clear_over_a_farther_death():
    results = [
        {"cleared": False, "reward": 2600.0, "steps": 400, "max_x": 3000},
        {"cleared": True, "reward": 2000.0, "steps": 500, "max_x": 2850},
    ]
    assert evaluate.best_episode_index(results) == 1


def test_best_episode_prefers_furthest_death_when_none_clear():
    results = [
        {"cleared": False, "reward": 231.0, "steps": 40, "max_x": 296},
        {"cleared": False, "reward": 2395.0, "steps": 2002, "max_x": 2850},
        {"cleared": False, "reward": 2612.0, "steps": 414, "max_x": 2750},
    ]
    assert evaluate.best_episode_index(results) == 1


def test_best_episode_breaks_clear_ties_on_reward():
    fast = {"cleared": True, "reward": 3100.0, "steps": 420, "max_x": 3161}
    slow = {"cleared": True, "reward": 2800.0, "steps": 700, "max_x": 3161}
    assert evaluate.best_episode_index([slow, fast]) == 1
