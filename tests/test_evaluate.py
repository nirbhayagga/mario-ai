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
