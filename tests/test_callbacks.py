"""
Tests for EpisodeStatsCallback in src/callbacks.py, driven with hand-built
locals dicts. Skipped if stable-baselines3 is not installed.
"""

import pytest

sb3 = pytest.importorskip("stable_baselines3")
from stable_baselines3.common.logger import Logger  # noqa: E402

from callbacks import EpisodeStatsCallback  # noqa: E402


def make_callback(window=100):
    callback = EpisodeStatsCallback(window=window)
    callback.logger = Logger(folder=None, output_formats=[])
    return callback


def step(callback, xs, dones, flags=None, levels=None):
    flags = flags or [False] * len(xs)
    infos = [{"x_pos": x, "flag_get": f} for x, f in zip(xs, flags)]
    for info, level in zip(infos, levels or []):
        info["world"], info["stage"] = level
    callback.locals = {"infos": infos, "dones": dones}
    return callback._on_step()


def test_nothing_logged_before_an_episode_ends():
    callback = make_callback()
    step(callback, [10, 20], [False, False])
    callback._on_rollout_end()
    assert "rollout/flag_rate" not in callback.logger.name_to_value


def test_tracks_max_x_per_env_and_flag_rate():
    callback = make_callback()
    step(callback, [100, 50], [False, False])
    step(callback, [300, 40], [False, False])  # env 1 walked backwards
    step(callback, [250, 60], [True, True], flags=[True, False])  # env 0 cleared, env 1 died
    callback._on_rollout_end()
    values = callback.logger.name_to_value
    assert values["rollout/flag_rate"] == 0.5
    assert values["rollout/max_x_mean"] == (300 + 60) / 2


def test_max_x_resets_after_episode_end():
    callback = make_callback()
    step(callback, [900], [True])
    step(callback, [30], [True])
    callback._on_rollout_end()
    assert callback.logger.name_to_value["rollout/max_x_mean"] == (900 + 30) / 2


def test_rolling_window_drops_old_episodes():
    callback = make_callback(window=2)
    step(callback, [10], [True], flags=[True])
    step(callback, [10], [True], flags=[False])
    step(callback, [10], [True], flags=[False])
    callback._on_rollout_end()
    assert callback.logger.name_to_value["rollout/flag_rate"] == 0.0


def test_on_step_never_interrupts_training():
    callback = make_callback()
    assert step(callback, [1], [False]) is True


def test_single_level_logs_only_pooled_stats():
    callback = make_callback()
    step(callback, [500, 600], [True, True], flags=[True, False], levels=[(1, 1), (1, 1)])
    callback._on_rollout_end()
    values = callback.logger.name_to_value
    assert values["rollout/flag_rate"] == 0.5
    assert not any(key.startswith("rollout/flag_rate_") for key in values)


def test_multi_level_logs_per_level_stats():
    callback = make_callback()
    levels = [(1, 1), (1, 2)]
    step(callback, [3000, 400], [True, True], flags=[True, False], levels=levels)
    step(callback, [2000, 900], [True, True], flags=[False, False], levels=levels)
    callback._on_rollout_end()
    values = callback.logger.name_to_value
    assert values["rollout/flag_rate"] == 0.25
    assert values["rollout/flag_rate_1-1"] == 0.5
    assert values["rollout/flag_rate_1-2"] == 0.0
    assert values["rollout/max_x_mean_1-1"] == 2500
    assert values["rollout/max_x_mean_1-2"] == 650


def test_episode_level_comes_from_its_first_step():
    callback = make_callback()
    step(callback, [100, 100], [False, False], levels=[(1, 1), (1, 2)])
    # env 0 clears 1-1 and the RAM already reports the next stage on the final step
    step(callback, [3161, 200], [True, True], flags=[True, False], levels=[(1, 2), (1, 2)])
    callback._on_rollout_end()
    values = callback.logger.name_to_value
    assert values["rollout/flag_rate_1-1"] == 1.0
    assert values["rollout/flag_rate_1-2"] == 0.0
