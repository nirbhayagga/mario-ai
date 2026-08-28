"""
Tests for the reward and frame wrappers in src/env_setup.py, run against a
tiny scripted gym.Env so no NES emulator is required. Skipped if the RL stack
(gym / stable-baselines3 / nes-py) is not installed.
"""

import numpy as np
import pytest

gym = pytest.importorskip("gym")
env_setup = pytest.importorskip("env_setup")

OBS_SHAPE = (4, 4, 3)


class ScriptedMario(gym.Env):
    """Action 1 moves Mario right by `stride` pixels; action 0 stands still.
    Each frame's observation is filled with the frame index so max-pooling is
    observable. Optionally terminates on a given frame."""

    observation_space = gym.spaces.Box(0, 255, OBS_SHAPE, np.uint8)
    action_space = gym.spaces.Discrete(2)

    def __init__(self, stride=5, reward=1.0, done_at=None):
        self.stride = stride
        self.frame_reward = reward
        self.done_at = done_at
        self.reset()

    def reset(self, **kwargs):
        self.x = 0
        self.frame = 0
        return np.zeros(OBS_SHAPE, np.uint8)

    def step(self, action):
        self.frame += 1
        if action == 1:
            self.x += self.stride
        obs = np.full(OBS_SHAPE, self.frame % 256, np.uint8)
        done = self.done_at is not None and self.frame >= self.done_at
        return obs, self.frame_reward, done, {"x_pos": self.x}


def run_frames(env, action, n):
    """Steps n times and returns the list of (reward, done) pairs."""
    out = []
    for _ in range(n):
        _, r, d, _ = env.step(action)
        out.append((r, d))
    return out


class TestCustomRewardWrapper:
    def test_moving_right_is_never_penalized(self):
        env = env_setup.CustomRewardWrapper(ScriptedMario())
        env.reset()
        assert all(r == 1.0 and not d for r, d in run_frames(env, 1, 700))

    def test_penalty_starts_after_80_stuck_frames(self):
        env = env_setup.CustomRewardWrapper(ScriptedMario())
        env.reset()
        results = run_frames(env, 0, 81)
        assert all(r == 1.0 for r, _ in results[:80])
        assert results[80] == (0.0, False)

    def test_episode_terminates_after_600_stuck_frames(self):
        env = env_setup.CustomRewardWrapper(ScriptedMario())
        env.reset()
        results = run_frames(env, 0, 601)
        assert results[599] == (0.0, False)
        assert results[600] == (1.0 - 1.0 - 15.0, True)

    def test_sub_two_pixel_drift_still_counts_as_stuck(self):
        env = env_setup.CustomRewardWrapper(ScriptedMario(stride=1))
        env.reset()
        assert run_frames(env, 1, 81)[80][0] == 0.0

    def test_two_pixel_movement_resets_counter(self):
        env = env_setup.CustomRewardWrapper(ScriptedMario(stride=2))
        env.reset()
        assert all(r == 1.0 for r, _ in run_frames(env, 1, 200))

    def test_reset_clears_stuck_counter(self):
        env = env_setup.CustomRewardWrapper(ScriptedMario())
        env.reset()
        run_frames(env, 0, 100)
        env.reset()
        assert run_frames(env, 0, 80)[-1][0] == 1.0


class TestSkipFrame:
    def test_repeats_action_and_sums_reward(self):
        base = ScriptedMario()
        env = env_setup.SkipFrame(base, skip=4)
        env.reset()
        _, reward, done, info = env.step(1)
        assert base.frame == 4
        assert info["x_pos"] == 20
        assert reward == 4.0
        assert not done

    def test_observation_is_max_of_last_two_frames(self):
        env = env_setup.SkipFrame(ScriptedMario(), skip=4)
        env.reset()
        obs, *_ = env.step(0)
        assert obs.shape == OBS_SHAPE
        assert obs.max() == 4  # frames 3 and 4 are pooled; frame 4 wins

    def test_stops_early_when_episode_ends(self):
        base = ScriptedMario(done_at=2)
        env = env_setup.SkipFrame(base, skip=4)
        env.reset()
        _, reward, done, _ = env.step(1)
        assert done
        assert base.frame == 2
        assert reward == 2.0


class TestRewardClipWrapper:
    @pytest.mark.parametrize("raw, expected", [(100.0, 15.0), (-100.0, -15.0), (3.5, 3.5), (-15.0, -15.0)])
    def test_clips_to_bounds(self, raw, expected):
        env = env_setup.RewardClipWrapper(ScriptedMario(reward=raw))
        env.reset()
        _, reward, _, _ = env.step(0)
        assert reward == expected
        assert isinstance(reward, float)


class TestNoopResetEnv:
    def test_reset_advances_between_1_and_30_frames(self):
        base = ScriptedMario()
        env = env_setup.NoopResetEnv(base, noop_max=30)
        np.random.seed(0)
        frames = set()
        for _ in range(50):
            obs = env.reset()
            frames.add(base.frame)
            assert obs.max() == base.frame % 256
        assert frames <= set(range(1, 31))
        assert len(frames) > 1


class TestEnvironmentStack:
    def test_single_env_observation_shape(self):
        """Builds the real emulator stack once: (84, 84, 4) uint8 for CnnPolicy."""
        env = env_setup.create_mario_environment("SuperMarioBros-1-1-v0", n_envs=1, use_custom_reward=True)
        try:
            obs = env.reset()
            assert obs.shape == (1, 84, 84, 4)
            assert obs.dtype == np.uint8
            obs, reward, done, info = env.step([0])
            assert obs.shape == (1, 84, 84, 4)
            assert "x_pos" in info[0]
        finally:
            env.close()

    def test_stuck_episode_terminates_and_reports_monitor_stats(self):
        """NOOP forever: Mario never moves, so CustomRewardWrapper ends the
        episode after 600 stuck NES frames (~150 agent steps at skip=4), and
        VecMonitor attaches the episode return/length SB3 needs for logging."""
        env = env_setup.create_mario_environment("SuperMarioBros-1-1-v0", n_envs=1, use_custom_reward=True)
        try:
            env.reset()
            step = 0
            done = [False]
            while not done[0] and step < 400:
                step += 1
                _, _, done, info = env.step([0])
            assert done[0], "stuck episode never terminated"
            assert 140 <= step <= 160, step
            assert {"r", "l"} <= set(info[0]["episode"])
            assert info[0]["episode"]["l"] == step
        finally:
            env.close()
