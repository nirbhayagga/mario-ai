"""
Environment Setup Module
Constructs the wrapped gym-super-mario-bros environment with observation
transformations required for PPO training with a CNN policy.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY
from nes_py.wrappers import JoypadSpace
from gym.wrappers import GrayScaleObservation, ResizeObservation
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv, VecFrameStack, VecMonitor


class CustomRewardWrapper(gym.Wrapper):
    """
    Stuck penalty and early termination only.

    The base gym-super-mario-bros environment already provides good reward
    shaping (x-velocity + clock penalty + death penalty). This wrapper adds
    only what the base env lacks: detection and punishment of the agent
    getting physically stuck against obstacles like pipes.

    Applied BEFORE SkipFrame, so frame counts are real NES frames.
    """

    def __init__(self, env: gym.Env) -> None:
        super(CustomRewardWrapper, self).__init__(env)
        self.last_x = 0
        self.x_stuck_frames = 0

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        state, reward, done, info = self.env.step(action)

        x_pos = info.get('x_pos', 0)

        # Rolling window: stuck means x hasn't changed by more than 1 pixel
        # since the last frame. This avoids penalizing legitimate backward
        # movement (e.g. underground pipe exits in 1-2, 4-2).
        if abs(x_pos - self.last_x) < 2:
            self.x_stuck_frames += 1
        else:
            self.x_stuck_frames = 0
        self.last_x = x_pos

        # ~1.3 seconds stuck (80 real frames at 60fps): continuous small penalty
        if self.x_stuck_frames > 80:
            reward -= 1.0

        # ~10 seconds stuck (600 real frames): end episode to save training time
        if self.x_stuck_frames > 600:
            done = True
            reward -= 15

        return state, reward, done, info

    def reset(self, **kwargs: Any) -> np.ndarray:
        self.last_x = 0
        self.x_stuck_frames = 0
        return self.env.reset(**kwargs)


class RewardClipWrapper(gym.RewardWrapper):
    """
    Clips rewards to [-15, 15] to prevent gradient explosions in PPO.
    Applied AFTER SkipFrame so the accumulated 4-frame reward is clipped.
    """

    def __init__(self, env: gym.Env, low: float = -15.0, high: float = 15.0) -> None:
        super(RewardClipWrapper, self).__init__(env)
        self.low = low
        self.high = high

    def reward(self, reward: float) -> float:
        return float(np.clip(reward, self.low, self.high))


class NoopResetEnv(gym.Wrapper):
    """
    Randomizes the starting state by executing 1-30 NOOP actions on reset.

    Uses its own RandomState rather than numpy's global one: SubprocVecEnv
    workers are forked with an identical copy of the global RNG, which would
    give every worker the same no-op sequence, and seed() makes it reproducible.
    """

    def __init__(self, env: gym.Env, noop_max: int = 30) -> None:
        super(NoopResetEnv, self).__init__(env)
        self.noop_max = noop_max
        self.rng = np.random.RandomState()

    def seed(self, seed: int | None = None) -> list[int]:
        self.rng = np.random.RandomState(seed)
        return self.env.seed(seed)

    def reset(self, **kwargs: Any) -> np.ndarray:
        self.env.reset(**kwargs)
        noops = self.rng.randint(1, self.noop_max + 1)
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(0)
            if done:
                obs = self.env.reset(**kwargs)
        return obs


class SkipFrame(gym.Wrapper):
    """
    Repeats the chosen action for `skip` consecutive frames, accumulates
    reward, and returns the pixel-wise maximum of the last 2 frames.

    Max-pooling prevents flickering NES sprites (rendered on alternating
    frames due to hardware limitations) from becoming invisible to the CNN.
    This is the standard MaxAndSkip technique from the original DQN paper.
    """

    def __init__(self, env: gym.Env, skip: int = 4) -> None:
        super(SkipFrame, self).__init__(env)
        self._skip = skip
        self._obs_buffer = np.zeros(
            (2,) + env.observation_space.shape, dtype=env.observation_space.dtype
        )

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        total_reward = 0.0
        done = False
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            # Store last 2 observations for max-pooling
            if i >= self._skip - 2:
                self._obs_buffer[i - (self._skip - 2)] = obs
            total_reward += reward
            if done:
                break
        # Pixel-wise max of last 2 frames
        max_frame = self._obs_buffer.max(axis=0)
        return max_frame, total_reward, done, info


def create_mario_environment(env_id: str | list[str] = "SuperMarioBros-v0", n_envs: int = 1,
                             use_custom_reward: bool = False, use_noop: bool = False, action_space: str = "simple",
                             subproc: bool = True) -> VecEnv:
    """
    Builds the full environment stack:
      [NoopReset] -> JoypadSpace -> [CustomReward] -> SkipFrame -> [RewardClip]
      -> GrayScale -> Resize -> VecEnv -> VecMonitor -> VecFrameStack(4)

    VecMonitor is required for SB3 to log rollout/ep_rew_mean and
    rollout/ep_len_mean: SB3 only auto-wraps plain gym envs in Monitor, not
    VecEnvs, so without it the reward curve never gets written.

    Stage-specific env IDs (SuperMarioBros-<w>-<s>-v<n>) already return done on
    death and on flag capture, so no episodic-life wrapper is needed.

    CustomReward is placed BEFORE SkipFrame so it sees individual frames.
    RewardClip is placed AFTER SkipFrame so the accumulated reward is clipped.

    Multi-env stacks use SubprocVecEnv (one worker process per env) unless
    subproc=False, which keeps every env in-process; evaluation uses that so
    it does not spawn workers that sit idle between evaluations.
    """

    def make_env(target_id: str) -> Callable[[], gym.Env]:
        def _init() -> gym.Env:
            env = gym_super_mario_bros.make(target_id)
            if use_noop:
                env = NoopResetEnv(env)
            if action_space == "complex":
                movement = COMPLEX_MOVEMENT
            elif action_space == "right_only":
                movement = RIGHT_ONLY
            else:
                movement = SIMPLE_MOVEMENT
            env = JoypadSpace(env, movement)
            if use_custom_reward:
                env = CustomRewardWrapper(env)
            env = SkipFrame(env, skip=4)
            if use_custom_reward:
                env = RewardClipWrapper(env, low=-15.0, high=15.0)
            env = GrayScaleObservation(env, keep_dim=True)
            env = ResizeObservation(env, shape=84)
            return env
        return _init

    vec_env_cls = SubprocVecEnv if subproc else DummyVecEnv
    if isinstance(env_id, list):
        # Cycle levels across n_envs so each level gets balanced coverage
        actual_n = max(n_envs, len(env_id))
        print(f"Creating multi-level environment: {env_id} ({actual_n} envs)")
        env = vec_env_cls([make_env(env_id[i % len(env_id)]) for i in range(actual_n)])
    elif n_envs > 1:
        env = vec_env_cls([make_env(env_id) for _ in range(n_envs)])
    else:
        env = DummyVecEnv([make_env(env_id)])

    env = VecMonitor(env)
    env = VecFrameStack(env, n_stack=4, channels_order='last')
    return env
