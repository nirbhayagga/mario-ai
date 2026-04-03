"""
Environment Setup Module
Constructs the wrapped gym-super-mario-bros environment with observation
transformations required for PPO training with a CNN policy.
"""

import gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY
from nes_py.wrappers import JoypadSpace
from gym.wrappers import GrayScaleObservation, ResizeObservation
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, SubprocVecEnv


class CustomRewardWrapper(gym.Wrapper):
    """
    Reward shaping: flagpole bonus, death penalty, stuck penalty,
    coin collection, score delta (kill proxy), power-up tracking,
    and time bonus on level clear.
    """

    def __init__(self, env):
        super(CustomRewardWrapper, self).__init__(env)
        self.current_x = 0
        self.x_stuck_frames = 0
        self.last_score = 0
        self.last_coins = 0
        self.last_status = 'small'

    def step(self, action):
        state, reward, done, info = self.env.step(action)

        # --- Flagpole bonus / death penalty ---
        if info.get('flag_get', False):
            reward += 500
            # Time bonus: reward finishing quickly (time_remaining * 0.1)
            time_left = info.get('time', 0)
            reward += time_left * 0.1

        if done and not info.get('flag_get', False):
            reward -= 50

        # --- Coin collection bonus ---
        coins = info.get('coins', 0)
        coin_delta = coins - self.last_coins
        if coin_delta > 0:
            reward += coin_delta * 2.0
        self.last_coins = coins

        # --- Score delta (proxy for enemy kills, bricks, etc.) ---
        score = info.get('score', 0)
        score_delta = score - self.last_score
        if score_delta > 0:
            reward += score_delta * 0.025
        self.last_score = score

        # --- Power-up / hit tracking ---
        status = info.get('status', 'small')
        if self.last_status == 'small' and status in ('tall', 'fireball'):
            reward += 25   # Got mushroom or flower
        elif self.last_status == 'fireball' and status == 'tall':
            reward += 25   # Got flower from tall (still a power-up)
        elif self.last_status in ('tall', 'fireball') and status == 'small':
            reward -= 25   # Got hit and shrunk
        self.last_status = status

        # --- Stuck penalty (not moving right) ---
        x_pos = info.get('x_pos', 0)

        if x_pos <= self.current_x:
            self.x_stuck_frames += 1
        else:
            self.x_stuck_frames = 0
            self.current_x = max(self.current_x, x_pos)

        # ~2.5 seconds stuck: continuous small penalty
        if self.x_stuck_frames > 20:
            reward -= 1.0

        # ~10 seconds stuck: end episode to save training time
        if self.x_stuck_frames > 150:
            done = True
            reward -= 20

        return state, reward, done, info

    def reset(self, **kwargs):
        self.current_x = 0
        self.x_stuck_frames = 0
        self.last_score = 0
        self.last_coins = 0
        self.last_status = 'small'
        return self.env.reset(**kwargs)


class NoopResetEnv(gym.Wrapper):
    """Randomizes the starting state by executing 1-30 NOOP actions on reset."""

    def __init__(self, env, noop_max=30):
        super(NoopResetEnv, self).__init__(env)
        self.noop_max = noop_max

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        noops = self.unwrapped.np_random.randint(1, self.noop_max + 1)
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(0)
            if done:
                obs = self.env.reset(**kwargs)
        return obs


class SkipFrame(gym.Wrapper):
    """
    Repeats the chosen action for `skip` consecutive frames and accumulates
    reward. Returns only the final frame. Reduces training time by ~75% and
    gives the frame stack a wider temporal window (~260ms at skip=4).
    """

    def __init__(self, env, skip=4):
        super(SkipFrame, self).__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info


class EpisodicLifeEnv(gym.Wrapper):
    """
    Treats each life loss as a terminal state. Without this wrapper, Mario
    respawns after death without a done signal, breaking the MDP assumption
    that the agent's state should not teleport discontinuously.
    """

    def __init__(self, env):
        super(EpisodicLifeEnv, self).__init__(env)
        self.lives = 0

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        lives = info.get('life', 0)
        if self.lives > lives > 0:
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.lives = 0
        return obs


def create_mario_environment(env_id="SuperMarioBros-v0", n_envs=1,
                             use_custom_reward=False, use_noop=False, action_space="simple"):
    """
    Builds the full environment stack: JoypadSpace, EpisodicLife, SkipFrame,
    optional reward shaping, grayscale, resize to 84x84, then VecFrameStack(4).
    Supports single level, parallel single level, and multi-level training.
    """

    def make_env(target_id):
        def _init():
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
            env = EpisodicLifeEnv(env)
            env = SkipFrame(env, skip=4)
            if use_custom_reward:
                env = CustomRewardWrapper(env)
            env = GrayScaleObservation(env, keep_dim=True)
            env = ResizeObservation(env, shape=84)
            return env
        return _init

    if isinstance(env_id, list):
        # Cycle levels across n_envs so each level gets balanced coverage
        actual_n = max(n_envs, len(env_id))
        print(f"Creating multi-level environment: {env_id} ({actual_n} envs)")
        env = SubprocVecEnv([make_env(env_id[i % len(env_id)]) for i in range(actual_n)])
    elif n_envs > 1:
        env = SubprocVecEnv([make_env(env_id) for _ in range(n_envs)])
    else:
        env = DummyVecEnv([make_env(env_id)])

    env = VecFrameStack(env, n_stack=4, channels_order='last')
    return env
