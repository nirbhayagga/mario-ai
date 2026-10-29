"""
Training Callbacks
Logs the two numbers that actually say whether Mario is learning a level:
the fraction of episodes that reach the flag and how far Mario gets. The
shaped episode reward SB3 logs by default is hard to interpret on its own.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


def level_key(info: dict[str, Any]) -> str | None:
    """"1-2" from a gym-super-mario-bros info dict, or None if it carries no level."""
    if "world" in info and "stage" in info:
        return f"{info['world']}-{info['stage']}"
    return None


class EpisodeStatsCallback(BaseCallback):
    """
    Tracks per-env max x-position, and on each episode end records whether
    the flag was reached. At the end of every rollout it logs rolling means
    over the last `window` finished episodes, matching the window SB3 uses
    for rollout/ep_rew_mean:

      rollout/flag_rate   fraction of recent episodes that cleared the level
      rollout/max_x_mean  mean furthest x-position reached per episode

    Once episodes from more than one level have finished (multi-level runs),
    the same two numbers are also logged per level, each over its own window,
    as rollout/flag_rate_<w>-<s> and rollout/max_x_mean_<w>-<s>. The pooled
    numbers alone hide a level that never clears behind one that always does.
    An episode's level is taken from its first step: on the final step the
    game may already be loading the next stage (or a warp zone's target).
    """

    def __init__(self, window: int = 100, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.window = window
        self._flags = deque(maxlen=window)
        self._max_xs = deque(maxlen=window)
        self._level_flags: dict[str, deque] = {}
        self._level_max_xs: dict[str, deque] = {}
        self._episode_max_x = None
        self._episode_level: list[str | None] = []

    def _on_step(self) -> bool:
        infos = self.locals["infos"]
        dones = self.locals["dones"]
        if self._episode_max_x is None or len(self._episode_max_x) != len(infos):
            self._episode_max_x = np.zeros(len(infos))
            self._episode_level = [None] * len(infos)

        for i, (info, done) in enumerate(zip(infos, dones)):
            if self._episode_level[i] is None:
                self._episode_level[i] = level_key(info)
            self._episode_max_x[i] = max(self._episode_max_x[i], info.get("x_pos", 0))
            if done:
                flag = float(info.get("flag_get", False))
                self._flags.append(flag)
                self._max_xs.append(self._episode_max_x[i])
                level = self._episode_level[i]
                if level is not None:
                    self._level_flags.setdefault(level, deque(maxlen=self.window)).append(flag)
                    self._level_max_xs.setdefault(level, deque(maxlen=self.window)).append(self._episode_max_x[i])
                self._episode_max_x[i] = 0
                self._episode_level[i] = None
        return True

    def _on_rollout_end(self) -> None:
        if self._flags:
            self.logger.record("rollout/flag_rate", float(np.mean(self._flags)))
            self.logger.record("rollout/max_x_mean", float(np.mean(self._max_xs)))
        if len(self._level_flags) > 1:
            for level in sorted(self._level_flags):
                self.logger.record(f"rollout/flag_rate_{level}", float(np.mean(self._level_flags[level])))
                self.logger.record(f"rollout/max_x_mean_{level}", float(np.mean(self._level_max_xs[level])))
