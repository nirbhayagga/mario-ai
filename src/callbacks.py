"""
Training Callbacks
Logs the two numbers that actually say whether Mario is learning a level:
the fraction of episodes that reach the flag and how far Mario gets. The
shaped episode reward SB3 logs by default is hard to interpret on its own.
"""

from __future__ import annotations

from collections import deque

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class EpisodeStatsCallback(BaseCallback):
    """
    Tracks per-env max x-position, and on each episode end records whether
    the flag was reached. At the end of every rollout it logs rolling means
    over the last `window` finished episodes, matching the window SB3 uses
    for rollout/ep_rew_mean:

      rollout/flag_rate   fraction of recent episodes that cleared the level
      rollout/max_x_mean  mean furthest x-position reached per episode
    """

    def __init__(self, window: int = 100, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.window = window
        self._flags = deque(maxlen=window)
        self._max_xs = deque(maxlen=window)
        self._episode_max_x = None

    def _on_step(self) -> bool:
        infos = self.locals["infos"]
        dones = self.locals["dones"]
        if self._episode_max_x is None or len(self._episode_max_x) != len(infos):
            self._episode_max_x = np.zeros(len(infos))

        for i, (info, done) in enumerate(zip(infos, dones)):
            self._episode_max_x[i] = max(self._episode_max_x[i], info.get("x_pos", 0))
            if done:
                self._flags.append(float(info.get("flag_get", False)))
                self._max_xs.append(self._episode_max_x[i])
                self._episode_max_x[i] = 0
        return True

    def _on_rollout_end(self) -> None:
        if self._flags:
            self.logger.record("rollout/flag_rate", float(np.mean(self._flags)))
            self.logger.record("rollout/max_x_mean", float(np.mean(self._max_xs)))
