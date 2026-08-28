"""
Media Helpers
GIF export for README-embeddable clips. GitHub renders GIFs inline but not
repository .mp4 files, so evaluation runs are written in both formats.
"""

from __future__ import annotations

import numpy as np

from PIL import Image


class GifRecorder:
    """
    Collects every `every`-th RGB frame, palette-quantizes it immediately to
    bound memory, and writes an infinitely looping GIF on save().
    """

    def __init__(self, path: str, fps: float = 30.0, every: int = 2, colors: int = 128) -> None:
        self.path = path
        self.every = max(1, int(every))
        self.colors = colors
        self.duration_ms = int(round(1000.0 * self.every / fps))
        self._frames = []
        self._seen = 0

    def add(self, rgb_frame: np.ndarray) -> None:
        if self._seen % self.every == 0:
            self._frames.append(Image.fromarray(rgb_frame).quantize(colors=self.colors))
        self._seen += 1

    def __len__(self) -> int:
        return len(self._frames)

    def save(self) -> bool:
        """Writes the GIF. Returns False if no frames were recorded."""
        if not self._frames:
            return False
        first, rest = self._frames[0], self._frames[1:]
        first.save(self.path, save_all=True, append_images=rest, duration=self.duration_ms, loop=0)
        return True
