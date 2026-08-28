"""Tests for the GIF recorder in src/media.py. Pure Pillow, no emulator."""

import numpy as np
from PIL import Image

from media import GifRecorder


def frame(value):
    return np.full((8, 8, 3), value, np.uint8)


def test_records_every_nth_frame_and_loops_forever(tmp_path):
    path = tmp_path / "clip.gif"
    recorder = GifRecorder(str(path), fps=30.0, every=3)
    for i in range(10):
        recorder.add(frame(i * 20))
    assert len(recorder) == 4  # frames 0, 3, 6, 9
    assert recorder.save()

    with Image.open(path) as gif:
        assert gif.n_frames == 4
        assert gif.info["loop"] == 0
        assert gif.info["duration"] == 100  # 3 frames at 30 fps


def test_save_without_frames_writes_nothing(tmp_path):
    path = tmp_path / "empty.gif"
    assert not GifRecorder(str(path)).save()
    assert not path.exists()
