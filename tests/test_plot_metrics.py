"""
Tests for src/plot_metrics.py against synthetic TensorBoard logs written with
torch's SummaryWriter. Skipped if torch is not installed.
"""

import pytest

torch = pytest.importorskip("torch")
from torch.utils.tensorboard import SummaryWriter  # noqa: E402

import plot_metrics as pm  # noqa: E402


def write_events(log_dir, points_by_tag):
    writer = SummaryWriter(str(log_dir))
    for tag, points in points_by_tag.items():
        for step, value in points:
            writer.add_scalar(tag, value, step)
    writer.close()


def test_load_scalars_merges_event_files_and_sorts_by_step(tmp_path):
    write_events(tmp_path / "PPO_1", {pm.TRAIN_REWARD_TAG: [(100, 1.0), (300, 3.0)]})
    write_events(tmp_path / "PPO_2", {pm.TRAIN_REWARD_TAG: [(200, 2.0)]})
    scalars = pm.load_scalars(str(tmp_path))
    assert [step for step, _ in scalars[pm.TRAIN_REWARD_TAG]] == [100, 200, 300]
    assert [value for _, value in scalars[pm.TRAIN_REWARD_TAG]] == [1.0, 2.0, 3.0]


def test_plot_writes_png_when_reward_tags_exist(tmp_path):
    write_events(tmp_path / "PPO_1", {
        pm.TRAIN_REWARD_TAG: [(i * 1000, float(i)) for i in range(10)],
        pm.EVAL_REWARD_TAG: [(5000, 50.0), (10000, 80.0)],
    })
    output = tmp_path / "out" / "reward_curve.png"
    assert pm.plot_training_curves(str(tmp_path), str(output))
    assert output.stat().st_size > 0


def test_plot_fails_loudly_without_reward_tags(tmp_path, capsys):
    write_events(tmp_path / "PPO_1", {"time/fps": [(100, 200.0)]})
    output = tmp_path / "reward_curve.png"
    assert not pm.plot_training_curves(str(tmp_path), str(output))
    assert not output.exists()
    assert "time/fps" in capsys.readouterr().out


def test_plot_handles_missing_log_dir(tmp_path):
    assert not pm.plot_training_curves(str(tmp_path / "nope"), str(tmp_path / "x.png"))
