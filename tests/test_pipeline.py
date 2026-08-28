"""
Tests for run_pipeline.py argument routing. No emulator needed: subprocess.run
is stubbed and the argv each mode would execute is inspected directly.
"""

import os
import subprocess
import sys

import pytest

import run_pipeline


@pytest.fixture
def run(monkeypatch, tmp_path):
    """Runs the orchestrator with the given argv and returns the argv of each
    subprocess it would have launched (with the interpreter stripped)."""
    monkeypatch.chdir(tmp_path)
    calls = []

    def fake_run(cmd, check=True):
        calls.append(cmd)

    monkeypatch.setattr(subprocess, "run", fake_run)

    def _run(*argv):
        calls.clear()
        monkeypatch.setattr(sys, "argv", ["run_pipeline.py", *argv])
        run_pipeline.main()
        return [c[1:] if c[0] == sys.executable else c for c in calls]

    return _run


def scripts(calls):
    return [c[0] for c in calls]


def test_map_level_adds_default_version():
    assert run_pipeline.map_level("1-1") == "SuperMarioBros-1-1-v0"
    assert run_pipeline.map_level(" 2-3 ") == "SuperMarioBros-2-3-v0"


def test_map_level_keeps_explicit_version():
    assert run_pipeline.map_level("1-1-v3") == "SuperMarioBros-1-1-v3"


def test_all_mode_runs_every_stage_in_order(run):
    calls = run("--mode", "all", "--level", "1-1")
    assert scripts(calls) == [
        "src/validate.py",
        "src/train.py",
        "src/evaluate.py",
        "src/visualize_progress.py",
        "src/plot_metrics.py",
    ]


def test_all_mode_forwards_extra_args_only_to_train(run):
    calls = run("--mode", "all", "--learning_rate", "1e-4", "--gamma", "0.95")
    for cmd in calls:
        forwarded = "--learning_rate" in cmd
        assert forwarded == (cmd[0] == "src/train.py"), cmd


def test_train_mode_forwards_extra_args(run):
    (cmd,) = run("--mode", "train", "--learning_rate", "1e-4")
    assert cmd[0] == "src/train.py"
    assert cmd[-2:] == ["--learning_rate", "1e-4"]


def test_multi_level_env_id_is_comma_joined(run):
    (cmd,) = run("--mode", "train", "--level", "1-1,1-2")
    env_id = cmd[cmd.index("--env_id") + 1]
    assert env_id == "SuperMarioBros-1-1-v0,SuperMarioBros-1-2-v0"


def test_run_id_encodes_flags(run):
    def output_dir(*argv):
        (cmd,) = run("--mode", "train", *argv)
        return cmd[cmd.index("--output_dir") + 1]

    assert output_dir("--level", "1-1") == os.path.join("runs", "1-1_2000000steps_rew_lr")
    assert output_dir("--level", "1-1", "--no_custom_reward", "--no_lr_schedule") == os.path.join(
        "runs", "1-1_2000000steps"
    )
    assert output_dir("--level", "1-1,1-2", "--action_space", "right_only", "--load_model", "m.zip") == os.path.join(
        "runs", "1-1_1-2_2000000steps_rew_lr_resume_right_only"
    )


def test_validate_passes_action_space_and_reward_flag(run):
    (cmd,) = run("--mode", "validate", "--action_space", "right_only", "--no_custom_reward")
    assert cmd[0] == "src/validate.py"
    assert cmd[cmd.index("--action_space") + 1] == "right_only"
    assert "--no_custom_reward" in cmd


def test_evaluate_prefers_best_model_when_present(run, tmp_path):
    run_dir = tmp_path / "runs" / "1-1_2000000steps_rew_lr"
    run_dir.mkdir(parents=True)
    (run_dir / "mario_ppo_final.zip").touch()

    (cmd,) = run("--mode", "evaluate", "--level", "1-1")
    assert cmd[cmd.index("--model_path") + 1].endswith("mario_ppo_final.zip")

    (run_dir / "best_model.zip").touch()
    (cmd,) = run("--mode", "evaluate", "--level", "1-1")
    assert cmd[cmd.index("--model_path") + 1].endswith("best_model.zip")


def test_evaluate_explicit_model_path_wins(run):
    (cmd,) = run("--mode", "evaluate", "--model_path", "custom.zip")
    assert cmd[cmd.index("--model_path") + 1] == "custom.zip"


def test_device_defaults_to_auto(run):
    calls = run("--mode", "all")
    for cmd in calls:
        if "--device" in cmd:
            assert cmd[cmd.index("--device") + 1] == "auto"
