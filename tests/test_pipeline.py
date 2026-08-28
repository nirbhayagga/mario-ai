"""
Tests for run_pipeline.py argument routing. No emulator needed: subprocess.run
is stubbed and the argv each command would execute is inspected directly.
"""

import json
import os
import subprocess
import sys
import time

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
        run_pipeline.main(list(argv))
        return [c[1:] if c[0] == sys.executable else c for c in calls]

    return _run


def scripts(calls):
    return [c[0] for c in calls]


def flag(cmd, name):
    return cmd[cmd.index(name) + 1]


def make_run(tmp_path, name="1-1_2000000steps_rew_lr", config=None, files=("mario_ppo_final.zip",), age=0):
    """Creates runs/<name> with optional run_config.json (age seconds old) and model files."""
    run_dir = tmp_path / "runs" / name
    run_dir.mkdir(parents=True)
    if config is not None:
        path = run_dir / "run_config.json"
        path.write_text(json.dumps(config))
        stamp = time.time() - age
        os.utime(path, (stamp, stamp))
    for f in files:
        (run_dir / f).touch()
    return run_dir


# ---- level mapping ----

def test_map_level_adds_default_version():
    assert run_pipeline.map_level("1-1") == "SuperMarioBros-1-1-v0"
    assert run_pipeline.map_level(" 2-3 ") == "SuperMarioBros-2-3-v0"


def test_map_level_keeps_explicit_version():
    assert run_pipeline.map_level("1-1-v3") == "SuperMarioBros-1-1-v3"


# ---- parser strictness ----

def test_command_is_required(run):
    with pytest.raises(SystemExit):
        run()


def test_unknown_flag_is_rejected(run):
    with pytest.raises(SystemExit):
        run("train", "--bogus", "1")


def test_eval_only_flags_are_rejected_on_train(run):
    with pytest.raises(SystemExit):
        run("train", "--eval_episodes", "3")


# ---- train ----

def test_train_derives_run_dir_from_flags(run):
    def output_dir(*argv):
        (cmd,) = run("train", *argv)
        return flag(cmd, "--output_dir")

    assert output_dir() == os.path.join("runs", "1-1_2000000steps_rew_lr")
    assert output_dir("--level", "1-1", "--no_custom_reward", "--no_lr_schedule") == os.path.join(
        "runs", "1-1_2000000steps"
    )
    assert output_dir("--level", "1-1,1-2", "--action_space", "right_only", "--load_model", "m.zip") == os.path.join(
        "runs", "1-1_1-2_2000000steps_rew_lr_resume_right_only"
    )


def test_train_run_dir_overrides_derived_path(run):
    (cmd,) = run("train", "--run_dir", "runs/custom")
    assert flag(cmd, "--output_dir") == "runs/custom"


def test_train_forwards_ppo_flags_only_when_set(run):
    (cmd,) = run("train")
    assert "--learning_rate" not in cmd
    assert "--n_epochs" not in cmd
    (cmd,) = run("train", "--learning_rate", "1e-4", "--n_epochs", "5", "--clip_range", "0.2")
    assert flag(cmd, "--learning_rate") == "0.0001"
    assert flag(cmd, "--n_epochs") == "5"
    assert flag(cmd, "--clip_range") == "0.2"


def test_train_forwards_resume_and_reward_flags(run):
    (cmd,) = run("train", "--load_model", "m.zip", "--no_custom_reward", "--no_lr_schedule", "--save_freq", "500")
    assert flag(cmd, "--load_model") == "m.zip"
    assert "--no_custom_reward" in cmd and "--no_lr_schedule" in cmd
    assert flag(cmd, "--save_freq") == "500"


def test_multi_level_env_id_is_comma_joined(run):
    (cmd,) = run("train", "--level", "1-1,1-2")
    assert flag(cmd, "--env_id") == "SuperMarioBros-1-1-v0,SuperMarioBros-1-2-v0"


# ---- validate / manual ----

def test_validate_passes_action_space_and_reward_flag(run):
    (cmd,) = run("validate", "--action_space", "right_only", "--no_custom_reward")
    assert cmd[0] == "src/validate.py"
    assert flag(cmd, "--action_space") == "right_only"
    assert "--no_custom_reward" in cmd


def test_manual_maps_level(run):
    (cmd,) = run("manual", "--level", "4-2")
    assert cmd == ["src/manual_play.py", "--env_id", "SuperMarioBros-4-2-v0"]


# ---- commands on existing runs ----

def test_evaluate_uses_latest_run_when_run_dir_omitted(run, tmp_path):
    make_run(tmp_path, "old", config={"env_id": "SuperMarioBros-1-1-v0"}, age=100)
    make_run(tmp_path, "new", config={"env_id": "SuperMarioBros-2-1-v0", "action_space": "right_only"})
    (cmd,) = run("evaluate")
    assert flag(cmd, "--env_id") == "SuperMarioBros-2-1-v0"
    assert flag(cmd, "--action_space") == "right_only"
    assert flag(cmd, "--model_path") == os.path.join("runs", "new", "mario_ppo_final.zip")
    assert flag(cmd, "--video_output") == os.path.join("runs", "new", "evaluation_run.mp4")


def test_evaluate_without_any_run_exits(run):
    with pytest.raises(SystemExit):
        run("evaluate")


def test_evaluate_prefers_best_model_when_present(run, tmp_path):
    run_dir = make_run(tmp_path, files=("mario_ppo_final.zip", "best_model.zip"))
    (cmd,) = run("evaluate", "--run_dir", str(run_dir))
    assert flag(cmd, "--model_path") == str(run_dir / "best_model.zip")


def test_explicit_flags_override_run_config(run, tmp_path):
    run_dir = make_run(tmp_path, config={"env_id": "SuperMarioBros-4-2-v0", "action_space": "right_only"})
    (cmd,) = run("evaluate", "--run_dir", str(run_dir), "--level", "1-1", "--action_space", "complex",
                 "--model_path", "custom.zip")
    assert flag(cmd, "--env_id") == "SuperMarioBros-1-1-v0"
    assert flag(cmd, "--action_space") == "complex"
    assert flag(cmd, "--model_path") == "custom.zip"


def test_run_dir_without_config_falls_back_to_defaults(run, tmp_path):
    run_dir = make_run(tmp_path, "bare")
    (cmd,) = run("evaluate", "--run_dir", str(run_dir))
    assert flag(cmd, "--env_id") == "SuperMarioBros-1-1-v0"
    assert flag(cmd, "--action_space") == "simple"


def test_evaluate_forwards_episode_count_and_video_speed(run, tmp_path):
    run_dir = make_run(tmp_path)
    (cmd,) = run("evaluate", "--run_dir", str(run_dir))
    assert flag(cmd, "--episodes") == "10"
    assert flag(cmd, "--video_speed") == "2.0"
    (cmd,) = run("evaluate", "--run_dir", str(run_dir), "--eval_episodes", "25", "--video_speed", "1")
    assert flag(cmd, "--episodes") == "25"
    assert flag(cmd, "--video_speed") == "1.0"


def test_visualize_defaults_and_num_steps(run, tmp_path):
    run_dir = make_run(tmp_path)
    (cmd,) = run("visualize", "--run_dir", str(run_dir), "--num_steps", "300")
    assert cmd[0] == "src/visualize_progress.py"
    assert flag(cmd, "--checkpoint_dir") == str(run_dir / "checkpoints")
    assert flag(cmd, "--output_path") == str(run_dir / "progress_montage.mp4")
    assert flag(cmd, "--num_steps") == "300"


def test_plot_uses_run_tensorboard_dir(run, tmp_path):
    run_dir = make_run(tmp_path)
    (cmd,) = run("plot", "--run_dir", str(run_dir))
    assert cmd == ["src/plot_metrics.py", "--log_dir", str(run_dir / "tensorboard"),
                   "--output_path", str(run_dir / "reward_curve.png")]


def test_dashboard_uses_run_tensorboard_dir_and_port(run, tmp_path):
    run_dir = make_run(tmp_path)
    (cmd,) = run("dashboard", "--run_dir", str(run_dir), "--port", "7007")
    assert cmd == ["tensorboard", "--logdir", str(run_dir / "tensorboard"), "--port", "7007"]


# ---- all ----

def test_all_runs_every_stage_in_order(run):
    calls = run("all", "--level", "1-1")
    assert scripts(calls) == [
        "src/validate.py", "src/train.py", "src/evaluate.py", "src/visualize_progress.py", "src/plot_metrics.py",
    ]


def test_all_keeps_training_flags_off_downstream_stages(run):
    for cmd in run("all", "--learning_rate", "1e-4", "--gamma", "0.95"):
        assert ("--learning_rate" in cmd) == (cmd[0] == "src/train.py"), cmd


def test_all_stages_share_one_run_dir(run):
    calls = run("all", "--run_dir", "runs/custom")
    assert flag(calls[1], "--output_dir") == "runs/custom"
    assert flag(calls[2], "--model_path") == os.path.join("runs", "custom", "mario_ppo_final.zip")
    assert flag(calls[3], "--checkpoint_dir") == os.path.join("runs", "custom", "checkpoints")
    assert flag(calls[4], "--log_dir") == os.path.join("runs", "custom", "tensorboard")


def test_seed_forwarded_only_to_train_and_evaluate(run):
    for cmd in run("all", "--seed", "7"):
        seeded = "--seed" in cmd
        assert seeded == (cmd[0] in ("src/train.py", "src/evaluate.py")), cmd
        if seeded:
            assert flag(cmd, "--seed") == "7"
    (cmd,) = run("train")
    assert "--seed" not in cmd


def test_device_defaults_to_auto(run):
    for cmd in run("all"):
        if "--device" in cmd:
            assert flag(cmd, "--device") == "auto"
