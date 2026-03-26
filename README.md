# Super Mario Bros PPO Agent

This repository trains a Reinforcement Learning agent to play Super Mario Bros using Proximal Policy Optimization (PPO) with Stable Baselines3.

## Setup

Python 3.8 or 3.9 is required due to dependency constraints.

### Step 1: Create Environment

**Option A (Mamba/Conda, recommended):**
```bash
mamba create -n mario_rl python=3.9
mamba activate mario_rl
```

**Option B (Standard venv):**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 2: Install Dependencies
```bash
pip install "pip<24.1" "setuptools==65.5.0" "wheel==0.38.4"
pip install -r requirements.txt
```

---

## Usage

All tasks are run through `run_pipeline.py`. Specify the level with `--level` (e.g. `1-1`, `2-1`).

### Command Line Arguments

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--mode` | (required) | `validate`, `train`, `evaluate`, `visualize`, `plot`, `dashboard`, `train_general`, `manual`, `compete`, `all` |
| `--level` | `1-1` | Target level. Supports comma-separated lists (e.g. `1-1,1-2,2-1`). |
| `--device` | `cuda` | `cuda` or `cpu`. |
| `--total_timesteps` | `2000000` | Total training steps. |
| `--n_envs` | `4` | Parallel environments. |
| `--use_custom_reward` | off | Enables flagpole bonus (+500) and death penalty (-50). |
| `--lr_schedule` | off | Linear learning rate decay over training. |
| `--load_model` | none | Path to a `.zip` model to resume from. |
| `--model_path` | none | Model for evaluation. |
| `--checkpoint_dir` | none | Checkpoint directory for montage visualization. |
| `--video_output` | none | Output path for `.mp4` video files. |

### PPO Hyperparameters

These defaults are based on community benchmarks from GitHub and Reddit for Super Mario Bros:

| Parameter | Default | Rationale |
| :--- | :--- | :--- |
| `--n_steps` | `2048` | Mario levels are long. A wide trajectory window lets the agent connect early actions to late rewards. |
| `--gamma` | `0.99` | High discount factor prioritizes reaching the flagpole over grabbing local coins. |
| `--batch_size` | `64` | Small batches improve gradient signal quality for pixel-based input. |
| `--ent_coef` | `0.01` | Entropy bonus encourages exploration and prevents the agent from getting stuck in loops. |
| `--vf_coef` | `0.5` | Balanced actor-critic weighting, standard for PPO. |
| `max_grad_norm` | `0.5` | Gradient clipping prevents catastrophic policy updates during long runs. |

### Stage Versions

Each level has 4 visual variations (append `-v0` through `-v3` to the level):

| Version | Description |
| :--- | :--- |
| `-v0` | Original NES graphics (default, recommended). |
| `-v1` | Downsampled sprites. |
| `-v2` | Pixel-only, black background. |
| `-v3` | Sprites only, no background. |

### Level Types

The 32 levels fall into four categories requiring different AI strategies:

* **Overworld**: Classic platforming (e.g. 1-1, 2-1).
* **Underground**: Ceiling-constrained stages (e.g. 1-2, 4-2).
* **Underwater**: Swimming physics (e.g. 2-2, 7-2).
* **Castle**: Firebars, lava, and timing hazards (e.g. 1-4, 8-4).

### Level Reference

| World | Stage 1 | Stage 2 | Stage 3 | Stage 4 |
| :--- | :--- | :--- | :--- | :--- |
| 1 | 1-1 (Over) | 1-2 (Under) | 1-3 (Over) | 1-4 (Castle) |
| 2 | 2-1 (Over) | 2-2 (Water) | 2-3 (Over) | 2-4 (Castle) |
| 3 | 3-1 (Over) | 3-2 (Over) | 3-3 (Over) | 3-4 (Castle) |
| 4 | 4-1 (Over) | 4-2 (Under) | 4-3 (Over) | 4-4 (Castle) |
| 5 | 5-1 (Over) | 5-2 (Over) | 5-3 (Over) | 5-4 (Castle) |
| 6 | 6-1 (Over) | 6-2 (Under) | 6-3 (Over) | 6-4 (Castle) |
| 7 | 7-1 (Over) | 7-2 (Water) | 7-3 (Over) | 7-4 (Castle) |
| 8 | 8-1 (Over) | 8-2 (Over) | 8-3 (Over) | 8-4 (Castle) |

---

## Pipeline Modes

The `--mode` argument defaults to `all` if omitted. You can run the full pipeline with just:
```bash
python run_pipeline.py --level 1-1 --total_timesteps 5000000 --n_envs 24 --use_custom_reward --lr_schedule
```

### Mode Reference

| Mode | Description | Requires Model | Produces |
| :--- | :--- | :--- | :--- |
| `validate` | Confirms the environment initializes and can step without crashing. | No | Pass/fail output. |
| `train` | Runs PPO training on one or more levels. | No (optional `--load_model` to resume) | Model weights, checkpoints, TensorBoard logs. |
| `train_general` | Trains across a curated set of 5 diverse levels (1-1, 1-2, 2-1, 3-1, 4-1). | No (optional `--load_model`) | Same as `train`. |
| `evaluate` | Loads a trained model and records a gameplay video with metadata overlay. | Yes (`--model_path`) | `.mp4` video file. |
| `visualize` | Renders up to 4 checkpoints in a side-by-side grid video. | Yes (`--checkpoint_dir`) | `.mp4` montage video. |
| `plot` | Generates a static reward curve from TensorBoard logs. | No | `.png` plot file. |
| `dashboard` | Launches TensorBoard in the browser. | No | Web UI. |
| `manual` | Human-controlled gameplay via keyboard. Resets on death. Press Escape to quit. | No | None. |
| `compete` | Human plays first, then the AI plays the same level for comparison. | Yes (`--model_path`) | None (visual only). |
| `all` | Runs validate, train, evaluate, visualize, and plot in sequence (default). | No | All artifacts. |

### Run Order (When Executing Modes Separately)

If you choose to run modes individually instead of using `all`, follow this order:

1. **validate** - Confirm your environment works before committing to a long training run.
2. **train** (or **train_general**) - Run PPO training. This produces model weights and TensorBoard logs.
3. **evaluate** - Generate a video of the trained agent playing. Requires the model from step 2.
4. **visualize** - Compare checkpoint performance over time. Requires checkpoints from step 2.
5. **plot** - Generate a reward curve image. Requires TensorBoard logs from step 2.
6. **dashboard** - (Optional) Launch TensorBoard for live monitoring. Can run during step 2.

The `manual` and `compete` modes are standalone and can be run at any time.

### Examples

Validate the environment:
```bash
python run_pipeline.py --mode validate --level 1-1
```

Train on a single level:
```bash
python run_pipeline.py --mode train --level 1-1 --total_timesteps 2000000 --n_envs 4 --use_custom_reward --lr_schedule
```

Resume from a checkpoint:
```bash
python run_pipeline.py --mode train --level 1-1 --load_model runs/1-1_2000000steps_rew_lr/mario_ppo_final.zip --total_timesteps 500000 --use_custom_reward --lr_schedule
```

Train on multiple levels simultaneously:
```bash
python run_pipeline.py --mode train --level 1-1,1-2,2-1,3-1 --total_timesteps 5000000 --n_envs 16 --use_custom_reward --lr_schedule
```

General training across diverse levels:
```bash
python run_pipeline.py --mode train_general --total_timesteps 10000000 --use_custom_reward --lr_schedule
```

Generate a gameplay video with metadata overlay:
```bash
python run_pipeline.py --mode evaluate --model_path runs/1-1_2000000steps_rew_lr/mario_ppo_final.zip
```

Render a side-by-side checkpoint comparison montage:
```bash
python run_pipeline.py --mode visualize --checkpoint_dir runs/1-1_2000000steps_rew_lr/checkpoints
```

Generate a static reward curve:
```bash
python run_pipeline.py --mode plot --level 1-1
```

Launch TensorBoard:
```bash
python run_pipeline.py --mode dashboard --level 1-1
```

Play Mario manually (arrow keys, Space/Up to jump, Shift to sprint, Escape to quit):
```bash
python run_pipeline.py --mode manual
```

Compete against the AI:
```bash
python run_pipeline.py --mode compete --level 1-1 --model_path runs/1-1_2000000steps_rew_lr/mario_ppo_final.zip
```

Run the full pipeline (default when `--mode` is omitted):
```bash
python run_pipeline.py --level 1-1 --total_timesteps 5000000 --n_envs 24 --use_custom_reward --lr_schedule
```

---

## Research Log and Training Results

### Run 1: Single-Level Speedrun (1-1)

**Goal**: Master World 1-1 with high parallelism.

```bash
python run_pipeline.py --mode train --level 1-1 --total_timesteps 5000000 --n_envs 24 --use_custom_reward --lr_schedule
```

* **Status**: Pending
* **Reward Curve**: `runs/<run_id>/reward_curve.png`
* **Evaluation Video**: `runs/<run_id>/evaluation_run.mp4`

| Mean Reward | Episode Length | Success Rate |
| :--- | :--- | :--- |
| -- | -- | -- |

### Run 2: General Intelligence (Multi-Level)

**Goal**: Build a generalized agent across 5 diverse levels.

```bash
python run_pipeline.py --mode train_general --total_timesteps 25000000 --n_envs 20 --use_custom_reward --lr_schedule
```

* **Status**: Pending
* **Reward Curve**: `runs/<run_id>/reward_curve.png`
* **Evaluation Video**: `runs/<run_id>/evaluation_run.mp4`

| Mean Reward | Episode Length | Success Rate |
| :--- | :--- | :--- |
| -- | -- | -- |

### Run 3: World 1 Conqueror (1-1 through 1-4)

**Goal**: Master all four stages of World 1.

```bash
python run_pipeline.py --mode train --level 1-1,1-2,1-3,1-4 --total_timesteps 15000000 --n_envs 16 --use_custom_reward --lr_schedule
```

* **Status**: Pending
* **Reward Curve**: `runs/<run_id>/reward_curve.png`
* **Evaluation Video**: `runs/<run_id>/evaluation_run.mp4`

| Mean Reward | Episode Length | Success Rate |
| :--- | :--- | :--- |
| -- | -- | -- |

### Run 4: Fine-Tuning (Resume from Checkpoint)

**Goal**: Refine an existing model past a learning plateau.

```bash
python run_pipeline.py --mode train --level 1-1 --total_timesteps 5000000 --n_envs 24 --load_model runs/1-1_5000000steps_rew_lr/checkpoints/mario_ppo_agent_5000000_steps.zip --use_custom_reward --lr_schedule
```

* **Status**: Pending
* **Reward Curve**: `runs/<run_id>/reward_curve.png`
* **Evaluation Video**: `runs/<run_id>/evaluation_run.mp4`

| Mean Reward | Episode Length | Success Rate |
| :--- | :--- | :--- |
| -- | -- | -- |

---

## Directory Organization

* `src/`: Python source modules.
* `runs/`: Each experiment generates an isolated folder using the pattern `<level>_<timesteps>steps[_rew][_lr][_resume]`:
    * `tensorboard/`: TensorBoard event logs.
    * `checkpoints/`: Periodic model snapshots.
    * `mario_ppo_final.zip`: Final trained model weights.
    * `evaluation_run.mp4`: Gameplay video with overlay.
    * `progress_montage.mp4`: Side-by-side checkpoint comparison.
    * `reward_curve.png`: Static reward plot.
