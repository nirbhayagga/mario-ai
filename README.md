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
| `--mode` | `all` | `validate`, `train`, `evaluate`, `visualize`, `plot`, `dashboard`, `manual`, `all` |
| `--level` | `1-1` | Target level. Supports comma-separated lists (e.g. `1-1,1-2,2-1`). |
| `--device` | `cuda` | `cuda` or `cpu`. |
| `--action_space` | `simple` | Movement space: `simple` (7 actions), `right_only` (5 actions - recommended for speed), or `complex` (12 actions). |
| `--total_timesteps` | `2000000` | Total training steps. |
| `--n_envs` | `4` | Parallel environments. For multi-level, levels cycle across envs. |
| `--no_custom_reward` | off | Disables stuck penalty, early termination, and reward clipping. Custom rewards are ON by default. |
| `--no_lr_schedule` | off | Disables linear learning rate decay. LR schedule is ON by default. |
| `--load_model` | none | Path to a `.zip` model to resume from. All CLI hyperparameters (LR, clip_range, etc.) are applied on resume. |
| `--model_path` | none | Model for evaluation. |
| `--checkpoint_dir` | none | Directory containing checkpoints for montage visualization. |
| `--video_output` | none | Destination path for exported `.mp4` video files. |
| `--save_freq` | auto | Checkpoint save frequency. Auto-computed for ~15 checkpoints if not set. |
| `--n_epochs` | `10` | Number of PPO optimization epochs per rollout. |
| `--max_steps` | `3000` | Maximum evaluation steps (stops earlier on level clear). |

### PPO Hyperparameters

These defaults are based on community benchmarks from GitHub and Reddit for Super Mario Bros:

| Parameter | Default | Rationale |
| :--- | :--- | :--- |
| `--n_steps` | `512` | Shorter rollouts converge faster for Mario. Longer buffers (2048) dilute the learning signal. |
| `--gamma` | `0.99` | High discount factor prioritizes reaching the flagpole over grabbing local coins. |
| `--batch_size` | `256` | Larger batches stabilize gradients with the clipped reward range. |
| `--ent_coef` | `0.01` | Entropy bonus encourages exploration and prevents the agent from getting stuck in loops. |
| `--vf_coef` | `0.5` | Balanced actor-critic weighting, standard for PPO. |
| `max_grad_norm` | `0.5` | Gradient clipping prevents catastrophic policy updates during long runs. |
| `--clip_range` | `0.1` | Tighter clipping than the SB3 default (0.2) prevents catastrophic policy swings. |
| `--n_epochs` | `10` | Sufficient optimization passes per batch without overfitting to the rollout. |
| `--learning_rate` | `2.5e-4` | SB3 default. Paired with linear decay (disable with `--no_lr_schedule`). |

---

## Recommended Configurations

### Single-Level Training (e.g. mastering 1-1)

```bash
python run_pipeline.py --mode train --level 1-1 \
    --total_timesteps 5000000 --n_envs 8
```

* **Timesteps**: 2-5M for World 1 levels, 5-10M for harder levels (castles, underwater).
* **Envs**: 4-8 on a standard GPU, up to 24 if you have the VRAM/CPU cores.
* **Reward/LR**: Custom rewards and LR schedules are enabled by default for optimal convergence.
* **Action Space**: Use `--action_space right_only` to reduce the action space from 7 to 5, significantly speeding up training by preventing Mario from walking backward.
* **Checkpoints**: Auto-computed for ~15 checkpoints per run. Override with `--save_freq`.

### Multi-Level Training (e.g. World 1)

```bash
python run_pipeline.py --mode train --level 1-1,1-2,1-3,1-4 \
    --total_timesteps 15000000 --n_envs 16
```

* **Timesteps**: ~3-5M per level in the set (so 4 levels = 12-20M).
* **Envs**: Use at least 4x the number of levels (e.g. 16 envs for 4 levels). Levels cycle evenly across envs.
* **Entropy**: Consider `--ent_coef 0.02` (pass via extra args) for more exploration across diverse levels.

### Fine-Tuning from Checkpoint

```bash
python run_pipeline.py --mode train --level 1-1 \
    --load_model runs/1-1_5000000steps_rew_lr/mario_ppo_final.zip \
    --total_timesteps 2000000 --n_envs 8 \
    --learning_rate 5e-5
```

* Use a lower learning rate (5e-5 or 2.5e-5) to refine without destroying learned behavior.
* The linear LR schedule applies automatically on resume (disable with `--no_lr_schedule`).

### Which Map Version to Use

Each level has 4 visual variants (`-v0` through `-v3`). Append to the level (e.g. `--level 1-1-v0`).

| Version | Description | Use For |
| :--- | :--- | :--- |
| `-v0` | Original NES graphics | **Training and video output (recommended)**. Full-detail sprites give the CNN the most information. |
| `-v1` | Downsampled sprites | Research only. Loses detail without clear benefit since we already resize to 84x84. |
| `-v2` | Pixel-only, black background | Research only. Removes background but also removes useful spatial cues. |
| `-v3` | Sprites only, no background | Research only. Cleaner signal but agents trained on v3 don't transfer to v0. |

**TL;DR**: Always use `-v0` (the default). It gives the best training signal and the best-looking evaluation videos.

---

## Pipeline Modes

### Mode Reference

| Mode | Description | Requires Model | Produces |
| :--- | :--- | :--- | :--- |
| `validate` | Confirms the environment initializes and can step without crashing. | No | Pass/fail output. |
| `train` | Runs PPO training on one or more levels. | No (optional `--load_model` to resume) | Model weights, checkpoints, TensorBoard logs. |
| `evaluate` | Loads a trained model and records a gameplay video with metadata overlay. Runs until level clear or `--max_steps`. | Yes (`--model_path`) | `.mp4` video file. |
| `visualize` | Renders up to 4 checkpoints in a side-by-side grid video. | Yes (`--checkpoint_dir`) | `.mp4` montage video. |
| `plot` | Generates a static reward curve from TensorBoard logs. | No | `.png` plot file. |
| `dashboard` | Launches TensorBoard in the browser. | No | Web UI. |
| `manual` | Human-controlled gameplay via keyboard. Resets on death. Press Escape to quit. | No | None. |
| `all` | Runs validate, train, evaluate, visualize, and plot in sequence (default). | No | All artifacts. |

### Run Order (When Executing Modes Separately)

If you choose to run modes individually instead of using `all`, follow this order:

1. **validate** - Confirm your environment works before committing to a long training run.
2. **train** - Run PPO training. This produces model weights and TensorBoard logs.
3. **evaluate** - Generate a video of the trained agent playing. Requires the model from step 2.
4. **visualize** - Compare checkpoint performance over time. Requires checkpoints from step 2.
5. **plot** - Generate a reward curve image. Requires TensorBoard logs from step 2.
6. **dashboard** - (Optional) Launch TensorBoard for live monitoring. Can run during step 2.

The `manual` mode is standalone and can be run at any time.

### Examples

Validate the environment:
```bash
python run_pipeline.py --mode validate --level 1-1
```

Train on a single level:
```bash
python run_pipeline.py --mode train --level 1-1 --total_timesteps 5000000 --n_envs 8
```

Resume from a checkpoint:
```bash
python run_pipeline.py --mode train --level 1-1 --load_model runs/1-1_5000000steps_rew_lr/mario_ppo_final.zip --total_timesteps 2000000
```

Train on multiple levels simultaneously:
```bash
python run_pipeline.py --mode train --level 1-1,1-2,2-1,3-1 --total_timesteps 15000000 --n_envs 16
```

Generate a gameplay video with metadata overlay:
```bash
python run_pipeline.py --mode evaluate --model_path runs/1-1_5000000steps_rew_lr/mario_ppo_final.zip
```

Render a side-by-side checkpoint comparison montage:
```bash
python run_pipeline.py --mode visualize --checkpoint_dir runs/1-1_5000000steps_rew_lr/checkpoints
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

Run the full pipeline (default when `--mode` is omitted):
```bash
python run_pipeline.py --level 1-1 --total_timesteps 5000000 --n_envs 8
```

---

## Stage Versions

Each level has 4 visual variations (append `-v0` through `-v3` to the level):

| Version | Description |
| :--- | :--- |
| `-v0` | Original NES graphics (default, recommended for both training and video). |
| `-v1` | Downsampled sprites. |
| `-v2` | Pixel-only, black background. |
| `-v3` | Sprites only, no background. |

## Level Types

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

## Directory Organization

* `src/`: Python source modules.
* `runs/`: Each experiment generates an isolated folder using the pattern `<level>_<timesteps>steps[_rew][_lr][_resume]`:
    * `tensorboard/`: TensorBoard event logs.
    * `checkpoints/`: Periodic model snapshots.
    * `eval_logs/`: Evaluation results from `EvalCallback`.
    * `best_model.zip`: Best-performing model weights (selected by periodic evaluation, protects against catastrophic forgetting).
    * `mario_ppo_final.zip`: Final trained model weights (may be worse than best_model due to policy collapse).
    * `evaluation_run.mp4`: Gameplay video with overlay.
    * `progress_montage.mp4`: Side-by-side checkpoint comparison.
    * `reward_curve.png`: Static reward plot.
