# Super Mario Bros PPO Agent

This repository contains scripts to train a Reinforcement Learning agent to play Super Mario Bros using Proximal Policy Optimization (PPO).

## Setup Instructions

Python 3.8 or 3.9 is required due to dependencies that are incompatible with newer versions.

### Step 1: Virtual Environment Creation
Create a virtual environment:
```bash
python3 -m venv venv
```

### Step 2: Activation & Dependency Installation
Activate the environment and install dependencies:
```bash
source venv/bin/activate
pip install "pip<24.1" "setuptools==65.5.0" "wheel==0.38.4"
pip install -r requirements.txt
```

---

## Usage (Unified Pipeline)

Use the `run_pipeline.py` script to execute all core tasks. You can specify different levels using the `--level` argument (e.g., `1-1`, `1-2`, `2-1`, `2-2`, etc.).

### 1. Validate Environment
Confirm that the emulator and wrappers are working correctly for a specific level:
```bash
python run_pipeline.py --mode validate --level 1-1
```

### 2. Training
Train the agent on a specific level with parallel environments:
```bash
python run_pipeline.py --mode train --level 1-1 --total_timesteps 2000000 --device cuda --n_envs 4
```
**Resume Training**: Load an existing model and continue training on the same or a different level:
```bash
python run_pipeline.py --mode train --level 1-2 --load_model models/mario_ppo_final.zip --total_timesteps 500000
```

### 3. Evaluation & Visualization
* **Single Render**: Create a video of one model.
  ```bash
  python run_pipeline.py --mode evaluate --model_path models/mario_ppo_final.zip
  ```
* **Progress Montage**: Render a side-by-side comparison of multiple checkpoints from `checkpoint_models/`.
  ```bash
  python run_pipeline.py --mode visualize --video_output videos/progress_montage.mp4
  ```
* **Performance Plot**: Generate a static reward curve from TensorBoard logs.
  ```bash
  python run_pipeline.py --mode plot --output_path logs/reward_curve.png
  ```
* **Dashboard (Web UI)**: Launch TensorBoard to see detailed training analytics.
  ```bash
  python run_pipeline.py --mode dashboard
  ```

### 4. Interactive Tools
* **Manual Play**: Control Mario yourself.
  ```bash
  python run_pipeline.py --mode manual
  ```
* **Competition**: Compete against the trained agent.
  ```bash
  python run_pipeline.py --mode compete
  ```

---

## Directory Organization
* `src/`: Core Python modules and script logic.
* `models/`: Saved model weights (`.zip`).
* `videos/`: Exported evaluation videos.
* `logs/`: Installation and runtime logs.
* `tensorboard_logs/`: Analytical dashboard data.
