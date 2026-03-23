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

Use the `run_pipeline.py` script to execute all core tasks:

### 1. Validate Environment
Confirm that the emulator and wrappers are working correctly:
```bash
python run_pipeline.py --mode validate
```

### 2. Training
Train the agent using CLI hyperparameters:
```bash
python run_pipeline.py --mode train --total_timesteps 2000000 --device cuda
```

### 3. Evaluation (Rendering)
Render the agent's performance into a video:
```bash
python run_pipeline.py --mode evaluate --model_path models/mario_ppo_final.zip
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
