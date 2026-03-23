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

Use the `run_pipeline.py` script to execute all core tasks. You can specify different levels using the `--level` argument (e.g., `1-1`, `1-2`, `2-1`).

### Stage Versions
Each level has 4 visual variations which can be targeted by appending `-v0` to `-v3` to the level (defaults to `-v0`):
*   **`-v0`**: Original NES graphics.
*   **`-v1`**: No background (removes clouds/scenery).
*   **`-v2`**: Minimalist (black background, essentials only).
*   **`-v3`**: Transparent world (only sprites like Mario/enemies).

### Level Types
The 32 levels fall into four distinct environmental categories, each requiring different AI skills:
*   **Overworld (Standard)**: Classic platforming (e.g., 1-1, 2-1). focuses on jumping and mobility.
*   **Underground**: Ceiling-constrained stages (e.g., 1-2, 4-2). requires tighter movement control.
*   **Underwater**: Complete physics shift (e.g., 2-2, 7-2). AI must learn to "swim" instead of jump.
*   **Castle (Fortress)**: Hazards like firebars and lava (e.g., 1-4, 8-4). tests timing and hazard avoidance.

### Level Reference Table
| World | 1 (Overworld) | 2 (Underground/Water) | 3 (Overworld/Other) | 4 (Castle) |
| :--- | :--- | :--- | :--- | :--- |
| **World 1** | 1-1 (Over) | 1-2 (Under) | 1-3 (Over) | 1-4 (Castle) |
| **World 2** | 2-1 (Over) | 2-2 (Water) | 2-3 (Over) | 2-4 (Castle) |
| **World 3** | 3-1 (Over) | 3-2 (Over) | 3-3 (Over) | 3-4 (Castle) |
| **World 4** | 4-1 (Over) | 4-2 (Under) | 4-3 (Over) | 4-4 (Castle) |
| **World 5** | 5-1 (Over) | 5-2 (Over) | 5-3 (Over) | 5-4 (Castle) |
| **World 6** | 6-1 (Over) | 6-2 (Under) | 6-3 (Over) | 6-4 (Castle) |
| **World 7** | 7-1 (Over) | 7-2 (Water) | 7-3 (Over) | 7-4 (Castle) |
| **World 8** | 8-1 (Over) | 8-2 (Over) | 8-3 (Over) | 8-4 (Castle) |

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
**General Training**: Train on a curated set of diverse levels (1-1, 1-2, 2-1, 3-1, 4-1) to improve agent adaptability:
```bash
python run_pipeline.py --mode train_general --total_timesteps 2000000 --use_custom_reward --lr_schedule
```
**Optimized Training Flags**:
- `--use_custom_reward`: Actives flagpole bonuses and death penalties.
- `--lr_schedule`: Linear learning rate annealing for better convergence.
**Custom Multi-Level**: Specify your own list of levels for simultaneous training:
```bash
python run_pipeline.py --mode train --level 1-1,1-2,2-1,3-1 --total_timesteps 1000000
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
