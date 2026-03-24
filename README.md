# Super Mario Bros PPO Agent

This repository contains scripts to train a Reinforcement Learning agent to play Super Mario Bros using Proximal Policy Optimization (PPO).

## Setup Instructions

Python 3.8 or 3.9 is required due to dependencies that are incompatible with newer versions.

### Step 1: Virtual Environment Creation
You can use standard `venv` or `mamba` (recommended for faster package resolution).

**Option A (Mamba/Conda) - Recommended**:
```bash
mamba create -n mario_rl python=3.9
mamba activate mario_rl
```

**Option B (Standard venv)**:
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

### Global Command Line Arguments
The pipeline accepts the following arguments across various modes:
*   `--mode`: Required. Choose from: `validate`, `train`, `evaluate`, `visualize`, `plot`, `dashboard`, `train_general`, `manual`, `compete`, `all`.
*   `--level`: The stage to target (e.g., `1-1`, `1-1-v1`). Can accept comma-separated lists for multi-level combinations.
*   `--device`: `cuda` or `cpu` (default: `cuda`).
*   `--total_timesteps`: Number of steps for training (default: 1000000).
*   `--n_envs`: Number of parallel environments (default: 1).
*   `--use_custom_reward`: Enables advanced reward shaping (flag bonuses, death penalties).
*   `--lr_schedule`: Enables linear learning rate annealing over time.
*   `--load_model`: Path to a `.zip` model to resume training from.
*   `--model_path`: Target model for evaluation.
*   `--checkpoint_dir`: Target directory for generating visualization montages.
*   `--video_output`: Destination path for exported `.mp4` video files.

### Unified Execution Mode
To run the entire pipeline autonomously (Validation -> Training -> Evaluation -> Visualization Montage -> Performance Plot), use the `all` mode:
```bash
python run_pipeline.py --mode all --level 1-1 --total_timesteps 1000000 --use_custom_reward --lr_schedule
```

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
python run_pipeline.py --mode train --level 1-2 --load_model runs/1-1_2000000steps/mario_ppo_final.zip --total_timesteps 500000
```
**General Training**: Train on a curated set of diverse levels (1-1, 1-2, 2-1, 3-1, 4-1) to improve agent adaptability:
```bash
python run_pipeline.py --mode train_general --total_timesteps 2000000 --use_custom_reward --lr_schedule
```
**Optimized Training Flags**:
- `--use_custom_reward`: Actives flagpole bonuses and death penalties.
- `--lr_schedule`: Linear learning rate annealing for better convergence.
- `--n_steps 2048`: Massively expands the trajectory horizon (standard GitHub consensus for long 2D levels).
- `--gamma 0.99`: Mathematical discounting factor explicitly tuned for deep long-term planning.
- `--vf_coef 0.5`: Perfectly balances actor-critic variance mathematically.
**Custom Multi-Level**: Specify your own list of levels for simultaneous training:
```bash
python run_pipeline.py --mode train --level 1-1,1-2,2-1,3-1 --total_timesteps 1000000
```

### 3. Evaluation & Visualization
* **Unified Pipeline (All Modes)**: Sequentially executes validation, training, video evaluation, monitoring montage, and plotting autonomously explicitly mapped into a specific `run_id` folder.
  ```bash
  python run_pipeline.py --mode all --level 1-1 --total_timesteps 2000000 --use_custom_reward --lr_schedule
  ```
* **Single Render**: Create a video of one model explicitly from a generated run namespace.
  ```bash
  python run_pipeline.py --mode evaluate --model_path runs/1-1_2000000steps/mario_ppo_final.zip
  ```
* **Progress Montage**: Render a side-by-side comparison of multiple checkpoints dynamically.
  ```bash
  python run_pipeline.py --mode visualize --checkpoint_dir runs/1-1_2000000steps/checkpoints
  ```
* **Performance Plot**: Generate a static reward curve from TensorBoard logs.
  ```bash
  python run_pipeline.py --mode plot --level 1-1
  ```
* **Dashboard (Web UI)**: Launch TensorBoard to see detailed training analytics.
  ```bash
  python run_pipeline.py --mode dashboard --level 1-1
  ```

### 4. Interactive Tools
* **Manual Play**: Control Mario yourself.
  ```bash
  python run_pipeline.py --mode manual
  ```
* **Competition**: Compete against the trained agent natively directly utilizing the `mario_ppo_final` weights specific to the level logic natively.
  ```bash
  ---

## Research Log & Training Results

Use this section to document and initiate your high-performance training runs. Each log entry includes the optimized command for your i9-13900K / RTX 4080 setup.

### 📝 1. Single-Level Speedrun (Mastering 1-1)
*   **Goal**: Achieve expert-level performance on World 1-1 as quickly as possible using high parallelism.
*   **Command**:
    ```bash
    python run_pipeline.py --mode train --level 1-1 --total_timesteps 5000000 --n_envs 24 --use_custom_reward --lr_schedule
    ```
*   **Status**: [Pending / Completed]
*   **Visual Results**:
    - ![Reward Curve](runs/[run_id]/reward_curve.png)
    - ![Evaluation Video](runs/[run_id]/evaluation_run.mp4)
*   **Metrics**:
    | Mean Reward | Episode Length | Success Rate |
    | :--- | :--- | :--- |
    | [Value] | [Value] | [%] |

### 📝 2. General Intelligence Mastery (Multi-Level)
*   **Goal**: Build a robust agent that generalizes across 5 diverse environments (Overworld, Underground, Water, Castle).
*   **Command**:
    ```bash
    python run_pipeline.py --mode train_general --total_timesteps 25000000 --n_envs 20 --use_custom_reward --lr_schedule
    ```
*   **Status**: [Pending / Completed]
*   **Visual Results**:
    - ![Reward Curve](runs/[run_id]/reward_curve.png)
    - ![Evaluation Video](runs/[run_id]/evaluation_run.mp4)
*   **Metrics**:
    | Mean Reward | Episode Length | Success Rate |
    | :--- | :--- | :--- |
    | [Value] | [Value] | [%] |

### 📝 3. World 1 Conqueror (Curated Sequence)
*   **Goal**: Master the entire first world (1-1 through 1-4) in sequence with balanced parallelism.
*   **Command**:
    ```bash
    python run_pipeline.py --mode train --level 1-1,1-2,1-3,1-4 --total_timesteps 15000000 --n_envs 16 --use_custom_reward --lr_schedule
    ```
*   **Status**: [Pending / Completed]
*   **Visual Results**:
    - ![Reward Curve](runs/[run_id]/reward_curve.png)
    - ![Evaluation Video](runs/[run_id]/evaluation_run.mp4)
*   **Metrics**:
    | Mean Reward | Episode Length | Success Rate |
    | :--- | :--- | :--- |
    | [Value] | [Value] | [%] |

### 📝 4. Continuous Fine-Tuning (Resume Mode)
*   **Goal**: Refine an existing model or overcome a learning plateau on a specific level.
*   **Command**:
    ```bash
    python run_pipeline.py --mode train --level 1-1 --total_timesteps 5000000 --n_envs 24 --load_model runs/1-1_5000000steps_rew_lr/checkpoints/mario_ppo_agent_5000000_steps.zip --use_custom_reward --lr_schedule
    ```
*   **Status**: [Pending / Completed]
*   **Visual Results**:
    - ![Reward Curve](runs/[run_id]/reward_curve.png)
    - ![Evaluation Video](runs/[run_id]/evaluation_run.mp4)
*   **Metrics**:
    | Mean Reward | Episode Length | Success Rate |
    | :--- | :--- | :--- |
    | [Value] | [Value] | [%] |

    | [Value] | [Value] | [%] |

---

## Directory Organization
* `src/`: Core Python modules and script logic.
* `runs/`: Dynamic grouping namespaces. Each experiment generates an isolated folder named using the convention `<level>_<timesteps>steps_[rew]_[lr]_[resume]`:
    * `<level>`: The Mario level (e.g., `1-1`) or `general` for mixed training.
    * `<timesteps>`: Total training steps evaluated.
    * `_rew`: Appended if `--use_custom_reward` was active.
    * `_lr`: Appended if `--lr_schedule` was active.
    * `_resume`: Appended if `--load_model` was used to explicitly resume training from older weights.
    
    Inside each specific namespace (e.g., `runs/1-1_2000000steps_rew_lr/`), you will find exactly the outputs constrained to that run:
    * `tensorboard/`: Nested analytical logs actively updated during the training loop.
    * `checkpoints/`: Models systematically dumped at specific intervals natively.
    * `mario_ppo_final.zip`: The ultimate structurally finalized agent architecture weights natively.
    * `evaluation_run.mp4`: A rendered visual tracing mathematical inference natively natively.
    * `progress_montage.mp4`: Comparison videos showing step-by-step intelligence maturation visually explicitly tracking.
    * `reward_curve.png`: Plotted static graphic metrics formally structurally.
