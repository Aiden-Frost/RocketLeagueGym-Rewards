# RocketLeagueGym-Rewards

## Project Links

- 🌐 [Project Website](https://aiden-frost.github.io/RocketLeagueGym-Rewards/)
- 📑 [Project Slides](https://docs.google.com/presentation/d/1Losh2EeNvpRQ31Fi9mhvghKpchoFJYR-8D2LuFRhyDY/edit?slide=id.g356eb091964_0_317#slide=id.g356eb091964_0_317)

## Setup

Install [Miniconda3](https://www.anaconda.com/docs/getting-started/miniconda/install), then set up your environment:

```bash
conda create --name ddrl-project python=3.10
conda activate ddrl-project
```

On **CUDA CIMS**, use `/scratch` for installing packages and saving checkpoints.  
On **Greene**, use `/scratch/<NET-ID>` for the same.

Ensure the pip you're using points to your conda environment:

```bash
which pip
pip install --cache-dir=/scratch -r requirements.txt
```

## Training

There are two training modes: single GPU and multi-GPU with DDP.

### Environment Variable

Before training, set the **authorize key** as an environment variable inside `train.py` or `train_ddp.py`.

### Training on Different Systems

- **CUDA CIMS**: Use `screen` to run and detach (press `Ctrl+A`, then `D`)
- **Greene**: Use `sbatch` to submit jobs

### Run Training

```bash
cd train

# Single GPU
python train.py

# Multi-GPU DDP
python train_ddp.py
```

For now, prefer using single GPU (`train.py`).  
To use `cuda:1`:

```python
Learner(device="cuda:1")
```

### Before DDP Training

```bash
pip uninstall rlgym_ppo
cd rlgym-ppo
pip install -e .
```

Then run:

```bash
python train_ddp.py
```

## Reward Functions

Reward functions are located in the `rewards` directory.  
To use a specific reward, import it in `train.py` or `train_ddp.py`.

## Inference

### Installing `rlviser` on macOS

```bash

# Install Rust from https://www.rust-lang.org/tools/install

# Clone and build
git clone https://github.com/VirxEC/rlviser.git
cd rlviser
git checkout v0.7.17
rustup install nightly
cargo +nightly build --release -Z build-std=std --target aarch64-apple-darwin

# Copy executable
cp target/aarch64-apple-darwin/release/rlviser ../eval
```

### Visualizing Matches with `rlviser`

Download the weights from the training cluster to your local machine. Change lines 73-74 in `visual_bot_match.py` to the policy weights checkpoint directory
on your local machine.
```bash
cd eval
python visual_bot_match.py
```

### Simulating Matches with `rlgymsim`

Used for simulating matches between two bots to compare performance of bots based on rewards cumulated and goals scored.
```bash
cd eval
python simulate_bot_match.py
```

### Testing reward functions in `rlviser`

You can test the reward functions by playing manually as the **blue agent**.

**Controls:**
- `W`, `A`, `S`, `D`: Movement  
- `Space`: Jump  
- `Left Shift`: Boost  
- `Q`, `E`: Roll  
- `X`: Handbrake  

```bash
cd eval
python human_match_improved.py
```

### Play against a bot

Challenge any of the bots by loading the agent weight and playing.
```bash
cd eval
python human_vs_bot.py
```

## TODO

1. Add metrics to `simulate_bot_match.py` for publishing to Weights & Biases (wandb)
2. (Srivats) Send updated `Learner` script for checkpoint save/load with wandb only
3. Brainstorm and implement new reward functions to improve performance
4. Train using your custom reward
5. Visualize or simulate your model every 500 steps on wandb; adjust reward weights based on insights
