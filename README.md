# Flappy Bird DQN

A compact reinforcement learning project that trains a Deep Q-Network (DQN) to play the `FlappyBird-v0` Gymnasium environment. It uses an experience replay buffer and a periodically synchronized target network, with training settings stored in YAML.

## Features

- DQN policy and target networks implemented in PyTorch
- Experience replay for sampling prior transitions
- Configurable exploration, learning rate, discount factor, and replay settings
- Automatic device selection (Apple Metal, CUDA, or CPU)
- Best-performing model checkpoints and reward logs saved under `runs/`
- Optional rendered evaluation of a trained checkpoint

## Requirements

Python 3.9 or newer is recommended. Install the project dependencies:

```bash
python -m pip install torch gymnasium flappy-bird-gymnasium pyyaml pygame
```

`pygame` is used by the interactive demo in `bird.py`; rendering the Gymnasium environment may also require it, depending on the environment installation.

## Usage

Run commands from the project directory. The parameter set name is the top-level key in `parameters.yaml` (`flappybirdv0` by default).

Train a model:

```bash
python agent.py flappybirdv0 --train
```

Training runs continuously. Stop it with `Ctrl+C`. The best policy checkpoint is written to `runs/flappybirdv0.pt`, and improvements are appended to `runs/flappybirdv0.log`.

Evaluate the saved checkpoint with a rendered game window:

```bash
python agent.py flappybirdv0
```

This loads `runs/flappybirdv0.pt` and renders the agent playing. A checkpoint must exist before evaluation.

To play the environment manually with Space to flap, run:

```bash
python bird.py
```

## Configuration

Edit `parameters.yaml` to adjust the training parameters. The current `flappybirdv0` settings include:

| Parameter | Purpose |
| --- | --- |
| `alpha` | Adam optimizer learning rate |
| `gamma` | Discount factor for future rewards |
| `epsilon_init`, `epsilon_min`, `epsilon_decay` | Exploration schedule |
| `replay_memory_size` | Maximum number of stored transitions |
| `mini_batch_size` | Number of transitions sampled for an update |
| `network_sync_rate` | Steps between target network updates |
| `reward_threshold` | Episode reward limit used by the training loop |

## Project files

| File | Description |
| --- | --- |
| `agent.py` | Training and checkpoint evaluation loop |
| `dqn.py` | PyTorch Q-network definition |
| `experience_replay.py` | Replay memory implementation |
| `bird.py` | Interactive human-play demo |
| `parameters.yaml` | Hyperparameter sets |
| `runs/` | Generated model checkpoint and training log |

## Notes

- Model and log filenames are based on the selected parameter set name.
- Evaluation expects the matching checkpoint under `runs/`.
- Checkpoints and logs are generated artifacts; keep or remove them according to your workflow.
