# StarJAXRL

Deep reinforcement learning environments and training infrastructure, implemented in [JAX](https://github.com/google/jax).

## Environments

| Environment | Task | Obs | Action |
|---|---|---|---|
| **Starship Landing** | Flip and burn to land on catch arms | 7D | 2D continuous (throttle, gimbal) |
| **CartPole** | Balance a pole on a cart | 4D | 1D continuous (force) |

Both environments share the same stateless functional API and work identically inside the PPO training loop.

## Quick start

**Requirements:** Python 3.12, [uv](https://docs.astral.sh/uv/)

```bash
git clone https://github.com/tymo77/starjaxrl
cd starjaxrl
uv sync
uv pip install -e .
```

### Train Starship

```bash
uv run python train.py                          # default config
uv run python train.py wandb.mode=disabled      # no W&B
uv run python train.py ppo.lr=1e-3 n_updates=500
```

### Train CartPole

```bash
uv run python train.py --config-name cartpole wandb.mode=disabled
```

### Evaluate

```bash
uv run python evaluate.py                                          # best checkpoint
uv run python evaluate.py checkpoint=checkpoints/step_0500
uv run python evaluate.py output=renders/landing.gif
```

### Run tests

```bash
uv run pytest
```

## Key design choices

- **Pure-JAX environments** — `jit`, `vmap`, and `lax.scan` compatible; no Python-side loops during rollout
- **Flax NNX** actor-critic with the `nnx.split` / `nnx.merge` pattern to thread parameters through `lax.scan`
- **Hydra** config management — every hyperparameter is overridable from the CLI
- **Generic training loop** — `init_runner` and `make_train_step` accept env functions as arguments; adding a new environment requires no changes to the PPO code
- **Gaussian reward shaping** — smooth dense signal that provides gradient throughout the episode, not just at success

## Project layout

```
starjaxrl/
├── configs/               # Hydra configs (env, ppo, network, train, cartpole)
├── docs/                  # This site
├── src/starjaxrl/
│   ├── physics/           # Rigid-body dynamics (Starship, CartPole)
│   ├── env/               # Environments + shared utilities
│   │   ├── types.py       # Shared StepInfo
│   │   ├── reward_utils.py # Shared Gaussian reward helper
│   │   ├── starship_env.py
│   │   └── cartpole_env.py
│   ├── agents/            # Actor-Critic networks and PPO agent (Flax NNX)
│   ├── training/          # Generic runner, optimizer, checkpointing, logging
│   └── utils/             # Trajectory visualization
├── tests/                 # 162 pytest tests
├── train.py               # Training entry point
└── evaluate.py            # Evaluation / animation entry point
```
