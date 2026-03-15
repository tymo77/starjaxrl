# starjaxrl

A demonstration of deep reinforcement learning applied to SpaceX's Starship belly-flop landing maneuver, implemented in JAX with Flax NNX.

## Overview

Starship executes a dramatic high-altitude bellyflop and then ignites its engines for a last-second flip-and-burn to land on the catch arms of the launch tower. This project implements that problem as a 2-D reinforcement learning environment and trains a PPO agent to solve it.

**Highlights:**
- Pure-JAX environment — `jit`, `vmap`, and `lax.scan` for fast, hardware-agnostic rollouts
- Flax NNX actor-critic policy network with orthogonal initialization
- Full PPO training loop (GAE-λ, clipped objective, entropy bonus, linear LR annealing)
- Hydra config management — all hyperparameters overridable from the CLI
- W&B logging, numpy checkpoint management, Matplotlib trajectory animation
- 148 unit and integration tests (pytest)

---

## Physics model

| Quantity | Value |
|---|---|
| Vehicle length | 50 m |
| Dry mass | 100 t |
| Propellant | 1 200 t (full load) |
| Max thrust | 6 MN (3 × sea-level Raptor) |
| Specific impulse | 330 s |
| Minimum throttle | 40 % |
| Max gimbal angle | ±20° (0.35 rad) |
| Timestep | 50 ms (Euler) |

The episode starts with the vehicle at 3 000 m altitude, falling at 80 m/s, oriented belly-down (θ = π/2). The agent must flip upright, slow down, and catch within ±1 m of the target at ≤ 2 m/s vertical velocity. Because T/W < 1 at full propellant, the agent must manage fuel carefully and execute a suicide-burn approach.

Termination conditions: catch height reached | out of bounds | tumbling | fuel exhausted | timeout.

---

## Project structure

```
starjaxrl/
├── configs/               # Hydra configs (env, ppo, network, train)
├── docs/design/           # Design documents
├── src/starjaxrl/
│   ├── physics/           # 2-D rigid-body dynamics (euler_step, StarshipState)
│   ├── env/               # JAX environment (reset, step, reward, obs)
│   │   └── gym_wrapper.py # Gymnasium wrapper for evaluation
│   ├── agents/            # Actor-Critic networks and PPO agent (Flax NNX)
│   ├── training/          # Runner, optimizer, checkpoint manager, logging
│   └── utils/             # Trajectory visualization (Matplotlib animation)
├── tests/                 # 148 pytest tests
├── train.py               # Hydra training entry point
└── evaluate.py            # Load checkpoint → render animation
```

---

## Quick start

**Requirements:** Python 3.12, [uv](https://docs.astral.sh/uv/)

```bash
git clone https://github.com/tymo77/starjaxrl
cd starjaxrl
uv sync
uv pip install -e .
```

### Run tests

```bash
uv run pytest
```

### Train

```bash
# Default config (1000 updates, wandb online)
uv run python train.py

# Override hyperparameters from the CLI
uv run python train.py wandb.mode=disabled n_updates=500
uv run python train.py ppo.lr=1e-3 ppo.n_envs=32
```

Checkpoints are saved to `checkpoints/` every 50 updates; the best checkpoint (by greedy eval return) is saved to `checkpoints/best.npz`.

### Evaluate

```bash
# Render a GIF of the best checkpoint
uv run python evaluate.py

# Custom checkpoint / output path
uv run python evaluate.py checkpoint=checkpoints/step_0500 output=renders/landing.gif
```

---

## Key configuration

`configs/env.yaml` — physics and reward:

```yaml
y0: 3000.0          # m — starting altitude
vy0: -80.0          # m/s — initial descent rate
T_max: 6000000.0    # N — max thrust
T_min: 0.4          # minimum throttle fraction (Raptor floor)

# Success tolerances
success_x_tol:     1.0    # m
success_vy_tol:    2.0    # m/s
success_vx_tol:    1.0    # m/s
success_theta_tol: 0.175  # rad (~10 deg)

# Reward weights
w_x:       0.01
w_vy:      0.01
w_vx:      0.01
w_theta:   0.01
R_success: 100.0
```

`configs/ppo.yaml` — algorithm:

```yaml
gamma: 0.99          # discount factor
gae_lambda: 0.95     # GAE lambda
clip_eps: 0.2        # PPO clip ratio
lr: 3.0e-4           # Adam learning rate
lr_anneal: true      # linearly anneal to zero
n_epochs: 10         # update epochs per rollout
n_envs: 16           # parallel environments
rollout_len: 128     # steps per env per update
```

---

## Implementation notes

### NNX + lax.scan

Flax NNX modules are OOP, but `lax.scan` requires functional-style carry. We use the `nnx.split` / `nnx.merge` pattern:

```python
graphdef, agent_state = nnx.split(agent)   # graphdef is static, agent_state is the pytree

def train_step(runner_state, _):
    agent = nnx.merge(graphdef, agent_state)   # reconstruct for rollout
    ...
    _, agent_state = nnx.split(agent)          # re-extract after update
```

`graphdef` is captured as a closure; `agent_state` flows as a JAX pytree through `lax.scan`.

### Observation normalization

The 7-element observation vector is normalized to approximately [-1, 1] before being passed to the network:

```
[x/500, y/3000, vx/100, vy/100, theta/pi, omega/2, m_prop]
```

### Checkpointing

Agent parameters are serialized via `jax.tree_util.tree_flatten` → `np.savez`, avoiding Orbax version fragility. The `CheckpointManager` tracks the best eval return and saves periodic snapshots.

---

## Results (v0.1.0)

Training for 200 updates (~3 min on Apple Silicon CPU, ~51k environment steps):

| Metric | Updates 1-20 avg | Updates 181-200 avg |
|---|---|---|
| Mean step reward | -2.37 | -1.69 |
| Policy entropy | 2.90 | 2.62 |

The policy learns to reduce step-reward by ~29% within 200 updates, with entropy clearly decreasing as the agent commits to structured behaviour. Full landing success requires longer training runs (1000+ updates). This is a hard exploration problem — the suicide-burn maneuver leaves little margin for trial-and-error.

---

## Roadmap / TODOs

See [`docs/design/06_todos.md`](docs/design/06_todos.md) for the full list. Short version:

- [ ] Aerodynamic drag (speed-squared) and flap modeling
- [ ] Initial condition randomization for domain robustness
- [ ] Observation noise
- [ ] Return / advantage normalization (PopArt) for more stable VF training
- [ ] SAC as an alternative to PPO
- [ ] Reward curriculum (gradually tighten tolerances)
- [ ] JAX Metal backend (`jax[metal]`) for Apple Silicon GPU training
- [ ] 3-D extension

---

## License

MIT
