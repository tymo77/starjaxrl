# 07 — Project Plan

Step-by-step build order. Each step ends with working, tested code before moving on.
Tests are written alongside implementation — not after.

---

## Step 1 — Project Skeleton & Config

**Goal:** Importable package, Hydra config wired up, CI-ready test suite.

Tasks:
- `src/starjaxrl/` package structure with `__init__.py` files
- `configs/` directory with Hydra structured configs:
  - `env.yaml` — physics params, initial conditions, termination thresholds
  - `ppo.yaml` — PPO hyperparameters
  - `network.yaml` — layer sizes, activation
  - `train.yaml` — top-level (num updates, num envs, seed, logging)
- `train.py` entry point (Hydra `@hydra.main`)
- `pyproject.toml` — add `wandb`, `hydra-core`, `pytest` dependencies
- `.gitignore` updates (outputs/, wandb/, checkpoints/)

Tests (`tests/test_config.py`):
- Config loads without error
- All required keys present
- Overrides work from CLI

---

## Step 2 — Physics Simulator

**Goal:** A pure JAX function that correctly propagates Starship state forward
one timestep under thrust and gravity.

Tasks:
- `src/starjaxrl/physics/dynamics.py`
  - `StarshipState` (NamedTuple of JAX arrays)
  - `StarshipParams` (frozen config struct)
  - `derivatives(state, action, params) -> StarshipState`
  - `euler_step(state, action, params, dt) -> StarshipState`
- `jit` and `vmap` compatibility verified

Tests (`tests/physics/test_dynamics.py`):
- **Free-fall:** zero thrust → vy decreases at -g, x/θ/ω unchanged
- **Hover:** throttle = T_hover → zero net vertical acceleration
- **Thrust direction:** θ = 0, δ = 0 → pure upward force
- **Gimbal torque:** δ ≠ 0 → non-zero angular acceleration
- **Mass depletion:** propellant decreases monotonically with thrust
- **No NaN:** random actions × random states never produce NaN
- **vmap:** batch of 16 states steps correctly

---

## Step 3 — Environment

**Goal:** A fully JAX-compatible environment with reset, step, and reward.

Tasks:
- `src/starjaxrl/env/starship_env.py`
  - `StarshipEnv` class
  - `reset(key, params) -> EnvState` — initializes at 3km, belly-down
  - `step(state, action, params) -> (EnvState, obs, reward, done, info)`
  - `get_obs(state) -> jnp.ndarray` — extracts observation vector
  - `compute_reward(state, action, params) -> float`
  - `is_done(state, params) -> bool`
  - `is_success(state, params) -> bool`
- Gymnasium compatibility shim (`src/starjaxrl/env/gym_wrapper.py`)

Tests (`tests/env/test_starship_env.py`):
- **Obs shape:** reset returns obs of shape (7,)
- **Step shapes:** step returns correct shapes for all outputs
- **Termination — catch height:** y ≤ y_catch triggers done
- **Termination — out of bounds:** |x| > x_max triggers done
- **Termination — tumbling:** |θ| > θ_max triggers done
- **Termination — fuel:** m_prop = 0 triggers done
- **Success check:** state within tolerances → is_success = True
- **Reward:** dense terms fire on each step; success bonus fires once
- **vmap:** batch reset and step work over N=16 envs

---

## Step 4 — PPO Agent (Networks)

**Goal:** Actor and critic networks that produce valid outputs.

Tasks:
- `src/starjaxrl/agents/networks.py`
  - `Actor(nnx.Module)` — outputs μ, log_std
  - `Critic(nnx.Module)` — outputs V(s)
  - Orthogonal initialization helper
- `src/starjaxrl/agents/ppo.py`
  - `PPOAgent` — holds actor + critic
  - `get_action_and_value(obs, key)` — sample action, compute log_prob and value
  - `evaluate_actions(obs, action)` — log_prob, entropy, value for collected data

Tests (`tests/agents/test_networks.py`):
- **Actor output shape:** (action_dim,) for μ, (action_dim,) for log_std
- **Critic output shape:** scalar
- **log_prob finite:** sampled actions produce finite log-probabilities
- **Entropy positive:** entropy > 0
- **Action bounds:** tanh-squashed actions stay in [-1, 1]
- **vmap:** batch forward pass over obs batch

---

## Step 5 — PPO Training Loop

**Goal:** A single `train_step` that can be `jit`'d and `lax.scan`'d.

Tasks:
- `src/starjaxrl/agents/ppo.py` (continued)
  - `collect_rollout(runner_state)` — `lax.scan` over T steps × N envs
  - `compute_gae(rewards, values, dones, params)` — GAE advantage estimation
  - `ppo_update(runner_state, batch)` — K epochs of minibatched updates
  - `train_step(runner_state, _)` — collect + update, returns metrics
- `src/starjaxrl/training/runner.py`
  - `RunnerState` dataclass
  - `init_runner(config, key)` — initializes everything
  - `train(config)` — `lax.scan` over `train_step` for N updates

Tests (`tests/agents/test_ppo.py`):
- **GAE shapes:** advantages same shape as rewards
- **GAE normalization:** zero mean, unit variance after normalization
- **Loss finite:** policy loss, value loss, entropy all finite after one update
- **Params change:** network params differ before and after one update
- **No NaN:** gradients finite throughout

---

## Step 6 — Logging & Checkpointing

**Goal:** wandb integration and Orbax checkpointing wired into training.

Tasks:
- `src/starjaxrl/training/logging.py`
  - `log_metrics(metrics, step)` — wandb + console
  - `log_trajectory_gif(env, policy, params, step)` — render eval rollout
- `src/starjaxrl/training/checkpoint.py`
  - `save_checkpoint(runner_state, path)`
  - `load_checkpoint(path) -> runner_state`
- `train.py` — wire up wandb init, periodic logging, checkpoint saves

Tests (`tests/training/test_checkpoint.py`):
- **Save/load round-trip:** params identical after save and restore
- **Best checkpoint tracking:** only overwrites when return improves

---

## Step 7 — Visualization

**Goal:** Animated playback of a trained policy rollout.

Tasks:
- `src/starjaxrl/utils/visualization.py`
  - `render_frame(ax, state, action)` — draws vehicle, thrust arrow, pad
  - `render_trajectory(states, actions, params)` — FuncAnimation
  - `save_animation(anim, path)` — gif or mp4
- `evaluate.py` — CLI script: load checkpoint, run eval rollout, save animation

Tests (`tests/utils/test_visualization.py`):
- **Render doesn't crash:** render_frame runs on valid state
- **Animation frame count:** matches trajectory length

---

## Step 8 — End-to-End Integration Test & Polish

**Goal:** Confirm the full pipeline runs, reward improves, and a successful
landing can be demonstrated.

Tasks:
- Integration test: train for 100 updates, assert mean return increases
- Tune reward weights and PPO hyperparameters until landings succeed
- Write a clean `README.md` with setup instructions, training command, example gif
- Final commit and tag `v0.1.0`

Tests (`tests/test_integration.py`):
- **Full train run:** 100 updates, no crash, return trending up
- **Eval rollout:** greedy policy produces a valid trajectory
- **Animation export:** gif saved to disk without error

---

## Running Tests

```bash
# All tests
uv run pytest tests/ -v

# Single module
uv run pytest tests/physics/ -v

# With coverage
uv run pytest tests/ --cov=starjaxrl --cov-report=term-missing
```
