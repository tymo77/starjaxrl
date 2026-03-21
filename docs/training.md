# Training

StarJAXRL uses Proximal Policy Optimization (PPO) with Generalized Advantage Estimation (GAE-λ). The training loop is implemented entirely in JAX and is generic across environments.

## Algorithm

**PPO** with the following components:

- **Clipped surrogate objective** — ratio clipping at ε = 0.2
- **GAE-λ** advantage estimation (γ = 0.99, λ = 0.95)
- **Entropy bonus** — encourages exploration early in training
- **Value function loss** — MSE between predicted and target returns
- **Linear LR annealing** — learning rate decays to zero over all gradient steps
- **Advantage normalization** — per-batch mean/std normalization

## Training loop

```
for update in range(n_updates):
    1. Collect T-step rollout across N parallel environments (lax.scan)
    2. Compute GAE advantages and returns
    3. Flatten (T, N) → (T*N,) batch
    4. For K epochs:
         shuffle batch into M minibatches
         for each minibatch:
             compute PPO loss (policy + value + entropy)
             gradient step via Adam
```

The inner loops (rollout scan, epoch scan, minibatch scan) are all compiled via `jax.jit` + `lax.scan`, so only the outer Python loop touches Python. Logging and checkpointing happen between updates.

## Generic environment interface

`init_runner` and `make_train_step` accept env functions as arguments:

```python
runner_state, graphdef, optimizer = init_runner(
    cfg, key, env_params,
    env_reset=reset,
    env_get_obs=get_obs,
    obs_dim=StarshipEnv.OBS_DIM,
    action_dim=StarshipEnv.ACTION_DIM,
)

train_step = jax.jit(make_train_step(
    graphdef, optimizer, env_params, cfg,
    env_reset=reset, env_get_obs=get_obs, env_step=step,
))
```

Any environment that implements `reset(key, params)`, `get_obs(state)`, and `step(state, action, params)` works with no changes to the PPO code.

## Networks

Both actor and critic use a 2-layer MLP with tanh activations and orthogonal initialization (recommended for PPO):

```
Actor:   [obs_dim] → [64] → [64] → [2 * action_dim]   (outputs μ and log_std)
Critic:  [obs_dim] → [64] → [64] → [1]                 (outputs V(s))
```

`log_std` is a learned state-independent parameter, clipped to [-1, 2].

## Hyperparameters

Full defaults in `configs/ppo/ppo.yaml`:

```yaml
gamma:                0.99
gae_lambda:           0.95
clip_eps:             0.2
lr:                   3.0e-4
lr_anneal:            true
grad_clip:            0.5
n_epochs:             10
minibatch_size:       64
rollout_len:          128
n_envs:               16
entropy_coeff:        0.01
value_coeff:          0.5
normalize_advantages: true
```

Override any value from the CLI:

```bash
uv run python train.py ppo.lr=1e-3 ppo.n_envs=32 ppo.n_epochs=5
```

## Checkpointing

`CheckpointManager` saves:

- **Periodic snapshots** every `checkpoint_every` updates → `checkpoints/step_NNNNNN.npz`
- **Best checkpoint** whenever eval return improves → `checkpoints/best.npz`

Parameters are serialized via `jax.tree_util.tree_flatten` → `np.savez`, avoiding Orbax version fragility.

## Logging

W&B logging is enabled by default. Disable with:

```bash
uv run python train.py wandb.mode=disabled
```

Metrics logged per update: total loss, policy loss, value loss, entropy, mean reward, eval return, eval success rate, curriculum gravity.

## NNX + lax.scan pattern

Flax NNX modules are OOP, but `lax.scan` requires functional-style carry. We use the split/merge pattern:

```python
# Before scan: split into static structure + mutable pytree
graphdef, agent_state = nnx.split(agent)

# Inside scan: reconstruct to call methods
agent = nnx.merge(graphdef, agent_state)
actions, log_probs, values, _ = jax.vmap(
    lambda o, k: agent.get_action_and_value(o, k)
)(obs, keys)

# After update: re-extract state
_, agent_state = nnx.split(agent)
```

`graphdef` is captured as a closure; `agent_state` flows as a JAX pytree.
