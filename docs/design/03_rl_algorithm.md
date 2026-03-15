# 03 — RL Algorithm

## Algorithm

**Decision: PPO (Proximal Policy Optimization).**

On-policy, actor-critic, well-suited for continuous control. Maps cleanly onto
a JAX `vmap` + `lax.scan` training loop. SAC is in the TODO list as a future
alternative.

---

## PPO Loss

```
L_CLIP = E[min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)]
L_VF   = E[(V(s) - V_target)²]
L_ENT  = H[π]

L = -L_CLIP + c1 * L_VF - c2 * L_ENT
```

---

## Hyperparameters (starting point — all in Hydra config)

| Parameter | Value | Notes |
|-----------|-------|-------|
| γ (discount) | 0.99 | |
| λ (GAE lambda) | 0.95 | |
| ε (clip ratio) | 0.2 | |
| Learning rate | 3e-4 | Adam |
| LR annealing | linear to 0 | Configurable on/off |
| Gradient clip norm | 0.5 | |
| Epochs per update | 10 | Minibatch passes per rollout |
| Minibatch size | 64 | |
| Rollout length | 128 | Steps per env per update |
| Num parallel envs | 16 | Small to start; increase for GPU |
| Entropy coeff c2 | 0.01 | |
| Value coeff c1 | 0.5 | |
| Normalize advantages | True | |

All values are Hydra config parameters.

---

## Training Loop Style

**Decision: PureJaxRL style** — entire rollout collection and parameter update
compiled with `jit`, `vmap` over environments, `lax.scan` over timesteps.

```python
@jax.jit
def train_step(runner_state, _):
    # 1. Collect rollout: lax.scan over T steps, N envs via vmap
    # 2. Compute GAE advantages
    # 3. PPO update: K epochs, minibatched
    return runner_state, metrics

runner_state, metrics = jax.lax.scan(train_step, init_runner_state, None, n_updates)
```

`runner_state` carries: env states, policy/value params, optimizer states, PRNG key.

---

## Configuration

**Decision: Hydra** for config management.

- All hyperparameters (physics, reward weights, PPO, network) in structured configs
- Override from CLI: `python train.py ppo.lr=1e-3 env.y_start=2000`
- Config groups for experiment sweeps

---

## Deferred

See [06_todos.md](06_todos.md):
- SAC implementation
- Reward curriculum (phase out dense shaping)
- Hyperparameter sweeps / tuning
