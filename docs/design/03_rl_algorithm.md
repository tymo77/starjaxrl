# 03 — RL Algorithm

## Algorithm Choice

| Algorithm | Type | Pros | Cons |
|-----------|------|------|------|
| **PPO** | On-policy, actor-critic | Stable, well-understood, good for continuous control | Sample-inefficient |
| **SAC** | Off-policy, actor-critic | Sample-efficient, handles continuous actions well | More complex, replay buffer memory |
| **TD3** | Off-policy | Stable, less hyperparameter-sensitive than SAC | Similar complexity to SAC |
| **DDPG** | Off-policy | Simple baseline | Less stable than TD3/SAC |

> **Proposed: PPO** — it's the de-facto standard for continuous control demos,
> stable to train, and maps cleanly onto a JAX `vmap` + `jit` training loop.
> Pure-JAX PPO (à la CleanRL or PureJaxRL style) is an excellent fit.

---

## PPO Specifics

### Clipped Objective
```
L_CLIP = E[min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)]
```

### Value Loss
```
L_VF = E[(V(s) - V_target)²]
```

### Entropy Bonus
```
L_ENT = -β * H[π]
```

### Combined Loss
```
L = -L_CLIP + c1 * L_VF - c2 * L_ENT
```

### Key Hyperparameters (starting point)

| Param | Value | Notes |
|-------|-------|-------|
| γ (discount) | 0.99 | |
| λ (GAE lambda) | 0.95 | |
| ε (clip ratio) | 0.2 | |
| Learning rate | 3e-4 | Adam |
| Epochs per update | 10 | Minibatch passes over collected data |
| Minibatch size | 64 | |
| Rollout length | 128 | Steps per env before update |
| Num envs (vmap) | 64–256 | Parallel environments |
| Entropy coeff c2 | 0.01 | |
| Value coeff c1 | 0.5 | |

**Open questions:**
- Linear LR annealing over training? (Common in PPO, often helps)
- Gradient clipping (max norm)? (Typically 0.5)
- Normalize advantages? (Yes, almost always)
- Normalize observations? Running mean/std? (Recommended)

---

## Training Loop Style

**Decision: PureJaxRL style vs. standard Python loop**

| Style | Description | Pros | Cons |
|-------|-------------|------|------|
| **PureJaxRL** | Entire rollout + update compiled with `jit`, `vmap`, `lax.scan` | Extremely fast, no Python overhead | Harder to debug, less flexible |
| **Python loop + JAX steps** | Python controls episodes, JAX computes steps/updates | Easier to debug, flexible logging | Slower (Python overhead per step) |

> **Proposed: PureJaxRL style** — the whole point of JAX is speed. Use `lax.scan`
> over timesteps, `vmap` over environments, and `jit` the entire update.
> This lets us train thousands of episodes in seconds on CPU/GPU/TPU.

### Rough Training Loop Structure

```python
@jax.jit
def train_step(runner_state):
    # 1. Collect rollout with lax.scan over T timesteps, N envs via vmap
    # 2. Compute GAE advantages
    # 3. Update actor/critic for K epochs with minibatching
    return runner_state, metrics

runner_state, metrics = jax.lax.scan(train_step, init_runner_state, None, n_updates)
```
