# 04 — Neural Network Design

## Library

**Decision: Flax NNX.**

NNX uses an object-oriented, PyTorch-style API where parameters live inside
the module. More intuitive to write and inspect. We'll use `nnx.jit` and
`nnx.vmap` wrappers for JAX compilation.

```python
class Actor(nnx.Module):
    def __init__(self, obs_dim, action_dim, rngs):
        self.l1 = nnx.Linear(obs_dim, 64, rngs=rngs)
        self.l2 = nnx.Linear(64, 64, rngs=rngs)
        self.mu_head = nnx.Linear(64, action_dim, rngs=rngs)
        self.log_std = nnx.Param(jnp.zeros(action_dim))
```

---

## Architecture

### Actor (Policy Network)

Outputs mean μ of a Gaussian policy over continuous actions.
`log_std` is a learned state-independent parameter vector, clipped to [-1, 2].

```
obs (7) → Linear(64) → Tanh → Linear(64) → Tanh → Linear(action_dim) → μ
log_std: learned parameter, shape (action_dim,)

Action: a = μ + exp(log_std) * ε,  ε ~ N(0, I)
```

### Critic (Value Network)

```
obs (7) → Linear(64) → Tanh → Linear(64) → Tanh → Linear(1) → V(s)
```

### Decisions

| Choice | Decision | Notes |
|--------|----------|-------|
| Width | 64 units | Start small; easy to scale up |
| Depth | 2 hidden layers | Standard for simple continuous control |
| Activation | Tanh | Bounded outputs, standard for RL |
| Initialization | Orthogonal | Empirically better for PPO |
| Actor/critic | Separate networks | Simpler, more stable |

---

## Deferred

See [06_todos.md](06_todos.md):
- Shared trunk architecture (actor and critic share a feature extractor)
- Recurrent policy (LSTM / GRU) for partial observability / sensor noise scenarios
