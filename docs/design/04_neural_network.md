# 04 — Neural Network Design

## Library: Flax Linen vs. Flax NNX

This is probably the most important JAX-specific decision. Both are from the
Flax team but represent fundamentally different paradigms.

### Flax Linen (legacy API)

```python
class Actor(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256)(x)
        x = nn.tanh(x)
        return x
```

- **Functional style** — parameters are external to the module, passed explicitly
- Requires `model.init(key, x)` to create params, then `model.apply(params, x)`
- Params live in a PyTree dict, separate from the module
- **Stable, battle-tested**, most tutorials and papers use this
- Works naturally with JAX's functional/stateless design philosophy

### Flax NNX (new API, stable since ~2024)

```python
class Actor(nnx.Module):
    def __init__(self, rngs):
        self.linear = nnx.Linear(obs_dim, 256, rngs=rngs)

    def __call__(self, x):
        return nnx.tanh(self.linear(x))
```

- **Object-oriented / stateful style** — parameters live inside the module
- More familiar if coming from PyTorch
- Explicit state management via `nnx.state()` / `nnx.graphdef()`
- Newer, less community examples for RL specifically

### Comparison

| Aspect | Linen | NNX |
|--------|-------|-----|
| Style | Functional | OOP (PyTorch-like) |
| Params location | External PyTree | Inside module |
| `jit` / `vmap` compatibility | Native | Requires `nnx.jit` wrapper |
| Community RL examples | Many (PureJaxRL, CleanRL-JAX) | Few (emerging) |
| Maturity | Stable, widely used | Stable but newer |
| Debugging | Harder (no object state) | Easier (inspect module directly) |

> **Proposed: Flax Linen** — the existing PureJaxRL-style PPO ecosystem uses
> Linen. More reference implementations, better community support for RL.
> NNX is worth a future migration once the ecosystem matures.

**Open question:** Do you have a preference here? If you're more comfortable with
PyTorch-style OOP, NNX might be worth the extra friction.

---

## Network Architecture

### Actor (Policy Network)

Outputs mean of a Gaussian policy over continuous actions.

```
Input: obs (7-dim state vector)
  → Linear(256) → Tanh
  → Linear(256) → Tanh
  → Linear(action_dim)  → mean μ
log_std: learned parameter (state-independent), clipped to [-1, 2]
```

Action sampled as: `a = μ + exp(log_std) * ε`, ε ~ N(0, I)

### Critic (Value Network)

```
Input: obs (7-dim state vector)
  → Linear(256) → Tanh
  → Linear(256) → Tanh
  → Linear(1)            → V(s)
```

### Shared vs. Separate Networks

| Option | Pros | Cons |
|--------|------|------|
| **Separate** actor + critic | Independent learning rates possible, stable | More parameters |
| **Shared trunk** | Feature reuse, fewer params | Gradient interference between actor/critic losses |

> **Proposed: Separate networks.** Simpler to implement and debug, standard for PPO.

**Open questions:**
- Network width: 256 units per layer? 64 is common for simple envs.
- Network depth: 2 hidden layers? 3?
- Activation: Tanh (bounded, common for RL) or ReLU?
- Orthogonal initialization? (Recommended for PPO — empirically better)
