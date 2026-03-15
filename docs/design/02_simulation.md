# 02 — Environment Simulation

## Physics Model

### Vehicle Parameters (Starship approximations)

| Parameter | Symbol | Value | Notes |
|-----------|--------|-------|-------|
| Dry mass | m_dry | 100,000 kg | Approximate |
| Max propellant | m_prop_max | 1,200,000 kg | Full tank |
| Max thrust (3 sea-level Raptors) | T_max | ~6,000,000 N | ~2 MN each |
| Specific impulse | Isp | 330 s | Sea-level |
| Vehicle length | L | 50 m | For moment of inertia |
| Moment of inertia | I | ~m*L²/12 | Uniform rod approximation |
| Drag coefficient × area | CdA | TBD | Belly-flop area vs nose-first |

**Open questions:**
- How closely do we want to match real Starship numbers? (Demo vs. realism tradeoff)
- Model aerodynamic drag? Important for the belly-flop phase.
- Model varying CdA as a function of angle (high drag belly-down, low drag nose-first)?

### Equations of Motion

2D rigid body under gravity, thrust, and drag:

```
# Forces
F_thrust_x = T * sin(θ + δ)          # thrust in world x
F_thrust_y = T * cos(θ + δ)          # thrust in world y
F_drag_x   = -0.5 * ρ * Cd * A(θ) * vx * |v|
F_drag_y   = -0.5 * ρ * Cd * A(θ) * vy * |v|

# Translational
ax = (F_thrust_x + F_drag_x) / m
ay = (F_thrust_y + F_drag_y) / m - g

# Rotational
τ  = T * sin(δ) * L/2                # torque from gimbaled thrust
α  = τ / I                           # angular acceleration

# Tsiolkovsky (mass flow)
dm/dt = -T / (Isp * g0)
```

---

## Numerical Integration

**Decision: Euler vs. RK4**

| Method | Pros | Cons |
|--------|------|------|
| **Euler** | Simple, fast, JAX-friendly | Less accurate, needs small dt |
| **RK4** | More accurate, stable at larger dt | 4x more function evaluations |

> **Proposed:** Start with Euler at small dt (~0.05 s), option to swap in RK4.

The integrator should be a pure JAX function so it can be `jit`-compiled and
`vmap`-ed over parallel environments.

---

## Environment Interface

**Decision: Gymnasium wrapper or pure JAX environment?**

| Option | Pros | Cons |
|--------|------|------|
| **Gymnasium wrapper** | Familiar API, works with SB3, easy eval | CPU↔GPU copies, breaks JAX pipeline |
| **Pure JAX env** | Fully jit-able, vmap for vectorization, blazing fast | Custom training loop required, less tooling |

> **Proposed:** Pure JAX environment with a thin Gymnasium compatibility shim for
> evaluation/rendering. Training uses the raw JAX env with `vmap` for parallelism.

### Pure JAX env API

```python
# Functional, stateless interface
state = env.reset(key: PRNGKey) -> EnvState
state, obs, reward, done, info = env.step(state, action)
```

`EnvState` is a JAX-compatible dataclass (NamedTuple or `flax.struct`).

---

## Randomization / Domain Randomization

**Open questions:**
- Randomize initial conditions per episode? (Yes — makes policy robust)
- Randomize physics params (mass, drag)? (Optional — for sim-to-real style training)
- Wind gusts as disturbances?
