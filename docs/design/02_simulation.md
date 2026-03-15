# 02 — Environment Simulation

## Physics Model

All parameters are configuration values (Hydra config). Starting values below
are nominal approximations — research into real Starship numbers is a TODO.

### Vehicle Parameters

| Parameter | Symbol | Nominal Value | Notes |
|-----------|--------|--------------|-------|
| Dry mass | m_dry | 100,000 kg | |
| Max propellant mass | m_prop_max | 1,200,000 kg | Full tank |
| Max thrust (3 sea-level Raptors) | T_max | 6,000,000 N | ~2 MN each |
| Specific impulse | Isp | 330 s | Sea-level |
| Min throttle fraction | T_min | 0.4 | Raptors can't throttle below ~40% |
| Max gimbal angle | δ_max | 0.35 rad | ~20° |
| Vehicle length | L | 50 m | For moment of inertia |
| Moment of inertia | I | m_dry * L² / 12 | Uniform rod approximation |
| Gravitational acceleration | g | 9.81 m/s² | |

### Equations of Motion

2D rigid body under gravity and thrust (no aerodynamic drag initially):

```
# Total mass
m = m_dry + m_prop

# Thrust vector in world frame (θ = 0 is vertical/nose-up)
F_thrust_x = T * T_max * sin(θ + δ)
F_thrust_y = T * T_max * cos(θ + δ)

# Translational accelerations
ax = F_thrust_x / m
ay = F_thrust_y / m - g

# Rotational
τ = T * T_max * sin(δ) * (L / 2)    # torque from gimbaled thrust
α = τ / I                             # angular acceleration

# Propellant consumption (Tsiolkovsky)
dm_prop/dt = -(T * T_max) / (Isp * g)
```

Aerodynamic drag (variable CdA as a function of θ) and flap modeling are
deferred to the TODO list.

---

## Numerical Integration

**Decision: Euler, dt = 0.05 s.** RK4 and integration with external simulators
are in the TODO list.

```python
# One Euler step — pure JAX, jit/vmap compatible
def euler_step(state, action, params, dt):
    derivs = compute_derivatives(state, action, params)
    return jax.tree.map(lambda s, d: s + d * dt, state, derivs)
```

---

## Environment Interface

**Decision: Pure JAX environment, stateless functional API.**
A thin Gymnasium compatibility shim is added for evaluation/rendering only.

```python
# Functional, stateless — fully jit/vmap compatible
state: EnvState = env.reset(key: PRNGKey)
state, obs, reward, done, info = env.step(state, action, params)
```

`EnvState` is a Flax-compatible struct (NamedTuple or `flax.struct`) holding all
simulation state as JAX arrays.

`StarshipParams` is a separate frozen config struct holding all physics and reward
parameters, passed explicitly to allow `vmap` over parameter distributions later.

---

## Deferred Features

See [06_todos.md](06_todos.md):
- Aerodynamic drag (angle-dependent CdA)
- Flap / control surface modeling
- Wind gusts and disturbances
- Domain randomization over physics parameters
- Observation noise
- Initial condition randomization
- RK4 / higher-order integration
- External simulator integration (e.g. JSBSim, RocketPy)
