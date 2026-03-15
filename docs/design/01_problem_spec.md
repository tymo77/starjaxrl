# 01 — Problem Specification

## Dimensionality

**Decision: 2D planar.** x/y position, θ pitch. Extend to 3D is in the TODO list.

---

## State Space

Continuous state vector (7 elements, no normalization initially):

| Variable | Symbol | Units | Notes |
|----------|--------|-------|-------|
| Horizontal position | x | m | Relative to pad |
| Altitude | y | m | Above ground |
| Horizontal velocity | vx | m/s | |
| Vertical velocity | vy | m/s | Negative = falling |
| Pitch angle | θ | rad | 0 = vertical/nose-up, π/2 = horizontal |
| Angular velocity | ω | rad/s | |
| Remaining propellant fraction | m_prop | [0, 1] | Normalized mass fraction |

Propellant mass is included — it drives the mass change throughout the burn (Tsiolkovsky).
Observation noise and domain randomization are deferred to the TODO list.

---

## Action Space

**Decision: Continuous, 2D.**

| Action | Symbol | Range | Notes |
|--------|--------|-------|-------|
| Throttle | T | [T_min, 1] | Fraction of max thrust; T_min in config (Raptors can't throttle to zero) |
| Gimbal angle | δ | [-δ_max, δ_max] | Thrust vector angle relative to vehicle axis; δ_max in config |

All action limits are configuration parameters. Engine shutdown (throttle below T_min)
and aerodynamic control surfaces (flaps) are deferred to the TODO list.

---

## Initial Conditions

Episode starts with Starship in free-fall, belly-down:

| Variable | Value | Notes |
|----------|-------|-------|
| x | 0 m | Centered on pad |
| y | 3000 m | Starting altitude |
| vx | 0 m/s | No lateral drift initially |
| vy | -80 m/s | Approximate terminal velocity belly-down |
| θ | π/2 rad | Horizontal / belly-down |
| ω | 0 rad/s | |
| m_prop | 1.0 | Full propellant |

Initial condition randomization is deferred to the TODO list.

---

## Termination Conditions

| Condition | Result | Notes |
|-----------|--------|-------|
| y ≤ y_catch | Terminal | Success if within tolerances, else crash |
| \|x\| > x_max | Crash | Out of bounds |
| m_prop ≤ 0 | Crash | Out of fuel |
| t > t_max | Crash | Timeout |
| \|θ\| > θ_max (e.g. 3π/2) | Crash | Tumbling / uncontrolled rotation |

`y_catch`, `x_max`, `t_max`, `θ_max` are all configuration parameters.
Landing is on the catch arms — we target a specific height, not y = 0.

---

## Success Criteria

Landing is successful if all of the following hold at termination (y ≤ y_catch):

| Condition | Tolerance | Notes |
|-----------|-----------|-------|
| Horizontal offset | \|x\| ≤ 1 m | Catch arm width |
| Vertical velocity | \|vy\| ≤ 2 m/s | Soft catch |
| Horizontal velocity | \|vx\| ≤ 1 m/s | |
| Pitch angle | \|θ\| ≤ 10° | Near-vertical |

Tolerances are configuration parameters.

---

## Reward Function

**Decision: Dense shaping now, sparse success bonus as primary signal.**

```
r(t) = - w_x  * |x|        # penalize horizontal offset
       - w_vy * |vy|        # penalize vertical speed
       - w_vx * |vx|        # penalize horizontal speed
       - w_θ  * |θ|         # penalize deviation from vertical

On success (terminal, within tolerances):
       + R_success           # large sparse bonus

On crash (terminal, outside tolerances):
       (no extra penalty — absence of R_success is sufficient signal)
```

- Weights `w_x`, `w_vy`, `w_vx`, `w_θ`, and `R_success` are configuration parameters.
- Dense shaping is kept initially to guide early learning.
- Curriculum to phase out dense terms and move toward pure sparse reward is in the TODO list.
- Fuel penalty is deferred — fuel use is implicitly penalized by limiting propellant.
