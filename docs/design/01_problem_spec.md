# 01 — Problem Specification

## Dimensionality

**Decision: 2D or 3D?**

| Option | Pros | Cons |
|--------|------|------|
| **2D planar** | Simple, fast to train, easy to visualize, good for demonstration | Ignores roll/yaw, lateral drift in one axis only |
| **3D** | Realistic, richer dynamics | Much harder to train, harder to visualize, overkill for a demo |

> **Proposed:** 2D planar (x, y position; theta orientation). Extend to 3D later if desired.

---

## State Space

Proposed continuous state vector:

| Variable | Symbol | Units | Notes |
|----------|--------|-------|-------|
| Horizontal position | x | m | Relative to pad |
| Altitude | y | m | Above ground |
| Horizontal velocity | vx | m/s | |
| Vertical velocity | vy | m/s | Negative = falling |
| Pitch angle | θ | rad | 0 = vertical/nose-up, π/2 = horizontal |
| Angular velocity | ω | rad/s | |
| Remaining propellant mass | m_prop | kg | Normalized to [0, 1]? |

**Open questions:**
- Do we include propellant mass in the state, or treat fuel as unlimited?
- Do we normalize the state vector? If so, how?
- Do we add sensor noise to make it more realistic?

---

## Action Space

**Decision: Continuous or Discrete?**

| Option | Pros | Cons |
|--------|------|------|
| **Continuous** | Realistic, smooth control | Harder to train, requires actor-critic |
| **Discrete** | Simpler, faster to train | Unrealistic quantized thrust |

> **Proposed:** Continuous 2D action space.

| Action | Symbol | Range | Notes |
|--------|--------|-------|-------|
| Throttle | T | [0, 1] | Fraction of max thrust |
| Gimbal angle | δ | [-δ_max, δ_max] | Thrust vector angle relative to vehicle axis |

**Open questions:**
- Should we allow throttle = 0 (engine off)? Or minimum throttle floor (real Raptors have a min throttle)?
- Max gimbal angle — real Raptor is ~±20°. Use that?
- Do we model grid fins as a separate control or fold into gimbal?

---

## Initial Conditions

The episode starts with Starship in free-fall, belly-down (θ ≈ π/2), at some altitude.

**Open questions:**
- Fixed initial state (easier to learn, less robust) or randomized (harder, more general)?
- Suggested starting point: y ≈ 500–2000 m, vy ≈ -80 m/s, θ ≈ π/2, vx ≈ 0

---

## Termination Conditions

| Condition | Terminal? | Notes |
|-----------|-----------|-------|
| y ≤ 0 (ground contact) | Yes | Success or crash depending on velocity/angle |
| \|x\| > x_max (out of bounds) | Yes | Failure |
| θ out of range | Yes? | Tumbling — failure |
| t > t_max | Yes | Timeout — failure |
| m_prop ≤ 0 | Yes | Out of fuel — failure (if modeled) |

---

## Reward Function

**This is the most important design decision.**

A sparse reward (only on landing) is realistic but very hard to learn. A shaped
reward guides learning but can cause unintended behavior.

**Proposed: Dense shaping + terminal bonus**

```
r(t) = - w1 * |x|           # penalize horizontal offset
       - w2 * |vy|           # penalize vertical speed
       - w3 * |vx|           # penalize horizontal speed
       - w4 * |θ|            # penalize deviation from vertical
       - w5 * |T|            # penalize fuel use (optional)
       + R_success           # large bonus on successful landing
       - R_crash             # large penalty on crash
```

**Open questions:**
- What weights? (These will need tuning.)
- Define "successful landing": e.g., |vy| < 2 m/s, |vx| < 1 m/s, |θ| < 10°, |x| < 5 m
- Should we penalize fuel use? (Encourages efficiency but complicates learning.)
- Potential-based shaping to guarantee policy invariance?
