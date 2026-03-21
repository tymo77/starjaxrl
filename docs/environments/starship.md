# Starship Landing

A 2D simulation of SpaceX's Starship belly-flop landing maneuver. The agent must flip the vehicle from horizontal to vertical, execute a suicide burn, and catch on the tower arms within tight tolerances — all without running out of propellant.

## The maneuver

1. **Free-fall** — vehicle falls belly-down at 80 m/s from 3 000 m altitude
2. **Flip** — fire engines to rotate nose-up (θ = π/2 → 0)
3. **Landing burn** — decelerate to near-zero velocity above the catch height
4. **Catch** — arrive within ±1 m lateral, ≤ 2 m/s vertical velocity at 50 m altitude

The core difficulty: thrust-to-weight ratio is below 1 at full propellant load, so the agent cannot hover and must execute a precisely timed burn.

## Physics model

| Parameter | Value |
|---|---|
| Vehicle length | 50 m |
| Dry mass | 100 t |
| Max propellant | 12 t |
| Max thrust | 6 MN (3 × sea-level Raptor) |
| Specific impulse | 330 s |
| Minimum throttle | 40 % |
| Max gimbal angle | ±20° (0.35 rad) |
| Timestep | 50 ms (forward Euler) |

Equations of motion are 2D rigid-body dynamics with gimbaled thrust:

```
F_x = T · sin(θ + δ)
F_y = T · cos(θ + δ)
τ   = T · sin(δ) · L/2      # torque about CoM
ṁ   = -T / (Isp · g · m_prop_max)
```

## Observation space (7D)

| Index | Quantity | Normalization |
|---|---|---|
| 0 | x — lateral position | / 500 m |
| 1 | y — altitude | / 3 000 m |
| 2 | vx — lateral velocity | / 100 m/s |
| 3 | vy — vertical velocity | / 100 m/s |
| 4 | θ — pitch angle | / π |
| 5 | ω — angular velocity | / 2 rad/s |
| 6 | m_prop — propellant fraction | [0, 1] |

## Action space (2D continuous)

| Index | Quantity | Range |
|---|---|---|
| 0 | throttle | [0, 1] (engine off below T_min = 0.4) |
| 1 | gimbal angle δ | [−0.35, +0.35] rad |

## Reward

Dense Gaussian shaping plus a sparse success bonus:

```
r(t) = gauss(x, σ_x) · gauss(y, 10·σ_x) · gauss(vy, σ_vy) · gauss(vx, σ_vx) · gauss(θ, σ_θ)
     + R_success   if done and within all tolerances
```

| Parameter | Default |
|---|---|
| σ_x | 50 m |
| σ_vy | 30 m/s |
| σ_vx | 10 m/s |
| σ_θ | 0.5 rad |
| R_success | 100 |

## Termination

| Condition | Description |
|---|---|
| Catch height | y ≤ 50 m |
| Out of bounds | \|x\| > 500 m |
| Tumbling | \|θ\| > 3π/2 |
| Fuel exhausted | m_prop ≤ 0 |
| Timeout | t ≥ 120 s |

**Success** requires landing within all tolerances simultaneously: \|x\| ≤ 1 m, \|vy\| ≤ 2 m/s, \|vx\| ≤ 1 m/s, \|θ\| ≤ 0.175 rad.

## Gravity curriculum

Training starts with g = 2.0 m/s² (making T/W > 1 early, easier to learn) and linearly ramps to 9.81 m/s² over 8 000 updates. Configure in `configs/train.yaml`:

```yaml
curriculum:
  g_start: 2.0
  g_updates: 8000
```
