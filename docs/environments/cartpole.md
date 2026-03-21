# CartPole Balancing

A classic control problem: keep a pole balanced upright on a moving cart by applying horizontal forces. The continuous-action formulation uses the same Gaussian reward shaping and PPO training loop as the Starship environment.

## Physics model

| Parameter | Default |
|---|---|
| Cart mass | 1.0 kg |
| Pole mass | 0.1 kg |
| Half-pole length | 0.5 m |
| Max force | ±10 N |
| Timestep | 20 ms (forward Euler) |

Equations of motion from the Lagrangian:

```
θ̈ = (g·sin θ − cos θ · (F + m_p·l·θ̇²·sin θ) / M) / (l · (4/3 − m_p·cos²θ / M))
ẍ = (F + m_p·l·(θ̇²·sin θ − θ̈·cos θ)) / M
```

where M = m_cart + m_pole.

## Observation space (4D)

| Index | Quantity | Normalization |
|---|---|---|
| 0 | x — cart position | / 2.4 m |
| 1 | ẋ — cart velocity | / 5 m/s |
| 2 | θ — pole angle from vertical | / 0.2094 rad |
| 3 | θ̇ — angular velocity | / 5 rad/s |

## Action space (1D continuous)

| Index | Quantity | Range |
|---|---|---|
| 0 | F — horizontal force | [−F_max, +F_max] |

## Reward

```
r(t) = w_θ · gauss(θ, σ_θ) + w_x · gauss(x, σ_x) − w_time
     + R_success   if episode ends by timeout (pole survived full episode)
```

| Parameter | Default |
|---|---|
| w_θ | 2.0 |
| σ_θ | 0.1 rad |
| w_x | 0.5 |
| σ_x | 1.0 m |
| w_time | 0.01 |
| R_success | 100 |

## Termination

| Condition | Description |
|---|---|
| Pole falls | \|θ\| ≥ 0.2094 rad (~12°) |
| Out of bounds | \|x\| ≥ 2.4 m |
| Timeout | t ≥ 10 s (500 steps) |

**Success** is defined as surviving the full 10 s episode (timeout). The sparse bonus rewards the agent for keeping the pole upright throughout.

## Training

```bash
uv run python train.py --config-name cartpole wandb.mode=disabled
```

CartPole uses a fixed gravity (no curriculum) since it converges in ~500 updates. Configure in `configs/cartpole.yaml`:

```yaml
curriculum:
  g_start: 9.81   # no ramping
  g_updates: 1
```
