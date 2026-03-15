# StarJAXRL — Design Overview

A demonstration of deep reinforcement learning applied to Starship's belly-flop
landing maneuver, implemented in JAX.

## Open Design Documents

| Doc | Topic | Status |
|-----|-------|--------|
| [01_problem_spec.md](01_problem_spec.md) | State/action spaces, reward, termination | 🔴 Open |
| [02_simulation.md](02_simulation.md) | Physics model, integration, env interface | 🔴 Open |
| [03_rl_algorithm.md](03_rl_algorithm.md) | Algorithm choice and training loop | 🔴 Open |
| [04_neural_network.md](04_neural_network.md) | Library (Linen vs NNX), architecture | 🔴 Open |
| [05_training_infra.md](05_training_infra.md) | Vectorization, logging, checkpointing | 🔴 Open |

## The Maneuver (Shared Reference)

Starship's belly-flop is a controlled reentry sequence:

1. **Free-fall** — vehicle falls horizontally, belly-down, using aerodynamic drag
2. **Flip** — Raptor engines fire to rotate the vehicle nose-up to vertical
3. **Boost-back burn** — engines slow descent and correct trajectory
4. **Landing burn** — final deceleration to near-zero velocity at pad
5. **Catch** — "Mechazilla" chopstick arms catch the vehicle (optional in sim)

The core RL challenge is the **flip + landing burn**: control thrust and gimbal
to arrive at the pad vertically, with near-zero velocity, without running out of
propellant.
