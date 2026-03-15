# StarJAXRL — Design Overview

A demonstration of deep reinforcement learning applied to Starship's belly-flop
landing maneuver, implemented in JAX.

## Design Documents

| Doc | Topic | Status |
|-----|-------|--------|
| [01_problem_spec.md](01_problem_spec.md) | State/action spaces, reward, termination | ✅ Decided |
| [02_simulation.md](02_simulation.md) | Physics model, integration, env interface | ✅ Decided |
| [03_rl_algorithm.md](03_rl_algorithm.md) | Algorithm choice and training loop | ✅ Decided |
| [04_neural_network.md](04_neural_network.md) | Library (NNX), architecture | ✅ Decided |
| [05_training_infra.md](05_training_infra.md) | Vectorization, logging, checkpointing | ✅ Decided |
| [06_todos.md](06_todos.md) | Deferred features and future extensions | 🔄 Living |
| [07_project_plan.md](07_project_plan.md) | Step-by-step build plan with tests | 🔄 Living |

## The Maneuver

Starship's belly-flop is a controlled reentry sequence:

1. **Free-fall** — vehicle falls horizontally, belly-down
2. **Flip** — Raptor engines fire to rotate the vehicle nose-up to vertical
3. **Landing burn** — final deceleration to near-zero velocity above the pad
4. **Catch** — "Mechazilla" chopstick arms catch the vehicle at a target height

The core RL challenge is the **flip + landing burn**: control thrust magnitude
and gimbal angle to arrive at the catch point vertically, with near-zero velocity,
without running out of propellant.
