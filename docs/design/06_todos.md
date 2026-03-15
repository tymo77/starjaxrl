# 06 — TODO / Future Extensions

Items deferred from the initial implementation. Organized by category.

---

## Physics & Simulation

- [ ] **Aerodynamic drag** — angle-dependent CdA (high drag belly-down, low drag nose-first)
- [ ] **Flap / control surface modeling** — separate flap deflection as an action
- [ ] **Variable CdA(θ)** — smooth function of pitch angle
- [ ] **Wind gusts** — stochastic lateral disturbance forces
- [ ] **RK4 integration** — swap in as alternative to Euler; configurable
- [ ] **External simulator integration** — RocketPy, JSBSim, or similar for high-fidelity physics
- [ ] **Engine shutdown modeling** — commanding below T_min cuts engines; model restart
- [ ] **Realistic Starship parameters** — research actual mass, Isp, T_max, gimbal limits

---

## Environment

- [ ] **Domain randomization** — randomize physics params per episode (mass, Isp, etc.)
- [ ] **Observation noise** — add Gaussian noise to sensor readings
- [ ] **Initial condition randomization** — randomize starting altitude, velocity, angle
- [ ] **3D extension** — full 6-DOF dynamics (x, y, z, roll, pitch, yaw)

---

## RL & Training

- [ ] **SAC implementation** — off-policy alternative to PPO
- [ ] **Reward curriculum** — phase out dense shaping terms over training; move toward sparse success-only reward
- [ ] **Difficulty curriculum** — start from easier initial conditions, increase difficulty as agent improves
- [ ] **Hyperparameter sweeps** — wandb sweeps over PPO config
- [ ] **Multi-seed evaluation** — report mean ± std over N seeds

---

## Neural Network

- [ ] **Shared trunk architecture** — actor and critic share a feature extractor
- [ ] **Recurrent policy (LSTM / GRU)** — handle partial observability once observation noise is added
- [ ] **Larger network sweep** — benchmark 64 vs 128 vs 256 width

---

## Infrastructure

- [ ] **GPU training** — test and tune on CUDA GPU
- [ ] **`jax[metal]`** — Apple Silicon GPU acceleration
- [ ] **Parallel env scaling** — profile N=64, 128, 256 on different hardware
- [ ] **Trajectory video export** — high-quality `.mp4` with ffmpeg backend
