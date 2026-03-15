# 05 — Training Infrastructure

## Vectorized Environments

**Decision: `vmap` over N parallel environments, small N to start (16).**

```python
batch_step = jax.vmap(env.step, in_axes=(0, 0))
```

Increase N when moving to GPU. Curriculum over environment difficulty is in the TODO list.

---

## Experiment Logging

**Decision: Weights & Biases (wandb).**

- Free account, rich UI, easy to share run comparisons
- Log: episodic return, success rate, value loss, policy loss, entropy, LR
- Log trajectory animations periodically (eval rollouts rendered to gif)

---

## Checkpointing

**Decision: Orbax** (ships with Flax).

- Save checkpoint every N updates
- Keep best checkpoint by mean episodic return
- Save full runner state: network params, optimizer state, step count

```python
checkpointer = orbax.checkpoint.PyTreeCheckpointer()
checkpointer.save(path, nnx.state(runner_state))
```

---

## Hardware

**Decision: Apple Silicon Mac (CPU/Metal).**

Code is hardware-agnostic — JAX selects the backend automatically.
`jax[metal]` plugin for Apple GPU acceleration is optional but recommended.

---

## Evaluation & Rendering

Separate from training. Run the greedy (deterministic) policy for one episode,
record the trajectory, animate with Matplotlib.

- Vehicle: rectangle rotated to θ
- Thrust vector: arrow from engine nozzle
- Trajectory: fading line of (x, y) history
- HUD: altitude, speed, fuel remaining
- Export: `.gif` for sharing, `.mp4` for quality

---

## Testing

**Decision: pytest.**

Tests live in `tests/` mirroring `src/starjaxrl/`. Every module gets a
test file written alongside the implementation. Key test categories:

| Category | What we test |
|----------|-------------|
| Physics | Free-fall trajectory, hover equilibrium, mass depletion, angular dynamics |
| Environment | Step/reset shapes, termination conditions, reward bounds |
| Agent | Network forward pass shapes, action distribution validity |
| Training | Loss decreases over a few updates, no NaN/Inf in gradients |
| Integration | Full training run for N steps without crash |

Run with: `uv run pytest tests/ -v`
