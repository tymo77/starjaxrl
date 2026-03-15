# 05 — Training Infrastructure

## Vectorized Environments

JAX's `vmap` lets us run N environments in parallel with zero overhead —
the physics step and environment reset are batched across all envs simultaneously.

```python
# Vectorized step: runs N envs in one XLA kernel
batch_step = jax.vmap(env.step, in_axes=(0, 0))
```

**Open questions:**
- How many parallel envs? 64–256 is typical for PPO on CPU. More on GPU.
- Do we want a curriculum (start easy, increase difficulty)?

---

## Experiment Logging

| Option | Pros | Cons |
|--------|------|------|
| **Weights & Biases (wandb)** | Rich UI, easy comparison, free tier | External dependency, requires account |
| **TensorBoard** | Local, no account needed | Less polished UI |
| **CSV + matplotlib** | Minimal, no setup | Manual plotting |

> **Proposed: wandb** for a demo project — the visualizations are compelling and
> shareable. Easy to add with a single `wandb.log()` call.

**Open question:** Do you have a wandb account / preference here?

---

## Checkpointing

Use **Orbax** (already installed as a Flax dependency) to save/restore training state.

```python
# Save
checkpointer = orbax.checkpoint.PyTreeCheckpointer()
checkpointer.save(path, runner_state)

# Restore
runner_state = checkpointer.restore(path)
```

Save checkpoints every N updates and keep the best by mean episode return.

---

## Hardware Target

| Target | Notes |
|--------|-------|
| **CPU (M-series Mac)** | Works out of the box, JAX metal plugin for GPU accel |
| **CUDA GPU** | `pip install jax[cuda]`, fastest for large batch training |
| **TPU** | Overkill for this project |

> **Proposed:** Write hardware-agnostic code (JAX handles this automatically).
> Optionally add `jax[metal]` for Apple Silicon GPU acceleration.

**Open question:** Are you training on your Mac or a cloud GPU?

---

## Evaluation & Rendering

Separate from training — run a greedy (deterministic) policy, record trajectory,
render with Matplotlib animation.

- `matplotlib.animation.FuncAnimation` for trajectory playback
- Save as `.gif` or `.mp4` for sharing
- Render: vehicle as a rectangle, thrust vector as an arrow, trajectory as a line

---

## Project Milestones

| Milestone | Description |
|-----------|-------------|
| M1 | Physics simulator passes sanity checks (free-fall, hover) |
| M2 | Environment step/reset works, basic reward fires |
| M3 | PPO training loop runs, reward increases over time |
| M4 | Successful landings demonstrated, animation rendered |
| M5 | Ablations, reward shaping experiments, writeup |
