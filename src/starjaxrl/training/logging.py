"""Experiment logging (wandb) and greedy evaluation rollout."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from starjaxrl.agents.ppo import TrainMetrics
from starjaxrl.env.starship_env import EnvParams, get_obs, is_success, reset, step as env_step
from starjaxrl.physics import StarshipState


# ---------------------------------------------------------------------------
# wandb helpers
# ---------------------------------------------------------------------------

def init_logging(cfg: Any) -> bool:
    """Initialise wandb. Returns True if wandb is active.

    Gracefully handles wandb being unavailable or disabled via config.
    """
    mode = str(cfg.wandb.get("mode", "disabled"))
    if mode == "disabled":
        return False

    try:
        import wandb
        wandb.init(
            project=str(cfg.wandb.get("project", "starjaxrl")),
            entity=cfg.wandb.get("entity", None) or None,
            config=dict(cfg),
            mode=mode,
        )
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[logging] wandb init failed ({exc}), continuing without it.")
        return False


def log_metrics(
    metrics: TrainMetrics,
    step: int,
    extra: dict | None = None,
    wandb_active: bool = False,
) -> None:
    """Log TrainMetrics to wandb (if active)."""
    if not wandb_active:
        return
    import wandb

    payload = {
        "train/total_loss":  float(metrics.total_loss),
        "train/pg_loss":     float(metrics.pg_loss),
        "train/vf_loss":     float(metrics.vf_loss),
        "train/entropy":     float(metrics.entropy),
        "train/mean_reward": float(metrics.mean_reward),
        "step": step,
    }
    if extra:
        payload.update(extra)
    wandb.log(payload, step=step)


def finish_logging(wandb_active: bool = False) -> None:
    """Finalise wandb run."""
    if wandb_active:
        import wandb
        wandb.finish()


def log_trajectory_artifact(
    states:       list,
    actions:      list,
    env_params:   EnvParams,
    step:         int,
    wandb_active: bool = False,
    render_dir:   str  = "renders",
) -> None:
    """Render a trajectory GIF and upload it as a W&B artifact.

    Generates an animated playback of the provided trajectory, saves it as a
    GIF under *render_dir*, and then uploads it to the active W&B run both as
    a versioned ``Artifact`` (for long-term storage) and as an inline
    ``wandb.Video`` (for in-run media preview).

    Args:
        states:       List of ``StarshipState`` objects from an eval episode.
        actions:      Corresponding list of action arrays.
        env_params:   Environment parameters used for the episode.
        step:         Current training update number (used for naming).
        wandb_active: Whether a W&B run is currently active.
        render_dir:   Directory in which to save the temporary GIF file.
    """
    if not wandb_active:
        return

    import tempfile
    from pathlib import Path

    import wandb

    from starjaxrl.utils.visualization import render_trajectory, save_animation

    fig, anim = render_trajectory(states, actions, env_params)

    renders_path = Path(render_dir)
    renders_path.mkdir(parents=True, exist_ok=True)
    gif_path = renders_path / f"trajectory_{step:04d}.gif"

    save_animation(anim, gif_path)

    # Upload as a versioned artifact for long-term storage
    artifact = wandb.Artifact(
        name=f"trajectory-step-{step:04d}",
        type="trajectory",
        description=f"Greedy eval trajectory playback at training update {step}",
        metadata={"step": step},
    )
    artifact.add_file(str(gif_path))
    wandb.log_artifact(artifact)

    # Also log inline so the GIF is visible in the run's Media panel
    wandb.log({"eval/trajectory": wandb.Video(str(gif_path), fps=30, format="gif")}, step=step)

    import matplotlib.pyplot as plt
    plt.close(fig)


# ---------------------------------------------------------------------------
# Greedy evaluation rollout
# ---------------------------------------------------------------------------

def run_eval_episode(
    agent_state: Any,
    graphdef:    Any,
    env_params:  EnvParams,
    key:         jax.Array,
    max_steps:   int = 2400,
) -> tuple[list[StarshipState], list[jax.Array], bool, float]:
    """Run one greedy (mean-action) episode and return trajectory data.

    Returns:
        states:        list of StarshipState at each step
        actions:       list of action arrays
        success:       whether the episode ended in a successful landing
        total_reward:  cumulative reward
    """
    from flax import nnx

    agent = nnx.merge(graphdef, agent_state)

    key, reset_key = jax.random.split(key)
    state = reset(reset_key, env_params)

    states:  list[StarshipState] = [state]
    actions: list[jax.Array]    = []
    total_reward = 0.0

    for _ in range(max_steps):
        obs = get_obs(state)
        # Greedy: take the mean action (no sampling noise)
        mu, _log_std = agent.actor(obs)
        action = mu

        state, _obs, reward, done, info = env_step(state, action, env_params)
        states.append(state)
        actions.append(action)
        total_reward += float(reward)

        if bool(done):
            break

    success = bool(is_success(state, env_params))
    return states, actions, success, total_reward
