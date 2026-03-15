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
