"""Evaluation script: load a checkpoint, run a greedy rollout, save animation.

Usage:
    uv run python evaluate.py                              # uses best checkpoint
    uv run python evaluate.py checkpoint=checkpoints/best
    uv run python evaluate.py output=renders/landing.gif
"""

from pathlib import Path

import hydra
import jax
import numpy as np
from omegaconf import DictConfig, OmegaConf

from starjaxrl.env.starship_env import env_params_from_cfg
from starjaxrl.training.checkpoint import load_checkpoint
from starjaxrl.training.logging import run_eval_episode
from starjaxrl.training.runner import init_runner
from starjaxrl.utils.visualization import render_trajectory, save_animation


@hydra.main(config_path="configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    # --- Resolve paths from cfg overrides or defaults ---
    checkpoint_path = Path(cfg.get("checkpoint", "checkpoints/best"))
    output_path     = Path(cfg.get("output",     "renders/landing.gif"))
    seed            = int(cfg.seed)

    env_params = env_params_from_cfg(cfg.env)
    key        = jax.random.PRNGKey(seed)

    # --- Load checkpoint ---
    print(f"Loading checkpoint from: {checkpoint_path}")
    key, init_key = jax.random.split(key)
    runner_state, graphdef, _ = init_runner(cfg, init_key)

    ckpt_file = checkpoint_path
    if not (ckpt_file.with_suffix(".npz")).exists() and not ckpt_file.with_suffix(".npz").exists():
        print(f"  Checkpoint not found at {checkpoint_path}.npz — using random weights.")
        agent_state = runner_state.agent_state
    else:
        agent_state = load_checkpoint(checkpoint_path, runner_state.agent_state)
        print("  Checkpoint loaded.")

    # --- Greedy eval rollout ---
    key, eval_key = jax.random.split(key)
    states, actions, success, total_reward = run_eval_episode(
        agent_state, graphdef, env_params, eval_key
    )

    print(f"Episode: {len(states) - 1} steps | "
          f"return {total_reward:.2f} | "
          f"{'SUCCESS ✓' if success else 'crash ✗'}")

    # --- Render and save ---
    np_actions = [np.array(a) for a in actions]
    fig, anim  = render_trajectory(states, np_actions, env_params)
    save_animation(anim, output_path)
    print(f"Done → {output_path}")


if __name__ == "__main__":
    main()
