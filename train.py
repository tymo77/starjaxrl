"""Entry point for StarJAXRL training."""

import hydra
import jax
from omegaconf import DictConfig


@hydra.main(config_path="configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    # Dispatch to CartPole training when the env config has CartPole-specific fields
    is_cartpole = "m_cart" in cfg.env
    if is_cartpole:
        from starjaxrl.training.runner import train_cartpole as train_fn
    else:
        from starjaxrl.training.runner import train as train_fn

    debug_mode = cfg.get("debug", False)
    with jax.disable_jit(disable=debug_mode):
        runner_state, metrics = train_fn(cfg)
        final = metrics[-1]
        print(
            f"\nTraining complete. "
            f"Final mean reward: {float(final.mean_reward):.4f} | "
            f"pg: {float(final.pg_loss):.4f} | "
            f"vf: {float(final.vf_loss):.4f}"
        )


if __name__ == "__main__":
    main()
