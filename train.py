"""Entry point for StarJAXRL training."""

import hydra
from omegaconf import DictConfig

from starjaxrl.training.runner import train


@hydra.main(config_path="configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    runner_state, metrics = train(cfg)
    final = metrics[-1]
    print(
        f"\nTraining complete. "
        f"Final mean reward: {float(final.mean_reward):.4f} | "
        f"pg: {float(final.pg_loss):.4f} | "
        f"vf: {float(final.vf_loss):.4f}"
    )


if __name__ == "__main__":
    main()
