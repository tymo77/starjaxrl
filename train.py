"""Entry point for StarJAXRL training."""

import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    # TODO(Step 5): wire up runner.train(cfg)


if __name__ == "__main__":
    main()
