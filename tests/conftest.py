"""Shared pytest fixtures for StarJAXRL tests."""

import pytest
from omegaconf import OmegaConf


@pytest.fixture(scope="session")
def env_cfg():
    return OmegaConf.load("configs/env/env.yaml")


@pytest.fixture(scope="session")
def ppo_cfg():
    return OmegaConf.load("configs/ppo/ppo.yaml")


@pytest.fixture(scope="session")
def network_cfg():
    return OmegaConf.load("configs/network/network.yaml")


@pytest.fixture(scope="session")
def train_cfg():
    """Full merged config (mirrors what Hydra produces at runtime)."""
    env = OmegaConf.load("configs/env/env.yaml")
    ppo = OmegaConf.load("configs/ppo/ppo.yaml")
    network = OmegaConf.load("configs/network/network.yaml")
    base = OmegaConf.load("configs/train.yaml")
    # Merge sub-configs under their keys, then overlay base _self_ keys
    return OmegaConf.merge(
        OmegaConf.create({"env": env, "ppo": ppo, "network": network}),
        {k: v for k, v in base.items() if k not in ("defaults",)},
    )
