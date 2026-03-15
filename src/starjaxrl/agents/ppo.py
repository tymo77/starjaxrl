"""PPOAgent: actor + critic with action sampling and evaluation API."""

import jax
import jax.numpy as jnp
from flax import nnx
from omegaconf import DictConfig

from starjaxrl.agents.networks import Actor, Critic, gaussian_entropy, gaussian_log_prob
from starjaxrl.env.starship_env import StarshipEnv


class PPOAgent(nnx.Module):
    """Holds Actor and Critic and exposes the PPO action/value interface.

    All methods work on a single (unbatched) observation. Use jax.vmap
    externally to batch across environments.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int,
        n_hidden: int,
        rngs: nnx.Rngs,
    ) -> None:
        self.actor  = Actor(obs_dim, action_dim, hidden_dim, n_hidden, rngs)
        self.critic = Critic(obs_dim, hidden_dim, n_hidden, rngs)

    def get_action_and_value(
        self,
        obs: jax.Array,
        key: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """Sample an action and return (action, log_prob, value, entropy).

        Used during rollout collection.
        """
        mu, log_std = self.actor(obs)
        std = jnp.exp(log_std)
        eps = jax.random.normal(key, mu.shape)
        action   = mu + std * eps
        log_prob = gaussian_log_prob(action, mu, log_std)
        value    = self.critic(obs)
        entropy  = gaussian_entropy(log_std)
        return action, log_prob, value, entropy

    def evaluate_actions(
        self,
        obs: jax.Array,
        action: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Re-evaluate stored actions under the current policy.

        Returns (log_prob, entropy, value). Used during the PPO update.
        """
        mu, log_std = self.actor(obs)
        log_prob = gaussian_log_prob(action, mu, log_std)
        value    = self.critic(obs)
        entropy  = gaussian_entropy(log_std)
        return log_prob, entropy, value


def agent_from_cfg(cfg: DictConfig, key: jax.Array) -> PPOAgent:
    """Construct a PPOAgent from a Hydra train config."""
    rngs = nnx.Rngs(params=key)
    return PPOAgent(
        obs_dim=StarshipEnv.OBS_DIM,
        action_dim=StarshipEnv.ACTION_DIM,
        hidden_dim=int(cfg.network.hidden_dim),
        n_hidden=int(cfg.network.n_hidden),
        rngs=rngs,
    )
