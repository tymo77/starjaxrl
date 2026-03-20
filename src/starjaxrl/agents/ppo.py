"""PPOAgent, rollout data structures, and GAE computation."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from flax import nnx
from omegaconf import DictConfig

from starjaxrl.agents.networks import Actor, Critic, gaussian_entropy, gaussian_log_prob


# ---------------------------------------------------------------------------
# Rollout data structures
# ---------------------------------------------------------------------------

class Transition(NamedTuple):
    """One timestep of experience from a single environment."""
    obs:      jax.Array   # (obs_dim,)
    action:   jax.Array   # (action_dim,)
    log_prob: jax.Array   # scalar — log π(a|s) under behaviour policy
    value:    jax.Array   # scalar — V(s) from critic
    reward:   jax.Array   # scalar
    done:     jax.Array   # bool scalar


class TrainMetrics(NamedTuple):
    """Scalar metrics returned by each train_step."""
    total_loss:  jax.Array
    pg_loss:     jax.Array
    vf_loss:     jax.Array
    entropy:     jax.Array
    mean_reward: jax.Array


# ---------------------------------------------------------------------------
# GAE
# ---------------------------------------------------------------------------

def compute_gae(
    rewards:    jax.Array,   # (T, N)
    values:     jax.Array,   # (T, N)
    dones:      jax.Array,   # (T, N)  bool
    last_value: jax.Array,   # (N,)
    gamma:      float,
    gae_lambda: float,
) -> tuple[jax.Array, jax.Array]:
    """Generalised Advantage Estimation (GAE-λ).

    Returns:
        advantages: (T, N)
        returns:    (T, N)  = advantages + values (targets for value head)
    """
    not_done = 1.0 - dones.astype(jnp.float32)

    # next_values[t] = V(s_{t+1}); last step bootstraps from last_value
    next_values = jnp.concatenate([values[1:], last_value[None]], axis=0)  # (T, N)

    # Reverse all time-axis arrays for a forward scan over reversed time
    rewards_r     = jnp.flip(rewards,     axis=0)
    values_r      = jnp.flip(values,      axis=0)
    not_done_r    = jnp.flip(not_done,    axis=0)
    next_values_r = jnp.flip(next_values, axis=0)

    def gae_step(last_gae: jax.Array, xs: tuple) -> tuple:
        reward, value, nd, next_val = xs
        delta    = reward + gamma * next_val * nd - value
        gae      = delta + gamma * gae_lambda * nd * last_gae
        return gae, gae

    _, advantages_r = jax.lax.scan(
        gae_step,
        jnp.zeros_like(last_value),
        (rewards_r, values_r, not_done_r, next_values_r),
    )

    advantages = jnp.flip(advantages_r, axis=0)
    returns    = advantages + values
    return advantages, returns


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


def agent_from_cfg(cfg: DictConfig, key: jax.Array, obs_dim: int, action_dim: int) -> PPOAgent:
    """Construct a PPOAgent from a Hydra train config.

    Args:
        cfg:        Hydra config containing ``network`` sub-config.
        key:        PRNG key for parameter initialisation.
        obs_dim:    Observation dimensionality (env-specific).
        action_dim: Action dimensionality (env-specific).
    """
    rngs = nnx.Rngs(params=key)
    return PPOAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=int(cfg.network.hidden_dim),
        n_hidden=int(cfg.network.n_hidden),
        rngs=rngs,
    )
