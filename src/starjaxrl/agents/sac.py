"""Soft Actor-Critic (SAC) — networks, replay buffer, and train metrics."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from flax import nnx
from omegaconf import DictConfig

from starjaxrl.agents.networks import gaussian_log_prob


# ---------------------------------------------------------------------------
# Train metrics
# ---------------------------------------------------------------------------

class SACTrainMetrics(NamedTuple):
    """Scalar metrics returned by each SAC train_step."""
    actor_loss:  jax.Array
    critic_loss: jax.Array
    alpha_loss:  jax.Array
    alpha:       jax.Array
    mean_reward: jax.Array


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

class ReplayBuffer(NamedTuple):
    """Fixed-capacity circular replay buffer stored as JAX arrays."""
    obs:      jax.Array  # (capacity, obs_dim)
    action:   jax.Array  # (capacity, action_dim)
    reward:   jax.Array  # (capacity,)
    next_obs: jax.Array  # (capacity, obs_dim)
    done:     jax.Array  # (capacity,) bool
    ptr:      jax.Array  # scalar int32 — next write index
    size:     jax.Array  # scalar int32 — number of valid entries


def make_replay_buffer(capacity: int, obs_dim: int, action_dim: int) -> ReplayBuffer:
    """Allocate an empty replay buffer."""
    return ReplayBuffer(
        obs      = jnp.zeros((capacity, obs_dim)),
        action   = jnp.zeros((capacity, action_dim)),
        reward   = jnp.zeros(capacity),
        next_obs = jnp.zeros((capacity, obs_dim)),
        done     = jnp.zeros(capacity, dtype=bool),
        ptr      = jnp.zeros((), dtype=jnp.int32),
        size     = jnp.zeros((), dtype=jnp.int32),
    )


def buffer_add_batch(
    buffer:   ReplayBuffer,
    obs:      jax.Array,   # (N, obs_dim)
    action:   jax.Array,   # (N, action_dim)
    reward:   jax.Array,   # (N,)
    next_obs: jax.Array,   # (N, obs_dim)
    done:     jax.Array,   # (N,) bool
) -> ReplayBuffer:
    """Add N transitions to the circular buffer (jit-compatible)."""
    n        = obs.shape[0]
    capacity = buffer.obs.shape[0]
    indices  = (buffer.ptr + jnp.arange(n)) % capacity
    return ReplayBuffer(
        obs      = buffer.obs.at[indices].set(obs),
        action   = buffer.action.at[indices].set(action),
        reward   = buffer.reward.at[indices].set(reward),
        next_obs = buffer.next_obs.at[indices].set(next_obs),
        done     = buffer.done.at[indices].set(done),
        ptr      = (buffer.ptr + n) % capacity,
        size     = jnp.minimum(buffer.size + n, capacity),
    )


def buffer_sample(
    buffer:     ReplayBuffer,
    batch_size: int,
    key:        jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Uniformly sample a batch. Safe even when buffer is not full yet."""
    indices = jax.random.randint(key, (batch_size,), 0, jnp.maximum(buffer.size, 1))
    return (
        buffer.obs[indices],
        buffer.action[indices],
        buffer.reward[indices],
        buffer.next_obs[indices],
        buffer.done[indices],
    )


# ---------------------------------------------------------------------------
# Network hyperparameters
# ---------------------------------------------------------------------------

_LOG_STD_MIN = -5.0
_LOG_STD_MAX = 2.0


# ---------------------------------------------------------------------------
# Q-network
# ---------------------------------------------------------------------------

class SACQNetwork(nnx.Module):
    """Q(s, a) function approximator (MLP with ReLU activations)."""

    def __init__(
        self,
        obs_dim:    int,
        action_dim: int,
        hidden_dim: int,
        n_hidden:   int,
        rngs:       nnx.Rngs,
    ) -> None:
        dims = [obs_dim + action_dim] + [hidden_dim] * n_hidden
        self.hidden = nnx.List([
            nnx.Linear(dims[i], dims[i + 1], rngs=rngs)
            for i in range(n_hidden)
        ])
        self.q_head = nnx.Linear(hidden_dim, 1, rngs=rngs)

    def __call__(self, obs: jax.Array, action: jax.Array) -> jax.Array:
        """Return scalar Q-value estimate."""
        x = jnp.concatenate([obs, action], axis=-1)
        for layer in self.hidden:
            x = jax.nn.relu(layer(x))
        return self.q_head(x).squeeze(-1)


class SACTwinnedQ(nnx.Module):
    """Two Q-networks for double Q-learning (reduces overestimation bias)."""

    def __init__(
        self,
        obs_dim:    int,
        action_dim: int,
        hidden_dim: int,
        n_hidden:   int,
        rngs:       nnx.Rngs,
    ) -> None:
        self.q1 = SACQNetwork(obs_dim, action_dim, hidden_dim, n_hidden, rngs)
        self.q2 = SACQNetwork(obs_dim, action_dim, hidden_dim, n_hidden, rngs)

    def __call__(
        self, obs: jax.Array, action: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        """Return (Q1, Q2) estimates."""
        return self.q1(obs, action), self.q2(obs, action)


# ---------------------------------------------------------------------------
# Squashed Gaussian actor
# ---------------------------------------------------------------------------

class SACActor(nnx.Module):
    """Squashed Gaussian policy for SAC.

    Samples actions via the reparameterisation trick and applies tanh
    squashing so that all outputs lie in (-1, 1).  Environments clip
    actions to their valid physical ranges internally.

    The log-probability accounts for the tanh change-of-variables:
        log π(a|s) = log N(u | μ, σ) − Σ log(1 − tanh²(u) + ε)
    where a = tanh(u), u ~ N(μ, σ).
    """

    def __init__(
        self,
        obs_dim:    int,
        action_dim: int,
        hidden_dim: int,
        n_hidden:   int,
        rngs:       nnx.Rngs,
    ) -> None:
        dims = [obs_dim] + [hidden_dim] * n_hidden
        self.hidden = nnx.List([
            nnx.Linear(dims[i], dims[i + 1], rngs=rngs)
            for i in range(n_hidden)
        ])
        self.mu_head      = nnx.Linear(hidden_dim, action_dim, rngs=rngs)
        self.log_std_head = nnx.Linear(hidden_dim, action_dim, rngs=rngs)

    def _trunk(self, obs: jax.Array) -> jax.Array:
        x = obs
        for layer in self.hidden:
            x = jax.nn.relu(layer(x))
        return x

    def __call__(self, obs: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Return (mu, log_std) of the pre-squash Gaussian distribution."""
        x       = self._trunk(obs)
        mu      = self.mu_head(x)
        log_std = jnp.clip(self.log_std_head(x), _LOG_STD_MIN, _LOG_STD_MAX)
        return mu, log_std

    def sample(
        self, obs: jax.Array, key: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        """Sample a squashed action and return (action, log_prob).

        Uses the reparameterisation trick: u = μ + σ·ε, a = tanh(u).
        """
        mu, log_std = self(obs)
        std = jnp.exp(log_std)
        u   = mu + std * jax.random.normal(key, mu.shape)
        a   = jnp.tanh(u)
        log_prob = (
            gaussian_log_prob(u, mu, log_std)
            - jnp.sum(jnp.log(1.0 - a ** 2 + 1e-6), axis=-1)
        )
        return a, log_prob

    def mean_action(self, obs: jax.Array) -> jax.Array:
        """Deterministic (greedy) action — tanh of the distribution mean."""
        mu, _ = self(obs)
        return jnp.tanh(mu)


# ---------------------------------------------------------------------------
# Convenience constructor
# ---------------------------------------------------------------------------

def sac_actor_from_cfg(
    cfg:        DictConfig,
    key:        jax.Array,
    obs_dim:    int,
    action_dim: int,
) -> SACActor:
    """Build a SACActor from a Hydra train config."""
    return SACActor(
        obs_dim    = obs_dim,
        action_dim = action_dim,
        hidden_dim = int(cfg.network.hidden_dim),
        n_hidden   = int(cfg.network.n_hidden),
        rngs       = nnx.Rngs(params=key),
    )
