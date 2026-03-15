"""Actor and Critic network modules using Flax NNX."""

import jax
import jax.numpy as jnp
from flax import nnx


# ---------------------------------------------------------------------------
# Initialization helper
# ---------------------------------------------------------------------------

def _ortho_linear(in_dim: int, out_dim: int, scale: float, rngs: nnx.Rngs) -> nnx.Linear:
    """Linear layer with orthogonal weight init (recommended for PPO)."""
    return nnx.Linear(
        in_dim, out_dim,
        kernel_init=jax.nn.initializers.orthogonal(scale),
        bias_init=jax.nn.initializers.zeros,
        rngs=rngs,
    )


# ---------------------------------------------------------------------------
# Probability utilities
# ---------------------------------------------------------------------------

def gaussian_log_prob(action: jax.Array, mu: jax.Array, log_std: jax.Array) -> jax.Array:
    """Log probability of action under a diagonal Gaussian N(mu, exp(log_std)²).

    Returns a scalar (sum over action dimensions).
    """
    std = jnp.exp(log_std)
    return -0.5 * jnp.sum(
        ((action - mu) / std) ** 2 + 2.0 * log_std + jnp.log(2.0 * jnp.pi),
        axis=-1,
    )


def gaussian_entropy(log_std: jax.Array) -> jax.Array:
    """Differential entropy of a diagonal Gaussian (sum over action dims)."""
    return jnp.sum(log_std + 0.5 * jnp.log(2.0 * jnp.pi * jnp.e), axis=-1)


# ---------------------------------------------------------------------------
# Actor
# ---------------------------------------------------------------------------

class Actor(nnx.Module):
    """Gaussian policy network.

    Outputs mean μ from a MLP; log_std is a learned state-independent parameter
    clipped to [-1, 2] for numerical stability.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int,
        n_hidden: int,
        rngs: nnx.Rngs,
    ) -> None:
        hidden_scale = jnp.sqrt(2.0)
        dims = [obs_dim] + [hidden_dim] * n_hidden

        self.hidden = nnx.List([
            _ortho_linear(dims[i], dims[i + 1], hidden_scale, rngs)
            for i in range(n_hidden)
        ])
        # Small output scale → starts near-deterministic, entropy grows during training
        self.mu_head = _ortho_linear(hidden_dim, action_dim, 0.01, rngs)
        self.log_std = nnx.Param(jnp.zeros(action_dim))

    def __call__(self, obs: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Returns (mu, log_std) for the action distribution."""
        x = obs
        for layer in self.hidden:
            x = jax.nn.tanh(layer(x))
        mu = self.mu_head(x)
        log_std = jnp.clip(self.log_std[...], -1.0, 2.0)
        return mu, log_std


# ---------------------------------------------------------------------------
# Critic
# ---------------------------------------------------------------------------

class Critic(nnx.Module):
    """Value network. Outputs scalar V(s)."""

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int,
        n_hidden: int,
        rngs: nnx.Rngs,
    ) -> None:
        hidden_scale = jnp.sqrt(2.0)
        dims = [obs_dim] + [hidden_dim] * n_hidden

        self.hidden = nnx.List([
            _ortho_linear(dims[i], dims[i + 1], hidden_scale, rngs)
            for i in range(n_hidden)
        ])
        self.value_head = _ortho_linear(hidden_dim, 1, 1.0, rngs)

    def __call__(self, obs: jax.Array) -> jax.Array:
        """Returns scalar value estimate V(s)."""
        x = obs
        for layer in self.hidden:
            x = jax.nn.tanh(layer(x))
        return self.value_head(x).squeeze(-1)
