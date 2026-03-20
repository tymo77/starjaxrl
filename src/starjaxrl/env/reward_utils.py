"""Shared reward shaping utilities used across environments."""

import jax
import jax.numpy as jnp


def gauss(value: jax.Array, sigma: float) -> jax.Array:
    """Unit Gaussian: 1.0 at value=0, exp(-0.5)≈0.6 at value=±sigma."""
    return jnp.exp(-0.5 * (value / sigma) ** 2)
