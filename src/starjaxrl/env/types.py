"""Shared environment types used across all environments."""

from typing import NamedTuple

import jax


class StepInfo(NamedTuple):
    """Fixed-structure info dict (required for lax.scan compatibility)."""
    success: jax.Array  # bool scalar
