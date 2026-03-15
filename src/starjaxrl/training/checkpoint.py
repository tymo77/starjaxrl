"""Checkpoint save/load using numpy serialisation (reliable across Orbax versions)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


# ---------------------------------------------------------------------------
# Primitive save / load
# ---------------------------------------------------------------------------

def save_checkpoint(agent_state: Any, path: Path) -> None:
    """Serialise an NNX agent_state pytree to a .npz file.

    Args:
        agent_state: nnx.State pytree of JAX arrays.
        path:        Destination path (without .npz extension).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    leaves, _treedef = jax.tree_util.tree_flatten(agent_state)
    payload = {f"param_{i}": np.array(leaf) for i, leaf in enumerate(leaves)}
    payload["n_params"] = np.array(len(leaves))
    np.savez(str(path), **payload)


def load_checkpoint(path: Path, template: Any) -> Any:
    """Load a previously saved checkpoint and reconstruct the pytree.

    Args:
        path:     Path to the .npz file (with or without extension).
        template: An agent_state with the same pytree structure as the saved one.

    Returns:
        Restored agent_state pytree with JAX arrays.
    """
    path = Path(path)
    if path.suffix != ".npz":
        path = path.with_suffix(".npz")

    data   = np.load(str(path))
    n      = int(data["n_params"])
    leaves = [jnp.array(data[f"param_{i}"]) for i in range(n)]

    _leaves_template, treedef = jax.tree_util.tree_flatten(template)
    return treedef.unflatten(leaves)


# ---------------------------------------------------------------------------
# Checkpoint manager
# ---------------------------------------------------------------------------

class CheckpointManager:
    """Tracks periodic and best-return checkpoints during training."""

    BEST_NAME     = "best"
    PERIODIC_FMT  = "step_{step:06d}"

    def __init__(self, checkpoint_dir: str | Path) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_return: float = float("-inf")

    # ------------------------------------------------------------------
    def maybe_save_best(
        self,
        agent_state: Any,
        mean_return: float,
        step: int,
    ) -> bool:
        """Save a 'best' checkpoint if mean_return beats the current best.

        Returns True if a new best was saved.
        """
        if mean_return > self.best_return:
            self.best_return = mean_return
            path = self.checkpoint_dir / self.BEST_NAME
            save_checkpoint(agent_state, path)
            return True
        return False

    def save_periodic(self, agent_state: Any, step: int) -> None:
        """Save a periodic checkpoint keyed by step number."""
        name = self.PERIODIC_FMT.format(step=step)
        save_checkpoint(agent_state, self.checkpoint_dir / name)

    # ------------------------------------------------------------------
    def load_best(self, template: Any) -> Any:
        """Load the best checkpoint."""
        return load_checkpoint(self.checkpoint_dir / self.BEST_NAME, template)

    def load_step(self, step: int, template: Any) -> Any:
        """Load a periodic checkpoint for a given step."""
        name = self.PERIODIC_FMT.format(step=step)
        return load_checkpoint(self.checkpoint_dir / name, template)
