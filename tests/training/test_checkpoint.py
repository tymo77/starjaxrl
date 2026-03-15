"""Tests for checkpoint save/load and CheckpointManager."""

import jax
import jax.numpy as jnp
import pytest

from starjaxrl.training.checkpoint import CheckpointManager, load_checkpoint, save_checkpoint
from starjaxrl.training.runner import init_runner

KEY = jax.random.PRNGKey(99)


@pytest.fixture(scope="module")
def agent_state(train_cfg):
    runner_state, _graphdef, _opt = init_runner(train_cfg, KEY)
    return runner_state.agent_state


# ---------------------------------------------------------------------------
# save / load round-trip
# ---------------------------------------------------------------------------

def test_save_creates_file(tmp_path, agent_state):
    save_checkpoint(agent_state, tmp_path / "test_ckpt")
    assert (tmp_path / "test_ckpt.npz").exists()


def test_load_restores_identical_params(tmp_path, agent_state):
    path = tmp_path / "roundtrip"
    save_checkpoint(agent_state, path)
    restored = load_checkpoint(path, agent_state)

    orig_leaves = jax.tree.leaves(agent_state)
    rest_leaves = jax.tree.leaves(restored)

    assert len(orig_leaves) == len(rest_leaves)
    for o, r in zip(orig_leaves, rest_leaves):
        assert jnp.allclose(o, r), "Param mismatch after load"


def test_load_with_npz_extension(tmp_path, agent_state):
    """load_checkpoint should accept path with or without .npz extension."""
    path = tmp_path / "ext_test"
    save_checkpoint(agent_state, path)
    # Load with explicit extension
    restored = load_checkpoint(tmp_path / "ext_test.npz", agent_state)
    orig_leaves = jax.tree.leaves(agent_state)
    rest_leaves = jax.tree.leaves(restored)
    for o, r in zip(orig_leaves, rest_leaves):
        assert jnp.allclose(o, r)


def test_restored_params_are_jax_arrays(tmp_path, agent_state):
    path = tmp_path / "jax_check"
    save_checkpoint(agent_state, path)
    restored = load_checkpoint(path, agent_state)
    for leaf in jax.tree.leaves(restored):
        assert isinstance(leaf, jax.Array)


# ---------------------------------------------------------------------------
# CheckpointManager
# ---------------------------------------------------------------------------

def test_ckpt_manager_saves_periodic(tmp_path, agent_state):
    mgr = CheckpointManager(tmp_path / "ckpts")
    mgr.save_periodic(agent_state, step=10)
    assert (tmp_path / "ckpts" / "step_000010.npz").exists()


def test_ckpt_manager_saves_best_on_improvement(tmp_path, agent_state):
    mgr = CheckpointManager(tmp_path / "best_test")
    saved = mgr.maybe_save_best(agent_state, mean_return=-50.0, step=1)
    assert saved is True
    assert (tmp_path / "best_test" / "best.npz").exists()


def test_ckpt_manager_no_save_on_regression(tmp_path, agent_state):
    mgr = CheckpointManager(tmp_path / "regression")
    mgr.maybe_save_best(agent_state, mean_return=-10.0, step=1)
    # Worse return — should not save again
    saved = mgr.maybe_save_best(agent_state, mean_return=-20.0, step=2)
    assert saved is False


def test_ckpt_manager_best_return_tracks_correctly(tmp_path, agent_state):
    mgr = CheckpointManager(tmp_path / "tracking")
    mgr.maybe_save_best(agent_state, mean_return=-10.0, step=1)
    assert mgr.best_return == pytest.approx(-10.0)
    mgr.maybe_save_best(agent_state, mean_return=-5.0, step=2)
    assert mgr.best_return == pytest.approx(-5.0)
    mgr.maybe_save_best(agent_state, mean_return=-8.0, step=3)
    assert mgr.best_return == pytest.approx(-5.0)  # unchanged


def test_ckpt_manager_load_best_roundtrip(tmp_path, agent_state):
    mgr = CheckpointManager(tmp_path / "rt_best")
    mgr.maybe_save_best(agent_state, mean_return=1.0, step=5)
    restored = mgr.load_best(agent_state)
    orig_leaves = jax.tree.leaves(agent_state)
    rest_leaves = jax.tree.leaves(restored)
    for o, r in zip(orig_leaves, rest_leaves):
        assert jnp.allclose(o, r)


def test_ckpt_manager_load_step_roundtrip(tmp_path, agent_state):
    mgr = CheckpointManager(tmp_path / "rt_step")
    mgr.save_periodic(agent_state, step=42)
    restored = mgr.load_step(42, agent_state)
    orig_leaves = jax.tree.leaves(agent_state)
    rest_leaves = jax.tree.leaves(restored)
    for o, r in zip(orig_leaves, rest_leaves):
        assert jnp.allclose(o, r)
