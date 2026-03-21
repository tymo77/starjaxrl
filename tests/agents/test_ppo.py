"""Tests for GAE, PPO loss, and the training loop."""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from starjaxrl.agents import TrainMetrics, compute_gae
from starjaxrl.training.runner import (
    RunnerState,
    init_runner,
    make_train_step,
)
from starjaxrl.env.starship_env import (
    StarshipEnv,
    env_params_from_cfg,
    get_obs,
    reset,
    step as env_step,
)

KEY = jax.random.PRNGKey(7)


def _init_starship(cfg, key=KEY):
    """Helper: init runner for Starship env with updated signature."""
    env_params = env_params_from_cfg(cfg.env)
    return init_runner(
        cfg, key, env_params, reset, get_obs,
        obs_dim=StarshipEnv.OBS_DIM, action_dim=StarshipEnv.ACTION_DIM,
    )


def _make_step(graphdef, optimizer, env_params, cfg):
    """Helper: make train_step for Starship env with updated signature."""
    return make_train_step(graphdef, optimizer, env_params, cfg, reset, get_obs, env_step)


# ---------------------------------------------------------------------------
# compute_gae
# ---------------------------------------------------------------------------

def test_gae_output_shapes():
    T, N = 10, 4
    rewards    = jnp.ones((T, N))
    values     = jnp.zeros((T, N))
    dones      = jnp.zeros((T, N), dtype=bool)
    last_value = jnp.zeros(N)
    adv, ret = compute_gae(rewards, values, dones, last_value, 0.99, 0.95)
    assert adv.shape == (T, N)
    assert ret.shape == (T, N)


def test_gae_terminal_zeros_bootstrap():
    """When done=True, bootstrap from next state is zeroed out."""
    T, N = 1, 1
    rewards    = jnp.array([[5.0]])
    values     = jnp.array([[2.0]])
    dones      = jnp.array([[True]])
    last_value = jnp.array([999.0])  # Should be ignored since done=True

    adv, ret = compute_gae(rewards, values, dones, last_value, 0.99, 0.95)
    # delta = reward - value = 5 - 2 = 3;  advantage = delta = 3
    assert float(adv[0, 0]) == pytest.approx(3.0, rel=1e-5)


def test_gae_bootstrap_non_terminal():
    """With done=False and lambda=1, advantage = delta (one-step)."""
    T, N = 1, 1
    rewards    = jnp.array([[1.0]])
    values     = jnp.array([[0.0]])
    dones      = jnp.array([[False]])
    last_value = jnp.array([1.0])
    gamma, lam = 0.9, 1.0
    adv, _ = compute_gae(rewards, values, dones, last_value, gamma, lam)
    # delta = 1.0 + 0.9 * 1.0 - 0.0 = 1.9
    assert float(adv[0, 0]) == pytest.approx(1.9, rel=1e-4)


def test_gae_returns_equal_adv_plus_values():
    T, N = 8, 4
    rewards    = jax.random.normal(KEY, (T, N))
    values     = jax.random.normal(KEY, (T, N))
    dones      = jnp.zeros((T, N), dtype=bool)
    last_value = jax.random.normal(KEY, (N,))
    adv, ret = compute_gae(rewards, values, dones, last_value, 0.99, 0.95)
    assert jnp.allclose(ret, adv + values, atol=1e-5)


def test_gae_finite():
    T, N = 16, 8
    rewards    = jax.random.normal(KEY, (T, N))
    values     = jax.random.normal(KEY, (T, N))
    dones      = jax.random.bernoulli(KEY, 0.1, (T, N))
    last_value = jax.random.normal(KEY, (N,))
    adv, ret = compute_gae(rewards, values, dones, last_value, 0.99, 0.95)
    assert jnp.all(jnp.isfinite(adv))
    assert jnp.all(jnp.isfinite(ret))


# ---------------------------------------------------------------------------
# init_runner
# ---------------------------------------------------------------------------

def test_init_runner(train_cfg):
    runner_state, graphdef, optimizer = _init_starship(train_cfg)
    n_envs  = int(train_cfg.ppo.n_envs)
    obs_dim = 7  # StarshipEnv.OBS_DIM

    assert runner_state.obs.shape == (n_envs, obs_dim)
    assert runner_state.env_states.y.shape == (n_envs,)
    assert jnp.all(jnp.isfinite(runner_state.obs))
    assert int(runner_state.step) == 0


def test_init_runner_agent_state_finite(train_cfg):
    runner_state, _, _ = _init_starship(train_cfg)
    leaves = jax.tree.leaves(runner_state.agent_state)
    for leaf in leaves:
        assert jnp.all(jnp.isfinite(leaf)), "Non-finite agent parameter at init"


# ---------------------------------------------------------------------------
# train_step — single update
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def runner_and_step(train_cfg):
    """Initialise and run one train_step at full Earth gravity."""
    env_params = env_params_from_cfg(train_cfg.env)
    runner_state, graphdef, optimizer = _init_starship(train_cfg)
    train_step = jax.jit(_make_step(graphdef, optimizer, env_params, train_cfg))
    current_g = jnp.array(float(train_cfg.env.g), dtype=jnp.float32)
    new_runner, metrics = train_step(runner_state, current_g)
    return runner_state, new_runner, metrics


def test_train_step_returns_metrics(runner_and_step):
    _, _, metrics = runner_and_step
    assert isinstance(metrics, TrainMetrics)


def test_train_step_metrics_finite(runner_and_step):
    _, _, metrics = runner_and_step
    for field in metrics:
        assert jnp.isfinite(field), f"Non-finite metric: {field}"


def test_train_step_params_change(runner_and_step):
    """Agent parameters must change after a gradient update."""
    old_runner, new_runner, _ = runner_and_step
    old_leaves = jax.tree.leaves(old_runner.agent_state)
    new_leaves = jax.tree.leaves(new_runner.agent_state)
    any_changed = any(
        not jnp.allclose(o, n)
        for o, n in zip(old_leaves, new_leaves)
    )
    assert any_changed, "No parameter changed after train_step"


def test_train_step_no_nan_in_grads(runner_and_step):
    """New parameters must all be finite (no NaN from bad gradients)."""
    _, new_runner, _ = runner_and_step
    leaves = jax.tree.leaves(new_runner.agent_state)
    for leaf in leaves:
        assert jnp.all(jnp.isfinite(leaf)), "NaN/Inf in agent state after update"


def test_train_step_obs_shape(runner_and_step):
    _, new_runner, _ = runner_and_step
    assert new_runner.obs.shape == runner_and_step[0].obs.shape


def test_train_step_increments_step(runner_and_step):
    old_runner, new_runner, _ = runner_and_step
    assert int(new_runner.step) == int(old_runner.step) + 1


# ---------------------------------------------------------------------------
# Multi-step: reward should not diverge
# ---------------------------------------------------------------------------

def test_multiple_train_steps_stable(train_cfg):
    """Run 5 train steps; losses and rewards must stay finite."""
    env_params = env_params_from_cfg(train_cfg.env)
    runner_state, graphdef, optimizer = _init_starship(train_cfg)
    train_step = jax.jit(_make_step(graphdef, optimizer, env_params, train_cfg))
    current_g = jnp.array(float(train_cfg.env.g), dtype=jnp.float32)

    for _ in range(5):
        runner_state, metrics = train_step(runner_state, current_g)
        for field in metrics:
            assert jnp.isfinite(field), "Divergence detected during multi-step test"
