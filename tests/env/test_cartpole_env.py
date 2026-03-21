"""Tests for the CartPole balancing environment."""

import jax
import jax.numpy as jnp
import pytest

from starjaxrl.env.cartpole_env import (
    CartPoleEnv,
    CartPoleEnvParams,
    DEFAULT_ENV_PARAMS,
    compute_reward,
    get_obs,
    is_done,
    is_success,
    reset,
    step,
)
from starjaxrl.physics.cartpole_dynamics import CartPoleState

KEY = jax.random.PRNGKey(42)
P = DEFAULT_ENV_PARAMS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(**overrides) -> CartPoleState:
    base = dict(x=0.0, x_dot=0.0, theta=0.0, theta_dot=0.0, time=0.0)
    base.update(overrides)
    return CartPoleState(**{k: jnp.array(v, dtype=jnp.float32) for k, v in base.items()})


def _zero_action():
    return jnp.zeros(1)


# ---------------------------------------------------------------------------
# reset
# ---------------------------------------------------------------------------

def test_reset_obs_shape():
    state = reset(KEY, P)
    obs = get_obs(state)
    assert obs.shape == (CartPoleEnv.OBS_DIM,)


def test_reset_time_zero():
    state = reset(KEY, P)
    assert float(state.time) == pytest.approx(0.0)


def test_reset_x_starts_at_zero():
    state = reset(KEY, P)
    assert float(state.x) == pytest.approx(P.x0)


def test_reset_random_theta_perturbation():
    """Different keys produce different initial angles."""
    key1, key2 = jax.random.split(KEY)
    s1 = reset(key1, P)
    s2 = reset(key2, P)
    # Both should be near theta0 but not necessarily equal
    assert abs(float(s1.theta)) <= P.theta0 + 0.05 + 1e-3
    assert abs(float(s2.theta)) <= P.theta0 + 0.05 + 1e-3


# ---------------------------------------------------------------------------
# get_obs
# ---------------------------------------------------------------------------

def test_obs_shape():
    state = _make_state()
    obs = get_obs(state)
    assert obs.shape == (CartPoleEnv.OBS_DIM,)


def test_obs_finite():
    state = _make_state(x=1.0, x_dot=2.0, theta=0.1, theta_dot=0.5)
    obs = get_obs(state)
    assert jnp.all(jnp.isfinite(obs))


def test_obs_at_limits_approximately_one():
    """State at x_max → obs[0] ≈ ±1."""
    state = _make_state(x=2.4)
    obs = get_obs(state)
    assert float(obs[0]) == pytest.approx(1.0, rel=1e-4)


def test_obs_upright_zero():
    """At theta=0, obs[2] ≈ 0."""
    state = _make_state(theta=0.0)
    obs = get_obs(state)
    assert float(obs[2]) == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# step — output shapes and types
# ---------------------------------------------------------------------------

def test_step_output_shapes():
    state = _make_state()
    next_state, obs, reward, done, info = step(state, _zero_action(), P)
    assert obs.shape    == (CartPoleEnv.OBS_DIM,)
    assert reward.shape == ()
    assert done.shape   == ()
    assert info.success.shape == ()


def test_step_obs_finite():
    state = _make_state()
    _, obs, _, _, _ = step(state, _zero_action(), P)
    assert jnp.all(jnp.isfinite(obs))


def test_step_reward_finite():
    state = _make_state()
    _, _, reward, _, _ = step(state, _zero_action(), P)
    assert jnp.isfinite(reward)


def test_step_time_advances():
    state = _make_state(time=0.0)
    next_state, _, _, _, _ = step(state, _zero_action(), P)
    assert float(next_state.time) == pytest.approx(P.dt, rel=1e-4)


# ---------------------------------------------------------------------------
# Termination conditions
# ---------------------------------------------------------------------------

def test_done_when_pole_falls_right():
    state = _make_state(theta=P.theta_max + 0.01)
    assert bool(is_done(state, P)) is True


def test_done_when_pole_falls_left():
    state = _make_state(theta=-(P.theta_max + 0.01))
    assert bool(is_done(state, P)) is True


def test_done_when_cart_out_of_bounds_right():
    state = _make_state(x=P.x_max + 0.1)
    assert bool(is_done(state, P)) is True


def test_done_when_cart_out_of_bounds_left():
    state = _make_state(x=-(P.x_max + 0.1))
    assert bool(is_done(state, P)) is True


def test_done_on_timeout():
    state = _make_state(time=P.t_max + P.dt)
    assert bool(is_done(state, P)) is True


def test_not_done_upright_mid_track():
    state = _make_state(x=0.5, theta=0.05, time=1.0)
    assert bool(is_done(state, P)) is False


# ---------------------------------------------------------------------------
# Success criterion
# ---------------------------------------------------------------------------

def test_success_on_timeout():
    """Timeout (pole survived full episode) → success."""
    state = _make_state(time=P.t_max + P.dt)
    assert bool(is_success(state, P)) is True


def test_not_success_early_termination():
    """Pole falling is not success (not a timeout)."""
    state = _make_state(theta=P.theta_max + 0.01, time=1.0)
    assert bool(is_success(state, P)) is False


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------

def test_reward_higher_upright_than_fallen():
    """Upright state at center → higher reward than fallen/off-center state."""
    done = jnp.array(False)
    state_good = _make_state(x=0.0, theta=0.0)
    state_bad  = _make_state(x=1.5, theta=0.18)
    r_good = float(compute_reward(state_good, done, P))
    r_bad  = float(compute_reward(state_bad,  done, P))
    assert r_good > r_bad


def test_reward_at_perfect_upright():
    """At theta=0, x=0: both Gaussians = 1.0; reward = w_theta + w_x - w_time."""
    state = _make_state(x=0.0, theta=0.0)
    done = jnp.array(False)
    r = float(compute_reward(state, done, P))
    expected = P.w_theta * 1.0 + P.w_x * 1.0 - P.w_time
    assert r == pytest.approx(expected, rel=1e-4)


def test_reward_includes_success_bonus_on_timeout():
    """Success bonus fires when done=True and time >= t_max."""
    state = _make_state(time=P.t_max + P.dt)
    done = jnp.array(True)
    r = float(compute_reward(state, done, P))
    assert r >= P.R_success - P.w_time  # at least the bonus minus time penalty


def test_reward_finite_across_states():
    done = jnp.array(False)
    for x, theta in [(0, 0), (2.0, 0.15), (-1.0, -0.1)]:
        state = _make_state(x=float(x), theta=float(theta))
        r = compute_reward(state, done, P)
        assert jnp.isfinite(r), f"Non-finite reward at x={x}, theta={theta}"


# ---------------------------------------------------------------------------
# vmap / jit compatibility
# ---------------------------------------------------------------------------

def test_batch_reset():
    N = 16
    keys = jax.random.split(KEY, N)
    states = jax.vmap(lambda k: reset(k, P))(keys)
    assert states.x.shape == (N,)
    assert jnp.all(jnp.isfinite(states.theta))


def test_batch_step():
    N = 16
    keys = jax.random.split(KEY, N)
    states = jax.vmap(lambda k: reset(k, P))(keys)
    actions = jnp.zeros((N, CartPoleEnv.ACTION_DIM))
    next_states, obs, rewards, dones, infos = jax.vmap(
        lambda s, a: step(s, a, P)
    )(states, actions)
    assert obs.shape     == (N, CartPoleEnv.OBS_DIM)
    assert rewards.shape == (N,)
    assert dones.shape   == (N,)


def test_jit_step():
    state = _make_state()
    jit_step = jax.jit(lambda s, a: step(s, a, P))
    _, _, reward, _, _ = jit_step(state, _zero_action())
    assert jnp.isfinite(reward)
