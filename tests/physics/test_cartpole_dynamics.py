"""Tests for CartPole physics dynamics."""

import jax
import jax.numpy as jnp
import pytest

from starjaxrl.physics.cartpole_dynamics import (
    CartPoleParams,
    CartPoleState,
    DEFAULT_PARAMS,
    derivatives,
    euler_step,
)

KEY = jax.random.PRNGKey(0)
P = DEFAULT_PARAMS


def _make_state(**overrides) -> CartPoleState:
    base = dict(x=0.0, x_dot=0.0, theta=0.0, theta_dot=0.0, time=0.0)
    base.update(overrides)
    return CartPoleState(**{k: jnp.array(v, dtype=jnp.float32) for k, v in base.items()})


def _zero_action():
    return jnp.zeros(1)


# ---------------------------------------------------------------------------
# derivatives
# ---------------------------------------------------------------------------

def test_derivatives_shape():
    state = _make_state()
    d = derivatives(state, _zero_action(), P)
    assert d.x.shape == ()
    assert d.x_dot.shape == ()
    assert d.theta.shape == ()
    assert d.theta_dot.shape == ()


def test_derivatives_zero_action_upright():
    """At theta=0, zero action → no angular acceleration, no linear acceleration."""
    state = _make_state(theta=0.0, theta_dot=0.0)
    d = derivatives(state, _zero_action(), P)
    assert float(d.theta) == pytest.approx(0.0)       # theta_dot = 0
    assert float(d.theta_dot) == pytest.approx(0.0)   # pole is balanced
    assert float(d.x_dot) == pytest.approx(0.0)       # F=0, pole=0 → no x accel


def test_derivatives_tilted_pole_falls():
    """A tilted pole with no action should have angular acceleration away from upright."""
    state = _make_state(theta=0.1, theta_dot=0.0)
    d = derivatives(state, _zero_action(), P)
    # Positive theta (tilted right) → positive theta_ddot (falls further right)
    assert float(d.theta_dot) > 0.0


def test_derivatives_force_accelerates_cart():
    """Positive force → positive x_ddot."""
    state = _make_state(theta=0.0, theta_dot=0.0)
    action = jnp.array([5.0])
    d = derivatives(state, action, P)
    assert float(d.x_dot) > 0.0


def test_derivatives_force_clipped():
    """Force is clipped to F_max; very large actions produce same result as F_max."""
    state = _make_state()
    d_max = derivatives(state, jnp.array([P.F_max]), P)
    d_huge = derivatives(state, jnp.array([1e6]), P)
    assert float(d_max.x_dot) == pytest.approx(float(d_huge.x_dot), rel=1e-5)


def test_derivatives_finite():
    """Derivatives are finite for typical states."""
    state = _make_state(x=1.0, x_dot=0.5, theta=0.1, theta_dot=0.2)
    d = derivatives(state, jnp.array([3.0]), P)
    for field in d:
        assert jnp.isfinite(field), f"Non-finite derivative: {field}"


def test_derivatives_time_rate_is_one():
    """Time derivative is always 1 (dt is handled in euler_step)."""
    state = _make_state()
    d = derivatives(state, _zero_action(), P)
    assert float(d.time) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# euler_step
# ---------------------------------------------------------------------------

def test_euler_step_time_advances():
    state = _make_state(time=0.0)
    next_state = euler_step(state, _zero_action(), P)
    assert float(next_state.time) == pytest.approx(P.dt)


def test_euler_step_position_updates():
    """Non-zero velocity → position changes."""
    state = _make_state(x=0.0, x_dot=1.0)
    next_state = euler_step(state, _zero_action(), P)
    assert float(next_state.x) == pytest.approx(P.dt * 1.0, rel=1e-4)


def test_euler_step_pole_falls_without_control():
    """Tilted pole without control gains angular velocity, then tilts further."""
    state = _make_state(theta=0.05, theta_dot=0.0)
    # After one step: theta_dot increases (due to gravity), but theta only moves by old theta_dot*dt
    next_state = euler_step(state, _zero_action(), P)
    assert float(next_state.theta_dot) > float(state.theta_dot)  # angular vel grows
    # After two steps: theta itself has moved
    next_next_state = euler_step(next_state, _zero_action(), P)
    assert float(next_next_state.theta) > float(state.theta)


def test_euler_step_jit_compatible():
    state = _make_state()
    jit_step = jax.jit(lambda s, a: euler_step(s, a, P))
    next_state = jit_step(state, _zero_action())
    assert jnp.isfinite(next_state.x)


def test_euler_step_vmap_compatible():
    """Batch of states steps in parallel."""
    N = 8
    xs = jnp.linspace(-1.0, 1.0, N)
    states = jax.vmap(lambda xi: _make_state(x=xi))(xs)
    actions = jnp.zeros((N, 1))
    next_states = jax.vmap(lambda s, a: euler_step(s, a, P))(states, actions)
    assert next_states.x.shape == (N,)
    assert jnp.all(jnp.isfinite(next_states.x))


def test_euler_step_symmetric():
    """Equal and opposite forces from theta=0 produce mirrored x_dot."""
    state = _make_state(theta=0.0)
    s_pos = euler_step(state, jnp.array([ 5.0]), P)
    s_neg = euler_step(state, jnp.array([-5.0]), P)
    assert float(s_pos.x_dot) == pytest.approx(-float(s_neg.x_dot), rel=1e-5)
