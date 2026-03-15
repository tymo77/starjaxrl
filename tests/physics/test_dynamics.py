"""Tests for Starship physics dynamics."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from starjaxrl.physics import (
    DEFAULT_PARAMS,
    StarshipState,
    derivatives,
    euler_step,
    hover_throttle,
    params_from_cfg,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(**overrides) -> StarshipState:
    """Default state: vertical, stationary, full fuel at altitude."""
    base = dict(
        x=0.0, y=3000.0, vx=0.0, vy=0.0,
        theta=0.0, omega=0.0, mprop=1.0, time=0.0,
    )
    base.update(overrides)
    return StarshipState(**{k: jnp.array(v, dtype=jnp.float32) for k, v in base.items()})


def _zero_action() -> jax.Array:
    """Zero throttle, zero gimbal. Engine off (below T_min floor)."""
    return jnp.zeros(2)


def _hover_action(state: StarshipState) -> jax.Array:
    """Throttle for zero net vertical accel at theta=0, zero gimbal."""
    T = hover_throttle(state, DEFAULT_PARAMS)
    return jnp.array([T, 0.0])


# ---------------------------------------------------------------------------
# params_from_cfg
# ---------------------------------------------------------------------------

def test_params_from_cfg(env_cfg):
    params = params_from_cfg(env_cfg)
    assert params.g == pytest.approx(env_cfg.g)
    assert params.T_max == pytest.approx(env_cfg.T_max)
    assert params.dt == pytest.approx(env_cfg.dt)


# ---------------------------------------------------------------------------
# Free-fall (no thrust)
# ---------------------------------------------------------------------------

def test_freefall_vertical_acceleration():
    """With no thrust, vy should decrease at -g each second."""
    state = _make_state(vy=0.0, theta=0.0)
    action = _zero_action()
    d = derivatives(state, action, DEFAULT_PARAMS)
    assert d.vy == pytest.approx(-DEFAULT_PARAMS.g, rel=1e-5)


def test_freefall_no_horizontal_accel():
    """No thrust and theta=0 → zero horizontal acceleration."""
    state = _make_state(vx=0.0, theta=0.0)
    action = _zero_action()
    d = derivatives(state, action, DEFAULT_PARAMS)
    assert d.vx == pytest.approx(0.0, abs=1e-6)


def test_freefall_no_angular_accel():
    """No thrust → zero torque → zero angular acceleration."""
    state = _make_state()
    action = _zero_action()
    d = derivatives(state, action, DEFAULT_PARAMS)
    assert d.omega == pytest.approx(0.0, abs=1e-6)


def test_freefall_no_mass_change():
    """No thrust → no propellant consumption."""
    state = _make_state()
    action = _zero_action()
    d = derivatives(state, action, DEFAULT_PARAMS)
    assert d.mprop == pytest.approx(0.0, abs=1e-10)


def test_freefall_trajectory():
    """After N steps with no thrust, altitude drops and vy grows negative."""
    state = _make_state(vy=0.0)
    action = _zero_action()
    for _ in range(20):
        state = euler_step(state, action, DEFAULT_PARAMS)
    assert float(state.vy) < 0.0
    assert float(state.y) < 3000.0


# ---------------------------------------------------------------------------
# Hover
# ---------------------------------------------------------------------------

def test_hover_zero_vertical_accel():
    """Hover throttle produces zero net vertical acceleration at theta=0.

    At full propellant Starship's T/W < 1 — hover is only feasible
    during the late landing burn when most fuel has been expended.
    """
    state = _make_state(theta=0.0, mprop=0.2)  # late-burn fuel state
    action = _hover_action(state)
    d = derivatives(state, action, DEFAULT_PARAMS)
    assert d.vy == pytest.approx(0.0, abs=1.0)  # within 1 m/s^2


def test_hover_throttle_in_range():
    """Hover throttle must be between T_min and 1.0 for a feasible hover.

    This is only physically achievable at low fuel fractions.
    At full propellant T/W ≈ 0.47, which is intentional (suicide-burn design).
    """
    state = _make_state(mprop=0.2, theta=0.0)
    T = hover_throttle(state, DEFAULT_PARAMS)
    assert DEFAULT_PARAMS.T_min <= T <= 1.0


# ---------------------------------------------------------------------------
# Thrust direction (theta / gimbal)
# ---------------------------------------------------------------------------

def test_vertical_thrust_no_horizontal_force():
    """theta=0, delta=0 → thrust is purely vertical (ax=0).

    At low fuel the vehicle is light enough for net upward acceleration.
    At full fuel T/W < 1, so we only assert no horizontal component.
    """
    state = _make_state(theta=0.0, mprop=0.1)  # nearly empty — T/W > 1
    action = jnp.array([1.0, 0.0])
    d = derivatives(state, action, DEFAULT_PARAMS)
    assert d.vx == pytest.approx(0.0, abs=1e-4)
    assert d.vy > 0.0  # net upward accel at full throttle, low mass


def test_horizontal_thrust_direction():
    """theta=pi/2, delta=0 → thrust is purely horizontal."""
    state = _make_state(theta=jnp.pi / 2)
    action = jnp.array([1.0, 0.0])
    d = derivatives(state, action, DEFAULT_PARAMS)
    assert d.vy == pytest.approx(-DEFAULT_PARAMS.g, abs=1.0)  # only gravity in y
    assert abs(float(d.vx)) > 0.0  # thrust in x


def test_gimbal_creates_horizontal_force():
    """theta=0, delta!=0 → some horizontal thrust component."""
    state = _make_state(theta=0.0)
    action = jnp.array([1.0, 0.1])
    d = derivatives(state, action, DEFAULT_PARAMS)
    assert abs(float(d.vx)) > 0.0


# ---------------------------------------------------------------------------
# Angular dynamics
# ---------------------------------------------------------------------------

def test_zero_gimbal_no_torque():
    """delta=0 → zero torque regardless of throttle."""
    state = _make_state()
    action = jnp.array([1.0, 0.0])
    d = derivatives(state, action, DEFAULT_PARAMS)
    assert d.omega == pytest.approx(0.0, abs=1e-6)


def test_positive_gimbal_positive_torque():
    """Positive gimbal angle → positive angular acceleration."""
    state = _make_state()
    action = jnp.array([1.0, 0.2])
    d = derivatives(state, action, DEFAULT_PARAMS)
    assert float(d.omega) > 0.0


def test_negative_gimbal_negative_torque():
    """Negative gimbal angle → negative angular acceleration."""
    state = _make_state()
    action = jnp.array([1.0, -0.2])
    d = derivatives(state, action, DEFAULT_PARAMS)
    assert float(d.omega) < 0.0


# ---------------------------------------------------------------------------
# Propellant / mass depletion
# ---------------------------------------------------------------------------

def test_mass_depletes_with_thrust():
    """Non-zero throttle → propellant decreases over time."""
    state = _make_state(mprop=1.0)
    action = jnp.array([1.0, 0.0])
    d = derivatives(state, action, DEFAULT_PARAMS)
    assert float(d.mprop) < 0.0


def test_mass_monotonically_decreasing():
    """mprop decreases monotonically during a burn."""
    state = _make_state(mprop=1.0)
    action = jnp.array([0.8, 0.0])
    mprop_prev = float(state.mprop)
    for _ in range(10):
        state = euler_step(state, action, DEFAULT_PARAMS)
        assert float(state.mprop) <= mprop_prev
        mprop_prev = float(state.mprop)


def test_mprop_never_negative():
    """mprop is clamped to 0 and never goes below zero."""
    state = _make_state(mprop=0.001)
    action = jnp.array([1.0, 0.0])
    for _ in range(50):
        state = euler_step(state, action, DEFAULT_PARAMS)
    assert float(state.mprop) >= 0.0


def test_engine_cut_when_empty():
    """When mprop=0, thrust is cut and vy changes only due to gravity."""
    state = _make_state(mprop=0.0, vy=0.0, theta=0.0)
    action = jnp.array([1.0, 0.0])  # command full throttle
    d = derivatives(state, action, DEFAULT_PARAMS)
    # With no fuel, should behave like free-fall
    assert d.vy == pytest.approx(-DEFAULT_PARAMS.g, rel=1e-5)
    assert d.vx == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Numerical sanity
# ---------------------------------------------------------------------------

def test_no_nan_random_states():
    """Random actions and states should never produce NaN derivatives."""
    key = jax.random.PRNGKey(0)
    for _ in range(50):
        key, k1, k2 = jax.random.split(key, 3)
        state = StarshipState(
            x=jax.random.uniform(k1, (), minval=-100, maxval=100),
            y=jax.random.uniform(k1, (), minval=100, maxval=5000),
            vx=jax.random.uniform(k1, (), minval=-50, maxval=50),
            vy=jax.random.uniform(k1, (), minval=-100, maxval=10),
            theta=jax.random.uniform(k1, (), minval=-jnp.pi, maxval=jnp.pi),
            omega=jax.random.uniform(k1, (), minval=-1, maxval=1),
            mprop=jax.random.uniform(k1, (), minval=0, maxval=1),
            time=jax.random.uniform(k1, (), minval=0, maxval=60),
        )
        action = jax.random.uniform(k2, (2,), minval=-1, maxval=1)
        d = derivatives(state, action, DEFAULT_PARAMS)
        for field in d:
            assert jnp.isfinite(field), f"NaN/Inf in derivative field"


# ---------------------------------------------------------------------------
# JIT and vmap
# ---------------------------------------------------------------------------

def test_derivatives_jittable():
    state = _make_state()
    action = jnp.array([0.5, 0.0])
    jit_fn = jax.jit(derivatives, static_argnums=(2,))
    d = jit_fn(state, action, DEFAULT_PARAMS)
    assert jnp.isfinite(d.vy)


def test_euler_step_jittable():
    state = _make_state()
    action = jnp.array([0.5, 0.0])
    jit_fn = jax.jit(euler_step, static_argnums=(2,))
    new_state = jit_fn(state, action, DEFAULT_PARAMS)
    assert jnp.isfinite(new_state.y)


def test_euler_step_vmappable():
    """vmap over a batch of 16 independent states."""
    N = 16
    states = StarshipState(
        x=jnp.zeros(N),
        y=jnp.full(N, 3000.0),
        vx=jnp.zeros(N),
        vy=jnp.full(N, -80.0),
        theta=jnp.full(N, jnp.pi / 2),
        omega=jnp.zeros(N),
        mprop=jnp.ones(N),
        time=jnp.zeros(N),
    )
    actions = jnp.tile(jnp.array([0.5, 0.0]), (N, 1))

    batch_step = jax.vmap(lambda s, a: euler_step(s, a, DEFAULT_PARAMS))
    new_states = batch_step(states, actions)

    assert new_states.y.shape == (N,)
    assert jnp.all(jnp.isfinite(new_states.y))


def test_time_advances():
    """time field increments by dt each step."""
    state = _make_state(time=0.0)
    action = _zero_action()
    new_state = euler_step(state, action, DEFAULT_PARAMS)
    assert float(new_state.time) == pytest.approx(DEFAULT_PARAMS.dt, rel=1e-6)
