"""Starship landing environment — pure JAX, stateless functional API."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from omegaconf import DictConfig

from starjaxrl.physics.dynamics import (
    StarshipParams,
    StarshipState,
    euler_step,
)


# ---------------------------------------------------------------------------
# Environment configuration
# ---------------------------------------------------------------------------

class EnvParams(NamedTuple):
    """All environment parameters: physics, ICs, termination, reward."""

    # --- Physics (mirrors StarshipParams) ---
    m_dry: float
    m_prop_max: float
    T_max: float
    Isp: float
    T_min: float
    delta_max: float
    L: float
    g: float
    dt: float

    # --- Initial conditions ---
    x0: float
    y0: float
    vx0: float
    vy0: float
    theta0: float
    omega0: float
    mprop0: float

    # --- Termination thresholds ---
    y_catch: float
    x_max: float
    theta_max: float
    t_max: float

    # --- Success tolerances ---
    success_x_tol: float
    success_vy_tol: float
    success_vx_tol: float
    success_theta_tol: float

    # --- Reward weights ---
    w_x: float
    w_vy: float
    w_vx: float
    w_theta: float
    R_success: float


class StepInfo(NamedTuple):
    """Fixed-structure info dict (required for lax.scan compatibility)."""
    success: jax.Array  # bool scalar


def env_params_from_cfg(cfg: DictConfig) -> EnvParams:
    """Build EnvParams from a Hydra env config node."""
    return EnvParams(**{field: float(cfg[field]) for field in EnvParams._fields})


def to_physics_params(params: EnvParams) -> StarshipParams:
    """Extract the physics sub-config from EnvParams."""
    return StarshipParams(
        m_dry=params.m_dry,
        m_prop_max=params.m_prop_max,
        T_max=params.T_max,
        Isp=params.Isp,
        T_min=params.T_min,
        delta_max=params.delta_max,
        L=params.L,
        g=params.g,
        dt=params.dt,
    )


# Default parameters matching configs/env.yaml
DEFAULT_ENV_PARAMS = EnvParams(
    m_dry=100_000.0, m_prop_max=1_200_000.0, T_max=6_000_000.0,
    Isp=330.0, T_min=0.4, delta_max=0.35, L=50.0, g=9.81, dt=0.05,
    x0=0.0, y0=3000.0, vx0=0.0, vy0=-80.0,
    theta0=jnp.pi / 2, omega0=0.0, mprop0=1.0,
    y_catch=50.0, x_max=500.0, theta_max=3 * jnp.pi / 2, t_max=120.0,
    success_x_tol=1.0, success_vy_tol=2.0, success_vx_tol=1.0,
    success_theta_tol=0.175,
    w_x=0.01, w_vy=0.01, w_vx=0.01, w_theta=0.01, R_success=100.0,
)


# ---------------------------------------------------------------------------
# Core environment functions
# ---------------------------------------------------------------------------

def reset(key: jax.Array, params: EnvParams) -> StarshipState:
    """Return the fixed initial state. key is reserved for future randomization."""
    del key  # not used yet — placeholder for randomized ICs
    return StarshipState(
        x=jnp.array(params.x0),
        y=jnp.array(params.y0),
        vx=jnp.array(params.vx0),
        vy=jnp.array(params.vy0),
        theta=jnp.array(params.theta0),
        omega=jnp.array(params.omega0),
        mprop=jnp.array(params.mprop0),
        time=jnp.zeros(()),
    )


def get_obs(state: StarshipState) -> jax.Array:
    """Extract the 7-element observation vector from the state."""
    return jnp.array([
        state.x,
        state.y,
        state.vx,
        state.vy,
        state.theta,
        state.omega,
        state.mprop,
    ])


def is_done(state: StarshipState, params: EnvParams) -> jax.Array:
    """Return True if the episode should terminate."""
    at_catch   = state.y <= params.y_catch
    out_bounds = jnp.abs(state.x) > params.x_max
    tumbling   = jnp.abs(state.theta) > params.theta_max
    no_fuel    = state.mprop <= 0.0
    timeout    = state.time >= params.t_max
    return at_catch | out_bounds | tumbling | no_fuel | timeout


def is_success(state: StarshipState, params: EnvParams) -> jax.Array:
    """Return True if the terminal state satisfies landing tolerances."""
    x_ok     = jnp.abs(state.x)     <= params.success_x_tol
    vy_ok    = jnp.abs(state.vy)    <= params.success_vy_tol
    vx_ok    = jnp.abs(state.vx)    <= params.success_vx_tol
    theta_ok = jnp.abs(state.theta) <= params.success_theta_tol
    return x_ok & vy_ok & vx_ok & theta_ok


def compute_reward(
    next_state: StarshipState,
    done: jax.Array,
    params: EnvParams,
) -> jax.Array:
    """
    Dense shaping reward every step plus a sparse success bonus on landing.

    r(t) = -w_x*|x| - w_vy*|vy| - w_vx*|vx| - w_theta*|theta|
           + R_success  (if done and within tolerances)
    """
    dense = -(
        params.w_x     * jnp.abs(next_state.x)
        + params.w_vy  * jnp.abs(next_state.vy)
        + params.w_vx  * jnp.abs(next_state.vx)
        + params.w_theta * jnp.abs(next_state.theta)
    )
    bonus = jnp.where(done & is_success(next_state, params), params.R_success, 0.0)
    return dense + bonus


def step(
    state: StarshipState,
    action: jax.Array,
    params: EnvParams,
) -> tuple[StarshipState, jax.Array, jax.Array, jax.Array, StepInfo]:
    """
    Advance the environment one timestep.

    Returns: (next_state, obs, reward, done, info)
    """
    physics = to_physics_params(params)
    next_state = euler_step(state, action, physics)

    done   = is_done(next_state, params)
    reward = compute_reward(next_state, done, params)
    obs    = get_obs(next_state)
    info   = StepInfo(success=is_success(next_state, params) & done)

    return next_state, obs, reward, done, info


# ---------------------------------------------------------------------------
# Convenience class (thin namespace wrapper)
# ---------------------------------------------------------------------------

class StarshipEnv:
    """Stateless namespace class — all methods are module-level functions."""

    reset          = staticmethod(reset)
    get_obs        = staticmethod(get_obs)
    step           = staticmethod(step)
    is_done        = staticmethod(is_done)
    is_success     = staticmethod(is_success)
    compute_reward = staticmethod(compute_reward)

    OBS_DIM    = 7
    ACTION_DIM = 2
