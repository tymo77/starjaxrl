"""Slow-approach environment — reach a target point with near-zero velocity.

The goal is simple: starting from the usual Starship initial conditions, slow
down and arrive at a configurable target point (x_target, y_target) with
near-zero velocity.  The target coordinates are part of the observation so the
policy can generalise across different target positions.

Observation (9-D):
    [x/500, y/3000, vx/100, vy/100, theta/π, omega/2, mprop,
     x_target/500, y_target/3000]

Done conditions (any one triggers episode end):
    • y ≤ y_catch             — hit the ground / catch-arm level
    • |x| > x_max             — left the horizontal flight corridor
    • |theta| > theta_max     — tumbling
    • mprop ≤ 0               — out of propellant
    • time ≥ t_max            — episode timeout
    • dist(pos, target) ≤ success_dist_tol  — arrived at target

Success: episode ended by arrival AND speed ≤ success_vel_tol.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from omegaconf import DictConfig

from starjaxrl.physics.dynamics import (
    StarshipParams,
    StarshipState,
    euler_step,
)
from starjaxrl.env.types import StepInfo
from starjaxrl.env.reward_utils import gauss


# ---------------------------------------------------------------------------
# Environment configuration
# ---------------------------------------------------------------------------

class SlowApproachEnvParams(NamedTuple):
    """All environment parameters: physics, ICs, target, termination, reward."""

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

    # --- Target point (where the robot should arrive with ~zero velocity) ---
    x_target: float
    y_target: float

    # --- Termination thresholds ---
    y_catch: float
    x_max: float
    theta_max: float
    t_max: float

    # --- Success tolerances ---
    success_dist_tol: float   # m   — max distance to target at arrival
    success_vel_tol: float    # m/s — max speed at arrival

    # --- Reward: Gaussian shaping weights ---
    w_dist: float   # proximity to target
    w_vel:  float   # low speed

    # --- Reward: Gaussian spreads ---
    sigma_dist: float   # m   — 1-sigma distance where reward ≈ 0.6
    sigma_vel:  float   # m/s — 1-sigma speed where reward ≈ 0.6

    # --- Reward: time penalty and success bonus ---
    w_time:    float
    R_success: float


def env_params_from_cfg(cfg: DictConfig) -> SlowApproachEnvParams:
    """Build SlowApproachEnvParams from a Hydra env config node."""
    return SlowApproachEnvParams(
        **{field: float(cfg[field]) for field in SlowApproachEnvParams._fields}
    )


def to_physics_params(params: SlowApproachEnvParams) -> StarshipParams:
    """Extract the physics sub-config from SlowApproachEnvParams."""
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


DEFAULT_ENV_PARAMS = SlowApproachEnvParams(
    # Physics
    m_dry=100_000.0, m_prop_max=12_000.0, T_max=6_000_000.0,
    Isp=330.0, T_min=0.4, delta_max=0.35, L=50.0, g=9.81, dt=0.05,
    # Initial conditions
    x0=0.0, y0=3000.0, vx0=0.0, vy0=-80.0,
    theta0=jnp.pi / 2, omega0=0.0, mprop0=1.0,
    # Target
    x_target=0.0, y_target=500.0,
    # Termination
    y_catch=50.0, x_max=500.0, theta_max=3 * jnp.pi / 2, t_max=120.0,
    # Success
    success_dist_tol=20.0, success_vel_tol=5.0,
    # Reward
    w_dist=1.0, w_vel=1.0,
    sigma_dist=100.0, sigma_vel=30.0,
    w_time=0.01, R_success=100.0,
)


# ---------------------------------------------------------------------------
# Core environment functions
# ---------------------------------------------------------------------------

def reset(key: jax.Array, params: SlowApproachEnvParams) -> StarshipState:
    """Return the fixed initial state."""
    del key  # reserved for future randomised ICs
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


def get_obs(state: StarshipState, params: SlowApproachEnvParams) -> jax.Array:
    """Extract and normalize the 9-element observation vector.

    The first 7 elements are the current pose (same normalisation as the
    standard Starship env).  The final 2 elements are the target coordinates,
    normalized by the same scale factors so the network sees comparable
    magnitudes regardless of physical units.
    """
    return jnp.array([
        state.x     / 500.0,           # x:        ±500 m    → ±1
        state.y     / 3000.0,          # y:        0–3000 m  → 0–1
        state.vx    / 100.0,           # vx:       ±100 m/s  → ±1
        state.vy    / 100.0,           # vy:       ±100 m/s  → ±1
        state.theta / jnp.pi,          # theta:    [0, 2π]   → [0, 2]
        state.omega / 2.0,             # omega:    ±2 rad/s  → ±1
        state.mprop,                   # mprop:    [0, 1]    — already normalised
        params.x_target / 500.0,       # x_target: ±500 m    → ±1
        params.y_target / 3000.0,      # y_target: 0–3000 m  → 0–1
    ])


def _dist_to_target(state: StarshipState, params: SlowApproachEnvParams) -> jax.Array:
    """Euclidean distance from current position to target."""
    dx = state.x - params.x_target
    dy = state.y - params.y_target
    return jnp.sqrt(dx ** 2 + dy ** 2)


def _speed(state: StarshipState) -> jax.Array:
    """Total translational speed."""
    return jnp.sqrt(state.vx ** 2 + state.vy ** 2)


def is_done(state: StarshipState, params: SlowApproachEnvParams) -> jax.Array:
    """Return True if the episode should terminate."""
    at_catch    = state.y <= params.y_catch
    out_bounds  = jnp.abs(state.x) > params.x_max
    tumbling    = jnp.abs(state.theta) > params.theta_max
    no_fuel     = state.mprop <= 0.0
    timeout     = state.time >= params.t_max
    near_target = _dist_to_target(state, params) <= params.success_dist_tol
    return at_catch | out_bounds | tumbling | no_fuel | timeout | near_target


def is_success(state: StarshipState, params: SlowApproachEnvParams) -> jax.Array:
    """Return True if the terminal state counts as a successful slow approach."""
    near = _dist_to_target(state, params) <= params.success_dist_tol
    slow = _speed(state) <= params.success_vel_tol
    return near & slow


def compute_reward(
    next_state: StarshipState,
    done: jax.Array,
    params: SlowApproachEnvParams,
) -> jax.Array:
    """Dense shaping on distance + speed, time penalty, sparse success bonus.

    r(t) = w_dist * gauss(dist_to_target, sigma_dist)
         + w_vel  * gauss(speed,          sigma_vel)
         - w_time
         + R_success   (if done and slow near target)

    Both Gaussians peak at 1.0 when the argument is 0, so the maximum dense
    reward per step is (w_dist + w_vel).
    """
    dist = _dist_to_target(next_state, params)
    spd  = _speed(next_state)

    dense = (
        params.w_dist * gauss(dist, params.sigma_dist)
        + params.w_vel  * gauss(spd,  params.sigma_vel)
        - params.w_time
    )
    success_bonus = jnp.where(
        done & is_success(next_state, params), params.R_success, 0.0
    )
    return dense + success_bonus


def step(
    state: StarshipState,
    action: jax.Array,
    params: SlowApproachEnvParams,
) -> tuple[StarshipState, jax.Array, jax.Array, jax.Array, StepInfo]:
    """Advance the environment one timestep.

    Returns: (next_state, obs, reward, done, info)
    """
    physics    = to_physics_params(params)
    next_state = euler_step(state, action, physics)

    done   = is_done(next_state, params)
    reward = compute_reward(next_state, done, params)
    obs    = get_obs(next_state, params)
    info   = StepInfo(success=is_success(next_state, params) & done)

    return next_state, obs, reward, done, info


# ---------------------------------------------------------------------------
# Convenience class (thin namespace wrapper)
# ---------------------------------------------------------------------------

class SlowApproachEnv:
    """Stateless namespace class — all methods are module-level functions."""

    reset          = staticmethod(reset)
    get_obs        = staticmethod(get_obs)
    step           = staticmethod(step)
    is_done        = staticmethod(is_done)
    is_success     = staticmethod(is_success)
    compute_reward = staticmethod(compute_reward)

    OBS_DIM    = 9   # 7 pose + 2 target
    ACTION_DIM = 2
