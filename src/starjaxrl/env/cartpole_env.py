"""CartPole balancing environment — pure JAX, stateless functional API."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from omegaconf import DictConfig

from starjaxrl.physics.cartpole_dynamics import (
    CartPoleParams,
    CartPoleState,
    euler_step,
)
from starjaxrl.env.types import StepInfo
from starjaxrl.env.reward_utils import gauss


# ---------------------------------------------------------------------------
# Environment configuration
# ---------------------------------------------------------------------------

class CartPoleEnvParams(NamedTuple):
    """All CartPole environment parameters: physics, ICs, termination, reward."""

    # --- Physics (mirrors CartPoleParams) ---
    m_cart: float
    m_pole: float
    l:      float
    g:      float
    F_max:  float
    dt:     float

    # --- Initial conditions ---
    x0:         float
    x_dot0:     float
    theta0:     float
    theta_dot0: float

    # --- Termination thresholds ---
    x_max:     float
    theta_max: float
    t_max:     float

    # --- Reward: Gaussian shaping weights ---
    w_theta: float
    w_x:     float

    # --- Reward: Gaussian spreads (1-sigma = value where reward ≈ 0.6) ---
    sigma_theta: float
    sigma_x:     float

    # --- Reward: time penalty and success bonus ---
    w_time:    float
    R_success: float


def env_params_from_cfg(cfg: DictConfig) -> CartPoleEnvParams:
    """Build CartPoleEnvParams from a Hydra env config node."""
    return CartPoleEnvParams(**{field: float(cfg[field]) for field in CartPoleEnvParams._fields})


def to_physics_params(params: CartPoleEnvParams) -> CartPoleParams:
    """Extract the physics sub-config from CartPoleEnvParams."""
    return CartPoleParams(
        m_cart=params.m_cart,
        m_pole=params.m_pole,
        l=params.l,
        g=params.g,
        F_max=params.F_max,
        dt=params.dt,
    )


# Default parameters matching configs/env/cartpole.yaml
DEFAULT_ENV_PARAMS = CartPoleEnvParams(
    m_cart=1.0, m_pole=0.1, l=0.5, g=9.81, F_max=10.0, dt=0.02,
    x0=0.0, x_dot0=0.0, theta0=0.05, theta_dot0=0.0,
    x_max=2.4, theta_max=0.2094, t_max=10.0,
    w_theta=2.0, w_x=0.5,
    sigma_theta=0.1, sigma_x=1.0,
    w_time=0.01, R_success=100.0,
)


# ---------------------------------------------------------------------------
# Core environment functions
# ---------------------------------------------------------------------------

def reset(key: jax.Array, params: CartPoleEnvParams) -> CartPoleState:
    """Return the initial state with a small random perturbation.

    A slight random angle offset (±0.05 rad) prevents the policy from
    learning a trivial solution that only works from a perfectly balanced start.
    """
    perturb = jax.random.uniform(key, minval=-0.05, maxval=0.05)
    return CartPoleState(
        x=jnp.array(params.x0),
        x_dot=jnp.array(params.x_dot0),
        theta=jnp.array(params.theta0) + perturb,
        theta_dot=jnp.array(params.theta_dot0),
        time=jnp.zeros(()),
    )


def get_obs(state: CartPoleState) -> jax.Array:
    """Extract and normalize the 4-element observation vector.

    Components are scaled to roughly [-1, 1]:
      x / x_max_scale  (cart position)
      x_dot / v_scale  (cart velocity)
      theta / pi       (pole angle — 0=upright)
      theta_dot / w_scale (angular velocity)
    """
    return jnp.array([
        state.x         / 2.4,    # x:         ±2.4 m    → ±1
        state.x_dot     / 5.0,    # x_dot:     ±5 m/s    → ±1
        state.theta     / 0.2094, # theta:     ±0.21 rad → ±1
        state.theta_dot / 5.0,    # theta_dot: ±5 rad/s  → ±1
    ])


def is_done(state: CartPoleState, params: CartPoleEnvParams) -> jax.Array:
    """Return True if the episode should terminate."""
    pole_fell  = jnp.abs(state.theta) >= params.theta_max
    out_bounds = jnp.abs(state.x)     >= params.x_max
    timeout    = state.time           >= params.t_max
    return pole_fell | out_bounds | timeout


def is_success(state: CartPoleState, params: CartPoleEnvParams) -> jax.Array:
    """Return True if the episode ended due to timeout (pole balanced for full duration)."""
    return state.time >= params.t_max


def compute_reward(
    next_state: CartPoleState,
    done: jax.Array,
    params: CartPoleEnvParams,
) -> jax.Array:
    """Gaussian shaping + time penalty + sparse success bonus.

    r(t) = w_theta * gauss(theta, sigma_theta)
         + w_x     * gauss(x,     sigma_x)
         - w_time
         + R_success   (if done due to timeout — pole survived full episode)

    Each Gaussian returns 1.0 at the target and smoothly falls to 0
    as the state moves away, with the 1-sigma point set by sigma_* params.
    """
    dense = (
        params.w_theta * gauss(next_state.theta, params.sigma_theta)
        + params.w_x   * gauss(next_state.x,     params.sigma_x)
        - params.w_time
    )
    success_bonus = jnp.where(
        done & is_success(next_state, params), params.R_success, 0.0
    )
    return dense + success_bonus


def step(
    state: CartPoleState,
    action: jax.Array,
    params: CartPoleEnvParams,
) -> tuple[CartPoleState, jax.Array, jax.Array, jax.Array, StepInfo]:
    """Advance the environment one timestep.

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

class CartPoleEnv:
    """Stateless namespace class — all methods are module-level functions."""

    reset          = staticmethod(reset)
    get_obs        = staticmethod(get_obs)
    step           = staticmethod(step)
    is_done        = staticmethod(is_done)
    is_success     = staticmethod(is_success)
    compute_reward = staticmethod(compute_reward)

    OBS_DIM    = 4
    ACTION_DIM = 1
