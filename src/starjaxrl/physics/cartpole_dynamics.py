"""CartPole rigid-body dynamics — pure JAX, jit/vmap compatible."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from omegaconf import DictConfig


class CartPoleState(NamedTuple):
    """Full simulation state. All fields are scalar JAX arrays."""
    x:         jax.Array  # m,     cart position (+ right)
    x_dot:     jax.Array  # m/s,   cart velocity
    theta:     jax.Array  # rad,   pole angle from vertical (0 = upright, + right)
    theta_dot: jax.Array  # rad/s, pole angular velocity (+ counterclockwise)
    time:      jax.Array  # s,     elapsed episode time


class CartPoleParams(NamedTuple):
    """Physics and integration parameters."""
    m_cart: float  # kg, cart mass
    m_pole: float  # kg, pole mass
    l:      float  # m,  half-pole length (pivot to CoM)
    g:      float  # m/s^2, gravitational acceleration
    F_max:  float  # N, maximum force magnitude
    dt:     float  # s, Euler timestep


def params_from_cfg(cfg: DictConfig) -> CartPoleParams:
    """Build CartPoleParams from a Hydra env config node."""
    return CartPoleParams(
        m_cart=float(cfg.m_cart),
        m_pole=float(cfg.m_pole),
        l=float(cfg.l),
        g=float(cfg.g),
        F_max=float(cfg.F_max),
        dt=float(cfg.dt),
    )


DEFAULT_PARAMS = CartPoleParams(
    m_cart=1.0,
    m_pole=0.1,
    l=0.5,
    g=9.81,
    F_max=10.0,
    dt=0.02,
)


def derivatives(
    state: CartPoleState,
    action: jax.Array,
    params: CartPoleParams,
) -> CartPoleState:
    """Compute time derivatives of CartPole state given a force action.

    Action is a 1-element array: [force] clipped to [-F_max, F_max].

    Equations of motion derived from the Lagrangian for a cart-pole system.
    Returns a CartPoleState whose fields are *rates of change* (dx/dt).
    """
    F = jnp.clip(action[0], -params.F_max, params.F_max)

    sin_theta = jnp.sin(state.theta)
    cos_theta = jnp.cos(state.theta)
    M = params.m_cart + params.m_pole

    # Numerator of theta_ddot before dividing by the effective inertia term
    # From Lagrangian: l*(4/3 - m_p*cos^2/M)*theta_ddot = g*sin - cos*(F + m_p*l*theta_dot^2*sin)/M
    temp = F + params.m_pole * params.l * state.theta_dot ** 2 * sin_theta

    theta_ddot = (
        (params.g * sin_theta - cos_theta * temp / M)
        / (params.l * (4.0 / 3.0 - params.m_pole * cos_theta ** 2 / M))
    )
    x_ddot = (temp - params.m_pole * params.l * theta_ddot * cos_theta) / M

    return CartPoleState(
        x=state.x_dot,
        x_dot=x_ddot,
        theta=state.theta_dot,
        theta_dot=theta_ddot,
        time=jnp.ones_like(state.time),
    )


def euler_step(
    state: CartPoleState,
    action: jax.Array,
    params: CartPoleParams,
) -> CartPoleState:
    """Advance state one timestep using forward Euler integration."""
    d = derivatives(state, action, params)
    return CartPoleState(
        x=state.x + d.x * params.dt,
        x_dot=state.x_dot + d.x_dot * params.dt,
        theta=state.theta + d.theta * params.dt,
        theta_dot=state.theta_dot + d.theta_dot * params.dt,
        time=state.time + params.dt,
    )
