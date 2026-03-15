"""2D Starship rigid-body dynamics — pure JAX, jit/vmap compatible."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from omegaconf import DictConfig


class StarshipState(NamedTuple):
    """Full simulation state. All fields are scalar JAX arrays."""
    x: jax.Array       # m,   horizontal position (+ right)
    y: jax.Array       # m,   altitude (+ up)
    vx: jax.Array      # m/s, horizontal velocity
    vy: jax.Array      # m/s, vertical velocity (negative = falling)
    theta: jax.Array   # rad, pitch from vertical (0 = nose-up, pi/2 = belly-down)
    omega: jax.Array   # rad/s, angular velocity (+ counterclockwise)
    mprop: jax.Array   # [-], propellant fraction in [0, 1]
    time: jax.Array    # s,   elapsed episode time


class StarshipParams(NamedTuple):
    """Physics and integration parameters. Passed explicitly so vmap can batch over them."""
    m_dry: float        # kg
    m_prop_max: float   # kg
    T_max: float        # N
    Isp: float          # s
    T_min: float        # throttle fraction floor
    delta_max: float    # rad, max gimbal angle
    L: float            # m, vehicle length
    g: float            # m/s^2
    dt: float           # s, Euler timestep


def params_from_cfg(cfg: DictConfig) -> StarshipParams:
    """Build StarshipParams from a Hydra env config node."""
    return StarshipParams(
        m_dry=float(cfg.m_dry),
        m_prop_max=float(cfg.m_prop_max),
        T_max=float(cfg.T_max),
        Isp=float(cfg.Isp),
        T_min=float(cfg.T_min),
        delta_max=float(cfg.delta_max),
        L=float(cfg.L),
        g=float(cfg.g),
        dt=float(cfg.dt),
    )


# Default parameters matching configs/env.yaml
DEFAULT_PARAMS = StarshipParams(
    m_dry=100_000.0,
    m_prop_max=1_200_000.0,
    T_max=6_000_000.0,
    Isp=330.0,
    T_min=0.4,
    delta_max=0.35,
    L=50.0,
    g=9.81,
    dt=0.05,
)


def derivatives(
    state: StarshipState,
    action: jax.Array,
    params: StarshipParams,
) -> StarshipState:
    """
    Compute time derivatives of state given action.

    Action is a 2-element array: [throttle, gimbal].
    Throttle is clipped to [T_min, 1]; gimbal to [-delta_max, delta_max].
    When mprop == 0 the engine is automatically cut.

    Returns a StarshipState whose fields are *rates of change* (dx/dt).
    """
    commanded = jnp.clip(action[0], 0.0, 1.0)
    gimbal    = jnp.clip(action[1], -params.delta_max, params.delta_max)

    # Engine is on only when commanded throttle >= T_min.
    # Below T_min → engine off (throttle = 0), not floored to T_min.
    # Also cut thrust when out of fuel.
    engine_on = (commanded >= params.T_min) & (state.mprop > 0.0)
    throttle  = jnp.where(engine_on, commanded, 0.0)

    m_total = params.m_dry + state.mprop * params.m_prop_max
    # Moment of inertia: uniform-rod approximation
    I = m_total * params.L ** 2 / 12.0

    thrust = throttle * params.T_max

    # Thrust vector in world frame.
    # theta=0  → nose-up  → thrust points straight up   (sin=0, cos=1)
    # theta=pi/2 → belly-down → thrust points horizontal (sin=1, cos=0)
    F_x = thrust * jnp.sin(state.theta + gimbal)
    F_y = thrust * jnp.cos(state.theta + gimbal)

    ax = F_x / m_total
    ay = F_y / m_total - params.g

    # Torque from gimbaled thrust about CoM (engine at L/2 below CoM)
    torque = thrust * jnp.sin(gimbal) * (params.L / 2.0)
    alpha  = torque / I

    # Propellant mass-flow rate (as fraction of m_prop_max per second)
    d_mprop = -thrust / (params.Isp * params.g * params.m_prop_max)

    return StarshipState(
        x=state.vx,
        y=state.vy,
        vx=ax,
        vy=ay,
        theta=state.omega,
        omega=alpha,
        mprop=d_mprop,
        time=jnp.ones_like(state.time),
    )


def euler_step(
    state: StarshipState,
    action: jax.Array,
    params: StarshipParams,
) -> StarshipState:
    """Advance state one timestep using forward Euler integration."""
    d = derivatives(state, action, params)
    return StarshipState(
        x=state.x + d.x * params.dt,
        y=state.y + d.y * params.dt,
        vx=state.vx + d.vx * params.dt,
        vy=state.vy + d.vy * params.dt,
        theta=state.theta + d.theta * params.dt,
        omega=state.omega + d.omega * params.dt,
        mprop=jnp.clip(state.mprop + d.mprop * params.dt, 0.0, 1.0),
        time=state.time + params.dt,
    )


def hover_throttle(state: StarshipState, params: StarshipParams) -> float:
    """Throttle fraction required for zero net vertical acceleration at theta=0."""
    m_total = params.m_dry + state.mprop * params.m_prop_max
    return m_total * params.g / params.T_max
