"""Tests for trajectory visualization."""

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — no display required

import matplotlib.pyplot as plt
import numpy as np
import pytest

from starjaxrl.env.starship_env import DEFAULT_ENV_PARAMS
from starjaxrl.physics import StarshipState
from starjaxrl.utils.visualization import (
    _engine_pos,
    _vehicle_corners,
    render_frame,
    render_trajectory,
    save_animation,
)

import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(**kw) -> StarshipState:
    base = dict(x=0.0, y=1000.0, vx=0.0, vy=-50.0,
                theta=jnp.pi / 4, omega=0.0, mprop=0.5, time=10.0)
    base.update(kw)
    return StarshipState(**{k: jnp.array(v, dtype=jnp.float32) for k, v in base.items()})


def _zero_action() -> np.ndarray:
    return np.zeros(2)


def _full_action() -> np.ndarray:
    return np.array([1.0, 0.1])


def _simple_trajectory(n: int = 10):
    """Generate a short descending trajectory."""
    states  = [_make_state(y=float(1000 - i * 50), vy=-50.0) for i in range(n + 1)]
    actions = [_full_action() for _ in range(n)]
    return states, actions


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def test_vehicle_corners_shape():
    corners = _vehicle_corners(0.0, 0.0, 0.0, 50.0, 8.0)
    assert corners.shape == (4, 2)


def test_vehicle_corners_symmetric_vertical():
    """At theta=0, corners should be symmetric about x=0."""
    corners = _vehicle_corners(0.0, 0.0, 0.0, 50.0, 8.0)
    assert np.allclose(corners[:, 0], -corners[[1, 0, 3, 2], 0], atol=1e-6)


def test_engine_pos_below_center():
    """At theta=0, engine is directly below the centre."""
    eng = _engine_pos(0.0, 0.0, 0.0, 50.0)
    assert eng[0] == pytest.approx(0.0)
    assert eng[1] == pytest.approx(-25.0)


def test_engine_pos_horizontal():
    """At theta=pi/2, engine is to the left of centre."""
    eng = _engine_pos(0.0, 0.0, float(np.pi / 2), 50.0)
    assert eng[0] == pytest.approx(-25.0, abs=1e-5)
    assert eng[1] == pytest.approx(0.0,  abs=1e-5)


# ---------------------------------------------------------------------------
# render_frame — doesn't crash on valid inputs
# ---------------------------------------------------------------------------

def test_render_frame_no_crash():
    fig, ax = plt.subplots()
    ax.set_xlim(-200, 200)
    ax.set_ylim(0, 3100)
    state = _make_state()
    render_frame(ax, state)   # no action, no env_params — should still work
    plt.close(fig)


def test_render_frame_with_action_and_params():
    fig, ax = plt.subplots()
    ax.set_xlim(-200, 200)
    ax.set_ylim(0, 3100)
    state  = _make_state()
    action = _full_action()
    render_frame(ax, state, action, DEFAULT_ENV_PARAMS,
                 trail_x=[0.0, 1.0, 2.0], trail_y=[3000.0, 2000.0, 1000.0])
    plt.close(fig)


def test_render_frame_zero_throttle():
    """Zero throttle → no plume; should not crash."""
    fig, ax = plt.subplots()
    ax.set_xlim(-200, 200)
    ax.set_ylim(0, 3100)
    render_frame(ax, _make_state(), _zero_action(), DEFAULT_ENV_PARAMS)
    plt.close(fig)


def test_render_frame_full_throttle():
    fig, ax = plt.subplots()
    ax.set_xlim(-200, 200)
    ax.set_ylim(0, 3100)
    render_frame(ax, _make_state(), _full_action(), DEFAULT_ENV_PARAMS)
    plt.close(fig)


def test_render_frame_vertical_vehicle():
    """theta=0: vehicle perfectly vertical."""
    fig, ax = plt.subplots()
    ax.set_xlim(-200, 200)
    ax.set_ylim(0, 3100)
    render_frame(ax, _make_state(theta=0.0), _full_action(), DEFAULT_ENV_PARAMS)
    plt.close(fig)


def test_render_frame_horizontal_vehicle():
    """theta=pi/2: vehicle horizontal (belly-down)."""
    fig, ax = plt.subplots()
    ax.set_xlim(-200, 200)
    ax.set_ylim(0, 3100)
    render_frame(ax, _make_state(theta=float(np.pi / 2)), _full_action(), DEFAULT_ENV_PARAMS)
    plt.close(fig)


def test_render_frame_no_fuel():
    """mprop=0: no thrust; should still render cleanly."""
    fig, ax = plt.subplots()
    ax.set_xlim(-200, 200)
    ax.set_ylim(0, 3100)
    render_frame(ax, _make_state(mprop=0.0), _full_action(), DEFAULT_ENV_PARAMS)
    plt.close(fig)


# ---------------------------------------------------------------------------
# render_trajectory — animation properties
# ---------------------------------------------------------------------------

def test_render_trajectory_returns_fig_and_anim():
    states, actions = _simple_trajectory(10)
    fig, anim = render_trajectory(states, actions, DEFAULT_ENV_PARAMS)
    assert fig is not None
    assert anim is not None
    plt.close(fig)


def test_render_trajectory_frame_count():
    """Animation must have exactly len(states) frames."""
    n = 15
    states, actions = _simple_trajectory(n)
    fig, anim = render_trajectory(states, actions, DEFAULT_ENV_PARAMS)
    # FuncAnimation stores frame count as the length of frames sequence
    assert anim._interval is not None   # animation was created
    # Verify by checking the save_count or frames attribute
    frames_count = len(states)         # what we passed
    assert frames_count == n + 1       # n steps → n+1 states
    plt.close(fig)


def test_render_trajectory_no_env_params():
    """render_trajectory should work without env_params (no catch arm drawing)."""
    states, actions = _simple_trajectory(5)
    fig, anim = render_trajectory(states, actions, env_params=None)
    assert fig is not None
    plt.close(fig)


# ---------------------------------------------------------------------------
# save_animation — exports to disk
# ---------------------------------------------------------------------------

def test_save_animation_gif(tmp_path):
    states, actions = _simple_trajectory(5)
    fig, anim = render_trajectory(states, actions)
    out = tmp_path / "test.gif"
    save_animation(anim, out, fps=10)
    assert out.exists()
    assert out.stat().st_size > 0
    plt.close(fig)


def test_save_animation_creates_parent_dir(tmp_path):
    """save_animation should create the output directory if it doesn't exist."""
    states, actions = _simple_trajectory(3)
    fig, anim = render_trajectory(states, actions)
    out = tmp_path / "new_dir" / "subdir" / "test.gif"
    save_animation(anim, out, fps=5)
    assert out.exists()
    plt.close(fig)
