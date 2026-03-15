"""Trajectory visualization for the Starship landing environment."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib
import matplotlib.animation as animation
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from starjaxrl.env.starship_env import EnvParams
from starjaxrl.physics import StarshipState

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_VEHICLE_WIDTH  = 8.0    # m — visual width of vehicle body
_PLUME_SCALE    = 40.0   # m — max plume length at full throttle
_BG_COLOR       = "#0b0c1a"
_GROUND_COLOR   = "#2a2a3a"
_VEHICLE_COLOR  = "#c8ccd4"
_NOSE_COLOR     = "#e8eaed"
_PLUME_COLOR    = "#ff8800"
_TRAIL_COLOR    = "#4499ff"
_PAD_COLOR      = "#f5c518"
_TEXT_COLOR     = "#e0e0e0"


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _vehicle_corners(
    cx: float, cy: float, theta: float, L: float, W: float
) -> np.ndarray:
    """Return (4, 2) array of rectangle corners in world frame.

    theta=0 → nose-up vertical; theta=pi/2 → nose pointing right.
    """
    along = np.array([np.sin(theta), np.cos(theta)])   # tail → nose
    perp  = np.array([np.cos(theta), -np.sin(theta)])  # lateral
    c     = np.array([cx, cy])
    return np.array([
        c + (L / 2) * along + (W / 2) * perp,
        c + (L / 2) * along - (W / 2) * perp,
        c - (L / 2) * along - (W / 2) * perp,
        c - (L / 2) * along + (W / 2) * perp,
    ])


def _engine_pos(cx: float, cy: float, theta: float, L: float) -> np.ndarray:
    """World-frame position of the engine nozzle (vehicle tail)."""
    along = np.array([np.sin(theta), np.cos(theta)])
    return np.array([cx, cy]) - (L / 2) * along


# ---------------------------------------------------------------------------
# Single-frame renderer
# ---------------------------------------------------------------------------

def render_frame(
    ax:         plt.Axes,
    state:      StarshipState,
    action:     np.ndarray | None = None,
    env_params: EnvParams | None = None,
    trail_x:    Sequence[float] | None = None,
    trail_y:    Sequence[float] | None = None,
    vehicle_L:  float = 50.0,
) -> None:
    """Draw one frame onto *ax*.

    Args:
        ax:         Matplotlib axes (already sized and limited by caller).
        state:      Current StarshipState.
        action:     [throttle, gimbal] array, or None for no thrust visual.
        env_params: Used for catch-arm height and success zone width.
        trail_x:    X history (oldest first) for the trajectory trail.
        trail_y:    Y history (oldest first).
        vehicle_L:  Vehicle length (m); must match physics params.
    """
    cx    = float(state.x)
    cy    = float(state.y)
    theta = float(state.theta)
    mprop = float(state.mprop)
    t     = float(state.time)
    speed = float((state.vx ** 2 + state.vy ** 2) ** 0.5)

    # ---- Ground ----
    xlim = ax.get_xlim()
    ax.add_patch(mpatches.Rectangle(
        (xlim[0], -200), xlim[1] - xlim[0], 200,
        color=_GROUND_COLOR, zorder=0,
    ))

    # ---- Catch arms ----
    if env_params is not None:
        y_catch = float(env_params.y_catch)
        arm_hw  = float(env_params.success_x_tol) * 4
        for sign in (-1, 1):
            ax.add_patch(mpatches.Rectangle(
                (sign * arm_hw - arm_hw * 0.15, y_catch - 2),
                arm_hw * 0.15, 6,
                color=_PAD_COLOR, zorder=1, linewidth=0,
            ))
        ax.plot([-arm_hw, -arm_hw * 0.15], [y_catch + 1, y_catch + 1],
                color=_PAD_COLOR, lw=3, zorder=1)
        ax.plot([arm_hw * 0.15, arm_hw], [y_catch + 1, y_catch + 1],
                color=_PAD_COLOR, lw=3, zorder=1)

    # ---- Trajectory trail (fading blue line) ----
    if trail_x and len(trail_x) > 1:
        n = len(trail_x)
        for i in range(1, n):
            alpha = 0.1 + 0.9 * (i / n)
            ax.plot(trail_x[i - 1:i + 1], trail_y[i - 1:i + 1],
                    color=_TRAIL_COLOR, alpha=alpha, lw=1, zorder=2)

    # ---- Thrust plume ----
    throttle = float(action[0]) if action is not None else 0.0
    gimbal   = float(action[1]) if action is not None else 0.0
    if env_params is not None:
        firing = throttle >= float(env_params.T_min)
    else:
        firing = throttle > 0.3

    engine = _engine_pos(cx, cy, theta, vehicle_L)
    if firing:
        plume_len = throttle * _PLUME_SCALE
        # Plume direction is OPPOSITE to thrust (exhaust shoots down/back)
        dx = -np.sin(theta + gimbal) * plume_len
        dy = -np.cos(theta + gimbal) * plume_len
        ax.annotate(
            "", xy=(engine[0] + dx, engine[1] + dy), xytext=(engine[0], engine[1]),
            arrowprops=dict(arrowstyle="-|>", color=_PLUME_COLOR,
                            lw=2.5, mutation_scale=12),
            zorder=3,
        )
        # Inner bright core
        core_len = plume_len * 0.5
        ax.annotate(
            "", xy=(engine[0] - np.sin(theta + gimbal) * core_len,
                    engine[1] - np.cos(theta + gimbal) * core_len),
            xytext=(engine[0], engine[1]),
            arrowprops=dict(arrowstyle="-|>", color="#ffdd55",
                            lw=1.5, mutation_scale=8),
            zorder=4,
        )

    # ---- Vehicle body ----
    corners = _vehicle_corners(cx, cy, theta, vehicle_L, _VEHICLE_WIDTH)
    body    = mpatches.Polygon(corners, closed=True, color=_VEHICLE_COLOR, zorder=5)
    ax.add_patch(body)

    # Nose cap (darker tip — top 15 % of vehicle)
    nose_tip    = np.array([cx, cy]) + (vehicle_L / 2) * np.array([np.sin(theta), np.cos(theta)])
    nose_base_l = corners[0]
    nose_base_r = corners[1]
    nose_patch  = mpatches.Polygon(
        [nose_tip, nose_base_l, nose_base_r], closed=True,
        color=_NOSE_COLOR, zorder=6,
    )
    ax.add_patch(nose_patch)

    # ---- HUD text ----
    hud_lines = [
        f"t = {t:5.1f} s",
        f"alt  {cy:7.1f} m",
        f"spd  {speed:6.1f} m/s",
        f"vy   {float(state.vy):+7.1f} m/s",
        f"θ    {np.degrees(theta):+6.1f}°",
        f"fuel {mprop * 100:5.1f} %",
    ]
    hud_text = "\n".join(hud_lines)
    ax.text(
        0.02, 0.97, hud_text,
        transform=ax.transAxes,
        va="top", ha="left",
        fontsize=8, color=_TEXT_COLOR,
        fontfamily="monospace",
        bbox=dict(facecolor="black", alpha=0.4, boxstyle="round,pad=0.3"),
        zorder=10,
    )


# ---------------------------------------------------------------------------
# Full trajectory animation
# ---------------------------------------------------------------------------

def render_trajectory(
    states:     list[StarshipState],
    actions:    list[np.ndarray],
    env_params: EnvParams | None = None,
    figsize:    tuple[int, int] = (8, 14),
    fps:        int = 30,
    vehicle_L:  float = 50.0,
) -> tuple[plt.Figure, animation.FuncAnimation]:
    """Build a FuncAnimation for a full episode trajectory.

    Args:
        states:     List of StarshipState (length T+1, including initial).
        actions:    List of actions (length T, one per step).
        env_params: Environment parameters for catch-arm rendering.
        figsize:    Figure size in inches.
        fps:        Target frames per second.
        vehicle_L:  Vehicle length in metres.

    Returns:
        (fig, anim) — the matplotlib Figure and FuncAnimation.
    """
    all_x = [float(s.x) for s in states]
    all_y = [float(s.y) for s in states]

    # Fixed view limits computed from trajectory
    x_span   = max(400.0, max(all_x) - min(all_x) + 200.0)
    x_center = np.mean(all_x)
    x_lim    = (x_center - x_span / 2, x_center + x_span / 2)
    y_lim    = (-80.0, max(all_y) * 1.06)

    fig, ax = plt.subplots(figsize=figsize, facecolor=_BG_COLOR)
    ax.set_facecolor(_BG_COLOR)

    def update(frame: int) -> None:
        ax.clear()
        ax.set_facecolor(_BG_COLOR)
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_xlabel("x  (m)", color=_TEXT_COLOR, fontsize=9)
        ax.set_ylabel("altitude  (m)", color=_TEXT_COLOR, fontsize=9)
        ax.tick_params(colors=_TEXT_COLOR)
        for spine in ax.spines.values():
            spine.set_color("#333355")

        action = np.array(actions[frame]) if frame < len(actions) else None
        render_frame(
            ax, states[frame], action, env_params,
            trail_x=all_x[: frame + 1],
            trail_y=all_y[: frame + 1],
            vehicle_L=vehicle_L,
        )

    anim = animation.FuncAnimation(
        fig, update,
        frames=len(states),
        interval=int(1000 / fps),
        blit=False,
    )
    return fig, anim


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def save_animation(
    anim:   animation.FuncAnimation,
    path:   str | Path,
    fps:    int = 30,
) -> None:
    """Save animation to disk as gif or mp4 based on file extension.

    Args:
        anim: FuncAnimation to save.
        path: Output path; extension determines format (.gif or .mp4).
        fps:  Frames per second.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix == ".mp4":
        writer = animation.FFMpegWriter(fps=fps, bitrate=1800)
        anim.save(str(path), writer=writer)
    else:
        writer = animation.PillowWriter(fps=fps)
        anim.save(str(path), writer=writer)

    print(f"Animation saved → {path}")
