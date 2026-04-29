"""Headless top-down rendering of an IADS sim state to a numpy RGB array.

Used by scripts/ppo_impl.py to record periodic eval videos for TensorBoard.
Output frame is shape (H, W, 3) uint8.
"""

from __future__ import annotations

from typing import Tuple

import matplotlib

matplotlib.use("Agg")  # headless, no display required

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from iads.sim_engine import EntityType, InterceptorState, MissileState


# Color scheme.
_REAL_COLOR = "#1f77ff"   # blue
_DECOY_COLOR = "#9467bd"  # purple
_SAM_COLOR = "#d62728"    # red
_RADAR_COLOR = "#ff7f0e"  # orange
_TARGET_COLOR = "#2ca02c" # green
_INTC_COLOR = "#8b0000"   # dark red

_TRAIL_LEN = 30  # last N positions in the agent trail


def render_frame(
    sim,
    *,
    figsize: Tuple[float, float] = (5.0, 7.5),
    dpi: int = 80,
    title: str = "",
) -> np.ndarray:
    """Render the current sim state as an (H, W, 3) uint8 RGB array."""
    W = sim.config["area_width"]
    H = sim.config["area_height"]

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.set_facecolor("#0e1117")
    ax.tick_params(colors="#888888", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444444")

    # Threats: SAM engagement rings + radar detection rings (faint, behind agents).
    for sam in sim.sams:
        if not sam.alive:
            continue
        active = sam.interceptors_remaining > 0
        ring = Circle(
            (sam.pos[0], sam.pos[1]),
            radius=sam.engagement_range,
            fill=True,
            facecolor=_SAM_COLOR,
            alpha=0.07 if active else 0.02,
            edgecolor=_SAM_COLOR,
            linewidth=0.6,
            linestyle="-" if active else ":",
        )
        ax.add_patch(ring)
        ax.plot(
            sam.pos[0],
            sam.pos[1],
            marker="X",
            color=_SAM_COLOR,
            markersize=7 if active else 5,
            markeredgecolor="white",
            markeredgewidth=0.5,
        )

    for radar in sim.radars:
        if not radar.alive:
            continue
        ring = Circle(
            (radar.pos[0], radar.pos[1]),
            radius=radar.detection_range,
            fill=False,
            edgecolor=_RADAR_COLOR,
            alpha=0.25,
            linewidth=0.6,
            linestyle="--",
        )
        ax.add_patch(ring)
        ax.plot(
            radar.pos[0],
            radar.pos[1],
            marker="^",
            color=_RADAR_COLOR,
            markersize=7,
            markeredgecolor="white",
            markeredgewidth=0.5,
        )

    # Targets.
    for tgt in sim.targets:
        ax.plot(
            tgt.pos[0],
            tgt.pos[1],
            marker="*",
            color=_TARGET_COLOR if tgt.alive else "#444444",
            markersize=15 if tgt.alive else 10,
            markeredgecolor="white",
            markeredgewidth=0.6,
        )

    # In-flight interceptors.
    for ic in sim.interceptors:
        if ic.state != InterceptorState.IN_FLIGHT:
            continue
        ax.plot(
            ic.pos[0],
            ic.pos[1],
            marker="o",
            color=_INTC_COLOR,
            markersize=4,
            markeredgecolor="white",
            markeredgewidth=0.4,
        )

    # Agent trails + current positions.
    for i, m in enumerate(sim.missiles):
        is_decoy = m.missile_type == EntityType.DECOY_MISSILE
        color = _DECOY_COLOR if is_decoy else _REAL_COLOR

        trail = m.trail[-_TRAIL_LEN:] if m.trail else []
        if len(trail) > 1:
            xs = [p[0] for p in trail]
            ys = [p[1] for p in trail]
            ax.plot(xs, ys, color=color, alpha=0.55, linewidth=1.2)

        alive = m.state == MissileState.IN_FLIGHT
        if alive:
            face = color
            edge = "white"
        elif m.state == MissileState.HIT_TARGET:
            face = _TARGET_COLOR
            edge = color
        else:  # intercepted, missed
            face = "#222222"
            edge = color
        marker = "s" if is_decoy else "o"
        ax.plot(
            m.pos[0],
            m.pos[1],
            marker=marker,
            markerfacecolor=face,
            markeredgecolor=edge,
            markersize=8,
            markeredgewidth=0.7,
        )
        ax.text(
            m.pos[0] + 6,
            m.pos[1] + 6,
            f"{i}{'d' if is_decoy else 'r'}",
            color=color,
            fontsize=6,
            fontweight="bold",
        )

    if title:
        ax.set_title(title, color="#dddddd", fontsize=8, loc="left")

    fig.patch.set_facecolor("#0e1117")
    fig.tight_layout(pad=0.4)

    # Pull RGB pixels from the canvas.
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img = buf.reshape(h, w, 4)[..., :3].copy()
    plt.close(fig)
    return img
