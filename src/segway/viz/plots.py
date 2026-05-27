"""Matplotlib plotting helpers (headless-safe)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe; UIs that need interactivity switch backends themselves
import matplotlib.pyplot as plt  # noqa: E402

from ..sim.runner import Trajectory  # noqa: E402

# Use mathtext for Greek/symbols so rendering is font-independent.
_ANGLE = r"Tilt $\theta$ (deg)"
_TORQUE = r"Torque $\tau$ (N$\cdot$m)"


def plot_trajectory(traj: Trajectory, path: str | Path | None = None, title: str | None = None):
    """Three-panel plot (tilt, base position, control torque) for one run."""
    import numpy as np

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    t = traj.t
    axes[0].plot(t, np.degrees(traj.theta), color="#e63946", lw=2, label="tilt")
    axes[0].axhline(0, color="k", ls="--", lw=0.8)
    axes[0].set_ylabel(_ANGLE)
    axes[1].plot(t, traj.x, color="#1d75d8", lw=2, label="position")
    axes[1].set_ylabel("Base x (m)")
    axes[2].plot(t, traj.controls, color="#2a9d8f", lw=2, label="torque")
    axes[2].set_ylabel(_TORQUE)
    axes[2].set_xlabel("Time (s)")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle(title or f"{traj.controller_name.upper()} - {traj.scenario.name}", fontweight="bold")
    fig.tight_layout()
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_roa(result, path: str | Path | None = None, title: str | None = None):
    """Filled contour of a Region-of-Attraction grid (green = recovered, red = fell)."""
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.contourf(
        result.thetadot_vals, result.theta_vals, result.grid.astype(float),
        levels=[-0.5, 0.5, 1.5], colors=["#e63946", "#2a9d8f"], alpha=0.85,
    )
    ax.set_xlabel(r"Initial $\dot{\theta}$ (rad/s)")
    ax.set_ylabel(r"Initial $\theta$ (rad)")
    ax.set_title(title or f"Region of attraction — {result.controller_name.upper()} "
                 f"(area {result.area_fraction:.0%})", fontweight="bold")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_comparison(trajs: dict[str, Trajectory], path: str | Path | None = None,
                    title: str = "Controller comparison"):
    """Overlay tilt and torque for several controllers run on the same scenario."""
    import numpy as np

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for name, tr in trajs.items():
        axes[0].plot(tr.t, np.degrees(tr.theta), lw=2, label=name)
        axes[1].plot(tr.t, tr.controls, lw=1.6, label=name)
    axes[0].axhline(0, color="k", ls="--", lw=0.8)
    axes[0].set_ylabel(_ANGLE)
    axes[1].set_ylabel(_TORQUE)
    axes[1].set_xlabel("Time (s)")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(ncol=3, fontsize=9)
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)
    return fig
