"""Research-grade navigation, terrain, benchmark, and RL figures.

All plotting for the navigation stack lives here, on the shared publication theme
(:mod:`segway.viz.style`). Functions take duck-typed results (``NavResult``, ``Terrain``,
benchmark row dicts) so this module stays decoupled from the navigation package and importing
:mod:`segway.viz` never drags in the controllers/solvers.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe; interactive UIs switch backends themselves
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

from .style import CYCLE, PALETTE, SEQ_CMAP, apply_paper_style  # noqa: E402

apply_paper_style()


def _save(fig, path):
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
    return fig


def _draw_world(ax, world, *, occupancy=True):
    """Draw bounds, the inflated occupancy field, and obstacles onto ``ax``."""
    from matplotlib.patches import Circle

    ax.set_xlim(0, world.width)
    ax.set_ylim(0, world.height)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    if occupancy and getattr(world, "occupancy", None) is not None and world.obstacles:
        ax.imshow(world.occupancy, extent=[0, world.width, 0, world.height], origin="lower",
                  cmap="Reds", alpha=0.12, vmin=0, vmax=1, zorder=1,
                  interpolation="nearest")
    for o in world.obstacles:
        ax.add_patch(Circle((o.x, o.y), o.r, color=PALETTE["obstacle"], zorder=3))
        ax.add_patch(Circle((o.x, o.y), o.r + world.robot_radius, fill=False, ls="--",
                            ec=PALETTE["obstacle"], lw=1.0, alpha=0.7, zorder=3))
    ax.set_aspect("equal")   # after imshow, which would otherwise reset the aspect


def _speed_colored_path(ax, x, y, speed, cmap="plasma", lw=3.0, zorder=4):
    """Plot a trajectory as a line coloured by speed; return the mappable for a colorbar."""
    pts = np.array([x, y]).T.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, cmap=cmap, zorder=zorder)
    lc.set_array(np.asarray(speed[:-1]))
    lc.set_linewidth(lw)
    ax.add_collection(lc)
    return lc


def _markers(ax, start, goal):
    ax.plot(*start, "o", color=PALETTE["start"], ms=9, zorder=6, label="start")
    ax.plot(*goal, "*", color=PALETTE["goal"], ms=18, zorder=6, label="goal")


def _cross_track(xy, polyline):
    """Per-point distance from a path to the nearest segment of ``polyline``."""
    poly = np.asarray(polyline, float)
    if len(poly) < 2:
        return np.zeros(len(xy))
    a, b = poly[:-1], poly[1:]
    ab = b - a
    denom = np.sum(ab**2, axis=1)
    denom[denom == 0] = 1e-9
    out = np.empty(len(xy))
    for i, p in enumerate(xy):
        t = np.clip(np.sum((p - a) * ab, axis=1) / denom, 0.0, 1.0)
        proj = a + t[:, None] * ab
        out[i] = np.min(np.hypot(*(p - proj).T))
    return out


# ===== top-down navigation =================================================
def plot_navigation(result, ax=None, *, title=None, show_planned=True, n_arrows=12,
                    occupancy=True):
    """Top-down map: inflated occupancy, obstacles, planned vs driven path, start/goal."""
    if ax is None:
        _fig, ax = plt.subplots(figsize=(7, 7))
    _draw_world(ax, result.world, occupancy=occupancy)

    if show_planned and result.path is not None and len(result.path) > 1:
        ax.plot(result.path[:, 0], result.path[:, 1], "--", color=PALETTE["planned"],
                lw=1.6, alpha=0.9, label=f"planned ({result.planner_name})", zorder=4)

    xy = result.driven_path
    if len(xy) > 1:
        ok = result.success
        ax.plot(xy[:, 0], xy[:, 1], "-", color=PALETTE["driven"] if ok else PALETTE["driven_bad"],
                lw=2.4, label=f"driven ({result.follower_name})", zorder=5)
        _heading_arrows(ax, result.trajectory, n_arrows)

    _markers(ax, result.start, result.goal)
    if title is None:
        status = "reached" if result.success else ("fell" if result.fell else "did not reach")
        title = f"{result.balance_name} · {result.planner_name} · {result.follower_name}  —  {status}"
    ax.set_title(title)
    ax.legend(loc="best")
    return ax


def _heading_arrows(ax, traj, n_arrows):
    if traj is None or len(traj.x) < 2 or n_arrows <= 0:
        return
    idx = np.linspace(0, len(traj.x) - 1, min(n_arrows, len(traj.x))).astype(int)
    scale = 0.04 * max(ax.get_xlim()[1], ax.get_ylim()[1])
    for k in idx:
        ax.arrow(traj.x[k], traj.y[k], scale * np.cos(traj.psi[k]), scale * np.sin(traj.psi[k]),
                 head_width=scale * 0.5, head_length=scale * 0.5, fc="0.2", ec="0.2",
                 alpha=0.6, zorder=5, length_includes_head=True)


# ===== full navigation run analysis ========================================
def plot_nav_analysis(result, path=None, fig=None):
    """Multi-panel analysis of one run: speed-coloured route + tracking/stability time series.

    Pass ``fig`` to draw into an existing figure (e.g. an embedded GUI canvas)."""
    tr = result.trajectory
    if tr is None:
        raise ValueError("result has no trajectory to analyse")
    t = tr.t
    own = fig is None
    fig = fig or plt.figure(figsize=(15, 8.5))
    fig.clear()
    gs = fig.add_gridspec(3, 2, width_ratios=[1.25, 1.0], hspace=0.45, wspace=0.22)

    # left: speed-coloured route over the map
    axm = fig.add_subplot(gs[:, 0])
    _draw_world(axm, result.world)
    if result.path is not None and len(result.path) > 1:
        axm.plot(result.path[:, 0], result.path[:, 1], "--", color=PALETTE["planned"],
                 lw=1.5, alpha=0.85, label="planned", zorder=4)
    lc = _speed_colored_path(axm, tr.x, tr.y, tr.v)
    _markers(axm, result.start, result.goal)
    fig.colorbar(lc, ax=axm, fraction=0.046, pad=0.02, label="speed v [m/s]")
    status = "reached" if result.success else ("fell" if result.fell else "did not reach")
    axm.set_title(f"Route — {result.balance_name}·{result.planner_name}·{result.follower_name} ({status})")
    axm.legend(loc="best")

    vdes = tr.commands[:, 0] if tr.commands.ndim == 2 else np.zeros_like(t)
    wdes = tr.commands[:, 1] if tr.commands.ndim == 2 else np.zeros_like(t)

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.plot(t, vdes, "--", color=PALETTE["muted"], label=r"$v_{des}$")
    ax1.plot(t, tr.v, color=PALETTE["driven"], label=r"$v$")
    ax1.set_ylabel(r"speed [m/s]")
    ax1.set_title("Forward-speed tracking")
    ax1.legend(ncol=2)

    ax2 = fig.add_subplot(gs[1, 1], sharex=ax1)
    ax2.plot(t, np.degrees(tr.theta), color=PALETTE["accent"])
    ax2.axhline(0, color="k", lw=0.7, ls="--")
    ax2.set_ylabel(r"pitch $\theta$ [deg]")
    ax2.set_title("Balance (body pitch)")

    ax3 = fig.add_subplot(gs[2, 1], sharex=ax1)
    ax3.plot(t, wdes, "--", color=PALETTE["muted"], label=r"$\dot{\psi}_{des}$")
    ax3.plot(t, tr.states[:, 5], color=CYCLE[1], label=r"$\dot{\psi}$")
    ax3.plot(t, tr.torques[:, 0], color=CYCLE[3], lw=1.2, alpha=0.8, label=r"$\tau_L$")
    ax3.plot(t, tr.torques[:, 1], color=CYCLE[4], lw=1.2, alpha=0.8, label=r"$\tau_R$")
    ax3.set_ylabel("yaw-rate / torque")
    ax3.set_xlabel("time [s]")
    ax3.set_title("Yaw-rate tracking & wheel torques")
    ax3.legend(ncol=2, fontsize=8)

    fig.suptitle("Navigation run analysis", y=0.98)
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight")
        if own:
            plt.close(fig)
    return fig


# ===== terrain =============================================================
def plot_terrain(terrain, ax=None, path=None):
    """Filled-contour height map with a colorbar."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.5, 6.5))
    else:
        fig = ax.figure
    xs = np.linspace(0, terrain.width, terrain.ncol)
    ys = np.linspace(0, terrain.height, terrain.nrow)
    cf = ax.contourf(xs, ys, terrain.heights, levels=24, cmap=SEQ_CMAP)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"Terrain '{terrain.name}' — amplitude {terrain.amplitude:.2f} m, "
                 f"max slope {np.degrees(terrain.max_slope()):.0f}°")
    fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.02, label="height [m]")
    return _save(fig, path) if ax.figure is fig and path else ax


def plot_terrain_run(result, terrain, path=None):
    """Route over the terrain height map + elevation profile + pitch-vs-slope along the path."""
    tr = result.trajectory
    fig = plt.figure(figsize=(15, 6.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.3, 1.0], hspace=0.5, wspace=0.2)

    axm = fig.add_subplot(gs[:, 0])
    xs = np.linspace(0, terrain.width, terrain.ncol)
    ys = np.linspace(0, terrain.height, terrain.nrow)
    cf = axm.contourf(xs, ys, terrain.heights, levels=24, cmap=SEQ_CMAP, zorder=1)
    _draw_world(axm, result.world, occupancy=False)
    axm.plot(tr.x, tr.y, color="white", lw=2.6, zorder=5,
             label=f"driven ({result.follower_name})")
    _markers(axm, result.start, result.goal)
    fig.colorbar(cf, ax=axm, fraction=0.046, pad=0.02, label="ground height [m]")
    axm.set_title(f"Route over '{terrain.name}' terrain")
    axm.legend(loc="best")

    xy = result.driven_path
    seg = np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1]))
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    elev = np.array([terrain.height_at(x, y) for x, y in xy])
    slope = np.degrees([terrain.slope_at(x, y) for x, y in xy])

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.plot(arc, elev, color=CYCLE[5])
    ax1.fill_between(arc, elev, elev.min() - 0.02, color=CYCLE[5], alpha=0.2)
    ax1.set_ylabel("ground height [m]")
    ax1.set_title("Elevation profile along the path")

    ax2 = fig.add_subplot(gs[1, 1], sharex=ax1)
    ax2.plot(arc, np.degrees(tr.theta), color=PALETTE["accent"], label=r"body pitch $\theta$")
    ax2.plot(arc, slope, color=PALETTE["muted"], ls="--", label="ground slope")
    ax2.set_xlabel("distance along path [m]")
    ax2.set_ylabel("angle [deg]")
    ax2.set_title("Body pitch vs ground slope")
    ax2.legend(ncol=2)

    fig.suptitle("Terrain traversal analysis", y=0.99)
    return _save(fig, path)


# ===== benchmark ============================================================
def plot_benchmark_summary(rows, path=None):
    """Success/time matrices for each navigation sweep (method × map)."""
    sweeps = ["planner", "follower", "balance"]
    titles = {"planner": "Planner (lqr · pure_pursuit)",
              "follower": "Follower (lqr · a_star)",
              "balance": "Balance controller (a_star · pure_pursuit)"}
    present = [s for s in sweeps if any(r["sweep"] == s for r in rows)]
    fig, axes = plt.subplots(1, len(present), figsize=(6.2 * len(present), 5.4), squeeze=False)
    for ax, sweep in zip(axes[0], present, strict=True):
        srows = [r for r in rows if r["sweep"] == sweep]
        methods = sorted({r[sweep] for r in srows})
        maps = sorted({r["map"] for r in srows})
        M = np.full((len(methods), len(maps)), np.nan)
        for r in srows:
            i, j = methods.index(r[sweep]), maps.index(r["map"])
            M[i, j] = 1.0 if r["success"] else 0.0
        ax.imshow(M, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(maps)), maps, rotation=30, ha="right")
        ax.set_yticks(range(len(methods)), methods)
        for r in srows:
            i, j = methods.index(r[sweep]), maps.index(r["map"])
            txt = f"{r['time_s']:.1f}s" if r["success"] else "✗"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8, color="#222")
        ax.set_title(titles.get(sweep, sweep))
    fig.suptitle("Navigation benchmark — reached (green) vs failed (red), time-to-goal annotated", y=1.02)
    fig.tight_layout()
    return _save(fig, path)


# ===== learned vs classical ================================================
def _read_learning_curve(learning_csv):
    """Extract (timesteps, mean episode reward) from an SB3 progress.csv, if present."""
    if not learning_csv or not Path(learning_csv).exists():
        return [], []
    import csv as _csv
    steps, rew = [], []
    with open(learning_csv, encoding="utf-8") as f:
        reader = _csv.DictReader(f)
        ycol = None
        for row in reader:
            if ycol is None:
                ycol = next((c for c in ("rollout/ep_rew_mean", "rollout/ep_reward_mean")
                             if c in row), None)
            if ycol and row.get("time/total_timesteps") and row.get(ycol):
                steps.append(float(row["time/total_timesteps"]))
                rew.append(float(row[ycol]))
    return steps, rew


def plot_rl_analysis(rl_rows, learning_csv=None, path=None):
    """Learned-vs-classical: arrival-distance distribution, success/time, and a learning curve."""
    methods = ["classical", "rl"]
    colors = {"classical": CYCLE[0], "rl": CYCLE[1]}
    steps, rew = _read_learning_curve(learning_csv)
    ncols = 3 if steps else 2     # only show the curve panel when reward data is available
    fig, axes = plt.subplots(1, ncols, figsize=(5.4 * ncols, 4.6))

    ax0 = axes[0]
    for m in methods:
        d = [r["final_dist_m"] for r in rl_rows if r["method"] == m]
        if d:
            ax0.hist(d, bins=10, alpha=0.6, color=colors[m], label=m, edgecolor="white")
    ax0.set_xlabel("final distance to goal [m]")
    ax0.set_ylabel("count")
    ax0.set_title("Arrival-distance distribution")
    ax0.legend()

    ax1 = axes[1]
    rates, times = [], []
    for m in methods:
        mr = [r for r in rl_rows if r["method"] == m]
        rates.append(100.0 * sum(r["success"] for r in mr) / max(len(mr), 1))
        times.append(np.mean([r["time_s"] for r in mr]) if mr else 0.0)
    xpos = np.arange(len(methods))
    ax1.bar(xpos, rates, color=[colors[m] for m in methods], width=0.6)
    ax1.set_xticks(xpos, methods)
    ax1.set_ylabel("success rate [%]")
    ax1.set_ylim(0, 105)
    ax1.set_title("Goal-reach success")
    for x, r, tm in zip(xpos, rates, times, strict=True):
        ax1.text(x, r + 1.5, f"{r:.0f}%\n{tm:.1f}s", ha="center", va="bottom", fontsize=9)

    if steps:
        ax2 = axes[2]
        ax2.plot(np.array(steps) / 1e3, rew, color=PALETTE["accent"])
        ax2.set_xlabel("timesteps [×10³]")
        ax2.set_ylabel("mean episode reward")
        ax2.set_title("PPO learning curve")

    fig.suptitle("Learned navigation vs classical stack", y=1.02)
    fig.tight_layout()
    return _save(fig, path)
