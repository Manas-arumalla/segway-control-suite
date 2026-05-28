"""Path followers (local planners) + a name-based registry.

A follower turns a global path + the robot's current pose into a motion command
``(v_des, yaw_rate_des, done)`` for the TWIP inner loop. They share one interface so the
navigator and benchmark can mix and match. Implemented: Pure Pursuit, Stanley, DWA (reactive,
obstacle-aware), a sampling MPC path-tracker, and a Vector/Potential field (reactive).

Each follower densifies the (possibly sparse) waypoint path and tracks a *monotonically
advancing* look-ahead target, so it never aims at a waypoint it has already passed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


def _wrap(a: float) -> float:
    return float(np.arctan2(np.sin(a), np.cos(a)))


def _densify(path: np.ndarray, spacing: float = 0.25) -> np.ndarray:
    path = np.asarray(path, dtype=float)
    if len(path) < 2:
        return path
    out = [path[0]]
    for a, b in zip(path[:-1], path[1:], strict=False):
        d = float(np.hypot(*(b - a)))
        n = max(1, int(d / spacing))
        for k in range(1, n + 1):
            out.append(a + (b - a) * (k / n))
    return np.array(out, dtype=float)


def _clearance(world, x: float, y: float) -> float:
    if world is None or not world.obstacles:
        return 10.0
    return min(np.hypot(x - o.x, y - o.y) - o.r - world.robot_radius for o in world.obstacles)


def _rollout(pose, v, w, dt, steps):
    x, y, psi = pose
    pts = []
    for _ in range(steps):
        x += v * np.cos(psi) * dt
        y += v * np.sin(psi) * dt
        psi += w * dt
        pts.append((x, y))
    return pts, (x, y, psi)


class Follower(ABC):
    """Common interface for path followers (with shared look-ahead bookkeeping)."""

    name: str = "base"

    def __init__(self, v_max=0.8, goal_tol=0.25, slow_radius=1.0, yaw_max=3.0, lookahead=0.8):
        self.v_max, self.goal_tol, self.slow_radius = v_max, goal_tol, slow_radius
        self.yaw_max, self.lookahead = yaw_max, lookahead
        self._dense: np.ndarray | None = None
        self._progress = 0

    def reset(self) -> None:
        self._dense = None
        self._progress = 0

    def _prepare(self, path) -> None:
        if self._dense is None:
            self._dense = _densify(np.asarray(path, dtype=float))
            self._progress = 0

    def _advance_target(self, p, ld: float) -> np.ndarray:
        """Monotonically advance along the dense path and return a look-ahead target."""
        dense = self._dense
        seg = dense[self._progress:]
        self._progress += int(np.argmin(np.hypot(seg[:, 0] - p[0], seg[:, 1] - p[1])))
        for k in range(self._progress, len(dense)):
            if np.hypot(dense[k, 0] - p[0], dense[k, 1] - p[1]) >= ld:
                return dense[k]
        return dense[-1]

    def _goal_distance(self, p) -> float:
        return float(np.hypot(*(self._dense[-1] - p)))

    def _reached(self, p) -> bool:
        """Done if within tolerance of the goal, or past it along the final path segment."""
        dense = self._dense
        if self._goal_distance(p) < self.goal_tol:
            return True
        if self._progress >= len(dense) - 1 and len(dense) >= 2:
            final_dir = dense[-1] - dense[-2]
            if np.dot(np.asarray(p) - dense[-1], final_dir) > -1e-9:
                return True   # the goal is at or behind the robot
        return False

    @abstractmethod
    def command(self, pose, path, world=None) -> tuple[float, float, bool]:
        """Return ``(v_des, yaw_rate_des, done)`` for ``pose=(x,y,psi)`` following ``path``."""
        raise NotImplementedError


class PurePursuit(Follower):
    name = "pure_pursuit"

    def command(self, pose, path, world=None):
        self._prepare(path)
        x, y, psi = pose
        p = np.array([x, y])
        tgt = self._advance_target(p, self.lookahead)
        if self._reached(p):
            return 0.0, 0.0, True
        dg = self._goal_distance(p)
        alpha = _wrap(np.arctan2(tgt[1] - y, tgt[0] - x) - psi)
        v = self.v_max * min(1.0, dg / self.slow_radius) * max(0.15, np.cos(alpha))
        yaw = 2.0 * max(v, 0.25) * np.sin(alpha) / self.lookahead
        return v, float(np.clip(yaw, -self.yaw_max, self.yaw_max)), False


class Stanley(Follower):
    name = "stanley"

    def __init__(self, k_e=1.2, k_h=1.5, **kw):
        super().__init__(**kw)
        self.k_e, self.k_h = k_e, k_h

    def command(self, pose, path, world=None):
        self._prepare(path)
        x, y, psi = pose
        p = np.array([x, y])
        dense = self._dense
        self._advance_target(p, 0.0)            # update progress to the nearest dense index
        if self._reached(p):
            return 0.0, 0.0, True
        dg = self._goal_distance(p)
        i0 = self._progress
        i1 = min(i0 + 1, len(dense) - 1)
        psi_p = float(np.arctan2(dense[i1, 1] - dense[max(i1 - 1, 0), 1],
                                 dense[i1, 0] - dense[max(i1 - 1, 0), 0]))
        e = (x - dense[i0, 0]) * -np.sin(psi_p) + (y - dense[i0, 1]) * np.cos(psi_p)
        v = self.v_max * min(1.0, dg / self.slow_radius)
        yaw = self.k_h * _wrap(psi_p - psi) + np.arctan2(-self.k_e * e, v + 0.3)
        return v, float(np.clip(yaw, -self.yaw_max, self.yaw_max)), False


class VectorField(Follower):
    name = "vector_field"

    def __init__(self, k_rep=1.5, influence=1.2, **kw):
        super().__init__(**kw)
        self.k_rep, self.influence = k_rep, influence

    def command(self, pose, path, world=None):
        self._prepare(path)
        x, y, psi = pose
        p = np.array([x, y])
        tgt = self._advance_target(p, self.lookahead)
        if self._reached(p):
            return 0.0, 0.0, True
        dg = self._goal_distance(p)
        vec = tgt - p
        vec = vec / (np.hypot(*vec) + 1e-9)
        if world is not None:
            for o in world.obstacles:
                to = p - np.array([o.x, o.y])
                surf = np.hypot(*to) - o.r - world.robot_radius
                if 0.0 < surf < self.influence:
                    vec += self.k_rep * (1.0 / surf - 1.0 / self.influence) / surf**2 * to / (np.hypot(*to) + 1e-9)
        alpha = _wrap(np.arctan2(vec[1], vec[0]) - psi)
        v = self.v_max * min(1.0, dg / self.slow_radius) * max(0.15, np.cos(alpha))
        yaw = 1.5 * alpha
        return v, float(np.clip(yaw, -self.yaw_max, self.yaw_max)), False


class DWA(Follower):
    """Dynamic Window Approach — reactive, obstacle-aware sampling over (v, omega)."""

    name = "dwa"

    def __init__(self, horizon=1.2, dt=0.2, n_v=6, n_w=15,
                 w_dist=1.0, w_head=0.6, w_clear=0.4, w_speed=0.15, **kw):
        super().__init__(yaw_max=kw.pop("yaw_max", 3.0), lookahead=kw.pop("lookahead", 1.2), **kw)
        self.horizon, self.dt, self.n_v, self.n_w = horizon, dt, n_v, n_w
        self.w_dist, self.w_head, self.w_clear, self.w_speed = w_dist, w_head, w_clear, w_speed

    def command(self, pose, path, world=None):
        self._prepare(path)
        p = np.array([pose[0], pose[1]])
        tgt = self._advance_target(p, self.lookahead)
        if self._reached(p):
            return 0.0, 0.0, True
        steps = max(1, int(self.horizon / self.dt))
        best, best_v, best_w = -np.inf, 0.0, 0.0
        for v in np.linspace(0.0, self.v_max, self.n_v):
            for w in np.linspace(-self.yaw_max, self.yaw_max, self.n_w):
                pts, end = _rollout(pose, v, w, self.dt, steps)
                clear = min(_clearance(world, px, py) for px, py in pts)
                if clear <= 0.0:
                    continue
                head_err = abs(_wrap(np.arctan2(tgt[1] - end[1], tgt[0] - end[0]) - end[2]))
                score = (-self.w_dist * np.hypot(end[0] - tgt[0], end[1] - tgt[1])
                         - self.w_head * head_err + self.w_clear * min(clear, 1.0) + self.w_speed * v)
                if score > best:
                    best, best_v, best_w = score, v, w
        return best_v, best_w, False


class MPCFollower(Follower):
    """Sampling-based model-predictive path tracker (cross-track + look-ahead + effort)."""

    name = "mpc"

    def __init__(self, horizon=1.2, dt=0.2, n_v=6, n_w=15,
                 w_track=0.5, w_term=1.0, w_head=0.4, w_effort=0.02, **kw):
        super().__init__(yaw_max=kw.pop("yaw_max", 3.0), lookahead=kw.pop("lookahead", 1.0), **kw)
        self.horizon, self.dt, self.n_v, self.n_w = horizon, dt, n_v, n_w
        self.w_track, self.w_term, self.w_head, self.w_effort = w_track, w_term, w_head, w_effort

    def command(self, pose, path, world=None):
        self._prepare(path)
        dense = self._dense
        p = np.array([pose[0], pose[1]])
        tgt = self._advance_target(p, self.lookahead)
        if self._reached(p):
            return 0.0, 0.0, True
        steps = max(1, int(self.horizon / self.dt))
        best, best_v, best_w = np.inf, 0.0, 0.0
        for v in np.linspace(0.1, self.v_max, self.n_v):
            for w in np.linspace(-self.yaw_max, self.yaw_max, self.n_w):
                pts, end = _rollout(pose, v, w, self.dt, steps)
                track = np.mean([np.min(np.hypot(dense[:, 0] - px, dense[:, 1] - py)) for px, py in pts])
                term = np.hypot(end[0] - tgt[0], end[1] - tgt[1])
                head_err = abs(_wrap(np.arctan2(tgt[1] - end[1], tgt[0] - end[0]) - end[2]))
                cost = self.w_track * track + self.w_term * term + self.w_head * head_err + self.w_effort * w**2
                if cost < best:
                    best, best_v, best_w = cost, v, w
        return best_v, best_w, False


_REGISTRY: dict[str, type[Follower]] = {
    PurePursuit.name: PurePursuit,
    Stanley.name: Stanley,
    DWA.name: DWA,
    MPCFollower.name: MPCFollower,
    VectorField.name: VectorField,
}


def build_follower(name: str, **kwargs) -> Follower:
    key = name.strip().lower()
    if key not in _REGISTRY:
        raise KeyError(f"unknown follower {name!r}; available: {sorted(_REGISTRY)}")
    return _REGISTRY[key](**kwargs)


def list_followers() -> list[str]:
    return sorted(_REGISTRY)
