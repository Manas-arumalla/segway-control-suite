"""Global path planners + a name-based registry.

All planners share one interface — ``plan(world, start, goal) -> (K,2) path or None`` — so
the navigator and benchmark can swap them freely. Implemented: grid search (A*, Dijkstra),
sampling (RRT, RRT*, PRM), and an artificial Potential Field.
"""

from __future__ import annotations

import heapq
from abc import ABC, abstractmethod

import numpy as np

from .world import World

_DIAG = np.sqrt(2.0)
_MOVES = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
          (-1, -1, _DIAG), (-1, 1, _DIAG), (1, -1, _DIAG), (1, 1, _DIAG)]


class Planner(ABC):
    """Common interface for global planners."""

    name: str = "base"

    @abstractmethod
    def plan(self, world: World, start, goal) -> np.ndarray | None:
        """Return an ``(K, 2)`` collision-free path from ``start`` to ``goal``, or ``None``."""
        raise NotImplementedError


# ===== grid search (A* / Dijkstra) ======================================
def _grid_search(world: World, start, goal, use_heuristic: bool):
    si, sj = world.to_cell(*start)
    gi, gj = world.to_cell(*goal)
    if not world.cell_free(si, sj) or not world.cell_free(gi, gj):
        return None
    res = world.resolution

    def h(i, j):
        return np.hypot(i - gi, j - gj) * res if use_heuristic else 0.0

    open_heap = [(h(si, sj), 0.0, (si, sj))]
    g = {(si, sj): 0.0}
    came: dict = {}
    reached = (si, sj) == (gi, gj)
    while open_heap:
        _, gc, (ci, cj) = heapq.heappop(open_heap)
        if (ci, cj) == (gi, gj):
            reached = True
            break
        if gc > g.get((ci, cj), np.inf):
            continue
        for di, dj, cost in _MOVES:
            ni, nj = ci + di, cj + dj
            if not world.cell_free(ni, nj):
                continue
            if di != 0 and dj != 0 and not (world.cell_free(ci + di, cj) and world.cell_free(ci, cj + dj)):
                continue  # forbid corner-cutting through obstacle corners
            ng = gc + cost * res
            if ng < g.get((ni, nj), np.inf):
                g[(ni, nj)] = ng
                came[(ni, nj)] = (ci, cj)
                heapq.heappush(open_heap, (ng + h(ni, nj), ng, (ni, nj)))

    if not reached:
        return None
    cells = [(gi, gj)]
    cur = (gi, gj)
    while cur != (si, sj):
        cur = came.get(cur)
        if cur is None:
            return None
        cells.append(cur)
    cells.reverse()
    pts = [list(world.cell_center(i, j)) for i, j in cells]
    pts[0] = list(start)
    pts[-1] = list(goal)
    return np.array(pts, dtype=float)


class AStarPlanner(Planner):
    name = "a_star"

    def plan(self, world, start, goal):
        return _grid_search(world, start, goal, use_heuristic=True)


class DijkstraPlanner(Planner):
    name = "dijkstra"

    def plan(self, world, start, goal):
        return _grid_search(world, start, goal, use_heuristic=False)


# ===== RRT ===============================================================
class RRTPlanner(Planner):
    name = "rrt"

    def __init__(self, step: float = 0.6, goal_bias: float = 0.1, max_iter: int = 5000, seed: int = 0):
        self.step, self.goal_bias, self.max_iter, self.seed = step, goal_bias, max_iter, seed

    def plan(self, world, start, goal):
        rng = np.random.default_rng(self.seed)
        start, goal = np.asarray(start, float), np.asarray(goal, float)
        if not world.is_free(*start) or not world.is_free(*goal):
            return None
        nodes = [start]
        parent = [-1]
        for _ in range(self.max_iter):
            samp = goal if rng.random() < self.goal_bias else np.array(
                [rng.uniform(0, world.width), rng.uniform(0, world.height)])
            ni = int(np.argmin([np.hypot(*(samp - n)) for n in nodes]))
            near = nodes[ni]
            direction = samp - near
            dist = np.hypot(*direction)
            if dist < 1e-9:
                continue
            new = near + direction / dist * min(self.step, dist)
            if not world.is_segment_free(near, new):
                continue
            nodes.append(new)
            parent.append(ni)
            if np.hypot(*(goal - new)) <= self.step and world.is_segment_free(new, goal):
                nodes.append(goal)
                parent.append(len(nodes) - 2)
                return _backtrack(nodes, parent)
        return None


# ===== RRT* ==============================================================
class RRTStarPlanner(Planner):
    name = "rrt_star"

    def __init__(self, step: float = 0.6, goal_bias: float = 0.1, radius: float = 1.2,
                 max_iter: int = 5000, seed: int = 0):
        self.step, self.goal_bias, self.radius = step, goal_bias, radius
        self.max_iter, self.seed = max_iter, seed

    def plan(self, world, start, goal):
        rng = np.random.default_rng(self.seed)
        start, goal = np.asarray(start, float), np.asarray(goal, float)
        if not world.is_free(*start) or not world.is_free(*goal):
            return None
        nodes = [start]
        parent = [-1]
        cost = [0.0]
        goal_idx, goal_cost = -1, np.inf
        for _ in range(self.max_iter):
            samp = goal if rng.random() < self.goal_bias else np.array(
                [rng.uniform(0, world.width), rng.uniform(0, world.height)])
            arr = np.array(nodes)
            ni = int(np.argmin(np.hypot(arr[:, 0] - samp[0], arr[:, 1] - samp[1])))
            near = nodes[ni]
            direction = samp - near
            dist = np.hypot(*direction)
            if dist < 1e-9:
                continue
            new = near + direction / dist * min(self.step, dist)
            if not world.is_segment_free(near, new):
                continue
            # choose best parent within radius
            arr = np.array(nodes)
            near_ids = [k for k in range(len(nodes))
                        if np.hypot(*(arr[k] - new)) <= self.radius and world.is_segment_free(nodes[k], new)]
            best_p, best_c = ni, cost[ni] + np.hypot(*(new - near))
            for k in near_ids:
                c = cost[k] + np.hypot(*(nodes[k] - new))
                if c < best_c:
                    best_p, best_c = k, c
            nodes.append(new)
            parent.append(best_p)
            cost.append(best_c)
            new_idx = len(nodes) - 1
            # rewire neighbors through the new node
            for k in near_ids:
                c = best_c + np.hypot(*(nodes[k] - new))
                if c < cost[k]:
                    parent[k] = new_idx
                    cost[k] = c
            if np.hypot(*(goal - new)) <= self.step and world.is_segment_free(new, goal):
                gc = best_c + np.hypot(*(goal - new))
                if gc < goal_cost:
                    goal_cost, goal_idx = gc, new_idx
        if goal_idx == -1:
            return None
        nodes.append(goal)
        parent.append(goal_idx)
        return _backtrack(nodes, parent)


# ===== PRM ===============================================================
class PRMPlanner(Planner):
    name = "prm"

    def __init__(self, n_samples: int = 300, radius: float = 1.5, seed: int = 0):
        self.n_samples, self.radius, self.seed = n_samples, radius, seed

    def plan(self, world, start, goal):
        rng = np.random.default_rng(self.seed)
        start, goal = np.asarray(start, float), np.asarray(goal, float)
        if not world.is_free(*start) or not world.is_free(*goal):
            return None
        pts = [start, goal]
        while len(pts) < self.n_samples + 2:
            p = np.array([rng.uniform(0, world.width), rng.uniform(0, world.height)])
            if world.is_free(*p):
                pts.append(p)
        pts_arr = np.array(pts)
        adj: dict[int, list[tuple[int, float]]] = {i: [] for i in range(len(pts))}
        for i in range(len(pts)):
            d = np.hypot(pts_arr[:, 0] - pts_arr[i, 0], pts_arr[:, 1] - pts_arr[i, 1])
            for j in np.where((d > 0) & (d <= self.radius))[0]:
                j = int(j)
                if j > i and world.is_segment_free(pts[i], pts[j]):
                    w = float(d[j])
                    adj[i].append((j, w))
                    adj[j].append((i, w))
        order = _dijkstra_graph(adj, 0, 1)   # start=0, goal=1
        if order is None:
            return None
        return np.array([pts[k] for k in order], dtype=float)


# ===== Potential Field ===================================================
class PotentialFieldPlanner(Planner):
    name = "potential_field"

    def __init__(self, k_att: float = 1.0, k_rep: float = 2.5, influence: float = 1.5,
                 step: float = 0.1, max_iter: int = 4000, goal_tol: float = 0.3, seed: int = 0):
        self.k_att, self.k_rep, self.influence = k_att, k_rep, influence
        self.step, self.max_iter, self.goal_tol, self.seed = step, max_iter, goal_tol, seed

    def plan(self, world, start, goal):
        rng = np.random.default_rng(self.seed)
        start, goal = np.asarray(start, float), np.asarray(goal, float)
        if not world.is_free(*start) or not world.is_free(*goal):
            return None
        p = start.copy()
        path = [p.copy()]
        for _ in range(self.max_iter):
            if np.hypot(*(goal - p)) < self.goal_tol:
                path.append(goal)
                return np.array(path, dtype=float)
            force = -self.k_att * (p - goal)
            force = force / (np.hypot(*force) + 1e-9)
            for o in world.obstacles:
                to = p - np.array([o.x, o.y])
                surf = np.hypot(*to) - o.r - world.robot_radius
                if 0.0 < surf < self.influence:
                    force += self.k_rep * (1.0 / surf - 1.0 / self.influence) / surf**2 * to / (np.hypot(*to) + 1e-9)
            mag = np.hypot(*force)
            if mag < 1e-6:
                force = rng.normal(size=2)          # nudge out of a flat spot
                mag = np.hypot(*force)
            new = p + force / mag * self.step
            if not world.is_free(*new):
                new = p + np.array([-force[1], force[0]]) / mag * self.step  # slide tangentially
                if not world.is_free(*new):
                    return None
            p = new
            path.append(p.copy())
        return None


# ===== helpers ===========================================================
def _backtrack(nodes, parent):
    idx = len(nodes) - 1
    path = []
    while idx != -1:
        path.append(nodes[idx])
        idx = parent[idx]
    path.reverse()
    return np.array(path, dtype=float)


def _dijkstra_graph(adj, src, dst):
    heap = [(0.0, src)]
    dist = {src: 0.0}
    came: dict = {}
    while heap:
        d, u = heapq.heappop(heap)
        if u == dst:
            order = [dst]
            while order[-1] != src:
                order.append(came[order[-1]])
            order.reverse()
            return order
        if d > dist.get(u, np.inf):
            continue
        for v, w in adj[u]:
            nd = d + w
            if nd < dist.get(v, np.inf):
                dist[v] = nd
                came[v] = u
                heapq.heappush(heap, (nd, v))
    return None


_REGISTRY: dict[str, type[Planner]] = {
    AStarPlanner.name: AStarPlanner,
    DijkstraPlanner.name: DijkstraPlanner,
    RRTPlanner.name: RRTPlanner,
    RRTStarPlanner.name: RRTStarPlanner,
    PRMPlanner.name: PRMPlanner,
    PotentialFieldPlanner.name: PotentialFieldPlanner,
}


def build_planner(name: str, **kwargs) -> Planner:
    key = name.strip().lower()
    if key not in _REGISTRY:
        raise KeyError(f"unknown planner {name!r}; available: {sorted(_REGISTRY)}")
    return _REGISTRY[key](**kwargs)


def list_planners() -> list[str]:
    return sorted(_REGISTRY)
