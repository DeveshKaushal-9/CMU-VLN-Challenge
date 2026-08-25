"""Occupancy mapping and frontier selection built from /terrain_map_ext.

The base autonomy stack publishes terrain points whose intensity channel is
height above the estimated ground plane, so intensity doubles as a
traversability cost: low means drivable floor, high means obstacle.
"""

import numpy as np
from collections import deque

UNKNOWN = 0
FREE = 1
OBSTACLE = 2

# Traversability: points lower than this above ground are drivable floor.
OBSTACLE_HEIGHT = 0.15


class OccupancyGrid:
    """Fixed-extent 2D grid in the map frame."""

    def __init__(self, resolution=0.2, extent=40.0, robot_radius=0.35):
        self.res = resolution
        self.extent = extent
        self.n = int(2 * extent / resolution)
        self.origin = -extent
        self.grid = np.zeros((self.n, self.n), dtype=np.uint8)
        self.inflate_cells = max(1, int(round(robot_radius / resolution)))

    # -- coordinate helpers -------------------------------------------------
    def to_cell(self, x, y):
        i = int((x - self.origin) / self.res)
        j = int((y - self.origin) / self.res)
        return i, j

    def to_world(self, i, j):
        return self.origin + (i + 0.5) * self.res, self.origin + (j + 0.5) * self.res

    def in_bounds(self, i, j):
        return 0 <= i < self.n and 0 <= j < self.n

    # -- updating -----------------------------------------------------------
    def update(self, pts):
        """Fold a terrain cloud (N,4 x/y/z/intensity) into the grid."""
        if pts.shape[0] == 0:
            return
        idx = ((pts[:, :2] - self.origin) / self.res).astype(np.int32)
        ok = ((idx >= 0) & (idx < self.n)).all(axis=1)
        idx, inten = idx[ok], pts[ok, 3]
        if idx.shape[0] == 0:
            return

        free = inten < OBSTACLE_HEIGHT
        # Free first, then obstacles, so an obstacle observation always wins
        # within a single update.
        self.grid[idx[free, 0], idx[free, 1]] = FREE
        blocked = ~free
        self.grid[idx[blocked, 0], idx[blocked, 1]] = OBSTACLE

    def inflated_obstacles(self):
        """Boolean mask of obstacle cells widened by the robot radius."""
        occ = self.grid == OBSTACLE
        out = occ.copy()
        r = self.inflate_cells
        for di in range(-r, r + 1):
            for dj in range(-r, r + 1):
                if di == 0 and dj == 0:
                    continue
                out |= np.roll(np.roll(occ, di, axis=0), dj, axis=1)
        return out

    # -- planning -----------------------------------------------------------
    def distance_field(self, start_xy):
        """BFS over drivable cells; returns int32 distances (-1 = unreachable)."""
        blocked = self.inflated_obstacles()
        drivable = (self.grid == FREE) & ~blocked

        dist = np.full((self.n, self.n), -1, dtype=np.int32)
        si, sj = self.to_cell(*start_xy)
        if not self.in_bounds(si, sj):
            return dist

        # The robot's own cell may read as unknown or inflated; seed from the
        # nearest drivable cell so exploration can still start.
        if not drivable[si, sj]:
            seed = self._nearest_drivable(si, sj, drivable)
            if seed is None:
                return dist
            si, sj = seed

        dist[si, sj] = 0
        q = deque([(si, sj)])
        while q:
            i, j = q.popleft()
            d = dist[i, j] + 1
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                a, b = i + di, j + dj
                if 0 <= a < self.n and 0 <= b < self.n and dist[a, b] < 0 and drivable[a, b]:
                    dist[a, b] = d
                    q.append((a, b))
        return dist

    def _nearest_drivable(self, si, sj, drivable, max_radius=25):
        for r in range(1, max_radius):
            lo_i, hi_i = max(0, si - r), min(self.n, si + r + 1)
            lo_j, hi_j = max(0, sj - r), min(self.n, sj + r + 1)
            window = drivable[lo_i:hi_i, lo_j:hi_j]
            hits = np.argwhere(window)
            if hits.size:
                best = hits[np.abs(hits - [si - lo_i, sj - lo_j]).sum(axis=1).argmin()]
                return lo_i + best[0], lo_j + best[1]
        return None

    def frontiers(self):
        """Free cells that border unmapped space.

        Inflation is deliberately not applied here: the frontier band lies at
        the edge of sensor range, which is exactly where inflated obstacles
        would erase it and stall exploration.
        """
        free = self.grid == FREE
        unknown = self.grid == UNKNOWN
        touches_unknown = np.zeros_like(unknown)
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            touches_unknown |= np.roll(np.roll(unknown, di, axis=0), dj, axis=1)
        return free & touches_unknown

    def next_frontier(self, robot_xy, visited=(), min_gap=1.5, stride=2):
        """Pick the frontier that best trades information gain against travel.

        Returns a drivable (x, y) goal in the map frame, or None when nothing
        reachable borders unmapped space.
        """
        front = self.frontiers()
        if not front.any():
            return None

        dist = self.distance_field(robot_xy)
        blocked = self.inflated_obstacles()
        drivable = (self.grid == FREE) & ~blocked
        unknown = self.grid == UNKNOWN

        cells = np.argwhere(front)
        if cells.shape[0] == 0:
            return None
        if stride > 1 and cells.shape[0] > 400:
            cells = cells[::stride]

        best, best_score = None, -1e18
        for i, j in cells:
            approach = self._approach_cell(i, j, drivable, dist)
            if approach is None:
                continue
            ai, aj = approach
            x, y = self.to_world(ai, aj)
            if any((x - vx) ** 2 + (y - vy) ** 2 < min_gap ** 2 for vx, vy in visited):
                continue
            lo_i, hi_i = max(0, i - 5), min(self.n, i + 6)
            lo_j, hi_j = max(0, j - 5), min(self.n, j + 6)
            gain = int(unknown[lo_i:hi_i, lo_j:hi_j].sum())
            travel = dist[ai, aj] * self.res
            score = gain - 3.0 * travel
            if score > best_score:
                best, best_score = (x, y), score
        return best

    def _approach_cell(self, i, j, drivable, dist, radius=3):
        """Nearest reachable drivable cell to a frontier cell."""
        if drivable[i, j] and dist[i, j] >= 0:
            return i, j
        best, best_d = None, None
        for di in range(-radius, radius + 1):
            for dj in range(-radius, radius + 1):
                a, b = i + di, j + dj
                if not (0 <= a < self.n and 0 <= b < self.n):
                    continue
                if not drivable[a, b] or dist[a, b] < 0:
                    continue
                d = di * di + dj * dj
                if best_d is None or d < best_d:
                    best, best_d = (a, b), d
        return best

    def coverage(self):
        """Number of cells mapped so far, as a crude progress signal."""
        return int((self.grid != UNKNOWN).sum())

    def is_drivable(self, x, y):
        i, j = self.to_cell(x, y)
        if not self.in_bounds(i, j):
            return False
        return bool(self.grid[i, j] == FREE)
