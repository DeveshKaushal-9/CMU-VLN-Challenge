"""Object candidate extraction from the accumulated LiDAR scan.

Registered scans are folded into a voxel set as the robot explores. On demand
the voxels are segmented into connected clusters, each of which becomes a
candidate object with a centroid and an axis-aligned bounding box. The
perception layer labels these candidates; this module only finds them.
"""

import numpy as np
from collections import deque

VOXEL = 0.08          # metres, accumulation + clustering resolution
FLOOR_MARGIN = 0.08   # ignore points this close to the floor
CEILING = 2.2         # ignore points above this height over the floor


class ObjectCandidate:
    __slots__ = ('center', 'extent', 'n_points', 'label', 'score')

    def __init__(self, center, extent, n_points):
        self.center = center      # (x, y, z) map frame
        self.extent = extent      # (length, width, height)
        self.n_points = n_points
        self.label = None         # filled in by the perception layer
        self.score = 0.0

    @property
    def xy(self):
        return self.center[0], self.center[1]

    @property
    def volume(self):
        return float(np.prod(self.extent))

    def __repr__(self):
        return (f'<Candidate {self.label or "?"} at '
                f'({self.center[0]:.2f}, {self.center[1]:.2f}, {self.center[2]:.2f}) '
                f'n={self.n_points}>')


class ObjectMap:
    """Accumulates registered scans and segments them into candidates."""

    def __init__(self, voxel=VOXEL, max_voxels=400000):
        self.voxel = voxel
        self.max_voxels = max_voxels
        self._sums = {}    # voxel key -> [sx, sy, sz, count]

    def add_scan(self, pts):
        """Fold an (N,4) registered scan into the voxel accumulator."""
        if pts.shape[0] == 0 or len(self._sums) >= self.max_voxels:
            return
        xyz = pts[:, :3]
        keys = np.floor(xyz / self.voxel).astype(np.int32)
        for k, p in zip(map(tuple, keys), xyz):
            slot = self._sums.get(k)
            if slot is None:
                self._sums[k] = [float(p[0]), float(p[1]), float(p[2]), 1]
            else:
                slot[0] += float(p[0])
                slot[1] += float(p[1])
                slot[2] += float(p[2])
                slot[3] += 1

    @property
    def n_voxels(self):
        return len(self._sums)

    def floor_height(self):
        if not self._sums:
            return 0.0
        zs = np.fromiter((v[2] / v[3] for v in self._sums.values()),
                         dtype=np.float32, count=len(self._sums))
        return float(np.percentile(zs, 2.0))

    def candidates(self, min_points=25, max_extent=3.0):
        """Segment accumulated voxels into object candidates.

        Clusters wider than `max_extent` in both horizontal axes are dropped as
        walls or floor rather than objects.
        """
        if not self._sums:
            return []

        floor = self.floor_height()
        lo = floor + FLOOR_MARGIN
        hi = floor + CEILING

        live = {k: v for k, v in self._sums.items() if lo <= v[2] / v[3] <= hi}
        if not live:
            return []

        neighbours = [(dx, dy, dz)
                      for dx in (-1, 0, 1)
                      for dy in (-1, 0, 1)
                      for dz in (-1, 0, 1)
                      if not (dx == dy == dz == 0)]

        seen = set()
        out = []
        for start in live:
            if start in seen:
                continue
            seen.add(start)
            q = deque([start])
            members = []
            while q:
                key = q.popleft()
                members.append(key)
                kx, ky, kz = key
                for dx, dy, dz in neighbours:
                    nb = (kx + dx, ky + dy, kz + dz)
                    if nb in live and nb not in seen:
                        seen.add(nb)
                        q.append(nb)

            pts = np.array([[live[k][0] / live[k][3],
                             live[k][1] / live[k][3],
                             live[k][2] / live[k][3]] for k in members],
                           dtype=np.float32)
            count = int(sum(live[k][3] for k in members))
            if count < min_points:
                continue

            mins, maxs = pts.min(axis=0), pts.max(axis=0)
            extent = maxs - mins
            if extent[0] > max_extent and extent[1] > max_extent:
                continue  # wall / floor slab, not an object

            center = (mins + maxs) / 2.0
            # Pad thin extents so the published marker has real volume.
            extent = np.maximum(extent, 0.1)
            out.append(ObjectCandidate(tuple(map(float, center)),
                                       tuple(map(float, extent)),
                                       count))

        out.sort(key=lambda c: c.n_points, reverse=True)
        return out
