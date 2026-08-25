"""Panoramic camera model and 2D->3D back-projection.

The 360 camera publishes a 1920x640 equirectangular image with 360 deg
horizontal and 120 deg vertical field of view. The system remaps images so the
camera frame stays aligned with the lidar frame, so a pixel maps to a bearing
in the sensor frame with no extra calibration.
"""

import numpy as np

WIDTH = 1920
HEIGHT = 640
HFOV = 2.0 * np.pi          # 360 deg
VFOV = np.deg2rad(120.0)    # 120 deg


def pixel_to_bearing(u, v, width=WIDTH, height=HEIGHT):
    """Pixel -> (azimuth, elevation) in radians, sensor frame.

    Azimuth grows counter-clockwise and is zero straight ahead; elevation is
    positive upwards.
    """
    az = (0.5 - np.asarray(u, dtype=np.float64) / width) * HFOV
    el = (0.5 - np.asarray(v, dtype=np.float64) / height) * VFOV
    return az, el


def points_to_bearing(pts):
    """Sensor-frame points (N,3) -> (azimuth, elevation, range)."""
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    rng = np.sqrt(x * x + y * y + z * z)
    az = np.arctan2(y, x)
    el = np.arctan2(z, np.sqrt(x * x + y * y))
    return az, el, rng


def wrap_angle(a):
    """Wrap to [-pi, pi)."""
    return (a + np.pi) % (2.0 * np.pi) - np.pi


class Keyframe:
    """One captured view: the image plus the pose and scan that go with it."""

    __slots__ = ('image', 'position', 'yaw', 'points', 'stamp')

    def __init__(self, image, position, yaw, points, stamp):
        self.image = image        # HxWx3 uint8, BGR
        self.position = position  # (x, y, z) map frame
        self.yaw = yaw            # robot heading, map frame
        self.points = points      # (N,3) map-frame scan points
        self.stamp = stamp

    def points_in_box(self, x0, y0, x1, y1, margin=0.02):
        """Map-frame scan points whose bearing falls inside an image box.

        Returns the subset of `self.points` seen through the given pixel
        rectangle, which is what turns a 2D detection into a 3D location.
        """
        if self.points.shape[0] == 0:
            return np.empty((0, 3), dtype=np.float32)

        # Scan points are in the map frame; rotate into the sensor frame so the
        # bearings line up with image columns.
        rel = self.points - np.asarray(self.position, dtype=np.float32)
        c, s = np.cos(-self.yaw), np.sin(-self.yaw)
        local = np.empty_like(rel)
        local[:, 0] = c * rel[:, 0] - s * rel[:, 1]
        local[:, 1] = s * rel[:, 0] + c * rel[:, 1]
        local[:, 2] = rel[:, 2]

        az, el, rng = points_to_bearing(local)

        az0, el0 = pixel_to_bearing(x1, y1)   # right/bottom -> min az, min el
        az1, el1 = pixel_to_bearing(x0, y0)   # left/top     -> max az, max el
        el_lo, el_hi = min(el0, el1) - margin, max(el0, el1) + margin

        # Azimuth needs circular containment because the panorama wraps.
        centre = wrap_angle((az0 + az1) / 2.0)
        half = abs(wrap_angle(az1 - az0)) / 2.0 + margin
        in_az = np.abs(wrap_angle(az - centre)) <= half
        in_el = (el >= el_lo) & (el <= el_hi)

        keep = in_az & in_el & (rng > 0.15)
        return self.points[keep]
