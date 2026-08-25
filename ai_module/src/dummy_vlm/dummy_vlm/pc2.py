"""Minimal PointCloud2 -> numpy conversion.

Written by hand rather than using sensor_msgs_py so the module has no
dependency beyond numpy, which keeps the submitted docker image identical
to the one the organizers provide.
"""

import numpy as np

# sensor_msgs/PointField datatype enum -> numpy dtype
_DTYPE = {
    1: np.int8, 2: np.uint8,
    3: np.int16, 4: np.uint16,
    5: np.int32, 6: np.uint32,
    7: np.float32, 8: np.float64,
}

_EMPTY = np.empty((0, 4), dtype=np.float32)


def to_xyzi(msg):
    """Convert a PointCloud2 into an (N, 4) float32 array of x, y, z, intensity.

    Intensity is zero-filled when the cloud does not carry the field.
    Rows holding non-finite coordinates are dropped.
    """
    n = msg.width * msg.height
    if n == 0:
        return _EMPTY

    fields = {f.name: f for f in msg.fields}
    if not all(k in fields for k in ('x', 'y', 'z')):
        return _EMPTY

    raw = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    if raw.size < n * msg.point_step:
        n = raw.size // msg.point_step
        if n == 0:
            return _EMPTY
    raw = raw[:n * msg.point_step].reshape(n, msg.point_step)

    out = np.zeros((n, 4), dtype=np.float32)
    for col, name in enumerate(('x', 'y', 'z', 'intensity')):
        f = fields.get(name)
        if f is None:
            continue
        dt = _DTYPE.get(f.datatype)
        if dt is None:
            continue
        width = np.dtype(dt).itemsize
        chunk = raw[:, f.offset:f.offset + width].copy()
        out[:, col] = chunk.view(dt).ravel().astype(np.float32)

    ok = np.isfinite(out[:, :3]).all(axis=1)
    # Registered scans carry exact-zero padding rows; left in, they pile up at
    # the map origin and back-project into a phantom object there.
    ok &= ~np.all(np.abs(out[:, :3]) < 1e-6, axis=1)
    return out[ok]
