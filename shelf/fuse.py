"""Camera + lidar association.

The camera says what an object is and which direction it lies. The lidar says
how far. Neither sensor can do the other's job; joining them is the point of
this project and it is about fifteen lines of arithmetic.

Bearing 0 is the lidar's front, increasing clockwise, wrapped to [0, 360).
Floor-plan units are the frontend's: 1 unit = 0.2 m.
"""
import math
from dataclasses import dataclass

M_PER_UNIT = 0.2


@dataclass
class Detection:
    cls_name: str
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass
class Station:
    id: str
    name: str
    x: float
    y: float
    heading: float


@dataclass
class Sighting:
    cls_name: str
    conf: float
    bearing: float
    range_m: float
    x: float
    y: float


def wrap(deg: float) -> float:
    return deg % 360.0


def angular_delta(a: float, b: float) -> float:
    """Signed a - b, in (-180, 180]."""
    d = (a - b) % 360.0
    return d - 360.0 if d > 180.0 else d


def returns_within(rev, bearing: float, half_width: float) -> list[float]:
    return [m for a, m in rev.pts if abs(angular_delta(a, bearing)) <= half_width]


def robust_min(ranges: list[float]) -> float:
    """Roughly the 10th percentile, so one spurious near return cannot drag
    the range in. Below five samples there is no percentile worth trusting:
    use the median. From five samples up, the index is clamped to at least 1
    (the second-smallest) so a single outlier is always rejected even in the
    common 5-9 sample band, where a literal 10th percentile would round down
    to index 0 and let the outlier straight through."""
    if not ranges:
        raise ValueError("robust_min of an empty window")
    s = sorted(ranges)
    n = len(s)
    if n < 5:
        return s[n // 2]
    idx = min(n - 1, max(1, int(n * 0.10)))
    return s[idx]


def associate(det: Detection, frame_w: float, rev, station: Station, cfg):
    """One detection + one revolution -> a positioned sighting, or None.

    None means the lidar could not corroborate the detection. Never substitute
    a default range here; that is the difference between a measurement and a
    guess wearing a measurement's clothes.
    """
    cx = (det.x1 + det.x2) / 2.0
    if cfg.edge_reject > 0:
        margin = frame_w * cfg.edge_reject
        if cx < margin or cx > frame_w - margin:
            return None                      # lens distortion is worst at the edges

    theta_cam = (cx / frame_w - 0.5) * cfg.hfov
    bearing = wrap(station.heading + cfg.yaw_offset + theta_cam)

    box_w_frac = abs(det.x2 - det.x1) / frame_w
    half_width = max(0.75, box_w_frac * cfg.hfov / 2.0)

    window = returns_within(rev, bearing, half_width)
    if not window:
        return None

    rng = robust_min(window)
    units = rng / M_PER_UNIT
    rad = math.radians(bearing)
    return Sighting(cls_name=det.cls_name, conf=det.conf,
                    bearing=bearing, range_m=rng,
                    x=station.x + math.sin(rad) * units,
                    y=station.y - math.cos(rad) * units)
