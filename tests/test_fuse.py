import math
import pytest
from shelf.capture import Revolution
from shelf.config import Config
from shelf.fuse import (Detection, Station, wrap, angular_delta,
                        returns_within, robust_min, associate)

CFG = Config(hfov=68.0, yaw_offset=0.0, edge_reject=0.0)
STATION = Station(id="s1", name="Aisle", x=50.0, y=34.0, heading=0.0)
FRAME_W = 1000.0


def ring(range_m=3.0, step=0.72):
    """A revolution that returns the same range in every direction."""
    n = int(360 / step)
    return Revolution(t=0.0, pts=[(i * step, range_m) for i in range(n)])


def box(center_x, width=100.0):
    return Detection(cls_name="cup", conf=0.9,
                     x1=center_x - width / 2, y1=100.0,
                     x2=center_x + width / 2, y2=200.0)


def test_wrap_normalises():
    assert wrap(370.0) == pytest.approx(10.0)
    assert wrap(-10.0) == pytest.approx(350.0)


def test_angular_delta_crosses_zero():
    assert angular_delta(10.0, 350.0) == pytest.approx(20.0)
    assert angular_delta(350.0, 10.0) == pytest.approx(-20.0)


def test_centred_box_maps_to_station_heading():
    s = associate(box(FRAME_W / 2), FRAME_W, ring(3.0), STATION, CFG)
    assert s is not None
    assert s.bearing == pytest.approx(0.0, abs=0.5)
    assert s.range_m == pytest.approx(3.0, abs=0.01)


def test_box_right_of_centre_maps_clockwise():
    # A quarter of the way right of centre is a quarter of the half-FOV.
    s = associate(box(FRAME_W * 0.75), FRAME_W, ring(3.0), STATION, CFG)
    assert s.bearing == pytest.approx(17.0, abs=0.5)     # 0.25 * 68


def test_yaw_offset_and_heading_both_apply():
    cfg = Config(hfov=68.0, yaw_offset=10.0, edge_reject=0.0)
    st = Station(id="s", name="n", x=0.0, y=0.0, heading=90.0)
    s = associate(box(FRAME_W / 2), FRAME_W, ring(2.0), st, cfg)
    assert s.bearing == pytest.approx(100.0, abs=0.5)


def test_no_lidar_return_in_window_returns_none():
    """The honesty rule: recognised in mid-air is not a sighting."""
    empty = Revolution(t=0.0, pts=[(180.0, 2.0)])       # nothing near bearing 0
    assert associate(box(FRAME_W / 2), FRAME_W, empty, STATION, CFG) is None


def test_robust_min_ignores_a_single_spurious_near_return():
    ranges = [0.2] + [3.0] * 20
    assert robust_min(ranges) == pytest.approx(3.0, abs=0.05)


def test_robust_min_falls_back_to_median_when_sparse():
    assert robust_min([2.0, 3.0, 4.0]) == pytest.approx(3.0)


def test_returns_within_wraps_around_zero():
    rev = Revolution(t=0.0, pts=[(359.0, 1.0), (1.0, 1.5), (180.0, 9.0)])
    got = returns_within(rev, bearing=0.0, half_width=2.0)
    assert sorted(got) == [1.0, 1.5]


def test_position_is_station_plus_polar_offset():
    st = Station(id="s", name="n", x=10.0, y=20.0, heading=0.0)
    s = associate(box(FRAME_W / 2), FRAME_W, ring(2.0), st, CFG)
    # bearing 0 is "up" on the floor plan, so y decreases; 2 m = 10 units
    assert s.x == pytest.approx(10.0, abs=0.1)
    assert s.y == pytest.approx(10.0, abs=0.1)


def test_edge_detections_are_rejected():
    cfg = Config(hfov=68.0, yaw_offset=0.0, edge_reject=0.10)
    assert associate(box(FRAME_W * 0.02, width=20), FRAME_W,
                     ring(3.0), STATION, cfg) is None
