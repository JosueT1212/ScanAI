import numpy as np
from fastapi.testclient import TestClient
from shelf.app import create_app
from shelf.config import Config
from shelf.detect import StubDetector
from shelf.pipeline import Pipeline
from shelf.store import Store
from shelf.capture import Revolution


class FakeLidar:
    def revolutions(self): return iter(())
    def close(self): pass


class FakeCamera:
    def start(self): pass
    def latest(self): return np.zeros((480, 640, 3), dtype=np.uint8)
    def available(self): return True
    def close(self): pass


def build(tmp_path):
    store = Store(str(tmp_path / "a.db")); store.init(); store.seed_coco_defaults()
    p = Pipeline(Config(edge_reject=0.0), store, FakeLidar(), FakeCamera(),
                 StubDetector(Config()))
    return TestClient(create_app(p, store)), p, store


def test_locations_endpoint(tmp_path):
    c, _, _ = build(tmp_path)
    r = c.get("/api/locations")
    assert r.status_code == 200
    assert len(r.json()) == 8


def test_activate_station_switches_pose(tmp_path):
    c, p, store = build(tmp_path)
    from shelf.fuse import Station
    store.upsert_station(Station("far", "Far corner", 10.0, 10.0, 90.0))
    assert c.post("/api/stations/far/activate").status_code == 200
    assert p.station.id == "far"


def test_websocket_pushes_a_snapshot(tmp_path):
    c, p, _ = build(tmp_path)
    p.tick(Revolution(t=0.0, pts=[(i * 0.72, 3.0) for i in range(500)]))
    with c.websocket_connect("/ws/live") as ws:
        msg = ws.receive_json()
        assert "scan" in msg and "sightings" in msg
        assert len(msg["scan"]) == 500


def test_calibrate_yaw_needs_a_detection_and_a_scan(tmp_path):
    c, p, _ = build(tmp_path)
    r = c.post("/api/calibrate/yaw")
    assert r.status_code == 200
    assert r.json()["ok"] is False


def test_calibrate_yaw_ignores_a_closer_object_behind_the_rig(tmp_path):
    """The operator's target is dead ahead; something nearer elsewhere in the
    room must not win the alignment."""
    c, p, _ = build(tmp_path)
    rev = Revolution(t=0.0, pts=[(0.0, 3.0), (180.0, 0.4)])   # decoy much closer, behind
    p.tick(rev)
    body = c.post("/api/calibrate/yaw").json()
    assert body["ok"] is True
    assert body["target_bearing"] == 0.0, "aligned to the decoy behind the rig"
    assert body["target_range_m"] == 3.0


def test_web_dir_is_resolved_from_the_module_not_the_cwd():
    from shelf.app import WEB_DIR
    assert WEB_DIR.is_absolute() and (WEB_DIR / "index.html").exists()


def test_index_is_served_at_root(tmp_path):
    c, _, _ = build(tmp_path)
    assert c.get("/").status_code == 200
