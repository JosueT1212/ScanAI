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
