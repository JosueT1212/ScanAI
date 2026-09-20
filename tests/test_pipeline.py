import numpy as np
from shelf.capture import Revolution
from shelf.config import Config
from shelf.detect import StubDetector
from shelf.fuse import Station
from shelf.pipeline import Pipeline
from shelf.store import Store


class FakeLidar:
    def __init__(self, revs): self.revs = revs
    def revolutions(self):
        for r in self.revs:
            yield r
    def close(self): pass


class FakeCamera:
    def start(self): pass
    def latest(self): return np.zeros((480, 640, 3), dtype=np.uint8)
    def available(self): return True
    def close(self): pass


def ring(m=3.0):
    return Revolution(t=0.0, pts=[(i * 0.72, m) for i in range(500)])


def test_one_tick_produces_a_stored_sighting(tmp_path):
    store = Store(str(tmp_path / "p.db")); store.init(); store.seed_coco_defaults()
    p = Pipeline(Config(edge_reject=0.0), store, FakeLidar([ring()]),
                 FakeCamera(), StubDetector(Config()))
    p.tick(ring())
    snap = p.snapshot()
    assert len(snap["sightings"]) == 1
    assert snap["sightings"][0]["cls_name"] == "cup"
    assert len(store.items()) == 1


def test_snapshot_reports_camera_state(tmp_path):
    store = Store(str(tmp_path / "q.db")); store.init()
    p = Pipeline(Config(), store, FakeLidar([]), FakeCamera(), StubDetector(Config()))
    assert p.snapshot()["camera_ok"] is True


def test_a_detection_the_lidar_cannot_corroborate_is_not_recorded(tmp_path):
    """The honesty rule. A recognised object with no lidar return behind it is an
    association failure, not a sighting at a guessed distance. Deleting the skip in
    tick() must fail this test."""
    store = Store(str(tmp_path / "n.db")); store.init(); store.seed_coco_defaults()
    empty = Revolution(t=0.0, pts=[(180.0, 2.0)])      # nothing near the box's bearing
    p = Pipeline(Config(edge_reject=0.0), store, FakeLidar([empty]),
                 FakeCamera(), StubDetector(Config()))
    p.tick(empty)
    snap = p.snapshot()
    assert snap["detections"], "the detector should still report the box"
    assert snap["sightings"] == [], "no lidar support means no sighting"
    assert store.items() == [], "and nothing may reach the store"
