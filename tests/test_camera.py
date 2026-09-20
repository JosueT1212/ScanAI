from shelf import camera
from shelf.config import Config


def test_picks_last_enumerated_when_index_is_default(monkeypatch):
    monkeypatch.setattr(camera, "list_cameras", lambda: [(0, "camera 0"), (2, "camera 2")])
    src = camera.CameraSource(Config())
    assert src.index == 2


def test_no_cameras_found_falls_back_to_zero(monkeypatch):
    monkeypatch.setattr(camera, "list_cameras", lambda: [])
    src = camera.CameraSource(Config())
    assert src.index == 0


def test_config_camera_index_overrides_pick(monkeypatch):
    monkeypatch.setattr(camera, "list_cameras", lambda: [(0, "camera 0"), (2, "camera 2")])
    src = camera.CameraSource(Config(camera_index=5))
    assert src.index == 5


def test_explicit_index_argument_wins_over_everything(monkeypatch):
    monkeypatch.setattr(camera, "list_cameras", lambda: [(0, "camera 0"), (2, "camera 2")])
    src = camera.CameraSource(Config(camera_index=5), index=7)
    assert src.index == 7
