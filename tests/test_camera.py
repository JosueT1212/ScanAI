from shelf import camera
from shelf.config import Config


def test_picks_last_enumerated_when_index_is_default(monkeypatch):
    monkeypatch.setattr(camera, "list_cameras", lambda: [(0, "camera 0"), (2, "camera 2")])
    src = camera.CameraSource(Config())
    assert src.index == 2


def test_auto_pick_warns_which_index_it_chose(monkeypatch, caplog):
    monkeypatch.setattr(camera, "list_cameras", lambda: [(0, "camera 0"), (2, "camera 2")])
    with caplog.at_level("WARNING"):
        camera.CameraSource(Config())
    assert any("2" in r.getMessage() and "auto" in r.getMessage()
               for r in caplog.records)


def test_no_cameras_found_also_warns(monkeypatch, caplog):
    monkeypatch.setattr(camera, "list_cameras", lambda: [])
    with caplog.at_level("WARNING"):
        camera.CameraSource(Config())
    assert any("0" in r.getMessage() for r in caplog.records)


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


def test_camera_index_zero_is_honoured_not_treated_as_unset(monkeypatch):
    """0 is a legitimate device index. A truthiness check here would silently
    fall through to auto-pick and grab the wrong camera."""
    monkeypatch.setattr(camera, "list_cameras", lambda: [(0, "camera 0"), (1, "camera 1")])
    src = camera.CameraSource(Config(camera_index=0))
    assert src.index == 0


def test_close_joins_the_reader_thread(monkeypatch):
    """close() must not return while leaving a reader thread alive against a
    released capture — a later start() would then run two readers."""
    import threading

    class FakeCap:
        def __init__(self):
            self.released = False

        def isOpened(self):
            return True

        def read(self):
            # Block until close() signals stop, simulating a wedged Continuity read.
            stop_event.wait(timeout=5.0)
            return False, None

        def release(self):
            self.released = True

    stop_event = threading.Event()
    src = camera.CameraSource(Config())
    fake_cap = FakeCap()
    src._cap = fake_cap
    src._stop = stop_event
    src._thread = threading.Thread(target=src._run, daemon=True)
    src._thread.start()

    src.close()

    assert src._thread is None
    assert fake_cap.released
