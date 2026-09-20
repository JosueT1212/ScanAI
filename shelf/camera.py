"""Continuity Camera as an ordinary capture device.

The iPhone enumerates on macOS as a normal camera, which removes the whole
HTTPS/tunnel problem a phone-browser capture page would have created.
A background thread reads continuously so a slow consumer never backs up
the capture queue.
"""
import logging
import threading

import cv2

log = logging.getLogger(__name__)


def list_cameras(limit: int = 6) -> list[tuple[int, str]]:
    found = []
    for i in range(limit):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            found.append((i, f"camera {i}"))
        cap.release()
    return found


class CameraSource:
    def __init__(self, cfg, index: int | None = None) -> None:
        self.cfg = cfg
        if index is not None:
            self.index = index
        elif cfg.camera_index >= 0:
            self.index = cfg.camera_index
        else:
            self.index = self._pick()
        self._frame = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._cap = None
        self._thread = None

    def _pick(self) -> int:
        cams = list_cameras()
        if not cams:
            log.warning("no cameras enumerated; falling back to index 0. "
                        "That is likely the built-in webcam, not Continuity. "
                        "Pin camera_index in config.local.toml.")
            return 0
        index = cams[-1][0]
        log.warning("camera_index is -1 (auto); picked index %d (last enumerated). "
                    "This is only the iPhone while Continuity is active -- with the "
                    "phone asleep it silently falls back to the built-in webcam. "
                    "Run `python -m shelf --list-cameras`, confirm with "
                    "`system_profiler SPCameraDataType`, then pin camera_index in "
                    "config.local.toml.", index)
        return index    # Continuity usually enumerates last

    def start(self) -> None:
        self._cap = cv2.VideoCapture(self.index)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            if self._cap is None or not self._cap.isOpened():
                with self._lock:
                    self._frame = None       # degrade; do not raise
                self._stop.wait(1.0)
                continue
            ok, frame = self._cap.read()
            if not ok:
                with self._lock:
                    self._frame = None
                self._stop.wait(0.2)
                continue
            with self._lock:
                self._frame = frame

    def latest(self):
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def available(self) -> bool:
        return self.latest() is not None

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            # A stale Continuity device can hang inside cap.read(); bound the wait
            # rather than block shutdown forever. The thread is a daemon, so a
            # genuinely wedged read cannot keep the process alive.
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
