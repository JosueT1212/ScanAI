"""The loop. One lidar revolution plus one camera frame becomes zero or more
positioned sightings. Runs on a thread; the API reads snapshot() from asyncio."""
import threading
from dataclasses import asdict

from .fuse import Station, associate


class Pipeline:
    def __init__(self, cfg, store, lidar, camera, detector) -> None:
        self.cfg, self.store = cfg, store
        self.lidar, self.camera, self.detector = lidar, camera, detector
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._snap = {"scan": [], "detections": [], "sightings": [],
                      "station": None, "camera_ok": camera.available()}
        sts = store.stations()
        self.station = sts[0] if sts else Station("home", "Home", 50.0, 34.0, 0.0)

    def set_station(self, station_id: str) -> None:
        for s in self.store.stations():
            if s.id == station_id:
                self.station = s
                return

    def tick(self, rev) -> None:
        frame = self.camera.latest()
        dets, sights = [], []
        if frame is not None:
            h, w = frame.shape[:2]
            for d in self.detector.infer(frame):
                dets.append(asdict(d))
                s = associate(d, float(w), rev, self.station, self.cfg)
                if s is None:
                    continue          # lidar could not corroborate it
                self.store.record_sighting(s)
                sights.append(asdict(s))
        with self._lock:
            self._snap = {
                "scan": [[round(a, 2), round(m, 3)] for a, m in rev.pts],
                "detections": dets, "sightings": sights,
                "station": asdict(self.station),
                "camera_ok": frame is not None,
            }

    def _run(self) -> None:
        for rev in self.lidar.revolutions():
            if self._stop.is_set():
                break
            self.tick(rev)

    def start(self) -> None:
        self.camera.start()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self.lidar.close()
        self.camera.close()

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self._snap)
