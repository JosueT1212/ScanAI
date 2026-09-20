"""Entry point.

    python -m shelf --stub                 # no hardware at all
    python -m shelf --replay data/room.jsonl
    python -m shelf                        # live lidar + Continuity camera
    python -m shelf --camera 1             # override the enumerated camera index
    python -m shelf --list-cameras         # print enumerated cameras and exit
"""
import argparse

import uvicorn

from .app import create_app
from .camera import CameraSource, list_cameras
from .capture import LidarSource, ReplaySource
from .config import load
from .detect import Detector, StubDetector
from .pipeline import Pipeline
from .store import Store


def main() -> None:
    ap = argparse.ArgumentParser(prog="shelf")
    ap.add_argument("--replay", help="JSONL recorded by scanner/record.py")
    ap.add_argument("--stub", action="store_true", help="stub detector, no weights")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--camera", type=int, default=None,
                     help="override cfg.camera_index")
    ap.add_argument("--list-cameras", action="store_true",
                     help="print enumerated cameras and exit, no pipeline started")
    args = ap.parse_args()

    if args.list_cameras:
        for index, name in list_cameras():
            print(f"{index}: {name}")
        print("OpenCV cannot name devices on macOS. Cross-check with:\n"
              "  system_profiler SPCameraDataType\n"
              "then pin the chosen index as camera_index in config.local.toml.")
        return

    cfg = load()
    store = Store(cfg.db_path)
    store.init()
    if not store.locations():
        store.seed_coco_defaults()

    lidar = ReplaySource(args.replay) if args.replay else LidarSource(cfg)
    camera = CameraSource(cfg, index=args.camera)
    detector = StubDetector(cfg) if args.stub else Detector(cfg)

    pipeline = Pipeline(cfg, store, lidar, camera, detector)
    pipeline.start()
    try:
        uvicorn.run(create_app(pipeline, store), host="127.0.0.1", port=args.port)
    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()
