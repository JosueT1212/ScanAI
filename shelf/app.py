"""Local API. Binds to 127.0.0.1; there is no auth and none is wanted."""
import asyncio
import json
import time
from pathlib import Path

import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from .fuse import angular_delta, wrap

WEB_DIR = Path(__file__).resolve().parent.parent / "web"


def create_app(pipeline, store) -> FastAPI:
    app = FastAPI(title="ScanAI")

    @app.get("/api/items")
    def items():
        return store.items()

    @app.get("/api/locations")
    def locations():
        return store.locations()

    @app.get("/api/stations")
    def stations():
        return [s.__dict__ for s in store.stations()]

    @app.post("/api/stations/{station_id}/activate")
    def activate(station_id: str):
        pipeline.set_station(station_id)
        return {"station": pipeline.station.__dict__}

    @app.post("/api/calibrate/yaw")
    def calibrate_yaw():
        """Align a detection dead ahead with the lidar return in that direction.

        Place one distinctive object directly in front of the rig, then call this.
        The offset is the phone's rotation relative to the lidar's zero — the
        constant that makes fusion possible at all.
        """
        snap = pipeline.snapshot()
        if not snap["detections"] or not snap["scan"]:
            return {"ok": False, "reason": "need one detection and a live scan"}
        d = max(snap["detections"], key=lambda x: x["conf"])
        cx = (d["x1"] + d["x2"]) / 2
        frame = pipeline.camera.latest()
        w = frame.shape[1] if frame is not None else 1.0
        theta_cam = (cx / w - 0.5) * pipeline.cfg.hfov

        # Only returns near the lidar's front can be the object the operator put
        # dead ahead. Searching the whole scan lets anything closer anywhere in
        # the room win, and a wrong offset here is silent and permanent.
        AHEAD = 20.0
        ahead = [p for p in snap["scan"] if abs(angular_delta(p[0], 0.0)) <= AHEAD]
        if not ahead:
            return {"ok": False,
                    "reason": f"no lidar return within {AHEAD:.0f} deg of the front; "
                              "is the object actually in front of the rig?"}
        nearest = min(ahead, key=lambda p: p[1])
        offset = wrap(nearest[0] - pipeline.station.heading - theta_cam)
        return {"ok": True, "yaw_offset": round(offset, 2),
                "target_bearing": round(nearest[0], 2),
                "target_range_m": round(nearest[1], 3),
                "note": "write this into config.local.toml"}

    def frames():
        while True:
            f = pipeline.camera.latest()
            if f is None:
                time.sleep(0.05)          # camera gone; do not spin a core
                continue
            ok, buf = cv2.imencode(".jpg", f)
            if ok:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                       + buf.tobytes() + b"\r\n")
            time.sleep(0.03)              # ~30 fps ceiling

    @app.get("/api/camera.mjpg")
    def camera_mjpg():
        return StreamingResponse(
            frames(), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.websocket("/ws/live")
    async def live(ws: WebSocket):
        await ws.accept()
        try:
            while True:
                await ws.send_text(json.dumps(pipeline.snapshot()))
                await asyncio.sleep(0.1)          # ~10 Hz, the lidar's own rate
        except WebSocketDisconnect:
            pass
        except Exception:                          # noqa: BLE001
            # Transports differ on write-after-close; anything here means the
            # client is gone. Leaking a task per stale client is worse.
            pass

    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
    return app
