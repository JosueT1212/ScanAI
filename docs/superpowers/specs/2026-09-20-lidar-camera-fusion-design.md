# Lidar + camera fusion backend — design

Date: 2026-09-20
Status: approved, not yet implemented

## What this builds

A local service that turns a real RPLIDAR C1 and a real iPhone camera into
inventory sightings, and serves the existing ScanAI page against live data
instead of a synthetic ray-cast.

The camera says **what** an object is and **which direction** it lies. The lidar
says **how far**. Neither sensor can do the other's job. Fusing them is the entire
point of the system, and it reduces to about fifteen lines of arithmetic.

## Goals

1. Stream real 360° lidar revolutions to the browser at ~10 Hz.
2. Detect objects in real camera frames with YOLOv8n.
3. Associate each detection with a lidar range, producing a positioned sighting.
4. Persist sightings so the index answers "where is X" from measured data.
5. Reject detections the lidar cannot corroborate, rather than guessing a range.

## Non-goals (explicitly out of scope for v1)

- **SLAM / odometry.** The rig does not move during a scan. See "Stations".
- Fine-tuned detection on makerspace-specific objects. v1 uses COCO classes.
- Multi-floor, multi-room, or map stitching.
- Authentication. This binds to localhost on a trusted machine.
- Running on anything but macOS. Continuity Camera is a macOS feature.

## Decisions taken, with reasons

| Decision | Chosen | Why |
|---|---|---|
| Camera transport | **Continuity Camera** | The iPhone already enumerates as a capture device (`iPhone14,5`). No HTTPS, no cert, no app, no tunnel. `getUserMedia` on a plain-HTTP LAN address is refused by every modern browser, which rules out the obvious alternative. |
| Rig motion | **Fixed scan stations** | No odometry, no drift, and it makes lidar/camera time-sync trivial because nothing moves between a revolution and a frame. |
| Detector | **YOLOv8n, local** | `torch 2.8.0` is already installed. Runs on CPU or MPS at usable rates. |
| Lidar capture | **Parse `ultra_simple` stdout** | Already proven in `scanner/record.py`. Implementing the RPLIDAR binary protocol is hours of descriptor and checksum work that buys nothing here. |
| Process model | **One process, threads + asyncio** | `cv2.read` and YOLO inference are blocking; they run in a thread executor. FastAPI and the WebSocket are async. Two processes would need a queue and a supervisor for no gain at this size. |

## Architecture

```
RPLIDAR C1 ─USB→ ultra_simple ─stdout→ LidarSource ─┐
                                                     ├→ Fuse ─→ Store (sqlite3)
iPhone ─Continuity→ cv2.VideoCapture → Detector ────┘         │
                                                               ├→ WS   /ws/live
                                                               ├→ HTTP /api/*
                                                               └→ MJPEG /api/camera.mjpg
                                                     web/index.html served locally
```

## Modules

### `shelf/capture.py`
`LidarSource` spawns `ultra_simple --channel --serial <port> 460800`, parses
`theta: <deg> Dist: <mm>` from stdout, and assembles revolutions by detecting
angle wrap. Yields `Revolution(t: float, pts: list[tuple[float, int]])`.

`ReplaySource` yields the same type from a recorded JSONL file, honouring the
recorded inter-revolution timing. **Same interface**, so every module downstream
develops with no hardware attached. This is the single biggest time-saver in the
project and must not be bolted on later.

Zero-distance samples are dropped at capture, not downstream.

### `shelf/camera.py`
`CameraSource` wraps `cv2.VideoCapture(index)`. Enumerates devices and picks the
Continuity device by name where possible, by index otherwise. Exposes
`latest_frame()` returning the most recent decoded frame; a background thread
reads continuously so a slow consumer never backs up the capture queue.

Handles the device vanishing (phone sleeps, Continuity drops) by reporting
unavailable rather than raising, so the service stays up.

### `shelf/detect.py`
`Detector.infer(frame) -> list[Detection]` where
`Detection = {cls_name, conf, x1, y1, x2, y2}`. Wraps ultralytics YOLOv8n.
Confidence floor and class allow-list come from config. A `StubDetector` with the
same interface exists for development without weights.

### `shelf/fuse.py` — the core, and the only module with real tests

```python
def associate(det, frame_w, revolution, station, calib) -> Sighting | None:
    theta_cam   = (center_x(det) / frame_w - 0.5) * calib.hfov
    theta_world = wrap(station.heading + calib.yaw_offset + theta_cam)
    half        = angular_half_width(det, frame_w, calib.hfov)
    window      = returns_within(revolution, theta_world, half)
    if not window:
        return None                    # recognised in mid-air: association failed
    rng = robust_min(window)           # nearest surface in the box's angular window
    return Sighting(cls_name=det.cls_name, conf=det.conf,
                    bearing=theta_world, range_m=rng,
                    xy=station.xy + polar(theta_world, rng))
```

Rules that matter:
- **Return `None` rather than guess.** A detection with no lidar support is not a
  sighting. This is what makes the system honest and it is the first thing a
  reviewer should check still holds.
- `robust_min` is the **10th percentile** of the window's ranges, not a bare
  `min`, so one spurious near return does not drag the range in. With fewer than
  5 returns in the window it falls back to the median.
- All angles wrap to `[0, 360)`; bearing 0 is the lidar's front, clockwise,
  matching both the C1's convention and the frontend's.

### `shelf/store.py`
`sqlite3` from the stdlib. No ORM — there are three tables.

```
items     (id, name, tags, cls_name, location_id, qty, last_seen_at,
           confidence, source)
locations (id, name, zone, x, y)
stations  (id, name, x, y, heading)
```

Sightings update the matching item's `last_seen_at`, `confidence`, `location_id`
and set `source='fusion'`. An unmatched class creates a new item, so walking the
rig around populates the index rather than requiring it to be pre-filled.

### `shelf/app.py`
FastAPI. Serves `web/` statically.

| Route | Purpose |
|---|---|
| `GET /api/items`, `/api/locations`, `/api/stations` | index reads |
| `POST /api/stations/{id}/activate` | declare which station the rig is parked at |
| `POST /api/calibrate/yaw` | one-shot yaw alignment against a target dead ahead |
| `GET /api/camera.mjpg` | MJPEG stream for the camera panel |
| `WS /ws/live` | `{scan, detections, sightings, station}` at ~10 Hz |

## Calibration

Two constants, both measured, neither guessable. They exist because the phone is
**rigidly mounted to the lidar** — handheld makes fusion impossible, since the
relationship changes every second.

- `hfov` — the iPhone's horizontal field of view. Ships with a default, corrected
  with a slider against the live lidar overlay.
- `yaw_offset` — rotation of the phone relative to the lidar's zero. Place a
  distinctive object dead ahead, run `POST /api/calibrate/yaw`; it aligns the
  detection's bearing with the nearest lidar return.

Stored in `config.toml`. `config.local.toml` overrides it and is gitignored.

## Inventory reseed

YOLOv8n knows 80 COCO classes, and "jumper wires" is not among them. It does know
*laptop, cup, bottle, book, scissors, keyboard, mouse, cell phone, remote, clock,
chair, backpack*.

v1 seeds the index with objects the detector can actually find. These are real
things people lose, the loop is genuinely end-to-end, and nothing is faked. The
makerspace framing survives. Fine-tuning on real bins is a later, honest upgrade,
not a pretence held now.

## Frontend integration

`web/index.html` is the page already built and approved. One change: a source
adapter. It tries `ws://localhost:8000/ws/live`; on success the scope renders
measured returns and the camera panel shows the MJPEG stream with **real** boxes.
On failure it falls back to the synthetic ray-cast exactly as today, so the
published artifact keeps working untouched.

The `SIM` tag on boxes disappears only when boxes come from the detector. That
distinction is load-bearing and must survive refactoring.

## Testing

`tests/test_fuse.py` — the module that earns tests:
- synthetic revolution (a ring at known ranges) + a box at a known pixel offset →
  asserts the recovered bearing and range
- a detection whose angular window contains no returns → asserts `None`
- a spurious single near return inside the window → asserts `robust_min` ignores it
- angle wrap across 0°/360°

`tests/check_los.py` — already written; pins the frontend's demo geometry.

## Risks

1. **Camera permission.** macOS requires it for whichever terminal runs this, and
   the prompt is easy to miss. First-run failure mode looks like "no camera".
2. **Continuity drops** when the phone sleeps or wanders. `CameraSource` must
   degrade, not crash.
3. **stdout parsing throughput** is ~5,000 lines/sec. Expected fine; measure it.
   If it is not, that is when the direct serial protocol earns its cost — not before.
4. **COCO classes are coarse.** Two bottles on two shelves are indistinguishable.
   v1 tracks class-level presence, not instance identity.
5. **Detections near the edge of frame** have the worst bearing accuracy, because
   the linear pixel→angle mapping ignores lens distortion. Consider rejecting the
   outer 10% of frame width.

## Open question deferred, deliberately

Instance identity — telling *this* bottle from *that* bottle — is what would make
the index truly reliable, and it is not solvable with COCO classes. Options later:
fine-tuning, fiducial tags on bins, or text recognition on shelf labels. Not now.
