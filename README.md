# ScanAI

Ask where something is. The index answers from the last time a sensor actually
saw it, and says how stale that memory is.

A real RPLIDAR C1 supplies range. An iPhone over Continuity Camera supplies
identity. Neither sensor can do the other's job.

## Run

```bash
uv venv --python 3.11 && uv pip install -e .
uv run python -m shelf --stub --replay data/room.jsonl   # no hardware, needs a recording (see "Record a room" below)
uv run python -m shelf                                   # live: real lidar + Continuity camera
```

`uv venv` creates `.venv` but does not activate it, so every command below
goes through `uv run` rather than assuming an activated shell.

`data/room.jsonl` is gitignored and none ships in this repo, so the
hardware-free command only works after you've recorded one yourself — see
"Record a room" below, which itself needs the lidar attached once. There is
currently no hardware-free path that works on a completely fresh clone with
nothing recorded yet. `--stub` alone (no `--replay`) still opens the real
`LidarSource`; it only swaps out the detector, not the lidar.

Open http://127.0.0.1:8000.

## Camera pinning

`camera_index = -1` (the config default) means "last enumerated device." That
is only the iPhone while Continuity Camera is active — with the phone asleep,
the last device is the MacBook's built-in webcam, and the pipeline will fuse
webcam detections with lidar ranges without any error. `CameraSource._pick()`
logs a warning naming the index it chose whenever it has to guess, but the
recommended setup is to not make it guess:

1. `uv run python -m shelf --list-cameras` — prints what OpenCV enumerates. On
   macOS this is unhelpful by itself: `[(0, 'camera 0'), (1, 'camera 1')]`,
   since OpenCV cannot read device names on this platform. It also prints an
   `OpenCV: camera failed to properly initialize!` line per absent index it
   probes up to its limit — cosmetic, expected, not a fault.
2. `system_profiler SPCameraDataType` — lists the same devices with real
   names, in the same (AVFoundation) order. Verified on this machine:
   index 0 = "MacBook Pro Camera", index 1 = "iPad de Josue Camera" (Model ID
   iPhone14,5 — the Continuity iPhone).
3. Put the number from step 2 in `config.local.toml` (gitignored):
   ```toml
   camera_index = 1
   ```

Once picked, the first `cap.read()` on a Continuity device commonly returns
`False` and needs roughly a second to warm up. `CameraSource._run()` already
loops rather than giving up, so a blank first read or two is expected, not a
fault — don't chase it.

## Calibrate

The phone must be **rigidly mounted** to the lidar. Two constants are physical
measurements, not guesses:

- `yaw_offset` — `curl -X POST http://127.0.0.1:8000/api/calibrate/yaw` with a
  distinctive object dead ahead. The endpoint restricts itself to lidar
  returns within ±20° of the front arc and returns either
  `{ok, offset, target_bearing, target_range_m}` or `{ok: False, reason}`.
- `hfov` — the phone's horizontal field of view; tune until a box's bearing
  agrees with the lidar's reading of the same object.

Both go in `config.local.toml`, which is gitignored. This procedure has not
been run against hardware yet on this machine — see the note under "Record a
room" below.

## Record a room

```bash
uv run python scripts/record.py data/room.jsonl 60
```

Writes lidar revolutions to JSONL so `--replay` and calibration can run
offline afterward. The second argument is seconds, not a revolution count —
at the C1's 10 Hz sweep, 60s is roughly 600 revolutions. If it writes zero,
the lidar port in `config.toml` is wrong; check `ls /dev/cu.usbserial*`.
`data/*.jsonl` is gitignored, so `--replay` only works locally after you (or
someone) has recorded a file — none ships in this repo.

**Not run against hardware this session.** No RPLIDAR was attached
(`ls /dev/cu.usbserial*` found nothing) when this recorder and the
calibration procedure above were written, so `data/room.jsonl` does not exist
here and the calibration curl above has not been exercised against a live
sensor. The `ultra_simple` binary is present and built
(`~/Documents/HackMIT/rplidar_sdk/output/Darwin/Release/ultra_simple`); what's
missing is the device on the port. `uv run pytest -q` covers the code paths
with recorded/replayed and stub data, which is not the same as a hardware
run — treat this section as a documented procedure to follow, not a result
already confirmed on this rig.

## What is real, and what is not

Real: the scan, the ranges, the detections, the association, the index.

Not real: instance identity. YOLOv8n reports COCO classes, so two cups on two
shelves are one item to v1. Fixing that means fine-tuning, fiducial tags, or
reading shelf labels — see the design spec.

The rig does not move during a scan. Sightings are placed relative to the
active station's known pose. That is a deliberate limit, not an oversight:
moving means SLAM.

- Design: `docs/superpowers/specs/2026-09-20-lidar-camera-fusion-design.md`
- Frontend: `web/index.html` (also published as a standalone demo artifact)
