# ScanAI Lidar + Camera Fusion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn a real RPLIDAR C1 and an iPhone camera into positioned inventory sightings, served to the existing ScanAI page over WebSocket.

**Architecture:** One Python process. Blocking work (lidar subprocess reads, `cv2.read`, YOLO inference) runs on threads; FastAPI and the WebSocket run on asyncio. The camera supplies object identity and bearing, the lidar supplies range, and `fuse.associate` joins them — returning `None` rather than guessing when the lidar cannot corroborate a detection.

**Tech Stack:** Python 3.11 (via `uv`), FastAPI, uvicorn, `sqlite3` (stdlib), OpenCV, ultralytics YOLOv8n, pytest. Lidar data comes from the prebuilt `ultra_simple` binary's stdout.

**Spec:** `docs/superpowers/specs/2026-09-20-lidar-camera-fusion-design.md`

## Global Constraints

- **Bearing convention:** 0° is the lidar's front, increasing **clockwise**, wrapped to `[0, 360)`. This matches the C1 and the existing frontend. Every angle in the codebase obeys it.
- **Units:** lidar reports millimetres; everything above `capture.py` uses **metres**. Floor-plan coordinates are the frontend's units, where **1 unit = 0.2 m**.
- **Fusion honesty rule:** a detection with no lidar return in its angular window returns `None`. Never substitute a default or estimated range. This rule is load-bearing and must survive refactoring.
- **`SIM` tag rule:** the frontend's `SIM` prefix on bounding-box labels disappears **only** when boxes originate from the detector. Never remove it for synthetic boxes.
- **Platform:** macOS only. Continuity Camera is a macOS feature.
- **Lidar stdout format:** lines matching `theta:\s*([\d.]+)\s*Dist:\s*([\d.]+)`, verified against the working `scanner/record.py`.
- **Binary path:** `~/Documents/HackMIT/rplidar_sdk/output/Darwin/Release/ultra_simple`, invoked as `ultra_simple --channel --serial <port> 460800`.
- **Zero-distance samples** are dropped at capture, never downstream.
- **No secrets, no auth.** The service binds to `127.0.0.1` only.

---

### Task 1: Project scaffolding and dependency environment

**Files:**
- Create: `pyproject.toml`
- Create: `shelf/__init__.py`
- Create: `tests/__init__.py`
- Create: `config.toml`
- Create: `shelf/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `shelf.config.Config` dataclass with fields `hfov: float`, `yaw_offset: float`, `lidar_port: str`, `lidar_binary: str`, `camera_name: str`, `conf_floor: float`, `edge_reject: float`, `db_path: str`. Loader `shelf.config.load(path: str = "config.toml") -> Config` which overlays `config.local.toml` when present.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
import textwrap
from shelf.config import load, Config


def test_load_reads_values(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text(textwrap.dedent("""
        hfov = 68.0
        yaw_offset = 12.5
        lidar_port = "/dev/cu.usbserial-210"
        lidar_binary = "/tmp/ultra_simple"
        camera_name = "iPhone"
        conf_floor = 0.45
        edge_reject = 0.1
        db_path = "scanai.db"
    """))
    cfg = load(str(p))
    assert isinstance(cfg, Config)
    assert cfg.hfov == 68.0
    assert cfg.yaw_offset == 12.5
    assert cfg.conf_floor == 0.45


def test_local_overlay_wins(tmp_path):
    (tmp_path / "config.toml").write_text('hfov = 68.0\nyaw_offset = 0.0\n')
    (tmp_path / "config.local.toml").write_text('yaw_offset = 33.0\n')
    cfg = load(str(tmp_path / "config.toml"))
    assert cfg.hfov == 68.0
    assert cfg.yaw_offset == 33.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'shelf.config'`

- [ ] **Step 3: Write minimal implementation**

```toml
# pyproject.toml
[project]
name = "scanai"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.115",
    "uvicorn[standard]>=0.30",
    "opencv-python>=4.10",
    "ultralytics>=8.3",
    "numpy>=1.26",
    "pytest>=8.0",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

```toml
# config.toml
hfov = 68.0            # iPhone wide-lens horizontal FOV, degrees. Calibrate.
yaw_offset = 0.0       # phone rotation vs lidar zero, degrees. Calibrate.
lidar_port = "/dev/cu.usbserial-210"
lidar_binary = "~/Documents/HackMIT/rplidar_sdk/output/Darwin/Release/ultra_simple"
camera_name = "iPhone" # substring matched against macOS camera names
conf_floor = 0.45      # YOLO confidence floor
edge_reject = 0.10     # reject detections in the outer 10% of frame width
db_path = "scanai.db"
```

```python
# shelf/config.py
"""Configuration. Two of these values — hfov and yaw_offset — are physical
measurements of how the phone sits on the rig. They cannot be guessed; see
the calibration section of the design spec."""
import os
import tomllib
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    hfov: float = 68.0
    yaw_offset: float = 0.0
    lidar_port: str = "/dev/cu.usbserial-210"
    lidar_binary: str = "ultra_simple"
    camera_name: str = "iPhone"
    conf_floor: float = 0.45
    edge_reject: float = 0.10
    db_path: str = "scanai.db"


def _read(path: str) -> dict:
    try:
        with open(path, "rb") as f:
            return tomllib.load(f)
    except FileNotFoundError:
        return {}


def load(path: str = "config.toml") -> Config:
    data = _read(path)
    local = path.replace(".toml", ".local.toml")
    data.update(_read(local))
    fields = {f for f in Config.__dataclass_fields__}
    data = {k: v for k, v in data.items() if k in fields}
    if "lidar_binary" in data:
        data["lidar_binary"] = os.path.expanduser(data["lidar_binary"])
    return Config(**data)
```

Create empty `shelf/__init__.py` and `tests/__init__.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv venv --python 3.11 && uv pip install -e ".[dev]" 2>/dev/null || uv pip install -e .` then `uv run pytest tests/test_config.py -v`
Expected: PASS, 2 passed

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml config.toml shelf/__init__.py shelf/config.py tests/__init__.py tests/test_config.py
git commit -m "feat: project scaffolding and config loader"
```

---

### Task 2: Lidar capture with a replay source

**Files:**
- Create: `shelf/capture.py`
- Test: `tests/test_capture.py`

**Interfaces:**
- Consumes: `shelf.config.Config`.
- Produces:
  - `Revolution` dataclass: `t: float`, `pts: list[tuple[float, float]]` where each tuple is `(bearing_deg, range_m)`.
  - `parse_line(line: str) -> tuple[float, float] | None` returning `(deg, metres)`.
  - `RevolutionAssembler.feed(deg, m) -> Revolution | None` — emits a Revolution when the angle wraps.
  - `ReplaySource(path: str)` and `LidarSource(cfg: Config)`, both with `revolutions() -> Iterator[Revolution]` and `close()`.

**Why a replay source exists:** every module above this one develops and tests with no hardware attached. Do not defer it.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_capture.py
import json
from shelf.capture import parse_line, RevolutionAssembler, ReplaySource


def test_parse_line_converts_mm_to_metres():
    assert parse_line("theta: 12.50 Dist: 1234.00") == (12.5, 1.234)


def test_parse_line_ignores_noise():
    assert parse_line("RPLIDAR S/N: ABCDEF") is None


def test_parse_line_drops_zero_distance():
    assert parse_line("theta: 90.00 Dist: 0.00") is None


def test_assembler_emits_on_angle_wrap():
    a = RevolutionAssembler()
    assert a.feed(10.0, 1.0) is None
    assert a.feed(200.0, 1.0) is None
    rev = a.feed(5.0, 1.0)          # wrapped past 360 -> revolution complete
    assert rev is not None
    assert [p[0] for p in rev.pts] == [10.0, 200.0]
    assert a.feed(30.0, 1.0) is None   # new revolution is accumulating


def test_replay_source_yields_recorded_revolutions(tmp_path):
    p = tmp_path / "scan.jsonl"
    p.write_text(
        json.dumps({"t": 0.0, "pts": [[0.0, 1000], [180.0, 2000]]}) + "\n"
        + json.dumps({"t": 0.1, "pts": [[90.0, 1500]]}) + "\n"
    )
    revs = list(ReplaySource(str(p), realtime=False).revolutions())
    assert len(revs) == 2
    assert revs[0].pts == [(0.0, 1.0), (180.0, 2.0)]
    assert revs[1].pts == [(90.0, 1.5)]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_capture.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'shelf.capture'`

- [ ] **Step 3: Write minimal implementation**

```python
# shelf/capture.py
"""Lidar capture.

Parses the prebuilt ultra_simple binary's stdout rather than reimplementing
the RPLIDAR binary protocol. This is already proven in scanner/record.py, and
the interesting problem in this project is not serial framing.

ReplaySource mirrors LidarSource exactly so that fusion, storage and the API
all develop from a recorded file with no hardware attached.
"""
import json
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from typing import Iterator

LINE = re.compile(r"theta:\s*([\d.]+)\s*Dist:\s*([\d.]+)")


@dataclass
class Revolution:
    t: float
    pts: list[tuple[float, float]] = field(default_factory=list)   # (deg, metres)


def parse_line(line: str) -> tuple[float, float] | None:
    """One stdout line -> (bearing_deg, range_m). None for noise or no-return."""
    m = LINE.search(line)
    if not m:
        return None
    mm = float(m.group(2))
    if mm <= 0:                      # no-return sample: drop here, never downstream
        return None
    return float(m.group(1)), mm / 1000.0


class RevolutionAssembler:
    """Accumulates points until the reported angle wraps past 360."""

    def __init__(self) -> None:
        self._pts: list[tuple[float, float]] = []
        self._last = 0.0
        self._t0 = time.time()

    def feed(self, deg: float, m: float) -> Revolution | None:
        done = None
        if deg < self._last and self._pts:
            done = Revolution(t=round(time.time() - self._t0, 3), pts=self._pts)
            self._pts = []
        self._last = deg
        self._pts.append((deg, m))
        return done


class ReplaySource:
    """Recorded revolutions from JSONL, optionally paced like the original."""

    def __init__(self, path: str, realtime: bool = True) -> None:
        self.path = path
        self.realtime = realtime

    def revolutions(self) -> Iterator[Revolution]:
        prev_t = None
        with open(self.path) as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                if self.realtime and prev_t is not None:
                    time.sleep(max(0.0, d["t"] - prev_t))
                prev_t = d["t"]
                yield Revolution(t=d["t"],
                                 pts=[(float(a), mm / 1000.0) for a, mm in d["pts"]])

    def close(self) -> None:
        pass


class LidarSource:
    """Live C1 via the ultra_simple binary."""

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.proc: subprocess.Popen | None = None

    def revolutions(self) -> Iterator[Revolution]:
        binary = os.path.expanduser(self.cfg.lidar_binary)
        self.proc = subprocess.Popen(
            [binary, "--channel", "--serial", self.cfg.lidar_port, "460800"],
            stdout=subprocess.PIPE, text=True, bufsize=1)
        asm = RevolutionAssembler()
        for line in self.proc.stdout:
            got = parse_line(line)
            if got is None:
                continue
            rev = asm.feed(*got)
            if rev is not None:
                yield rev

    def close(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_capture.py -v`
Expected: PASS, 5 passed

- [ ] **Step 5: Commit**

```bash
git add shelf/capture.py tests/test_capture.py
git commit -m "feat: lidar capture with matching replay source"
```

---

### Task 3: Fusion — the core module

**Files:**
- Create: `shelf/fuse.py`
- Test: `tests/test_fuse.py`

**Interfaces:**
- Consumes: `shelf.capture.Revolution`, `shelf.config.Config`.
- Produces:
  - `Detection` dataclass: `cls_name: str`, `conf: float`, `x1: float`, `y1: float`, `x2: float`, `y2: float`.
  - `Station` dataclass: `id: str`, `name: str`, `x: float`, `y: float`, `heading: float` (floor-plan units, degrees).
  - `Sighting` dataclass: `cls_name: str`, `conf: float`, `bearing: float`, `range_m: float`, `x: float`, `y: float`.
  - `wrap(deg: float) -> float` → `[0, 360)`.
  - `angular_delta(a: float, b: float) -> float` → signed difference in `(-180, 180]`.
  - `returns_within(rev, bearing, half_width) -> list[float]`.
  - `robust_min(ranges: list[float]) -> float`.
  - `associate(det, frame_w, rev, station, cfg) -> Sighting | None`.

**The rule this task exists to enforce:** `associate` returns `None` when no lidar return falls inside the detection's angular window. An object recognised in mid-air is an association failure, not a sighting with a guessed distance.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_fuse.py
import math
import pytest
from shelf.capture import Revolution
from shelf.config import Config
from shelf.fuse import (Detection, Station, wrap, angular_delta,
                        returns_within, robust_min, associate)

CFG = Config(hfov=68.0, yaw_offset=0.0, edge_reject=0.0)
STATION = Station(id="s1", name="Aisle", x=50.0, y=34.0, heading=0.0)
FRAME_W = 1000.0


def ring(range_m=3.0, step=0.72):
    """A revolution that returns the same range in every direction."""
    n = int(360 / step)
    return Revolution(t=0.0, pts=[(i * step, range_m) for i in range(n)])


def box(center_x, width=100.0):
    return Detection(cls_name="cup", conf=0.9,
                     x1=center_x - width / 2, y1=100.0,
                     x2=center_x + width / 2, y2=200.0)


def test_wrap_normalises():
    assert wrap(370.0) == pytest.approx(10.0)
    assert wrap(-10.0) == pytest.approx(350.0)


def test_angular_delta_crosses_zero():
    assert angular_delta(10.0, 350.0) == pytest.approx(20.0)
    assert angular_delta(350.0, 10.0) == pytest.approx(-20.0)


def test_centred_box_maps_to_station_heading():
    s = associate(box(FRAME_W / 2), FRAME_W, ring(3.0), STATION, CFG)
    assert s is not None
    assert s.bearing == pytest.approx(0.0, abs=0.5)
    assert s.range_m == pytest.approx(3.0, abs=0.01)


def test_box_right_of_centre_maps_clockwise():
    # A quarter of the way right of centre is a quarter of the half-FOV.
    s = associate(box(FRAME_W * 0.75), FRAME_W, ring(3.0), STATION, CFG)
    assert s.bearing == pytest.approx(17.0, abs=0.5)     # 0.25 * 68


def test_yaw_offset_and_heading_both_apply():
    cfg = Config(hfov=68.0, yaw_offset=10.0, edge_reject=0.0)
    st = Station(id="s", name="n", x=0.0, y=0.0, heading=90.0)
    s = associate(box(FRAME_W / 2), FRAME_W, ring(2.0), st, cfg)
    assert s.bearing == pytest.approx(100.0, abs=0.5)


def test_no_lidar_return_in_window_returns_none():
    """The honesty rule: recognised in mid-air is not a sighting."""
    empty = Revolution(t=0.0, pts=[(180.0, 2.0)])       # nothing near bearing 0
    assert associate(box(FRAME_W / 2), FRAME_W, empty, STATION, CFG) is None


def test_robust_min_ignores_a_single_spurious_near_return():
    ranges = [0.2] + [3.0] * 20
    assert robust_min(ranges) == pytest.approx(3.0, abs=0.05)


def test_robust_min_falls_back_to_median_when_sparse():
    assert robust_min([2.0, 3.0, 4.0]) == pytest.approx(3.0)


def test_returns_within_wraps_around_zero():
    rev = Revolution(t=0.0, pts=[(359.0, 1.0), (1.0, 1.5), (180.0, 9.0)])
    got = returns_within(rev, bearing=0.0, half_width=2.0)
    assert sorted(got) == [1.0, 1.5]


def test_position_is_station_plus_polar_offset():
    st = Station(id="s", name="n", x=10.0, y=20.0, heading=0.0)
    s = associate(box(FRAME_W / 2), FRAME_W, ring(2.0), st, CFG)
    # bearing 0 is "up" on the floor plan, so y decreases; 2 m = 10 units
    assert s.x == pytest.approx(10.0, abs=0.1)
    assert s.y == pytest.approx(10.0, abs=0.1)


def test_edge_detections_are_rejected():
    cfg = Config(hfov=68.0, yaw_offset=0.0, edge_reject=0.10)
    assert associate(box(FRAME_W * 0.02, width=20), FRAME_W,
                     ring(3.0), STATION, cfg) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_fuse.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'shelf.fuse'`

- [ ] **Step 3: Write minimal implementation**

```python
# shelf/fuse.py
"""Camera + lidar association.

The camera says what an object is and which direction it lies. The lidar says
how far. Neither sensor can do the other's job; joining them is the point of
this project and it is about fifteen lines of arithmetic.

Bearing 0 is the lidar's front, increasing clockwise, wrapped to [0, 360).
Floor-plan units are the frontend's: 1 unit = 0.2 m.
"""
import math
from dataclasses import dataclass

M_PER_UNIT = 0.2


@dataclass
class Detection:
    cls_name: str
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass
class Station:
    id: str
    name: str
    x: float
    y: float
    heading: float


@dataclass
class Sighting:
    cls_name: str
    conf: float
    bearing: float
    range_m: float
    x: float
    y: float


def wrap(deg: float) -> float:
    return deg % 360.0


def angular_delta(a: float, b: float) -> float:
    """Signed a - b, in (-180, 180]."""
    d = (a - b) % 360.0
    return d - 360.0 if d > 180.0 else d


def returns_within(rev, bearing: float, half_width: float) -> list[float]:
    return [m for a, m in rev.pts if abs(angular_delta(a, bearing)) <= half_width]


def robust_min(ranges: list[float]) -> float:
    """10th percentile, so one spurious near return cannot drag the range in.
    Below five samples there is no percentile worth trusting: use the median."""
    if not ranges:
        raise ValueError("robust_min of an empty window")
    s = sorted(ranges)
    if len(s) < 5:
        return s[len(s) // 2]
    return s[max(0, int(len(s) * 0.10) - 1)] if len(s) * 0.10 >= 1 else s[0]


def associate(det: Detection, frame_w: float, rev, station: Station, cfg):
    """One detection + one revolution -> a positioned sighting, or None.

    None means the lidar could not corroborate the detection. Never substitute
    a default range here; that is the difference between a measurement and a
    guess wearing a measurement's clothes.
    """
    cx = (det.x1 + det.x2) / 2.0
    if cfg.edge_reject > 0:
        margin = frame_w * cfg.edge_reject
        if cx < margin or cx > frame_w - margin:
            return None                      # lens distortion is worst at the edges

    theta_cam = (cx / frame_w - 0.5) * cfg.hfov
    bearing = wrap(station.heading + cfg.yaw_offset + theta_cam)

    box_w_frac = abs(det.x2 - det.x1) / frame_w
    half_width = max(0.75, box_w_frac * cfg.hfov / 2.0)

    window = returns_within(rev, bearing, half_width)
    if not window:
        return None

    rng = robust_min(window)
    units = rng / M_PER_UNIT
    rad = math.radians(bearing)
    return Sighting(cls_name=det.cls_name, conf=det.conf,
                    bearing=bearing, range_m=rng,
                    x=station.x + math.sin(rad) * units,
                    y=station.y - math.cos(rad) * units)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_fuse.py -v`
Expected: PASS, 11 passed

- [ ] **Step 5: Commit**

```bash
git add shelf/fuse.py tests/test_fuse.py
git commit -m "feat: camera-lidar association with the no-guess rule"
```

---

### Task 4: Storage

**Files:**
- Create: `shelf/store.py`
- Test: `tests/test_store.py`

**Interfaces:**
- Consumes: `shelf.fuse.Sighting`, `shelf.fuse.Station`.
- Produces: `Store(db_path: str)` with `init()`, `items() -> list[dict]`, `locations() -> list[dict]`, `stations() -> list[Station]`, `upsert_station(st)`, `add_location(id, name, zone, x, y)`, `record_sighting(s: Sighting) -> str` returning the item id, `seed_coco_defaults()`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_store.py
from shelf.fuse import Sighting, Station
from shelf.store import Store


def make(tmp_path):
    s = Store(str(tmp_path / "t.db"))
    s.init()
    return s


def test_sighting_creates_an_item_when_the_class_is_new(tmp_path):
    s = make(tmp_path)
    iid = s.record_sighting(Sighting("cup", 0.9, 10.0, 2.0, 12.0, 30.0))
    items = s.items()
    assert len(items) == 1
    assert items[0]["id"] == iid
    assert items[0]["cls_name"] == "cup"
    assert items[0]["source"] == "fusion"


def test_second_sighting_updates_rather_than_duplicates(tmp_path):
    s = make(tmp_path)
    a = s.record_sighting(Sighting("cup", 0.9, 10.0, 2.0, 12.0, 30.0))
    b = s.record_sighting(Sighting("cup", 0.7, 20.0, 3.0, 40.0, 50.0))
    assert a == b
    assert len(s.items()) == 1
    assert s.items()[0]["confidence"] == 0.7


def test_stations_round_trip(tmp_path):
    s = make(tmp_path)
    s.upsert_station(Station("s1", "Aisle", 50.0, 34.0, 90.0))
    got = s.stations()
    assert len(got) == 1
    assert got[0].heading == 90.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_store.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'shelf.store'`

- [ ] **Step 3: Write minimal implementation**

```python
# shelf/store.py
"""sqlite3 from the stdlib. Three tables do not need an ORM."""
import json
import sqlite3
import time
from datetime import datetime, timezone

from .fuse import Sighting, Station

SCHEMA = """
CREATE TABLE IF NOT EXISTS items (
  id TEXT PRIMARY KEY, name TEXT, tags TEXT, cls_name TEXT,
  location_id TEXT, qty INTEGER DEFAULT 1, last_seen_at TEXT,
  confidence REAL, source TEXT, x REAL, y REAL);
CREATE TABLE IF NOT EXISTS locations (
  id TEXT PRIMARY KEY, name TEXT, zone TEXT, x REAL, y REAL);
CREATE TABLE IF NOT EXISTS stations (
  id TEXT PRIMARY KEY, name TEXT, x REAL, y REAL, heading REAL);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Store:
    def __init__(self, db_path: str) -> None:
        self.db = sqlite3.connect(db_path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row

    def init(self) -> None:
        self.db.executescript(SCHEMA)
        self.db.commit()

    def items(self) -> list[dict]:
        return [dict(r) for r in self.db.execute("SELECT * FROM items")]

    def locations(self) -> list[dict]:
        return [dict(r) for r in self.db.execute("SELECT * FROM locations")]

    def stations(self) -> list[Station]:
        return [Station(**dict(r)) for r in self.db.execute("SELECT * FROM stations")]

    def upsert_station(self, st: Station) -> None:
        self.db.execute(
            "INSERT INTO stations (id,name,x,y,heading) VALUES (?,?,?,?,?) "
            "ON CONFLICT(id) DO UPDATE SET name=excluded.name, x=excluded.x, "
            "y=excluded.y, heading=excluded.heading",
            (st.id, st.name, st.x, st.y, st.heading))
        self.db.commit()

    def add_location(self, id: str, name: str, zone: str, x: float, y: float) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO locations (id,name,zone,x,y) VALUES (?,?,?,?,?)",
            (id, name, zone, x, y))
        self.db.commit()

    def record_sighting(self, s: Sighting) -> str:
        """Class-level identity for v1: two cups are one item. Instance identity
        is not solvable with COCO classes and is deferred in the spec."""
        row = self.db.execute(
            "SELECT id FROM items WHERE cls_name=?", (s.cls_name,)).fetchone()
        iid = row["id"] if row else f"itm_{s.cls_name}_{int(time.time()*1000)}"
        loc = self._nearest_location(s.x, s.y)
        if row:
            self.db.execute(
                "UPDATE items SET last_seen_at=?, confidence=?, source='fusion', "
                "location_id=?, x=?, y=? WHERE id=?",
                (_now(), s.conf, loc, s.x, s.y, iid))
        else:
            self.db.execute(
                "INSERT INTO items (id,name,tags,cls_name,location_id,qty,"
                "last_seen_at,confidence,source,x,y) VALUES (?,?,?,?,?,1,?,?,?,?,?)",
                (iid, s.cls_name.replace("_", " ").title(),
                 json.dumps([s.cls_name]), s.cls_name, loc, _now(),
                 s.conf, "fusion", s.x, s.y))
        self.db.commit()
        return iid

    def _nearest_location(self, x: float, y: float) -> str | None:
        best, bd = None, 1e9
        for l in self.locations():
            d = (l["x"] - x) ** 2 + (l["y"] - y) ** 2
            if d < bd:
                best, bd = l["id"], d
        return best

    def seed_coco_defaults(self) -> None:
        """Four zones matching the frontend's floor plan, so fused sightings
        land somewhere nameable on first run."""
        for lid, name, zone, x, y in [
            ("ELEC-A", "Shelf A, Electronics Bench", "Electronics", 15, 17),
            ("ELEC-B", "Bin Wall B, Electronics Bench", "Electronics", 34, 17),
            ("FAB-1", "Tool Chest 1, Fabrication", "Fabrication", 67, 17),
            ("FAB-2", "Under-bench Rack, Fabrication", "Fabrication", 86, 17),
            ("STOR-C", "Storage Wall, Row C", "Storage", 15, 52),
            ("STOR-D", "Storage Wall, Row D", "Storage", 34, 52),
            ("CLASS-1", "Supply Cabinet, Classroom", "Classroom", 67, 52),
            ("CHRG-1", "Charging Station, Classroom", "Classroom", 86, 52),
        ]:
            self.add_location(lid, name, zone, x, y)
        self.upsert_station(Station("home", "Cross aisle", 50.0, 34.0, 0.0))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_store.py -v`
Expected: PASS, 3 passed

- [ ] **Step 5: Commit**

```bash
git add shelf/store.py tests/test_store.py
git commit -m "feat: sqlite store for items, locations and stations"
```

---

### Task 5: Camera source and detector

**Files:**
- Create: `shelf/camera.py`
- Create: `shelf/detect.py`
- Test: `tests/test_detect.py`

**Interfaces:**
- Consumes: `shelf.config.Config`, `shelf.fuse.Detection`.
- Produces:
  - `CameraSource(cfg)` with `start()`, `latest() -> numpy.ndarray | None`, `available() -> bool`, `close()`.
  - `list_cameras() -> list[tuple[int, str]]`.
  - `Detector(cfg)` with `infer(frame) -> list[Detection]`.
  - `StubDetector(cfg)` with the same `infer` signature, returning one fixed centred box — used by tests and by anyone without weights.

**Degrade, never crash:** Continuity drops the device when the phone sleeps. `available()` goes False and the service stays up.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_detect.py
import numpy as np
from shelf.config import Config
from shelf.detect import StubDetector
from shelf.fuse import Detection


def test_stub_detector_returns_a_centred_box():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    dets = StubDetector(Config()).infer(frame)
    assert len(dets) == 1
    d = dets[0]
    assert isinstance(d, Detection)
    assert (d.x1 + d.x2) / 2 == 320.0
    assert d.conf >= 0.5


def test_stub_detector_scales_to_frame_width():
    frame = np.zeros((240, 1000, 3), dtype=np.uint8)
    d = StubDetector(Config()).infer(frame)[0]
    assert (d.x1 + d.x2) / 2 == 500.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_detect.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'shelf.detect'`

- [ ] **Step 3: Write minimal implementation**

```python
# shelf/camera.py
"""Continuity Camera as an ordinary capture device.

The iPhone enumerates on macOS as a normal camera, which removes the whole
HTTPS/tunnel problem a phone-browser capture page would have created.
A background thread reads continuously so a slow consumer never backs up
the capture queue.
"""
import threading

import cv2


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
        self.index = index if index is not None else self._pick()
        self._frame = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._cap = None
        self._thread = None

    def _pick(self) -> int:
        cams = list_cameras()
        return cams[-1][0] if cams else 0    # Continuity usually enumerates last

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
        if self._cap is not None:
            self._cap.release()
```

```python
# shelf/detect.py
"""Object identity.

YOLOv8n knows 80 COCO classes. "Jumper wires" is not among them; laptop, cup,
bottle, book, scissors, keyboard, mouse, backpack and chair are. v1 tracks what
the detector can actually find rather than pretending to recognise more.
"""
from .fuse import Detection


class StubDetector:
    """One centred box. Lets the whole pipeline run without weights."""

    def __init__(self, cfg) -> None:
        self.cfg = cfg

    def infer(self, frame) -> list[Detection]:
        h, w = frame.shape[:2]
        bw, bh = w * 0.15, h * 0.25
        return [Detection(cls_name="cup", conf=0.88,
                          x1=w / 2 - bw / 2, y1=h / 2 - bh / 2,
                          x2=w / 2 + bw / 2, y2=h / 2 + bh / 2)]


class Detector:
    """ultralytics YOLOv8n. Loaded lazily so importing this module is cheap."""

    def __init__(self, cfg, weights: str = "yolov8n.pt") -> None:
        self.cfg = cfg
        self.weights = weights
        self._model = None

    def _load(self):
        if self._model is None:
            from ultralytics import YOLO
            self._model = YOLO(self.weights)
        return self._model

    def infer(self, frame) -> list[Detection]:
        model = self._load()
        out = []
        for r in model.predict(frame, verbose=False, conf=self.cfg.conf_floor):
            names = r.names
            for b in r.boxes:
                x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())
                out.append(Detection(cls_name=names[int(b.cls[0])],
                                     conf=float(b.conf[0]),
                                     x1=x1, y1=y1, x2=x2, y2=y2))
        return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_detect.py -v`
Expected: PASS, 2 passed

- [ ] **Step 5: Commit**

```bash
git add shelf/camera.py shelf/detect.py tests/test_detect.py
git commit -m "feat: continuity camera source and YOLOv8n detector"
```

---

### Task 6: Pipeline — the loop that joins everything

**Files:**
- Create: `shelf/pipeline.py`
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `Pipeline(cfg, store, lidar, camera, detector)` with `start()`, `stop()`, `snapshot() -> dict` shaped as `{"scan": [[deg, m], ...], "detections": [...], "sightings": [...], "station": {...}, "camera_ok": bool}`, and `set_station(station_id)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pipeline.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'shelf.pipeline'`

- [ ] **Step 3: Write minimal implementation**

```python
# shelf/pipeline.py
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
                      "station": None, "camera_ok": False}
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_pipeline.py -v`
Expected: PASS, 2 passed

- [ ] **Step 5: Commit**

```bash
git add shelf/pipeline.py tests/test_pipeline.py
git commit -m "feat: fusion pipeline joining lidar revolutions to camera frames"
```

---

### Task 7: HTTP API, WebSocket and MJPEG

**Files:**
- Create: `shelf/app.py`
- Create: `shelf/__main__.py`
- Test: `tests/test_app.py`

**Interfaces:**
- Consumes: `Pipeline`, `Store`, `Config`.
- Produces: `create_app(pipeline, store) -> FastAPI`, and `python -m shelf` as the entry point with `--replay PATH`, `--stub`, `--port`.

**Routes:** `GET /api/items`, `GET /api/locations`, `GET /api/stations`, `POST /api/stations/{id}/activate`, `POST /api/calibrate/yaw`, `GET /api/camera.mjpg`, `WS /ws/live`, and `web/` served at `/`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_app.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_app.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'shelf.app'`

- [ ] **Step 3: Write minimal implementation**

```python
# shelf/app.py
"""Local API. Binds to 127.0.0.1; there is no auth and none is wanted."""
import asyncio
import json

import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from .fuse import wrap


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
        """Align a detection dead ahead with the nearest lidar return.

        Place one distinctive object directly in front of the rig, then call
        this. The offset it computes is the phone's rotation relative to the
        lidar's zero — the constant that makes fusion possible at all."""
        snap = pipeline.snapshot()
        if not snap["detections"] or not snap["scan"]:
            return {"ok": False, "reason": "need one detection and a live scan"}
        d = max(snap["detections"], key=lambda x: x["conf"])
        cx = (d["x1"] + d["x2"]) / 2
        frame = pipeline.camera.latest()
        w = frame.shape[1] if frame is not None else 1.0
        theta_cam = (cx / w - 0.5) * pipeline.cfg.hfov
        nearest = min(snap["scan"], key=lambda p: p[1])
        offset = wrap(nearest[0] - pipeline.station.heading - theta_cam)
        return {"ok": True, "yaw_offset": round(offset, 2),
                "note": "write this into config.local.toml"}

    def frames():
        while True:
            f = pipeline.camera.latest()
            if f is None:
                continue
            ok, buf = cv2.imencode(".jpg", f)
            if ok:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                       + buf.tobytes() + b"\r\n")

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

    app.mount("/", StaticFiles(directory="web", html=True), name="web")
    return app
```

```python
# shelf/__main__.py
"""Entry point.

    python -m shelf --stub                 # no hardware at all
    python -m shelf --replay data/room.jsonl
    python -m shelf                        # live lidar + Continuity camera
"""
import argparse

import uvicorn

from .app import create_app
from .camera import CameraSource
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
    args = ap.parse_args()

    cfg = load()
    store = Store(cfg.db_path)
    store.init()
    if not store.locations():
        store.seed_coco_defaults()

    lidar = ReplaySource(args.replay) if args.replay else LidarSource(cfg)
    camera = CameraSource(cfg)
    detector = StubDetector(cfg) if args.stub else Detector(cfg)

    pipeline = Pipeline(cfg, store, lidar, camera, detector)
    pipeline.start()
    try:
        uvicorn.run(create_app(pipeline, store), host="127.0.0.1", port=args.port)
    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_app.py -v`
Expected: PASS, 3 passed

- [ ] **Step 5: Commit**

```bash
git add shelf/app.py shelf/__main__.py tests/test_app.py
git commit -m "feat: local API, live websocket and mjpeg camera stream"
```

---

### Task 8: Frontend live source adapter

**Files:**
- Modify: `web/index.html` — the `scope` module's `points()` source, the `cam` module's background, and the boot block.

**Interfaces:**
- Consumes: `WS /ws/live` payload `{scan, detections, sightings, station, camera_ok}`, and `GET /api/camera.mjpg`.
- Produces: nothing other modules consume.

**The rule this task must not break:** the `SIM` prefix on box labels disappears **only** when boxes come from the detector. Synthetic boxes keep it forever.

- [ ] **Step 1: Write the failing check**

There is no DOM test harness in this project and adding one for a single adapter is not worth it. The check is a script that asserts the adapter's contract exists in the file and that the `SIM` rule is still conditional on a live source.

```python
# tests/test_frontend_contract.py
import pathlib
import re

HTML = pathlib.Path("web/index.html").read_text()


def test_live_source_is_wired():
    assert "ws://" in HTML or "wss://" in HTML
    assert "/ws/live" in HTML
    assert "/api/camera.mjpg" in HTML


def test_sim_tag_is_conditional_on_a_live_detector():
    """The SIM prefix may only be dropped when boxes come from the detector."""
    m = re.search(r'"SIM  "\s*:\s*""|\(\s*(\w+)\s*\?\s*""\s*:\s*"SIM  "\s*\)', HTML)
    assert m, "SIM prefix must remain conditional, never removed outright"


def test_synthetic_fallback_survives():
    assert "FALLBACK" in HTML
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_frontend_contract.py -v`
Expected: FAIL on `test_live_source_is_wired` — no WebSocket URL in the file yet.

- [ ] **Step 3: Write the adapter**

Insert this immediately before the `/* ---------- boot ---------- */` comment in `web/index.html`:

```javascript
/* ---------- live backend source ---------------------------------------
   Served locally by `python -m shelf`, the page runs on measured data: the
   scope draws real returns and the camera panel shows the real MJPEG stream
   with real detector boxes. Opened as the published artifact there is no
   backend, the socket never opens, and everything falls back to the
   synthetic ray-cast unchanged.                                        */
const backend = (() => {
  let live = null, ws = null;
  function connect(){
    const host = location.origin.startsWith("http")
      && !location.origin.includes("claude.ai") ? location.host : null;
    if (!host) return;                        // published artifact: stay synthetic
    try { ws = new WebSocket("ws://" + host + "/ws/live"); } catch (e) { return; }
    ws.onmessage = ev => {
      try { live = JSON.parse(ev.data); } catch (e) { return; }
      scope.useLive(live.scan);
      cam.useLive(live.detections, live.camera_ok);
    };
    ws.onclose = () => { live = null; scope.useLive(null); cam.useLive(null, false); };
    ws.onerror = () => { try { ws.close(); } catch (e) {} };
  }
  connect();
  return {isLive: () => live !== null};
})();
```

In the `scope` module, add to its returned object and use it in `draw`:

```javascript
  let liveScan = null;
  function useLive(scan){
    liveScan = scan && scan.length
      ? scan.map(([a, m]) => ({a, m}))
      : null;
  }
```

Change the point loop in `draw` from `for (const p of scan)` to
`for (const p of (liveScan || scan))`, and add `useLive` to the returned object.

In the `cam` module, add:

```javascript
  let liveDets = null, mjpeg = null;
  function useLive(dets, ok){
    liveDets = dets;
    if (ok && !mjpeg){
      mjpeg = new Image();
      mjpeg.src = "/api/camera.mjpg";
    }
    if (!ok) mjpeg = null;
  }
```

In `render`, draw `mjpeg` as the background when present (in place of the
`drawVideo()` branch), and when `liveDets` is non-null draw those boxes using
their pixel coordinates directly, **without** the `SIM` prefix — the prefix
stays for every synthetic box. Add `useLive` to the returned object.

- [ ] **Step 4: Run the check to verify it passes**

Run: `uv run pytest tests/test_frontend_contract.py -v`
Expected: PASS, 3 passed

Then verify by hand: `python -m shelf --stub --replay data/room.jsonl`, open
`http://127.0.0.1:8000`, confirm the scope shows recorded returns rather than
the ray-cast, and the camera panel shows the live stream.

- [ ] **Step 5: Commit**

```bash
git add web/index.html tests/test_frontend_contract.py
git commit -m "feat: frontend falls back to synthetic when no backend answers"
```

---

### Task 9: Record a room, calibrate, and document the run

**Files:**
- Create: `scripts/record.py` (adapted from `~/Documents/HackMIT/scanner/record.py`)
- Modify: `README.md`

**Interfaces:**
- Consumes: `shelf.capture.LidarSource`.
- Produces: `data/*.jsonl` recordings; documented run procedure.

- [ ] **Step 1: Write the recorder**

```python
# scripts/record.py
"""Record revolutions to JSONL so everything downstream replays offline.

    python scripts/record.py data/room.jsonl 60
"""
import json
import sys

from shelf.capture import LidarSource
from shelf.config import load

out = sys.argv[1] if len(sys.argv) > 1 else "data/scan.jsonl"
limit = int(sys.argv[2]) if len(sys.argv) > 2 else 60

src = LidarSource(load())
n = 0
try:
    with open(out, "w") as f:
        for rev in src.revolutions():
            f.write(json.dumps({"t": rev.t,
                                "pts": [[a, int(m * 1000)] for a, m in rev.pts]}) + "\n")
            n += 1
            if n % 10 == 0:
                print(f"\r{n} revolutions", end="", flush=True)
            if rev.t > limit:
                break
finally:
    src.close()
    print(f"\nwrote {n} revolutions to {out}")
```

- [ ] **Step 2: Record a real room**

Run: `uv run python scripts/record.py data/room.jsonl 60`
Expected: a JSONL file with roughly 600 revolutions. If it writes zero, the
lidar port in `config.toml` is wrong — check `ls /dev/cu.usbserial*`.

- [ ] **Step 3: Calibrate**

Mount the phone rigidly to the lidar, pointing the same way. Place one
distinctive object directly in front. Run `python -m shelf`, then:

```bash
curl -X POST http://127.0.0.1:8000/api/calibrate/yaw
```

Write the returned `yaw_offset` into `config.local.toml`. Sweep `hfov` by hand
until a box's bearing agrees with the lidar's reading of the same object.

- [ ] **Step 4: Document it**

Replace `README.md` with the real run procedure:

````markdown
# ScanAI

Ask where something is. The index answers from the last time a sensor actually
saw it, and says how stale that memory is.

A real RPLIDAR C1 supplies range. An iPhone over Continuity Camera supplies
identity. Neither sensor can do the other's job.

## Run

```bash
uv venv --python 3.11 && uv pip install -e .
python -m shelf --stub --replay data/room.jsonl   # no hardware
python -m shelf                                   # live
```

Open http://127.0.0.1:8000.

## Calibrate

The phone must be **rigidly mounted** to the lidar. Two constants are physical
measurements, not guesses:

- `yaw_offset` — `curl -X POST http://127.0.0.1:8000/api/calibrate/yaw` with a
  distinctive object dead ahead.
- `hfov` — the phone's horizontal field of view; tune until a box's bearing
  agrees with the lidar's reading of the same object.

Both go in `config.local.toml`, which is gitignored.

## What is real, and what is not

Real: the scan, the ranges, the detections, the association, the index.

Not real: instance identity. YOLOv8n reports COCO classes, so two cups on two
shelves are one item to v1. Fixing that means fine-tuning, fiducial tags, or
reading shelf labels — see the design spec.

The rig does not move during a scan. Sightings are placed relative to the
active station's known pose. That is a deliberate limit, not an oversight:
moving means SLAM.
````

- [ ] **Step 5: Commit**

```bash
git add scripts/record.py README.md
git commit -m "feat: room recorder and documented calibration procedure"
```

---

## Self-Review

**Spec coverage.** Every spec section maps to a task: capture → 2, camera → 5,
detect → 5, fuse → 3, store → 4, app → 7, calibration → 7 and 9, frontend
integration → 8, testing → 3/4/5/6/7/8, inventory reseed → 4
(`seed_coco_defaults`). The spec's `Fuse` box in the architecture diagram is
Task 3 plus Task 6's `Pipeline`, which the spec folded together.

**Placeholders.** None. Every code step carries runnable code.

**Type consistency.** `Detection`, `Station` and `Sighting` are defined once in
`shelf/fuse.py` and imported everywhere else — `detect.py`, `store.py`,
`pipeline.py` and the tests all use those exact names. `Revolution` is defined
once in `capture.py`. `associate` has one signature, used identically in Task 3's
tests and Task 6's pipeline. `Config` field names match between Task 1's loader,
Task 3's fusion and Task 5's detector.

**Gap found and closed.** The spec lists `shelf/{capture,camera,detect,fuse,store,app}.py`
but the loop that drives them has no home in that list. It became
`shelf/pipeline.py` (Task 6) rather than being buried in `app.py`, so the fusion
loop is testable without HTTP.
