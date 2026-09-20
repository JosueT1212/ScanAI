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
                                 pts=[(float(a), mm / 1000.0) for a, mm in d["pts"] if mm > 0])

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
        rc = self.proc.wait()
        if rc != 0:
            raise RuntimeError(
                f"{binary} exited {rc} (port={self.cfg.lidar_port}). "
                "Check the port with: ls /dev/cu.usbserial*")

    def close(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
