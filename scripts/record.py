"""Record revolutions to JSONL so everything downstream replays offline.

    uv run python scripts/record.py data/room.jsonl 60

Arg 2 is seconds of wall-clock time (checked against rev.t, the elapsed
time reported per revolution), not a revolution count. At the C1's 10 Hz
sweep that's roughly 600 revolutions, not exactly, since the serial link
still drops the odd line.
"""
import json
import os
import sys

from shelf.capture import LidarSource
from shelf.config import load

out = sys.argv[1] if len(sys.argv) > 1 else "data/scan.jsonl"
limit = int(sys.argv[2]) if len(sys.argv) > 2 else 60

os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

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

# ultra_simple can exit 0 on a bind failure (its stderr goes to the terminal,
# not caught here), so a bad port silently produces an empty, "successful"
# recording. Make that visible to anything checking the exit code rather than
# reading the file size.
if n == 0:
    sys.exit(1)
