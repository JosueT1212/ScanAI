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
