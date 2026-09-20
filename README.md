# ScanAI

Ask where something is. The index answers from the last time a sensor actually
saw it, and says how stale that memory is.

A real RPLIDAR C1 supplies range. An iPhone over Continuity Camera supplies
identity. Neither sensor can do the other's job — fusing them is the point.

- Design: `docs/superpowers/specs/2026-09-20-lidar-camera-fusion-design.md`
- Frontend: `web/findit.html` (also published as a standalone demo artifact)

Status: design approved, backend not yet implemented.
