import pathlib
import re

HTML = (pathlib.Path(__file__).resolve().parent.parent / "web" / "index.html").read_text()

# These are substring/regex checks over the static HTML source, not a DOM or
# JS behavioural harness (none exists for this project). Each one confirms a
# specific literal has not been deleted from the file; none of them execute
# the code or would catch it being made a no-op (e.g. wrapping the guard in
# `if (false)`). Verified reading the actual control flow is still required
# for the behaviour these literals are part of.


def test_live_source_is_wired():
    """The websocket URL, /ws/live and /api/camera.mjpg strings are present
    in the source — i.e. the live-adapter wiring hasn't been deleted."""
    assert "ws://" in HTML or "wss://" in HTML
    assert "/ws/live" in HTML
    assert "/api/camera.mjpg" in HTML


def test_sim_tag_is_conditional_on_a_live_detector():
    """The `"SIM  " : ""` (or equivalent) ternary literal is still present,
    i.e. the SIM prefix has not been hardcoded to always-on or always-off."""
    m = re.search(r'"SIM  "\s*:\s*""|\(\s*(\w+)\s*\?\s*""\s*:\s*"SIM  "\s*\)', HTML)
    assert m, "SIM prefix must remain conditional, never removed outright"


def test_synthetic_fallback_survives():
    """The FALLBACK array — the synthetic seed data used when no store and
    no backend are reachable — is still present in the source."""
    assert "FALLBACK" in HTML


def test_simulated_warning_banner_is_conditional_on_live_detector():
    """DEFAULT_NOTE (the captured synthetic-mode note text) and the live-mode
    note copy ("Live camera, live detector") are both still present in the
    source. This does not verify the note actually flips between them at
    runtime — that logic has no behavioural coverage."""
    assert "DEFAULT_NOTE" in HTML
    assert "Live camera, live detector" in HTML
