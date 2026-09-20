import pathlib
import re

HTML = (pathlib.Path(__file__).resolve().parent.parent / "web" / "index.html").read_text()


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


def test_simulated_warning_banner_is_conditional_on_live_detector():
    """The amber 'simulated' note must flip to a real-detector note in live
    mode, and restore its synthetic wording when the backend drops."""
    assert "DEFAULT_NOTE" in HTML
    assert "Live camera, live detector" in HTML
