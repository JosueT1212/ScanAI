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
