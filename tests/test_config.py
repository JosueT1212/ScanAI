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
