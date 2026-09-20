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
