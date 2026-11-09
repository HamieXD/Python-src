import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class Settings:
    camera_index: int = 0
    ear_threshold: float = 0.25
    eyes_closed_seconds: float = 1.5
    face_lost_warning_seconds: float = 2.0
    face_lost_alarm_delay_seconds: float = 2.0
    alarm_volume: float = 1.0
    calibration_seconds: float = 5.0
    yawn_mar_threshold: float = 0.6
    head_nod_drop_ratio: float = 0.15


def _valid_setting(name, value):
    if name == "camera_index":
        return type(value) is int and value >= 0
    if type(value) not in (int, float):
        return False
    try:
        if not math.isfinite(value):
            return False
    except OverflowError:
        return False
    if name == "alarm_volume":
        return 0 <= value <= 1
    if name == "ear_threshold":
        return value > 0
    if name in ("calibration_seconds", "yawn_mar_threshold"):
        return value > 0
    if name == "head_nod_drop_ratio":
        return 0 < value <= 1
    return value >= 0


def load_settings(path):
    defaults = Settings()
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return defaults

    if not isinstance(data, dict):
        return defaults

    values = {}
    for name in asdict(defaults):
        if name not in data:
            continue
        if _valid_setting(name, data[name]):
            values[name] = data[name]
        else:
            print(f"Invalid setting '{name}'; using the default.")
    return Settings(**values)


def save_settings(settings, path):
    contents = json.dumps(asdict(settings), indent=2)
    Path(path).write_text(contents + "\n", encoding="utf-8")
