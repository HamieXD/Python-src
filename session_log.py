import csv
import time
from datetime import datetime
from pathlib import Path


class SessionLog:
    """Collect one session's events and write them as a CSV file."""

    ALERT_EVENTS = {"drowsy", "face_lost_alarm"}
    EVENT_TYPES = ALERT_EVENTS | {"yawn", "calibration"}

    def __init__(self, log_dir="logs", *, started_at=None, monotonic_start=None):
        self.log_dir = Path(log_dir)
        self.started_at = started_at or datetime.now()
        self.monotonic_start = time.monotonic() if monotonic_start is None else monotonic_start
        self.events = []

    def record(self, event_type, details="", *, timestamp=None):
        if event_type not in self.EVENT_TYPES:
            raise ValueError(f"Unknown session event: {event_type}")
        self.events.append((timestamp or datetime.now(), event_type, str(details)))

    def summary(self, *, monotonic_end=None, longest_eyes_closed=0.0):
        ended = time.monotonic() if monotonic_end is None else monotonic_end
        return {
            "total_seconds": max(0.0, ended - self.monotonic_start),
            "alerts": sum(event in self.ALERT_EVENTS for _time, event, _details in self.events),
            "yawns": sum(event == "yawn" for _time, event, _details in self.events),
            "longest_eyes_closed": max(0.0, float(longest_eyes_closed)),
        }

    def write(self):
        self.log_dir.mkdir(parents=True, exist_ok=True)
        base = f"session_{self.started_at:%Y-%m-%d_%H-%M-%S}"
        suffix = 1
        while True:
            suffix_text = "" if suffix == 1 else f"_{suffix}"
            path = self.log_dir / f"{base}{suffix_text}.csv"
            try:
                output = path.open("x", newline="", encoding="utf-8")
                break
            except FileExistsError:
                suffix += 1
        with output:
            writer = csv.writer(output)
            writer.writerow(("timestamp", "event", "details"))
            for timestamp, event, details in self.events:
                writer.writerow(
                    (timestamp.strftime("%Y-%m-%d %H:%M:%S"), event, details)
                )
        return path


def format_duration(total_seconds):
    """Format elapsed seconds as H:MM:SS, truncating partial seconds."""
    seconds = max(0, int(total_seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours}:{minutes:02d}:{seconds:02d}"
