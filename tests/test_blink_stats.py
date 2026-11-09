from detector import DrowsinessMonitor
from config import Settings


def test_monitor_counts_short_blinks_and_tracks_last_blink_time():
    monitor = DrowsinessMonitor(Settings(eyes_closed_seconds=1.5))
    monitor.update(0.1, 10.0)
    monitor.update(0.1, 10.1)
    for now in (10.2, 10.3, 10.4, 10.5, 10.6):
        status = monitor.update(0.3, now)
    assert status.state == "ok"
    assert monitor.blink_count == 1
    assert monitor.last_blink_at == 10.5


def test_monitor_does_not_count_a_long_eye_closure_as_blink():
    monitor = DrowsinessMonitor(Settings(eyes_closed_seconds=1.0))
    monitor.update(0.1, 0.0)
    monitor.update(0.1, 0.1)
    monitor.update(0.1, 1.1)
    for now in (1.2, 1.3, 1.4, 1.5, 1.6):
        monitor.update(0.3, now)
    assert monitor.blink_count == 0
    assert monitor.last_blink_at is None
