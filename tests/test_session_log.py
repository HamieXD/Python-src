from datetime import datetime

from session_log import SessionLog


def test_session_log_records_events_and_summary_counts_once(tmp_path):
    start = datetime(2026, 11, 4, 12, 0, 5)
    log = SessionLog(tmp_path, started_at=start, monotonic_start=10.0)
    log.record("drowsy", "eyes closed", timestamp=start)
    log.record("face_lost_alarm", "face missing", timestamp=start)
    log.record("yawn", "count=1", timestamp=start)
    path = log.write()

    assert path.name == "session_2026-11-04_12-00-05.csv"
    assert path.read_text(encoding="utf-8").splitlines() == [
        "timestamp,event,details",
        "2026-11-04 12:00:05,drowsy,eyes closed",
        "2026-11-04 12:00:05,face_lost_alarm,face missing",
        "2026-11-04 12:00:05,yawn,count=1",
    ]
    assert log.summary(monotonic_end=130.0, longest_eyes_closed=4.2) == {
        "total_seconds": 120.0,
        "alerts": 2,
        "yawns": 1,
        "longest_eyes_closed": 4.2,
    }


def test_session_log_uses_next_available_name(tmp_path):
    start = datetime(2026, 11, 4, 12, 0, 5)
    expected = [
        "session_2026-11-04_12-00-05.csv",
        "session_2026-11-04_12-00-05_2.csv",
        "session_2026-11-04_12-00-05_3.csv",
    ]
    for name in expected[:2]:
        (tmp_path / name).write_text("previous session", encoding="utf-8")

    path = SessionLog(tmp_path, started_at=start).write()

    assert path.name == expected[2]


def test_session_log_name_conflict_suffixes_do_not_overwrite(tmp_path):
    started = datetime(2026, 11, 4, 12, 0, 5)
    first = tmp_path / 'session_2026-11-04_12-00-05.csv'
    first.write_text('keep', encoding='utf-8')
    log = SessionLog(tmp_path, started_at=started)
    path = log.write()
    assert first.read_text(encoding='utf-8') == 'keep'
    assert path.name == 'session_2026-11-04_12-00-05_2.csv'


def test_format_duration_uses_hours_minutes_seconds():
    from session_log import format_duration

    assert format_duration(1453.2) == "0:24:13"
