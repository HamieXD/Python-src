from types import SimpleNamespace
import inspect
import detector

import pytest

from detector import (
    DrowsinessMonitor,
    eye_aspect_ratio,
    eye_landmarks,
    mouth_aspect_ratio,
    mouth_landmarks,
)
from config import Settings


def test_eye_aspect_ratio_is_higher_for_open_eye():
    open_eye = [(0, 0), (2, 2), (4, 2), (6, 0), (4, -2), (2, -2)]
    closed_eye = [(0, 0), (2, 0.5), (4, 0.5), (6, 0), (4, -0.5), (2, -0.5)]

    assert eye_aspect_ratio(open_eye) == pytest.approx(2 / 3)
    assert eye_aspect_ratio(closed_eye) == pytest.approx(1 / 6)


def test_eye_aspect_ratio_returns_zero_when_eye_width_is_zero():
    eye = [(0, 0), (1, 2), (2, 2), (0, 0), (2, -2), (1, -2)]

    assert eye_aspect_ratio(eye) == 0.0


def test_eye_aspect_ratio_uses_full_euclidean_distance():
    eye = [
        (0, 0, 0),
        (2, 2, 1),
        (4, 2, 1),
        (6, 0, 0),
        (4, -2, -1),
        (2, -2, -1),
    ]

    assert eye_aspect_ratio(eye) == pytest.approx((2 * (20**0.5)) / 12)


def test_calibrator_uses_median_ignores_missing_and_keeps_nose_baseline():
    calibrator_type = getattr(detector, "Calibrator", None)
    assert calibrator_type is not None

    calibrator = calibrator_type(Settings(calibration_seconds=1.0))
    calibrator.start(10.0)
    ears = [0.30 + index / 100 for index in range(10)]
    for index, ear in enumerate(ears):
        calibrator.update(ear, 10.0 + index / 10, nose_y=0.52)
    calibrator.update(None, 11.0, nose_y=None)

    assert calibrator.complete is True
    assert calibrator.succeeded is True
    assert calibrator.threshold == pytest.approx(0.75 * 0.345)
    assert calibrator.nose_baseline == pytest.approx(0.52)


def test_calibration_with_too_few_samples_keeps_existing_threshold():
    calibrator_type = getattr(detector, "Calibrator", None)
    assert calibrator_type is not None

    calibrator = calibrator_type(
        Settings(ear_threshold=0.27, calibration_seconds=1.0)
    )
    calibrator.start(0.0)
    calibrator.update(0.4, 0.1, nose_y=0.5)
    calibrator.update(None, 1.0)

    assert calibrator.complete is True
    assert calibrator.succeeded is False
    assert calibrator.threshold == 0.27


def test_monitor_uses_average_of_recent_ear_values():
    monitor = DrowsinessMonitor(Settings(ear_threshold=0.15))
    monitor.update(0.1, 0.0)

    status = monitor.update(0.3, 0.1)

    assert status.ear == pytest.approx(0.2)
    assert status.state == "ok"


def test_yawn_requires_one_second_counts_once_per_open_close_without_alarm():
    monitor = DrowsinessMonitor(Settings(yawn_mar_threshold=0.6))
    assert "mar" in inspect.signature(monitor.update).parameters

    assert monitor.update(0.3, 0.0, mar=0.7).state == "ok"
    before_yawn = monitor.update(0.3, 0.99, mar=0.7)
    assert before_yawn.state == "ok"
    assert before_yawn.yawn_count == 0

    yawning = monitor.update(0.3, 1.0, mar=0.7)
    held_open = monitor.update(0.3, 1.2, mar=0.8)
    assert yawning.state == "yawning"
    assert yawning.alarm_on is False
    assert yawning.yawn_count == 1
    assert held_open.yawn_count == 1

    closed = monitor.update(0.3, 1.3, mar=0.2)
    monitor.update(0.3, 1.4, mar=0.7)
    second_yawn = monitor.update(0.3, 2.4, mar=0.7)
    assert closed.state == "ok"
    assert second_yawn.yawn_count == 2


def test_sustained_head_nod_triggers_drowsy_until_head_returns():
    monitor = DrowsinessMonitor(Settings(head_nod_drop_ratio=0.15))
    assert "nose_y" in inspect.signature(monitor.update).parameters
    assert hasattr(monitor, "set_nose_baseline")
    monitor.set_nose_baseline(0.5)

    assert monitor.update(0.3, 0.0, nose_y=0.66).state == "ok"
    before_alarm = monitor.update(0.3, 0.99, nose_y=0.66)
    assert before_alarm.state == "ok"

    nod = monitor.update(0.3, 1.0, nose_y=0.66)
    upright = monitor.update(0.3, 1.1, nose_y=0.5)
    assert nod.state == "drowsy"
    assert nod.alarm_on is True
    assert upright.state == "ok"
    assert upright.alarm_on is False


def test_eye_landmarks_extracts_dlib_style_eye_points():
    landmarks = SimpleNamespace(
        part=lambda index: SimpleNamespace(x=index, y=-index)
    )

    left_eye, right_eye = eye_landmarks(landmarks)

    assert left_eye == [(index, -index) for index in range(36, 42)]
    assert right_eye == [(index, -index) for index in range(42, 48)]


def test_mouth_aspect_ratio_and_landmarks():
    open_mouth = [
        (0, 0),
        (1, 1),
        (2, 1),
        (3, 1),
        (6, 0),
        (3, -1),
        (2, -1),
        (1, -1),
    ]
    landmarks = SimpleNamespace(
        part=lambda index: SimpleNamespace(x=index, y=index + 1)
    )

    assert mouth_aspect_ratio(open_mouth) == pytest.approx(0.5)
    assert mouth_landmarks(landmarks) == [
        (index, index + 1) for index in range(60, 68)
    ]


def test_short_blink_does_not_trigger_alarm():
    monitor = DrowsinessMonitor(Settings())

    assert monitor.update(0.1, 0.0).state == "eyes_closing"
    monitor.update(0.3, 0.5)
    monitor.update(0.3, 0.6)
    reopened = monitor.update(0.3, 0.7)

    assert reopened.state == "ok"
    assert reopened.alarm_on is False


def test_closed_eyes_trigger_alarm_at_threshold():
    monitor = DrowsinessMonitor(Settings())

    monitor.update(0.1, 10.0)
    status = monitor.update(0.1, 11.5)

    assert status.state == "drowsy"
    assert status.alarm_on is True
    assert status.ear == 0.1


def test_open_eyes_stop_alarm_and_reset_closed_timer():
    monitor = DrowsinessMonitor(Settings())
    monitor.update(0.1, 0.0)
    monitor.update(0.1, 1.5)

    monitor.update(0.3, 1.6)
    monitor.update(0.3, 1.7)
    monitor.update(0.3, 1.8)
    status = monitor.update(0.3, 1.9)

    assert status.state == "ok"
    assert status.alarm_on is False
    assert status.ear == pytest.approx(0.26)


def test_face_loss_warns_then_alarms_after_additional_delay():
    monitor = DrowsinessMonitor(Settings())

    assert monitor.update(None, 0.0).state == "ok"
    assert monitor.update(None, 1.99).state == "ok"

    warning = monitor.update(None, 2.0)
    alarm = monitor.update(None, 4.0)

    assert warning.state == "face_lost_warning"
    assert warning.alarm_on is False
    assert warning.ear is None
    assert alarm.state == "face_lost_alarm"
    assert alarm.alarm_on is True


def test_face_return_resets_face_loss_and_eye_timers():
    monitor = DrowsinessMonitor(Settings())
    monitor.update(None, 0.0)
    monitor.update(None, 4.0)

    returned = monitor.update(0.1, 4.1)
    still_closing = monitor.update(0.1, 5.59)

    assert returned.state == "eyes_closing"
    assert returned.alarm_on is False
    assert still_closing.state == "eyes_closing"
    assert still_closing.alarm_on is False


def test_short_face_dropout_keeps_drowsy_alarm_on():
    monitor = DrowsinessMonitor(Settings())
    monitor.update(0.1, 0.0)
    drowsy = monitor.update(0.1, 1.5)

    missing = monitor.update(None, 1.6)

    assert drowsy.state == "drowsy"
    assert missing.state == "drowsy"
    assert missing.alarm_on is True


def test_short_face_dropout_keeps_closed_eye_timer_running():
    monitor = DrowsinessMonitor(Settings())
    monitor.update(0.1, 0.0)
    monitor.update(None, 0.8)

    status = monitor.update(0.1, 1.5)

    assert status.state == "drowsy"
    assert status.alarm_on is True


def test_calibration_rejects_low_open_eye_ear_and_clamps_threshold():
    from detector import Calibrator

    settings = Settings(ear_threshold=0.27, calibration_seconds=1.0)
    low = Calibrator(settings)
    low.start(0.0)
    for index in range(10):
        low.update(0.17, index / 10)
    low.update(None, 1.0)
    assert low.succeeded is False
    assert low.threshold == 0.27

    high = Calibrator(Settings(calibration_seconds=1.0))
    high.start(0.0)
    for index in range(10):
        high.update(0.60, index / 10)
    high.update(None, 1.0)
    assert high.threshold == 0.35


def test_face_loss_longer_than_warning_clears_ear_smoothing():
    monitor = DrowsinessMonitor(Settings(face_lost_warning_seconds=1.0))
    monitor.update(0.1, 0.0)
    monitor.update(None, 0.1)
    returned = monitor.update(0.3, 1.2)
    assert returned.ear == pytest.approx(0.3)
