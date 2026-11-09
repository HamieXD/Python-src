from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest

from app import DrowsinessApp
from config import Settings, load_settings
from detector import Calibrator, DrowsinessMonitor
from session_log import SessionLog


class FakeCapture:
    def read(self):
        return True, np.zeros((2, 2, 3), dtype=np.uint8)


class FakeRoot:
    def __init__(self):
        self.scheduled = []

    def after(self, delay, callback):
        self.scheduled.append((delay, callback))
        return len(self.scheduled)


class FakeStatusLabel:
    def configure(self, **options):
        self.options = options


def test_frame_processing_error_is_logged_once_and_next_frame_is_scheduled(capsys):
    app = object.__new__(DrowsinessApp)
    app.is_closing = False
    app.capture = FakeCapture()
    app.camera_retry_id = None
    app._frame_error_logged = False
    app.is_paused = False
    app.root = FakeRoot()

    def fail_detection(_gray):
        raise RuntimeError("detector failed")

    app.face_detector = fail_detection

    app.process_frame()
    app.process_frame()

    output = capsys.readouterr().out
    assert output.count("Frame processing error: detector failed") == 1
    assert app.root.scheduled == [(20, app.process_frame), (20, app.process_frame)]


def test_camera_retry_only_reopens_last_working_index_and_backs_off():
    app = object.__new__(DrowsinessApp)
    app.is_closing = False
    app.camera_retry_id = 1
    app.is_paused = False
    app.camera_index = 2
    app.settings = Settings(camera_index=2)
    app.camera_retry_delay_ms = 1000
    app.root = FakeRoot()
    attempts = []
    app._open_camera = lambda indices=None: attempts.append(indices) or False
    app._set_visual_state = lambda *_args: None

    for attempt in range(4):
        app.camera_retry_id = 1
        app._retry_camera()

    assert attempts == [(2,), (2,), (2,), (2,)]
    assert [delay for delay, _callback in app.root.scheduled] == [
        1000,
        2000,
        4000,
        5000,
    ]


def test_startup_camera_fallback_is_shown_in_status(monkeypatch):
    app = object.__new__(DrowsinessApp)
    app.settings = Settings(camera_index=2)
    app.capture = None
    app.camera_index = None
    app.notify_label = FakeStatusLabel()
    app.fps = 0.0
    attempts = []

    class Capture:
        def __init__(self, index):
            attempts.append(index)
            self.index = index

        def isOpened(self):
            return self.index == 1

        def release(self):
            pass

    monkeypatch.setattr("app.cv2.VideoCapture", Capture)

    assert app._open_camera()
    assert attempts == [2, 0, 1]
    assert app.camera_index == 1


def test_face_detection_downscales_and_scales_rectangle_back(monkeypatch):
    app = object.__new__(DrowsinessApp)
    observed_shapes = []
    app.face_detector = lambda gray: observed_shapes.append(gray.shape) or [
        SimpleNamespace(left=lambda: 1, top=lambda: 1, right=lambda: 3, bottom=lambda: 2)
    ]

    class ScaledFace:
        def __init__(self, left, top, right, bottom):
            self.coordinates = (left, top, right, bottom)

    monkeypatch.setattr("app.dlib.rectangle", ScaledFace)
    assert hasattr(app, "_detect_faces")
    faces = app._detect_faces(np.zeros((8, 10), dtype=np.uint8))

    assert observed_shapes == [(4, 5)]
    assert faces[0].coordinates == (2, 2, 6, 4)


def test_fps_uses_a_smoothed_window_of_frame_times():
    app = object.__new__(DrowsinessApp)
    app._frame_times = deque(maxlen=10)
    app.fps = 0.0
    assert hasattr(app, "_record_fps")

    for timestamp in (1.0, 1.1, 1.2, 1.3):
        app._record_fps(timestamp)

    assert app.fps == pytest.approx(10.0)


def test_font_uses_supported_tk_weight_and_medium_family(monkeypatch):
    app = object.__new__(DrowsinessApp)
    app._fonts = {
        (False, "regular"): "Sans",
        (False, "medium"): "Sans Medium",
        (True, "regular"): "Mono",
        (True, "medium"): "Mono Medium",
    }
    calls = []

    def font_factory(**options):
        calls.append(options)
        return options

    monkeypatch.setattr("app.ctk.CTkFont", font_factory)

    result = app._font(16, weight="medium", mono=True)

    assert result["family"] == "Mono Medium"
    assert all(call["weight"] != "medium" for call in calls)


def test_app_calibration_saves_threshold_and_sets_nose_baseline(tmp_path):
    settings = Settings(calibration_seconds=1.0, ear_threshold=0.22)
    app = object.__new__(DrowsinessApp)
    app.settings = settings
    app.calibrator = Calibrator(settings)
    app.monitor = DrowsinessMonitor(settings)
    app.monitor.longest_eyes_closed_seconds = 3.4
    app.session_log = SessionLog(tmp_path / "logs")
    app.settings_path = tmp_path / "settings.json"
    app.camera_index = 0
    app.fps = 0.0
    app._frame_times = deque(maxlen=10)
    app.is_calibrating = False
    app.calibration_started_at = 0.0
    app._status_hold_until = 0.0
    statuses = []
    app._set_visual_state = lambda state, word, detail: statuses.append((word, detail))
    app._stop_alarm = lambda: None
    assert hasattr(app, "_start_calibration")
    assert hasattr(app, "_collect_calibration")

    app._start_calibration(now=0.0)
    for index in range(10):
        app._collect_calibration(0.30 + index / 100, index / 10, nose_y=0.52)
    app._collect_calibration(None, 1.0)

    assert statuses[0][0] == "Calibrating"
    assert app.monitor.longest_eyes_closed_seconds == 3.4
    assert settings.ear_threshold == pytest.approx(0.75 * 0.345)
    assert app.monitor._nose_baseline == pytest.approx(0.52)
    assert load_settings(app.settings_path).ear_threshold == pytest.approx(
        settings.ear_threshold
    )
    assert statuses[-1][0] == "Awake"


def test_yawning_status_is_orange_and_does_not_start_alarm():
    app = object.__new__(DrowsinessApp)
    alarm_calls = []
    status_calls = []
    app._stop_alarm = lambda: alarm_calls.append("stop")
    app._set_visual_state = lambda state, word, detail: status_calls.append((state, word, detail))
    app._update_stats = lambda: None

    app._apply_status(SimpleNamespace(state="yawning", alarm_on=False))

    assert alarm_calls == ["stop"]
    assert status_calls == [("yawning", "Yawning", "Yawn detected")]


def test_app_logs_alarm_only_when_alarm_starts(tmp_path):
    app = object.__new__(DrowsinessApp)
    app.session_log = SessionLog(tmp_path)
    app._last_alarm_on = False
    app._last_yawn_count = 0
    app._status_hold_until = 0.0
    app._start_alarm = lambda: None
    app._stop_alarm = lambda: None
    app._set_visual_state = lambda *_args: None
    app._update_stats = lambda: None

    alarm = SimpleNamespace(state="drowsy", alarm_on=True, yawn_count=0)
    clear = SimpleNamespace(state="ok", alarm_on=False, yawn_count=0)
    app._apply_status(alarm)
    app._apply_status(alarm)
    app._apply_status(clear)
    app._apply_status(alarm)

    assert [event for _time, event, _details in app.session_log.events] == [
        "drowsy",
        "drowsy",
    ]


