import math
import statistics
from collections import deque
from dataclasses import dataclass
from typing import Literal

from config import Settings


def eye_aspect_ratio(eye):
    vertical_a = math.dist(eye[1], eye[5])
    vertical_b = math.dist(eye[2], eye[4])
    horizontal = math.dist(eye[0], eye[3])
    if horizontal == 0:
        return 0.0
    return (vertical_a + vertical_b) / (2.0 * horizontal)


def landmark_points(landmarks, indices):
    points = []
    for index in indices:
        point = landmarks.part(index)
        points.append((point.x, point.y))
    return points


def eye_landmarks(landmarks):
    left_eye = landmark_points(landmarks, range(36, 42))
    right_eye = landmark_points(landmarks, range(42, 48))
    return left_eye, right_eye


def mouth_aspect_ratio(mouth):
    vertical_a = math.dist(mouth[1], mouth[7])
    vertical_b = math.dist(mouth[2], mouth[6])
    vertical_c = math.dist(mouth[3], mouth[5])
    horizontal = math.dist(mouth[0], mouth[4])
    if horizontal == 0:
        return 0.0
    return (vertical_a + vertical_b + vertical_c) / (2.0 * horizontal)


def mouth_landmarks(landmarks):
    return landmark_points(landmarks, range(60, 68))


class Calibrator:
    minimum_samples = 10

    def __init__(self, settings=None):
        self.settings = settings or Settings()
        self.threshold = self.settings.ear_threshold
        self.nose_baseline = None
        self.complete = False
        self.succeeded = False
        self._started_at = None
        self._ear_samples = []
        self._nose_samples = []

    def start(self, now):
        self._started_at = now
        self._ear_samples.clear()
        self._nose_samples.clear()
        self.complete = False
        self.succeeded = False
        self.threshold = self.settings.ear_threshold
        self.nose_baseline = None

    def update(self, ear_or_none, now, nose_y=None):
        if self._started_at is None:
            self.start(now)
        if self.complete:
            return True

        if ear_or_none is not None:
            ear = float(ear_or_none)
            if math.isfinite(ear):
                self._ear_samples.append(ear)
                if nose_y is not None and math.isfinite(float(nose_y)):
                    self._nose_samples.append(float(nose_y))

        if now - self._started_at >= self.settings.calibration_seconds:
            self.complete = True
            if len(self._ear_samples) >= self.minimum_samples:
                median_ear = statistics.median(self._ear_samples)
                if median_ear >= 0.18:
                    self.threshold = min(0.35, max(0.15, 0.75 * median_ear))
                    if self._nose_samples:
                        self.nose_baseline = statistics.median(self._nose_samples)
                    self.succeeded = True
        return self.complete


@dataclass(frozen=True)
class Status:
    state: Literal[
        "ok",
        "eyes_closing",
        "yawning",
        "face_lost_warning",
        "drowsy",
        "face_lost_alarm",
    ]
    alarm_on: bool
    ear: float | None
    yawn_count: int = 0


class DrowsinessMonitor:
    def __init__(self, settings=None):
        self.settings = settings or Settings()
        self._closed_eyes_since = None
        self._face_missing_since = None
        self.yawn_count = 0
        self._yawning_since = None
        self._yawn_counted = False
        self._nose_baseline = None
        self._head_nod_since = None
        self._last_face_status = Status(
            state="ok", alarm_on=False, ear=None, yawn_count=0
        )
        self._ear_samples = deque(maxlen=5)
        self.longest_eyes_closed_seconds = 0.0
        self.blink_count = 0
        self.last_blink_at = None

    def update(self, ear_or_none, now, *, mar=None, nose_y=None):
        if ear_or_none is None:
            self._update_yawn(None, now)
            if self._face_missing_since is None:
                self._face_missing_since = now

            missing_for = now - self._face_missing_since
            warning_after = self.settings.face_lost_warning_seconds
            alarm_after = warning_after + self.settings.face_lost_alarm_delay_seconds
            if missing_for < warning_after:
                return Status(
                    state=self._last_face_status.state,
                    alarm_on=self._last_face_status.alarm_on,
                    ear=None,
                    yawn_count=self.yawn_count,
                )

            self._closed_eyes_since = None
            self._head_nod_since = None
            if missing_for >= warning_after:
                self._ear_samples.clear()
            if missing_for >= alarm_after:
                state = "face_lost_alarm"
                alarm_on = True
            elif missing_for >= warning_after:
                state = "face_lost_warning"
                alarm_on = False
            return Status(
                state=state,
                alarm_on=alarm_on,
                ear=None,
                yawn_count=self.yawn_count,
            )

        if (
            self._face_missing_since is not None
            and now - self._face_missing_since
            >= self.settings.face_lost_warning_seconds
        ):
            self._ear_samples.clear()
        self._face_missing_since = None
        self._ear_samples.append(float(ear_or_none))
        ear = sum(self._ear_samples) / len(self._ear_samples)
        is_yawning = self._update_yawn(mar, now)
        is_nodding = self._update_head_nod(nose_y, now)
        if ear < self.settings.ear_threshold:
            if self._closed_eyes_since is None:
                self._closed_eyes_since = now
            closed_for = now - self._closed_eyes_since
            self.longest_eyes_closed_seconds = max(
                self.longest_eyes_closed_seconds, closed_for
            )
            if closed_for >= self.settings.eyes_closed_seconds:
                status = Status(state="drowsy", alarm_on=True, ear=ear)
            else:
                status = Status(state="eyes_closing", alarm_on=False, ear=ear)
        else:
            if self._closed_eyes_since is not None:
                if now - self._closed_eyes_since < self.settings.eyes_closed_seconds:
                    self.blink_count += 1
                    self.last_blink_at = now
            self._closed_eyes_since = None
            status = Status(state="ok", alarm_on=False, ear=ear)

        if is_nodding:
            status = Status(
                state="drowsy",
                alarm_on=True,
                ear=ear,
                yawn_count=self.yawn_count,
            )
        elif is_yawning and status.state != "drowsy":
            status = Status(
                state="yawning",
                alarm_on=False,
                ear=ear,
                yawn_count=self.yawn_count,
            )
        else:
            status = Status(
                state=status.state,
                alarm_on=status.alarm_on,
                ear=ear,
                yawn_count=self.yawn_count,
            )
        self._last_face_status = status
        return status

    def _update_yawn(self, mar, now):
        if mar is None or mar < self.settings.yawn_mar_threshold:
            self._yawning_since = None
            self._yawn_counted = False
            return False

        if self._yawning_since is None:
            self._yawning_since = now
        if now - self._yawning_since >= 1.0 and not self._yawn_counted:
            self.yawn_count += 1
            self._yawn_counted = True
        return now - self._yawning_since >= 1.0

    def set_nose_baseline(self, nose_y):
        self._nose_baseline = None if nose_y is None else float(nose_y)
        self._head_nod_since = None

    def _update_head_nod(self, nose_y, now):
        if nose_y is None:
            return False
        nose_y = float(nose_y)
        if self._nose_baseline is None:
            self._nose_baseline = nose_y
            return False

        drop_limit = self._nose_baseline + self.settings.head_nod_drop_ratio
        if nose_y <= drop_limit:
            self._head_nod_since = None
            return False

        if self._head_nod_since is None:
            self._head_nod_since = now
        return now - self._head_nod_since >= 1.0
