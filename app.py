import math
import time
import tkinter as tk
from collections import deque
from pathlib import Path
from tkinter import messagebox

import customtkinter as ctk
import cv2
import dlib
import numpy as np
import pygame
from PIL import Image, ImageDraw, ImageTk

from config import load_settings, save_settings
from detector import (
    Calibrator,
    DrowsinessMonitor,
    eye_aspect_ratio,
    eye_landmarks,
    mouth_aspect_ratio,
    mouth_landmarks,
)
from session_log import SessionLog, format_duration


COLORS = {
    "deep": "#0f0f10",
    "page": "#141415",
    "card": "#19191b",
    "raised": "#232325",
    "line": "#262628",
    "text": "#ededed",
    "secondary": "#a1a1a6",
    "muted": "#85858b",
    "amber": "#e8a33d",
    "ok": "#4cc38a",
    "alarm": "#e5484d",
}


def load_fonts(script_dir):
    """Load bundled IBM Plex fonts when the platform font manager supports it."""
    # Tk has no "medium" weight, so each weight is its own family name.
    # Returns {(mono, weight): family}.
    fonts_dir = Path(script_dir).resolve() / "assets" / "fonts"
    try:
        loaded = all(
            ctk.FontManager.load_font(str(fonts_dir / filename))
            for filename in (
                "IBMPlexSans-Regular.ttf",
                "IBMPlexSans-Medium.ttf",
                "IBMPlexMono-Regular.ttf",
                "IBMPlexMono-Medium.ttf",
            )
        )
    except Exception:
        loaded = False
    if loaded:
        return {
            (False, "regular"): "IBM Plex Sans",
            (False, "medium"): "IBM Plex Sans Medm",
            (True, "regular"): "IBM Plex Mono",
            (True, "medium"): "IBM Plex Mono Medium",
        }
    return {
        (False, "regular"): "Segoe UI",
        (False, "medium"): "Segoe UI Semibold",
        (True, "regular"): "Consolas",
        (True, "medium"): "Consolas",
    }


class DrowsinessApp:
    def __init__(self, root, settings, model_path, settings_path=None):
        self.root = root
        self.settings = settings
        self.settings_path = Path(settings_path or Path(__file__).with_name("settings.json"))
        self.capture = None
        self.camera_index = settings.camera_index
        self.monitor = DrowsinessMonitor(settings)
        self.calibrator = Calibrator(settings)
        self.is_calibrating = False
        self.calibration_started = False
        self.calibration_started_at = None
        self.is_paused = False
        self.current_status = None
        self._frame_times = deque(maxlen=10)
        self.fps = 0.0
        self.camera_retry_id = None
        self.camera_retry_delay_ms = 1000
        self.is_closing = False
        self._frame_error_logged = False
        self.session_log = SessionLog(Path(__file__).with_name("logs"))
        self._last_alarm_on = False
        self._last_yawn_count = 0
        self._status_hold_until = 0.0
        self._trace = deque()
        self._last_trace_draw = 0.0
        self._face_found = False
        self._last_ear = None
        self.camera_photo = None
        self._camera_image_size = None
        self._camera_mask = None
        self._camera_alarm_active = None
        self._camera_border_size = None
        self._fonts = load_fonts(Path(__file__).parent)
        self.face_detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(str(model_path))

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.bind("<Escape>", self.leave_fullscreen)
        self.root.bind("<F11>", self.toggle_fullscreen)
        self.root.bind("q", self.quit_from_key)
        self._update_header()
        self._set_visual_state("calibrating", "Calibrating", "Keep your eyes open")
        if self._open_camera():
            self._start_calibration()
            self.root.after(0, self.process_frame)
        else:
            self._set_visual_state("face_lost_alarm", "Face not found", "Camera disconnected \u00b7 retrying")
            self._schedule_camera_retry()

    def _font(self, size, weight="regular", mono=False):
        family = self._fonts[(mono, "medium" if weight == "medium" else "regular")]
        return ctk.CTkFont(family=family, size=size, weight="normal")

    def _build_ui(self):
        ctk.set_appearance_mode("dark")
        self.root.configure(fg_color=COLORS["page"])
        self.root.title("Drowsiness Detection")
        self.root.geometry("1280x800")
        self.root.minsize(1024, 680)
        self.root.grid_rowconfigure(0, minsize=56, weight=0)
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        top = ctk.CTkFrame(self.root, height=56, corner_radius=0, fg_color=COLORS["deep"])
        top.grid(row=0, column=0, sticky="ew")
        top.grid_propagate(False)
        ctk.CTkFrame(top, height=1, corner_radius=0, fg_color=COLORS["line"]).place(
            relx=0, rely=1, relwidth=1, anchor="sw"
        )
        title = ctk.CTkLabel(top, text="Drowsiness Detection", text_color=COLORS["text"], font=self._font(16, "medium"))
        title.pack(side="left", padx=(24, 16))
        ctk.CTkLabel(top, text="SMET Surawittayakarn School", text_color=COLORS["muted"], font=self._font(13)).pack(side="left")
        self.header_meta = ctk.CTkLabel(top, text="", text_color=COLORS["muted"], font=self._font(12, mono=True))
        self.header_meta.pack(side="right", padx=24)

        body = ctk.CTkFrame(self.root, fg_color="transparent")
        body.grid(row=1, column=0, sticky="nsew", padx=24, pady=(16, 24))
        body.grid_rowconfigure(0, weight=1)
        body.grid_columnconfigure(0, weight=1)
        body.grid_columnconfigure(1, minsize=296, weight=0)

        left = ctk.CTkFrame(body, fg_color="transparent")
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 16))
        left.grid_columnconfigure(0, weight=1)
        left.grid_rowconfigure(0, weight=1)
        left.grid_rowconfigure(1, minsize=216, weight=0)

        self.camera_card = ctk.CTkFrame(left, fg_color=COLORS["deep"], corner_radius=16, border_width=1, border_color=COLORS["line"])
        self.camera_card.grid(row=0, column=0, sticky="nsew", pady=(0, 16))
        self.camera_card.grid_rowconfigure(0, weight=1)
        self.camera_card.grid_columnconfigure(0, weight=1)
        self.camera_canvas = tk.Canvas(self.camera_card, bg=COLORS["deep"], highlightthickness=0, bd=0)
        self.camera_canvas.grid(row=0, column=0, sticky="nsew", padx=1, pady=1)
        # A dark copy 1px below keeps the label readable on bright frames.
        self.camera_canvas.create_text(
            17, 17, text="Live \u00b7 largest face", fill=COLORS["deep"],
            font=self._font(12, mono=True), anchor="nw", tags="camera_tag",
        )
        self.camera_canvas.create_text(
            16, 16, text="Live \u00b7 largest face", fill=COLORS["text"],
            font=self._font(12, mono=True), anchor="nw", tags="camera_tag",
        )
        self._camera_alarm_border = self.camera_canvas.create_line(
            1, 1, 1, 1, fill="", width=2, tags="camera_alarm_border"
        )

        trace_card = ctk.CTkFrame(left, height=216, fg_color=COLORS["card"], corner_radius=16, border_width=1, border_color=COLORS["line"])
        trace_card.grid(row=1, column=0, sticky="nsew")
        trace_card.grid_propagate(False)
        trace_card.grid_columnconfigure(0, weight=1)
        trace_card.grid_rowconfigure(1, weight=1)
        header = ctk.CTkFrame(trace_card, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=20, pady=(16, 4))
        ctk.CTkLabel(header, text="Eye openness", text_color=COLORS["text"], font=self._font(14)).pack(side="left")
        self.trace_value = ctk.CTkLabel(header, text="last 10 s \u00b7 now --", text_color=COLORS["muted"], font=self._font(12, mono=True))
        self.trace_value.pack(side="right")
        self.trace_canvas = tk.Canvas(trace_card, bg=COLORS["card"], highlightthickness=0, bd=0)
        self.trace_canvas.grid(row=1, column=0, sticky="nsew", padx=20, pady=(0, 12))
        self.trace_canvas.bind("<Configure>", lambda _event: self._draw_trace(force=True))

        side = ctk.CTkFrame(body, width=296, fg_color="transparent")
        side.grid(row=0, column=1, sticky="nsew")
        side.grid_propagate(False)
        side.grid_columnconfigure(0, weight=1)
        side.grid_rowconfigure(2, weight=1)

        status_card = ctk.CTkFrame(side, fg_color=COLORS["card"], corner_radius=16, border_width=1, border_color=COLORS["line"])
        status_card.grid(row=0, column=0, sticky="ew", pady=(0, 16))
        status_card.grid_columnconfigure(0, weight=1)
        state_line = ctk.CTkFrame(status_card, fg_color="transparent")
        state_line.grid(row=0, column=0, sticky="w", padx=20, pady=(20, 0))
        self.state_dot = ctk.CTkFrame(state_line, width=8, height=8, corner_radius=4, fg_color=COLORS["ok"])
        self.state_dot.pack_propagate(False)
        self.state_dot.pack(side="left", padx=(0, 8))
        self.watching = ctk.CTkLabel(state_line, text="Watching", text_color=COLORS["muted"], font=self._font(13))
        self.watching.pack(side="left")
        self.state_label = ctk.CTkLabel(status_card, text="Awake", text_color=COLORS["text"], font=self._font(32, "medium"))
        self.state_label.grid(row=1, column=0, sticky="w", padx=20, pady=(4, 0))
        self.detail_label = ctk.CTkLabel(status_card, text="Eyes open", text_color=COLORS["secondary"], font=self._font(14), wraplength=250, justify="left")
        self.detail_label.grid(row=2, column=0, sticky="w", padx=20, pady=(0, 20))

        stats = ctk.CTkFrame(side, fg_color=COLORS["card"], corner_radius=16, border_width=1, border_color=COLORS["line"])
        stats.grid(row=1, column=0, sticky="ew", pady=(0, 16))
        stats.grid_columnconfigure(0, weight=1)
        self.stat_values = {}
        for row, (key, label) in enumerate((("blinks", "Blinks"), ("yawns", "Yawns"), ("alerts", "Alerts"), ("longest", "Longest closed"))):
            line = ctk.CTkFrame(stats, fg_color="transparent", height=40)
            line.grid(row=row, column=0, sticky="ew", padx=20)
            line.grid_propagate(False)
            if row:
                ctk.CTkFrame(line, height=1, fg_color=COLORS["line"]).pack(side="top", fill="x")
            ctk.CTkLabel(line, text=label, text_color=COLORS["secondary"], font=self._font(14)).pack(side="left", pady=8)
            value = ctk.CTkLabel(line, text="0", text_color=COLORS["text"], font=self._font(15, mono=True))
            value.pack(side="right", pady=8)
            self.stat_values[key] = value

        actions = ctk.CTkFrame(side, fg_color="transparent")
        actions.grid(row=2, column=0, sticky="sew")
        actions.grid_columnconfigure((0, 1), weight=1)
        self.pause_button = self._button(actions, "Pause", self.toggle_pause, primary=True)
        self.pause_button.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        self._button(actions, "Calibrate", self._start_calibration).grid(row=1, column=0, sticky="ew", padx=(0, 4))
        self._button(actions, "Settings", self.open_settings).grid(row=1, column=1, sticky="ew", padx=(4, 0))
        quit_button = ctk.CTkButton(actions, text="Quit and save log", command=self.close, height=32, fg_color="transparent", hover_color=COLORS["raised"], text_color=COLORS["muted"], font=self._font(13), corner_radius=10)
        quit_button.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(4, 0))

    def _button(self, parent, text, command, primary=False):
        return ctk.CTkButton(
            parent, text=text, command=command, height=44, corner_radius=10,
            fg_color=COLORS["text"] if primary else COLORS["raised"],
            hover_color=COLORS["text"] if primary else COLORS["raised"],
            text_color=COLORS["page"] if primary else COLORS["text"],
            border_width=0 if primary else 1,
            border_color=COLORS["line"], font=self._font(14, "medium"),
        )

    def _set_visual_state(self, state, word, detail):
        self.state_label.configure(text=word)
        self.detail_label.configure(text=detail)
        alarm = state in ("drowsy", "face_lost_alarm")
        accent = COLORS["alarm"] if alarm else COLORS["amber"] if state in ("yawning", "face_lost_warning", "calibrating") else COLORS["text"]
        dot = COLORS["alarm"] if alarm else COLORS["amber"] if state in ("yawning", "face_lost_warning", "calibrating") else COLORS["ok"]
        self.state_label.configure(text_color=accent)
        self.state_dot.configure(fg_color=dot)
        self.watching.configure(text="Alarm on" if alarm else "Watching", text_color=COLORS["muted"])
        self.camera_card.configure(border_width=2 if alarm else 1, border_color=COLORS["alarm"] if alarm else COLORS["line"])
        if hasattr(self, "camera_canvas"):
            self._update_camera_border(alarm)
        if state == "paused":
            self.state_dot.configure(fg_color=COLORS["muted"])
            self.watching.configure(text="Paused", text_color=COLORS["muted"])
        self._draw_trace()

    def _update_header(self):
        elapsed = self.session_log.summary(longest_eyes_closed=0)["total_seconds"] if hasattr(self, "session_log") else 0
        active_camera = "—" if self.camera_index is None else str(self.camera_index)
        self.header_meta.configure(text=f"Session {format_duration(elapsed)}     Camera {active_camera}     {self.fps:.0f} fps")
        if not self.is_closing:
            self.root.after(1000, self._update_header)

    def _camera_indices(self):
        return tuple(dict.fromkeys((self.settings.camera_index, 0, 1, 2, 3)))

    def _open_camera(self, indices=None):
        for index in self._camera_indices() if indices is None else indices:
            capture = None
            try:
                capture = cv2.VideoCapture(index)
                if capture.isOpened():
                    self.capture = capture
                    self.camera_index = index
                    return True
            except cv2.error:
                pass
            if capture is not None:
                capture.release()
        self.capture = None
        return False

    def _schedule_camera_retry(self):
        if not self.is_closing and self.camera_retry_id is None and not self.is_paused:
            delay = self.camera_retry_delay_ms
            self.camera_retry_id = self.root.after(delay, self._retry_camera)
            self.camera_retry_delay_ms = min(delay * 2, 5000)

    def _retry_camera(self):
        self.camera_retry_id = None
        if self.is_closing or getattr(self, "is_paused", False):
            return
        retry_index = (
            self.settings.camera_index
            if self.settings.camera_index != self.camera_index
            else self.camera_index
        )
        if self._open_camera((retry_index,)):
            self.camera_retry_delay_ms = 1000
            if not self.calibration_started:
                self._start_calibration()
            self.root.after(0, self.process_frame)
        else:
            self._set_visual_state("face_lost_alarm", "Face not found", "Camera disconnected \u00b7 retrying")
            self._schedule_camera_retry()

    def process_frame(self):
        if self.is_closing or getattr(self, "is_paused", False):
            return
        if self.capture is None:
            self._set_visual_state("face_lost_alarm", "Face not found", "Camera disconnected \u00b7 retrying")
            self._schedule_camera_retry()
            return
        try:
            ret, frame = self.capture.read()
            if not ret or frame is None:
                self.capture.release()
                self.capture = None
                self._set_visual_state("face_lost_alarm", "Face not found", "Camera disconnected \u00b7 retrying")
                self._schedule_camera_retry()
                return
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self._detect_faces(gray)
            now = time.monotonic()
            self._face_found = bool(faces)
            if faces:
                largest_face = max(faces, key=lambda face: face.width() * face.height())
                status = self._process_face(gray, frame, largest_face, now)
            elif self.is_calibrating:
                self._collect_calibration(None, now)
                status = None
            else:
                status = self.monitor.update(None, now)
            self._record_fps(time.monotonic())
            if status is not None:
                self.current_status = status
                self._apply_status(status)
            self._show_camera(frame)
            self._update_stats()
            self._append_trace(now, status.ear if status else None)
            self._frame_error_logged = False
        except Exception as error:
            if not self._frame_error_logged:
                print(f"Frame processing error: {error}")
                self._frame_error_logged = True
        finally:
            if not self.is_closing and not getattr(self, "is_paused", False) and self.capture is not None and self.camera_retry_id is None:
                self.root.after(20, self.process_frame)

    def _show_camera(self, frame):
        target_w = max(1, self.camera_canvas.winfo_width())
        target_h = max(1, self.camera_canvas.winfo_height())
        height, width = frame.shape[:2]
        target_ratio = target_w / target_h
        source_ratio = width / height
        if source_ratio > target_ratio:
            crop_w = max(1, round(height * target_ratio))
            left = (width - crop_w) // 2
            frame = frame[:, left:left + crop_w]
        elif source_ratio < target_ratio:
            crop_h = max(1, round(width / target_ratio))
            top = (height - crop_h) // 2
            frame = frame[top:top + crop_h, :]
        resized = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
        image = Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)).convert("RGBA")
        if self.camera_photo is None or self._camera_image_size != (target_w, target_h):
            self._camera_mask = Image.new("L", (target_w, target_h), 0)
            ImageDraw.Draw(self._camera_mask).rounded_rectangle(
                (0, 0, target_w - 1, target_h - 1), radius=16, fill=255
            )
            image.putalpha(self._camera_mask)
            self.camera_photo = ImageTk.PhotoImage(image)
            self._camera_image_size = (target_w, target_h)
            self.camera_canvas.delete("camera_image")
            self.camera_canvas.create_image(0, 0, image=self.camera_photo, anchor="nw", tags="camera_image")
            self.camera_canvas.tag_raise("camera_tag")
        else:
            image.putalpha(self._camera_mask)
            self.camera_photo.paste(image)
        if self._camera_border_size != (target_w, target_h):
            self._update_camera_border(bool(self._camera_alarm_active))

    def _update_camera_border(self, alarm):
        alarm = bool(alarm)
        if not hasattr(self, "camera_canvas"):
            self._camera_alarm_active = alarm
            return
        width = max(1, self.camera_canvas.winfo_width())
        height = max(1, self.camera_canvas.winfo_height())
        if self._camera_border_size == (width, height) and self._camera_alarm_active == alarm:
            return
        self._camera_alarm_active = alarm
        self._camera_border_size = (width, height)
        radius = min(16, (width - 2) / 2, (height - 2) / 2)
        centers = (
            (radius, radius, 180, 270),
            (width - 1 - radius, radius, 270, 360),
            (width - 1 - radius, height - 1 - radius, 0, 90),
            (radius, height - 1 - radius, 90, 180),
        )
        points = []
        for center_x, center_y, start, end in centers:
            for step in range(9):
                angle = math.radians(start + (end - start) * step / 8)
                points.extend((center_x + radius * math.cos(angle), center_y + radius * math.sin(angle)))
        points.extend(points[:2])
        color = COLORS["alarm"] if alarm else ""
        self.camera_canvas.coords(self._camera_alarm_border, *points)
        self.camera_canvas.itemconfigure(self._camera_alarm_border, fill=color)
        self.camera_canvas.tag_raise("camera_alarm_border")

    def _process_face(self, gray, frame, face, now=None):
        landmarks = self.predictor(gray, face)
        left_eye, right_eye = eye_landmarks(landmarks)
        left_eye = np.array(left_eye, dtype=np.int32)
        right_eye = np.array(right_eye, dtype=np.int32)
        eye_color = (77, 72, 232) if self.current_status and self.current_status.state == "drowsy" else (61, 163, 232)
        self._draw_eye_outline(frame, left_eye, eye_color)
        self._draw_eye_outline(frame, right_eye, eye_color)
        ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
        mar = mouth_aspect_ratio(mouth_landmarks(landmarks))
        face_height = max(1, face.bottom() - face.top())
        nose_y = (landmarks.part(30).y - face.top()) / face_height
        now = time.monotonic() if now is None else now
        if self.is_calibrating:
            self._collect_calibration(ear, now, nose_y)
            return None
        return self.monitor.update(ear, now, mar=mar, nose_y=nose_y)

    def _detect_faces(self, gray):
        small_gray = cv2.resize(gray, None, fx=0.5, fy=0.5)
        small_faces = self.face_detector(small_gray)
        return [dlib.rectangle(face.left() * 2, face.top() * 2, face.right() * 2, face.bottom() * 2) for face in small_faces]

    def _record_fps(self, now):
        self._frame_times.append(now)
        if len(self._frame_times) > 1:
            elapsed = self._frame_times[-1] - self._frame_times[0]
            if elapsed > 0:
                self.fps = (len(self._frame_times) - 1) / elapsed

    def _start_calibration(self, now=None):
        now = time.monotonic() if now is None else now
        previous_yawn_count = self.monitor.yawn_count
        previous_longest = self.monitor.longest_eyes_closed_seconds
        previous_blinks = self.monitor.blink_count
        previous_last_blink = self.monitor.last_blink_at
        self.monitor = DrowsinessMonitor(self.settings)
        self.monitor.yawn_count = previous_yawn_count
        self.monitor.longest_eyes_closed_seconds = previous_longest
        self.monitor.blink_count = previous_blinks
        self.monitor.last_blink_at = previous_last_blink
        self.calibrator.start(now)
        self.calibration_started_at = now
        self.calibration_started = True
        self.is_calibrating = True
        self._stop_alarm()
        self._last_alarm_on = False
        self._status_hold_until = 0.0
        self._set_visual_state("calibrating", "Calibrating", f"Keep your eyes open \u00b7 {math.ceil(self.settings.calibration_seconds)} s")

    def _collect_calibration(self, ear_or_none, now, nose_y=None):
        if not self.is_calibrating:
            return False
        if self.calibrator.update(ear_or_none, now, nose_y):
            self.is_calibrating = False
            if self.calibrator.succeeded:
                self.settings.ear_threshold = self.calibrator.threshold
                self.monitor.settings.ear_threshold = self.calibrator.threshold
                if self.calibrator.nose_baseline is not None:
                    self.monitor.set_nose_baseline(self.calibrator.nose_baseline)
                try:
                    save_settings(self.settings, self.settings_path)
                except OSError as error:
                    print(f"Could not save calibration: {error}")
                    self._set_visual_state("yawning", "Calibration complete", "Could not save settings")
                    self.session_log.record("calibration", "succeeded; settings not saved")
                    self._status_hold_until = time.monotonic() + 3.0
                    return False
                self._set_visual_state("ok", "Awake", "Calibration complete")
                self.session_log.record("calibration", "succeeded")
            else:
                self._set_visual_state("yawning", "Awake", "Calibration failed \u00b7 keeping previous threshold")
                self.session_log.record("calibration", "failed")
            self._status_hold_until = time.monotonic() + 3.0
            return False
        remaining = max(0, math.ceil(self.settings.calibration_seconds - (now - self.calibration_started_at)))
        if hasattr(self, "detail_label"):
            self.detail_label.configure(text=f"Keep your eyes open - {remaining} s")
        return True

    def _draw_eye_outline(self, frame, eye_points, color=(61, 163, 232)):
        for index in range(len(eye_points)):
            start = tuple(eye_points[index])
            end = tuple(eye_points[(index + 1) % len(eye_points)])
            cv2.line(frame, start, end, color, 2)

    def _apply_status(self, status):
        if status.alarm_on:
            self._start_alarm()
        else:
            self._stop_alarm()
        if status.alarm_on and not getattr(self, "_last_alarm_on", False):
            event = "face_lost_alarm" if status.state == "face_lost_alarm" else "drowsy"
            self.session_log.record(event, status.state)
        self._last_alarm_on = status.alarm_on
        yawn_count = getattr(status, "yawn_count", 0)
        if yawn_count > getattr(self, "_last_yawn_count", 0):
            for count in range(getattr(self, "_last_yawn_count", 0) + 1, yawn_count + 1):
                self.session_log.record("yawn", f"count={count}")
        self._last_yawn_count = yawn_count
        if time.monotonic() < getattr(self, "_status_hold_until", 0.0):
            return
        closed_since = getattr(getattr(self, "monitor", None), "_closed_eyes_since", None)
        closed_for = 0.0 if closed_since is None else max(0.0, time.monotonic() - closed_since)
        face_missing_since = getattr(getattr(self, "monitor", None), "_face_missing_since", None)
        states = {
            "ok": ("Awake", "Eyes open"),
            "eyes_closing": ("Awake", f"Eyes closed for {closed_for:.1f} s"),
            "yawning": ("Yawning", "Yawn detected"),
            "face_lost_warning": ("Face not found", "Move into the camera view"),
            "drowsy": (
                "Drowsy",
                f"Eyes closed for {closed_for:.1f} s"
                if closed_since is not None
                else "Sustained downward head nod",
            ),
            "face_lost_alarm": (
                "Face not found",
                f"No face for {max(0, int(time.monotonic() - face_missing_since))} s"
                if face_missing_since is not None
                else "No face detected",
            ),
        }
        word, detail = states[status.state]
        self._set_visual_state(status.state, word, detail)
        self._update_stats()

    def _update_stats(self):
        self.stat_values["blinks"].configure(text=str(self.monitor.blink_count))
        self.stat_values["yawns"].configure(text=str(self.monitor.yawn_count))
        self.stat_values["alerts"].configure(text=str(sum(event in self.session_log.ALERT_EVENTS for _, event, _ in self.session_log.events)))
        self.stat_values["longest"].configure(text=f"{self.monitor.longest_eyes_closed_seconds:.1f} s")
        if self.monitor.last_blink_at is not None and getattr(getattr(self, "current_status", None), "state", "ok") in ("ok", "eyes_closing"):
            ago = max(0, int(time.monotonic() - self.monitor.last_blink_at))
            self.detail_label.configure(text=f"Eyes open \u00b7 last blink {ago} s ago")

    def _append_trace(self, now, ear):
        if ear is not None:
            self._trace.append((now, ear))
        while self._trace and now - self._trace[0][0] > 10:
            self._trace.popleft()
        if now - self._last_trace_draw >= 0.1:
            self._draw_trace()

    def _draw_trace(self, force=False):
        if not hasattr(self, "trace_canvas"):
            return
        now = time.monotonic()
        if not force and now - self._last_trace_draw < 0.1:
            return
        self._last_trace_draw = now
        canvas = self.trace_canvas
        canvas.delete("all")
        width = max(1, canvas.winfo_width())
        height = max(1, canvas.winfo_height())
        threshold = self.settings.ear_threshold
        value = self._trace[-1][1] if self._trace else None
        text = "last 10 s \u00b7 now --" if value is None else f"last 10 s \u00b7 now {value:.2f}"
        color = COLORS["alarm"] if value is not None and value < threshold else COLORS["muted"]
        self.trace_value.configure(text=text, text_color=color)
        for fraction in (0.2, 0.5, 0.8):
            y = int(height * fraction)
            canvas.create_line(0, y, width, y, fill=COLORS["line"], width=1)
        max_ear = max(0.5, threshold * 1.8)
        threshold_y = height * (1 - min(1, threshold / max_ear))
        for x in range(0, width, 8):
            canvas.create_line(x, threshold_y, min(width, x + 3), threshold_y, fill=COLORS["amber"], width=1)
        canvas.create_text(4, max(9, threshold_y - 9), text=f"your line {threshold:.2f}", fill=COLORS["amber"], anchor="w", font=self._font(10, mono=True))
        if not self._trace:
            return
        points = [(width * max(0, 1 - (now - stamp) / 10), height * (1 - min(1, ear / max_ear))) for stamp, ear in self._trace]
        for index, (start, end) in enumerate(zip(points, points[1:])):
            segment_color = COLORS["alarm"] if (self._trace[index][1] + self._trace[index + 1][1]) / 2 < threshold else COLORS["text"]
            canvas.create_line(*start, *end, fill=segment_color, width=2)
        x, y = points[-1]
        dot_color = COLORS["alarm"] if value is not None and value < threshold else COLORS["text"]
        canvas.create_oval(x-2, y-2, x+2, y+2, fill=dot_color, outline=dot_color)

    def toggle_pause(self):
        self.is_paused = not self.is_paused
        if self.is_paused:
            self._stop_alarm()
            if self.camera_retry_id is not None:
                self.root.after_cancel(self.camera_retry_id)
                self.camera_retry_id = None
            self.pause_button.configure(text="Resume")
            self._set_visual_state("paused", "Paused", "Processing stopped")
        else:
            self.pause_button.configure(text="Pause")
            self._set_visual_state("ok", "Awake", "Resuming camera")
            if self.capture is None:
                self._schedule_camera_retry()
            else:
                self.root.after(0, self.process_frame)

    def open_settings(self):
        window = ctk.CTkToplevel(self.root)
        window.title("Settings")
        window.geometry("360x420")
        window.resizable(False, False)
        window.configure(fg_color=COLORS["page"])
        window.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(window, text="Settings", font=self._font(22, "medium"), text_color=COLORS["text"]).grid(row=0, column=0, sticky="w", padx=24, pady=(24, 16))
        fields = (("Camera index", "camera_index", self.settings.camera_index), ("Eyes closed seconds", "eyes_closed_seconds", self.settings.eyes_closed_seconds), ("Yawn threshold", "yawn_mar_threshold", self.settings.yawn_mar_threshold))
        entries = {}
        for row, (label, key, value) in enumerate(fields, 1):
            ctk.CTkLabel(window, text=label, text_color=COLORS["secondary"], font=self._font(13)).grid(row=row, column=0, sticky="w", padx=24, pady=(8, 2))
            entry = ctk.CTkEntry(window, fg_color=COLORS["raised"], border_color=COLORS["line"], text_color=COLORS["text"], font=self._font(14, mono=True))
            entry.insert(0, str(value))
            entry.grid(row=row, column=0, sticky="ew", padx=24)
            entries[key] = entry
        ctk.CTkLabel(window, text="Alarm volume", text_color=COLORS["secondary"], font=self._font(13)).grid(row=4, column=0, sticky="w", padx=24, pady=(12, 2))
        volume = ctk.CTkSlider(
            window, from_=0, to=1, number_of_steps=100,
            progress_color=COLORS["secondary"],
            button_color=COLORS["muted"],
            button_hover_color=COLORS["secondary"],
            command=lambda value: pygame.mixer.music.set_volume(value),
        )
        volume.set(self.settings.alarm_volume)
        volume.grid(row=5, column=0, sticky="ew", padx=24)

        def save_and_close():
            try:
                camera_index = int(entries["camera_index"].get())
                eyes_closed = float(entries["eyes_closed_seconds"].get())
                yawn_threshold = float(entries["yawn_mar_threshold"].get())
                if (
                    camera_index < 0
                    or not math.isfinite(eyes_closed)
                    or not math.isfinite(yawn_threshold)
                    or eyes_closed <= 0
                    or yawn_threshold <= 0
                ):
                    raise ValueError
            except ValueError:
                messagebox.showerror("Invalid settings", "Enter a camera index of 0 or more and positive threshold values", parent=window)
                return
            updated_settings = type(self.settings)(**vars(self.settings))
            updated_settings.camera_index = camera_index
            updated_settings.eyes_closed_seconds = eyes_closed
            updated_settings.yawn_mar_threshold = yawn_threshold
            updated_settings.alarm_volume = float(volume.get())
            try:
                save_settings(updated_settings, self.settings_path)
            except OSError as error:
                messagebox.showerror("Settings not saved", str(error), parent=window)
                return
            self.settings = updated_settings
            self.monitor.settings = updated_settings
            self.calibrator.settings = updated_settings
            window.destroy()

        self._button(window, "Save", save_and_close, primary=True).grid(row=6, column=0, sticky="ew", padx=24, pady=(20, 8))
        self._button(window, "Re-run calibration", lambda: (window.destroy(), self._start_calibration())).grid(row=7, column=0, sticky="ew", padx=24)
        window.transient(self.root)
        window.grab_set()

    def _start_alarm(self):
        if not pygame.mixer.music.get_busy():
            pygame.mixer.music.play(-1)

    def _stop_alarm(self):
        if pygame.mixer.music.get_busy():
            pygame.mixer.music.stop()

    def leave_fullscreen(self, _event=None):
        self.root.attributes("-fullscreen", False)

    def toggle_fullscreen(self, _event=None):
        fullscreen = self.root.attributes("-fullscreen")
        self.root.attributes("-fullscreen", fullscreen not in (True, 1, "1"))

    def quit_from_key(self, _event=None):
        self.close()

    def close(self):
        if self.is_closing:
            return
        self.is_closing = True
        if self.camera_retry_id is not None:
            self.root.after_cancel(self.camera_retry_id)
            self.camera_retry_id = None
        if self.capture is not None:
            try:
                self.capture.release()
            except cv2.error:
                pass
            self.capture = None
        try:
            pygame.mixer.music.stop()
            pygame.mixer.quit()
        except pygame.error:
            pass
        try:
            try:
                path = self.session_log.write()
                print(f"Session log: {path}")
            except Exception as error:
                print(f"Could not write session log: {error}")
            summary = self.session_log.summary(longest_eyes_closed=self.monitor.longest_eyes_closed_seconds)
            messagebox.showinfo(
                "Session summary",
                f"Total time: {format_duration(summary['total_seconds'])}\nAlerts: {summary['alerts']}\nYawns: {summary['yawns']}\nLongest eyes closed: {summary['longest_eyes_closed']:.1f} seconds",
                parent=self.root,
            )
        except Exception as error:
            print(f"Could not show session summary: {error}")
        finally:
            self.root.destroy()


def _show_startup_error(root, title, message):
    root.withdraw()
    messagebox.showerror(title, message, parent=root)
    root.destroy()


def main():
    script_dir = Path(__file__).resolve().parent
    ctk.set_appearance_mode("dark")
    root = ctk.CTk()
    model_path = script_dir / "shape_predictor_68_face_landmarks.dat"
    alarm_sound_path = script_dir / "alarm_sound.wav"
    model_download_url = "https://drive.google.com/file/d/171wLhOsdIdax5XHYfLwCSA3tuAl-myu0/view"
    settings = load_settings(script_dir / "settings.json")
    missing_files = []
    if not model_path.is_file():
        missing_files.append(f"Missing face landmark model:\n{model_path}")
    if not alarm_sound_path.is_file():
        missing_files.append(f"Missing alarm sound:\n{alarm_sound_path}")
    if missing_files:
        message = "\n\n".join(missing_files) + f"\n\nModel download: {model_download_url}"
        _show_startup_error(root, "Required file not found", message)
        return
    try:
        pygame.mixer.init()
        pygame.mixer.music.load(str(alarm_sound_path))
        pygame.mixer.music.set_volume(settings.alarm_volume)
    except pygame.error as error:
        pygame.mixer.quit()
        _show_startup_error(root, "Alarm audio error", f"Could not load or play {alarm_sound_path}:\n{error}")
        return
    try:
        DrowsinessApp(root, settings, model_path, script_dir / "settings.json")
    except Exception as error:
        pygame.mixer.quit()
        _show_startup_error(root, "Startup error", str(error))
        return
    root.mainloop()


if __name__ == "__main__":
    main()
