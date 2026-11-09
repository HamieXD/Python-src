# Drowsiness Detection

A small Python desktop app that uses your webcam and facial landmarks to estimate eye openness. At startup it calibrates your open-eye EAR for five seconds and sets the eye threshold to 75% of the median. Calibration needs at least ten samples and a median EAR of at least 0.18; the resulting threshold is bounded to 0.15–0.35. Otherwise, it keeps the previous threshold. The calibration result stays visible for three seconds.

It smooths EAR over the last five frames, detects yawns, and detects a sustained downward head nod. Yawns are counted and shown as an orange warning without sounding the alarm. A head nod held for one second or closed eyes held for the configured time trigger the alarm. It also warns when your face is out of view and sounds the alarm if it remains missing.

The app uses a dark CustomTkinter dashboard with a live camera preview, eye-openness trace, blink and yawn counts, alerts, and calibration controls. Good lighting helps; glasses can make eye detection less reliable.

## Screenshots

See app.

## Required files

Download `shape_predictor_68_face_landmarks.dat` and put it in the project folder:

[Download the 68-point face landmark model](https://drive.google.com/file/d/171wLhOsdIdax5XHYfLwCSA3tuAl-myu0/view)

Keep `alarm_sound.wav` in the project folder as well.

## Install

On Windows with Python 3.12:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The requirements use `dlib-bin`, a prebuilt Windows-friendly package. Installing `dlib` from source on Windows requires CMake and the Visual C++ build tools.

To install the test dependency too:

```powershell
python -m pip install -r requirements-dev.txt
```

## Run

Either command starts the app:

```powershell
python app.py
```

```powershell
python drowsiness_detection.py
```

At startup, the app tries the selected camera index and then indexes 0 through 3. The active camera index, session time, and frame rate appear in the dashboard header. If the camera disconnects, the app retries that same index with delays of 1, 2, 4, then 5 seconds. The face detector uses a half-size grayscale image; landmarks are measured on the full-size image. The interface updates the camera preview and session statistics as frames arrive.

If no face is detected for 2 seconds, the app shows a warning; after 2 more seconds, it sounds the alarm. Brief face dropouts keep the current eye state and alarm, and do not reset the closed-eye timer.

## Keyboard shortcuts

- `Esc`: leave fullscreen
- `F11`: toggle fullscreen
- `q`: quit
- `Quit and save log` button: quit

## Settings

The app reads `settings.json` from the project folder. Open Settings to change the camera index, eye-closure duration, alarm volume, and yawn threshold. Saved changes apply immediately except camera selection, which applies on reconnect. Missing or invalid fields use their defaults individually, and invalid fields are named in a warning. Unknown keys are ignored. Calibration saves a successful threshold to this file.

```json
{
  "camera_index": 0,
  "ear_threshold": 0.25,
  "eyes_closed_seconds": 1.5,
  "face_lost_warning_seconds": 2.0,
  "face_lost_alarm_delay_seconds": 2.0,
  "alarm_volume": 1.0,
  "calibration_seconds": 5.0,
  "yawn_mar_threshold": 0.6,
  "head_nod_drop_ratio": 0.15
}
```

The face-lost alarm delay is the additional time after the warning appears. `yawn_mar_threshold` is the mouth aspect ratio held for at least one second to count a yawn. `head_nod_drop_ratio` is the nose-tip drop as a fraction of face height, held for one second to trigger drowsiness. `alarm_volume` ranges from `0.0` (silent) to `1.0` (full volume). Use the `Calibrate` button to run calibration again.

## Session log

When the app closes, it writes session events to `logs/session_YYYY-MM-DD_HH-MM-SS.csv`; if that name already exists, it adds `_2`, `_3`, and so on and shows a summary with session time, alert count, yawn count, and longest eyes-closed time. Alarm events are counted once when an alarm starts. The `logs/` folder is ignored by Git.

## Tests

The unit tests do not open the camera or start the GUI:

```powershell
python -m pytest -q
```
