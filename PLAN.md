# Drowsiness Detection – Improvement Plan

Project: `drowsiness_detection.py` (Tkinter + OpenCV + dlib eye-aspect-ratio detector with an alarm).

Do the tasks **in order**. Each task is written so it can be pasted as-is to the coder (AI or human).
After each task, send the diff back for review before starting the next one.

---

## Bugs found in the current code (fix these first)

| # | Problem | Where |
|---|---------|-------|
| B1 | Camera is hardcoded to index `1`. Most laptops only have camera `0`, so the app exits with "Failed to open the video capture." | `cv2.VideoCapture(1)` |
| B2 | If a frame fails to read, `root.destroy()` is called but the function keeps going and crashes on `cv2.flip(None)`. | top of `main()` |
| B3 | Pressing `q` does nothing: `cv2.waitKey` only works with an OpenCV window, and this app uses Tkinter. | end of `main()` |
| B4 | App opens fullscreen with no keyboard way out (only the Exit button). | `root.attributes('-fullscreen', True)` |
| B5 | Alarm plays once and stops, even if the person stays asleep. | `pygame.mixer.music.play()` |
| B6 | If the face disappears (head drops forward — a very common sign of sleep!), the drowsy state freezes and nothing happens. | `for face in faces:` |
| B7 | With 2+ faces in view, all faces share one counter, so results are mixed up. | `for face in faces:` |
| B8 | "16 frames" means different times on fast vs slow computers. | `EAR_CONSEC_FRAMES` |
| B9 | Camera release / mixer quit run at import time, outside `main`, and not on window close. | bottom of file |
| B10 | Missing files crash with an unclear error (`.dat` model, `alarm_sound.wav`). | model/sound loading |

---

## Task 1 – Project setup (small)

- Add `requirements.txt` (opencv-python, dlib, scipy, numpy, pygame, Pillow).
- Add `.gitignore` that ignores `shape_predictor_68_face_landmarks.dat`, `__pycache__/`, `.venv/`.
- Delete the `Read` file (it duplicates README.md).
- Rewrite README.md: what it does, install steps (note: on Windows `pip install dlib` needs CMake + Visual C++ build tools, or use a prebuilt wheel), where to get the `.dat` file, how to run, keyboard shortcuts, settings.

**Done when:** `pip install -r requirements.txt` lists every import used, and the repo has no `Read` file.

## Task 2 – Fix bugs B1–B10

Keep the same look and features. Specifically:
- Camera index comes from a setting (default `0`); if it fails, try 0–3 and show a clear message.
- On a failed frame read: show "Camera disconnected" and retry, never crash.
- Remove all `cv2.waitKey` calls. Bind `Esc` = leave fullscreen, `F11` = toggle fullscreen, `q` = quit.
- Alarm loops (`play(-1)`) until the eyes open again.
- **No face for > 2 seconds → treat as warning** ("Face not found") and after a few more seconds sound the alarm.
- Only track the largest face.
- Use **seconds** instead of frames: drowsy if eyes closed ≥ 1.5 s (setting).
- Clean shutdown via `root.protocol("WM_DELETE_WINDOW", ...)`: release camera, stop mixer.
- Check that the model and sound files exist at startup; show a Tkinter error box with the download link if not.

**Done when:** app starts on camera 0, Esc/q work, unplugging the camera does not crash it, closing the window frees the camera.

## Task 3 – Restructure the code

- Split into: `detector.py` (pure logic: EAR, MAR, state machine, no GUI), `app.py` (Tkinter UI), `config.py` (settings with defaults, loaded from/saved to `settings.json`).
- No `global` variables; use a class.
- Add `tests/test_detector.py` with pytest tests for `eye_aspect_ratio` and the drowsy state machine (fake timestamps + fake EAR values, no camera needed).

**Done when:** `pytest` passes without a webcam, and `python app.py` runs the same as before.

## Task 4 – Better detection

- **Calibration on start (5 s):** "Look at the camera with eyes open" → measure the person's normal EAR, set threshold = 75% of it. Fixes the "take off your glasses" problem in the README for many people.
- **Yawn detection:** mouth aspect ratio (landmarks 60–67); count yawns, show "Yawning" warning.
- **Head-nod detection:** if the nose point drops quickly and stays down, count as drowsy.
- **Smooth EAR** with a short moving average (e.g. last 5 values) to stop flicker.
- Downscale the frame for face detection (e.g. 0.5×) to make it faster; scale landmarks back.

**Done when:** tests cover calibration math and yawn logic; FPS is shown and higher than before.

## Task 5 – Better UI

- Live panel: EAR value, a small EAR graph (last 10 s), FPS, blink count, yawn count, session time.
- Status colours: green (OK), yellow (warning: yawning / face lost), red (drowsy, plus screen border flashes red).
- Settings window: camera index, EAR threshold, seconds before alarm, alarm volume, re-run calibration.
- Buttons: Start/Pause, Calibrate, Settings, Exit.

## Task 6 – Session log (optional, nice for a school project)

- Save each drowsy event / yawn with a time to `logs/session_YYYY-MM-DD_HH-MM.csv`.
- At exit, show a summary: total time, number of alerts, longest eyes-closed time.

## Task 7 (optional, bigger) – Swap dlib for MediaPipe Face Mesh

dlib is hard to install on Windows and needs the 100 MB `.dat` file. MediaPipe installs with `pip`, needs no extra file, and works better with glasses and side angles. Keep the same `detector.py` interface so the UI does not change.

---

## Review checklist (used after each task)

- App still starts and shows the camera.
- No crash when: no camera, no face, camera unplugged, files missing.
- `pytest` passes.
- No leftover debug prints / dead code; README matches the behaviour.
