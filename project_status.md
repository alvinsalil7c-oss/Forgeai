# PROJECT STATUS: Gaming Behavior AI (Camera + Microphone Module)

Use this as context if continuing this project in a new conversation.

## 1. ENVIRONMENT

- OS: Windows
- Project folder: `C:\Users\Pavan\OneDrive\Desktop\Projects\Gaming project`
- Python version: **3.12.0**, installed via python.org (multiple versions coexist on this machine: 3.11 and 3.13 also present)
- Virtual environment: `venv\` inside the project folder, created with:
  ```
  py -3.12 -m venv venv
  ```
- Activate before every session with:
  ```
  venv\Scripts\activate.bat
  ```

## 2. LIBRARIES INSTALLED (confirmed working versions)

| Library | Version | Installed for |
|---|---|---|
| opencv-python | 5.0.0 | webcam capture, drawing, display |
| mediapipe | 1.0.1 | face landmark detection |
| numpy | 2.5.3 | numerical/array operations |
| pandas | 3.0.5 | (installed, not yet used in code) |

Not yet installed (planned for later stages):
- `sounddevice`, `scipy` — Stage 10 (microphone)
- `librosa`, `webrtcvad-wheels` — Stage 11 (audio features)
- `scikit-learn` — Stage 16-17 (Isolation Forest anomaly detection)
- `xgboost` — Stage 21 (supervised classifier)
- `matplotlib` (optional), `torch` (optional, Stage 23 only)

Full categorized list lives in `requirements.txt` in the project root.

## 3. PROJECT STRUCTURE SO FAR

```
Gaming project/
├── venv/                          (Python 3.12 virtual environment)
├── requirements.txt                (categorized: NOW / SOON / LATER / OPTIONAL)
├── verify_setup.py                 (Stage 1: checks core libs import correctly)
├── webcam_test.py                  (Stage 2: opens webcam, shows live preview)
├── mediapipe_test.py               (Stage 3: loads model, detects face in 1 frame)
├── face_landmarks_live.py          (Stage 4: live face mesh + iris overlay)
├── blink_test.py                   (Stage 5: live EAR + blink counter — just created, not yet verified)
├── models/
│   └── face_landmarker.task        (downloaded MediaPipe model, ~3.58 MB)
└── vision/
    ├── __init__.py                 (empty, makes this an importable package)
    └── eye_features.py             (EAR calculation + BlinkDetector class)
```

Folders defined but not yet populated: `baseline/`, `audio/`, `features/`, `models/` (scripts, as opposed to the model file already inside it), `data/raw/`, `data/processed/`, `data/sessions/`, `visualization/`.

## 4. STAGE PROGRESS (of 25 total)

- [x] Stage 1 — Python 3.12 venv + opencv/mediapipe/numpy/pandas installed and verified
- [x] Stage 2 — Webcam test confirmed working
- [x] Stage 3 — MediaPipe model downloaded, single-frame detection confirmed (478 landmarks, 52 blendshapes)
- [x] Stage 4 — Live face mesh + iris tracking confirmed working smoothly
- [ ] Stage 5 — Blink detection: code written, **not yet run/verified**
- [ ] Stage 6 — Remaining eye features (gaze deviation, rapid gaze change)
- [ ] Stage 7 — Facial features via blendshapes
- [ ] Stage 8 — Head pose/movement
- [ ] Stage 9 — Posture (lightweight proxy, no second model)
- [ ] Stages 10-25 — not started

## 5. KEY TECHNICAL DECISIONS / CORRECTIONS MADE SO FAR

- Must use Python 3.9-3.12 — mediapipe has no Windows wheels for 3.13+.
- `mediapipe.solutions` / `mediapipe.framework` (the old drawing_utils-based API) is **removed** in mediapipe 1.0.1. All landmark drawing is done manually with `cv2.circle`/`cv2.putText`, not the legacy helpers.
- Face Landmarker model (`face_landmarker.task`) must be downloaded manually — it isn't bundled with `pip install mediapipe`. Source URL:
  `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`
- Use `cv2.VideoCapture(0, cv2.CAP_DSHOW)` on Windows — the default MSMF backend can be slow/hang on camera open.
- Iris landmarks are always at fixed indices: right eye `468-472`, left eye `473-477` (Face Landmarker includes iris refinement by default).
- EAR eye landmark indices: left eye corners `33, 133` + vertical pairs `(160,144),(158,153)`; right eye corners `362, 263` + vertical pairs `(385,380),(387,373)`.
- `EAR_THRESHOLD = 0.21` in `eye_features.py` is a **bootstrap value only** — the real system replaces this with each user's personalized baseline in Stage 14-16, not a universal threshold.
- `webrtcvad` fails to build on Windows without a C++ compiler — use `webrtcvad-wheels` instead (same `import webrtcvad` API, prebuilt Windows wheels).
- Posture tracking uses a lightweight proxy (face bounding-box size/position), not a second MediaPipe Pose model, to avoid the extra inference cost.
- Threading is deferred until Stage 12 (audio+video window sync) — a single-threaded loop is fast enough through the vision-only stages.

## 6. NEXT STEP

Run `blink_test.py`, confirm the blink counter increments correctly (once per blink, not double-counting), report back, then proceed to Stage 6.
