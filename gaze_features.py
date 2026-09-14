"""
Stage 6 — Gaze deviation + rapid gaze change.

Depends on: face_landmarker.task (already downloaded), the same landmark
result object you're already getting in face_landmarks_live.py.

Iris indices (fixed, from Stage 4/5 notes):
    right iris: 468-472   right eye corners: 33, 133*
    left iris:  473-477   left eye corners: 362, 263*

* NOTE: MediaPipe's "left/right" is from the SUBJECT's perspective, but the
  eye-corner indices your blink_test.py already uses (33/133 for one eye,
  362/263 for the other) pair with iris blocks 468-472 and 473-477
  respectively — keep the same left/right convention you used for EAR so
  gaze and blink features reference the same eye consistently.
"""

import time
import numpy as np


class GazeTracker:
    def __init__(self, deviation_threshold=0.35, rapid_change_threshold=0.15,
                 rapid_change_window=0.3, smoothing=5):
        """
        deviation_threshold: normalized iris offset from eye-center that
            counts as "looking away" (0 = dead center, ~1 = at eye corner).
        rapid_change_threshold: how much the normalized gaze position has to
            shift within `rapid_change_window` seconds to count as a rapid
            gaze change (saccade-like event).
        smoothing: number of frames to moving-average the raw offset over,
            to kill single-frame landmark jitter.
        """
        self.deviation_threshold = deviation_threshold
        self.rapid_change_threshold = rapid_change_threshold
        self.rapid_change_window = rapid_change_window
        self._history = []  # list of (timestamp, gaze_x, gaze_y)
        self._smooth_buf = []
        self.smoothing = smoothing

        self.rapid_change_count = 0
        self.current_deviation = 0.0
        self.is_looking_away = False

    @staticmethod
    def _iris_center(landmarks, iris_indices):
        pts = np.array([[landmarks[i].x, landmarks[i].y] for i in iris_indices])
        return pts.mean(axis=0)

    @staticmethod
    def _normalized_gaze(iris_center, corner_a, corner_b, eye_top, eye_bottom):
        """
        Map iris center into [-1, 1] on both axes relative to the eye's own
        bounding box, so it's resolution- and face-distance-independent.
        """
        x_span = corner_b[0] - corner_a[0]
        y_span = eye_bottom[1] - eye_top[1]
        if abs(x_span) < 1e-6 or abs(y_span) < 1e-6:
            return 0.0, 0.0
        gx = ((iris_center[0] - corner_a[0]) / x_span) * 2 - 1
        gy = ((iris_center[1] - eye_top[1]) / y_span) * 2 - 1
        return gx, gy

    def update(self, face_landmarks):
        """
        face_landmarks: the .face_landmarks[0] list from a FaceLandmarker
        result (normalized landmarks, same object you already use for EAR).
        Returns a dict of current gaze features for this frame.
        """
        lm = face_landmarks

        # Right eye (indices 33/133 corners, iris 468-472)
        r_iris = self._iris_center(lm, range(468, 473))
        r_corner_a = np.array([lm[33].x, lm[33].y])
        r_corner_b = np.array([lm[133].x, lm[133].y])
        r_top = np.array([lm[159].x, lm[159].y])
        r_bottom = np.array([lm[145].x, lm[145].y])
        rgx, rgy = self._normalized_gaze(r_iris, r_corner_a, r_corner_b, r_top, r_bottom)

        # Left eye (indices 362/263 corners, iris 473-477)
        l_iris = self._iris_center(lm, range(473, 478))
        l_corner_a = np.array([lm[362].x, lm[362].y])
        l_corner_b = np.array([lm[263].x, lm[263].y])
        l_top = np.array([lm[386].x, lm[386].y])
        l_bottom = np.array([lm[374].x, lm[374].y])
        lgx, lgy = self._normalized_gaze(l_iris, l_corner_a, l_corner_b, l_top, l_bottom)

        gaze_x = (rgx + lgx) / 2.0
        gaze_y = (rgy + lgy) / 2.0

        # smoothing
        self._smooth_buf.append((gaze_x, gaze_y))
        if len(self._smooth_buf) > self.smoothing:
            self._smooth_buf.pop(0)
        sx = float(np.mean([p[0] for p in self._smooth_buf]))
        sy = float(np.mean([p[1] for p in self._smooth_buf]))

        now = time.time()
        self._history.append((now, sx, sy))
        cutoff = now - max(self.rapid_change_window, 1.0)
        self._history = [h for h in self._history if h[0] >= cutoff]

        deviation = float(np.hypot(sx, sy))
        self.current_deviation = deviation
        self.is_looking_away = deviation > self.deviation_threshold

        rapid = self._check_rapid_change(now, sx, sy)
        if rapid:
            self.rapid_change_count += 1

        return {
            "gaze_x": sx,
            "gaze_y": sy,
            "deviation": deviation,
            "looking_away": self.is_looking_away,
            "rapid_change": rapid,
            "rapid_change_count_total": self.rapid_change_count,
        }

    def _check_rapid_change(self, now, sx, sy):
        window_start = now - self.rapid_change_window
        past_points = [h for h in self._history if h[0] <= window_start + 0.001]
        if not past_points and self._history:
            past_points = [self._history[0]]
        if not past_points:
            return False
        _, px, py = past_points[-1]
        shift = float(np.hypot(sx - px, sy - py))
        return shift > self.rapid_change_threshold

    def reset(self):
        self._history.clear()
        self._smooth_buf.clear()
        self.rapid_change_count = 0


if __name__ == "__main__":
    # Quick manual smoke test with a live webcam + FaceLandmarker, matching
    # the pattern from face_landmarks_live.py / blink_test.py.
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    MODEL_PATH = "models/face_landmarker.task"

    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_faces=1,
    )
    landmarker = mp_vision.FaceLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    tracker = GazeTracker()
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(frame_idx * (1000 / 30))
        result = landmarker.detect_for_video(mp_image, timestamp_ms)
        frame_idx += 1

        if result.face_landmarks:
            feats = tracker.update(result.face_landmarks[0])
            color = (0, 0, 255) if feats["looking_away"] else (0, 255, 0)
            cv2.putText(frame, f"dev: {feats['deviation']:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, f"rapid changes: {feats['rapid_change_count_total']}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow("Gaze Test", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
