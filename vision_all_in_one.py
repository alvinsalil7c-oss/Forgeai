"""
Combined single-file vision pipeline — Stages 5 through 9, plus the anger
composite index, all in one process/window.

IMPORTANT: the BlinkDetector class below is a REIMPLEMENTATION from the EAR
formula and landmark indices documented in your project status file
(EAR_THRESHOLD=0.21, left corners 33/133, right corners 362/263, etc.) —
it is NOT your original vision/eye_features.py, since that file's actual
contents were never shared in this conversation. If its blink counting
behaves differently than your already-verified blink_test.py, replace this
class's body with your real one; everything else in this file (gaze,
blendshapes, anger index, head pose, posture) is unchanged from what you
already tested individually.

CHANGE IN THIS VERSION: the main loop now displays every value each
tracker computes (blink EAR/count, full gaze data, posture data, full head
pose data, tension index, and the anger sub_scores breakdown) instead of
just Angry/Happy/Surprised. No tracker/model logic was changed — only the
on-screen text.

Run from the project root:
    venv\\Scripts\\activate.bat
    python vision_all_in_one.py
Press 'q' to quit.
"""

import time
from collections import deque

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

MODEL_PATH = "models/face_landmarker.task"


# ============================================================
# Stage 5 (reimplemented — see warning above) — Blink detection
# ============================================================
class BlinkDetector:
    def __init__(self, ear_threshold=0.21, consec_frames=2):
        self.ear_threshold = ear_threshold
        self.consec_frames = consec_frames
        self._below_count = 0
        self.blink_count = 0
        self.last_ear = 0.0

    @staticmethod
    def _dist(landmarks, i, j):
        a, b = landmarks[i], landmarks[j]
        return float(np.hypot(a.x - b.x, a.y - b.y))

    def _eye_ear(self, landmarks, corner_a, corner_b, v1a, v1b, v2a, v2b):
        horiz = self._dist(landmarks, corner_a, corner_b)
        if horiz < 1e-6:
            return 0.0
        vert1 = self._dist(landmarks, v1a, v1b)
        vert2 = self._dist(landmarks, v2a, v2b)
        return (vert1 + vert2) / (2.0 * horiz)

    def update(self, face_landmarks):
        left_ear = self._eye_ear(face_landmarks, 33, 133, 160, 144, 158, 153)
        right_ear = self._eye_ear(face_landmarks, 362, 263, 385, 380, 387, 373)
        ear = (left_ear + right_ear) / 2.0
        self.last_ear = ear

        if ear < self.ear_threshold:
            self._below_count += 1
        else:
            if self._below_count >= self.consec_frames:
                self.blink_count += 1
            self._below_count = 0

        return {"ear": ear, "blink_count": self.blink_count}


# ============================================================
# Stage 6 — Gaze deviation + rapid gaze change
# ============================================================
class GazeTracker:
    def __init__(self, deviation_threshold=0.35, rapid_change_threshold=0.15,
                 rapid_change_window=0.3, smoothing=5):
        self.deviation_threshold = deviation_threshold
        self.rapid_change_threshold = rapid_change_threshold
        self.rapid_change_window = rapid_change_window
        self._history = []
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
        x_span = corner_b[0] - corner_a[0]
        y_span = eye_bottom[1] - eye_top[1]
        if abs(x_span) < 1e-6 or abs(y_span) < 1e-6:
            return 0.0, 0.0
        gx = ((iris_center[0] - corner_a[0]) / x_span) * 2 - 1
        gy = ((iris_center[1] - eye_top[1]) / y_span) * 2 - 1
        return gx, gy

    def update(self, lm):
        r_iris = self._iris_center(lm, range(468, 473))
        r_corner_a = np.array([lm[33].x, lm[33].y]); r_corner_b = np.array([lm[133].x, lm[133].y])
        r_top = np.array([lm[159].x, lm[159].y]); r_bottom = np.array([lm[145].x, lm[145].y])
        rgx, rgy = self._normalized_gaze(r_iris, r_corner_a, r_corner_b, r_top, r_bottom)

        l_iris = self._iris_center(lm, range(473, 478))
        l_corner_a = np.array([lm[362].x, lm[362].y]); l_corner_b = np.array([lm[263].x, lm[263].y])
        l_top = np.array([lm[386].x, lm[386].y]); l_bottom = np.array([lm[374].x, lm[374].y])
        lgx, lgy = self._normalized_gaze(l_iris, l_corner_a, l_corner_b, l_top, l_bottom)

        gaze_x = (rgx + lgx) / 2.0
        gaze_y = (rgy + lgy) / 2.0

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
            "gaze_x": sx, "gaze_y": sy, "deviation": deviation,
            "looking_away": self.is_looking_away, "rapid_change": rapid,
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


# ============================================================
# Stage 7 — Blendshapes (tension/surprise/positive) + Anger Index
# ============================================================
TRACKED_CATEGORIES = [
    "browDownLeft", "browDownRight", "browInnerUp", "jawOpen",
    "mouthPressLeft", "mouthPressRight",
    "eyeSquintLeft", "eyeSquintRight",
    "mouthSmileLeft", "mouthSmileRight",
]

ANGER_BLENDSHAPE_CATEGORIES = [
    "browDownLeft", "browDownRight", "browInnerUp",
    "eyeWideLeft", "eyeWideRight", "eyeSquintLeft", "eyeSquintRight",
    "eyeBlinkLeft", "eyeBlinkRight",
    "jawOpen", "jawForward", "jawLeft", "jawRight",
    "mouthPressLeft", "mouthPressRight",
    "mouthRollLower", "mouthRollUpper",
    "mouthStretchLeft", "mouthStretchRight",
]

INNER_BROW_L, INNER_BROW_R = 55, 285
MOUTH_CORNER_L, MOUTH_CORNER_R = 61, 291
EYE_INNER_L, EYE_INNER_R = 133, 362

# Angry comes from AngerIndex, a fully separate composite/amplified score,
# not a raw blendshape delta like Happy/Sad/Surprised — so it needs its own,
# much higher threshold or it wins almost every frame just from baseline
# facial micro-movement. Tune these per-person if a given emotion still
# over- or under-triggers for you (watch the printed candidate numbers).
EMOTION_THRESHOLDS = {
    "Happy": 0.12,
    "Sad": 0.10,
    "Surprised": 0.12,
    "Angry": 0.45,
}


class BlendshapeTracker:
    def __init__(self, tension_gain=1.8):
        self.tension_gain = tension_gain

    @staticmethod
    def scores_dict(face_blendshapes_result):
        return {c.category_name: c.score for c in face_blendshapes_result}

    def update(self, face_blendshapes_result):
        s = self.scores_dict(face_blendshapes_result)
        tension_raw = float(np.mean([s.get("browDownLeft", 0), s.get("browDownRight", 0),
                                      s.get("mouthPressLeft", 0), s.get("mouthPressRight", 0)]))
        tension = float(np.clip(tension_raw * self.tension_gain, 0.0, 1.0))
        surprise = float(np.mean([s.get("browInnerUp", 0), s.get("jawOpen", 0)]))
        positive = float(np.mean([s.get("mouthSmileLeft", 0), s.get("mouthSmileRight", 0)]))
        return {"scores": s, "tension_index": tension, "surprise_index": surprise,
                "positive_affect_index": positive}


# ============================================================
# Emotion classifier — 60-frame "stay neutral" calibration,
# then classifies each frame as Happy / Sad / Surprised / Neutral
# ============================================================
# ============================================================
# Emotion classifier — 60-frame "stay neutral" calibration,
# then produces raw Happy / Sad / Surprised candidate scores
# (0-1, deviation from your personal neutral baseline).
# Final Happy vs Sad vs Surprised vs Angry decision is made in
# main() using EMOTION_THRESHOLDS, since Angry uses a different
# scoring scale (see AngerIndex) and can't be compared directly
# against these raw deltas without its own threshold.
# ============================================================
class EmotionClassifier:
    RELEVANT = [
        "mouthSmileLeft", "mouthSmileRight",
        "mouthFrownLeft", "mouthFrownRight",
        "browInnerUp", "jawOpen",
    ]

    def __init__(self, baseline_frames=60, happy_gain=1.0, sad_gain=2.2, surprise_gain=1.6):
        """
        *_gain: multiplier applied to each raw blendshape deviation before
        clipping to [0, 1]. mouthFrown / browInnerUp+jawOpen naturally stay
        low even on a genuine sad/surprised face, so sad_gain and
        surprise_gain are higher than happy_gain (mouthSmile already swings
        close to 1.0 on its own). Raise these further if a given emotion
        still under-reads for you; lower if it over-triggers.
        """
        self.baseline_frames = baseline_frames
        self.happy_gain = happy_gain
        self.sad_gain = sad_gain
        self.surprise_gain = surprise_gain
        self._baseline_buf = []
        self.baseline = None

    def _extract(self, scores):
        return {k: float(scores.get(k, 0.0)) for k in self.RELEVANT}

    @staticmethod
    def _clip01(v):
        return float(np.clip(v, 0.0, 1.0))

    def update(self, blendshape_scores):
        vals = self._extract(blendshape_scores)

        if self.baseline is None:
            self._baseline_buf.append(vals)
            if len(self._baseline_buf) >= self.baseline_frames:
                self.baseline = {
                    k: float(np.mean([b[k] for b in self._baseline_buf]))
                    for k in self.RELEVANT
                }
            return {
                "calibrating": True,
                "frames_remaining": max(0, self.baseline_frames - len(self._baseline_buf)),
            }

        smile = np.mean([vals["mouthSmileLeft"], vals["mouthSmileRight"]])
        frown = np.mean([vals["mouthFrownLeft"], vals["mouthFrownRight"]])
        surprise = np.mean([vals["browInnerUp"], vals["jawOpen"]])

        base_smile = np.mean([self.baseline["mouthSmileLeft"], self.baseline["mouthSmileRight"]])
        base_frown = np.mean([self.baseline["mouthFrownLeft"], self.baseline["mouthFrownRight"]])
        base_surprise = np.mean([self.baseline["browInnerUp"], self.baseline["jawOpen"]])

        happy_score = self._clip01(max(0.0, float(smile - base_smile)) * self.happy_gain)
        sad_score = self._clip01(max(0.0, float(frown - base_frown)) * self.sad_gain)
        surprised_score = self._clip01(max(0.0, float(surprise - base_surprise)) * self.surprise_gain)

        candidates = {"Happy": happy_score, "Sad": sad_score, "Surprised": surprised_score}

        return {
            "calibrating": False,
            "scores": candidates,
        }


class AngerIndex:
    def __init__(self, baseline_frames=60, blink_window_seconds=15, movement_window=10,
                 weights=None, anger_gain=2.2):
        """
        anger_gain: multiplier applied to the final composite score before
        clipping to [0, 1]. Raise this if anger_index reads low even when
        you're clearly frowning/tensing; 1.0 = no amplification.
        """
        self.baseline_frames = baseline_frames
        self.blink_window_seconds = blink_window_seconds
        self.anger_gain = anger_gain
        self._baseline_buf = []
        self.baseline_brow_dist = None
        self.baseline_mouth_width = None
        self.baseline_blink_rate = None
        self._blink_calib_buf = []
        self._blink_timestamps = deque(maxlen=200)
        self._prev_blink_state = {"left": False, "right": False}
        self._movement_history = deque(maxlen=movement_window)
        self._prev_scores_vec = None
        self.weights = weights or {
            "eyebrow_lowering": 1.2, "inner_eyebrow_distance": 1.2, "eye_intensity": 0.8,
            "blink_rate_shift": 0.6, "mouth_width": 0.7, "lip_compression": 1.0,
            "mouth_opening": 0.4, "jaw_movement": 0.9, "head_up": 0.5, "facial_movement": 0.6,
        }
        total_w = sum(self.weights.values())
        self._norm_weights = {k: v / total_w for k, v in self.weights.items()}

    @staticmethod
    def _dist(landmarks, i, j):
        a, b = landmarks[i], landmarks[j]
        return float(np.hypot(a.x - b.x, a.y - b.y))

    def _face_scale(self, landmarks):
        return max(self._dist(landmarks, EYE_INNER_L, EYE_INNER_R), 1e-6)

    def update(self, face_landmarks, blendshape_scores, head_pitch_deg):
        scale = self._face_scale(face_landmarks)
        brow_dist_norm = self._dist(face_landmarks, INNER_BROW_L, INNER_BROW_R) / scale
        mouth_width_norm = self._dist(face_landmarks, MOUTH_CORNER_L, MOUTH_CORNER_R) / scale

        now = time.time()
        blink_event = self._update_blinks(blendshape_scores, now)

        if self.baseline_brow_dist is None:
            self._baseline_buf.append((brow_dist_norm, mouth_width_norm))
            if len(self._baseline_buf) >= self.baseline_frames:
                self.baseline_brow_dist = float(np.mean([b[0] for b in self._baseline_buf]))
                self.baseline_mouth_width = float(np.mean([b[1] for b in self._baseline_buf]))
            return {"calibrating": True, "frames_remaining": max(0, self.baseline_frames - len(self._baseline_buf))}

        sub_scores = {
            "eyebrow_lowering": self._clip01(np.mean([blendshape_scores.get("browDownLeft", 0), blendshape_scores.get("browDownRight", 0)])),
            "inner_eyebrow_distance": self._brow_distance_score(brow_dist_norm),
            "eye_intensity": self._clip01(max(
                np.mean([blendshape_scores.get("eyeWideLeft", 0), blendshape_scores.get("eyeWideRight", 0)]),
                np.mean([blendshape_scores.get("eyeSquintLeft", 0), blendshape_scores.get("eyeSquintRight", 0)]))),
            "blink_rate_shift": self._blink_rate_shift_score(now),
            "mouth_width": self._mouth_width_score(mouth_width_norm, blendshape_scores),
            "lip_compression": self._clip01(np.mean([
                blendshape_scores.get("mouthPressLeft", 0), blendshape_scores.get("mouthPressRight", 0),
                blendshape_scores.get("mouthRollLower", 0), blendshape_scores.get("mouthRollUpper", 0)])),
            "mouth_opening": self._clip01(blendshape_scores.get("jawOpen", 0)),
            "jaw_movement": self._clip01(np.mean([
                blendshape_scores.get("jawForward", 0), blendshape_scores.get("jawLeft", 0), blendshape_scores.get("jawRight", 0)])),
            "head_up": self._head_up_score(head_pitch_deg),
            "facial_movement": self._facial_movement_score(blendshape_scores),
        }
        raw_composite = sum(sub_scores[k] * self._norm_weights[k] for k in sub_scores)
        anger_index = float(np.clip(raw_composite * self.anger_gain, 0.0, 1.0))
        return {"calibrating": False, "anger_index": anger_index, "sub_scores": sub_scores, "blink_event": blink_event}

    @staticmethod
    def _clip01(v):
        return float(np.clip(v, 0.0, 1.0))

    def _brow_distance_score(self, brow_dist_norm):
        if self.baseline_brow_dist is None or self.baseline_brow_dist < 1e-6:
            return 0.0
        shrink_ratio = 1.0 - (brow_dist_norm / self.baseline_brow_dist)
        return self._clip01(shrink_ratio / 0.15)

    def _mouth_width_score(self, mouth_width_norm, blendshape_scores):
        if self.baseline_mouth_width is None or self.baseline_mouth_width < 1e-6:
            narrowing = 0.0
        else:
            narrow_ratio = 1.0 - (mouth_width_norm / self.baseline_mouth_width)
            narrowing = self._clip01(narrow_ratio / 0.15)
        stretch = self._clip01(np.mean([blendshape_scores.get("mouthStretchLeft", 0), blendshape_scores.get("mouthStretchRight", 0)]))
        return max(narrowing, stretch)

    def _update_blinks(self, blendshape_scores, now, threshold=0.5):
        event = None
        for side, cat in (("left", "eyeBlinkLeft"), ("right", "eyeBlinkRight")):
            is_closed = blendshape_scores.get(cat, 0.0) > threshold
            if is_closed and not self._prev_blink_state[side]:
                self._blink_timestamps.append(now)
                event = side
            self._prev_blink_state[side] = is_closed
        return event

    def _blink_rate_shift_score(self, now):
        cutoff = now - self.blink_window_seconds
        recent = [t for t in self._blink_timestamps if t >= cutoff]
        rate_per_min = len(recent) * (60.0 / self.blink_window_seconds)
        if self.baseline_blink_rate is None:
            self._blink_calib_buf.append(rate_per_min)
            if len(self._blink_calib_buf) >= 30:
                self.baseline_blink_rate = float(np.mean(self._blink_calib_buf))
            return 0.0
        deviation = abs(rate_per_min - self.baseline_blink_rate)
        return self._clip01(deviation / 15.0)

    def _head_up_score(self, pitch_deg):
        # NOTE: flip sign here if "chin up" reads negative pitch for you
        return self._clip01((pitch_deg - 12.0) / 15.0) if pitch_deg > 0 else 0.0

    def _facial_movement_score(self, blendshape_scores):
        vec = np.array([blendshape_scores.get(c, 0.0) for c in ANGER_BLENDSHAPE_CATEGORIES])
        if self._prev_scores_vec is not None:
            delta = float(np.linalg.norm(vec - self._prev_scores_vec))
            self._movement_history.append(delta)
        self._prev_scores_vec = vec
        if not self._movement_history:
            return 0.0
        avg_delta = float(np.mean(self._movement_history))
        return self._clip01(avg_delta / 0.3)


# ============================================================
# Stage 8 — Head pose (yaw/pitch/roll + velocity-based sudden move)
# ============================================================
class HeadPoseTracker:
    def __init__(self, velocity_threshold_deg_per_sec=180):
        self.velocity_threshold_deg_per_sec = velocity_threshold_deg_per_sec
        self._last_sample = None
        self.sudden_movement_count = 0

    @staticmethod
    def _matrix_to_euler(matrix_4x4):
        m = np.array(matrix_4x4).reshape(4, 4)
        r = m[:3, :3]
        sy = np.sqrt(r[0, 0] ** 2 + r[1, 0] ** 2)
        singular = sy < 1e-6
        if not singular:
            pitch = np.arctan2(-r[2, 0], sy)
            yaw = np.arctan2(r[1, 0], r[0, 0])
            roll = np.arctan2(r[2, 1], r[2, 2])
        else:
            pitch = np.arctan2(-r[2, 0], sy)
            yaw = 0
            roll = np.arctan2(-r[1, 2], r[1, 1])
        return np.degrees(yaw), np.degrees(pitch), np.degrees(roll)

    def update(self, transformation_matrix):
        yaw, pitch, roll = self._matrix_to_euler(transformation_matrix)
        now = time.time()
        velocity = 0.0
        sudden = False
        if self._last_sample is not None:
            t0, y0, p0, r0 = self._last_sample
            dt = now - t0
            if dt > 1e-4:
                velocity = max(abs(yaw - y0), abs(pitch - p0), abs(roll - r0)) / dt
                if velocity > self.velocity_threshold_deg_per_sec:
                    sudden = True
                    self.sudden_movement_count += 1
        self._last_sample = (now, yaw, pitch, roll)
        return {"yaw": yaw, "pitch": pitch, "roll": roll,
                "angular_velocity_deg_s": velocity, "sudden_movement": sudden,
                "sudden_movement_count_total": self.sudden_movement_count}


# ============================================================
# Stage 9 — Posture proxy (face bbox size/position)
# ============================================================
class PostureTracker:
    def __init__(self, baseline_frames=60, drift_threshold=0.15, area_change_threshold=0.25):
        self.baseline_frames = baseline_frames
        self.drift_threshold = drift_threshold
        self.area_change_threshold = area_change_threshold
        self._baseline_buf = []
        self.baseline_area = None
        self.baseline_center = None
        self._areas_smooth = deque(maxlen=10)

    @staticmethod
    def _bbox(landmarks):
        xs = [lm.x for lm in landmarks]; ys = [lm.y for lm in landmarks]
        x_min, x_max = min(xs), max(xs); y_min, y_max = min(ys), max(ys)
        area = (x_max - x_min) * (y_max - y_min)
        center = ((x_min + x_max) / 2.0, (y_min + y_max) / 2.0)
        return area, center

    def update(self, face_landmarks):
        area, center = self._bbox(face_landmarks)
        self._areas_smooth.append(area)
        smoothed_area = float(np.mean(self._areas_smooth))

        if self.baseline_area is None:
            self._baseline_buf.append((smoothed_area, center))
            if len(self._baseline_buf) >= self.baseline_frames:
                self.baseline_area = float(np.mean([b[0] for b in self._baseline_buf]))
                self.baseline_center = (float(np.mean([b[1][0] for b in self._baseline_buf])),
                                         float(np.mean([b[1][1] for b in self._baseline_buf])))
            return {"calibrating": True, "frames_remaining": max(0, self.baseline_frames - len(self._baseline_buf))}

        area_change = (smoothed_area - self.baseline_area) / self.baseline_area
        drift = float(np.hypot(center[0] - self.baseline_center[0], center[1] - self.baseline_center[1]))
        return {"calibrating": False, "area_change_pct": area_change, "center_drift": drift,
                "leaning_in": area_change > self.area_change_threshold,
                "leaning_back": area_change < -self.area_change_threshold,
                "off_center": drift > self.drift_threshold}


# ============================================================
# Main combined loop
# ============================================================
def matrix_to_pitch(matrix_4x4):
    m = np.array(matrix_4x4).reshape(4, 4)
    r = m[:3, :3]
    sy = np.sqrt(r[0, 0] ** 2 + r[1, 0] ** 2)
    pitch = np.arctan2(-r[2, 0], sy)
    return float(np.degrees(pitch))


def main():
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
    )
    landmarker = mp_vision.FaceLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (check CAP_DSHOW / camera index).")

    blink_detector = BlinkDetector()
    gaze_tracker = GazeTracker()
    blendshape_tracker = BlendshapeTracker()
    emotion_classifier = EmotionClassifier()
    anger_tracker = AngerIndex()
    head_pose_tracker = HeadPoseTracker()
    posture_tracker = PostureTracker()

    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Frame grab failed, stopping.")
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(frame_idx * (1000 / 30))
        result = landmarker.detect_for_video(mp_image, timestamp_ms)
        frame_idx += 1

        # ------------------------------------------------------------
        # Text is drawn in two columns so everything fits on screen:
        # left column = blink / gaze / posture / head pose
        # right column = blendshape tension + anger sub_scores
        # ------------------------------------------------------------
        left_x = 10
        right_x = 480
        y_left = 25
        y_right = 25
        line_h = 22
        small_scale = 0.5
        small_color = (255, 255, 255)

        if result.face_landmarks:
            lm = result.face_landmarks[0]

            # ---- Stage 5: blink ----
            blink_out = blink_detector.update(lm)
            cv2.putText(frame, f"EAR: {blink_out['ear']:.3f}  Blinks: {blink_out['blink_count']}",
                        (left_x, y_left), cv2.FONT_HERSHEY_SIMPLEX, small_scale, small_color, 1)
            y_left += line_h

            # ---- Stage 6: gaze ----
            gaze_out = gaze_tracker.update(lm)
            cv2.putText(frame,
                        f"Gaze x:{gaze_out['gaze_x']:.2f} y:{gaze_out['gaze_y']:.2f} "
                        f"dev:{gaze_out['deviation']:.2f} away:{gaze_out['looking_away']}",
                        (left_x, y_left), cv2.FONT_HERSHEY_SIMPLEX, small_scale, small_color, 1)
            y_left += line_h
            cv2.putText(frame,
                        f"Rapid gaze change: {gaze_out['rapid_change']} "
                        f"(total: {gaze_out['rapid_change_count_total']})",
                        (left_x, y_left), cv2.FONT_HERSHEY_SIMPLEX, small_scale, small_color, 1)
            y_left += line_h

            # ---- Stage 9: posture ----
            posture_out = posture_tracker.update(lm)
            if posture_out["calibrating"]:
                cv2.putText(frame, f"Posture calibrating... {posture_out['frames_remaining']}",
                            (left_x, y_left), cv2.FONT_HERSHEY_SIMPLEX, small_scale, (0, 255, 255), 1)
                y_left += line_h
            else:
                cv2.putText(frame,
                            f"Posture area%:{posture_out['area_change_pct']*100:.1f} "
                            f"drift:{posture_out['center_drift']:.3f}",
                            (left_x, y_left), cv2.FONT_HERSHEY_SIMPLEX, small_scale, small_color, 1)
                y_left += line_h
                cv2.putText(frame,
                            f"Leaning in:{posture_out['leaning_in']} back:{posture_out['leaning_back']} "
                            f"off_center:{posture_out['off_center']}",
                            (left_x, y_left), cv2.FONT_HERSHEY_SIMPLEX, small_scale, small_color, 1)
                y_left += line_h

            # ---- Stage 8: head pose ----
            pitch = None
            if result.facial_transformation_matrixes:
                pose = head_pose_tracker.update(result.facial_transformation_matrixes[0])
                pitch = pose["pitch"]
                cv2.putText(frame,
                            f"Yaw:{pose['yaw']:.1f} Pitch:{pose['pitch']:.1f} Roll:{pose['roll']:.1f}",
                            (left_x, y_left), cv2.FONT_HERSHEY_SIMPLEX, small_scale, small_color, 1)
                y_left += line_h
                cv2.putText(frame,
                            f"Ang.vel:{pose['angular_velocity_deg_s']:.1f} deg/s "
                            f"Sudden:{pose['sudden_movement']} (total:{pose['sudden_movement_count_total']})",
                            (left_x, y_left), cv2.FONT_HERSHEY_SIMPLEX, small_scale, small_color, 1)
                y_left += line_h

            # ---- Stage 7: blendshapes + anger index ----
            if result.face_blendshapes:
                bs = blendshape_tracker.update(result.face_blendshapes[0])
                tension = bs["tension_index"]

                # ---- Happy / Sad / Surprised candidate scores (60-frame neutral calib) ----
                emotion_out = emotion_classifier.update(bs["scores"])

                # ---- Angry candidate score (its own 60-frame neutral calib) ----
                anger_calibrating = True
                anger_frames_remaining = 0
                angry_score = 0.0
                sub_scores = None
                blink_event = None
                if pitch is not None:
                    out = anger_tracker.update(lm, bs["scores"], pitch)
                    anger_calibrating = out["calibrating"]
                    if not anger_calibrating:
                        angry_score = out["anger_index"]
                        sub_scores = out["sub_scores"]
                        blink_event = out["blink_event"]
                    else:
                        anger_frames_remaining = out["frames_remaining"]

                still_calibrating = emotion_out["calibrating"] or anger_calibrating
                if still_calibrating:
                    frames_left = max(
                        emotion_out.get("frames_remaining", 0),
                        anger_frames_remaining,
                    )
                    cv2.putText(frame, f"Stay neutral... {frames_left} frames left",
                                (left_x, y_left), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    y_left += line_h + 5
                else:
                    # combine all four candidates into one decision.
                    # Compare MARGIN above each emotion's own threshold
                    # (not raw score) since Angry's composite scale is not
                    # comparable to the raw Happy/Sad/Surprised deltas.
                    candidates = dict(emotion_out["scores"])
                    candidates["Angry"] = angry_score

                    margins = {
                        name: val - EMOTION_THRESHOLDS.get(name, 0.12)
                        for name, val in candidates.items()
                    }
                    best_label = max(margins, key=margins.get)
                    label = best_label if margins[best_label] > 0 else "Neutral"

                    label_color = {
                        "Happy": (0, 255, 0),
                        "Sad": (255, 0, 0),
                        "Surprised": (0, 255, 255),
                        "Angry": (0, 0, 255),
                        "Neutral": (200, 200, 200),
                    }.get(label, (255, 255, 255))
                    cv2.putText(frame, f"Emotion: {label}", (left_x, y_left),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, label_color, 2)
                    y_left += line_h + 10
                    for name, val in candidates.items():
                        cv2.putText(frame, f"  {name}: {val:.2f}", (left_x, y_left),
                                    cv2.FONT_HERSHEY_SIMPLEX, small_scale, small_color, 1)
                        y_left += line_h

                cv2.putText(frame, f"Tension: {tension:.2f}", (left_x, y_left),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2); y_left += line_h + 5

                if not still_calibrating:
                    cv2.putText(frame, f"Blink event: {blink_event}", (left_x, y_left),
                                cv2.FONT_HERSHEY_SIMPLEX, small_scale, small_color, 1)
                    y_left += line_h

                    # anger sub_scores breakdown, right column
                    cv2.putText(frame, "Anger sub_scores:", (right_x, y_right),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
                    y_right += line_h
                    for name, val in sub_scores.items():
                        cv2.putText(frame, f"{name}: {val:.2f}", (right_x, y_right),
                                    cv2.FONT_HERSHEY_SIMPLEX, small_scale, small_color, 1)
                        y_right += line_h
        else:
            cv2.putText(frame, "no face detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Gaming Behavior AI - All Vision Features", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
