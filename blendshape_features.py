"""
Stage 7 — Facial features via blendshapes, PLUS a composite anger/stress
index (AngerIndex) that layers in raw landmark distances, blink rate, and
head pitch on top of the blendshapes.

FaceLandmarker gives you 52 blendshape scores (0-1) per frame when
`output_face_blendshapes=True` is set on FaceLandmarkerOptions. BlendshapeTracker
extracts/aggregates the subset that's informative for a gaming-behavior signal
(frustration, tension, surprise) instead of tracking all 52.

Chosen subset and why:
    browDownLeft/Right     -> frowning / tension / frustration
    browInnerUp            -> concern / surprise
    jawOpen                -> gasps, shouting, surprise
    mouthPressLeft/Right   -> lip pressing = tension/concentration
    eyeSquintLeft/Right    -> squinting = focus or strain
    mouthSmileLeft/Right   -> positive affect (contrast signal)

AngerIndex is a separate, more involved tracker underneath BlendshapeTracker.
It needs THREE outputs from the same FaceLandmarker result at once —
face_landmarks, face_blendshapes, and facial_transformation_matrixes — all of
which can be enabled together (see the combined __main__ block at the bottom).

Two of AngerIndex's ten sub-signals (inner eyebrow distance, mouth width) are
raw landmark distances rather than blendshape categories — these are
scale/face-shape dependent, so they self-calibrate against YOUR OWN neutral
face over the first `baseline_frames` frames, same pattern as
posture_features.py's PostureTracker. Don't make an angry face during
calibration.

NOTE on head pitch sign: `head_goes_up` assumes pitch increases as the chin
lifts, matching head_pose_features.py's decomposition. If you tested that
module and found pitch behaves the opposite way for you, flip the sign in
`_head_up_score` below (there's a one-line comment marking it).
"""

import time
from collections import deque
import numpy as np

TRACKED_CATEGORIES = [
    "browDownLeft", "browDownRight",
    "browInnerUp",
    "jawOpen",
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

# Landmark indices used for the two self-calibrated distance measures.
INNER_BROW_L, INNER_BROW_R = 55, 285   # inner eyebrow tips
MOUTH_CORNER_L, MOUTH_CORNER_R = 61, 291
EYE_INNER_L, EYE_INNER_R = 133, 362     # used to normalize distances by face scale


class BlendshapeTracker:
    def __init__(self, history_seconds=5, fps_estimate=30, tension_gain=1.8):
        """
        tension_gain: multiplier applied to the tension index before
        clipping to [0, 1]. Raise this if subtle frowning/lip-pressing
        isn't showing up strongly enough; 1.0 = no amplification.
        """
        maxlen = history_seconds * fps_estimate
        self._history = {cat: deque(maxlen=maxlen) for cat in TRACKED_CATEGORIES}
        self.tension_gain = tension_gain

    @staticmethod
    def _scores_dict(face_blendshapes_result):
        """
        face_blendshapes_result: result.face_blendshapes[0], a list of
        Category objects each with .category_name and .score.
        """
        return {c.category_name: c.score for c in face_blendshapes_result}

    def update(self, face_blendshapes_result):
        scores = self._scores_dict(face_blendshapes_result)
        frame_out = {}
        for cat in TRACKED_CATEGORIES:
            val = scores.get(cat, 0.0)
            self._history[cat].append(val)
            frame_out[cat] = val

        tension_raw = float(np.mean([
            frame_out["browDownLeft"], frame_out["browDownRight"],
            frame_out["mouthPressLeft"], frame_out["mouthPressRight"],
        ]))
        tension = float(np.clip(tension_raw * self.tension_gain, 0.0, 1.0))
        surprise = float(np.mean([frame_out["browInnerUp"], frame_out["jawOpen"]]))
        positive_affect = float(np.mean([frame_out["mouthSmileLeft"], frame_out["mouthSmileRight"]]))
        squint = float(np.mean([frame_out["eyeSquintLeft"], frame_out["eyeSquintRight"]]))

        return {
            "raw": frame_out,
            "tension_index": tension,
            "surprise_index": surprise,
            "positive_affect_index": positive_affect,
            "squint_index": squint,
        }

    def rolling_mean(self, category, seconds=None):
        vals = list(self._history[category])
        if seconds is not None:
            # deque has no timestamps; caller should size history_seconds
            # appropriately at construction if using this loosely
            pass
        return float(np.mean(vals)) if vals else 0.0


class AngerIndex:
    def __init__(self, baseline_frames=60, blink_window_seconds=15,
                 movement_window=10, weights=None):
        self.baseline_frames = baseline_frames
        self.blink_window_seconds = blink_window_seconds

        self._baseline_buf = []  # (brow_dist_norm, mouth_width_norm)
        self.baseline_brow_dist = None
        self.baseline_mouth_width = None
        self.baseline_blink_rate = None
        self._blink_calib_buf = []

        self._blink_timestamps = deque(maxlen=200)
        self._prev_blink_state = {"left": False, "right": False}

        self._movement_history = deque(maxlen=movement_window)
        self._prev_scores_vec = None

        # relative importance of each sub-signal in the final composite;
        # tune freely, they're re-normalized to sum to 1 automatically.
        self.weights = weights or {
            "eyebrow_lowering": 1.2,
            "inner_eyebrow_distance": 1.2,
            "eye_intensity": 0.8,
            "blink_rate_shift": 0.6,
            "mouth_width": 0.7,
            "lip_compression": 1.0,
            "mouth_opening": 0.4,
            "jaw_movement": 0.9,
            "head_up": 0.5,
            "facial_movement": 0.6,
        }
        total_w = sum(self.weights.values())
        self._norm_weights = {k: v / total_w for k, v in self.weights.items()}

    # ---------- landmark-based helpers ----------

    @staticmethod
    def _dist(landmarks, i, j):
        a, b = landmarks[i], landmarks[j]
        return float(np.hypot(a.x - b.x, a.y - b.y))

    def _face_scale(self, landmarks):
        # normalize by inter-eye-corner distance so results don't depend
        # on how close the face is to the camera
        return max(self._dist(landmarks, EYE_INNER_L, EYE_INNER_R), 1e-6)

    # ---------- per-frame update ----------

    def update(self, face_landmarks, blendshape_scores, head_pitch_deg):
        """
        face_landmarks: result.face_landmarks[0]
        blendshape_scores: dict of {category_name: score}, e.g. from
            BlendshapeTracker._scores_dict(result.face_blendshapes[0])
        head_pitch_deg: pitch value from HeadPoseTracker.update() (or your
            own decomposition of result.facial_transformation_matrixes[0])
        """
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
            return {"calibrating": True,
                    "frames_remaining": max(0, self.baseline_frames - len(self._baseline_buf))}

        sub_scores = {
            "eyebrow_lowering": self._clip01(np.mean([
                blendshape_scores.get("browDownLeft", 0.0),
                blendshape_scores.get("browDownRight", 0.0),
            ])),
            "inner_eyebrow_distance": self._brow_distance_score(brow_dist_norm),
            "eye_intensity": self._clip01(max(
                np.mean([blendshape_scores.get("eyeWideLeft", 0.0),
                         blendshape_scores.get("eyeWideRight", 0.0)]),
                np.mean([blendshape_scores.get("eyeSquintLeft", 0.0),
                         blendshape_scores.get("eyeSquintRight", 0.0)]),
            )),
            "blink_rate_shift": self._blink_rate_shift_score(now),
            "mouth_width": self._mouth_width_score(mouth_width_norm, blendshape_scores),
            "lip_compression": self._clip01(np.mean([
                blendshape_scores.get("mouthPressLeft", 0.0),
                blendshape_scores.get("mouthPressRight", 0.0),
                blendshape_scores.get("mouthRollLower", 0.0),
                blendshape_scores.get("mouthRollUpper", 0.0),
            ])),
            "mouth_opening": self._clip01(blendshape_scores.get("jawOpen", 0.0)),
            "jaw_movement": self._clip01(np.mean([
                blendshape_scores.get("jawForward", 0.0),
                blendshape_scores.get("jawLeft", 0.0),
                blendshape_scores.get("jawRight", 0.0),
            ])),
            "head_up": self._head_up_score(head_pitch_deg),
            "facial_movement": self._facial_movement_score(blendshape_scores),
        }

        anger_index = float(np.clip(
            sum(sub_scores[k] * self._norm_weights[k] for k in sub_scores), 0.0, 1.0
        ))

        return {
            "calibrating": False,
            "anger_index": anger_index,
            "sub_scores": sub_scores,
            "blink_event": blink_event,
        }

    # ---------- individual sub-signal scorers ----------

    @staticmethod
    def _clip01(v):
        return float(np.clip(v, 0.0, 1.0))

    def _brow_distance_score(self, brow_dist_norm):
        # brows pulled TOGETHER (furrowed) = distance shrinks below baseline
        if self.baseline_brow_dist is None or self.baseline_brow_dist < 1e-6:
            return 0.0
        shrink_ratio = 1.0 - (brow_dist_norm / self.baseline_brow_dist)
        # shrink_ratio ~0 at baseline, grows as brows pull in; scale so a
        # ~15% reduction maps to a strong (1.0) score — adjust divisor to taste
        return self._clip01(shrink_ratio / 0.15)

    def _mouth_width_score(self, mouth_width_norm, blendshape_scores):
        if self.baseline_mouth_width is None or self.baseline_mouth_width < 1e-6:
            narrowing = 0.0
        else:
            narrow_ratio = 1.0 - (mouth_width_norm / self.baseline_mouth_width)
            narrowing = self._clip01(narrow_ratio / 0.15)
        stretch = self._clip01(np.mean([
            blendshape_scores.get("mouthStretchLeft", 0.0),
            blendshape_scores.get("mouthStretchRight", 0.0),
        ]))
        # either a tightened (narrowed) mouth OR a tense horizontal stretch
        # counts — they're different presentations of the same tension
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
            if len(self._blink_calib_buf) >= 30:  # ~30 samples of rate estimate
                self.baseline_blink_rate = float(np.mean(self._blink_calib_buf))
            return 0.0

        deviation = abs(rate_per_min - self.baseline_blink_rate)
        # a swing of ~15 blinks/min from your own baseline reads as strong
        return self._clip01(deviation / 15.0)

    def _head_up_score(self, pitch_deg):
        # NOTE: verify this sign matches what you saw testing
        # head_pose_features.py — flip if "chin up" showed negative pitch
        # for you. Threshold: >12 degrees up counts as a deliberate
        # head-up/defiant posture rather than normal micro-movement.
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
        # scale: a raw per-frame L2 delta of ~0.3 across these categories
        # reads as high overall facial activity
        return self._clip01(avg_delta / 0.3)


if __name__ == "__main__":
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
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
    )
    landmarker = mp_vision.FaceLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    bs_tracker = BlendshapeTracker()
    anger_tracker = AngerIndex()
    frame_idx = 0

    def matrix_to_pitch(matrix_4x4):
        m = np.array(matrix_4x4).reshape(4, 4)
        r = m[:3, :3]
        sy = np.sqrt(r[0, 0] ** 2 + r[1, 0] ** 2)
        pitch = np.arctan2(-r[2, 0], sy)
        return float(np.degrees(pitch))

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

        y = 30
        if result.face_blendshapes:
            feats = bs_tracker.update(result.face_blendshapes[0])
            cv2.putText(frame, f"tension: {feats['tension_index']:.2f}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2); y += 28
            cv2.putText(frame, f"surprise: {feats['surprise_index']:.2f}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2); y += 28
            cv2.putText(frame, f"positive: {feats['positive_affect_index']:.2f}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2); y += 34

        if result.face_landmarks and result.face_blendshapes and result.facial_transformation_matrixes:
            scores = bs_tracker._scores_dict(result.face_blendshapes[0])
            pitch = matrix_to_pitch(result.facial_transformation_matrixes[0])
            out = anger_tracker.update(result.face_landmarks[0], scores, pitch)

            if out["calibrating"]:
                cv2.putText(frame, f"anger calibrating (stay neutral)... {out['frames_remaining']}",
                            (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            else:
                color = (0, 0, 255) if out["anger_index"] > 0.5 else (0, 255, 0)
                cv2.putText(frame, f"anger_index: {out['anger_index']:.2f}", (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2); y += 26
                for k, v in out["sub_scores"].items():
                    cv2.putText(frame, f"{k}: {v:.2f}", (10, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
                    y += 18

        cv2.imshow("Blendshape + Anger Test", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
