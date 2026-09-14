"""
Stage 9 — Posture (lightweight proxy, no second model).

Per the decision log: instead of running a separate MediaPipe Pose model,
we use the face's own bounding box (derived from the 478 face landmarks
we already have) as a cheap proxy:

    - box AREA shrinking/growing over time  -> leaning back / leaning in
    - box CENTER drifting off the frame center -> slouching sideways /
      off-center posture
    - box vertical position rising          -> hunching forward toward
      the camera (common "tilt-in when focused/tense" behavior)

This is a proxy, not a real posture estimate — it only tells you the head
moved closer/farther/off-center, which is what you have signal for without
paying for a second model.
"""

from collections import deque
import numpy as np


class PostureTracker:
    def __init__(self, baseline_frames=60, drift_threshold=0.15,
                 area_change_threshold=0.25):
        """
        baseline_frames: number of initial frames used to establish the
            "neutral" bounding-box size/position (simple running baseline,
            replaced properly in Stage 14-16 like EAR_THRESHOLD is).
        drift_threshold: normalized (0-1 of frame width/height) center
            drift from baseline that counts as "off-center."
        area_change_threshold: fractional change in box area from baseline
            that counts as "leaning in" (positive) or "leaning back"
            (negative).
        """
        self.baseline_frames = baseline_frames
        self.drift_threshold = drift_threshold
        self.area_change_threshold = area_change_threshold

        self._baseline_buf = []
        self.baseline_area = None
        self.baseline_center = None
        self._areas_smooth = deque(maxlen=10)

    @staticmethod
    def _bbox_from_landmarks(landmarks, frame_w, frame_h):
        xs = [lm.x for lm in landmarks]
        ys = [lm.y for lm in landmarks]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        area = (x_max - x_min) * (y_max - y_min)  # normalized 0-1 area
        center = ((x_min + x_max) / 2.0, (y_min + y_max) / 2.0)
        return area, center

    def update(self, face_landmarks, frame_w, frame_h):
        area, center = self._bbox_from_landmarks(face_landmarks, frame_w, frame_h)
        self._areas_smooth.append(area)
        smoothed_area = float(np.mean(self._areas_smooth))

        if self.baseline_area is None:
            self._baseline_buf.append((smoothed_area, center))
            if len(self._baseline_buf) >= self.baseline_frames:
                self.baseline_area = float(np.mean([b[0] for b in self._baseline_buf]))
                self.baseline_center = (
                    float(np.mean([b[1][0] for b in self._baseline_buf])),
                    float(np.mean([b[1][1] for b in self._baseline_buf])),
                )
            return {
                "calibrating": True,
                "frames_remaining": max(0, self.baseline_frames - len(self._baseline_buf)),
            }

        area_change = (smoothed_area - self.baseline_area) / self.baseline_area
        drift = float(np.hypot(center[0] - self.baseline_center[0],
                                center[1] - self.baseline_center[1]))

        leaning_in = area_change > self.area_change_threshold
        leaning_back = area_change < -self.area_change_threshold
        off_center = drift > self.drift_threshold

        return {
            "calibrating": False,
            "area": smoothed_area,
            "area_change_pct": area_change,
            "center_drift": drift,
            "leaning_in": leaning_in,
            "leaning_back": leaning_back,
            "off_center": off_center,
        }

    def reset_baseline(self):
        self._baseline_buf.clear()
        self.baseline_area = None
        self.baseline_center = None


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
    )
    landmarker = mp_vision.FaceLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    tracker = PostureTracker()
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(frame_idx * (1000 / 30))
        result = landmarker.detect_for_video(mp_image, timestamp_ms)
        frame_idx += 1

        if result.face_landmarks:
            feats = tracker.update(result.face_landmarks[0], w, h)
            if feats["calibrating"]:
                cv2.putText(frame, f"calibrating... {feats['frames_remaining']}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                status = []
                if feats["leaning_in"]:
                    status.append("LEANING IN")
                if feats["leaning_back"]:
                    status.append("LEANING BACK")
                if feats["off_center"]:
                    status.append("OFF-CENTER")
                cv2.putText(frame, " | ".join(status) or "neutral", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, f"area chg: {feats['area_change_pct']*100:.1f}%",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.imshow("Posture Test", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
