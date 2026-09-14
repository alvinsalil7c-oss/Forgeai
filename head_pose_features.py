"""
Stage 8 — Head pose / movement.

FaceLandmarker can output a 4x4 facial transformation matrix per face when
`output_facial_transformation_matrixes=True` is set. That matrix already
encodes head rotation relative to the camera, so we decompose it into
yaw/pitch/roll instead of hand-rolling solvePnP.
"""

import time
from collections import deque
import numpy as np


class HeadPoseTracker:
    def __init__(self, velocity_threshold_deg_per_sec=180):
        """
        velocity_threshold_deg_per_sec: angular velocity (max of yaw/pitch/
            roll rate of change, in degrees/second) above which a frame
            counts as a "sudden head movement." This looks at instantaneous
            speed between consecutive frames rather than total displacement
            over a window, so a slow-but-large turn won't falsely trigger,
            and a small-but-fast flick will.
        """
        self.velocity_threshold_deg_per_sec = velocity_threshold_deg_per_sec
        self._last_sample = None  # (timestamp, yaw, pitch, roll)
        self.sudden_movement_count = 0

    @staticmethod
    def _matrix_to_euler(matrix_4x4):
        """
        matrix_4x4: row-major list/array from
        result.facial_transformation_matrixes[0]. Returns yaw, pitch, roll
        in degrees.
        """
        m = np.array(matrix_4x4).reshape(4, 4)
        r = m[:3, :3]

        # standard rotation-matrix -> euler (yaw-pitch-roll) decomposition
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
            if dt > 1e-4:  # guard against duplicate/zero-time frames
                d_yaw = abs(yaw - y0)
                d_pitch = abs(pitch - p0)
                d_roll = abs(roll - r0)
                velocity = max(d_yaw, d_pitch, d_roll) / dt
                if velocity > self.velocity_threshold_deg_per_sec:
                    sudden = True
                    self.sudden_movement_count += 1

        self._last_sample = (now, yaw, pitch, roll)

        return {
            "yaw": yaw,
            "pitch": pitch,
            "roll": roll,
            "angular_velocity_deg_s": velocity,
            "sudden_movement": sudden,
            "sudden_movement_count_total": self.sudden_movement_count,
        }

    def reset(self):
        self._last_sample = None
        self.sudden_movement_count = 0


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
        output_facial_transformation_matrixes=True,
    )
    landmarker = mp_vision.FaceLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    tracker = HeadPoseTracker()
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

        if result.facial_transformation_matrixes:
            feats = tracker.update(result.facial_transformation_matrixes[0])
            color = (0, 0, 255) if feats["sudden_movement"] else (0, 255, 0)
            cv2.putText(frame, f"yaw:{feats['yaw']:.1f} pitch:{feats['pitch']:.1f} roll:{feats['roll']:.1f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(frame, f"velocity: {feats['angular_velocity_deg_s']:.0f} deg/s",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame, f"sudden moves: {feats['sudden_movement_count_total']}",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.imshow("Head Pose Test", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
