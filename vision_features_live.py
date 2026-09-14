"""
Combined live overlay — Stages 4 through 9 running in one loop.

Run this from the project root (not from inside vision/) so the relative
model path and package imports both resolve:
    venv\\Scripts\\activate.bat
    python vision_features_live.py

This is meant as an integration smoke test once you've verified each stage
individually — not a replacement for verifying blink_test.py and each new
module on its own first.
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from vision.eye_features import BlinkDetector  # your existing Stage 5 class
from vision.gaze_features import GazeTracker
from vision.blendshape_features import BlendshapeTracker
from vision.head_pose_features import HeadPoseTracker
from vision.posture_features import PostureTracker

MODEL_PATH = "models/face_landmarker.task"


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

    blink_detector = BlinkDetector()  # adjust constructor args to match your Stage 5 class
    gaze_tracker = GazeTracker()
    blendshape_tracker = BlendshapeTracker()
    head_pose_tracker = HeadPoseTracker()
    posture_tracker = PostureTracker()

    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Frame grab failed, stopping.")
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(frame_idx * (1000 / 30))
        result = landmarker.detect_for_video(mp_image, timestamp_ms)
        frame_idx += 1

        y = 30
        if result.face_landmarks:
            lm = result.face_landmarks[0]

            blink_state = blink_detector.update(lm)  # adapt to your actual method name/signature
            cv2.putText(frame, f"blinks: {getattr(blink_state, 'blink_count', blink_state)}",
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2); y += 25

            gaze = gaze_tracker.update(lm)
            cv2.putText(frame, f"gaze dev: {gaze['deviation']:.2f} rapid: {gaze['rapid_change_count_total']}",
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2); y += 25

            posture = posture_tracker.update(lm, w, h)
            if posture["calibrating"]:
                cv2.putText(frame, f"posture calibrating ({posture['frames_remaining']})",
                            (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2); y += 25
            else:
                cv2.putText(frame, f"posture drift: {posture['center_drift']:.2f}",
                            (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2); y += 25

            if result.face_blendshapes:
                bs = blendshape_tracker.update(result.face_blendshapes[0])
                cv2.putText(frame, f"tension: {bs['tension_index']:.2f}",
                            (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2); y += 25

            if result.facial_transformation_matrixes:
                pose = head_pose_tracker.update(result.facial_transformation_matrixes[0])
                cv2.putText(frame, f"yaw:{pose['yaw']:.0f} pitch:{pose['pitch']:.0f} roll:{pose['roll']:.0f}",
                            (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2); y += 25
        else:
            cv2.putText(frame, "no face detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Vision Features (Stages 4-9)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
