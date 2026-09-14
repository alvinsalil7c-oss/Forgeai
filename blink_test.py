"""
Stage 5: live blink detection test.

Shows the webcam feed with the current EAR value, running blink
count, and blinks-per-minute, updated in real time.

Press 'q' to quit.
"""

import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

from vision.eye_features import compute_avg_ear, BlinkDetector

MODEL_PATH = "models/face_landmarker.task"
CAMERA_INDEX = 0


def main():
    base_options = mp_tasks.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=True,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_faces=1,
    )
    landmarker = mp_vision.FaceLandmarker.create_from_options(options)
    blink_detector = BlinkDetector()

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("ERROR: could not open camera.")
        return

    print("Blink test started. Blink naturally, press 'q' to quit.")
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("WARNING: failed to read a frame. Stopping.")
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        now = time.time()
        timestamp_ms = int((now - start_time) * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        if result.face_landmarks:
            landmarks = result.face_landmarks[0]
            avg_ear = compute_avg_ear(landmarks)
            blink_detector.update(avg_ear, now - start_time)

            elapsed_minutes = max((now - start_time) / 60.0, 1e-6)
            blink_rate = blink_detector.blink_count / elapsed_minutes

            cv2.putText(frame, f"EAR: {avg_ear:.3f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Blinks: {blink_detector.blink_count}", (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Blink rate: {blink_rate:.1f}/min", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No face detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("Stage 5 - Blink Detection (press 'q' to quit)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nSession summary: {blink_detector.blink_count} blinks detected.")


if __name__ == "__main__":
    main()