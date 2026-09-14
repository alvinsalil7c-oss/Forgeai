"""
Stage 4: live face landmark detection.

Opens the webcam and draws all 478 face landmarks as dots on the
live video feed, with the iris points highlighted separately.

Press 'q' to quit.
"""

import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

MODEL_PATH = "models/face_landmarker.task"
CAMERA_INDEX = 0

# MediaPipe always reserves these index ranges for the iris points
# when using the Face Landmarker (which includes iris refinement).
LEFT_IRIS_RANGE = range(473, 478)
RIGHT_IRIS_RANGE = range(468, 473)


def draw_landmarks_on_frame(frame, face_landmarks):
    h, w = frame.shape[:2]

    for idx, lm in enumerate(face_landmarks):
        x_px = int(lm.x * w)
        y_px = int(lm.y * h)

        if idx in LEFT_IRIS_RANGE or idx in RIGHT_IRIS_RANGE:
            cv2.circle(frame, (x_px, y_px), 2, (0, 0, 255), -1)   # red = iris
        else:
            cv2.circle(frame, (x_px, y_px), 1, (0, 255, 0), -1)   # green = mesh


def main():
    base_options = mp_tasks.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=True,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_faces=1,
    )
    landmarker = mp_vision.FaceLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("ERROR: could not open camera.")
        return

    print("Live face landmark tracking started. Press 'q' to quit.")
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("WARNING: failed to read a frame. Stopping.")
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        timestamp_ms = int((time.time() - start_time) * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        if result.face_landmarks:
            draw_landmarks_on_frame(frame, result.face_landmarks[0])
            status_text = "Face detected"
            status_color = (0, 255, 0)
        else:
            status_text = "No face detected"
            status_color = (0, 0, 255)

        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, status_color, 2)

        cv2.imshow("Stage 4 - Face Landmarks (press 'q' to quit)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()