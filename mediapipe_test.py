"""
Stage 3: MediaPipe test.

Loads the Face Landmarker model and runs it once on a single frame
captured from the webcam, just to confirm the model initializes
correctly and can detect a face.

Does NOT do live video or draw anything yet - that's Stage 4.
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

MODEL_PATH = "models/face_landmarker.task"


def main():
    # 1. Set up the Face Landmarker in IMAGE mode (single-frame, not streaming)
    base_options = mp_tasks.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=True,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_faces=1,
    )
    landmarker = mp_vision.FaceLandmarker.create_from_options(options)
    print("Face Landmarker model loaded successfully.")

    # 2. Grab one frame from the webcam
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("ERROR: could not open camera.")
        return

    print("Capturing a frame - look at the camera...")
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("ERROR: failed to capture a frame.")
        return

    # 3. Run detection on that one frame
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result = landmarker.detect(mp_image)

    # 4. Report what was found
    num_faces = len(result.face_landmarks)
    print(f"\nFaces detected: {num_faces}")

    if num_faces > 0:
        num_landmarks = len(result.face_landmarks[0])
        print(f"Landmarks on first face: {num_landmarks}")
        if result.face_blendshapes:
            print(f"Blendshapes available: {len(result.face_blendshapes[0])}")
        print("\nMediaPipe is working correctly.")
    else:
        print("\nNo face detected. Make sure you were facing the camera")
        print("with reasonable lighting when the frame was captured.")


if __name__ == "__main__":
    main()