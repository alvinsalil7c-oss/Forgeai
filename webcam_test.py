"""
Stage 2: webcam test.

Opens the default webcam and shows a live preview window.
Press 'q' to quit.

This does NOT do any face detection yet - just confirms OpenCV
can access your camera. MediaPipe comes in Stage 3.
"""

import cv2

CAMERA_INDEX = 0  # try 1 if this doesn't find your webcam

def main():
    # CAP_DSHOW avoids the slow/hanging camera-open issue that OpenCV's
    # default backend (MSMF) sometimes has on Windows.
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print(f"ERROR: could not open camera at index {CAMERA_INDEX}.")
        print("Try changing CAMERA_INDEX to 1 near the top of this file,")
        print("or check that no other app (Zoom, Teams, etc.) is using the camera.")
        return

    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"Camera opened successfully. Resolution: {int(width)}x{int(height)}")
    print("Press 'q' in the preview window to quit.")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("WARNING: failed to read a frame from the camera. Stopping.")
            break

        # Mirror the frame horizontally so it feels like looking in a mirror -
        # purely cosmetic, doesn't affect anything downstream.
        frame = cv2.flip(frame, 1)

        cv2.imshow("Webcam Test - press 'q' to quit", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()