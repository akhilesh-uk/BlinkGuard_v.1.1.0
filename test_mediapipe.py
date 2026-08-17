import cv2
import mediapipe as mp
import time

# MediaPipe Tasks
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# --------------------------------------------------
# 1. MODEL PATH
# --------------------------------------------------

MODEL_PATH = "models/face_landmarker.task"


# --------------------------------------------------
# 2. CREATE FACE LANDMARKER
# --------------------------------------------------

base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)

options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False
)

landmarker = vision.FaceLandmarker.create_from_options(options)


# --------------------------------------------------
# 3. OPEN WEBCAM
# --------------------------------------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    landmarker.close()
    exit()

print("Webcam opened successfully.")
print("Move your face in front of the camera.")
print("Press Q to quit.")


# --------------------------------------------------
# 4. PROCESS WEBCAM
# --------------------------------------------------

start_time = time.time()

while True:

    success, frame = cap.read()

    if not success:
        print("ERROR: Could not read frame.")
        break

    # OpenCV uses BGR.
    # MediaPipe expects RGB.
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert OpenCV image to MediaPipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # Timestamp in milliseconds
    timestamp_ms = int((time.time() - start_time) * 1000)

    # Run face landmark detection
    result = landmarker.detect_for_video(
        mp_image,
        timestamp_ms
    )

    # --------------------------------------------------
    # 5. DRAW LANDMARKS
    # --------------------------------------------------

    if result.face_landmarks:

        for face_landmarks in result.face_landmarks:

            height, width, _ = frame.shape

            # Draw every facial landmark
            for landmark in face_landmarks:

                x = int(landmark.x * width)
                y = int(landmark.y * height)

                if 0 <= x < width and 0 <= y < height:

                    cv2.circle(
                        frame,
                        (x, y),
                        1,
                        (0, 255, 0),
                        -1
                    )

        # Status
        cv2.putText(
            frame,
            "FACE DETECTED",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

    else:

        cv2.putText(
            frame,
            "NO FACE DETECTED",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

    # --------------------------------------------------
    # 6. DISPLAY
    # --------------------------------------------------

    cv2.imshow(
        "BlinkGuard - MediaPipe Face Landmarks Test",
        frame
    )

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# --------------------------------------------------
# 7. CLEANUP
# --------------------------------------------------

cap.release()
cv2.destroyAllWindows()
landmarker.close()

print("Test completed.")