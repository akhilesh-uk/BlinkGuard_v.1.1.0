import cv2
import mediapipe as mp
import time


# ============================================================
# MediaPipe setup
# ============================================================

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


MODEL_PATH = "models/face_landmarker.task"


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


# ============================================================
# Webcam
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    landmarker.close()
    exit()


print("BlinkGuard eye-landmark test started.")
print("Press Q to quit.")


start_time = time.time()


# ============================================================
# Eye landmark indices
# ============================================================

# MediaPipe Face Mesh / Face Landmarker eye landmark indices.
# We will use these for the first visualization test.

LEFT_EYE = [
    33, 160, 158, 133, 153, 144
]

RIGHT_EYE = [
    362, 385, 387, 263, 373, 380
]


# ============================================================
# Main loop
# ============================================================

while True:

    success, frame = cap.read()

    if not success:
        print("ERROR: Could not read webcam frame.")
        break

    # Flip image so it behaves like a mirror.
    frame = cv2.flip(frame, 1)

    # Convert BGR → RGB
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # Create MediaPipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # Timestamp
    timestamp_ms = int(
        (time.time() - start_time) * 1000
    )

    # Detect landmarks
    result = landmarker.detect_for_video(
        mp_image,
        timestamp_ms
    )


    # ========================================================
    # Process detected face
    # ========================================================

    if result.face_landmarks:

        face_landmarks = result.face_landmarks[0]

        height, width, _ = frame.shape


        # ----------------------------------------------------
        # Draw left eye
        # ----------------------------------------------------

        for index in LEFT_EYE:

            landmark = face_landmarks[index]

            x = int(landmark.x * width)
            y = int(landmark.y * height)

            cv2.circle(
                frame,
                (x, y),
                5,
                (0, 255, 0),
                -1
            )


        # ----------------------------------------------------
        # Draw right eye
        # ----------------------------------------------------

        for index in RIGHT_EYE:

            landmark = face_landmarks[index]

            x = int(landmark.x * width)
            y = int(landmark.y * height)

            cv2.circle(
                frame,
                (x, y),
                5,
                (0, 255, 0),
                -1
            )


        cv2.putText(
            frame,
            "EYE LANDMARKS DETECTED",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    else:

        cv2.putText(
            frame,
            "NO FACE DETECTED",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )


    # ========================================================
    # Display
    # ========================================================

    cv2.imshow(
        "BlinkGuard - Eye Landmarks",
        frame
    )


    # Q → quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ============================================================
# Cleanup
# ============================================================

cap.release()
cv2.destroyAllWindows()
landmarker.close()

print("Eye landmark test completed.")