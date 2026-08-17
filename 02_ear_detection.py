import cv2
import mediapipe as mp
import time
import math

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# MediaPipe setup
# ============================================================

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
# Eye landmark indices
# ============================================================

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


# ============================================================
# Calculate Euclidean distance
# ============================================================

def distance(p1, p2):

    return math.sqrt(
        (p1.x - p2.x) ** 2 +
        (p1.y - p2.y) ** 2
    )


# ============================================================
# Calculate EAR
# ============================================================

def calculate_ear(landmarks, eye_indices):

    p1 = landmarks[eye_indices[0]]
    p2 = landmarks[eye_indices[1]]
    p3 = landmarks[eye_indices[2]]
    p4 = landmarks[eye_indices[3]]
    p5 = landmarks[eye_indices[4]]
    p6 = landmarks[eye_indices[5]]

    vertical_1 = distance(p2, p6)
    vertical_2 = distance(p3, p5)

    horizontal = distance(p1, p4)

    ear = (
        vertical_1 + vertical_2
    ) / (2.0 * horizontal)

    return ear


# ============================================================
# Webcam
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("ERROR: Could not open webcam.")

    landmarker.close()
    exit()


print("BlinkGuard EAR test started.")
print("Blink several times and observe the EAR values.")
print("Press Q to quit.")


start_time = time.time()


# ============================================================
# Main loop
# ============================================================

while True:

    success, frame = cap.read()

    if not success:

        print("ERROR: Could not read webcam frame.")
        break


    # Mirror the camera
    frame = cv2.flip(frame, 1)


    # BGR → RGB
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # MediaPipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )


    # Timestamp
    timestamp_ms = int(
        (time.time() - start_time) * 1000
    )


    # Run MediaPipe
    result = landmarker.detect_for_video(
        mp_image,
        timestamp_ms
    )


    # ========================================================
    # Process face
    # ========================================================

    if result.face_landmarks:

        face_landmarks = result.face_landmarks[0]


        # Calculate EAR
        left_ear = calculate_ear(
            face_landmarks,
            LEFT_EYE
        )

        right_ear = calculate_ear(
            face_landmarks,
            RIGHT_EYE
        )


        # Average EAR
        average_ear = (
            left_ear + right_ear
        ) / 2.0


        # ====================================================
        # Display EAR values
        # ====================================================

        cv2.putText(
            frame,
            f"Left EAR: {left_ear:.3f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Right EAR: {right_ear:.3f}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Average EAR: {average_ear:.3f}",
            (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )


        # ====================================================
        # Draw eye landmarks
        # ====================================================

        height, width, _ = frame.shape

        for index in LEFT_EYE:

            landmark = face_landmarks[index]

            x = int(landmark.x * width)
            y = int(landmark.y * height)

            cv2.circle(
                frame,
                (x, y),
                4,
                (0, 255, 0),
                -1
            )


        for index in RIGHT_EYE:

            landmark = face_landmarks[index]

            x = int(landmark.x * width)
            y = int(landmark.y * height)

            cv2.circle(
                frame,
                (x, y),
                4,
                (0, 255, 0),
                -1
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
        "BlinkGuard - EAR Test",
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

print("EAR test completed.")