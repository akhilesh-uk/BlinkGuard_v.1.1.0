import cv2
import mediapipe as mp
import time
import math
import csv
from datetime import datetime

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/face_landmarker.task"

RECORDING_DURATION = 30  # seconds

OUTPUT_FILE = (
    "data/processed/ear_recording_"
    + datetime.now().strftime("%Y%m%d_%H%M%S")
    + ".csv"
)


# ============================================================
# MediaPipe setup
# ============================================================

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
# Distance calculation
# ============================================================

def distance(p1, p2):

    return math.sqrt(
        (p1.x - p2.x) ** 2 +
        (p1.y - p2.y) ** 2
    )


# ============================================================
# EAR calculation
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

    if horizontal == 0:
        return 0.0

    return (
        vertical_1 + vertical_2
    ) / (2.0 * horizontal)


# ============================================================
# Create output directory if necessary
# ============================================================

import os

os.makedirs("data/processed", exist_ok=True)


# ============================================================
# Open CSV file
# ============================================================

csv_file = open(
    OUTPUT_FILE,
    "w",
    newline=""
)

csv_writer = csv.writer(csv_file)

csv_writer.writerow([
    "timestamp",
    "elapsed_seconds",
    "left_ear",
    "right_ear",
    "average_ear",
    "face_detected"
])


# ============================================================
# Open webcam
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("ERROR: Could not open webcam.")

    csv_file.close()
    landmarker.close()

    exit()


print()
print("==========================================")
print("       BLINKGUARD EAR DATA RECORDER")
print("==========================================")
print()
print("Recording duration:", RECORDING_DURATION, "seconds")
print()
print("Follow these instructions:")
print()
print("1. First keep your eyes naturally open.")
print("2. Blink normally several times.")
print("3. Look normally at the camera.")
print("4. Around the middle, close your eyes")
print("   continuously for about 2-3 seconds.")
print("5. Continue blinking normally.")
print()
print("DO NOT deliberately exaggerate your blinks.")
print()
print("Press Q to stop early.")
print()
print("Starting in 3 seconds...")
time.sleep(3)


# ============================================================
# Recording
# ============================================================

recording_start = time.time()

sample_count = 0

last_ear = 0.0


while True:

    success, frame = cap.read()

    if not success:

        print("ERROR: Could not read webcam frame.")
        break


    # Mirror camera
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
    current_time = time.time()

    elapsed = current_time - recording_start

    timestamp_ms = int(
        elapsed * 1000
    )


    # Run Face Landmarker
    result = landmarker.detect_for_video(
        mp_image,
        timestamp_ms
    )


    # ========================================================
    # Face detected
    # ========================================================

    if result.face_landmarks:

        face_landmarks = result.face_landmarks[0]

        left_ear = calculate_ear(
            face_landmarks,
            LEFT_EYE
        )

        right_ear = calculate_ear(
            face_landmarks,
            RIGHT_EYE
        )

        average_ear = (
            left_ear + right_ear
        ) / 2.0

        last_ear = average_ear

        face_detected = 1

    else:

        left_ear = 0.0
        right_ear = 0.0
        average_ear = 0.0

        face_detected = 0


    # ========================================================
    # Save data
    # ========================================================

    csv_writer.writerow([
        datetime.now().isoformat(),
        round(elapsed, 4),
        round(left_ear, 6),
        round(right_ear, 6),
        round(average_ear, 6),
        face_detected
    ])

    sample_count += 1


    # ========================================================
    # Display information
    # ========================================================

    remaining = max(
        0,
        RECORDING_DURATION - elapsed
    )

    cv2.putText(
        frame,
        f"EAR: {last_ear:.3f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Time: {elapsed:.1f}s / {RECORDING_DURATION}s",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Samples: {sample_count}",
        (20, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        "Recording...",
        (20, 145),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )


    # ========================================================
    # Show webcam
    # ========================================================

    cv2.imshow(
        "BlinkGuard - EAR Data Recorder",
        frame
    )


    # ========================================================
    # Stop conditions
    # ========================================================

    if elapsed >= RECORDING_DURATION:
        break

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ============================================================
# Cleanup
# ============================================================

cap.release()

cv2.destroyAllWindows()

csv_file.close()

landmarker.close()


print()
print("==========================================")
print("          RECORDING COMPLETED")
print("==========================================")
print()
print("Samples collected:", sample_count)
print()
print("Saved to:")
print(OUTPUT_FILE)
print()