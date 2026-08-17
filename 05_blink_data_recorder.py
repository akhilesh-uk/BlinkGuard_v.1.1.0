import cv2
import mediapipe as mp
import time
import math
import csv
import os
from datetime import datetime

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/face_landmarker.task"

EAR_CLOSED_THRESHOLD = 0.25

MIN_CLOSED_DURATION = 0.05

MAX_BLINK_DURATION = 0.80

SMOOTHING_ALPHA = 0.35

RECORDING_DURATION = 60


# ============================================================
# Output file
# ============================================================

os.makedirs("data/processed", exist_ok=True)

OUTPUT_FILE = (
    "data/processed/blink_recording_"
    + datetime.now().strftime("%Y%m%d_%H%M%S")
    + ".csv"
)


# ============================================================
# MediaPipe
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

landmarker = vision.FaceLandmarker.create_from_options(
    options
)


# ============================================================
# Eye landmarks
# ============================================================

LEFT_EYE = [
    33, 160, 158, 133, 153, 144
]

RIGHT_EYE = [
    362, 385, 387, 263, 373, 380
]


# ============================================================
# Distance
# ============================================================

def distance(p1, p2):

    return math.sqrt(
        (p1.x - p2.x) ** 2 +
        (p1.y - p2.y) ** 2
    )


# ============================================================
# EAR
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
# Blink state
# ============================================================

OPEN = "OPEN"
CLOSING = "CLOSING"
CLOSED = "CLOSED"
OPENING = "OPENING"

eye_state = OPEN


# ============================================================
# Blink statistics
# ============================================================

blink_count = 0

blink_start_time = None

last_blink_duration = 0.0


# ============================================================
# EAR smoothing
# ============================================================

smoothed_ear = None


def smooth_ear(current_ear):

    global smoothed_ear

    if smoothed_ear is None:

        smoothed_ear = current_ear

    else:

        smoothed_ear = (
            SMOOTHING_ALPHA * current_ear
            +
            (1 - SMOOTHING_ALPHA) * smoothed_ear
        )

    return smoothed_ear


# ============================================================
# CSV
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
    "smoothed_ear",
    "eye_state",
    "blink_event",
    "blink_duration",
    "face_detected"
])


# ============================================================
# Webcam
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("ERROR: Could not open webcam.")

    csv_file.close()
    landmarker.close()

    exit()


print()
print("==========================================")
print("     BLINKGUARD BLINK DATA RECORDER")
print("==========================================")
print()
print(f"Recording duration: {RECORDING_DURATION} seconds")
print()
print("Instructions:")
print()
print("1. Blink naturally.")
print("2. Do NOT count your blinks manually.")
print("3. Occasionally blink quickly.")
print("4. Occasionally blink normally.")
print("5. Once or twice, close your eyes for")
print("   approximately 1-2 seconds.")
print()
print("We will compare the recorded EAR signal")
print("with the detector output.")
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


while True:

    success, frame = cap.read()

    if not success:

        print("ERROR: Could not read webcam frame.")
        break


    # Mirror image
    frame = cv2.flip(frame, 1)


    # Convert BGR → RGB
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # MediaPipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )


    # Time
    current_time = time.time()

    elapsed = (
        current_time -
        recording_start
    )

    timestamp_ms = int(
        elapsed * 1000
    )


    # ========================================================
    # Face detection
    # ========================================================

    result = landmarker.detect_for_video(
        mp_image,
        timestamp_ms
    )


    blink_event = 0

    blink_duration = 0.0

    face_detected = 0

    left_ear = 0.0
    right_ear = 0.0
    average_ear = 0.0


    # ========================================================
    # Face found
    # ========================================================

    if result.face_landmarks:

        face_detected = 1

        face_landmarks = result.face_landmarks[0]


        # ----------------------------------------------------
        # EAR
        # ----------------------------------------------------

        left_ear = calculate_ear(
            face_landmarks,
            LEFT_EYE
        )

        right_ear = calculate_ear(
            face_landmarks,
            RIGHT_EYE
        )

        average_ear = (
            left_ear +
            right_ear
        ) / 2.0


        # ----------------------------------------------------
        # Smoothed EAR
        # ----------------------------------------------------

        ear = smooth_ear(
            average_ear
        )


        # ====================================================
        # STATE MACHINE
        # ====================================================

        if eye_state == OPEN:

            if ear < EAR_CLOSED_THRESHOLD:

                eye_state = CLOSING

                blink_start_time = current_time


        elif eye_state == CLOSING:

            if ear < EAR_CLOSED_THRESHOLD:

                closed_duration = (
                    current_time -
                    blink_start_time
                )

                if (
                    closed_duration >=
                    MIN_CLOSED_DURATION
                ):

                    eye_state = CLOSED


            else:

                eye_state = OPEN

                blink_start_time = None


        elif eye_state == CLOSED:

            if ear >= EAR_CLOSED_THRESHOLD:

                eye_state = OPENING


        elif eye_state == OPENING:

            if ear >= EAR_CLOSED_THRESHOLD:

                blink_duration = (
                    current_time -
                    blink_start_time
                )


                # --------------------------------------------
                # Blink confirmation
                # --------------------------------------------

                if (
                    MIN_CLOSED_DURATION
                    <= blink_duration
                    <= MAX_BLINK_DURATION
                ):

                    blink_count += 1

                    blink_event = 1

                    last_blink_duration = (
                        blink_duration
                    )


                eye_state = OPEN

                blink_start_time = None


            else:

                eye_state = CLOSED


    # ========================================================
    # Save CSV row
    # ========================================================

    csv_writer.writerow([
        datetime.now().isoformat(),
        round(elapsed, 4),
        round(left_ear, 6),
        round(right_ear, 6),
        round(average_ear, 6),
        round(
            smoothed_ear if smoothed_ear is not None else 0,
            6
        ),
        eye_state,
        blink_event,
        round(blink_duration, 6),
        face_detected
    ])


    sample_count += 1


    # ========================================================
    # Display
    # ========================================================

    current_display_ear = (
        smoothed_ear
        if smoothed_ear is not None
        else 0.0
    )


    cv2.putText(
        frame,
        f"EAR: {current_display_ear:.3f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )


    cv2.putText(
        frame,
        f"State: {eye_state}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )


    cv2.putText(
        frame,
        f"Blinks: {blink_count}",
        (20, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )


    cv2.putText(
        frame,
        f"Time: {elapsed:.1f}s / {RECORDING_DURATION}s",
        (20, 145),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )


    if blink_event == 1:

        cv2.putText(
            frame,
            "BLINK DETECTED",
            (20, 190),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )


    # ========================================================
    # Show
    # ========================================================

    cv2.imshow(
        "BlinkGuard - Blink Data Recorder",
        frame
    )


    # ========================================================
    # Stop
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
print("       RECORDING COMPLETED")
print("==========================================")
print()
print("Samples:", sample_count)
print("Detected blinks:", blink_count)
print()
print("Saved file:")
print(OUTPUT_FILE)
print()