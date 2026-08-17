import cv2
import mediapipe as mp
import time
import math


from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/face_landmarker.task"

# TEMPORARY prototype threshold.
# We will NOT treat this as the final research threshold.
EAR_CLOSED_THRESHOLD = 0.25

# Minimum amount of time the eyes must remain below
# the threshold before we consider it a real closure.
MIN_CLOSED_DURATION = 0.05

# Maximum duration for something to be considered
# a normal blink.
MAX_BLINK_DURATION = 0.80

# Exponential smoothing factor.
# Lower = smoother but slower response.
SMOOTHING_ALPHA = 0.35


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

landmarker = vision.FaceLandmarker.create_from_options(
    options
)


# ============================================================
# Eye landmark indices
# ============================================================

LEFT_EYE = [
    33,
    160,
    158,
    133,
    153,
    144
]

RIGHT_EYE = [
    362,
    385,
    387,
    263,
    373,
    380
]


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

last_blink_time = None


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
# Webcam
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("ERROR: Could not open webcam.")

    landmarker.close()

    exit()


print()
print("==========================================")
print("       BLINKGUARD BLINK DETECTOR")
print("==========================================")
print()
print("Temporary EAR threshold:",
      EAR_CLOSED_THRESHOLD)
print()
print("Blink naturally.")
print("Watch the eye state and blink count.")
print()
print("Press Q to quit.")
print()


# ============================================================
# Timing
# ============================================================

start_time = time.time()

previous_time = start_time


# ============================================================
# Main loop
# ============================================================

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


    # Current time
    current_time = time.time()

    elapsed = current_time - start_time

    timestamp_ms = int(
        elapsed * 1000
    )


    # ========================================================
    # Face landmark detection
    # ========================================================

    result = landmarker.detect_for_video(
        mp_image,
        timestamp_ms
    )


    # ========================================================
    # Face detected
    # ========================================================

    if result.face_landmarks:

        face_landmarks = result.face_landmarks[0]


        # ----------------------------------------------------
        # Calculate both eyes
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
            left_ear + right_ear
        ) / 2.0


        # ----------------------------------------------------
        # Smooth EAR
        # ----------------------------------------------------

        ear = smooth_ear(
            average_ear
        )


        # ====================================================
        # BLINK STATE MACHINE
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

                if closed_duration >= MIN_CLOSED_DURATION:

                    eye_state = CLOSED

            else:

                # EAR went back up too quickly.
                # Treat it as noise rather than a blink.

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
                # Confirm blink
                # --------------------------------------------

                if (
                    MIN_CLOSED_DURATION
                    <= blink_duration
                    <= MAX_BLINK_DURATION
                ):

                    blink_count += 1

                    last_blink_duration = (
                        blink_duration
                    )

                    last_blink_time = (
                        current_time
                    )


                # Reset state
                eye_state = OPEN

                blink_start_time = None


            else:

                # Eye went closed again.
                eye_state = CLOSED


        # ====================================================
        # Display
        # ====================================================

        cv2.putText(
            frame,
            f"EAR: {ear:.3f}",
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


        if last_blink_duration > 0:

            cv2.putText(
                frame,
                f"Last blink: "
                f"{last_blink_duration:.3f}s",
                (20, 145),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )


        # ----------------------------------------------------
        # Draw eye landmarks
        # ----------------------------------------------------

        height, width, _ = frame.shape


        for index in LEFT_EYE:

            landmark = face_landmarks[index]

            x = int(
                landmark.x * width
            )

            y = int(
                landmark.y * height
            )

            cv2.circle(
                frame,
                (x, y),
                4,
                (0, 255, 0),
                -1
            )


        for index in RIGHT_EYE:

            landmark = face_landmarks[index]

            x = int(
                landmark.x * width
            )

            y = int(
                landmark.y * height
            )

            cv2.circle(
                frame,
                (x, y),
                4,
                (0, 255, 0),
                -1
            )


    else:

        # ----------------------------------------------------
        # No face
        # ----------------------------------------------------

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
        "BlinkGuard - Blink Detector",
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


print()
print("==========================================")
print("          BLINK TEST COMPLETED")
print("==========================================")
print()
print("Total blinks:", blink_count)
print(
    "Last blink duration:",
    round(last_blink_duration, 3),
    "seconds"
)
print()