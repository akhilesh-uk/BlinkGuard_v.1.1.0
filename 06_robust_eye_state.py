import cv2
import mediapipe as mp
import time
import math
import statistics
import csv
import os
import json
from datetime import datetime


from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# BLINKGUARD
# ROBUST EYE STATE + EXPERIMENTAL DATA RECORDER
# ============================================================
#
# Current pipeline:
#
# Webcam
#    ↓
# MediaPipe Face Landmarker
#    ↓
# Face landmarks
#    ↓
# Left / Right EAR
#    ↓
# EAR smoothing
#    ↓
# Personal calibration
#    ↓
# Adaptive threshold
#    ↓
# Temporal eye-state machine
#    ↓
# Blink / prolonged closure
#    ↓
# CSV recording
#
# This version is intended for:
#
# 1. Development
# 2. Controlled testing
# 3. Spectacles comparison
# 4. EAR analysis
# 5. Later PERCLOS implementation
#
# IMPORTANT:
# This is NOT the final fatigue model.
#
# ============================================================


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/face_landmarker.task"


# ------------------------------------------------------------
# Calibration
# ------------------------------------------------------------

CALIBRATION_DURATION = 10.0


# ------------------------------------------------------------
# EAR smoothing
# ------------------------------------------------------------

SMOOTHING_ALPHA = 0.35


# ------------------------------------------------------------
# Adaptive closed-eye threshold
# ------------------------------------------------------------

CLOSED_RATIO = 0.65


# ------------------------------------------------------------
# Temporal filtering
# ------------------------------------------------------------

MIN_CLOSED_DURATION = 0.05

MAX_BLINK_DURATION = 0.80


# ------------------------------------------------------------
# Valid EAR range
# ------------------------------------------------------------

MIN_VALID_EAR = 0.10

MAX_VALID_EAR = 0.60


# ------------------------------------------------------------
# CSV flush interval
# ------------------------------------------------------------

CSV_FLUSH_INTERVAL = 1.0


# ============================================================
# EYE LANDMARK INDICES
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
# UTILITY FUNCTIONS
# ============================================================

def distance(p1, p2):
    """
    Euclidean distance between two MediaPipe
    normalized landmark points.
    """

    return math.sqrt(
        (p1.x - p2.x) ** 2
        +
        (p1.y - p2.y) ** 2
    )


# ============================================================

def calculate_ear(landmarks, eye_indices):
    """
    Calculate Eye Aspect Ratio.

    EAR =
        (vertical_1 + vertical_2)
        /
        (2 × horizontal)
    """

    p1 = landmarks[eye_indices[0]]
    p2 = landmarks[eye_indices[1]]
    p3 = landmarks[eye_indices[2]]
    p4 = landmarks[eye_indices[3]]
    p5 = landmarks[eye_indices[4]]
    p6 = landmarks[eye_indices[5]]

    vertical_1 = distance(p2, p6)

    vertical_2 = distance(p3, p5)

    horizontal = distance(p1, p4)

    if horizontal <= 1e-9:
        return 0.0

    ear = (
        vertical_1
        +
        vertical_2
    ) / (
        2.0 * horizontal
    )

    return ear


# ============================================================

def safe_round(value, digits=5):

    if value is None:
        return ""

    return round(value, digits)


# ============================================================
# EXPERIMENT INFORMATION
# ============================================================

print()
print("==============================================")
print("             BLINKGUARD")
print("      ROBUST EYE STATE EXPERIMENT")
print("==============================================")
print()


print("Enter the test condition.")
print()
print("Examples:")
print("  spectacles")
print("  no_spectacles")
print("  low_light")
print("  normal_light")
print("  high_light")
print()


condition = input(
    "Test condition: "
).strip()


if condition == "":
    condition = "unspecified"


print()
print("Select the intended processing mode.")
print()
print("1 = LOW POWER")
print("2 = BALANCED")
print("3 = HIGH POWER")
print()


mode_input = input(
    "Processing mode [1/2/3]: "
).strip()


if mode_input == "1":

    processing_mode = "low_power"

    target_fps = 5

elif mode_input == "3":

    processing_mode = "high_power"

    target_fps = 15

else:

    processing_mode = "balanced"

    target_fps = 10


print()
print(
    f"Mode: {processing_mode}"
)

print(
    f"Target FPS: {target_fps}"
)

print()


# ============================================================
# MEDIAPIPE INITIALIZATION
# ============================================================

print("Loading MediaPipe Face Landmarker...")
print()


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


print(
    "MediaPipe loaded successfully."
)

print()


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(0)


if not cap.isOpened():

    print()
    print("ERROR: Could not open webcam.")
    print()

    landmarker.close()

    raise SystemExit


# ============================================================
# GLOBAL MEDIA PIPE TIMESTAMP
# ============================================================
#
# MediaPipe VIDEO mode requires timestamps to be strictly
# monotonically increasing.
#
# We use one timestamp source for the entire program.
#
# DO NOT reset previous_timestamp.
# ============================================================

program_start = time.monotonic()

previous_timestamp = -1


def get_timestamp_ms():

    global previous_timestamp

    timestamp_ms = int(
        (
            time.monotonic()
            -
            program_start
        )
        * 1000
    )

    if timestamp_ms <= previous_timestamp:

        timestamp_ms = (
            previous_timestamp
            +
            1
        )

    previous_timestamp = timestamp_ms

    return timestamp_ms


# ============================================================
# CALIBRATION
# ============================================================

print("==============================================")
print("              CALIBRATION")
print("==============================================")
print()

print(
    "Keep your face in a normal working position."
)

print(
    "Look naturally at the screen."
)

print(
    "Keep your eyes naturally open."
)

print(
    "Do not intentionally keep your eyes wide open."
)

print(
    "Use your normal spectacles if applicable."
)

print()


for countdown in [3, 2, 1]:

    print(
        f"Starting calibration in {countdown}..."
    )

    time.sleep(1)


print()
print("CALIBRATION STARTED")
print()


calibration_values = []

calibration_start = time.monotonic()

calibration_valid_frames = 0

calibration_total_frames = 0


while True:

    success, frame = cap.read()


    if not success:

        print(
            "ERROR: Webcam frame could not be read."
        )

        break


    calibration_total_frames += 1


    frame = cv2.flip(
        frame,
        1
    )


    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )


    timestamp_ms = get_timestamp_ms()


    result = landmarker.detect_for_video(
        mp_image,
        timestamp_ms
    )


    elapsed = (
        time.monotonic()
        -
        calibration_start
    )


    if result.face_landmarks:

        landmarks = result.face_landmarks[0]


        left_ear = calculate_ear(
            landmarks,
            LEFT_EYE
        )


        right_ear = calculate_ear(
            landmarks,
            RIGHT_EYE
        )


        average_ear = (
            left_ear
            +
            right_ear
        ) / 2.0


        if (
            MIN_VALID_EAR
            <= average_ear
            <= MAX_VALID_EAR
        ):

            calibration_values.append(
                average_ear
            )

            calibration_valid_frames += 1


        cv2.putText(
            frame,
            "CALIBRATION",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )


        cv2.putText(
            frame,
            f"Time: "
            f"{elapsed:.1f}/"
            f"{CALIBRATION_DURATION:.0f}s",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2
        )


        cv2.putText(
            frame,
            f"EAR: "
            f"{average_ear:.3f}",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )


    else:

        cv2.putText(
            frame,
            "FACE NOT DETECTED",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )


    cv2.imshow(
        "BlinkGuard - Calibration",
        frame
    )


    key = cv2.waitKey(1) & 0xFF


    if key == ord("q"):

        cap.release()

        cv2.destroyAllWindows()

        landmarker.close()

        raise SystemExit


    if elapsed >= CALIBRATION_DURATION:

        break


# ============================================================
# CALIBRATION VALIDATION
# ============================================================

if len(calibration_values) < 30:

    print()
    print("==============================================")
    print("          CALIBRATION FAILED")
    print("==============================================")
    print()

    print(
        "Not enough valid EAR measurements."
    )

    print()
    print(
        f"Valid samples: "
        f"{len(calibration_values)}"
    )

    print()

    print(
        "Check lighting, face visibility, "
        "camera position, and spectacles."
    )

    print()


    cap.release()

    cv2.destroyAllWindows()

    landmarker.close()

    raise SystemExit


# ============================================================
# PERSONAL BASELINE
# ============================================================

baseline_ear = statistics.median(
    calibration_values
)


closed_threshold = (
    baseline_ear
    *
    CLOSED_RATIO
)


# Safety limits

closed_threshold = max(
    0.15,
    min(
        closed_threshold,
        0.35
    )
)


# ============================================================
# CALIBRATION STATISTICS
# ============================================================

calibration_mean = statistics.mean(
    calibration_values
)


calibration_min = min(
    calibration_values
)


calibration_max = max(
    calibration_values
)


if len(calibration_values) > 1:

    calibration_std = statistics.stdev(
        calibration_values
    )

else:

    calibration_std = 0.0


# ============================================================
# CALIBRATION OUTPUT
# ============================================================

print()
print("==============================================")
print("          CALIBRATION COMPLETE")
print("==============================================")
print()

print(
    f"Total frames: "
    f"{calibration_total_frames}"
)

print(
    f"Valid EAR frames: "
    f"{calibration_valid_frames}"
)

print(
    f"Baseline EAR: "
    f"{baseline_ear:.5f}"
)

print(
    f"Mean EAR: "
    f"{calibration_mean:.5f}"
)

print(
    f"EAR minimum: "
    f"{calibration_min:.5f}"
)

print(
    f"EAR maximum: "
    f"{calibration_max:.5f}"
)

print(
    f"EAR standard deviation: "
    f"{calibration_std:.5f}"
)

print(
    f"Closed threshold: "
    f"{closed_threshold:.5f}"
)

print()


# ============================================================
# RESET EAR SMOOTHING
# ============================================================

smoothed_ear = None


# ============================================================
# EYE STATES
# ============================================================

OPEN = "OPEN"

CLOSING = "CLOSING"

CLOSED = "CLOSED"

OPENING = "OPENING"


eye_state = OPEN


# ============================================================
# EVENT VARIABLES
# ============================================================

blink_count = 0

prolonged_closure_count = 0

closure_start_time = None

closure_was_prolonged = False

last_blink_duration = 0.0

last_prolonged_duration = 0.0


# ============================================================
# MANUAL TEST INPUT
# ============================================================

print("==============================================")
print("           MANUAL TEST INFORMATION")
print("==============================================")
print()

print(
    "After the camera test finishes, you can enter"
)

print(
    "the number of blinks YOU intentionally performed."
)

print(
    "This allows us to calculate missed detections."
)

print()

manual_blink_input = input(
    "Planned/observed blinks [press Enter to skip]: "
).strip()


if manual_blink_input:

    try:

        manual_blinks = int(
            manual_blink_input
        )

    except ValueError:

        manual_blinks = None

else:

    manual_blinks = None


print()


# ============================================================
# CSV DIRECTORY
# ============================================================

os.makedirs(
    "data/processed",
    exist_ok=True
)


# ============================================================
# SESSION IDENTIFIER
# ============================================================

session_timestamp = datetime.now().strftime(
    "%Y%m%d_%H%M%S"
)


# ============================================================
# CSV PATH
# ============================================================

csv_path = (
    "data/processed/"
    f"eye_state_recording_"
    f"{session_timestamp}.csv"
)


# ============================================================
# METADATA PATH
# ============================================================

metadata_path = (
    "data/processed/"
    f"eye_state_metadata_"
    f"{session_timestamp}.json"
)


# ============================================================
# OPEN CSV
# ============================================================

csv_file = open(
    csv_path,
    "w",
    newline="",
    encoding="utf-8"
)


csv_writer = csv.writer(
    csv_file
)


# ============================================================
# CSV HEADER
# ============================================================

csv_writer.writerow([

    "timestamp_ms",

    "elapsed_seconds",

    "condition",

    "processing_mode",

    "target_fps",

    "face_detected",

    "left_ear",

    "right_ear",

    "average_ear",

    "smoothed_ear",

    "threshold",

    "eye_state",

    "closure_duration",

    "blink_event",

    "blink_count",

    "prolonged_closure_event",

    "prolonged_closure_count"

])


# ============================================================
# METADATA
# ============================================================

metadata = {

    "session_timestamp":
        session_timestamp,

    "condition":
        condition,

    "processing_mode":
        processing_mode,

    "target_fps":
        target_fps,

    "calibration_duration_seconds":
        CALIBRATION_DURATION,

    "baseline_ear":
        baseline_ear,

    "calibration_mean":
        calibration_mean,

    "calibration_min":
        calibration_min,

    "calibration_max":
        calibration_max,

    "calibration_std":
        calibration_std,

    "closed_ratio":
        CLOSED_RATIO,

    "closed_threshold":
        closed_threshold,

    "smoothing_alpha":
        SMOOTHING_ALPHA,

    "min_closed_duration":
        MIN_CLOSED_DURATION,

    "max_blink_duration":
        MAX_BLINK_DURATION,

    "manual_blinks":
        manual_blinks

}


with open(
    metadata_path,
    "w",
    encoding="utf-8"
) as metadata_file:

    json.dump(
        metadata_file if False else metadata,
        metadata_file,
        indent=4
    )


# ============================================================
# DETECTION START
# ============================================================

print("==============================================")
print("           EYE STATE DETECTION")
print("==============================================")
print()

print(
    f"Condition: {condition}"
)

print(
    f"Mode: {processing_mode}"
)

print(
    f"Target FPS: {target_fps}"
)

print()

print(
    "Perform your controlled test now."
)

print()

print(
    "Recommended:"
)

print(
    "• Normal blinking"
)

print(
    "• Fast blinking"
)

print(
    "• Slow blinking"
)

print(
    "• 2-second eye closure"
)

print()

print(
    "Press Q to stop."
)

print()

print(
    f"CSV: {csv_path}"
)

print()


time.sleep(2)


# ============================================================
# DETECTION TIMING
# ============================================================

detection_start = time.monotonic()

last_csv_flush = detection_start

frame_count = 0

processed_frame_count = 0

face_detected_count = 0

face_lost_count = 0

processing_times = []

last_frame_time = time.monotonic()


# ============================================================
# MAIN LOOP
# ============================================================

try:

    while True:

        loop_start = time.monotonic()


        # ====================================================
        # FRAME CAPTURE
        # ====================================================

        success, frame = cap.read()


        if not success:

            print(
                "ERROR: Could not read webcam frame."
            )

            break


        frame_count += 1


        frame = cv2.flip(
            frame,
            1
        )


        # ====================================================
        # RGB
        # ====================================================

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )


        # ====================================================
        # TIMESTAMP
        # ====================================================

        timestamp_ms = get_timestamp_ms()


        # ====================================================
        # MEDIA PIPE
        # ====================================================

        detection_call_start = time.monotonic()


        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )


        detection_call_time = (
            time.monotonic()
            -
            detection_call_start
        )


        processing_times.append(
            detection_call_time
        )


        # ====================================================
        # ELAPSED TIME
        # ====================================================

        elapsed_seconds = (
            time.monotonic()
            -
            detection_start
        )


        # ====================================================
        # DEFAULT VALUES
        # ====================================================

        face_detected = False

        left_ear_value = None

        right_ear_value = None

        average_ear_value = None

        smoothed_ear_value = None

        closure_duration = 0.0

        blink_event = False

        prolonged_closure_event = False


        # ====================================================
        # FACE DETECTED
        # ====================================================

        if result.face_landmarks:

            face_detected = True

            face_detected_count += 1

            processed_frame_count += 1


            landmarks = result.face_landmarks[0]


            # =================================================
            # EAR
            # =================================================

            left_ear_value = calculate_ear(
                landmarks,
                LEFT_EYE
            )


            right_ear_value = calculate_ear(
                landmarks,
                RIGHT_EYE
            )


            average_ear_value = (
                left_ear_value
                +
                right_ear_value
            ) / 2.0


            # =================================================
            # VALID EAR
            # =================================================

            if (
                MIN_VALID_EAR
                <= average_ear_value
                <= MAX_VALID_EAR
            ):


                # =============================================
                # SMOOTH EAR
                # =============================================

                if smoothed_ear is None:

                    smoothed_ear = (
                        average_ear_value
                    )

                else:

                    smoothed_ear = (
                        SMOOTHING_ALPHA
                        *
                        average_ear_value
                        +
                        (
                            1.0
                            -
                            SMOOTHING_ALPHA
                        )
                        *
                        smoothed_ear
                    )


                smoothed_ear_value = (
                    smoothed_ear
                )


                current_time = (
                    time.monotonic()
                )


                # =============================================
                # OPEN
                # =============================================

                if eye_state == OPEN:

                    if (
                        smoothed_ear
                        <
                        closed_threshold
                    ):

                        eye_state = CLOSING

                        closure_start_time = (
                            current_time
                        )

                        closure_was_prolonged = False


                # =============================================
                # CLOSING
                # =============================================

                elif eye_state == CLOSING:

                    if (
                        smoothed_ear
                        <
                        closed_threshold
                    ):

                        closure_duration = (
                            current_time
                            -
                            closure_start_time
                        )


                        if (
                            closure_duration
                            >=
                            MIN_CLOSED_DURATION
                        ):

                            eye_state = CLOSED


                    else:

                        eye_state = OPEN

                        closure_start_time = None

                        closure_was_prolonged = False


                # =============================================
                # CLOSED
                # =============================================

                elif eye_state == CLOSED:

                    closure_duration = (
                        current_time
                        -
                        closure_start_time
                    )


                    if (
                        closure_duration
                        >
                        MAX_BLINK_DURATION
                    ):

                        closure_was_prolonged = True


                    if (
                        smoothed_ear
                        >=
                        closed_threshold
                    ):

                        eye_state = OPENING


                # =============================================
                # OPENING
                # =============================================

                elif eye_state == OPENING:

                    if (
                        smoothed_ear
                        >=
                        closed_threshold
                    ):

                        closure_duration = (
                            current_time
                            -
                            closure_start_time
                        )


                        # =====================================
                        # NORMAL BLINK
                        # =====================================

                        if (
                            MIN_CLOSED_DURATION
                            <=
                            closure_duration
                            <=
                            MAX_BLINK_DURATION
                            and
                            not
                            closure_was_prolonged
                        ):

                            blink_count += 1

                            blink_event = True

                            last_blink_duration = (
                                closure_duration
                            )


                        # =====================================
                        # PROLONGED CLOSURE
                        # =====================================

                        elif (
                            closure_duration
                            >
                            MAX_BLINK_DURATION
                        ):

                            prolonged_closure_count += 1

                            prolonged_closure_event = True

                            last_prolonged_duration = (
                                closure_duration
                            )


                        eye_state = OPEN

                        closure_start_time = None

                        closure_was_prolonged = False


                    else:

                        eye_state = CLOSED


                # =============================================
                # CURRENT CLOSURE DURATION
                # =============================================

                if closure_start_time is not None:

                    closure_duration = (
                        current_time
                        -
                        closure_start_time
                    )


                # =================================================
                # DISPLAY EAR
                # =================================================

                cv2.putText(
                    frame,
                    f"EAR: "
                    f"{smoothed_ear:.3f}",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )


                cv2.putText(
                    frame,
                    f"Threshold: "
                    f"{closed_threshold:.3f}",
                    (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2
                )


                cv2.putText(
                    frame,
                    f"State: "
                    f"{eye_state}",
                    (20, 105),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )


                cv2.putText(
                    frame,
                    f"Blinks: "
                    f"{blink_count}",
                    (20, 140),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )


                cv2.putText(
                    frame,
                    f"Long closures: "
                    f"{prolonged_closure_count}",
                    (20, 175),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 255),
                    2
                )


                cv2.putText(
                    frame,
                    f"Closure: "
                    f"{closure_duration:.2f}s",
                    (20, 210),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2
                )


                # =============================================
                # EVENT DISPLAY
                # =============================================

                if blink_event:

                    cv2.putText(
                        frame,
                        "BLINK DETECTED",
                        (20, 250),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2
                    )


                elif prolonged_closure_event:

                    cv2.putText(
                        frame,
                        "PROLONGED CLOSURE",
                        (20, 250),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (0, 255, 255),
                        2
                    )


        # ====================================================
        # FACE NOT DETECTED
        # ====================================================

        else:

            face_lost_count += 1


            cv2.putText(
                frame,
                "FACE NOT DETECTED",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )


            cv2.putText(
                frame,
                f"Blinks: "
                f"{blink_count}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )


        # ====================================================
        # CSV RECORDING
        # ====================================================

        csv_writer.writerow([

            timestamp_ms,

            round(
                elapsed_seconds,
                4
            ),

            condition,

            processing_mode,

            target_fps,

            face_detected,

            safe_round(
                left_ear_value
            ),

            safe_round(
                right_ear_value
            ),

            safe_round(
                average_ear_value
            ),

            safe_round(
                smoothed_ear_value
            ),

            safe_round(
                closed_threshold
            ),

            eye_state,

            round(
                closure_duration,
                4
            ),

            blink_event,

            blink_count,

            prolonged_closure_event,

            prolonged_closure_count

        ])


        # ====================================================
        # PERIODIC CSV FLUSH
        # ====================================================

        current_time = time.monotonic()


        if (
            current_time
            -
            last_csv_flush
            >=
            CSV_FLUSH_INTERVAL
        ):

            csv_file.flush()

            last_csv_flush = (
                current_time
            )


        # ====================================================
        # FPS
        # ====================================================

        frame_processing_time = (
            time.monotonic()
            -
            loop_start
        )


        if frame_processing_time > 0:

            instantaneous_fps = (
                1.0
                /
                frame_processing_time
            )

        else:

            instantaneous_fps = 0.0


        # ====================================================
        # DISPLAY FPS
        # ====================================================

        cv2.putText(
            frame,
            f"FPS: "
            f"{instantaneous_fps:.1f}",
            (20, 290),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )


        cv2.putText(
            frame,
            f"Mode: "
            f"{processing_mode}",
            (20, 325),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )


        # ====================================================
        # SHOW
        # ====================================================

        cv2.imshow(
            "BlinkGuard - Eye State Recorder",
            frame
        )


        # ====================================================
        # QUIT
        # ====================================================

        key = cv2.waitKey(1) & 0xFF


        if key == ord("q"):

            break


finally:

    # ========================================================
    # CSV CLOSE
    # ========================================================

    csv_file.flush()

    csv_file.close()


    # ========================================================
    # CAMERA
    # ========================================================

    cap.release()


    # ========================================================
    # WINDOWS
    # ========================================================

    cv2.destroyAllWindows()


    # ========================================================
    # MEDIAPIPE
    # ========================================================

    landmarker.close()


# ============================================================
# FINAL STATISTICS
# ============================================================

total_detection_time = (
    time.monotonic()
    -
    detection_start
)


if total_detection_time > 0:

    average_fps = (
        frame_count
        /
        total_detection_time
    )

else:

    average_fps = 0.0


if face_detected_count > 0:

    face_detection_rate = (
        face_detected_count
        /
        frame_count
        *
        100
    )

else:

    face_detection_rate = 0.0


if processing_times:

    average_mediapipe_time = (
        statistics.mean(
            processing_times
        )
        *
        1000
    )

else:

    average_mediapipe_time = 0.0


# ============================================================
# MANUAL ACCURACY
# ============================================================

if manual_blinks is not None:

    detected_blinks = blink_count

    missed_blinks = max(
        0,
        manual_blinks
        -
        detected_blinks
    )

    if manual_blinks > 0:

        detection_rate = (
            detected_blinks
            /
            manual_blinks
            *
            100
        )

    else:

        detection_rate = 0.0

else:

    detected_blinks = blink_count

    missed_blinks = None

    detection_rate = None


# ============================================================
# UPDATE METADATA
# ============================================================

metadata.update({

    "total_frames":
        frame_count,

    "face_detected_frames":
        face_detected_count,

    "face_lost_frames":
        face_lost_count,

    "face_detection_rate_percent":
        face_detection_rate,

    "average_fps":
        average_fps,

    "average_mediapipe_processing_ms":
        average_mediapipe_time,

    "detected_blinks":
        detected_blinks,

    "missed_blinks":
        missed_blinks,

    "manual_detection_rate_percent":
        detection_rate,

    "prolonged_closures":
        prolonged_closure_count,

    "last_blink_duration_seconds":
        last_blink_duration,

    "last_prolonged_closure_seconds":
        last_prolonged_duration

})


# ============================================================
# SAVE FINAL METADATA
# ============================================================

with open(
    metadata_path,
    "w",
    encoding="utf-8"
) as metadata_file:

    json.dump(
        metadata,
        metadata_file,
        indent=4
    )


# ============================================================
# FINAL REPORT
# ============================================================

print()
print("==============================================")
print("             TEST COMPLETED")
print("==============================================")
print()

print(
    f"Condition: "
    f"{condition}"
)

print(
    f"Processing mode: "
    f"{processing_mode}"
)

print(
    f"Target FPS: "
    f"{target_fps}"
)

print()

print(
    f"Baseline EAR: "
    f"{baseline_ear:.5f}"
)

print(
    f"Closed threshold: "
    f"{closed_threshold:.5f}"
)

print()

print(
    f"Frames processed: "
    f"{frame_count}"
)

print(
    f"Face detection rate: "
    f"{face_detection_rate:.2f}%"
)

print(
    f"Average FPS: "
    f"{average_fps:.2f}"
)

print(
    f"Average MediaPipe processing time: "
    f"{average_mediapipe_time:.2f} ms"
)

print()

print(
    f"Blinks detected: "
    f"{blink_count}"
)

print(
    f"Prolonged closures: "
    f"{prolonged_closure_count}"
)


if manual_blinks is not None:

    print(
        f"Manual blink count: "
        f"{manual_blinks}"
    )

    print(
        f"Missed blinks: "
        f"{missed_blinks}"
    )

    print(
        f"Blink detection rate: "
        f"{detection_rate:.2f}%"
    )


print()

print(
    "CSV data:"
)

print(
    csv_path
)

print()

print(
    "Session metadata:"
)

print(
    metadata_path
)

print()

print("==============================================")
print("                 DONE")
print("==============================================")
print()