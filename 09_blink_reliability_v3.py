import cv2
import mediapipe as mp
import time
import math
import statistics
import csv
import os
import json
from datetime import datetime
from collections import deque

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# BLINKGUARD
# BLINK DETECTION RELIABILITY EXPERIMENT -- v2
# (Hysteresis state machine + median filter + FPS diagnostics
#  + synchronized ground truth)
# ============================================================
#
# CHANGE FROM v1:
# v1 tested spectacles/high_power and got 24 detected vs 50 manual
# blinks (~52% miss rate). The most likely cause is that v1's EMA
# smoothing (alpha=0.35) took ~3 frames (~200ms at 15 FPS) to react,
# while a real blink lasts only ~100-150ms -- the smoothing was
# very likely averaging real blinks away before they crossed
# closing_threshold. v2 replaces EMA with a 3-frame MEDIAN FILTER,
# which suppresses noise without smearing fast transients, and adds
# achieved-FPS logging so we can tell whether frame rate itself is
# also a limiting factor.
# ============================================================
#
# WHY THIS SCRIPT EXISTS
# ------------------------------------------------------------
# Your last recorded session (eye_state_metadata_20260814_105829.json)
# logged manual_blinks = 5, but you performed 20+ blinks. That gap
# is NOT a detector accuracy number -- it means the way blinks were
# being counted was itself unreliable. You cannot evaluate whether
# any detector change "helped" without a ground truth you trust.
#
# This script fixes TWO things at once, and keeps them separable
# in the output data so you can still analyze each independently:
#
# 1. GROUND TRUTH LOGGING
#    Instead of typing a single total blink count before the test,
#    you press SPACE at the moment of every blink you perform.
#    Each press is timestamped. This gives you a *timestamped event
#    log*, not just a final integer -- which is what you need to
#    compute precision / recall, not just "detected vs claimed".
#
# 2. HYSTERESIS STATE MACHINE
#    The old state machine (04_blink_detector.py, 06_robust_eye_state.py)
#    used ONE threshold for both closing and opening transitions.
#    When smoothed EAR hovers near that single threshold (common with
#    spectacle glare / noisy tracking), the state can flicker back
#    and forth across it within a couple of frames. That flicker either
#    (a) fragments a real blink so it never satisfies MIN_CLOSED_DURATION,
#    or (b) creates a short phantom closure that isn't a real blink.
#
#    The fix is a hysteresis BAND instead of a single line:
#      - CLOSING_THRESHOLD (lower)  -- eye must drop below this to
#        start being considered "closing"
#      - OPENING_THRESHOLD (higher) -- eye must rise above this to
#        be considered "open" again
#    Small noise that stays inside the band no longer flips the state.
#    This is the same idea used in Schmitt-trigger circuits and is a
#    standard fix for threshold-chatter problems in signal processing.
#
# CALIBRATION IS DELIBERATELY UNCHANGED
# ------------------------------------------------------------
# 06_robust_eye_state.py already computes baseline_ear as a MEDIAN
# (not mean) of calibration samples, which is already fairly robust
# to a stray blink during the 10s calibration window. We are keeping
# that logic exactly as-is here. Only ONE variable changes in this
# experiment: the state machine (single threshold -> hysteresis band).
# This is intentional -- change one thing at a time so that if blink
# detection improves, you know WHY it improved.
#
# ============================================================


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/face_landmarker.task"


# ------------------------------------------------------------
# Calibration (unchanged from 06_robust_eye_state.py)
# ------------------------------------------------------------

CALIBRATION_DURATION = 10.0


# ------------------------------------------------------------
# EAR filtering
# ------------------------------------------------------------
# v1 used an exponential moving average (EMA) here. EMA smears fast
# transients: at ~15 FPS it takes ~3 frames (~200ms) to reflect a
# step change, but a real blink is only ~100-150ms start to finish.
# That lag was very likely eating real blinks before they ever
# crossed closing_threshold.
#
# v2 uses a MEDIAN FILTER over the last MEDIAN_WINDOW raw EAR values
# instead. A median filter rejects single-frame noise spikes just as
# well as EMA, but it does NOT smear real transients -- a genuine dip
# shows up almost as soon as it happens, because the median simply
# picks the middle value rather than blending old and new readings.
# This is the standard fix for "smoothing ate my fast event."

MEDIAN_WINDOW = 3


# ------------------------------------------------------------
# HYSTERESIS thresholds (ratios of baseline_ear)
# ------------------------------------------------------------
# CLOSING_RATIO must be LOWER than OPENING_RATIO. The gap between
# them is the "dead band" that absorbs noise. If you find the
# detector still flickers, widen this gap. If you find it is
# missing very fast/shallow blinks, narrow it.

CLOSING_RATIO = 0.68
OPENING_RATIO = 0.80

# Safety bounds so a bad calibration can't produce a degenerate
# threshold (e.g. negative, or above the open-eye EAR itself).
MIN_CLOSING_THRESHOLD = 0.15
MAX_CLOSING_THRESHOLD = 0.32

MIN_OPENING_THRESHOLD = 0.18
MAX_OPENING_THRESHOLD = 0.38

# Minimum required gap between the two thresholds. If calibration
# produces thresholds closer than this, we force them apart so the
# hysteresis band always does something.
MIN_BAND_GAP = 0.02


# ------------------------------------------------------------
# Temporal filtering
# ------------------------------------------------------------

MIN_CLOSED_DURATION = 0.05
MAX_BLINK_DURATION = 0.80


# ------------------------------------------------------------
# Valid EAR range (rejects obviously broken landmark frames)
# ------------------------------------------------------------

MIN_VALID_EAR = 0.10
MAX_VALID_EAR = 0.60


# ------------------------------------------------------------
# Ground truth matching tolerance
# ------------------------------------------------------------
# A human pressing SPACE has reaction-time lag relative to the
# physical blink (typically 150-350ms). We match a manual event
# to the NEAREST detected blink within this window. This tolerance
# is recorded in the output JSON so it's part of your methodology,
# not a hidden constant.

MATCH_TOLERANCE_SECONDS = 0.45


# ------------------------------------------------------------
# CSV flush interval
# ------------------------------------------------------------

CSV_FLUSH_INTERVAL = 1.0


# ============================================================
# EYE LANDMARK INDICES
# ============================================================

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def distance(p1, p2):
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)


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

    if horizontal <= 1e-9:
        return 0.0

    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def safe_round(value, digits=5):
    if value is None:
        return ""
    return round(value, digits)


def clamp(value, low, high):
    return max(low, min(value, high))


# ============================================================
# GROUND-TRUTH <-> DETECTED BLINK MATCHING
# ============================================================
#
# Greedy nearest-neighbor matching within a tolerance window.
# Each manual event can match at most one detected event and
# vice versa. This gives:
#
#   True Positives  (TP) = matched pairs
#   False Negatives (FN) = manual blinks with no matching detection
#                          (blinks you made that the detector missed)
#   False Positives (FP) = detected blinks with no matching manual
#                          event (detector triggered on nothing)
#
# Precision = TP / (TP + FP)   "of what it detected, how much was real"
# Recall    = TP / (TP + FN)   "of what actually happened, how much
#                               did it catch"
# F1        = harmonic mean of precision and recall
#
def match_events(manual_times, detected_times, tolerance):
    manual_times = sorted(manual_times)
    detected_times = sorted(detected_times)

    used_detected = [False] * len(detected_times)
    matches = []
    unmatched_manual = []

    for m_time in manual_times:
        best_index = None
        best_diff = None

        for i, d_time in enumerate(detected_times):
            if used_detected[i]:
                continue

            diff = abs(d_time - m_time)

            if diff <= tolerance:
                if best_diff is None or diff < best_diff:
                    best_diff = diff
                    best_index = i

        if best_index is not None:
            used_detected[best_index] = True
            matches.append((m_time, detected_times[best_index], best_diff))
        else:
            unmatched_manual.append(m_time)

    unmatched_detected = [
        d_time for i, d_time in enumerate(detected_times)
        if not used_detected[i]
    ]

    true_positives = len(matches)
    false_negatives = len(unmatched_manual)
    false_positives = len(unmatched_detected)

    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0 else None
    )

    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0 else None
    )

    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = None

    return {
        "matches": matches,
        "unmatched_manual": unmatched_manual,
        "unmatched_detected": unmatched_detected,
        "true_positives": true_positives,
        "false_negatives": false_negatives,
        "false_positives": false_positives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ============================================================
# EXPERIMENT SETUP
# ============================================================

print()
print("==============================================")
print("             BLINKGUARD")
print("     BLINK RELIABILITY EXPERIMENT")
print("==============================================")
print()

print("Enter the test condition.")
print("Examples: spectacles / no_spectacles / low_light / normal_light")
print()
condition = input("Condition: ").strip() or "unspecified"

print()
print("Select processing mode:")
print("  1 = low_power   (10 FPS)")
print("  2 = balanced     (20 FPS)")
print("  3 = high_power   (30 FPS)")
print()
mode_choice = input("Mode [1/2/3]: ").strip()

mode_map = {
    "1": ("low_power", 10),
    "2": ("balanced", 20),
    "3": ("high_power", 30),
}

processing_mode, target_fps = mode_map.get(mode_choice, ("balanced", 20))

print()
print(f"Condition:       {condition}")
print(f"Processing mode: {processing_mode} (target {target_fps} FPS)")
print()
print("HOW THIS TEST WORKS:")
print("  1. A 10-second calibration will run first. Keep your eyes")
print("     naturally open and look at the screen.")
print("  2. During the main recording, press SPACE at the exact")
print("     moment you perform each blink. Blink naturally, but")
print("     make each one deliberate so your keypress lines up.")
print("  3. Press Q at any time to end the recording early.")
print()
input("Press Enter when you are ready to begin calibration...")


# ============================================================
# MediaPipe setup
# ============================================================

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)

options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
)

landmarker = vision.FaceLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    landmarker.close()
    raise SystemExit

session_start = time.monotonic()


def get_timestamp_ms():
    return int((time.monotonic() - session_start) * 1000)


# ============================================================
# CALIBRATION (unchanged logic from 06_robust_eye_state.py)
# ============================================================

print()
# ============================================================
# FRAME RATE THROTTLING
# ============================================================
# Camera capture (cap.read()) is left free-running -- we still call
# it every loop iteration so the internal camera buffer doesn't fill
# up and start returning stale/delayed frames. What we throttle is
# PROCESSING: landmark detection, EAR calculation, the state machine,
# and CSV logging only run once per FRAME_INTERVAL. Frames that land
# inside that interval are captured and immediately discarded. This
# is what actually enforces "10 FPS" rather than just labeling the
# mode and running at whatever the camera + MediaPipe throughput is.

FRAME_INTERVAL = 1.0 / target_fps

print()
print(f"CALIBRATION STARTED (throttled to ~{target_fps} FPS)")
print()

calibration_values = []
calibration_start = time.monotonic()
last_calib_process_time = None

while True:
    success, frame = cap.read()

    if not success:
        print("ERROR: Webcam frame could not be read.")
        break

    now = time.monotonic()

    if last_calib_process_time is not None and (now - last_calib_process_time) < FRAME_INTERVAL:
        # Discard this frame -- not yet time for the next processed
        # sample. Still poll for quit so the window stays responsive.
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            cap.release()
            cv2.destroyAllWindows()
            landmarker.close()
            raise SystemExit
        continue

    last_calib_process_time = now

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    timestamp_ms = get_timestamp_ms()
    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    elapsed = time.monotonic() - calibration_start

    if result.face_landmarks:
        landmarks = result.face_landmarks[0]

        left_ear = calculate_ear(landmarks, LEFT_EYE)
        right_ear = calculate_ear(landmarks, RIGHT_EYE)
        average_ear = (left_ear + right_ear) / 2.0

        if MIN_VALID_EAR <= average_ear <= MAX_VALID_EAR:
            calibration_values.append(average_ear)

        cv2.putText(frame, "CALIBRATION", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Time: {elapsed:.1f}/{CALIBRATION_DURATION:.0f}s",
                    (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        cv2.putText(frame, f"EAR: {average_ear:.3f}",
                    (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "FACE NOT DETECTED", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("BlinkGuard - Calibration", frame)

    key = cv2.waitKey(1) & 0xFF

    # Accept 'q', 'Q', and ESC -- some systems/keyboard layouts send an
    # uppercase code even without an explicit shift (e.g. Caps Lock on),
    # which would silently fail a check against ord("q") alone.
    if key in (ord("q"), ord("Q"), 27):
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()
        raise SystemExit

    if elapsed >= CALIBRATION_DURATION:
        break


if len(calibration_values) < 30:
    print()
    print("CALIBRATION FAILED: not enough valid EAR samples "
          f"({len(calibration_values)}). Check lighting, face "
          "visibility, camera position, and spectacles.")
    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()
    raise SystemExit


# ============================================================
# BASELINE + HYSTERESIS THRESHOLDS
# ============================================================

baseline_ear = statistics.median(calibration_values)

calibration_mean = statistics.mean(calibration_values)
calibration_std = (
    statistics.stdev(calibration_values)
    if len(calibration_values) > 1 else 0.0
)

closing_threshold = clamp(
    baseline_ear * CLOSING_RATIO,
    MIN_CLOSING_THRESHOLD,
    MAX_CLOSING_THRESHOLD,
)

opening_threshold = clamp(
    baseline_ear * OPENING_RATIO,
    MIN_OPENING_THRESHOLD,
    MAX_OPENING_THRESHOLD,
)

# Force a minimum gap so the hysteresis band always exists,
# even if calibration produced values that would otherwise
# collapse the band to near-zero width.
if opening_threshold - closing_threshold < MIN_BAND_GAP:
    mid = (opening_threshold + closing_threshold) / 2.0
    closing_threshold = mid - (MIN_BAND_GAP / 2.0)
    opening_threshold = mid + (MIN_BAND_GAP / 2.0)

print()
print("==============================================")
print("          CALIBRATION COMPLETE")
print("==============================================")
print(f"Baseline EAR (median):     {baseline_ear:.5f}")
print(f"Calibration mean:          {calibration_mean:.5f}")
print(f"Calibration std:           {calibration_std:.5f}")
print(f"Closing threshold:         {closing_threshold:.5f}")
print(f"Opening threshold:         {opening_threshold:.5f}")
print(f"Hysteresis band width:     {opening_threshold - closing_threshold:.5f}")
print()
input("Press Enter to begin the main recording...")


# ============================================================
# OUTPUT FILES
# ============================================================

session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

os.makedirs("data/processed", exist_ok=True)

csv_path = f"data/processed/blink_reliability_{session_timestamp}.csv"
manual_csv_path = f"data/processed/blink_reliability_manual_events_{session_timestamp}.csv"
metadata_path = f"data/processed/blink_reliability_metadata_{session_timestamp}.json"

csv_file = open(csv_path, "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow([
    "timestamp", "elapsed", "face_detected",
    "left_ear", "right_ear", "average_ear", "smoothed_ear",
    "closing_threshold", "opening_threshold",
    "eye_state", "closure_duration",
    "blink_event", "blink_count",
])

manual_csv_file = open(manual_csv_path, "w", newline="")
manual_csv_writer = csv.writer(manual_csv_file)
manual_csv_writer.writerow(["manual_blink_index", "timestamp_seconds"])


# ============================================================
# MAIN RECORDING LOOP
# ============================================================

OPEN, CLOSING, CLOSED, OPENING = "OPEN", "CLOSING", "CLOSED", "OPENING"
eye_state = OPEN

ear_window = deque(maxlen=MEDIAN_WINDOW)
closure_start_time = None

# Actual achieved FPS tracking -- tells us whether the frame rate
# itself is the ceiling on how short a blink we can even resolve.
frame_intervals = []
last_frame_time = None

blink_count = 0
blink_confirm_times = []   # timestamps (seconds since recording start) of detected blinks

manual_blink_times = []    # timestamps (seconds since recording start) of SPACE presses

recording_start = time.monotonic()
last_flush_time = recording_start
last_record_process_time = None
frame_count = 0

print()
print(f"RECORDING STARTED (throttled to ~{target_fps} FPS). "
      "Press SPACE for each blink. Press Q to stop.")
print()

while True:
    success, frame = cap.read()

    if not success:
        print("ERROR: Could not read webcam frame.")
        break

    now = time.monotonic()
    elapsed = now - recording_start

    # SPACE/quit are checked on every captured frame, NOT throttled --
    # ground-truth keypress timing needs to stay at full time resolution
    # even though EAR processing below is throttled to target_fps.
    key = cv2.waitKey(1) & 0xFF

    if key == ord(" "):
        manual_blink_times.append(elapsed)
        manual_csv_writer.writerow([len(manual_blink_times), round(elapsed, 4)])
        manual_csv_file.flush()
        print(f"  Manual blink #{len(manual_blink_times)} logged at {elapsed:.2f}s")

    elif key in (ord("q"), ord("Q"), 27):
        break

    if last_record_process_time is not None and (now - last_record_process_time) < FRAME_INTERVAL:
        # Discard this frame for EAR/detection purposes -- not yet
        # time for the next processed sample.
        continue

    last_record_process_time = now

    frame_count += 1
    frame = cv2.flip(frame, 1)

    if last_frame_time is not None:
        frame_intervals.append(now - last_frame_time)
    last_frame_time = now

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    timestamp_ms = get_timestamp_ms()
    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    face_detected = bool(result.face_landmarks)

    left_ear_value = right_ear_value = average_ear_value = None
    smoothed_ear_value = None
    closure_duration_value = None
    blink_event = 0

    if face_detected:
        landmarks = result.face_landmarks[0]

        left_ear_value = calculate_ear(landmarks, LEFT_EYE)
        right_ear_value = calculate_ear(landmarks, RIGHT_EYE)
        average_ear_value = (left_ear_value + right_ear_value) / 2.0

        if MIN_VALID_EAR <= average_ear_value <= MAX_VALID_EAR:

            ear_window.append(average_ear_value)
            # Median filter: rejects single-frame spikes without
            # smearing genuine fast dips the way EMA did.
            smoothed_ear = statistics.median(ear_window)
            smoothed_ear_value = smoothed_ear

            # ================================================
            # HYSTERESIS STATE MACHINE
            # ================================================
            #
            # OPEN     -> CLOSING : ear drops below closing_threshold
            # CLOSING  -> CLOSED  : still below closing_threshold after
            #                       MIN_CLOSED_DURATION
            # CLOSING  -> OPEN    : ear rises back above opening_threshold
            #                       (noise -- not a real closure). Note
            #                       this uses the UPPER threshold, not
            #                       the lower one, so small wobble near
            #                       closing_threshold does not cancel a
            #                       real blink in progress.
            # CLOSED   -> OPENING : ear rises above opening_threshold
            # OPENING  -> OPEN    : confirm blink if duration is in range
            # OPENING  -> CLOSED  : ear drops below closing_threshold
            #                       again (eye closed back down)
            #
            if eye_state == OPEN:
                if smoothed_ear < closing_threshold:
                    eye_state = CLOSING
                    closure_start_time = now

            elif eye_state == CLOSING:
                if smoothed_ear < closing_threshold:
                    closure_duration_value = now - closure_start_time
                    if closure_duration_value >= MIN_CLOSED_DURATION:
                        eye_state = CLOSED
                elif smoothed_ear >= opening_threshold:
                    eye_state = OPEN
                    closure_start_time = None
                # else: stays CLOSING -- ear is inside the band,
                # treated as still-closing noise, not a cancellation.

            elif eye_state == CLOSED:
                closure_duration_value = now - closure_start_time
                if smoothed_ear >= opening_threshold:
                    eye_state = OPENING

            elif eye_state == OPENING:
                closure_duration_value = now - closure_start_time
                if smoothed_ear >= opening_threshold:
                    if MIN_CLOSED_DURATION <= closure_duration_value <= MAX_BLINK_DURATION:
                        blink_count += 1
                        # Record the ONSET time (when the eye started
                        # closing), not the confirmation time (when it
                        # finished reopening). A person's SPACE press
                        # tracks close to when they feel themselves
                        # blink -- i.e. onset -- not when the eye is
                        # fully back open. Logging the confirmation
                        # instant here introduced a systematic lag
                        # equal to the full closure duration (found to
                        # average ~330ms in testing), which pushed
                        # otherwise-correct detections outside the
                        # match tolerance against ground truth.
                        blink_onset_elapsed = closure_start_time - recording_start
                        blink_confirm_times.append(blink_onset_elapsed)
                        blink_event = 1
                    eye_state = OPEN
                    closure_start_time = None
                elif smoothed_ear < closing_threshold:
                    eye_state = CLOSED

        # ---- on-screen overlay ----
        cv2.putText(frame, f"EAR: {smoothed_ear_value:.3f}" if smoothed_ear_value else "EAR: --",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"State: {eye_state}",
                    (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Detected blinks: {blink_count}",
                    (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Manual (SPACE) blinks: {len(manual_blink_times)}",
                    (20, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)
        cv2.putText(frame, f"Close/Open thr: {closing_threshold:.3f}/{opening_threshold:.3f}",
                    (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    else:
        cv2.putText(frame, "NO FACE DETECTED", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("BlinkGuard - Blink Reliability Test", frame)

    csv_writer.writerow([
        round(elapsed, 4), round(elapsed, 4), int(face_detected),
        safe_round(left_ear_value), safe_round(right_ear_value),
        safe_round(average_ear_value), safe_round(smoothed_ear_value),
        round(closing_threshold, 5), round(opening_threshold, 5),
        eye_state, safe_round(closure_duration_value),
        blink_event, blink_count,
    ])

    if now - last_flush_time >= CSV_FLUSH_INTERVAL:
        csv_file.flush()
        last_flush_time = now


# ============================================================
# CLEANUP
# ============================================================

cap.release()
cv2.destroyAllWindows()
landmarker.close()
csv_file.close()
manual_csv_file.close()


# ============================================================
# EVALUATION
# ============================================================

evaluation = match_events(
    manual_blink_times,
    blink_confirm_times,
    MATCH_TOLERANCE_SECONDS,
)

if frame_intervals:
    mean_interval = statistics.mean(frame_intervals)
    achieved_fps = 1.0 / mean_interval if mean_interval > 0 else None
    max_interval = max(frame_intervals)
    min_resolvable_blink_ms = mean_interval * 1000 * 2  # need ~2 frames to register a dip
else:
    achieved_fps = None
    max_interval = None
    min_resolvable_blink_ms = None

metadata = {
    "session_timestamp": session_timestamp,
    "condition": condition,
    "processing_mode": processing_mode,
    "target_fps": target_fps,
    "frames_processed": frame_count,
    "calibration_duration_seconds": CALIBRATION_DURATION,
    "baseline_ear": baseline_ear,
    "calibration_mean": calibration_mean,
    "calibration_std": calibration_std,
    "closing_threshold": closing_threshold,
    "opening_threshold": opening_threshold,
    "hysteresis_band_width": opening_threshold - closing_threshold,
    "filter_type": "median",
    "median_window": MEDIAN_WINDOW,
    "achieved_fps": achieved_fps,
    "max_frame_interval_seconds": max_interval,
    "min_resolvable_blink_duration_ms_estimate": min_resolvable_blink_ms,
    "min_closed_duration": MIN_CLOSED_DURATION,
    "max_blink_duration": MAX_BLINK_DURATION,
    "match_tolerance_seconds": MATCH_TOLERANCE_SECONDS,
    "detected_blink_count": blink_count,
    "manual_blink_count": len(manual_blink_times),
    "true_positives": evaluation["true_positives"],
    "false_negatives_missed_blinks": evaluation["false_negatives"],
    "false_positives_phantom_blinks": evaluation["false_positives"],
    "precision": evaluation["precision"],
    "recall": evaluation["recall"],
    "f1_score": evaluation["f1"],
    "csv_path": csv_path,
    "manual_events_csv_path": manual_csv_path,
    "note": (
        "Ground truth collected via synchronized SPACE keypress during "
        "recording, not a post-hoc typed total. Precision/recall are "
        "only as reliable as the operator's keypress timing -- treat "
        "small sample sizes (<20 events) as preliminary, not final "
        "research numbers. Detected blink timestamps used for matching "
        "are the CLOSURE ONSET time (when the eye started closing), not "
        "the confirmation time (when it finished reopening) -- v1/v2 "
        "before this fix used confirmation time, which under-reported "
        "matches by the full closure duration."
    ),
}

with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=4)

print()
print("==============================================")
print("                 RESULTS")
print("==============================================")
print()
print(f"Frames processed:        {frame_count}")
if achieved_fps is not None:
    print(f"Achieved FPS:            {achieved_fps:.1f} (target {target_fps})")
    print(f"Est. min resolvable blink duration: {min_resolvable_blink_ms:.0f}ms "
          "(need ~2 frames to register a dip)")
print(f"Detected blinks:         {blink_count}")
print(f"Manual (SPACE) blinks:   {len(manual_blink_times)}")
print()
print(f"True positives:          {evaluation['true_positives']}")
print(f"Missed blinks (FN):      {evaluation['false_negatives']}")
print(f"Phantom blinks (FP):     {evaluation['false_positives']}")
print()

if evaluation["precision"] is not None:
    print(f"Precision:                {evaluation['precision']:.3f}")
else:
    print("Precision:                N/A (no detected blinks)")

if evaluation["recall"] is not None:
    print(f"Recall:                   {evaluation['recall']:.3f}")
else:
    print("Recall:                   N/A (no manual blinks logged)")

if evaluation["f1"] is not None:
    print(f"F1 score:                 {evaluation['f1']:.3f}")
else:
    print("F1 score:                 N/A")

print()
print("CSV data:      ", csv_path)
print("Manual events: ", manual_csv_path)
print("Metadata/eval: ", metadata_path)
print()
print("==============================================")
print("                  DONE")
print("==============================================")
print()