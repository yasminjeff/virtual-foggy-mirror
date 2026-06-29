"""
hand_tracker.py — Step 2: Hand tracking with MediaPipe Tasks API.

Adds 21-point hand landmark detection to the mirrored webcam feed.
Detects open palm (wipe pose) vs other hand states.
"""

import cv2
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core import base_options as base_opts
from mediapipe.tasks.python.vision.core.image import Image, ImageFormat
from mediapipe.tasks.python.vision.hand_landmarker import (
    HandLandmark,
    HandLandmarkerResult,
    HandLandmarksConnections,
)

MODEL_PATH = "hand_landmarker.task"

# -- colour constants for drawing --
LM_COLOUR = (100, 220, 100)     # BGR green for landmark dots
CONN_COLOUR = (180, 180, 180)   # BGR grey for connection lines
WIPE_COLOUR = (0, 255, 0)       # BGR green for wipe mode
TEXT_GREY = (120, 120, 120)
TEXT_AMBER = (50, 180, 230)


def _sq_dist(a, b) -> float:
    """Squared distance between two landmarks (avoids sqrt)."""
    return (a.x - b.x) ** 2 + (a.y - b.y) ** 2


def is_open_palm(landmarks: list) -> bool:
    """
    Returns True when all five fingers are extended (open palm = wipe pose).

    Finger logic (assuming hand faces camera, fingers pointing up):
      - Index, middle, ring, pinky: tip above PIP joint (tip.y < pip.y)
      - Thumb: tip farther from index MCP than the IP joint is (wide spread)
    """
    lm = landmarks

    # Thumb: extended if thumb-tip is spread away from the palm
    thumb_ext = _sq_dist(lm[HandLandmark.THUMB_TIP], lm[HandLandmark.INDEX_FINGER_MCP]) > \
                _sq_dist(lm[HandLandmark.THUMB_IP], lm[HandLandmark.INDEX_FINGER_MCP])

    # Four fingers: extended if the tip is above the PIP joint
    index_ext  = lm[HandLandmark.INDEX_FINGER_TIP].y  < lm[HandLandmark.INDEX_FINGER_PIP].y
    middle_ext = lm[HandLandmark.MIDDLE_FINGER_TIP].y < lm[HandLandmark.MIDDLE_FINGER_PIP].y
    ring_ext   = lm[HandLandmark.RING_FINGER_TIP].y   < lm[HandLandmark.RING_FINGER_PIP].y
    pinky_ext  = lm[HandLandmark.PINKY_TIP].y         < lm[HandLandmark.PINKY_PIP].y

    return thumb_ext and index_ext and middle_ext and ring_ext and pinky_ext


def draw_hand_skeleton(frame: np.ndarray, landmarks: list) -> None:
    """Draw hand landmarks and connections on the frame (in-place)."""
    h, w, _ = frame.shape

    # Draw connections (lines between landmark pairs)
    for conn in HandLandmarksConnections.HAND_CONNECTIONS:
        a = landmarks[conn.start]
        b = landmarks[conn.end]
        x1, y1 = int(a.x * w), int(a.y * h)
        x2, y2 = int(b.x * w), int(b.y * h)
        cv2.line(frame, (x1, y1), (x2, y2), CONN_COLOUR, 2, cv2.LINE_AA)

    # Draw landmark dots
    for lm in landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (cx, cy), 5, LM_COLOUR, -1, cv2.LINE_AA)


def main() -> None:
    # --- initialise hand landmarker ---
    base_opts_obj = base_opts.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_opts_obj,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = vision.HandLandmarker.create_from_options(options)

    # --- open webcam ---
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Webcam opened. Press 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame.")
            break

        # Mirror horizontally so it feels like a mirror
        frame = cv2.flip(frame, 1)

        # MediaPipe Tasks expects SRGB Image
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)
        result: HandLandmarkerResult = landmarker.detect(mp_image)

        wipe_mode = False

        if result.hand_landmarks:
            for hand_landmarks in result.hand_landmarks:
                draw_hand_skeleton(frame, hand_landmarks)

                if is_open_palm(hand_landmarks):
                    wipe_mode = True

        # --- On-screen status ---
        if wipe_mode:
            cv2.putText(
                frame, "WIPE MODE", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, WIPE_COLOUR, 3,
            )
        elif not result.hand_landmarks:
            cv2.putText(
                frame, "Show your hand to the mirror", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT_GREY, 2,
            )
        else:
            cv2.putText(
                frame, "Open your palm to wipe", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT_AMBER, 2,
            )

        cv2.imshow("Foggy Mirror", frame)

        # Exit on X button or 'q'
        if cv2.getWindowProperty("Foggy Mirror", cv2.WND_PROP_VISIBLE) < 1:
            break
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()
