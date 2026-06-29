"""
mirror_app.py — Blur-based foggy mirror.

Maintains three layers:
  1. Webcam frame (dimmed for contrast)
  2. Heavily blurred version of the webcam frame (the "fog" layer)
  3. Alpha mask   (float32, 0 = clear, 1 = fully fogged, starts ~0.9)

Final:  output = sharp * (1 - alpha) + blurred * alpha

When alpha ≈ 0.9 the mirror looks fogged (blurred).
Wiping sets alpha → 0, revealing the sharp camera feed.

Press [F] to toggle the blur overlay for debugging.
Press [q] to quit.
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

MODEL_PATH = "models/hand_landmarker.task"

# --- fog / blur configuration ---
ALPHA_INITIAL = 0.90         # starting fog opacity (0=clear, 1=fogged)
REFOG_RATE = 0.008           # alpha increase per frame (~3 s to full re-fog)
WIPE_RADIUS_FACTOR = 0.55    # wipe circle radius relative to hand size
BRIGHTNESS_FACTOR = 0.55     # keep 55% brightness (45% dim) for fog contrast
WIPE_BLUR_SIGMA = 8          # Gaussian blur sigma for smooth wipe edges
BLUR_STRENGTH = 31           # Gaussian blur sigma for the "fog" itself
BLUR_DOWNSCALE = 4           # downscale factor for the blur pass (4=16× fewer pixels)

# --- colours for UI ---
LM_COLOUR = (100, 220, 100)
CONN_COLOUR = (180, 180, 180)
TEXT_GREY = (120, 120, 120)
TEXT_AMBER = (50, 180, 230)
WIPE_GREEN = (0, 255, 0)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _sq_dist(a, b):
    return (a.x - b.x) ** 2 + (a.y - b.y) ** 2


def _dist(a, b):
    return _sq_dist(a, b) ** 0.5


def is_open_palm(landmarks) -> bool:
    """True when all five fingers are extended (open palm = wipe pose)."""
    lm = landmarks
    thumb_ext = _sq_dist(lm[HandLandmark.THUMB_TIP],
                         lm[HandLandmark.INDEX_FINGER_MCP]) > \
                _sq_dist(lm[HandLandmark.THUMB_IP],
                         lm[HandLandmark.INDEX_FINGER_MCP])
    index_ext  = lm[HandLandmark.INDEX_FINGER_TIP].y  < lm[HandLandmark.INDEX_FINGER_PIP].y
    middle_ext = lm[HandLandmark.MIDDLE_FINGER_TIP].y < lm[HandLandmark.MIDDLE_FINGER_PIP].y
    ring_ext   = lm[HandLandmark.RING_FINGER_TIP].y   < lm[HandLandmark.RING_FINGER_PIP].y
    pinky_ext  = lm[HandLandmark.PINKY_TIP].y         < lm[HandLandmark.PINKY_PIP].y
    return thumb_ext and index_ext and middle_ext and ring_ext and pinky_ext


def get_palm_center(landmarks):
    idx = [HandLandmark.WRIST, HandLandmark.INDEX_FINGER_MCP,
           HandLandmark.MIDDLE_FINGER_MCP, HandLandmark.RING_FINGER_MCP,
           HandLandmark.PINKY_MCP]
    return (sum(landmarks[i].x for i in idx) / len(idx),
            sum(landmarks[i].y for i in idx) / len(idx))


def get_hand_size(landmarks) -> float:
    return _dist(landmarks[HandLandmark.WRIST],
                 landmarks[HandLandmark.MIDDLE_FINGER_TIP])





# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def draw_hand_skeleton(frame, landmarks) -> None:
    h, w, _ = frame.shape
    for conn in HandLandmarksConnections.HAND_CONNECTIONS:
        a, b = landmarks[conn.start], landmarks[conn.end]
        cv2.line(frame, (int(a.x * w), int(a.y * h)),
                 (int(b.x * w), int(b.y * h)), CONN_COLOUR, 2, cv2.LINE_AA)
    for lm in landmarks:
        cv2.circle(frame, (int(lm.x * w), int(lm.y * h)),
                   5, LM_COLOUR, -1, cv2.LINE_AA)


def wipe_alpha(alpha: np.ndarray, landmarks, wipe_radius: int) -> None:
    """
    Clear alpha (set toward 0) in a hand-shaped area with smoothed edges.

    Circles are drawn on a full-res temp mask, then downscaled before the
    Gaussian blur (big speed win) and upscaled back for the alpha blend.
    """
    h, w = alpha.shape
    temp = np.zeros((h, w), dtype=np.float32)

    # palm centre
    cx, cy = get_palm_center(landmarks)
    cv2.circle(temp, (int(cx * w), int(cy * h)),
               wipe_radius, 1.0, -1, cv2.LINE_AA)

    # fingertips
    tips = [HandLandmark.THUMB_TIP, HandLandmark.INDEX_FINGER_TIP,
            HandLandmark.MIDDLE_FINGER_TIP, HandLandmark.RING_FINGER_TIP,
            HandLandmark.PINKY_TIP]
    r_tip = max(6, wipe_radius // 2)
    for tip in tips:
        tx, ty = landmarks[tip].x, landmarks[tip].y
        cv2.circle(temp, (int(tx * w), int(ty * h)),
                   r_tip, 1.0, -1, cv2.LINE_AA)

    # smooth edges (downscaled for speed)
    if WIPE_BLUR_SIGMA > 0:
        s = BLUR_DOWNSCALE
        small = cv2.resize(temp, (w // s, h // s), interpolation=cv2.INTER_LINEAR)
        small = cv2.GaussianBlur(small, (0, 0), sigmaX=WIPE_BLUR_SIGMA)
        temp = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

    # apply: alpha = min(alpha, 1 - wipe_intensity)
    np.minimum(alpha, 1.0 - temp, out=alpha)


# ---------------------------------------------------------------------------
# Compositing
# ---------------------------------------------------------------------------

def composite(sharp: np.ndarray, blurred: np.ndarray,
              alpha: np.ndarray) -> np.ndarray:
    """Alpha blend between sharp and blurred frames: output = sharp * (1-α) + blurred * α"""
    a = alpha[..., None]  # (h, w) → (h, w, 1) for broadcasting
    return (sharp.astype(np.float32) * (1.0 - a) +
            blurred.astype(np.float32) * a).astype(np.uint8)


def fast_blur(frame: np.ndarray, sigma: float, scale: int) -> np.ndarray:
    """
    Downscale → blur → upscale.  Much faster than a full-res blur when sigma
    is large, because the blur kernel operates on ~1/scale² fewer pixels.

    The sigma is divided by the scale factor so the visual blur radius matches
    what you'd get at full resolution.
    """
    h, w = frame.shape[:2]
    sh, sw = max(1, h // scale), max(1, w // scale)
    small = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_LINEAR)
    small = cv2.GaussianBlur(small, (0, 0), sigmaX=sigma / scale)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # --- MediaPipe hand landmarker ---
    base_opts_obj = base_opts.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_opts_obj,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = vision.HandLandmarker.create_from_options(options)

    # --- webcam (cap at 640×480 to reduce pixels) ---
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        landmarker.close()
        return

    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to read frame.")
        cap.release()
        landmarker.close()
        return

    h, w = frame.shape[:2]
    print(f"Camera: {w}x{h}")

    # ---- layers ----
    alpha = np.full((h, w), ALPHA_INITIAL, dtype=np.float32)

    show_fog = True
    print(f"Blur strength: {BLUR_STRENGTH}, alpha initial: {ALPHA_INITIAL}")
    print("Foggy mirror ready — open palm to wipe.")
    print("  [F]  toggle blur overlay")
    print("  [q]  quit")

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # Mirror + dim for contrast
        frame = cv2.flip(frame, 1)
        dimmed = cv2.convertScaleAbs(frame, alpha=BRIGHTNESS_FACTOR, beta=0)

        # ---- hand tracking (uses original frame, not dimmed) ----
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)
        result: HandLandmarkerResult = landmarker.detect(mp_image)

        wipe_active = False

        if result.hand_landmarks:
            for hand_landmarks in result.hand_landmarks:
                draw_hand_skeleton(dimmed, hand_landmarks)

                if is_open_palm(hand_landmarks):
                    wipe_active = True
                    hand_sz = get_hand_size(hand_landmarks)
                    radius = max(10, int(hand_sz * w * WIPE_RADIUS_FACTOR))
                    wipe_alpha(alpha, hand_landmarks, radius)

        # ---- re-fog: alpha creeps back toward ALPHA_INITIAL ----
        np.minimum(alpha + REFOG_RATE, ALPHA_INITIAL, out=alpha)

        # ---- blur the frame for the "fog" layer (downscaled for speed), then composite ----
        blurred = fast_blur(dimmed, BLUR_STRENGTH, BLUR_DOWNSCALE)

        if show_fog:
            output = composite(dimmed, blurred, alpha)
        else:
            output = dimmed.copy()

        # ---- debug overlay ----
        if show_fog:
            if wipe_active:
                info, colour = "WIPE MODE", WIPE_GREEN
            elif not result.hand_landmarks:
                info, colour = "Show your hand to the mirror", TEXT_GREY
            else:
                info, colour = "Open your palm to wipe", TEXT_AMBER
            cv2.putText(output, info, (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, colour, 2)
            fog_pct = int(alpha.mean() * 100)
            cv2.putText(output, f"Fog: {fog_pct}%", (20, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            # Debug: sample pixel values
            if frame_count % 30 == 0:
                print(f"[frame {frame_count}] alpha mean: {alpha.mean():.3f} "
                      f"dimmed[0,0]: {dimmed[0,0]} blurred[0,0]: {blurred[0,0]} "
                      f"output[0,0]: {output[0,0]} show_fog: {show_fog}")
        else:
            cv2.putText(output, "BLUR OFF  [F]", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, TEXT_GREY, 2)

        cv2.imshow("Foggy Mirror", output)

        # ---- key handling ----
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("f"):
            show_fog = not show_fog
            print(f"Fog overlay: {'ON' if show_fog else 'OFF'}")

        if cv2.getWindowProperty("Foggy Mirror", cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()
