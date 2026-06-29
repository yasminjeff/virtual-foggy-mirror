"""
webcam_capture.py — Step 1: Basic mirrored webcam feed.

Opens the default webcam, flips horizontally (mirror mode),
and displays the feed. Press 'q' to quit.
"""

import cv2


def main() -> None:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Webcam opened. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame.")
            break

        # Mirror horizontally so it acts like a mirror
        frame = cv2.flip(frame, 1)

        cv2.imshow("Foggy Mirror", frame)

        # Check if window was closed via X button
        if cv2.getWindowProperty("Foggy Mirror", cv2.WND_PROP_VISIBLE) < 1:
            break

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
