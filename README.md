# Virtual Foggy Mirror

Turn your webcam into a steamed mirror you can wipe clean with your hand.

Hold up an open palm — the fog clears where your hand passes, like wiping a fogged-up bathroom mirror. Close your hand or move away, and the fog slowly creeps back.

## How it works

Three layers composited in real time:

1. **Sharp webcam frame** (dimmed to 55% brightness for contrast)
2. **Blurred copy** of the frame (the "fog" layer, using a heavy Gaussian blur at sigma=31)
3. **Alpha mask** — float32, 0 = clear, 1 = fully fogged, starts at 0.90

**Output**: `sharp × (1 − α) + blurred × α`

Hand tracking uses MediaPipe Tasks API to detect open palms — the alpha mask is cleared in a smoothed area around the palm centre and fingertips. Every frame, the alpha creeps back toward 0.90 (re-fog rate: 0.008/frame, about 3 seconds to full fog).

### Performance optimizations

- Fog blur runs at 4× downscale (16× fewer pixels, 26× faster) with proportional sigma scaling — visually identical
- Wipe edge smoothing uses the same downscale-blur trick
- Camera capped at 640×480
- Wipe blur sigma reduced to 8

## Requirements

- Python 3.10+
- Webcam

## Setup

```bash
# Create virtual environment
python -m venv venv

# Activate
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download the MediaPipe hand landmarker model
# (auto-downloaded, or place in models/)
```

## Usage

```bash
python main.py
```

Hold your hand up with an open palm and move it across the camera to wipe the fog.

| Key | Action |
|-----|--------|
| `F` | Toggle fog/blur overlay on/off |
| `q` | Quit |

## Project structure

```
├── main.py                  # Entry point
├── src/
│   ├── __init__.py
│   └── foggy_mirror.py      # Core application logic
├── dev/
│   ├── hand_tracker.py      # Reference: hand tracking module
│   └── webcam_capture.py    # Reference: webcam capture module
├── models/
│   └── hand_landmarker.task # MediaPipe model
├── requirements.txt
└── .gitignore
```

## Controls

- **Open palm** = wipe the fog
- **Fist / no hand** = let the fog return
- **F** key = toggle fog overlay (for debugging)
- **q** key = quit
