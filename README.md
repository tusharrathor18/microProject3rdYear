# Road Accident Detection and Emergency Alert System

This phase provides live laptop-webcam monitoring, pretrained YOLOv8 nano
object detection, an on-screen object-count panel, a smoothed FPS counter, and
manual simulated-incident video evidence recording. It detects objects only;
it does not recognize accidents, classify incidents, or send emergency
notifications.

## Requirements

- Windows 10 or 11
- Python 3.10 or later
- A working laptop webcam

## Setup in VS Code

Open the project folder in VS Code and open a PowerShell terminal in the
project root. Create the virtual environment if needed, then install the
project dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

In VS Code, select `.venv\Scripts\python.exe` as the Python interpreter.

## Run

From the project root, run:

```powershell
.\.venv\Scripts\python.exe main.py
```

On the first run, Ultralytics downloads the pretrained `yolov8n.pt` model if it
is not already cached. Internet access is needed for that initial download.
Allow camera access if Windows prompts you.

## Controls

- Press **A** in the video window to simulate an incident and save a clip with
  the configured pre-event and post-event footage (10 seconds each by default).
- Press **Q** in the video window to quit. If a clip is still in progress, the
  available frames are finalized before exit.

Run the recorder tests from the project root with:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
```

The live window shows object bounding boxes and class labels, an overall object
count, separate counts for cars, motorcycles, buses, trucks, and people, and
the smoothed FPS. The interface indicates when an incident clip is recording.
The **A** key is a manual demonstration trigger, not automatic accident
detection. Clips are saved as `evidence\incident_YYYY-MM-DD_HH-MM-SS.mp4`;
colliding filenames receive a numeric suffix. No JPEG or PNG snapshots are
created, and no alerts or messages are sent.
Open the `evidence\` folder in File Explorer and double-click the MP4 to play
it in Windows Media Player or another MP4-compatible player.

## Troubleshooting

- **Webcam unavailable:** Close other apps that may be using the camera. Check
  Windows **Settings > Privacy & security > Camera** and allow desktop apps to
  access it. The app tries DirectShow first and then OpenCV's default backend.
  If you have multiple cameras, change `CAMERA_ID` in `config.py` (usually `0`
  for the built-in webcam).
- **Missing packages:** From the project root, run
  `.\.venv\Scripts\python.exe -m pip install -r requirements.txt`. Confirm VS
  Code is using `.venv\Scripts\python.exe`.
- **Model download fails:** Check internet access and retry. A firewall, proxy,
  or network policy may block the initial download. If the weight file is
  already available locally, set `YOLO_MODEL_NAME` in `config.py` to its path.
- **Window does not respond to Q:** Click the video window to focus it, then
  press **Q**.

## Project layout

```text
RoadAccidentDetection/
├── config.py
├── main.py
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── camera/
│   │   ├── __init__.py
│   │   └── camera.py
│   ├── detection/
│   │   ├── __init__.py
│   │   └── detector.py
│   └── evidence/
│       ├── __init__.py
│       └── video_recorder.py
├── tests/
│   └── test_video_recorder.py
├── models/
├── evidence/
└── output/
```

Video recording is started only by the manual **A** demonstration trigger;
there is no automatic incident detection.
