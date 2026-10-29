"""Shared configuration for webcam monitoring and evidence recording."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

CAMERA_ID = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

YOLO_MODEL_NAME = "yolov8n.pt"
CONFIDENCE_THRESHOLD = 0.4
YOLO_IMAGE_SIZE = 640
YOLO_MAX_DETECTIONS = 100
YOLO_DEVICE: int | str | None = None
DISPLAYED_CLASSES = ("car", "motorcycle", "bus", "truck", "person")

VIDEO_FPS = 20.0
VIDEO_CODEC = "mp4v"
VIDEO_EXTENSION = ".mp4"
PRE_EVENT_SECONDS = 10.0
POST_EVENT_SECONDS = 10.0
FPS_SMOOTHING_ALPHA = 0.2
DISPLAY_WINDOW_NAME = "Road Accident Detection - Phase 1"

MODELS_DIR = PROJECT_ROOT / "models"
EVIDENCE_DIR = PROJECT_ROOT / "evidence"
OUTPUT_DIR = PROJECT_ROOT / "output"


def ensure_directories() -> None:
    """Create project data directories if they do not already exist."""
    for directory in (MODELS_DIR, EVIDENCE_DIR, OUTPUT_DIR):
        directory.mkdir(parents=True, exist_ok=True)
