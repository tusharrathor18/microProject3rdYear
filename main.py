"""Run live webcam monitoring with YOLO object detection."""

import logging
import time
from collections import Counter

import cv2

from config import (
    DISPLAYED_CLASSES,
    DISPLAY_WINDOW_NAME,
    FPS_SMOOTHING_ALPHA,
    POST_EVENT_SECONDS,
    PRE_EVENT_SECONDS,
    ensure_directories,
)
from src.camera.camera import Camera
from src.detection.detector import Detection, Detector
from src.evidence.video_recorder import VideoRecorder

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)


def draw_monitoring_overlay(
    frame: cv2.typing.MatLike,
    detections: list[Detection],
    fps: float,
    incident_recording: bool,
) -> cv2.typing.MatLike:
    """Draw the monitoring heading, object counts, and FPS on the frame."""
    counts = Counter(detection.class_name.lower() for detection in detections)
    lines = [
        "LIVE TRAFFIC MONITORING",
        f"Objects detected: {len(detections)}",
        *(f"{name.title()}: {counts[name]}" for name in DISPLAYED_CLASSES),
        f"FPS: {fps:.1f}",
        f"Incident clip: {'RECORDING' if incident_recording else 'READY'}",
        f"A triggers SIMULATED INCIDENT ({PRE_EVENT_SECONDS:g}s before + "
        f"{POST_EVENT_SECONDS:g}s after)",
    ]

    panel_height = 18 + len(lines) * 28
    cv2.rectangle(frame, (8, 8), (720, panel_height), (20, 20, 20), thickness=-1)
    for index, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (18, 32 + index * 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65 if index == 0 else 0.6,
            (0, 255, 0) if index == 0 else (255, 255, 255),
            2 if index == 0 else 1,
            cv2.LINE_AA,
        )
    return frame


def main() -> None:
    """Start live webcam object detection until the user presses Q."""
    camera: Camera | None = None
    recorder: VideoRecorder | None = None
    try:
        ensure_directories()
        detector = Detector()
        camera = Camera()
        recorder = VideoRecorder()
        previous_frame_time = time.perf_counter()
        fps = 0.0
        LOGGER.info(
            "Live monitoring started. Press A to trigger a simulated incident; "
            "press Q to quit."
        )
        while True:
            frame = camera.read()
            annotated_frame, detections = detector.detect(frame)

            current_time = time.perf_counter()
            elapsed = current_time - previous_frame_time
            if elapsed > 0:
                instantaneous_fps = 1.0 / elapsed
                fps = (
                    instantaneous_fps
                    if fps == 0.0
                    else FPS_SMOOTHING_ALPHA * instantaneous_fps
                    + (1.0 - FPS_SMOOTHING_ALPHA) * fps
                )
            previous_frame_time = current_time

            draw_monitoring_overlay(
                annotated_frame,
                detections,
                fps,
                recorder.is_recording,
            )
            recorder.add_frame(annotated_frame)
            cv2.imshow(DISPLAY_WINDOW_NAME, annotated_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key in (ord("a"), ord("A")):
                try:
                    recorder.trigger()
                except (RuntimeError, OSError) as exc:
                    LOGGER.warning("%s", exc)
    except Exception:
        LOGGER.exception("Live monitoring stopped because of an error.")
    finally:
        if recorder is not None:
            try:
                recorder.close()
            except Exception:
                LOGGER.exception("Could not finalize the incident recording cleanly.")
        if camera is not None:
            camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
