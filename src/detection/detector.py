"""YOLOv8 object detection and annotated-frame generation."""

from dataclasses import dataclass

import cv2
from ultralytics import YOLO
from ultralytics.engine.results import Results

from config import (
    CONFIDENCE_THRESHOLD,
    YOLO_DEVICE,
    YOLO_IMAGE_SIZE,
    YOLO_MAX_DETECTIONS,
    YOLO_MODEL_NAME,
)


@dataclass(frozen=True)
class Detection:
    """Details for one detected object."""

    class_id: int
    class_name: str
    confidence: float
    box: tuple[int, int, int, int]


class Detector:
    """Load the configured pretrained YOLO model and detect objects in frames."""

    def __init__(self) -> None:
        self._model = YOLO(YOLO_MODEL_NAME)

    def detect(
        self,
        frame: cv2.typing.MatLike,
    ) -> tuple[cv2.typing.MatLike, list[Detection]]:
        """Return an annotated frame and object detections for one image."""
        results = list(
            self._model.predict(
                source=frame,
                conf=CONFIDENCE_THRESHOLD,
                imgsz=YOLO_IMAGE_SIZE,
                max_det=YOLO_MAX_DETECTIONS,
                device=YOLO_DEVICE,
                verbose=False,
            )
        )
        if not results:
            raise RuntimeError("The YOLO model returned no detection results.")
        result = results[0]
        if not isinstance(result, Results):
            raise RuntimeError("The YOLO model returned an unsupported result type.")
        annotated_frame = result.plot()
        if annotated_frame is None:
            raise RuntimeError("The YOLO model could not render detection results.")

        detections: list[Detection] = []
        boxes = result.boxes
        if boxes is not None:
            for class_id_value, confidence_value, coordinates in zip(
                boxes.cls.tolist(),
                boxes.conf.tolist(),
                boxes.xyxy.tolist(),
                strict=True,
            ):
                class_id = int(class_id_value)
                class_name = result.names[class_id]
                x1, y1, x2, y2 = (int(value) for value in coordinates)
                detections.append(
                    Detection(
                        class_id=class_id,
                        class_name=str(class_name),
                        confidence=float(confidence_value),
                        box=(x1, y1, x2, y2),
                    )
                )

        return annotated_frame, detections
