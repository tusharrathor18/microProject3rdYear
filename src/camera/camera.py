"""Laptop webcam capture using the configured camera and frame dimensions."""

import logging

import cv2

from config import CAMERA_ID, FRAME_HEIGHT, FRAME_WIDTH

LOGGER = logging.getLogger(__name__)


class Camera:
    """Open and read frames from the configured webcam."""

    def __init__(self) -> None:
        self._capture: cv2.VideoCapture | None = None
        try:
            self._capture = cv2.VideoCapture(CAMERA_ID, cv2.CAP_DSHOW)
        except cv2.error as exc:
            LOGGER.warning("DirectShow could not open the camera: %s", exc)

        if self._capture is None or not self._capture.isOpened():
            if self._capture is not None:
                self._capture.release()
            LOGGER.info("Trying the default OpenCV camera backend.")
            try:
                self._capture = cv2.VideoCapture(CAMERA_ID)
            except cv2.error as exc:
                raise RuntimeError(
                    f"Could not initialize camera {CAMERA_ID} with the default backend."
                ) from exc

        if not self._capture.isOpened():
            self._capture.release()
            self._capture = None
            raise RuntimeError(
                f"Could not open camera {CAMERA_ID}. Check that it is connected, "
                "camera access is enabled, and no other application is using it."
            )

        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    def read(self) -> cv2.typing.MatLike:
        """Return the next webcam frame or raise if capture fails."""
        if self._capture is None:
            raise RuntimeError("The webcam is not initialized.")
        success, frame = self._capture.read()
        if not success or frame is None:
            raise RuntimeError(
                f"Could not read a frame from camera {CAMERA_ID}. "
                "Check the camera connection and permissions."
            )
        return frame

    def release(self) -> None:
        """Release the webcam device."""
        if self._capture is not None:
            self._capture.release()
            self._capture = None
