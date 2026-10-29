"""Rolling pre-event buffer and asynchronous simulated-incident clip writer."""

from collections import deque
from datetime import datetime
from pathlib import Path
import threading
import time
from typing import Callable

import cv2
import numpy as np

from config import (
    EVIDENCE_DIR,
    POST_EVENT_SECONDS,
    PRE_EVENT_SECONDS,
    VIDEO_CODEC,
    VIDEO_EXTENSION,
    VIDEO_FPS,
)

TimedFrame = tuple[float, np.ndarray]


class VideoRecorder:
    """Keep recent frames and save a clip after a manual incident trigger."""

    def __init__(
        self,
        pre_event_seconds: float = PRE_EVENT_SECONDS,
        post_event_seconds: float = POST_EVENT_SECONDS,
        fps: float = VIDEO_FPS,
        evidence_dir: Path = EVIDENCE_DIR,
        codec: str = VIDEO_CODEC,
        extension: str = VIDEO_EXTENSION,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if pre_event_seconds < 0:
            raise ValueError("Pre-event duration cannot be negative.")
        if post_event_seconds <= 0:
            raise ValueError("Post-event duration must be greater than zero.")
        if fps <= 0:
            raise ValueError("Video FPS must be greater than zero.")
        if len(codec) != 4:
            raise ValueError("Video codec must contain exactly four characters.")
        if not extension.startswith("."):
            raise ValueError("Video extension must start with a period.")

        self._pre_event_seconds = pre_event_seconds
        self._post_event_seconds = post_event_seconds
        self._fps = fps
        self._evidence_dir = Path(evidence_dir)
        self._codec = codec
        self._extension = extension
        self._clock = clock

        self._frame_buffer: deque[TimedFrame] = deque()
        self._incident_frames: list[TimedFrame] | None = None
        self._trigger_time: float | None = None
        self._incident_end_time: float | None = None
        self._incident_output_path: Path | None = None
        self._writer_thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._closed = False
        self._last_saved_path: Path | None = None
        self._last_error: Exception | None = None

    @property
    def is_recording(self) -> bool:
        """Whether a triggered clip is being captured or written."""
        with self._lock:
            return self._incident_frames is not None or self._writer_thread is not None

    @property
    def last_saved_path(self) -> Path | None:
        """Path of the most recently completed clip, if any."""
        with self._lock:
            return self._last_saved_path

    @property
    def last_error(self) -> Exception | None:
        """Most recent clip-writing error, if any."""
        with self._lock:
            return self._last_error

    def add_frame(self, frame: cv2.typing.MatLike, timestamp: float | None = None) -> None:
        """Add a copied frame to the rolling buffer and active incident."""
        if self._closed:
            raise RuntimeError("Cannot add frames after the recorder is closed.")
        frame_time = self._clock() if timestamp is None else timestamp
        if self._frame_buffer and frame_time < self._frame_buffer[-1][0]:
            raise ValueError("Frame timestamps must be monotonic.")
        frame_copy = np.array(frame, copy=True)
        if (
            frame_copy.ndim != 3
            or frame_copy.shape[0] == 0
            or frame_copy.shape[1] == 0
            or frame_copy.shape[2] != 3
        ):
            raise ValueError("Video frames must be non-empty, three-channel color images.")

        timed_frame = (frame_time, frame_copy)
        self._frame_buffer.append(timed_frame)
        cutoff = frame_time - self._pre_event_seconds
        while self._frame_buffer and self._frame_buffer[0][0] < cutoff:
            self._frame_buffer.popleft()

        incident_frames = self._incident_frames
        trigger_time = self._trigger_time
        incident_end_time = self._incident_end_time
        if (
            incident_frames is None
            or trigger_time is None
            or incident_end_time is None
        ):
            return
        if frame_time > trigger_time and frame_time <= incident_end_time:
            incident_frames.append(timed_frame)
        if frame_time >= incident_end_time:
            self._finish_incident(incident_end_time)

    def trigger(self) -> Path:
        """Trigger a simulated incident clip using buffered and upcoming frames."""
        if self._closed:
            raise RuntimeError("Cannot trigger an incident after the recorder is closed.")
        if self.is_recording:
            raise RuntimeError("An incident clip is already being recorded or saved.")
        if not self._frame_buffer:
            raise RuntimeError("No webcam frames are available to record yet.")

        now = self._clock()
        cutoff = now - self._pre_event_seconds
        pre_event_frames = [
            (frame_time, frame.copy())
            for frame_time, frame in self._frame_buffer
            if frame_time >= cutoff and frame_time <= now
        ]
        if not pre_event_frames:
            raise RuntimeError("No recent webcam frames are available for the clip.")

        self._evidence_dir.mkdir(parents=True, exist_ok=True)
        expected_path = self._new_output_path()
        self._incident_frames = pre_event_frames
        self._trigger_time = now
        self._incident_end_time = now + self._post_event_seconds
        self._incident_output_path = expected_path
        self._last_error = None
        print(
            f"Simulated incident recording started. "
            f"Saving video to: {expected_path}"
        )
        return expected_path

    def close(self, wait: bool = True) -> None:
        """Finalize an active clip and optionally wait for the writer to finish."""
        self._closed = True
        if self._incident_frames is not None:
            self._finish_incident(self._incident_frames[-1][0])
        if wait:
            self.wait_for_completion()

    def wait_for_completion(self, timeout: float | None = None) -> bool:
        """Wait for an active background writer; return whether it has finished."""
        with self._lock:
            writer_thread = self._writer_thread
        if writer_thread is None:
            return True

        writer_thread.join(timeout)
        return not writer_thread.is_alive()

    def _new_output_path(self) -> Path:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        base_name = f"incident_{timestamp}"
        output_path = self._evidence_dir / f"{base_name}{self._extension}"
        suffix = 1
        while output_path.exists():
            output_path = self._evidence_dir / (
                f"{base_name}_{suffix:02d}{self._extension}"
            )
            suffix += 1
        return output_path

    def _finish_incident(self, clip_end_time: float) -> None:
        if self._incident_frames is None:
            return
        frames = self._incident_frames
        output_path = self._incident_output_path
        self._incident_frames = None
        self._trigger_time = None
        self._incident_end_time = None
        self._incident_output_path = None
        if not frames or output_path is None:
            self._last_error = RuntimeError("No frames were captured for the incident clip.")
            print(f"Incident video was not saved: {self._last_error}")
            return

        writer_thread = threading.Thread(
            target=self._write_clip,
            args=(frames, output_path, clip_end_time),
            name="incident-video-writer",
            daemon=False,
        )
        with self._lock:
            self._writer_thread = writer_thread
        writer_thread.start()

    def _write_clip(
        self,
        frames: list[TimedFrame],
        output_path: Path,
        clip_end_time: float,
    ) -> None:
        writer: cv2.VideoWriter | None = None
        try:
            first_frame = frames[0][1]
            height, width = first_frame.shape[:2]
            if width <= 0 or height <= 0:
                raise RuntimeError("The incident clip has invalid frame dimensions.")

            fourcc = cv2.VideoWriter.fourcc(*self._codec)
            writer = cv2.VideoWriter(
                str(output_path),
                fourcc,
                self._fps,
                (width, height),
            )
            if not writer.isOpened():
                raise RuntimeError(f"Could not open video writer for {output_path}.")

            for frame in self._resample_frames(frames, clip_end_time):
                if frame.shape[:2] != (height, width):
                    raise RuntimeError("Frame dimensions changed during incident recording.")
                writer.write(frame)

            writer.release()
            writer = None
            if not output_path.is_file() or output_path.stat().st_size == 0:
                raise RuntimeError(f"Video writer produced no usable file at {output_path}.")

            with self._lock:
                self._last_saved_path = output_path
            print(f"Incident video successfully saved: {output_path}")
        except Exception as exc:
            with self._lock:
                self._last_error = exc
            print(f"Incident video could not be saved: {exc}")
            if output_path.exists():
                try:
                    output_path.unlink()
                except OSError as cleanup_error:
                    print(f"Could not remove incomplete video {output_path}: {cleanup_error}")
        finally:
            if writer is not None:
                try:
                    writer.release()
                except Exception as exc:
                    print(f"Could not release the video writer cleanly: {exc}")
            with self._lock:
                self._writer_thread = None

    def _resample_frames(
        self,
        frames: list[TimedFrame],
        clip_end_time: float,
    ) -> list[np.ndarray]:
        """Sample the captured timeline at output FPS, repeating frames if needed."""
        if not frames:
            return []
        start_time = frames[0][0]
        end_time = max(start_time, clip_end_time)
        duration = max(0.0, end_time - start_time)
        sample_count = max(1, int(duration * self._fps) + 1)

        sampled_frames: list[np.ndarray] = []
        source_index = 0
        for sample_index in range(sample_count):
            target_time = min(start_time + sample_index / self._fps, end_time)
            while (
                source_index + 1 < len(frames)
                and frames[source_index + 1][0] <= target_time
            ):
                source_index += 1
            sampled_frames.append(frames[source_index][1])
        return sampled_frames
