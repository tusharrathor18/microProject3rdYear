"""Tests for rolling-buffer and simulated incident video recording."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.evidence.video_recorder import VideoRecorder


class FakeVideoWriter:
    """Small VideoWriter substitute that records frames to memory."""

    instances: list["FakeVideoWriter"] = []

    @staticmethod
    def fourcc(*_characters: str) -> int:
        return sum(ord(character) for character in _characters)

    def __init__(
        self,
        output_path: str,
        _fourcc: int,
        _fps: float,
        frame_size: tuple[int, int],
    ) -> None:
        self.output_path = Path(output_path)
        self.frame_size = frame_size
        self.fourcc_code = _fourcc
        self.fps = _fps
        self.frames: list[np.ndarray] = []
        self.released = False
        self.instances.append(self)

    def isOpened(self) -> bool:
        return True

    def write(self, frame: np.ndarray) -> None:
        self.frames.append(frame.copy())

    def release(self) -> None:
        self.released = True
        self.output_path.write_bytes(b"fake video")


class VideoRecorderTests(unittest.TestCase):
    def setUp(self) -> None:
        FakeVideoWriter.instances.clear()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.evidence_dir = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def recorder(self, clock) -> VideoRecorder:
        return VideoRecorder(
            pre_event_seconds=2.0,
            post_event_seconds=1.0,
            fps=2.0,
            evidence_dir=self.evidence_dir,
            clock=clock,
        )

    @patch("src.evidence.video_recorder.cv2.VideoWriter", FakeVideoWriter)
    def test_retains_recent_frames_and_writes_pre_and_post_event_clip(self) -> None:
        now = [0.0]
        recorder = self.recorder(lambda: now[0])
        recorder.add_frame(np.full((4, 6, 3), 0, dtype=np.uint8), timestamp=0.0)
        recorder.add_frame(np.full((4, 6, 3), 1, dtype=np.uint8), timestamp=1.0)
        recorder.add_frame(np.full((4, 6, 3), 2, dtype=np.uint8), timestamp=2.1)

        now[0] = 2.1
        expected_path = recorder.trigger()
        self.assertIn("incident_", expected_path.name)
        self.assertTrue(recorder.is_recording)

        recorder.add_frame(np.full((4, 6, 3), 3, dtype=np.uint8), timestamp=2.6)
        recorder.add_frame(np.full((4, 6, 3), 4, dtype=np.uint8), timestamp=3.1)
        self.assertTrue(recorder.wait_for_completion(timeout=5.0))

        self.assertFalse(recorder.is_recording)
        self.assertEqual(recorder.last_saved_path, expected_path)
        self.assertTrue(expected_path.is_file())
        writer = FakeVideoWriter.instances[0]
        self.assertTrue(writer.released)
        self.assertEqual(writer.frame_size, (6, 4))
        self.assertEqual(len(writer.frames), 5)
        self.assertEqual(int(writer.frames[0][0, 0, 0]), 1)
        self.assertEqual(int(writer.frames[-1][0, 0, 0]), 3)
        recorder.close()

    @patch("src.evidence.video_recorder.cv2.VideoWriter", FakeVideoWriter)
    def test_close_finalizes_clip_if_post_event_window_is_incomplete(self) -> None:
        now = [5.0]
        recorder = self.recorder(lambda: now[0])
        recorder.add_frame(np.ones((4, 6, 3), dtype=np.uint8), timestamp=5.0)
        recorder.trigger()
        now[0] = 5.4
        recorder.add_frame(np.full((4, 6, 3), 2, dtype=np.uint8), timestamp=5.4)

        recorder.close()

        self.assertFalse(recorder.is_recording)
        saved_path = recorder.last_saved_path
        self.assertIsNotNone(saved_path)
        assert saved_path is not None
        self.assertTrue(saved_path.is_file())
        self.assertEqual(len(FakeVideoWriter.instances), 1)

    def test_rejects_second_trigger_while_clip_is_active(self) -> None:
        now = [1.0]
        recorder = self.recorder(lambda: now[0])
        recorder.add_frame(np.ones((4, 6, 3), dtype=np.uint8), timestamp=1.0)
        recorder.trigger()

        with self.assertRaises(RuntimeError):
            recorder.trigger()

        recorder.close()

    def test_rejects_invalid_frame_dimensions_or_channels(self) -> None:
        recorder = self.recorder(lambda: 1.0)
        with self.assertRaises(ValueError):
            recorder.add_frame(np.zeros((4, 6), dtype=np.uint8))
        with self.assertRaises(ValueError):
            recorder.add_frame(np.zeros((4, 6, 4), dtype=np.uint8))
        recorder.close()


if __name__ == "__main__":
    unittest.main()
