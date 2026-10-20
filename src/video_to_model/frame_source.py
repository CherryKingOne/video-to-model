"""Incremental video decoding so a whole video never has to sit in memory."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np


@dataclass(frozen=True)
class FrameInfo:
    """Geometry and sampling of a source video after upstream's decoding rules."""

    height: int
    width: int
    fps: float
    stride: int
    frame_count: int | None  # estimate from the container; None when it is unknown


class FrameSource:
    """Yield sampled frames one by one, mirroring ``dc_utils.read_video_frames``.

    The upstream helper decodes an entire video into one ``numpy`` array, which makes
    peak memory grow linearly with video length. This class keeps the same sampling
    rules (fps stride, ``max_res`` downscale, ``max_len`` cut-off) but returns a
    generator so callers only ever hold a sliding window of frames.
    """

    def __init__(self, path: Path, *, target_fps: float = -1, max_res: int = 1280, max_len: int = -1) -> None:
        import cv2

        self._cv2 = cv2
        self.path = Path(path)
        self.max_len = max_len

        capture = cv2.VideoCapture(str(self.path))
        if not capture.isOpened():
            raise OSError(f"Could not open video: {self.path}")
        try:
            original_fps = capture.get(cv2.CAP_PROP_FPS)
            original_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            original_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            capture.release()

        if original_fps <= 0 or original_height <= 0 or original_width <= 0:
            raise ValueError(f"Cannot read video geometry from {self.path}")

        height, width = original_height, original_width
        self._resize = False
        if max_res > 0 and max(height, width) > max_res:
            scale = max_res / max(height, width)
            height = round(height * scale)
            width = round(width * scale)
            self._resize = True

        fps = original_fps if target_fps < 0 else target_fps
        stride = max(round(original_fps / fps), 1)

        frame_count: int | None = None
        if total_frames > 0:
            decoded = min(total_frames, max_len) if max_len > 0 else total_frames
            frame_count = -(-decoded // stride)  # ceil

        self.info = FrameInfo(height, width, float(fps), stride, frame_count)

    def frames(self) -> Iterator[np.ndarray]:
        """Decode and yield RGB frames, skipping ``stride - 1`` frames between kept ones."""
        cv2 = self._cv2
        info = self.info
        capture = cv2.VideoCapture(str(self.path))
        frame_count = 0
        try:
            while capture.isOpened():
                ret, frame = capture.read()
                if not ret or (self.max_len > 0 and frame_count >= self.max_len):
                    break
                if frame_count % info.stride == 0:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    if self._resize:
                        frame = cv2.resize(frame, (info.width, info.height))
                    yield frame
                frame_count += 1
        finally:
            capture.release()

    def counted(self) -> Iterator[np.ndarray]:
        """Yield frames from :meth:`frames` while recording how many were produced."""
        self.frames_read = 0
        for frame in self.frames():
            self.frames_read += 1
            yield frame

    frames_read = 0
