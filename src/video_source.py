"""Camera frame delivery. Real camera and mock source for testing."""

from __future__ import annotations

import threading
from typing import Optional

import cv2
import numpy as np


class VideoSource:
    """Thread-safe wrapper around cv2.VideoCapture."""

    def __init__(self, camera_index: int = 0, width: int = 640, height: int = 480, fps: float = 30.0) -> None:
        self._index = camera_index
        self._width = width
        self._height = height
        self._fps = fps
        self._cap: Optional[cv2.VideoCapture] = None
        self._lock = threading.Lock()

    def open(self) -> bool:
        self._cap = cv2.VideoCapture(self._index)
        if not self._cap.isOpened():
            return False
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self._cap.set(cv2.CAP_PROP_FPS, self._fps)
        return True

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        with self._lock:
            if self._cap is None or not self._cap.isOpened():
                return False, None
            return self._cap.read()

    def release(self) -> None:
        with self._lock:
            if self._cap is not None:
                self._cap.release()
                self._cap = None

    def is_opened(self) -> bool:
        with self._lock:
            return self._cap is not None and self._cap.isOpened()

    @property
    def frame_width(self) -> int:
        """Actual frame width reported by the capture device (falls back to requested width)."""
        with self._lock:
            if self._cap is not None and self._cap.isOpened():
                return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            return self._width

    @property
    def frame_height(self) -> int:
        """Actual frame height reported by the capture device (falls back to requested height)."""
        with self._lock:
            if self._cap is not None and self._cap.isOpened():
                return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return self._height


class MockVideoSource(VideoSource):
    """Serves frames from a static image or video file — useful for tests and demos."""

    def __init__(self, source: str, width: int = 640, height: int = 480) -> None:
        super().__init__(camera_index=0, width=width, height=height)
        self._source = source
        self._frame: Optional[np.ndarray] = None
        self._opened = False

    def open(self) -> bool:
        # Try as image first, then as video file
        img = cv2.imread(self._source)
        if img is not None:
            self._frame = cv2.resize(img, (self._width, self._height))
            self._opened = True
            return True
        # Try as video
        self._cap = cv2.VideoCapture(self._source)
        if self._cap.isOpened():
            self._opened = True
            return True
        return False

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        if not self._opened:
            return False, None
        if self._frame is not None:
            # Static image: always return the same frame
            return True, self._frame.copy()
        return super().read()

    def is_opened(self) -> bool:
        return self._opened
