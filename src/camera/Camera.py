from __future__ import annotations

import logging
import threading
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

class Camera:
    """
    Manages the camera feed and provides frames.
    """

    def __init__(
        self,
        source: int | str = 0,
        width: int = 640,
        height: int = 480,
        fps: float = 30.0,
    ) -> None:
        """
        Initializes the video source.
        :param source: Camera index
        :param width: Requested width in pixels.
        :param height: Requested height in pixels.
        :param fps: Requested framerate.
        """
        self._source = source
        self._req_width = width
        self._req_height = height
        self._req_fps = fps

        #Actual values (confirmed)
        self.frame_width = 0
        self.frame_height = 0
        self.fps: float = 0.0

        self._cap: Optional[cv2.VideoCapture] = None
        self._lock = threading.Lock()

    def start(self) -> bool:
        """Opens the camera stream. Returns True if successful."""
        with self._lock:
            self._cap = cv2.VideoCapture(self._source)
            if not self._cap.isOpened():
                logger.error(
                    f"[VideoSource] Error: Source {self._source} "
                    "could not be opened."
                )
                return False

            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._req_width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._req_height)
            self._cap.set(cv2.CAP_PROP_FPS, self._req_fps)

            self.fps = self._cap.get(cv2.CAP_PROP_FPS) or self._req_fps
            self.frame_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            logger.info(
                f"[VideoSource] Started: {self.frame_width}x"
                f"{self.frame_height} @ {self.fps} FPS"
            )
            return True

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        """
        Reads a frame from the stream. Thread-safe.

        :return: (success as bool, frame as np.ndarray or None)
        """
        with self._lock:
            if self._cap is None or not self._cap.isOpened():
                return False, None
            return self._cap.read()

    def release(self) -> None:
        """Releases the camera resources."""
        with self._lock:
            if self._cap is not None:
                self._cap.release()
                self._cap = None
            logger.info("[VideoSource] Stopped and resources released.")

    def is_opened(self) -> bool:
        """Checks if the source is opened."""
        with self._lock:
            return self._cap is not None and self._cap.isOpened()