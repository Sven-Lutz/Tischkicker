from __future__ import annotations

import logging

import cv2


class Camera:
    """
    Manages the camera feed and provides frames.
    """

    def __init__(self, source: int = 0):
        """
        :param source: Camera index (0 = default webcam) or video path
        """
        self.source = source
        self.cap: cv2.VideoCapture | None = None
        self.fps: float = 30.0
        self.frame_width: int = 0
        self.frame_height: int = 0

    def start(self) -> bool:
        """Opens the camera stream. Returns True if successful."""
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            logging.error(f"[Camera] Fehler: Kamera {self.source} konnte nicht geöffnet werden.")
            return False

        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[Camera] Gestartet: {self.frame_width}x{self.frame_height} @ {self.fps} FPS")
        return True

    def read_frame(self):
        """
        Reads a frame from the stream.
        :return: (success: bool, frame: np.ndarray | None)
        """
        if self.cap is None:
            return False, None
        return self.cap.read()

    def stop(self):
        """Releases the camera resource."""
        if self.cap:
            self.cap.release()
            self.cap = None
        print("[Camera] Gestoppt.")