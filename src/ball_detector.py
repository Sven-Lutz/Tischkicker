"""Ball detection via HSV colour filtering.

Assumes a white ball on a dark green field:
  H = 0–180 (any hue)
  S = 0–40  (very low saturation → white/grey)
  V = 200–255 (high brightness → white)
"""

from __future__ import annotations

import time
from typing import Optional

import cv2
import numpy as np

from src.game_events import BallPosition


class BallDetector:
    DEFAULT_HSV_LOWER = np.array([0, 0, 200], dtype=np.uint8)
    DEFAULT_HSV_UPPER = np.array([180, 40, 255], dtype=np.uint8)

    MIN_RADIUS = 8
    MAX_RADIUS = 40
    MIN_CIRCULARITY = 0.70

    def __init__(self) -> None:
        self._hsv_lower = self.DEFAULT_HSV_LOWER.copy()
        self._hsv_upper = self.DEFAULT_HSV_UPPER.copy()
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def detect(self, frame: np.ndarray) -> Optional[BallPosition]:
        """Return the ball's position in the frame, or None if not found."""
        blurred = cv2.GaussianBlur(frame, (7, 7), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self._hsv_lower, self._hsv_upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = self._pick_best_contour(contours)
        if best is None:
            return None

        (cx, cy), _ = cv2.minEnclosingCircle(best)
        return BallPosition(x=float(cx), y=float(cy), timestamp=time.time())

    def _pick_best_contour(self, contours: tuple) -> Optional[np.ndarray]:
        best = None
        best_area = 0.0
        for c in contours:
            area = cv2.contourArea(c)
            if area < 1:
                continue
            _, radius = cv2.minEnclosingCircle(c)
            if not (self.MIN_RADIUS <= radius <= self.MAX_RADIUS):
                continue
            perimeter = cv2.arcLength(c, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter ** 2)
            if circularity < self.MIN_CIRCULARITY:
                continue
            if area > best_area:
                best_area = area
                best = c
        return best

    def update_hsv_range(self, lower: np.ndarray, upper: np.ndarray) -> None:
        self._hsv_lower = lower.astype(np.uint8)
        self._hsv_upper = upper.astype(np.uint8)
