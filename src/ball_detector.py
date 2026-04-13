"""Ball detection using Background Subtraction (MOG2) and shape heuristics.

Instead of static colour filtering, this module detects moving objects
and identifies the ball based on its expected pixel area and aspect ratio.
Supply a field mask via set_field_mask() to restrict the search area.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import cv2
import numpy as np

from src.game_events import BallPosition

logger = logging.getLogger(__name__)


class BallDetector:
    # Heuristics for ball detection (tune these based on your video resolution)
    MIN_AREA = 30.0
    MAX_AREA = 800.0
    MIN_ASPECT_RATIO = 0.5   # Allows for motion blur elongation
    MAX_ASPECT_RATIO = 2.5

    def __init__(self) -> None:
        # Initialize the background subtractor
        # history: length of the history (frames)
        # varThreshold: distance threshold for pixel-background matching
        self._back_sub = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=50, detectShadows=False
        )
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self._field_mask: Optional[np.ndarray] = None
        logger.debug("BallDetector initialized with MOG2 background subtraction.")

    def set_field_mask(self, mask: np.ndarray) -> None:
        """Restrict detection to pixels where *mask* is non-zero."""
        self._field_mask = mask.astype(np.uint8)
        logger.debug("Field mask applied (%dx%d).", mask.shape[1], mask.shape[0])

    def clear_field_mask(self) -> None:
        """Remove the field mask; the full frame is searched."""
        self._field_mask = None
        logger.debug("Field mask cleared.")

    def detect(self, frame: np.ndarray) -> Optional[BallPosition]:
        """Return the ball position in *frame* based on movement, or None if not found."""
        roi = frame

        # 1. Restrict search area to the table to save processing power and avoid noise
        if self._field_mask is not None:
            if self._field_mask.shape != frame.shape[:2]:
                self._field_mask = cv2.resize(
                    self._field_mask,
                    (frame.shape[1], frame.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            roi = cv2.bitwise_and(frame, frame, mask=self._field_mask)

        # 2. Extract foreground (moving objects)
        fg_mask = self._back_sub.apply(roi)

        # 3. Clean up noise (morphological opening removes small speckles)
        cleaned = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self._kernel)

        # 4. Find contours of moving objects
        contours, _ = cv2.findContours(
            cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        best_contour = self._pick_best_contour(contours)
        if best_contour is None:
            return None

        # 5. Calculate the center of mass using moments
        moments = cv2.moments(best_contour)
        if moments["m00"] > 0:
            cx = moments["m10"] / moments["m00"]
            cy = moments["m01"] / moments["m00"]
            return BallPosition(x=float(cx), y=float(cy), timestamp=time.time())

        return None

    def _pick_best_contour(self, contours: tuple) -> Optional[np.ndarray]:
        """Filter contours based on size and aspect ratio to distinguish ball from players."""
        best: Optional[np.ndarray] = None
        best_area = 0.0

        for c in contours:
            area = cv2.contourArea(c)

            # Filter 1: Size
            if not (self.MIN_AREA <= area <= self.MAX_AREA):
                continue

            x, y, w, h = cv2.boundingRect(c)
            if h == 0:
                continue

            aspect_ratio = float(w) / float(h)

            # Filter 2: Aspect Ratio (Players are tall/thin, ball is square-ish or blurred)
            if not (self.MIN_ASPECT_RATIO <= aspect_ratio <= self.MAX_ASPECT_RATIO):
                continue

            if area > best_area:
                best_area = area
                best = c

        return best