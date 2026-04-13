"""Automatic playing-field detection via HSV colour segmentation.

The green felt surface is found by:
  1. Blurring to suppress noise
  2. HSV thresholding for typical green felt colours
  3. Morphological cleanup
  4. Picking the largest contour that covers >10 % of the frame
  5. Approximating it as a quadrilateral (or falling back to a bounding rect)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Broad HSV range that covers most green felt under varying lighting.
# Hue 30–90 captures yellow-green through cyan-green.
_FIELD_HSV_LOWER = np.array([30, 30, 25], dtype=np.uint8)
_FIELD_HSV_UPPER = np.array([95, 255, 210], dtype=np.uint8)

# Minimum fraction of total frame area that must be green for detection to succeed.
_MIN_FIELD_FRACTION = 0.10


@dataclass
class FieldBounds:
    """Detected boundaries of the playing field."""

    corners: list[tuple[int, int]]  # [TL, TR, BR, BL] in pixel coords
    x1: int
    y1: int
    x2: int
    y2: int

    @classmethod
    def from_rect(cls, x: int, y: int, w: int, h: int) -> "FieldBounds":
        return cls(
            corners=[(x, y), (x + w, y), (x + w, y + h), (x, y + h)],
            x1=x,
            y1=y,
            x2=x + w,
            y2=y + h,
        )

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    def create_mask(self, frame_shape: tuple[int, int]) -> np.ndarray:
        """Return a binary mask (uint8, 255 inside field) of the given frame shape."""
        h, w = frame_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        pts = np.array(self.corners, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)
        return mask


class FieldDetector:
    """Detects the green playing field in a single frame or a sequence of frames."""

    def __init__(self) -> None:
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

    # ── Public API ────────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> Optional[FieldBounds]:
        """Detect the field in a single frame. Returns None on failure."""
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, _FIELD_HSV_LOWER, _FIELD_HSV_UPPER)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            logger.debug("FieldDetector: no contours found")
            return None

        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        frame_area = frame.shape[0] * frame.shape[1]

        if area < frame_area * _MIN_FIELD_FRACTION:
            logger.debug(
                "FieldDetector: largest contour covers only %.1f %% of frame",
                100.0 * area / frame_area,
            )
            return None

        # Try to fit a quadrilateral
        peri = cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, 0.02 * peri, True)

        if len(approx) == 4:
            pts = approx.reshape(4, 2)
            pts = _order_corners(pts)
            x1 = int(pts[:, 0].min())
            y1 = int(pts[:, 1].min())
            x2 = int(pts[:, 0].max())
            y2 = int(pts[:, 1].max())
            bounds = FieldBounds(
                corners=[(int(p[0]), int(p[1])) for p in pts],
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
            )
            logger.debug("FieldDetector: quad fit — %s", bounds)
            return bounds

        # Fallback: axis-aligned bounding rect
        x, y, w, h = cv2.boundingRect(largest)
        logger.debug("FieldDetector: bounding rect fallback — x=%d y=%d w=%d h=%d", x, y, w, h)
        return FieldBounds.from_rect(x, y, w, h)

    def detect_from_frames(
        self,
        video_source,
        num_frames: int = 15,
    ) -> Optional[FieldBounds]:
        """Sample *num_frames* frames from *video_source* and return the median bounds.

        Uses the median of all successful detections so that occasional bad frames
        (e.g. a hand covering part of the field) do not throw off the result.
        """
        results: list[FieldBounds] = []

        for _ in range(num_frames):
            ok, frame = video_source.read()
            if not ok or frame is None:
                continue
            bounds = self.detect(frame)
            if bounds is not None:
                results.append(bounds)

        if not results:
            logger.warning("FieldDetector: could not detect field in %d frames", num_frames)
            return None

        if len(results) == 1:
            return results[0]

        # Median of each boundary coordinate for robustness.
        x1s = sorted(b.x1 for b in results)
        y1s = sorted(b.y1 for b in results)
        x2s = sorted(b.x2 for b in results)
        y2s = sorted(b.y2 for b in results)
        mid = len(results) // 2

        x1, y1, x2, y2 = x1s[mid], y1s[mid], x2s[mid], y2s[mid]
        # Recompute corners from the median bounding box
        bounds = FieldBounds.from_rect(x1, y1, x2 - x1, y2 - y1)
        logger.info(
            "FieldDetector: field detected from %d/%d frames — x1=%d y1=%d x2=%d y2=%d",
            len(results),
            num_frames,
            x1,
            y1,
            x2,
            y2,
        )
        return bounds


# ── Helpers ────────────────────────────────────────────────────────────────────


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Sort 4 (x,y) points into [TL, TR, BR, BL] order."""
    # Sort by y then split into top/bottom halves
    idx = np.argsort(pts[:, 1])
    pts = pts[idx]
    top = pts[:2][np.argsort(pts[:2, 0])]     # TL, TR  (smaller x first)
    bottom = pts[2:][np.argsort(pts[2:, 0])]  # BL, BR  (smaller x first)
    return np.array([top[0], top[1], bottom[1], bottom[0]])  # TL TR BR BL
