"""Automatic playing-field detection via HSV colour segmentation.

Finds the green felt surface and returns a FieldBounds with corners and a
bounding box. detect_from_frames() samples several frames and uses the median
result for robustness against partial occlusion.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Broad green range — adapted to be more lenient towards shadows/highlights.
_FIELD_HSV_LOWER = np.array([35, 40, 40], dtype=np.uint8)
_FIELD_HSV_UPPER = np.array([85, 255, 255], dtype=np.uint8)

# Field must cover at least this fraction of the frame to be accepted.
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
        """Return a uint8 binary mask (255 inside field) for *frame_shape*."""
        h, w = frame_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        pts = np.array(self.corners, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)
        return mask

    def get_perspective_matrix(self, target_width: int, target_height: int) -> np.ndarray:
        """Calculate the transformation matrix for the Bird's-Eye View."""
        # Source points: The 4 detected corners of the table
        src_pts = np.array(self.corners, dtype=np.float32)

        # Destination points: A perfect rectangle starting at (0,0)
        dst_pts = np.array([
            [0, 0],  # Top-Left
            [target_width - 1, 0],  # Top-Right
            [target_width - 1, target_height - 1],  # Bottom-Right
            [0, target_height - 1]  # Bottom-Left
        ], dtype=np.float32)

        # Calculate Homography matrix
        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        return matrix

class FieldDetector:
    """Detects the green playing field in a single frame or a sequence of frames."""

    def __init__(self) -> None:
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

    def detect(self, frame: np.ndarray) -> Optional[FieldBounds]:
        """Detect the field in a single frame. Returns None on failure."""
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, _FIELD_HSV_LOWER, _FIELD_HSV_UPPER)

        # Morphological operations to fill holes and remove noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            logger.debug("No green contours found.")
            return None

        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        frame_area = frame.shape[0] * frame.shape[1]

        if area < frame_area * _MIN_FIELD_FRACTION:
            logger.debug(
                "Largest contour covers only %.1f %% of frame — too small.",
                100.0 * area / frame_area,
            )
            return None

        # Polygon approximation to find the 4 corners of the table
        perimeter = cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, 0.02 * perimeter, True)

        if len(approx) == 4:
            pts = _order_corners(approx.reshape(4, 2))
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
            logger.debug("Quad fit successful: %s", bounds)
            return bounds

        x, y, w, h = cv2.boundingRect(largest)
        logger.debug(
            "Bounding rect fallback used: x=%d y=%d w=%d h=%d", x, y, w, h
        )
        return FieldBounds.from_rect(x, y, w, h)

    def detect_from_frames(
        self,
        video_source,
        num_frames: int = 15,
    ) -> Optional[FieldBounds]:
        """Sample *num_frames* frames and return the median field bounds."""
        results: list[FieldBounds] = []

        for _ in range(num_frames):
            ok, frame = video_source.read()
            if not ok or frame is None:
                continue
            bounds = self.detect(frame)
            if bounds is not None:
                results.append(bounds)

        if not results:
            logger.warning(
                "Field not detected in any of %d frames.", num_frames
            )
            return None

        if len(results) == 1:
            return results[0]

        # Calculate the median coordinates to ignore outlier frames
        x1s = sorted(b.x1 for b in results)
        y1s = sorted(b.y1 for b in results)
        x2s = sorted(b.x2 for b in results)
        y2s = sorted(b.y2 for b in results)
        mid = len(results) // 2

        x1, y1, x2, y2 = x1s[mid], y1s[mid], x2s[mid], y2s[mid]
        bounds = FieldBounds.from_rect(x1, y1, x2 - x1, y2 - y1)

        logger.info(
            "Field detected from %d/%d frames: x=%d–%d y=%d–%d",
            len(results),
            num_frames,
            x1,
            x2,
            y1,
            y2,
        )
        return bounds


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Sort four (x, y) points into [TL, TR, BR, BL] order."""
    pts = pts[np.argsort(pts[:, 1])]
    top = pts[:2][np.argsort(pts[:2, 0])]
    bottom = pts[2:][np.argsort(pts[2:, 0])]
    return np.array([top[0], top[1], bottom[1], bottom[0]])