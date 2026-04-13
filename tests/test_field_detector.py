"""Tests for FieldDetector and FieldBounds."""

import numpy as np
import pytest

from src.field_detector import FieldBounds, FieldDetector, _order_corners


# ── FieldBounds helpers ───────────────────────────────────────────────────────

class TestFieldBounds:
    def test_from_rect_sets_corners(self):
        fb = FieldBounds.from_rect(10, 20, 200, 100)
        assert fb.x1 == 10
        assert fb.y1 == 20
        assert fb.x2 == 210
        assert fb.y2 == 120
        assert len(fb.corners) == 4

    def test_width_height(self):
        fb = FieldBounds.from_rect(0, 0, 640, 480)
        assert fb.width == 640
        assert fb.height == 480

    def test_create_mask_fills_interior(self):
        fb = FieldBounds.from_rect(100, 50, 200, 150)  # x:100-300, y:50-200
        mask = fb.create_mask((480, 640))
        assert mask.shape == (480, 640)
        # A point inside the rectangle is 255
        assert mask[125, 200] == 255
        # A point outside is 0
        assert mask[10, 10] == 0
        assert mask[400, 400] == 0

    def test_create_mask_shape_matches_frame(self):
        fb = FieldBounds.from_rect(0, 0, 320, 240)
        mask = fb.create_mask((480, 640))
        assert mask.shape == (480, 640)
        assert mask.dtype == np.uint8


# ── _order_corners helper ─────────────────────────────────────────────────────

class TestOrderCorners:
    def test_canonical_order(self):
        # Supply in BR, TL, TR, BL order — expect TL TR BR BL back
        pts = np.array([
            [640, 480],  # BR
            [0,   0],    # TL
            [640, 0],    # TR
            [0,   480],  # BL
        ])
        ordered = _order_corners(pts)
        assert tuple(ordered[0]) == (0, 0)    # TL
        assert tuple(ordered[1]) == (640, 0)  # TR
        assert tuple(ordered[2]) == (640, 480)  # BR
        assert tuple(ordered[3]) == (0, 480)  # BL


# ── FieldDetector.detect on synthetic frames ──────────────────────────────────

class TestFieldDetector:
    def _green_frame(self, h=480, w=640, x1=80, y1=60, x2=560, y2=420) -> np.ndarray:
        """Create a black frame with a solid green rectangle."""
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        # Green in BGR: (0, 180, 0) — HSV hue ≈ 60, S≈255, V≈180 → within range
        frame[y1:y2, x1:x2] = (0, 160, 0)
        return frame

    def test_detect_returns_none_on_black_frame(self):
        detector = FieldDetector()
        black = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.detect(black)
        assert result is None

    def test_detect_returns_none_when_green_too_small(self):
        """A tiny green patch (<10 % of frame area) should not be detected."""
        detector = FieldDetector()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # 20x20 = 400 px < 10% of 307200
        frame[10:30, 10:30] = (0, 160, 0)
        result = detector.detect(frame)
        assert result is None

    def test_detect_finds_large_green_field(self):
        """A green rectangle covering >10 % of frame area must be found."""
        detector = FieldDetector()
        frame = self._green_frame()
        result = detector.detect(frame)
        assert result is not None

    def test_detect_approximate_bounds(self):
        """Detected bounds should roughly match the synthetic green rectangle."""
        detector = FieldDetector()
        frame = self._green_frame(x1=80, y1=60, x2=560, y2=420)
        result = detector.detect(frame)
        assert result is not None
        # Allow ±20 px tolerance for HSV blurring / morphology effects
        tol = 20
        assert abs(result.x1 - 80) <= tol
        assert abs(result.y1 - 60) <= tol
        assert abs(result.x2 - 560) <= tol
        assert abs(result.y2 - 420) <= tol

    def test_detect_width_height_positive(self):
        detector = FieldDetector()
        frame = self._green_frame()
        result = detector.detect(frame)
        assert result is not None
        assert result.width > 0
        assert result.height > 0

    def test_detect_from_frames_uses_multiple_readings(self):
        """detect_from_frames returns bounds even with some bad frames mixed in."""

        class _MockSource:
            def __init__(self, good_frame, num_total, num_bad):
                self._frame = good_frame
                self._total = num_total
                self._bad = num_bad
                self._count = 0

            def read(self):
                self._count += 1
                if self._count <= self._bad:
                    bad = np.zeros_like(self._frame)
                    return True, bad
                return True, self._frame.copy()

        detector = FieldDetector()
        good = self._green_frame()
        src = _MockSource(good, num_total=15, num_bad=5)
        result = detector.detect_from_frames(src, num_frames=15)
        assert result is not None

    def test_detect_from_frames_none_when_all_bad(self):
        """Returns None when no frame contains a detectable field."""

        class _BlackSource:
            def read(self):
                return True, np.zeros((480, 640, 3), dtype=np.uint8)

        detector = FieldDetector()
        result = detector.detect_from_frames(_BlackSource(), num_frames=5)
        assert result is None

    def test_detect_from_frames_handles_read_failures(self):
        """read() returning ok=False should not crash detect_from_frames."""

        class _BadSource:
            def read(self):
                return False, None

        detector = FieldDetector()
        result = detector.detect_from_frames(_BadSource(), num_frames=5)
        assert result is None

    def test_detected_bounds_have_four_corners(self):
        detector = FieldDetector()
        frame = self._green_frame()
        result = detector.detect(frame)
        assert result is not None
        assert len(result.corners) == 4
