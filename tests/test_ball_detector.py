"""Tests for BallDetector."""

import numpy as np
import pytest

from src.ball_detector import BallDetector


# ── helpers ───────────────────────────────────────────────────────────────────

def _white_ball_frame(cx: int = 320, cy: int = 240, radius: int = 12) -> np.ndarray:
    """640×480 BGR frame: dark green background + white filled circle."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:, :] = (0, 100, 0)  # dark green (BGR)
    cv2_import = __import__("cv2")
    cv2_import.circle(frame, (cx, cy), radius, (255, 255, 255), -1)
    return frame


def _dark_frame() -> np.ndarray:
    """All-black 640×480 frame — no ball."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


def _green_frame() -> np.ndarray:
    """Uniform green frame — no ball."""
    f = np.zeros((480, 640, 3), dtype=np.uint8)
    f[:, :] = (0, 100, 0)
    return f


# ── basic detection ───────────────────────────────────────────────────────────

class TestBasicDetection:
    def test_detect_returns_none_on_dark_frame(self):
        bd = BallDetector()
        assert bd.detect(_dark_frame()) is None

    def test_detect_returns_none_on_uniform_green_frame(self):
        bd = BallDetector()
        assert bd.detect(_green_frame()) is None

    def test_detect_finds_white_ball(self):
        bd = BallDetector()
        result = bd.detect(_white_ball_frame(cx=320, cy=240, radius=15))
        assert result is not None

    def test_detected_position_close_to_centre(self):
        """Detected x/y should be within 10 px of the true ball centre."""
        bd = BallDetector()
        result = bd.detect(_white_ball_frame(cx=200, cy=150, radius=14))
        assert result is not None
        assert abs(result.x - 200) <= 10
        assert abs(result.y - 150) <= 10

    def test_detected_position_top_left(self):
        bd = BallDetector()
        result = bd.detect(_white_ball_frame(cx=80, cy=80, radius=14))
        assert result is not None
        assert abs(result.x - 80) <= 10
        assert abs(result.y - 80) <= 10

    def test_detected_position_bottom_right(self):
        bd = BallDetector()
        result = bd.detect(_white_ball_frame(cx=560, cy=400, radius=14))
        assert result is not None
        assert abs(result.x - 560) <= 10
        assert abs(result.y - 400) <= 10

    def test_ball_too_small_not_detected(self):
        """A very small blob (radius < MIN_RADIUS) should be ignored."""
        frame = _green_frame()
        import cv2
        cv2.circle(frame, (320, 240), 3, (255, 255, 255), -1)  # radius 3 < MIN_RADIUS=6
        bd = BallDetector()
        assert bd.detect(frame) is None

    def test_ball_too_large_not_detected(self):
        """A very large blob (radius > MAX_RADIUS) should be ignored."""
        frame = _green_frame()
        import cv2
        cv2.circle(frame, (320, 240), 60, (255, 255, 255), -1)  # > MAX_RADIUS=45
        bd = BallDetector()
        assert bd.detect(frame) is None

    def test_non_circular_blob_not_detected(self):
        """A rectangular white patch (low circularity) should not be detected."""
        frame = _green_frame()
        import cv2
        cv2.rectangle(frame, (100, 100), (200, 130), (255, 255, 255), -1)  # wide thin rect
        bd = BallDetector()
        assert bd.detect(frame) is None

    def test_result_has_detected_true(self):
        bd = BallDetector()
        result = bd.detect(_white_ball_frame())
        assert result is not None
        assert result.detected is True

    def test_result_has_timestamp(self):
        bd = BallDetector()
        result = bd.detect(_white_ball_frame())
        assert result is not None
        assert result.timestamp > 0.0


# ── field mask ────────────────────────────────────────────────────────────────

class TestFieldMask:
    def test_mask_excludes_ball_outside_field(self):
        """Ball placed outside the field mask must not be detected."""
        bd = BallDetector()
        # Ball at (50, 50)
        frame = _white_ball_frame(cx=50, cy=50, radius=14)
        # Mask: only the right half of the frame is valid field
        mask = np.zeros((480, 640), dtype=np.uint8)
        mask[:, 320:] = 255
        bd.set_field_mask(mask)
        assert bd.detect(frame) is None

    def test_mask_allows_ball_inside_field(self):
        """Ball inside the mask region must still be detected."""
        bd = BallDetector()
        frame = _white_ball_frame(cx=400, cy=240, radius=14)
        mask = np.zeros((480, 640), dtype=np.uint8)
        mask[:, 320:] = 255  # right half is the field
        bd.set_field_mask(mask)
        result = bd.detect(frame)
        assert result is not None
        assert abs(result.x - 400) <= 10

    def test_clear_mask_restores_full_frame_search(self):
        """After clearing the mask, balls anywhere should be detected again."""
        bd = BallDetector()
        frame = _white_ball_frame(cx=50, cy=50, radius=14)
        mask = np.zeros((480, 640), dtype=np.uint8)
        mask[:, 320:] = 255
        bd.set_field_mask(mask)
        assert bd.detect(frame) is None  # masked out
        bd.clear_field_mask()
        assert bd.detect(frame) is not None  # now detectable


# ── HSV range update ──────────────────────────────────────────────────────────

class TestHSVRange:
    def test_custom_hsv_detects_coloured_ball(self):
        """After reconfiguring HSV for orange, an orange ball should be found."""
        import cv2

        frame = _green_frame()
        # Draw a bright orange circle: HSV ≈ (15, 200, 220) → BGR ≈ (0, 127, 220)
        cv2.circle(frame, (320, 240), 15, (0, 127, 220), -1)

        bd = BallDetector()
        # Default white thresholds: should NOT find orange
        assert bd.detect(frame) is None

        # Orange: H 5–25, S 100–255, V 100–255
        bd.update_hsv_range(
            np.array([5, 100, 100], dtype=np.uint8),
            np.array([25, 255, 255], dtype=np.uint8),
        )
        result = bd.detect(frame)
        assert result is not None
        assert abs(result.x - 320) <= 12
