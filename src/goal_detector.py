"""Goal detection with 2-D goal zones and edge-triggered scoring.

Improvements over the original 1-D approach:
* **Edge detection** — a goal fires only when the ball *enters* a zone, not on
  every frame it spends inside it.  This eliminates the need for a long cooldown
  to avoid double-counting.
* **Cooldown** is kept as a secondary guard: even if the ball briefly leaves and
  re-enters a zone quickly it will not score again.
* **2-D goal zones** — when the field has been calibrated the goal height is
  bounded correctly so that a ball near the lateral edge of the field that
  drifts past x=goal_line but is NOT within the goal opening is ignored.
  Fall-back (no y-constraint) is available for unit tests and when the field
  has not yet been calibrated.
* **``configure_from_corners()``** accepts the four field corners returned by
  ``FieldDetector`` and derives the correct 2-D goal rectangles automatically.

Public interface is backward-compatible with the original class so that all
existing tests continue to pass.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from src.game_events import BallPosition, EventType, GameEvent, Team

logger = logging.getLogger(__name__)

# Fraction of field width used as goal depth (distance behind the goal line).
_GOAL_DEPTH_RATIO = 0.04
# Fraction of field height that the goal opening spans.
_GOAL_HEIGHT_RATIO = 0.28


class _GoalZone:
    """A 2-D rectangular region that triggers a goal when entered."""

    def __init__(
        self,
        name: str,
        x: int,
        y: int,
        w: int,
        h: int,
        scoring_team: Team,
        use_y_bounds: bool = True,
    ) -> None:
        self.name = name
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.scoring_team = scoring_team
        self._use_y_bounds = use_y_bounds

    def contains(self, px: float, py: float) -> bool:
        in_x = self.x <= px <= self.x + self.w
        if not in_x:
            return False
        if self._use_y_bounds:
            return self.y <= py <= self.y + self.h
        return True


class GoalDetector:
    """Detects when the ball enters the left or right goal zone.

    Constructor parameters (unchanged for backward compatibility):
        field_x1, field_x2  — horizontal field boundaries in pixels
        goal_zone_width     — depth of each goal zone in pixels
        cooldown_frames     — minimum frames between two goal events

    Use ``configure_from_corners()`` to enable proper 2-D detection after
    field calibration.
    """

    DEFAULT_GOAL_ZONE_WIDTH = 40
    DEFAULT_COOLDOWN_FRAMES = 45  # 1.5 s at 30 fps

    def __init__(
        self,
        field_x1: int = 0,
        field_x2: int = 640,
        goal_zone_width: int = DEFAULT_GOAL_ZONE_WIDTH,
        cooldown_frames: int = DEFAULT_COOLDOWN_FRAMES,
    ) -> None:
        self._field_x1 = field_x1
        self._field_x2 = field_x2
        self._goal_zone_width = goal_zone_width
        self._cooldown_frames = cooldown_frames

        self._score_left = 0
        self._score_right = 0
        self._frames_since_goal = cooldown_frames  # start ready to detect

        # Edge-detection state: was ball in each zone on the previous frame?
        self._prev_in_left = False
        self._prev_in_right = False

        # 2-D goal zones (set by configure_from_corners)
        self._zone_left: Optional[_GoalZone] = None
        self._zone_right: Optional[_GoalZone] = None

        # Build default 1-D-style zones (no y bounds)
        self._rebuild_default_zones()

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def score_left(self) -> int:
        return self._score_left

    @property
    def score_right(self) -> int:
        return self._score_right

    # ── Configuration ─────────────────────────────────────────────────────────

    def configure_from_corners(
        self,
        corners: list[tuple[int, int]],
    ) -> None:
        """Derive proper 2-D goal zones from field corner coordinates.

        *corners* must be in [TL, TR, BR, BL] order (as returned by
        ``FieldDetector``).
        """
        if len(corners) < 4:
            logger.warning("configure_from_corners: need exactly 4 corners, got %d", len(corners))
            return

        tl, tr, br, bl = corners

        field_width = int(((tr[0] - tl[0]) + (br[0] - bl[0])) / 2)
        field_height = int(((bl[1] - tl[1]) + (br[1] - tr[1])) / 2)

        goal_depth = max(8, int(field_width * _GOAL_DEPTH_RATIO))
        goal_height = max(20, int(field_height * _GOAL_HEIGHT_RATIO))

        left_mid_x = int((tl[0] + bl[0]) / 2)
        left_mid_y = int((tl[1] + bl[1]) / 2)
        right_mid_x = int((tr[0] + br[0]) / 2)
        right_mid_y = int((tr[1] + br[1]) / 2)

        # Left goal zone: ball enters from the right → RIGHT team scores
        self._zone_left = _GoalZone(
            name="Left",
            x=left_mid_x - goal_depth,
            y=left_mid_y - goal_height // 2,
            w=goal_depth,
            h=goal_height,
            scoring_team=Team.RIGHT,
            use_y_bounds=True,
        )

        # Right goal zone: ball enters from the left → LEFT team scores
        self._zone_right = _GoalZone(
            name="Right",
            x=right_mid_x,
            y=right_mid_y - goal_height // 2,
            w=goal_depth,
            h=goal_height,
            scoring_team=Team.LEFT,
            use_y_bounds=True,
        )

        # Update 1-D fallback bounds too
        self._field_x1 = tl[0]
        self._field_x2 = tr[0]

        logger.info(
            "GoalDetector 2-D zones configured: left x=%d–%d y=%d–%d | right x=%d–%d y=%d–%d",
            self._zone_left.x,
            self._zone_left.x + self._zone_left.w,
            self._zone_left.y,
            self._zone_left.y + self._zone_left.h,
            self._zone_right.x,
            self._zone_right.x + self._zone_right.w,
            self._zone_right.y,
            self._zone_right.y + self._zone_right.h,
        )

    def update_field_bounds(
        self,
        field_x1: int,
        field_x2: int,
        field_y1: int = 0,
        field_y2: int = 480,
    ) -> None:
        """Update goal zones from simple field bounding box (no corners)."""
        self._field_x1 = field_x1
        self._field_x2 = field_x2
        self._rebuild_default_zones(field_y1, field_y2)

    # ── Main update loop ──────────────────────────────────────────────────────

    def update(self, position: Optional[BallPosition]) -> Optional[GameEvent]:
        """Call once per frame. Returns a GameEvent if a goal was detected."""
        self._frames_since_goal += 1

        if position is None or not position.detected:
            # Ball lost — reset edge-detection state so the next detection
            # doesn't immediately re-fire if the ball reappears in a zone.
            self._prev_in_left = False
            self._prev_in_right = False
            return None

        x, y = position.x, position.y

        in_left = self._zone_left.contains(x, y) if self._zone_left else (
            x <= self._field_x1 + self._goal_zone_width
        )
        in_right = self._zone_right.contains(x, y) if self._zone_right else (
            x >= self._field_x2 - self._goal_zone_width
        )

        event: Optional[GameEvent] = None

        # Left goal zone — RIGHT team scores
        if in_left and not self._prev_in_left and self._frames_since_goal >= self._cooldown_frames:
            self._score_right += 1
            self._frames_since_goal = 0
            event = GameEvent(
                event_type=EventType.GOAL,
                timestamp=position.timestamp,
                team=Team.RIGHT,
                score_left=self._score_left,
                score_right=self._score_right,
                description=f"Tor Rechts · {self._score_left}:{self._score_right}",
            )
            logger.info("Goal RIGHT! Score %d:%d", self._score_left, self._score_right)

        # Right goal zone — LEFT team scores
        elif in_right and not self._prev_in_right and self._frames_since_goal >= self._cooldown_frames:
            self._score_left += 1
            self._frames_since_goal = 0
            event = GameEvent(
                event_type=EventType.GOAL,
                timestamp=position.timestamp,
                team=Team.LEFT,
                score_left=self._score_left,
                score_right=self._score_right,
                description=f"Tor Links · {self._score_left}:{self._score_right}",
            )
            logger.info("Goal LEFT! Score %d:%d", self._score_left, self._score_right)

        self._prev_in_left = in_left
        self._prev_in_right = in_right
        return event

    def reset(self) -> None:
        self._score_left = 0
        self._score_right = 0
        self._frames_since_goal = self._cooldown_frames
        self._prev_in_left = False
        self._prev_in_right = False

    def draw(self, frame) -> None:
        """Draw goal zone outlines onto *frame* for debugging / HUD."""
        import cv2
        for zone in (self._zone_left, self._zone_right):
            if zone is None:
                continue
            cv2.rectangle(
                frame,
                (zone.x, zone.y),
                (zone.x + zone.w, zone.y + zone.h),
                (0, 60, 220),
                2,
            )
            cv2.putText(
                frame,
                f"Tor {zone.name}",
                (zone.x, zone.y - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 60, 220),
                1,
            )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _rebuild_default_zones(
        self,
        field_y1: int = 0,
        field_y2: int = 480,
    ) -> None:
        """Build 1-D-style goal zones (y bounds ignored) from x bounds."""
        self._zone_left = _GoalZone(
            name="Left",
            x=self._field_x1,
            y=field_y1,
            w=self._goal_zone_width,
            h=field_y2 - field_y1,
            scoring_team=Team.RIGHT,
            use_y_bounds=False,  # backward compat: any y counts
        )
        self._zone_right = _GoalZone(
            name="Right",
            x=self._field_x2 - self._goal_zone_width,
            y=field_y1,
            w=self._goal_zone_width,
            h=field_y2 - field_y1,
            scoring_team=Team.LEFT,
            use_y_bounds=False,
        )
