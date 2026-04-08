"""Goal zone detection with cooldown to prevent double-counting."""

from __future__ import annotations

import time
from typing import Optional

from src.game_events import BallPosition, EventType, GameEvent, Team


class GoalDetector:
    """Detects when the ball enters the left or right goal zone.

    Goal zone left:  x in [field_x1, field_x1 + goal_zone_width]
    Goal zone right: x in [field_x2 - goal_zone_width, field_x2]

    cooldown_frames: minimum frames between two goal detections (prevents double-count).
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

    @property
    def score_left(self) -> int:
        return self._score_left

    @property
    def score_right(self) -> int:
        return self._score_right

    def update(self, position: Optional[BallPosition]) -> Optional[GameEvent]:
        """Call once per frame. Returns a GameEvent if a goal was detected, else None."""
        self._frames_since_goal += 1

        if position is None or not position.detected:
            return None

        if self._frames_since_goal < self._cooldown_frames:
            return None

        x = position.x
        left_zone_end = self._field_x1 + self._goal_zone_width
        right_zone_start = self._field_x2 - self._goal_zone_width

        if x <= left_zone_end:
            # Ball in left goal zone → RIGHT team scores
            self._score_right += 1
            self._frames_since_goal = 0
            return GameEvent(
                event_type=EventType.GOAL,
                timestamp=position.timestamp,
                team=Team.RIGHT,
                score_left=self._score_left,
                score_right=self._score_right,
                description=f"Tor Rechts · {self._score_left}:{self._score_right}",
            )

        if x >= right_zone_start:
            # Ball in right goal zone → LEFT team scores
            self._score_left += 1
            self._frames_since_goal = 0
            return GameEvent(
                event_type=EventType.GOAL,
                timestamp=position.timestamp,
                team=Team.LEFT,
                score_left=self._score_left,
                score_right=self._score_right,
                description=f"Tor Links · {self._score_left}:{self._score_right}",
            )

        return None

    def reset(self) -> None:
        self._score_left = 0
        self._score_right = 0
        self._frames_since_goal = self._cooldown_frames
