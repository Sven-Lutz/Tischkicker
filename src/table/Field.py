"""Goal zone detection merging 2D spatial collision with event-driven state tracking."""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

from src.game_events import BallPosition, EventType, GameEvent, Team

logger = logging.getLogger(__name__)


class GoalZone:
    """Represents a 2D goal opening on the table."""

    DEFAULT_COOLDOWN_FRAMES = 30

    def __init__(
            self,
            name: str,
            x: int,
            y: int,
            w: int,
            h: int,
            scoring_team: Team):
        """
        :param name: Goal name (e.g., "Left" or "Right")
        :param x, y: Top-left corner of the zone
        :param w, h: Width and height of the zone
        :param scoring_team:
        The team that gets a point when the ball enters this zone
        """
        self.name = name
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.scoring_team = scoring_team

        self._cooldown_counter: int = 0
        self._ball_was_inside: bool = False

    def contains_point(self, px: float, py: float) -> bool:
        """Checks if the given point lies within the bounding box of the goal zone."""
        return (self.x <= px <= self.x + self.w and
                self.y <= py <= self.y + self.h)

    def check_goal(self, ball_x: Optional[float],
                   ball_y: Optional[float]) -> bool:
        """
        Evaluates if a new goal was scored in this frame.
        Uses edge detection (False -> True) and a cooldown to avoid double counting.
        """
        if self._cooldown_counter > 0:
            self._cooldown_counter -= 1

        if ball_x is None or ball_y is None:
            self._ball_was_inside = False
            return False

        inside = self.contains_point(ball_x, ball_y)

        # A goal is scored ONLY if the ball just entered the zone AND cooldown is zero
        is_new_goal = (inside and not self._ball_was_inside and
                       self._cooldown_counter == 0)

        if is_new_goal:
            self._cooldown_counter = self.DEFAULT_COOLDOWN_FRAMES

        self._ball_was_inside = inside
        return is_new_goal

    def draw(self, frame: np.ndarray,
             color: tuple[int, int, int] = (0, 0, 255)) -> None:
        """Draws the goal opening as a centered line segment on the field border."""
        # Draw on the outer edge of the goal zone
        border_x = self.x + self.w if self.name == "Left" else self.x
        cv2.line(frame, (border_x, self.y),
                 (border_x, self.y + self.h), color, 4)

        label_x = self.x if self.name == "Left" else max(0, self.x - 60)
        label_y = max(0, self.y - 6)
        cv2.putText(
            frame,
            f"Goal {self.name}",
            (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1
        )


class GoalDetector:
    """
    Manages the playing field calibration and detects goals based on 2D zones.
    Replaces the old 1D checking with dynamic, visually calibrated bounds.
    """

    GOAL_DEPTH_RATIO = 0.04
    GOAL_HEIGHT_RATIO = 0.27

    def __init__(self) -> None:
        self.goal_zones: list[GoalZone] = []
        self.corners: list[tuple[int, int]] = []
        self._click_buffer: list[tuple[int, int]] = []

        self._score_left = 0
        self._score_right = 0

    @property
    def score_left(self) -> int:
        return self._score_left

    @property
    def score_right(self) -> int:
        return self._score_right

    # Calibration--------------------------------------------------------

    def calibrate_interactive(self,
                              frame: np.ndarray,
                              window_name: str = "Calibration") -> None:
        """
        Interactive 4-point calibration for the playing field.
        Expects clicks in clockwise order: Top-Left, Top-Right, Bottom-Right, Bottom-Left.
        """
        self._click_buffer.clear()
        logger.info("Starting interactive field calibration. "
                    "Please click the 4 corners.")

        def on_mouse(event: int, x: int, y: int, flags: int, param: any) -> None:
            if event == cv2.EVENT_LBUTTONDOWN and len(self._click_buffer) < 4:
                self._click_buffer.append((x, y))
                logger.debug("Corner %d/4 set at (%d, %d)", len(self._click_buffer), x, y)

        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, on_mouse)

        while len(self._click_buffer) < 4:
            display = frame.copy()

            # Draw clicked points
            for i, pt in enumerate(self._click_buffer):
                cv2.circle(display, pt, 6, (0, 255, 255), -1)
                cv2.putText(
                    display,
                    str(i + 1),
                    (pt[0] + 8, pt[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2
                )

            # Draw lines between clicked points
            if len(self._click_buffer) >= 2:
                for i in range(len(self._click_buffer) - 1):
                    cv2.line(display, self._click_buffer[i], self._click_buffer[i + 1], (0, 255, 255), 3)

            cv2.imshow(window_name, display)
            cv2.waitKey(30)

        self.corners = list(self._click_buffer)
        self._compute_goal_zones()

        # Show the calculated zones briefly
        self._show_calibration_result(frame.copy(), window_name)
        logger.info("Field calibration completed successfully.")

    def _compute_goal_zones(self) -> None:
        """Calculates the two goal zones dynamically based on the 4 calibrated corners."""
        if len(self.corners) < 4:
            return

        tl, tr, br, bl = self.corners

        # Estimate field dimensions in pixels
        field_width = int(((tr[0] - tl[0]) + (br[0] - bl[0])) / 2)
        field_height = int(((bl[1] - tl[1]) + (br[1] - tr[1])) / 2)

        goal_depth = max(8, int(field_width * self.GOAL_DEPTH_RATIO))
        goal_height = max(20, int(field_height * self.GOAL_HEIGHT_RATIO))

        left_mid_x = int((tl[0] + bl[0]) / 2)
        left_mid_y = int((tl[1] + bl[1]) / 2)

        right_mid_x = int((tr[0] + br[0]) / 2)
        right_mid_y = int((tr[1] + br[1]) / 2)

        # Left goal zone -> Right team scores if the ball enters here
        gz_left = GoalZone(
            name="Left",
            x=left_mid_x - goal_depth,
            y=left_mid_y - goal_height // 2,
            w=goal_depth,
            h=goal_height,
            scoring_team=Team.RIGHT,
        )

        # Right goal zone -> Left team scores if the ball enters here
        gz_right = GoalZone(
            name="Right",
            x=right_mid_x,
            y=right_mid_y - goal_height // 2,
            w=goal_depth,
            h=goal_height,
            scoring_team=Team.LEFT,
        )

        self.goal_zones = [gz_left, gz_right]
        logger.debug("Goal Left computed: x=%d, y=%d, w=%d, h=%d", gz_left.x, gz_left.y, gz_left.w, gz_left.h)
        logger.debug("Goal Right computed: x=%d, y=%d, w=%d, h=%d", gz_right.x, gz_right.y, gz_right.w, gz_right.h)

    def _show_calibration_result(self, frame: np.ndarray, window_name: str) -> None:
        """Displays the calibration result briefly."""
        pts = np.array(self.corners, dtype=np.int32)
        cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 255), thickness=4)

        for gz in self.goal_zones:
            gz.draw(frame)

        cv2.imshow(window_name, frame)
        cv2.waitKey(1000)  # Show for 1 second

    #--Loop Logic & Visualization--------------------------------------------

    def update(self, position: Optional[BallPosition]) -> Optional[GameEvent]:
        """
        Called once per frame by the Controller.

        :return: A GameEvent if a goal was scored, else None.
        """
        if position is None or not position.detected:
            # Reset states if the ball is lost, but cooldowns tick down internally
            for gz in self.goal_zones:
                gz.check_goal(None, None)
            return None

        for gz in self.goal_zones:
            if gz.check_goal(position.x, position.y):

                # Increment the correct score
                if gz.scoring_team == Team.LEFT:
                    self._score_left += 1
                else:
                    self._score_right += 1

                score_str = f"{self._score_left}:{self._score_right}"
                logger.info("Goal scored by %s! Current Score: %s", gz.scoring_team.value, score_str)

                return GameEvent(
                    event_type=EventType.GOAL,
                    timestamp=position.timestamp,
                    team=gz.scoring_team,
                    score_left=self._score_left,
                    score_right=self._score_right,
                    description=f"Goal {gz.scoring_team.value} | {score_str}",
                )

        return None

    def draw(self, frame: np.ndarray) -> None:
        """Renders the field boundaries and goal zones onto the frame."""
        if self.corners:
            pts = np.array(self.corners, dtype=np.int32)
            cv2.polylines(frame, [pts], isClosed=True, color=(0, 200, 200), thickness=3)

        for gz in self.goal_zones:
            gz.draw(frame)

    def reset(self) -> None:
        """Resets the internal scores and goal cooldown states."""
        self._score_left = 0
        self._score_right = 0
        for gz in self.goal_zones:
            gz._cooldown_counter = gz.DEFAULT_COOLDOWN_FRAMES
            gz._ball_was_inside = False
        logger.info("GoalDetector scores and internal states have been reset.")