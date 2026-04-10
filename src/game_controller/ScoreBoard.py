from __future__ import annotations

import logging
import threading
import time
from typing import Optional

from src.GameEvents import GameEvent, EventType, Team

logger = logging.getLogger(__name__)

class ScoreBoard:
    """Manages the score and the goal history."""

    def __init__(
            self,
            left_team: Team = Team.RED,
            right_team: Team = Team.BLACK
    )->None:
        """Initializes the ScoreBoard."""
        self.left_team = left_team
        self.right_team = right_team

        self._scores: dict[Team, int] = {
            team: 0 for team in Team
        }
        self._goal_events: list[GameEvent] = []
        self._lock = threading.Lock()

        logger.info(
            f"[ScoreBoard] Initialized. "
            f"Left: {self.left_team.value} | Right: {self.right_team.value}"
        )

    def register_goal(
            self,
            team: Team,
            ball_speed_ms: float = 0.0
    )-> Optional[GameEvent]:
        """Registers a goal."""
        with self._lock:
            if team not in self._scores:
                logger.error(f"[ScoreBoard] Unknown team: {team}")
                return None

            self._scores[team] += 1

            score_l = self._scores[self.left_team]
            score_r = self._scores[self.right_team]

            event = GameEvent(
                event_type=EventType.GOAL,
                timestamp=time.time(),
                team=team,
                value=ball_speed_ms,
                score_left=score_l,
                score_right=score_r,
            )
            self._goal_events.append(event)

            logger.info(
                f"[ScoreBoard] GOAL for '{team.value}'! "
                f"Score: {score_l}:{score_r} | {ball_speed_ms:.2f} m/s"
            )
            return event

    def get_score_string(self) -> str:
        """Returns the score as a readable string formatted 'Left : Right'."""
        with self._lock:
            return (
                f"{self._scores[self.left_team]} : "
                f"{self._scores[self.right_team]}"
            )

    @property
    def goal_events(self) -> list[GameEvent]:
        """Returns a list of all recorded goals."""
        with self._lock:
            return list(self._goal_events)

    def reset(self) -> None:
        """Resets the score and clears history."""
        with self._lock:
            for team in self._scores:
                self._scores[team] = 0
            self._goal_events.clear()
            logger.info("[ScoreBoard] Score reset.")