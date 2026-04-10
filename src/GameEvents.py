"""Shared data classes and enums for the Kicker GT3 system."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)

class Team(Enum):
    """The two competing teams based on figure colors."""
    RED = "Team Red"
    BLACK = "Team Black"


class EventType(Enum):
    """All significant occurrences during a match."""
    GOAL = "Goal"
    MAX_SPEED = "MaxSpeed"
    REBOUND = "Rebound"
    GAME_START = "GameStart"
    GAME_END = "GameEnd"
    SHOT = "Shot"


class Rod(Enum):
    """
    The 8 physical rods on a standard foosball table.
    Ordered from left to right from the camera's perspective.
    The number in brackets denotes the standard player count (1-2-3-5-5-3-2-1).
    """
    LEFT_GOALIE = "Left Goalie (1)"
    LEFT_DEFENSE = "Left Defense (2)"
    RIGHT_ATTACK = "Right Attack (3)"
    LEFT_MIDFIELD = "Left Midfield (5)"
    RIGHT_MIDFIELD = "Right Midfield (5)"
    LEFT_ATTACK = "Left Attack (3)"
    RIGHT_DEFENSE = "Right Defense (2)"
    RIGHT_GOALIE = "Right Goalie (1)"


class Zone(Enum):
    """
    The 8 corresponding field zones directly beneath the rods.
    Crucial for heatmaps and precise possession tracking.
    """
    LEFT_GOAL_AREA = "Left Goal Area"
    LEFT_DEFENSE_AREA = "Left Defense Area"
    RIGHT_ATTACK_AREA = "Right Attack Area"
    LEFT_MIDFIELD_AREA = "Left Midfield Area"
    RIGHT_MIDFIELD_AREA = "Right Midfield Area"
    LEFT_ATTACK_AREA = "Left Attack Area"
    RIGHT_DEFENSE_AREA = "Right Defense Area"
    RIGHT_GOAL_AREA = "Right Goal Area"


@dataclass
class BallPosition:
    """Stores the exact position of the ball at a specific timestamp."""
    x: float
    y: float
    timestamp: float = field(default_factory=time.time)
    # Direction vector (dx, dy) to track where the ball is heading
    direction: Optional[tuple[float, float]] = None
    detected: bool = True

    def as_tuple(self) -> tuple[float, float]:
        """Returns coordinates as a simple tuple for geometric calculations."""
        return self.x, self.y


@dataclass
class GameEvent:
    """
    A unified data structure for any significant event.
    """
    event_type: EventType
    timestamp: float
    team: Optional[Team] = None
    value: Optional[float] = None

    # Keeping score references based on table sides for the UI
    score_left: int = 0
    score_right: int = 0
    description: str = ""


class GameMode(Enum):
    """Supported match configurations."""
    ONE_VS_ONE = "1v1"
    ONE_VS_TWO = "1v2"
    TWO_VS_ONE = "2v1"
    TWO_VS_TWO = "2v2"


@dataclass
class GameConfig:
    """Central configuration parameters for the system's game."""
    mode: GameMode = GameMode.ONE_VS_ONE

    team_left_names: list[str] = field(default_factory=lambda: ["Left"])
    team_right_names: list[str] = field(default_factory=lambda: ["Right"])

    pixels_per_meter: float = 500.0
    fps: float = 30.0
    camera_index: int = 0

    field_x1: int = 0
    field_y1: int = 0
    field_x2: int = 640
    field_y2: int = 480

    def __post_init__(self) -> None:
        """Automatically called after the configuration is initialized."""
        left_players = ", ".join(self.team_left_names)
        right_players = ", ".join(self.team_right_names)

        logger.info(
            f"[GameConfig] System initialized successfully. "
            f"Mode: {self.mode.value} | "
            f"Matchup: [{left_players}] vs [{right_players}]"
        )