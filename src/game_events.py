"""Shared data classes and enums for the Kicker GT3 system."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Team(Enum):
    LEFT = "Links"
    RIGHT = "Rechts"


class EventType(Enum):
    GOAL = "Tor"
    MAX_SPEED = "Hoechstgeschwindigkeit"
    REBOUND = "Abprall"
    GAME_START = "Spielstart"
    GAME_END = "Spielende"


class Zone(Enum):
    ATTACK_LEFT = "Angriff Links"
    MIDDLE = "Mitte"
    ATTACK_RIGHT = "Angriff Rechts"


class Rod(Enum):
    KEEPER_LEFT = "Tor Links"
    DEFENSE_LEFT = "Abwehr Links"
    MIDFIELD = "Mittelfeld"
    DEFENSE_RIGHT = "Abwehr Rechts"
    KEEPER_RIGHT = "Tor Rechts"


@dataclass
class BallPosition:
    x: float
    y: float
    timestamp: float = field(default_factory=time.time)
    detected: bool = True

    def as_tuple(self) -> tuple[float, float]:
        return (self.x, self.y)


@dataclass
class GameEvent:
    event_type: EventType
    timestamp: float
    team: Optional[Team] = None
    value: Optional[float] = None
    score_left: int = 0
    score_right: int = 0
    description: str = ""


class GameMode(Enum):
    ONE_VS_ONE = "1v1"
    TWO_VS_TWO = "2v2"


@dataclass
class GameConfig:
    mode: GameMode = GameMode.ONE_VS_ONE
    team_left_names: list[str] = field(default_factory=lambda: ["Links"])
    team_right_names: list[str] = field(default_factory=lambda: ["Rechts"])
    pixels_per_meter: float = 500.0
    fps: float = 30.0
    camera_index: int = 1
    field_x1: int = 0
    field_y1: int = 0
    field_x2: int = 640
    field_y2: int = 480
