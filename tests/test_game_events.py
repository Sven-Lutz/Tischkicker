"""Tests for game_events.py dataclasses and enums."""

import time
import pytest
from src.game_events import (
    Team, EventType, Zone, Rod, GameMode,
    BallPosition, GameEvent, GameConfig,
)


def test_ball_position_as_tuple():
    pos = BallPosition(x=100.0, y=200.0)
    assert pos.as_tuple() == (100.0, 200.0)


def test_ball_position_detected_default():
    pos = BallPosition(x=0.0, y=0.0)
    assert pos.detected is True


def test_ball_position_timestamp_auto():
    before = time.time()
    pos = BallPosition(x=10.0, y=20.0)
    after = time.time()
    assert before <= pos.timestamp <= after


def test_game_event_defaults():
    ts = time.time()
    event = GameEvent(event_type=EventType.GOAL, timestamp=ts)
    assert event.team is None
    assert event.value is None
    assert event.score_left == 0
    assert event.score_right == 0
    assert event.description == ""


def test_game_config_defaults():
    config = GameConfig()
    assert config.mode == GameMode.ONE_VS_ONE
    assert config.team_left_names == ["Links"]
    assert config.team_right_names == ["Rechts"]
    assert config.pixels_per_meter == 500.0
    assert config.fps == 30.0
    assert config.camera_index == 1


def test_team_enum_values():
    assert Team.LEFT.value == "Links"
    assert Team.RIGHT.value == "Rechts"


def test_rod_enum_has_five_rods():
    assert len(Rod) == 5


def test_zone_enum_has_three_zones():
    assert len(Zone) == 3
