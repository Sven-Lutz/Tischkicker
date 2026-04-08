"""Tests for GoalDetector."""

import pytest
from src.game_events import BallPosition, EventType, Team
from src.goal_detector import GoalDetector


def make_pos(x: float, y: float = 240.0) -> BallPosition:
    return BallPosition(x=x, y=y, timestamp=0.0)


def _ready_detector(**kwargs) -> GoalDetector:
    """GoalDetector with cooldown pre-expired so first detection fires immediately."""
    gd = GoalDetector(**kwargs)
    return gd  # frames_since_goal starts at cooldown_frames


def test_goal_left_zone_scores_right_team():
    gd = _ready_detector(field_x1=0, field_x2=640, goal_zone_width=40)
    event = gd.update(make_pos(x=20.0))
    assert event is not None
    assert event.team == Team.RIGHT
    assert event.score_right == 1
    assert event.score_left == 0
    assert event.event_type == EventType.GOAL


def test_goal_right_zone_scores_left_team():
    gd = _ready_detector(field_x1=0, field_x2=640, goal_zone_width=40)
    event = gd.update(make_pos(x=620.0))
    assert event is not None
    assert event.team == Team.LEFT
    assert event.score_left == 1
    assert event.score_right == 0


def test_no_goal_in_center():
    gd = _ready_detector(field_x1=0, field_x2=640, goal_zone_width=40)
    event = gd.update(make_pos(x=320.0))
    assert event is None


def test_cooldown_suppresses_double_count():
    gd = GoalDetector(field_x1=0, field_x2=640, goal_zone_width=40, cooldown_frames=5)
    # Force cooldown to be ready
    gd._frames_since_goal = 5
    first = gd.update(make_pos(x=20.0))
    assert first is not None
    # Immediately again — still in cooldown
    second = gd.update(make_pos(x=20.0))
    assert second is None
    assert gd.score_right == 1  # not 2


def test_reset_clears_scores():
    gd = _ready_detector(field_x1=0, field_x2=640)
    gd.update(make_pos(x=20.0))
    assert gd.score_right == 1
    gd.reset()
    assert gd.score_left == 0
    assert gd.score_right == 0


def test_none_position_does_not_score():
    gd = _ready_detector()
    event = gd.update(None)
    assert event is None
    assert gd.score_left == 0
    assert gd.score_right == 0
