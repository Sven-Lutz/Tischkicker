"""Extended tests for the improved GoalDetector (edge detection + 2-D zones)."""

import pytest
from src.game_events import BallPosition, EventType, Team
from src.goal_detector import GoalDetector


# ── helpers ───────────────────────────────────────────────────────────────────

def pos(x: float, y: float = 240.0, detected: bool = True) -> BallPosition:
    return BallPosition(x=x, y=y, timestamp=0.0, detected=detected)


def ready(field_x1=0, field_x2=640, goal_zone_width=40, cooldown=45) -> GoalDetector:
    """GoalDetector with cooldown pre-expired so detection fires immediately."""
    return GoalDetector(
        field_x1=field_x1,
        field_x2=field_x2,
        goal_zone_width=goal_zone_width,
        cooldown_frames=cooldown,
    )


# ── edge-detection (ball must ENTER zone, not just be inside it) ──────────────

class TestEdgeDetection:
    def test_first_entry_scores(self):
        gd = ready()
        evt = gd.update(pos(10.0))  # enters left zone from outside
        assert evt is not None
        assert evt.team == Team.RIGHT

    def test_staying_in_zone_does_not_double_count(self):
        gd = ready()
        gd.update(pos(10.0))          # enter → goal
        evt2 = gd.update(pos(15.0))   # still inside → no second goal
        evt3 = gd.update(pos(5.0))    # deeper inside → no goal
        assert evt2 is None
        assert evt3 is None
        assert gd.score_right == 1

    def test_leave_and_re_enter_after_cooldown_scores_again(self):
        gd = ready(cooldown=3)
        gd.update(pos(10.0))          # enter → goal, frames_since=0
        gd.update(pos(200.0))         # leave zone, frames_since=1, prev_in_left=False
        gd.update(pos(200.0))         # still outside, frames_since=2
        gd.update(pos(200.0))         # frames_since=3 >= cooldown
        evt = gd.update(pos(10.0))    # re-enter after cooldown → goal
        assert evt is not None
        assert gd.score_right == 2

    def test_leave_and_re_enter_within_cooldown_does_not_score(self):
        gd = ready(cooldown=10)
        gd.update(pos(10.0))          # enter → goal, frames_since=0
        gd.update(pos(200.0))         # leave, frames_since=1, prev_in_left=False
        evt = gd.update(pos(10.0))    # re-enter at frame 2, cooldown=10 → blocked
        assert evt is None
        assert gd.score_right == 1

    def test_ball_lost_resets_edge_state(self):
        """When ball disappears, entering zone afterwards must trigger a goal."""
        gd = ready(cooldown=3)
        gd.update(pos(10.0))          # enter → goal
        # Ball disappears for cooldown duration
        for _ in range(4):
            gd.update(None)           # resets prev_in_left to False
        evt = gd.update(pos(10.0))    # re-enter — prev_in_left=False → should score
        assert evt is not None
        assert gd.score_right == 2

    def test_right_zone_edge_detection(self):
        gd = ready(field_x1=0, field_x2=640, goal_zone_width=40)
        gd.update(pos(300.0))         # center — no goal, prev_in_right=False
        evt = gd.update(pos(610.0))   # enter right zone
        assert evt is not None
        assert evt.team == Team.LEFT
        evt2 = gd.update(pos(620.0))  # still inside
        assert evt2 is None
        assert gd.score_left == 1


# ── score increments correctly ────────────────────────────────────────────────

class TestScoring:
    def test_multiple_goals_tracked(self):
        gd = ready(cooldown=3)
        gd.update(pos(10.0))       # goal 1
        for _ in range(4):
            gd.update(pos(300.0))  # leave + wait
        gd.update(pos(10.0))       # goal 2
        assert gd.score_right == 2
        assert gd.score_left == 0

    def test_both_sides_score(self):
        gd = ready(cooldown=3)
        gd.update(pos(10.0))        # right scores on left goal
        for _ in range(4):
            gd.update(pos(300.0))
        gd.update(pos(630.0))       # left scores on right goal
        assert gd.score_right == 1
        assert gd.score_left == 1

    def test_event_carries_correct_score(self):
        gd = ready(cooldown=3)
        gd.update(pos(10.0))       # right scores: 0:1
        for _ in range(4):
            gd.update(pos(300.0))
        evt = gd.update(pos(10.0))  # right scores again: 0:2
        assert evt is not None
        assert evt.score_right == 2
        assert evt.score_left == 0

    def test_event_type_is_goal(self):
        gd = ready()
        evt = gd.update(pos(10.0))
        assert evt is not None
        assert evt.event_type == EventType.GOAL

    def test_reset_clears_edge_state_and_score(self):
        gd = ready()
        gd.update(pos(10.0))
        gd.reset()
        assert gd.score_right == 0
        assert gd.score_left == 0
        # After reset, entering zone should score again
        evt = gd.update(pos(10.0))
        assert evt is not None
        assert gd.score_right == 1


# ── 2-D goal zones from configure_from_corners ───────────────────────────────

class TestConfigureFromCorners:
    """Test that configure_from_corners enables y-bounded 2-D goal zones."""

    # Field: 100 px left to 500 px right, 50 px top to 350 px bottom.
    # Centre y = 200. GOAL_HEIGHT_RATIO=0.28 → height ≈ 0.28*300 ≈ 84 px
    # → goal y range: ~158 to ~242

    CORNERS = [(100, 50), (500, 50), (500, 350), (100, 350)]  # TL TR BR BL

    def _detector(self) -> GoalDetector:
        gd = GoalDetector(cooldown_frames=3)
        gd.configure_from_corners(self.CORNERS)
        gd._frames_since_goal = 100  # pre-expire cooldown
        return gd

    def test_ball_in_goal_y_range_scores(self):
        gd = self._detector()
        # x=100 (left boundary), y=200 (vertically centred → inside goal height)
        evt = gd.update(pos(x=100.0, y=200.0))
        assert evt is not None
        assert evt.team == Team.RIGHT

    def test_ball_outside_y_range_does_not_score(self):
        gd = self._detector()
        # x=100 but y=10 — above field top, way outside goal opening
        evt = gd.update(pos(x=100.0, y=10.0))
        assert evt is None

    def test_right_goal_with_y_range(self):
        gd = self._detector()
        # x=500 (right boundary), y=200 (inside goal height)
        evt = gd.update(pos(x=500.0, y=200.0))
        assert evt is not None
        assert evt.team == Team.LEFT

    def test_four_corners_required(self):
        """Fewer than 4 corners should log a warning and not crash."""
        gd = GoalDetector()
        gd.configure_from_corners([(0, 0), (640, 0)])  # only 2 points
        # Should still work with default zones
        evt = gd.update(pos(10.0))
        assert evt is not None  # default 1-D zone still fires


# ── detected=False positions ──────────────────────────────────────────────────

def test_undetected_position_never_scores():
    gd = ready()
    evt = gd.update(pos(10.0, detected=False))
    assert evt is None
    assert gd.score_right == 0
