"""Extended unit tests for Statistics.

Covers scenarios not tested in test_statistics.py:
  * shot count detection
  * rod goal attribution
  * game timer behaviour (start / stop / elapsed)
  * reset() clears all state
  * gap in ball detection resets kinematics chain
  * thread safety (basic concurrent access smoke test)
  * events list accumulation
  * possession percentages when ball always on one side
"""

import threading
import time

import numpy as np
import pytest

from src.game_events import (
    BallPosition,
    EventType,
    GameConfig,
    GameEvent,
    GameMode,
    Rod,
    Team,
    Zone,
)
from src.statistics import Statistics


# ── fixtures / helpers ────────────────────────────────────────────────────────

def cfg(
    fps: float = 30.0,
    ppm: float = 500.0,
    x1: int = 0,
    y1: int = 0,
    x2: int = 640,
    y2: int = 480,
) -> GameConfig:
    return GameConfig(
        mode=GameMode.ONE_VS_ONE,
        fps=fps,
        pixels_per_meter=ppm,
        field_x1=x1,
        field_y1=y1,
        field_x2=x2,
        field_y2=y2,
    )


def p(x: float, y: float = 240.0, ts: float = 0.0) -> BallPosition:
    return BallPosition(x=x, y=y, timestamp=ts)


def goal_event(team: Team, sl: int, sr: int) -> GameEvent:
    return GameEvent(
        event_type=EventType.GOAL,
        timestamp=0.0,
        team=team,
        score_left=sl,
        score_right=sr,
    )


# ── shot count ────────────────────────────────────────────────────────────────

class TestShotCount:
    def test_shot_counted_when_speed_crosses_threshold(self):
        """A rapid acceleration crossing SHOT_SPEED_THRESHOLD_HIGH triggers a shot."""
        s = Statistics(cfg(fps=30.0, ppm=500.0))
        # Need speed ≥ 3.0 m/s: 50 px / 500 ppm * 30 fps = 3.0 m/s
        s.update(p(0.0), None)
        s.update(p(50.0), None)   # 3.0 m/s → crosses threshold
        assert s.shot_count == 1

    def test_shot_not_double_counted_while_fast(self):
        """Consecutive fast frames only count as ONE shot."""
        s = Statistics(cfg(fps=30.0, ppm=500.0))
        s.update(p(0.0), None)
        s.update(p(50.0), None)   # 3.0 m/s  → shot 1
        s.update(p(100.0), None)  # 3.0 m/s  → still above threshold, no new shot
        assert s.shot_count == 1

    def test_shot_counted_again_after_slowing_down(self):
        """Ball slows down then speeds up again → second shot."""
        s = Statistics(cfg(fps=30.0, ppm=500.0))
        s.update(p(0.0), None)
        s.update(p(50.0), None)    # fast (shot 1)
        s.update(p(50.5), None)    # slow (below threshold)
        s.update(p(101.0), None)   # fast again (shot 2)
        assert s.shot_count == 2

    def test_slow_movement_no_shot(self):
        s = Statistics(cfg(fps=30.0, ppm=500.0))
        s.update(p(0.0), None)
        s.update(p(1.0), None)    # 0.06 m/s — well below threshold
        assert s.shot_count == 0

    def test_team_shot_counts_attributed_to_correct_side(self):
        s = Statistics(cfg(fps=30.0, ppm=500.0, x1=0, x2=640))
        # Shot from left side (x < 320)
        s.update(p(100.0), None)
        s.update(p(150.0), None)  # 3.0 m/s on left → left team shot
        tc = s.team_shot_counts
        assert tc[Team.LEFT] == 1
        assert tc[Team.RIGHT] == 0


# ── rod goal attribution ──────────────────────────────────────────────────────

class TestRodGoalAttribution:
    def test_goal_attributed_to_rod_at_keeper_left(self):
        s = Statistics(cfg(x1=0, x2=640))
        # Ball near left keeper (x < 20% of 640 = 128)
        for _ in range(6):
            s.update(p(60.0), None)   # fill trajectory buffer
        s.update(None, goal_event(Team.RIGHT, 0, 1))
        rods = s.rod_goal_counts
        assert rods[Rod.KEEPER_LEFT] == 1

    def test_goal_attributed_to_midfield_rod(self):
        s = Statistics(cfg(x1=0, x2=640))
        for _ in range(6):
            s.update(p(320.0), None)  # midfield (40%–60% → rod MIDFIELD)
        s.update(None, goal_event(Team.LEFT, 1, 0))
        assert s.rod_goal_counts[Rod.MIDFIELD] == 1

    def test_goal_attributed_to_keeper_right(self):
        s = Statistics(cfg(x1=0, x2=640))
        for _ in range(6):
            s.update(p(620.0), None)  # > 80% → KEEPER_RIGHT
        s.update(None, goal_event(Team.LEFT, 1, 0))
        assert s.rod_goal_counts[Rod.KEEPER_RIGHT] == 1

    def test_no_attribution_when_trajectory_empty(self):
        """Goal before any ball position is tracked — no crash, no attribution."""
        s = Statistics(cfg())
        s.update(None, goal_event(Team.RIGHT, 0, 1))
        # All rod counts remain 0
        assert all(v == 0 for v in s.rod_goal_counts.values())


# ── timer ─────────────────────────────────────────────────────────────────────

class TestTimer:
    def test_game_time_zero_before_start(self):
        s = Statistics(cfg())
        assert s.game_time_seconds == 0.0

    def test_game_time_increases_after_start(self):
        s = Statistics(cfg())
        s.start_timer()
        time.sleep(0.05)
        assert s.game_time_seconds >= 0.04

    def test_stop_timer_freezes_clock(self):
        s = Statistics(cfg())
        s.start_timer()
        time.sleep(0.05)
        s.stop_timer()
        t1 = s.game_time_seconds
        time.sleep(0.05)
        t2 = s.game_time_seconds
        assert t1 == pytest.approx(t2, abs=0.001)

    def test_start_timer_twice_resets_clock(self):
        """Calling start_timer() again resets the start reference."""
        s = Statistics(cfg())
        s.start_timer()
        time.sleep(0.1)
        s.start_timer()  # restart
        # Should be near zero again
        assert s.game_time_seconds < 0.05


# ── reset ─────────────────────────────────────────────────────────────────────

class TestReset:
    def test_reset_clears_speed(self):
        s = Statistics(cfg(fps=30.0, ppm=500.0))
        s.update(p(0.0), None)
        s.update(p(50.0), None)
        assert s.max_speed_ms > 0
        s.reset()
        assert s.max_speed_ms == 0.0
        assert s.current_speed_ms == 0.0
        assert s.average_speed_ms == 0.0

    def test_reset_clears_shot_and_rebound_counts(self):
        s = Statistics(cfg(fps=30.0, ppm=500.0))
        s.update(p(0.0), None)
        s.update(p(50.0), None)
        s.update(p(40.0), None)   # rebound
        assert s.shot_count > 0 or s.rebound_count > 0
        s.reset()
        assert s.shot_count == 0
        assert s.rebound_count == 0

    def test_reset_clears_trajectory(self):
        s = Statistics(cfg())
        for i in range(10):
            s.update(p(float(i)), None)
        assert len(s.trajectory) > 0
        s.reset()
        assert len(s.trajectory) == 0

    def test_reset_clears_heatmap(self):
        s = Statistics(cfg())
        s.update(p(320.0, 240.0), None)
        assert s.heatmap.max() > 0.0
        s.reset()
        assert s.heatmap.max() == 0.0

    def test_reset_clears_zone_counts(self):
        s = Statistics(cfg(x1=0, x2=600))
        s.update(p(100.0), None)
        s.reset()
        for v in s.zone_percentages.values():
            assert v == 0.0

    def test_reset_clears_events(self):
        s = Statistics(cfg())
        s.update(None, goal_event(Team.LEFT, 1, 0))
        assert len(s.events) == 1
        s.reset()
        assert len(s.events) == 0

    def test_reset_clears_rod_goal_counts(self):
        s = Statistics(cfg())
        for _ in range(6):
            s.update(p(320.0), None)
        s.update(None, goal_event(Team.LEFT, 1, 0))
        s.reset()
        assert all(v == 0 for v in s.rod_goal_counts.values())


# ── gap in detection resets kinematics ───────────────────────────────────────

class TestDetectionGap:
    def test_speed_not_calculated_across_gap(self):
        """If ball disappears then reappears, no spurious high speed is computed."""
        s = Statistics(cfg(fps=30.0, ppm=500.0))
        s.update(p(0.0), None)
        s.update(None, None)        # ball lost — prev_position cleared
        s.update(p(600.0), None)    # large jump but no prev → speed = 0
        # current_speed must be 0 (no two consecutive positions)
        assert s.current_speed_ms == 0.0

    def test_max_speed_not_corrupted_by_gap(self):
        s = Statistics(cfg(fps=30.0, ppm=500.0))
        s.update(p(0.0), None)
        s.update(p(10.0), None)     # moderate speed 0.6 m/s
        s.update(None, None)        # gap
        s.update(p(640.0), None)    # huge jump — should NOT be measured
        assert s.max_speed_ms < 50.0  # no ridiculous speed


# ── events accumulation ───────────────────────────────────────────────────────

class TestEvents:
    def test_goal_event_recorded(self):
        s = Statistics(cfg())
        evt = goal_event(Team.RIGHT, 0, 1)
        s.update(None, evt)
        assert evt in s.events

    def test_multiple_events_in_order(self):
        s = Statistics(cfg())
        e1 = goal_event(Team.RIGHT, 0, 1)
        e2 = goal_event(Team.LEFT, 1, 1)
        s.update(None, e1)
        s.update(None, e2)
        events = s.events
        assert events[0] is e1
        assert events[1] is e2

    def test_none_event_not_recorded(self):
        s = Statistics(cfg())
        s.update(p(320.0), None)
        assert len(s.events) == 0


# ── possession ────────────────────────────────────────────────────────────────

class TestPossession:
    def test_all_left_gives_100_pct_left(self):
        s = Statistics(cfg(x1=0, x2=640))
        for _ in range(5):
            s.update(p(100.0), None)  # left half
        pcts = s.team_possession_pct
        assert pcts[Team.LEFT] == pytest.approx(100.0)
        assert pcts[Team.RIGHT] == pytest.approx(0.0)

    def test_even_split_gives_50_50(self):
        s = Statistics(cfg(x1=0, x2=640))
        for _ in range(4):
            s.update(p(100.0), None)  # left
        for _ in range(4):
            s.update(p(500.0), None)  # right
        pcts = s.team_possession_pct
        assert pcts[Team.LEFT] == pytest.approx(50.0)
        assert pcts[Team.RIGHT] == pytest.approx(50.0)

    def test_empty_gives_zero_pct(self):
        s = Statistics(cfg())
        pcts = s.team_possession_pct
        assert pcts[Team.LEFT] == 0.0
        assert pcts[Team.RIGHT] == 0.0


# ── thread safety smoke test ──────────────────────────────────────────────────

class TestThreadSafety:
    def test_concurrent_updates_do_not_crash(self):
        """Multiple threads calling update() concurrently must not raise."""
        s = Statistics(cfg(fps=30.0, ppm=500.0))
        errors: list[Exception] = []

        def worker(x_offset: int) -> None:
            try:
                for i in range(50):
                    s.update(p(float(i + x_offset)), None)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i * 50,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread errors: {errors}"

    def test_concurrent_reads_and_writes(self):
        """Reading properties while another thread writes must not crash."""
        s = Statistics(cfg())
        s.start_timer()
        stop = threading.Event()
        errors: list[Exception] = []

        def writer() -> None:
            for i in range(100):
                s.update(p(float(i % 640)), None)

        def reader() -> None:
            while not stop.is_set():
                try:
                    _ = s.max_speed_ms
                    _ = s.rebound_count
                    _ = s.trajectory
                except Exception as exc:
                    errors.append(exc)

        r = threading.Thread(target=reader)
        w = threading.Thread(target=writer)
        r.start()
        w.start()
        w.join()
        stop.set()
        r.join()

        assert errors == [], f"Reader errors: {errors}"
