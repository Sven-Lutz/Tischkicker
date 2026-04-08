"""Tests for Statistics."""

import time
import pytest
import numpy as np

from src.game_events import BallPosition, GameConfig, GameMode, Team, Zone
from src.statistics import Statistics


def make_config(fps: float = 30.0, ppm: float = 500.0,
                field_x1: int = 0, field_y1: int = 0,
                field_x2: int = 640, field_y2: int = 480) -> GameConfig:
    return GameConfig(
        mode=GameMode.ONE_VS_ONE,
        fps=fps,
        pixels_per_meter=ppm,
        field_x1=field_x1, field_y1=field_y1,
        field_x2=field_x2, field_y2=field_y2,
    )


def make_pos(x: float, y: float = 240.0, ts: float = 0.0) -> BallPosition:
    return BallPosition(x=x, y=y, timestamp=ts)


# ── Speed calculations ─────────────────────────────────────────────────────

def test_speed_known_distance():
    """50 px at fps=30, ppm=500 → 50/500*30 = 3.0 m/s."""
    cfg = make_config(fps=30.0, ppm=500.0)
    stats = Statistics(cfg)
    stats.update(make_pos(0.0), None)
    stats.update(make_pos(50.0), None)
    assert pytest.approx(stats.current_speed_ms, rel=1e-6) == 3.0


def test_max_speed_tracked():
    cfg = make_config(fps=30.0, ppm=500.0)
    stats = Statistics(cfg)
    stats.update(make_pos(0.0), None)
    stats.update(make_pos(50.0), None)   # 3.0 m/s
    stats.update(make_pos(51.0), None)   # 0.06 m/s
    assert pytest.approx(stats.max_speed_ms, rel=1e-6) == 3.0


def test_max_speed_timestamp_recorded():
    cfg = make_config(fps=30.0, ppm=500.0)
    stats = Statistics(cfg)
    stats.start_timer()
    stats.update(make_pos(0.0), None)
    stats.update(make_pos(50.0), None)  # max speed here
    ts = stats.max_speed_timestamp
    assert ts >= 0.0


def test_average_speed():
    cfg = make_config(fps=30.0, ppm=500.0)
    stats = Statistics(cfg)
    # frame 0→1: 30 px → 30/500*30 = 1.8 m/s
    # frame 1→2: 70 px → 70/500*30 = 4.2 m/s
    stats.update(make_pos(0.0), None)
    stats.update(make_pos(30.0), None)
    stats.update(make_pos(100.0), None)
    assert pytest.approx(stats.average_speed_ms, rel=1e-4) == (1.8 + 4.2) / 2


# ── Zone distribution ──────────────────────────────────────────────────────

def test_zone_percentages_sum_to_100():
    cfg = make_config(field_x1=0, field_x2=600)
    stats = Statistics(cfg)
    for x in [100, 300, 500]:
        stats.update(make_pos(float(x)), None)
    pcts = stats.zone_percentages
    total = sum(pcts.values())
    assert pytest.approx(total, abs=0.01) == 100.0


def test_zone_percentages_correct_distribution():
    cfg = make_config(field_x1=0, field_x2=600)
    stats = Statistics(cfg)
    # Two positions in left third (0–200), one in right third (400–600)
    for x in [50.0, 100.0, 500.0]:
        stats.update(make_pos(x), None)
    pcts = stats.zone_percentages
    assert pytest.approx(pcts[Zone.ATTACK_LEFT], rel=0.01) == 200 / 3
    assert pytest.approx(pcts[Zone.ATTACK_RIGHT], rel=0.01) == 100 / 3


# ── Rebound detection ──────────────────────────────────────────────────────

def test_rebound_counted_on_sign_flip():
    cfg = make_config()
    stats = Statistics(cfg)
    # Move right (large dx > threshold), then move left (sign flip)
    stats.update(make_pos(100.0), None)
    stats.update(make_pos(110.0), None)  # dx=+10
    stats.update(make_pos(100.0), None)  # dx=-10 → rebound
    assert stats.rebound_count == 1


# ── Heatmap ────────────────────────────────────────────────────────────────

def test_heatmap_shape_and_nonzero():
    cfg = make_config(field_x1=0, field_y1=0, field_x2=640, field_y2=480)
    stats = Statistics(cfg)
    stats.update(make_pos(320.0, 240.0), None)
    hm = stats.heatmap
    assert hm.shape == (480, 640)
    assert hm.max() > 0.0


def test_team_heatmaps_shape():
    cfg = make_config(field_x1=0, field_y1=0, field_x2=640, field_y2=480)
    stats = Statistics(cfg)
    stats.update(make_pos(100.0, 240.0), None)  # left half
    stats.update(make_pos(500.0, 240.0), None)  # right half
    hms = stats.team_heatmaps
    assert hms[Team.LEFT].shape == (480, 640)
    assert hms[Team.RIGHT].shape == (480, 640)
    assert hms[Team.LEFT].max() > 0.0
    assert hms[Team.RIGHT].max() > 0.0


# ── Trajectory ─────────────────────────────────────────────────────────────

def test_trajectory_capped_at_trajectory_length():
    cfg = make_config()
    stats = Statistics(cfg)
    for i in range(Statistics.TRAJECTORY_LENGTH + 10):
        stats.update(make_pos(float(i)), None)
    assert len(stats.trajectory) == Statistics.TRAJECTORY_LENGTH


# ── Per-team stats ─────────────────────────────────────────────────────────

def test_team_possession_pct_sums_to_100():
    cfg = make_config(field_x1=0, field_x2=640)
    stats = Statistics(cfg)
    for x in [100.0, 200.0, 400.0, 500.0]:
        stats.update(make_pos(x), None)
    pcts = stats.team_possession_pct
    assert pytest.approx(pcts[Team.LEFT] + pcts[Team.RIGHT], abs=0.01) == 100.0


def test_team_max_speed_independent():
    cfg = make_config(fps=30.0, ppm=500.0, field_x1=0, field_x2=640)
    stats = Statistics(cfg)
    # Fast move on left side
    stats.update(make_pos(100.0), None)
    stats.update(make_pos(150.0), None)  # 3 m/s on left
    # Slower move on right side
    stats.update(make_pos(400.0), None)
    stats.update(make_pos(410.0), None)  # 0.6 m/s on right
    tm = stats.team_max_speed
    assert tm[Team.LEFT] > tm[Team.RIGHT]
