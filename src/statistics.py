"""Real-time and post-game statistics for the Kicker GT3 system."""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Optional

import numpy as np

from src.game_events import BallPosition, EventType, GameConfig, GameEvent, Rod, Team, Zone


class Statistics:
    """Tracks all game statistics from ball positions and events.

    Thread-safe: update() and all property reads are protected by a Lock.
    """

    TRAJECTORY_LENGTH = 30
    HEATMAP_GAUSSIAN_RADIUS = 15
    SHOT_SPEED_THRESHOLD_LOW = 1.0   # m/s — below this, ball is not in motion
    SHOT_SPEED_THRESHOLD_HIGH = 3.0  # m/s — crossing this triggers a shot count
    REBOUND_VX_THRESHOLD = 2.0       # pixels — minimum |vx| to count as rebound

    def __init__(self, config: GameConfig) -> None:
        self._config = config
        self._lock = threading.Lock()
        self._reset_internal()

    # ── Public interface ──────────────────────────────────────────────────────

    def update(self, position: Optional[BallPosition], event: Optional[GameEvent]) -> None:
        with self._lock:
            if event is not None:
                self._events.append(event)
                if event.event_type == EventType.GOAL and event.team is not None:
                    self._record_rod_goal(event)

            if position is None or not position.detected:
                self._prev_position = None
                return

            self._trajectory.append(position)

            if self._prev_position is not None:
                self._update_kinematics(position)

            self._update_heatmap(position)
            self._update_zone(position)
            self._update_team_stats(position)
            self._prev_position = position

    def start_timer(self) -> None:
        with self._lock:
            self._game_start_time = time.time()
            self._game_end_time = None

    def stop_timer(self) -> None:
        with self._lock:
            self._game_end_time = time.time()

    def reset(self) -> None:
        with self._lock:
            self._reset_internal()

    # ── Core properties ───────────────────────────────────────────────────────

    @property
    def current_speed_ms(self) -> float:
        with self._lock:
            return self._current_speed_ms

    @property
    def max_speed_ms(self) -> float:
        with self._lock:
            return self._max_speed_ms

    @property
    def max_speed_timestamp(self) -> float:
        """Seconds since game start when max speed was recorded."""
        with self._lock:
            return self._max_speed_timestamp

    @property
    def average_speed_ms(self) -> float:
        with self._lock:
            if self._speed_frame_count == 0:
                return 0.0
            return self._total_speed_sum / self._speed_frame_count

    @property
    def rebound_count(self) -> int:
        with self._lock:
            return self._rebound_count

    @property
    def shot_count(self) -> int:
        with self._lock:
            return self._shot_count

    @property
    def zone_percentages(self) -> dict[Zone, float]:
        with self._lock:
            total = sum(self._zone_frame_counts.values())
            if total == 0:
                return {z: 0.0 for z in Zone}
            return {z: (self._zone_frame_counts[z] / total) * 100.0 for z in Zone}

    @property
    def heatmap(self) -> np.ndarray:
        """Normalised heatmap [0, 1] as float32 array (height × width)."""
        with self._lock:
            return self._normalise(self._heatmap_raw.copy())

    @property
    def trajectory(self) -> list[BallPosition]:
        with self._lock:
            return list(self._trajectory)

    @property
    def rod_goal_counts(self) -> dict[Rod, int]:
        with self._lock:
            return dict(self._rod_goal_counts)

    @property
    def events(self) -> list[GameEvent]:
        with self._lock:
            return list(self._events)

    @property
    def game_time_seconds(self) -> float:
        with self._lock:
            if self._game_start_time is None:
                return 0.0
            end = self._game_end_time if self._game_end_time is not None else time.time()
            return end - self._game_start_time

    # ── Per-team properties ───────────────────────────────────────────────────

    @property
    def team_heatmaps(self) -> dict[Team, np.ndarray]:
        """Separate normalised heatmaps for left and right halves of the field."""
        with self._lock:
            return {
                Team.LEFT: self._normalise(self._team_heatmap_raw[Team.LEFT].copy()),
                Team.RIGHT: self._normalise(self._team_heatmap_raw[Team.RIGHT].copy()),
            }

    @property
    def team_shot_counts(self) -> dict[Team, int]:
        with self._lock:
            return dict(self._team_shot_counts)

    @property
    def team_max_speed(self) -> dict[Team, float]:
        with self._lock:
            return dict(self._team_max_speed)

    @property
    def team_possession_pct(self) -> dict[Team, float]:
        """Percentage of frames where ball was in each team's half."""
        with self._lock:
            total = sum(self._team_frame_counts.values())
            if total == 0:
                return {Team.LEFT: 0.0, Team.RIGHT: 0.0}
            return {t: (self._team_frame_counts[t] / total) * 100.0 for t in Team}

    # ── Private helpers ───────────────────────────────────────────────────────

    def _reset_internal(self) -> None:
        cfg = self._config
        h = cfg.field_y2 - cfg.field_y1
        w = cfg.field_x2 - cfg.field_x1

        self._current_speed_ms: float = 0.0
        self._max_speed_ms: float = 0.0
        self._max_speed_timestamp: float = 0.0
        self._total_speed_sum: float = 0.0
        self._speed_frame_count: int = 0
        self._rebound_count: int = 0
        self._shot_count: int = 0
        self._prev_was_fast: bool = False
        self._prev_vx: Optional[float] = None
        self._prev_position: Optional[BallPosition] = None

        self._zone_frame_counts: dict[Zone, int] = {z: 0 for z in Zone}
        self._heatmap_raw: np.ndarray = np.zeros((h, w), dtype=np.float32)
        self._rod_goal_counts: dict[Rod, int] = {r: 0 for r in Rod}
        self._events: list[GameEvent] = []
        self._trajectory: deque[BallPosition] = deque(maxlen=self.TRAJECTORY_LENGTH)

        # Per-team
        self._team_frame_counts: dict[Team, int] = {Team.LEFT: 0, Team.RIGHT: 0}
        self._team_heatmap_raw: dict[Team, np.ndarray] = {
            Team.LEFT: np.zeros((h, w), dtype=np.float32),
            Team.RIGHT: np.zeros((h, w), dtype=np.float32),
        }
        self._team_max_speed: dict[Team, float] = {Team.LEFT: 0.0, Team.RIGHT: 0.0}
        self._team_shot_counts: dict[Team, int] = {Team.LEFT: 0, Team.RIGHT: 0}

        self._game_start_time: Optional[float] = None
        self._game_end_time: Optional[float] = None

    def _update_kinematics(self, position: BallPosition) -> None:
        prev = self._prev_position
        cfg = self._config
        dx = position.x - prev.x
        dy = position.y - prev.y
        pixel_dist = (dx ** 2 + dy ** 2) ** 0.5
        speed_ms = (pixel_dist / cfg.pixels_per_meter) * cfg.fps

        self._current_speed_ms = speed_ms
        self._total_speed_sum += speed_ms
        self._speed_frame_count += 1

        game_secs = 0.0
        if self._game_start_time is not None:
            game_secs = time.time() - self._game_start_time

        if speed_ms > self._max_speed_ms:
            self._max_speed_ms = speed_ms
            self._max_speed_timestamp = game_secs

        # Attribute speed to where the ball came FROM (prev position)
        team = self._current_team(prev)

        # Shot detection: speed crossing from below SHOT_SPEED_THRESHOLD_HIGH
        is_fast = speed_ms >= self.SHOT_SPEED_THRESHOLD_HIGH
        if is_fast and not self._prev_was_fast:
            self._shot_count += 1
            self._team_shot_counts[team] += 1
        self._prev_was_fast = is_fast

        # Per-team max speed
        if speed_ms > self._team_max_speed[team]:
            self._team_max_speed[team] = speed_ms

        # Rebound: horizontal velocity sign flip with sufficient magnitude
        if self._prev_vx is not None:
            if (abs(dx) > self.REBOUND_VX_THRESHOLD and
                    abs(self._prev_vx) > self.REBOUND_VX_THRESHOLD and
                    dx * self._prev_vx < 0):
                self._rebound_count += 1
        self._prev_vx = dx

    def _update_heatmap(self, position: BallPosition) -> None:
        cfg = self._config
        px = int(position.x - cfg.field_x1)
        py = int(position.y - cfg.field_y1)
        h, w = self._heatmap_raw.shape
        if 0 <= px < w and 0 <= py < h:
            r = self.HEATMAP_GAUSSIAN_RADIUS
            x0, x1 = max(0, px - r), min(w, px + r + 1)
            y0, y1 = max(0, py - r), min(h, py + r + 1)
            for gy in range(y0, y1):
                for gx in range(x0, x1):
                    dist_sq = (gx - px) ** 2 + (gy - py) ** 2
                    val = np.exp(-dist_sq / (2 * (r / 2.5) ** 2))
                    self._heatmap_raw[gy, gx] += val
                    team = self._current_team(position)
                    self._team_heatmap_raw[team][gy, gx] += val

    def _update_zone(self, position: BallPosition) -> None:
        cfg = self._config
        field_width = cfg.field_x2 - cfg.field_x1
        third = field_width / 3
        rel_x = position.x - cfg.field_x1
        if rel_x < third:
            self._zone_frame_counts[Zone.ATTACK_LEFT] += 1
        elif rel_x < 2 * third:
            self._zone_frame_counts[Zone.MIDDLE] += 1
        else:
            self._zone_frame_counts[Zone.ATTACK_RIGHT] += 1

    def _update_team_stats(self, position: BallPosition) -> None:
        team = self._current_team(position)
        self._team_frame_counts[team] += 1

    def _current_team(self, position: BallPosition) -> Team:
        """Ball in left half → LEFT team territory; right half → RIGHT team territory."""
        cfg = self._config
        midpoint = (cfg.field_x1 + cfg.field_x2) / 2
        return Team.LEFT if position.x < midpoint else Team.RIGHT

    def _record_rod_goal(self, event: GameEvent) -> None:
        """Attribute goal to the rod where the ball was just before it scored.

        Per README: 'Top-Scorer-Stange – x-Zone des Balls in den Frames vor dem Tor'.
        Looks back ~5 frames in the trajectory to find which rod shot it.
        """
        if not self._trajectory:
            return
        lookback = min(5, len(self._trajectory))
        pos_before = self._trajectory[-lookback]
        rod = self._x_to_rod(pos_before.x)
        self._rod_goal_counts[rod] += 1

    def _x_to_rod(self, x: float) -> Rod:
        """Map x-coordinate to one of the 5 rods (equal bands)."""
        cfg = self._config
        width = cfg.field_x2 - cfg.field_x1
        if width <= 0:
            return Rod.MIDFIELD
        rel = (x - cfg.field_x1) / width
        if rel < 0.2:
            return Rod.KEEPER_LEFT
        if rel < 0.4:
            return Rod.DEFENSE_LEFT
        if rel < 0.6:
            return Rod.MIDFIELD
        if rel < 0.8:
            return Rod.DEFENSE_RIGHT
        return Rod.KEEPER_RIGHT

    @staticmethod
    def _normalise(arr: np.ndarray) -> np.ndarray:
        max_val = arr.max()
        if max_val == 0:
            return arr
        return arr / max_val
