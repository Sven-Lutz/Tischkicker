from __future__ import annotations

import logging
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
import time

from typing import Optional
import numpy as np

from src.game_controller.ScoreBoard import ScoreBoard
from src.GameEvents import (
    BallPosition,
    EventType,
    GameConfig,
    GameEvent,
    Rod,
    Team,
    Zone,
)

logger = logging.getLogger(__name__)


class Statistics:
    """
    Tracks all game statistics from ball positions and events.
    """
    TRAJECTORY_LENGTH = 30
    HEATMAP_GAUSSIAN_RADIUS = 15
    SHOT_SPEED_THRESHOLD_LOW = 1.0
    SHOT_SPEED_THRESHOLD_HIGH = 3.0
    REBOUND_VX_THRESHOLD = 2.0

    def __init__(self, config: GameConfig) -> None:
        self._config = config
        self._lock = threading.Lock()
        self._reset_internal()

    #--Public interface-------------------------------------------------------

    def update(
            self, position: Optional[BallPosition], event: Optional[GameEvent]
    )->None:
        """Updates internal stats based on new frames and events."""
        with self._lock:
            if event is not None:
                self._events.append(event)
                is_goal = event.event_type == EventType.GOAL
                if is_goal and event.team is not None:
                    self._record_rod_goal()

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
        """Starts the timer."""
        with self._lock:
            self._game_start_time = time.time()
            self._game_end_time = None

    def stop_timer(self) -> None:
        """Stops the timer."""
        with self._lock:
            self._game_end_time = time.time()

    def reset(self): -> None:
    """Resets all collected statistics."""
        with self._lock:
            self._reset_internal()

    #--Summary----------------------------------------------------------------

    def summary(self, scoreboard: ScoreBoard) -> str:
        """Returns a formatted summary."""
        with self._lock:
            avg_speed = self._average_speed()
            max_speed = self._max_speed()
            frames = self._speed_frame_count()
            rebound_cnt = self._rebound_count()
            shot_cnt = self._shot_count()

            total_frames = sum(self._team_frame_counts.values())
            poss_red = poss_black = 0.0
            if total_frames > 0:
                poss_red = self._team_frame_counts[Team.RED] / total_frames
                poss_black = self._team_frame_counts[Team.BLACK] / total_frames

        goals = scoreboard.goal_events

        lines = [
            "=" * 45,
            "             GAME SUMMARY",
            "=" * 45,
            f"  Final Score:       {scoreboard.get_score_string()}",
            f"  Total Goals:       {len(goals)}",
            f"  Avg. Speed:        {avg_speed:.2f} m/s",
            f"  Max Speed:         {max_speed:.2f} m/s",
            f"  Total Shots:       {shot_cnt}",
            f"  Total Rebounds:    {rebound_cnt}",
            f"  Analyzed Frames:   {frames}",
            "-" * 45,
            f" Possesion Red: {poss_red * 100:.1f} %",
            f"  Possesion Black: {poss_black * 100:.1f} %",
            "-" * 45,
        ]

        for i, goal in enumerate(goals, 1):
            time_obj = time.localtime(goal.timestamp)
            t_str = time.strftime("%H:%M:%S", time_obj)
            speed = goal.value if goal.value is not None else 0.0

            lines.append(
                f"  Goal {i}: {goal.team.value} | "
                f"{speed:.2f} m/s | {t_str}"
            )

        lines.append("=" * 45)
        return "\n".join(lines)

    #--Core------------------------------------------------------------------

    @property
    def current_speed_ms(self) -> float:
        with self._lock:
            return self._current_speed_ms

    @property
    def max_speed_ms(self) -> float:
        with self._lock:
            return self._max_speed_ms

    @property
    def average_speed_ms(self) -> float:
        with self._lock:
            return self._average_speed_ms_unlocked()

    def _average_speed_ms_unlocked(self) -> float:
        if self._speed_frame_count == 0:
            return 0.0
        return self._total_speed_sum / self._speed_frame_count

    @property
    def heatmap(self) -> np.ndarray:
        """Normalised heatmap as float array"""
        with self._lock:
            return self._normalize(self._heatmap_raw.copy())

    @property
    def trajectory(self) -> list[BallPosition]:
        with self._lock:
            return list(self._trajectory)

    @property
    def rod_goal_counts(self) -> dict[Rod, int]:
        """Returns the goal count attributed to each rod."""
        with self._lock:
            return dict(self._rod_goal_counts)

    #--Helpers---------------------------------------------------------------

    def _reset_internal(self)-> None:
        """Prepares all arrays and counters for a new session."""
        cfg = self._config
        h = max(1, cfg.field_y2 - cfg.field_y1)
        w = max(1, cfg.field_x2 - cfg.field_x1)

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
        self._trajectory: deque[BallPosition] = deque(
            maxlen=self.TRAJECTORY_LENGTH
        )

        self._team_frame_counts: dict[Team, int] = {
            Team.RED: 0, Team.BLACK: 0
        }

        self._team_max_speed: dict[Team, float] = {
            Team.RED: 0.0, Team.BLACK: 0.0
        }

        self._team_shot_counts: dict[Team, int] = {
            Team.RED: 0, Team.BLACK: 0
        }

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

        team = self._current_team(prev)

        is_fast = speed_ms >= self.SHOT_SPEED_THRESHOLD_HIGH
        if is_fast and not self._prev_was_fast:
            self._shot_count += 1
            self._team_shot_counts[team] += 1
        self._prev_was_fast = is_fast

        if speed_ms > self._team_max_speed[team]:
            self._team_max_speed[team] = speed_ms

        if self._prev_vx is not None:
            if (
                    abs(dx) > self.REBOUND_VX_THRESHOLD
                    and abs(self._prev_vx) > self.REBOUND_VX_THRESHOLD
                    and dx * self._prev_vx < 0
            ):
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


    def _update_zone(self, position: BallPosition) -> None:
        """Tracks the exact zone based on the 8 rod areas."""
        rod = self._x_to_rod(position.x)

        zone_mapping = {
            Rod.LEFT_GOALIE: Zone.LEFT_GOAL_AREA,
            Rod.LEFT_DEFENSE: Zone.LEFT_DEFENSE_AREA,
            Rod.RIGHT_ATTACK: Zone.RIGHT_ATTACK_AREA,
            Rod.LEFT_MIDFIELD: Zone.LEFT_MIDFIELD_AREA,
            Rod.RIGHT_MIDFIELD: Zone.RIGHT_MIDFIELD_AREA,
            Rod.LEFT_ATTACK: Zone.LEFT_ATTACK_AREA,
            Rod.RIGHT_DEFENSE: Zone.RIGHT_DEFENSE_AREA,
            Rod.RIGHT_GOALIE: Zone.RIGHT_GOAL_AREA,
        }
        zone = zone_mapping[rod]
        self._zone_frame_counts[zone] += 1

    def _update_team_stats(self, position: BallPosition) -> None:
        team = self._current_team(position)
        self._team_frame_counts[team] += 1

    def _current_team(self, postion: BallPosition) -> Team:
        """Determines possession based on which physical rod the ball is near."""
        rod = self._x_to_rod(postion.x)

        if rod in (
            Rod.LEFT_GOALIE,
            Rod.LEFT_DEFENSE,
            Rod.LEFT_MIDFIELD,
            Rod.LEFT_ATTACK
        ):
            return Team.RED
        else:
            return Team.BLACK

    def _record_rod_goal(self) -> None:
        """Attributes a goal to the last active rod."""
        if not self._trajectory:
            return
        lookback = min(5, len(self._trajectory))
        pos_before = self._trajectory[-lookback]
        rod = self._x_to_rod(pos_before.x)
        self._rod_goal_counts[rod] += 1

    def _x_to_rid(self, x: float) -> Rod:
        cfg = self._config
        width = cfg.field_x2 - cfg.field_x1
        if width <= 0:
            return Rod.LEFT_MIDFIELD

        rel = (x - cfg.field_x1) / width

        if rel < 0.125: return Rod.LEFT_GOALIE
        if rel < 0.250: return Rod.LEFT_DEFENSE
        if rel < 0.375: return Rod.RIGHT_ATTACK
        if rel < 0.500: return Rod.LEFT_MIDFIELD
        if rel < 0.625: return Rod.RIGHT_MIDFIELD
        if rel < 0.750: return Rod.LEFT_ATTACK
        if rel < 0.875: return Rod.RIGHT_DEFENSE
        return Rod.RIGHT_GOALIE

    @staticmethod
    def _normalise(arr: np.ndarray) -> np.ndarray:
        max_val = arr.max()
        if max_val == 0:
            return arr
        return arr / max_val