from dataclasses import dataclass, field
from datetime import datetime
import time


@dataclass
class GoalEvent:
    """Stores a single goal event."""
    team: str
    timestamp: datetime = field(default_factory=datetime.now)
    ball_speed_cm_s: float = 0.0


class ScoreBoard:
    """
    Manages the score and goal history.
    """

    def __init__(self, team_names: tuple[str, str] = ("Links", "Rechts")):
        """
        :param team_names: Names of the two teams / sides
        """
        self.team_names = team_names
        self._scores: dict[str, int] = {name: 0 for name in team_names}
        self._goal_events: list[GoalEvent] = []

    def register_goal(self, team: str, ball_speed: float = 0.0) -> None:
        """
        Records a goal for the specified team.

        :param team: Team name (must exist in team_names)
        :param ball_speed: Ball speed at the time of the goal (cm/s)
        """
        if team not in self._scores:
            print(f"[ScoreBoard] Unbekanntes Team: {team}")
            return

        self._scores[team] += 1
        event = GoalEvent(team=team, ball_speed_cm_s=ball_speed)
        self._goal_events.append(event)
        print(f"[ScoreBoard] TOR für '{team}'! Stand: {self.get_score_string()} | {ball_speed:.1f} cm/s")

    def get_score(self, team: str) -> int:
        return self._scores.get(team, 0)

    def get_score_string(self) -> str:
        """Returns the score as a readable string, e.g. '3 : 2'."""
        a, b = self.team_names
        return f"{self._scores[a]} : {self._scores[b]}"

    def reset(self) -> None:
        """Resets the score."""
        for name in self.team_names:
            self._scores[name] = 0
        self._goal_events.clear()
        print("[ScoreBoard] Spielstand zurückgesetzt.")

    @property
    def goal_events(self) -> list[GoalEvent]:
        return list(self._goal_events)


class Statistics:
    """
    Collects and computes game statistics.
    """

    def __init__(self):
        self._speed_samples: list[float] = []  # cm/s, alle Frames
        self._max_speed: float = 0.0
        self._frame_count: int = 0
        self.positions = [] #für die Trajektorie
        self.window_seconds: int = 5

    def record_speed(self, speed_cm_s: float) -> None:
        """Records a speed measurement."""
        if speed_cm_s <= 0:
            return
        self._speed_samples.append(speed_cm_s)
        if speed_cm_s > self._max_speed:
            self._max_speed = speed_cm_s
        self._frame_count += 1

    def average_speed(self) -> float:
        """Average speed over the entire game."""
        if not self._speed_samples:
            return 0.0
        return sum(self._speed_samples) / len(self._speed_samples)

    def max_speed(self) -> float:
        """Maximum recorded speed."""
        return self._max_speed

    def trajectory_add(self, position):
        now = time.time()
        x = position[0]
        y = position[1]
        self.positions.append((x, y, now))
        self._cleanup(now)

    def _cleanup(self, now):
        self.positions = [(x, y, t) for (x, y, t) in self.positions
                          if now - t <= self.window_seconds
                          ]

    def get_trajectory_count(self):
        return self.positions

    def summary(self, scoreboard: ScoreBoard) -> str:
        """Returns a formatted summary."""
        lines = [
            "=" * 40,
            "         SPIEL-ZUSAMMENFASSUNG",
            "=" * 40,
            f"  Endstand:          {scoreboard.get_score_string()}",
            f"  Tore gesamt:       {len(scoreboard.goal_events)}",
            f"  Ø Geschwindigkeit: {self.average_speed():.1f} cm/s",
            f"  Max Geschw.:       {self.max_speed():.1f} cm/s",
            f"  Frames analysiert: {self._frame_count}",
            "-" * 40,
        ]
        for i, event in enumerate(scoreboard.goal_events, 1):
            lines.append(f"  Tor {i}: {event.team} | {event.ball_speed_cm_s:.1f} cm/s "
                         f"| {event.timestamp.strftime('%H:%M:%S')}")
        lines.append("=" * 40)
        return "\n".join(lines)

    def reset(self) -> None:
        self._speed_samples.clear()
        self._max_speed = 0.0
        self._frame_count = 0
        self.positions.clear()
