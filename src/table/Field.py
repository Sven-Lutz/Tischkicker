from __future__ import annotations

import cv2
import numpy as np


class GoalZone:
    """
    Represents a goal area on the table.
    """

    COOLDOWN_FRAMES = 30  #Cooldown zwischen Torzählungen

    def __init__(self, name: str, x: int, y: int, w: int, h: int):
        """
        :param name: Goal name
        :param x, y: Top-left corner of the zone
        :param w, h: Width and height of the zone
        """
        self.name = name
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self._cooldown_counter: int = 0  # Frames seit letztem Tor
        self._ball_was_inside: bool = False

    # ------------------------------------------------------------------
    # Kollision
    # ------------------------------------------------------------------

    def contains_point(self, point: tuple[int, int]) -> bool:
        """Returns True if the point lies within the goal box."""
        px, py = point
        return self.x <= px <= self.x + self.w and self.y <= py <= self.y + self.h

    def check_goal(self, ball_center: tuple[int, int] | None) -> bool:
        """
        Checks whether a goal was scored in this frame.
        Uses cooldown and state tracking to avoid duplicate scoring.

        :param ball_center: (x, y) center of the ball, or None if not visible
        :return: True exactly when a new goal is counted in this frame
        """
        # Cooldown herunterzählen
        if self._cooldown_counter > 0:
            self._cooldown_counter -= 1

        if ball_center is None:
            self._ball_was_inside = False
            return False

        inside = self.contains_point(ball_center)

        # Tor nur zählen wenn:
        #  - Ball jetzt ERSTMALS in der Zone (Flanke False→True)
        #  - kein aktiver Cooldown
        is_new_goal = inside and not self._ball_was_inside and self._cooldown_counter == 0

        if is_new_goal:
            self._cooldown_counter = self.COOLDOWN_FRAMES

        self._ball_was_inside = inside
        return is_new_goal

    # ------------------------------------------------------------------
    # Visualisierung
    # ------------------------------------------------------------------

    def draw(self, frame: np.ndarray, color: tuple = (0, 0, 255)) -> None:
        """Draws the goal zone onto the frame."""
        cv2.rectangle(frame, (self.x, self.y), (self.x + self.w, self.y + self.h), color, 2)
        cv2.putText(frame, f"Tor {self.name}", (self.x, self.y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


# ---------------------------------------------------------------------------


class Field:
    """
    Represents the playing field.
    Manages calibration and all GoalZone objects.
    """

    def __init__(self):
        self.goal_zones: list[GoalZone] = []
        self._calibrated: bool = False
        # Klick-Puffer für interaktive Kalibrierung
        self._click_points: list[tuple[int, int]] = []
        self._current_goal_name: str = ""

    # ------------------------------------------------------------------
    # Kalibrierung
    # ------------------------------------------------------------------

    def add_goal_zone(self, name: str, x: int, y: int, w: int, h: int) -> None:
        """Adds a GoalZone."""
        self.goal_zones.append(GoalZone(name, x, y, w, h))

    def calibrate_interactive(self, frame: np.ndarray, window_name: str = "Kalibrierung") -> None:
        """
        The user clicks two points per goal (top-left, bottom-right).
        Ends once all goals have been calibrated.
        """
        goal_names = ["Links", "Rechts"]
        self._click_points = []

        def on_mouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                self._click_points.append((x, y))
                print(f"[Field] Klick {len(self._click_points)}: ({x}, {y})")

        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, on_mouse)

        for goal_name in goal_names:
            self._click_points = []
            print(f"\n[Field] Kalibrierung '{goal_name}': Klicke obere-links, dann untere-rechts Ecke des Tors.")

            while len(self._click_points) < 2:
                display = frame.copy()
                cv2.putText(display, f"Tor '{goal_name}': 2 Ecken klicken ({len(self._click_points)}/2)",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                # Vorherige Zonen schon einzeichnen
                for gz in self.goal_zones:
                    gz.draw(display)
                cv2.imshow(window_name, display)
                cv2.waitKey(30)

            x1, y1 = self._click_points[0]
            x2, y2 = self._click_points[1]
            self.add_goal_zone(goal_name, min(x1, x2), min(y1, y2),
                               abs(x2 - x1), abs(y2 - y1))
            print(f"[Field] Tor '{goal_name}' kalibriert.")

        cv2.destroyWindow(window_name)
        self._calibrated = True
        print("[Field] Kalibrierung abgeschlossen.")

    # ------------------------------------------------------------------
    # Tor-Check (delegiert an GoalZones)
    # ------------------------------------------------------------------

    def check_goals(self, ball_center: tuple[int, int] | None) -> list[str]:
        """
        Checks all goal zones and returns the names of the goals scored in this frame.
        """
        scored = []
        for gz in self.goal_zones:
            if gz.check_goal(ball_center):
                scored.append(gz.name)
        return scored

    # ------------------------------------------------------------------
    # Visualisierung
    # ------------------------------------------------------------------

    def draw(self, frame: np.ndarray) -> None:
        """Draws all goal zones onto the frame."""
        for gz in self.goal_zones:
            gz.draw(frame)