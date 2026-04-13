from __future__ import annotations

import cv2
import numpy as np


class GoalZone:
    """
    Represents a goal opening on the table.
    """

    COOLDOWN_FRAMES = 30  #Cooldown zwischen Torzählungen

    def __init__(self, name: str, x: int, y: int, w: int, h: int, side: str):
        """
        :param name: Goal name
        :param x, y: Top-left corner of the zone
        :param w, h: Width and height of the zone
        :param side: Side of the table the goal belongs to ("left" or "right")
        """
        self.name = name
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.side = side
        self._cooldown_counter: int = 0
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

    def draw(self, frame: np.ndarray, color: tuple = (0, 0, 255)) -> None:
        """Draws the goal opening as a centered line segment on the field border."""
        border_x = self.x + self.w if self.side == "left" else self.x
        cv2.line(frame, (border_x, self.y), (border_x, self.y + self.h), color, 4)

        label_x = self.x if self.side == "left" else max(0, self.x - 60)
        label_y = max(0, self.y - 6)
        cv2.putText(frame, f"Tor {self.name}", (label_x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


class Field:
    """
    Represents the playing field.
    Manages calibration and all GoalZone objects.
    """
    GOAL_DEPTH_RATIO  = 0.04
    GOAL_HEIGHT_RATIO = 0.27

    def __init__(self):
        self.goal_zones: list[GoalZone] = []
        self.corners: list[tuple[int, int]] = []   # 4 Ecken
        self._calibrated: bool = False
        self._click_buffer: list[tuple[int, int]] = []

    def calibrate_interactive(self, frame: np.ndarray, window_name: str = "Kalibrierung") -> None:
        """
        The user calibrates four points for field (top-left, bottom-right).
        Ends once all goals have been calibrated.
        """
        self._click_buffer = []

        def on_mouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and len(self._click_buffer) < 4:
                self._click_buffer.append((x, y))
                print(f"[Field] Ecke {len(self._click_buffer)}/4: ({x}, {y})")

        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, on_mouse)

        while len(self._click_buffer) < 4:
            display = frame.copy()

            # Bereits geklickte Ecken einzeichnen
            for i, pt in enumerate(self._click_buffer):
                cv2.circle(display, pt, 6, (0, 255, 255), -1)
                cv2.putText(display, str(i + 1), (pt[0] + 8, pt[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 5)

            # Verbindungslinien der bisher geklickten Punkte
            if len(self._click_buffer) >= 2:
                for i in range(len(self._click_buffer) - 1):
                    cv2.line(display, self._click_buffer[i], self._click_buffer[i + 1],
                             (0, 255, 255), 7)

            cv2.imshow(window_name, display)
            cv2.waitKey(30)

        self.corners = self._click_buffer.copy()
        self._compute_goal_zones()
        self._calibrated = True

        # Vorschau der berechneten Torzonen anzeigen
        self._show_calibration_result(frame.copy(), window_name)
        print("[Field] Kalibrierung abgeschlossen.")

    def _compute_goal_zones(self) -> None:
        """
        Berechnet die zwei Torzonen automatisch aus den 4 Ecken.

        Ecken-Reihenfolge (Uhrzeigersinn):
          0: oben-links   1: oben-rechts
          3: unten-links  2: unten-rechts

        Linkes Tor:  Mitte zwischen Ecke 0 und 3 (linker Rand)
        Rechtes Tor: Mitte zwischen Ecke 1 und 2 (rechter Rand)
        """
        tl, tr, br, bl = self.corners  # top-left, top-right, bottom-right, bottom-left

        # Spielfeldmaße schätzen (Pixel)
        field_width  = int(((tr[0] - tl[0]) + (br[0] - bl[0])) / 2)
        field_height = int(((bl[1] - tl[1]) + (br[1] - tr[1])) / 2)

        goal_depth  = max(8, int(field_width  * self.GOAL_DEPTH_RATIO))
        goal_height = max(20, int(field_height * self.GOAL_HEIGHT_RATIO))

        # Mitte des linken Randes
        left_mid_x = int((tl[0] + bl[0]) / 2)
        left_mid_y = int((tl[1] + bl[1]) / 2)

        # Mitte des rechten Randes
        right_mid_x = int((tr[0] + br[0]) / 2)
        right_mid_y = int((tr[1] + br[1]) / 2)

        # GoalZone links: ragt nach links aus dem Spielfeld raus
        gz_left = GoalZone(
            name="Links",
            x=left_mid_x - goal_depth,
            y=left_mid_y - goal_height // 2,
            w=goal_depth,
            h=goal_height,
            side="left",
        )

        # GoalZone rechts: ragt nach rechts aus dem Spielfeld raus
        gz_right = GoalZone(
            name="Rechts",
            x=right_mid_x,
            y=right_mid_y - goal_height // 2,
            w=goal_depth,
            h=goal_height,
            side="right",
        )

        self.goal_zones = [gz_left, gz_right]

        print(f"[Field] Tor Links:  x={gz_left.x}, y={gz_left.y}, "
              f"w={gz_left.w}, h={gz_left.h}")
        print(f"[Field] Tor Rechts: x={gz_right.x}, y={gz_right.y}, "
              f"w={gz_right.w}, h={gz_right.h}")

    def _show_calibration_result(self, frame: np.ndarray, window_name: str) -> None:
        """Zeigt 2 Sekunden lang das Ergebnis der Kalibrierung."""
        # Spielfeld-Polygon einzeichnen
        pts = np.array(self.corners, dtype=np.int32)
        cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 255), thickness=6)

        # Torzonen als Teil der Seitenlinie einzeichnen
        for gz in self.goal_zones:
            gz.draw(frame)


    # ------------------------------------------------------------------
    # Tor-Check
    # ------------------------------------------------------------------

    def check_goals(self, ball_center: tuple[int, int] | None) -> list[str]:
        """Gibt Namen aller Tore zurück die in diesem Frame erzielt wurden."""
        return [gz.name for gz in self.goal_zones if gz.check_goal(ball_center)]

    # ------------------------------------------------------------------
    # Visualisierung
    # ------------------------------------------------------------------

    def draw(self, frame: np.ndarray) -> None:
        """Zeichnet Spielfeld-Umriss und alle Torzonen."""
        if self.corners:
            pts = np.array(self.corners, dtype=np.int32)
            cv2.polylines(frame, [pts], isClosed=True, color=(0, 200, 200), thickness=5)

        for gz in self.goal_zones:
            gz.draw(frame)