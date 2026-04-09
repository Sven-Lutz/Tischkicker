import cv2
import numpy as np


class HUDRenderer:
    """Renders the head-up display (HUD) with score, speed, and status."""
    
    def __init__(self):
        """Initializes the HUD renderer."""
        pass
    
    def render_hud(
        self,
        frame: np.ndarray,
        scoreboard,
        statistics,
        ball_tracker,
        state: str
    ) -> None:
        """
        Renders score, speed, and status into the image.
        
        :param frame: The image to render into
        :param scoreboard: ScoreBoard-Instance
        :param statistics: Statistics-Instance
        :param ball_tracker: BallTracker-Instance
        :param state: Current game state (e.g. "RUNNING", "PAUSED")
        """
        h, w = frame.shape[:2]

        # Schwarzer Balken oben
        cv2.rectangle(frame, (0, 0), (w, 50), (0, 0, 0), -1)

        # Spielstand in der Mitte
        score_text = f"  {scoreboard.get_score_string()}  "
        cv2.putText(frame, score_text, (w // 2 - 60, 35),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)

        # Team-Namen links und rechts
        cv2.putText(frame, scoreboard.team_names[0], (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 200, 255), 2)
        cv2.putText(frame, scoreboard.team_names[1], (w - 100, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 200, 255), 2)

        # Geschwindigkeit unten
        avg_speed = statistics.average_speed()
        cv2.putText(frame, f"Akt: {ball_tracker.speed_cm_s:.1f} cm/s  |  Ø {avg_speed:.1f} cm/s",
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Pause-Anzeige
        if state == "PAUSED":
            cv2.putText(frame, "⏸ PAUSE", (w // 2 - 60, h // 2),
                        cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 255, 255), 3)
    
    def draw_trajectory(self, frame: np.ndarray, trajectory: list) -> None:
        """
        Draws the trajectory of the last few seconds.
        
        :param frame: The image to draw into
        :param trajectory: List of positions
        """
        if len(trajectory) < 2:
            return
        
        for i in range(1, len(trajectory)):
            # Extrahiere nur x,y Position
            p1 = (int(trajectory[i-1][0]), int(trajectory[i-1][1]))
            p2 = (int(trajectory[i][0]), int(trajectory[i][1]))
            cv2.line(frame, p1, p2, (255, 0, 0), 2)
