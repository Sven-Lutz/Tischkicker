import cv2
import numpy as np
import logging
logging.basicConfig(level=logging.INFO)
from camera.Camera import Camera
from ball_tracker.BallTracker import BallTracker
from table.Field import Field
from statistics.Statistics import Statistics, ScoreBoard


class GameController:
    # Spielzustände
    STATE_IDLE = "IDLE"
    STATE_CALIBRATING = "CALIBRATING"
    STATE_RUNNING = "RUNNING"
    STATE_PAUSED = "PAUSED"
    STATE_FINISHED = "FINISHED"

    WINDOW_NAME = "Tischkicker ball_tracker"

    def __init__(
            self,
            camera_source: int = 0,
            team_names: tuple[str, str] = ("Links", "Rechts"),
            cm_per_pixel: float = 0.1,
            goals_to_win: int = 10,

    ):
        """
        :param camera_source: Kamera-Index oder Videopfad
        :param team_names:    Namen der beiden Teams
        :param cm_per_pixel:  Pixel→cm Umrechnungsfaktor
        :param goals_to_win:  Tore bis zum Spielende
        """
        self.goals_to_win = goals_to_win
        self.state = self.STATE_IDLE

        # Komponenten
        self.camera = Camera(source=camera_source)
        self.ball_tracker = BallTracker(cm_per_pixel=cm_per_pixel)
        self.field = Field()
        self.scoreboard = ScoreBoard(team_names=team_names)
        self.statistics = Statistics()

    def start(self) -> None:
        """Startet das System: Kamera öffnen → kalibrieren → Spiel-Loop."""
        logging.info("[game_controller] Starte System …")

        if not self.camera.start():
            logging.error("[game_controller] Abbruch: Kamera nicht verfügbar.")
            return

        self.ball_tracker.fps = self.camera.fps

        self._run_calibration()

        self._run_game_loop()

        self._shutdown()

    def stop(self) -> None:
        """Beendet den Spiel-Loop von außen ."""
        self.state = self.STATE_FINISHED

    def _run_calibration(self) -> None:
        """Kalibrierungsphase: HSV-Werte und Tor-Zonen interaktiv festlegen."""
        self.state = self.STATE_CALIBRATING

        # 1. HSV-Kalibrierung
        logging.info(
            "[game_controller] HSV-Kalibrierung – Passe die Trackbars an, bis nur der Ball sichtbar ist. Drücke 'q' zum Fortfahren.")
        self.ball_tracker.calibrate_hsv_interactive(self.camera)

        # 2. Tor-Kalibrierung
        logging.info("[game_controller] Tor-Kalibrierung – Bitte Tore markieren.")
        ok, frame = self.camera.read_frame()
        if not ok:
            logging.error("[game_controller] Kein Frame für Kalibrierung.")
            return

        self.field.calibrate_interactive(frame, window_name=self.WINDOW_NAME)
        # 3. Wand-Kalibrierung

        self.state = self.STATE_RUNNING
        logging.info("[game_controller] Kalibrierung fertig – Spiel startet!")

    def _run_game_loop(self) -> None:
        """Haupt-Loop: Frame lesen → tracken → prüfen → anzeigen."""
        print(f"[game_controller] Spiel läuft. Tasten: [p] Pause  [r] Reset  [q] Beenden")

        while self.state in (self.STATE_RUNNING, self.STATE_PAUSED):
            ok, frame = self.camera.read_frame()
            if not ok:
                logging.error("[game_controller] Kein Frame mehr – Loop beendet.")
                break

            if self.state == self.STATE_RUNNING:
                self._process_frame(frame)

            self._render_hud(frame)
            cv2.imshow(self.WINDOW_NAME, frame)

            self._handle_keys(cv2.waitKey(1) & 0xFF)

    def _process_frame(self, frame: np.ndarray) -> None:
        """Führt Tracking, Tor-Check und Statistik für einen einzelnen Frame durch."""
        # 1. Ball tracken
        ball_pos = self.ball_tracker.update(frame)
        self.ball_tracker.draw(frame)

        # 2. Geschwindigkeit aufzeichnen
        self.statistics.record_speed(self.ball_tracker.speed_cm_s)

        # 3. Trajektorie aufzeichnen
        if ball_pos is not None:
            self.statistics.trajectory_add(ball_pos)
        
        # 4. Torzonen zeichnen
        self.field.draw(frame)
        
        # 5. Trajektorie zeichnen
        self.draw_trajectory(frame)

        # 6. Tor-Check
        scored_goals = self.field.check_goals(ball_pos)
        for goal_name in scored_goals:
            self.scoreboard.register_goal(goal_name, self.ball_tracker.speed_cm_s)
            self._on_goal(goal_name)

        # 7. Spielende prüfen
        for team in self.scoreboard.team_names:
            if self.scoreboard.get_score(team) >= self.goals_to_win:
                self._on_game_over(team)
                return
    def draw_trajectory(self, frame):
        """Zeichnet die Trajektorie der letzten 5 Sekunden."""
        trajectory = self.statistics.get_trajectory_count()
        if len(trajectory) < 2:
            return
        
        for i in range(1, len(trajectory)):
            # Extrahiere nur x,y position
            p1 = (int(trajectory[i-1][0]), int(trajectory[i-1][1]))
            p2 = (int(trajectory[i][0]), int(trajectory[i][1]))
            cv2.line(frame, p1, p2, (255, 0, 0), 2)
    # ------------------------------------------------------------------
    # Event-Handler
    # ------------------------------------------------------------------

    def _on_goal(self, team: str) -> None:
        """Wird aufgerufen wenn ein Tor fällt."""
        print(f"[game_controller] TOR für {team}! Neuer Stand: {self.scoreboard.get_score_string()}")
        # Hier könnten Animationen, Sounds etc. ausgelöst werden

    def _on_game_over(self, winner: str) -> None:
        """Wird aufgerufen wenn ein Team gewonnen hat."""
        print(f"\n[game_controller] Spiel vorbei! Gewinner: {winner}")
        print(self.statistics.summary(self.scoreboard))
        self.state = self.STATE_FINISHED

    def _handle_keys(self, key: int) -> None:
        """Verarbeitet Tastatur-Eingaben."""
        if key == ord('q'):
            print("[game_controller] Beenden durch User.")
            self.state = self.STATE_FINISHED

        elif key == ord('p'):
            if self.state == self.STATE_RUNNING:
                self.state = self.STATE_PAUSED
                print("[game_controller] Pausiert.")
            elif self.state == self.STATE_PAUSED:
                self.state = self.STATE_RUNNING
                print("[game_controller] Fortgesetzt.")

        elif key == ord('r'):
            self.scoreboard.reset()
            self.statistics.reset()
            print("[game_controller] Spielstand und Statistiken zurückgesetzt.")

    # ------------------------------------------------------------------
    # HUD (Head-Up-Display)
    # ------------------------------------------------------------------

    def _render_hud(self, frame: np.ndarray) -> None:
        """Rendert Spielstand, Geschwindigkeit und Status ins Bild."""
        h, w = frame.shape[:2]

        cv2.rectangle(frame, (0, 0), (w, 50), (0, 0, 0), -1)

        score_text = f"  {self.scoreboard.get_score_string()}  "
        cv2.putText(frame, score_text, (w // 2 - 60, 35),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)

        cv2.putText(frame, self.scoreboard.team_names[0], (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 200, 255), 2)
        cv2.putText(frame, self.scoreboard.team_names[1], (w - 100, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 200, 255), 2)

        avg_speed = self.statistics.average_speed()
        cv2.putText(frame, f"Akt: {self.ball_tracker.speed_cm_s:.1f} cm/s  |  Ø {avg_speed:.1f} cm/s",
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        if self.state == self.STATE_PAUSED:
            cv2.putText(frame, "⏸ PAUSE", (w // 2 - 60, h // 2),
                        cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 255, 255), 3)

    def _shutdown(self) -> None:
        """Gibt alle Ressourcen frei und zeigt die Zusammenfassung."""
        print("\n" + self.statistics.summary(self.scoreboard))
        self.camera.stop()
        cv2.destroyAllWindows()
        print("[game_controller] System beendet.")
