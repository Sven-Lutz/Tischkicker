from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING, Optional, Tuple

import cv2
import numpy as np

from camera.Camera import Camera
from ball_tracker.BallTracker import BallTracker
from table.Field import GoalDetector
from statistics.Statistics import Statistics, ScoreBoard
from game_controller.EventHandler import EventHandler
from game_controller.HUDRenderer import HUDRenderer
from game_controller.SnapshotManager import SnapshotManager

if TYPE_CHECKING:
    # Replace with the actual GUI class import used in your project
    from gui import KickerGUI

logger = logging.getLogger(__name__)

TARGET_FPS = 30.0
FRAME_INTERVAL = 1.0 / TARGET_FPS


class GameController:
    """Wires all backend components together."""

    STATE_IDLE = "IDLE"
    STATE_CALIBRATING = "CALIBRATING"
    STATE_RUNNING = "RUNNING"
    STATE_PAUSED = "PAUSED"
    STATE_FINISHED = "FINISHED"

    WINDOW_NAME = "Tischkicker Ball Tracker"

    def __init__(
            self,
            gui: Optional["KickerGUI"] = None,
            camera_source: int = 0,
            team_names: Tuple[str, str] = ("Left", "Right"),
            cm_per_pixel: float = 0.1,
            goals_to_win: int = 10,
            snapshot_dir: str = "../snapshots"
    ) -> None:
        """
        :param gui: Optional GUI instance for signal emitting.
        :param camera_source: Camera index or video path.
        :param team_names: Tuple containing the names of the two teams.
        :param cm_per_pixel: Conversion factor from pixels to centimeters.
        :param goals_to_win: Number of goals required to end the game.
        :param snapshot_dir: Directory to save goal snapshots.
        """
        self._gui = gui
        self.goals_to_win = goals_to_win
        self.state = self.STATE_IDLE

        # Core
        self.camera = Camera(source=camera_source)
        self.ball_tracker = BallTracker(cm_per_pixel=cm_per_pixel)
        self.field = GoalDetector()
        self.scoreboard = ScoreBoard(team_names=team_names)
        self.statistics = Statistics()

        # Helper
        self.snapshot_manager = SnapshotManager(snapshot_dir=snapshot_dir)
        self.hud_renderer = HUDRenderer()
        self.event_handler = EventHandler(game_controller=self)

        # Threading mechanisms
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # -- Public API ------------------------------------------------------------

    def start_game(self) -> None:
        """Initializes the camera, runs calibration, and starts the game loop thread."""
        logger.info("Starting system...")
        self._stop_event.clear()

        if not self.camera.start():
            logger.error("Camera could not be opened. Aborting start.")
            return

        self.ball_tracker.fps = self.camera.fps

        # Run interactive calibration on the main thread (OpenCV requirement)
        self._run_calibration()

        self.state = self.STATE_RUNNING

        if self._gui:
            self._gui.show_dashboard()

        # Start the game loop in a background thread
        self._thread = threading.Thread(
            target=self._game_loop,
            daemon=True,
            name="GameLoopThread"
        )
        self._thread.start()
        logger.info("Game loop thread started successfully.")

    def stop_game(self) -> None:
        """Stops the background loop and shuts down resources."""
        logger.info("Stopping game...")
        self.state = self.STATE_FINISHED
        self._stop_event.set()

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)

        self._shutdown()

    def new_game(self) -> None:
        """Resets everything and returns to the initial state for a new session."""
        logger.info("Starting a new game session.")
        self.stop_game()

        self.scoreboard.reset()
        self.statistics.reset()
        self.state = self.STATE_IDLE

        if self._gui:
            self._gui.show_start_screen()

    def quit(self) -> None:
        """Performs a clean shutdown of the application."""
        logger.info("Application quit requested.")
        self.stop_game()

    # -- Core Logic ------------------------------------------------------------

    def _run_calibration(self) -> None:
        """Calibration phase: interactively set HSV values and goal zones."""
        self.state = self.STATE_CALIBRATING
        logger.info("HSV Calibration: Adjust trackbars until only the ball is visible. Press 'q' to continue.")
        self.ball_tracker.calibrate_hsv_interactive(self.camera)

        logger.info("Goal Calibration: Please mark the goals on the field.")
        ok, frame = self.camera.read_frame()
        if not ok:
            logger.error("No frame available for calibration.")
            return

        self.field.calibrate_interactive(frame, window_name=self.WINDOW_NAME)
        logger.info("Calibration finished. Transitioning to RUNNING state.")

    def _game_loop(self) -> None:
        """Main background loop: reads frames, tracks the ball, and handles events."""
        logger.info("Game loop is active. Awaiting frame processing.")

        while not self._stop_event.is_set():
            t_start = time.monotonic()

            if self.state not in (self.STATE_RUNNING, self.STATE_PAUSED):
                time.sleep(0.1)
                continue

            ok, frame = self.camera.read_frame()
            if not ok:
                logger.warning("Failed to read frame. Ending game loop.")
                break

            ball_pos = None

            if self.state == self.STATE_RUNNING:
                ball_pos = self._process_frame(frame)

            self.hud_renderer.render_hud(
                frame,
                self.scoreboard,
                self.statistics,
                self.ball_tracker,
                self.state
            )

            if self._gui and hasattr(self._gui, 'frame_signal'):
                score_l = self.scoreboard.get_score(self.scoreboard.team_names[0])
                score_r = self.scoreboard.get_score(self.scoreboard.team_names[1])
                self._gui.frame_signal.update.emit(
                    frame,
                    ball_pos,
                    self.statistics,
                    score_l,
                    score_r
                )
            else:
                cv2.imshow(self.WINDOW_NAME, frame)
                key = cv2.waitKey(1) & 0xFF
                self.event_handler.handle_key_press(key)

            # Throttle loop to maintain the target FPS
            elapsed = time.monotonic() - t_start
            sleep_time = FRAME_INTERVAL - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        logger.info("Game loop has terminated cleanly.")

    def _process_frame(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        :param frame: The current video frame.
        :return: The (x, y) coordinates of the ball, if detected.
        """

        ball_pos = self.ball_tracker.update(frame)
        self.ball_tracker.draw(frame)

        self.statistics.record_speed(self.ball_tracker.speed_cm_s)
        if ball_pos is not None:
            self.statistics.trajectory_add(ball_pos)

        self.field.draw(frame)
        trajectory = self.statistics.get_trajectory_count()
        self.hud_renderer.draw_trajectory_gradient(frame, trajectory)

        scored_goals = self.field.check_goals(ball_pos)
        for goal_name in scored_goals:
            self.scoreboard.register_goal(goal_name, self.ball_tracker.speed_cm_s)
            self.event_handler.on_goal(goal_name, frame)

        for team in self.scoreboard.team_names:
            if self.scoreboard.get_score(team) >= self.goals_to_win:
                self.event_handler.on_game_over(team)
                self.state = self.STATE_FINISHED

        return ball_pos

    def _shutdown(self) -> None:
        """Releases all hardware resources and logs the final game summary."""
        try:
            summary = self.statistics.summary(self.scoreboard)
            logger.info("\n" + summary)
        except Exception as e:
            logger.debug(f"Could not print summary during shutdown: {e}")

        if self.camera:
            self.camera.stop()

        cv2.destroyAllWindows()

        if self._gui:
            score_l = self.scoreboard.get_score(self.scoreboard.team_names[0])
            score_r = self.scoreboard.get_score(self.scoreboard.team_names[1])
            self._gui.show_summary(self.statistics, score_l, score_r)

        logger.info("System shutdown complete.")