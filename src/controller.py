"""Controller — orchestrates the game loop between VideoSource, detectors, Statistics, and GUI."""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING, Optional

from src.ball_detector import BallDetector
from src.game_events import EventType, GameConfig, GameEvent
from src.goal_detector import GoalDetector
from src.statistics import Statistics
from src.video_source import VideoSource

if TYPE_CHECKING:
    from src.gui import KickerGUI

logger = logging.getLogger(__name__)

TARGET_FPS = 30.0
FRAME_INTERVAL = 1.0 / TARGET_FPS


class Controller:
    """Wires all backend components together and drives the game loop on a background thread."""

    def __init__(self, gui: "KickerGUI") -> None:
        self._gui = gui
        self._video: Optional[VideoSource] = None
        self._detector: Optional[BallDetector] = None
        self._goal_detector: Optional[GoalDetector] = None
        self._stats: Optional[Statistics] = None
        self._config: Optional[GameConfig] = None
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ── Public API ────────────────────────────────────────────────────────────

    def start_game(self, config: GameConfig) -> None:
        """Initialise all components and start the game loop."""
        self._config = config
        self._stop_event.clear()

        self._video = VideoSource(
            camera_index=config.camera_index,
            width=config.field_x2 - config.field_x1 or 640,
            height=config.field_y2 - config.field_y1 or 480,
            fps=config.fps,
        )
        opened = self._video.open()
        if not opened:
            logger.warning("Camera index %d not available — running without feed.", config.camera_index)

        # Update config with actual frame dimensions if camera opened
        if opened:
            config.field_x2 = self._video.frame_width
            config.field_y2 = self._video.frame_height

        self._detector = BallDetector()
        self._goal_detector = GoalDetector(
            field_x1=config.field_x1,
            field_x2=config.field_x2 if config.field_x2 > 0 else 640,
        )
        self._stats = Statistics(config)
        self._stats.start_timer()

        self._gui.show_dashboard()

        self._thread = threading.Thread(target=self._game_loop, daemon=True, name="game-loop")
        self._thread.start()
        logger.info("Game started. Config: %s", config)

    def end_game(self) -> None:
        """Stop the loop and show the summary screen."""
        self._stop_event.set()
        if self._stats:
            self._stats.stop_timer()
        score_left = self._goal_detector.score_left if self._goal_detector else 0
        score_right = self._goal_detector.score_right if self._goal_detector else 0
        if self._video:
            self._video.release()
        self._gui.show_summary(self._stats, self._config, score_left, score_right)
        logger.info("Game ended. Score: %d:%d", score_left, score_right)

    def new_game(self) -> None:
        """Reset everything and return to the start screen."""
        self._stop_event.set()
        if self._video:
            self._video.release()
            self._video = None
        self._stats = None
        self._goal_detector = None
        self._detector = None
        self._config = None
        self._gui.show_start_screen()
        logger.info("New game requested — back to start screen.")

    def quit(self) -> None:
        """Clean shutdown."""
        self._stop_event.set()
        if self._video:
            self._video.release()
        logger.info("Application quit.")

    # ── Background thread ─────────────────────────────────────────────────────

    def _game_loop(self) -> None:
        signal = self._gui.frame_signal  # FrameUpdateSignal

        while not self._stop_event.is_set():
            t_start = time.monotonic()

            frame = None
            position = None

            if self._video and self._video.is_opened():
                ok, frame = self._video.read()
                if not ok:
                    frame = None

            if frame is not None and self._detector:
                position = self._detector.detect(frame)

            goal_event: Optional[GameEvent] = None
            if self._goal_detector:
                goal_event = self._goal_detector.update(position)

            if self._stats:
                self._stats.update(position, goal_event)

            score_left = self._goal_detector.score_left if self._goal_detector else 0
            score_right = self._goal_detector.score_right if self._goal_detector else 0

            # Emit to main thread via queued signal connection
            signal.update.emit(frame, position, self._stats, score_left, score_right)

            # Throttle to TARGET_FPS
            elapsed = time.monotonic() - t_start
            sleep_time = FRAME_INTERVAL - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
