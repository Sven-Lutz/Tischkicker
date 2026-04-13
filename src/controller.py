"""Controller — orchestrates the game loop between VideoSource, detectors, Statistics, and GUI.

Auto-calibration sequence (runs once before the game loop starts):
  1. Open the camera.
  2. Grab *CALIBRATION_FRAMES* frames and detect the green playing field.
  3. If found:   update GameConfig bounds, set field mask on BallDetector,
                 configure 2-D goal zones on GoalDetector.
  4. If not found: log a warning and proceed with full-frame defaults so the
                   game still runs without calibration.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING, Optional

from src.ball_detector import BallDetector
from src.field_detector import FieldDetector
from src.game_events import EventType, GameConfig, GameEvent
from src.goal_detector import GoalDetector
from src.statistics import Statistics
from src.video_source import VideoSource

if TYPE_CHECKING:
    from src.gui import KickerGUI

logger = logging.getLogger(__name__)

TARGET_FPS = 30.0
FRAME_INTERVAL = 1.0 / TARGET_FPS
CALIBRATION_FRAMES = 20  # frames sampled for field auto-detection


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
        self._field_detector = FieldDetector()

    # ── Public API ────────────────────────────────────────────────────────────

    def start_game(self, config: GameConfig) -> None:
        """Initialise all components, auto-calibrate, and start the game loop."""
        self._config = config
        self._stop_event.clear()

        # Open camera with sensible request resolution
        req_w = config.field_x2 - config.field_x1 if config.field_x2 > config.field_x1 else 640
        req_h = config.field_y2 - config.field_y1 if config.field_y2 > config.field_y1 else 480

        self._video = VideoSource(
            camera_index=config.camera_index,
            width=req_w,
            height=req_h,
            fps=config.fps,
        )
        camera_opened = self._video.open()
        if not camera_opened:
            logger.warning(
                "Camera %d not available — running without video feed.", config.camera_index
            )

        # Read back actual frame dimensions from the camera
        if camera_opened:
            config.field_x2 = self._video.frame_width
            config.field_y2 = self._video.frame_height
            config.field_x1 = 0
            config.field_y1 = 0

        # ── Auto field detection ───────────────────────────────────────────────
        field_corners: Optional[list[tuple[int, int]]] = None
        if camera_opened:
            logger.info("Auto-calibration: sampling %d frames for field detection …", CALIBRATION_FRAMES)
            bounds = self._field_detector.detect_from_frames(
                self._video, num_frames=CALIBRATION_FRAMES
            )
            if bounds is not None:
                config.field_x1 = bounds.x1
                config.field_y1 = bounds.y1
                config.field_x2 = bounds.x2
                config.field_y2 = bounds.y2
                field_corners = bounds.corners
                logger.info(
                    "Field detected: x=%d–%d  y=%d–%d",
                    bounds.x1, bounds.x2, bounds.y1, bounds.y2,
                )
            else:
                logger.warning(
                    "Field auto-detection failed — using full-frame defaults (%dx%d).",
                    config.field_x2,
                    config.field_y2,
                )

        # ── Build detectors ────────────────────────────────────────────────────
        self._detector = BallDetector()

        if camera_opened and field_corners is not None:
            # Create a static field mask once; the camera resolution is now known.
            from src.field_detector import FieldBounds
            bounds_obj = FieldBounds(
                corners=field_corners,
                x1=config.field_x1,
                y1=config.field_y1,
                x2=config.field_x2,
                y2=config.field_y2,
            )
            mask = bounds_obj.create_mask((config.field_y2, config.field_x2))
            self._detector.set_field_mask(mask)

        self._goal_detector = GoalDetector(
            field_x1=config.field_x1,
            field_x2=config.field_x2,
        )
        if field_corners is not None:
            self._goal_detector.configure_from_corners(field_corners)
        else:
            self._goal_detector.update_field_bounds(
                config.field_x1,
                config.field_x2,
                config.field_y1,
                config.field_y2,
            )

        self._stats = Statistics(config)
        self._stats.start_timer()

        self._gui.show_dashboard()

        self._thread = threading.Thread(
            target=self._game_loop, daemon=True, name="game-loop"
        )
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
        signal = self._gui.frame_signal

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
