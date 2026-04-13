"""Controller — orchestrates the game loop and auto-calibration on startup.

On start_game(), the camera is opened, CALIBRATION_FRAMES are sampled for
field detection, and all detectors are configured before the loop starts.
Features a Bird's-Eye View perspective warp and a comet-tail ball trajectory.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import TYPE_CHECKING, Optional
import math

import cv2
import numpy as np

from src.ball_detector import BallDetector
from src.field_detector import FieldBounds, FieldDetector
from src.game_events import GameConfig, GameEvent
from src.goal_detector import GoalDetector
from src.statistics import Statistics
from src.video_source import VideoSource

if TYPE_CHECKING:
    from src.gui import KickerGUI

logger = logging.getLogger(__name__)

TARGET_FPS = 30.0
FRAME_INTERVAL = 1.0 / TARGET_FPS
CALIBRATION_FRAMES = 20

# Target dimensions for the Bird's-Eye View (10 pixels = 1 cm)
# Standard foosball table size: 120 cm x 68 cm
WARPED_WIDTH = 1200
WARPED_HEIGHT = 680


class Controller:
    """Wires all backend components together and drives the game loop."""

    def __init__(self, gui: "KickerGUI") -> None:
        self._gui = gui
        self._video: Optional[VideoSource] = None
        self._detector: Optional[BallDetector] = None
        self._goal_detector: Optional[GoalDetector] = None
        self._stats: Optional[Statistics] = None
        self._config: Optional[GameConfig] = None
        self._ball_history: deque = deque(maxlen=25)

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._field_detector = FieldDetector()

        # Perspective transformation matrix for Bird's-Eye View
        self._perspective_matrix: Optional[np.ndarray] = None

        # History queue for drawing the comet tail (max 25 frames)
        self._ball_history: deque[tuple[int, int]] = deque(maxlen=25)

    def start_game(self, config: GameConfig) -> None:
        """Initialise all components, auto-calibrate, and start the game loop."""
        self._config = config
        self._stop_event.clear()
        self._ball_history.clear()

        field_w = config.field_x2 - config.field_x1
        field_h = config.field_y2 - config.field_y1
        req_w = field_w if field_w > 0 else 640
        req_h = field_h if field_h > 0 else 480

        self._video = VideoSource(
            camera_index=config.camera_index,
            width=req_w,
            height=req_h,
            fps=config.fps,
        )

        camera_opened = self._video.open()
        if not camera_opened:
            logger.warning(
                "Camera %d not available — running without video feed.",
                config.camera_index,
            )

        if camera_opened:
            config.field_x1 = 0
            config.field_y1 = 0
            config.field_x2 = self._video.frame_width
            config.field_y2 = self._video.frame_height

        field_corners: Optional[list[tuple[int, int]]] = None
        if camera_opened:
            logger.info(
                "Auto-calibration: sampling %d frames for field detection.",
                CALIBRATION_FRAMES,
            )
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
                    "Field detected: x=%d–%d y=%d–%d",
                    bounds.x1, bounds.x2, bounds.y1, bounds.y2,
                )
            else:
                logger.warning(
                    "Field auto-detection failed — using full frame (%dx%d).",
                    config.field_x2,
                    config.field_y2,
                )

        self._detector = BallDetector()

        if camera_opened and field_corners is not None:
            bounds_obj = FieldBounds(
                corners=field_corners,
                x1=config.field_x1, y1=config.field_y1,
                x2=config.field_x2, y2=config.field_y2,
            )
            # Calculate matrix for Bird's-Eye View
            self._perspective_matrix = bounds_obj.get_perspective_matrix(WARPED_WIDTH, WARPED_HEIGHT)

            # Clear mask because the warped frame will only contain the field
            self._detector.clear_field_mask()

            # Set constant real-world dimensions for subsequent logic
            config.field_x1, config.field_y1 = 0, 0
            config.field_x2, config.field_y2 = WARPED_WIDTH, WARPED_HEIGHT
            config.pixels_per_meter = 1000.0

        self._goal_detector = GoalDetector(
            field_x1=config.field_x1,
            field_x2=config.field_x2,
        )

        # Apply strict goals bounds matching the warped dimensions
        if self._perspective_matrix is not None:
             self._goal_detector.update_field_bounds(
                0, WARPED_WIDTH, 0, WARPED_HEIGHT
            )
        elif field_corners is not None:
            self._goal_detector.configure_from_corners(field_corners)
        else:
            self._goal_detector.update_field_bounds(
                config.field_x1, config.field_x2, config.field_y1, config.field_y2
            )

        self._stats = Statistics(config)
        self._stats.start_timer()

        self._gui.show_dashboard()

        self._thread = threading.Thread(
            target=self._game_loop, daemon=True, name="game-loop"
        )
        self._thread.start()
        logger.info("Game started: %s", config)

    def end_game(self) -> None:
        """Stop the loop and show the summary screen."""
        self._stop_event.set()

        # Safely wait for the background thread to finish its last iteration
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)

        if self._stats:
            self._stats.stop_timer()

        score_left = self._goal_detector.score_left if self._goal_detector else 0
        score_right = self._goal_detector.score_right if self._goal_detector else 0

        if self._video:
            self._video.release()

        self._gui.show_summary(self._stats, self._config, score_left, score_right)
        logger.info("Game ended. Score: %d:%d", score_left, score_right)

    def new_game(self) -> None:
        """Stop the current game and return to the start screen."""
        self._stop_event.set()

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)

        if self._video:
            self._video.release()
            self._video = None

        self._stats = None
        self._goal_detector = None
        self._detector = None
        self._config = None
        self._perspective_matrix = None
        self._ball_history.clear()

        self._gui.show_start_screen()
        logger.info("New game — returning to start screen.")

    def quit(self) -> None:
        """Clean shutdown."""
        self._stop_event.set()

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)

        if self._video:
            self._video.release()

        logger.info("Application quit.")

    def _game_loop(self) -> None:
        signal = self._gui.frame_signal
        last_read_ok = True
        logger.debug("Game loop started.")

        while not self._stop_event.is_set():
            t_start = time.monotonic()
            frame = None
            position = None

            if self._video and self._video.is_opened():
                ok, raw_frame = self._video.read()

                if not ok:
                    if last_read_ok:
                        logger.warning("Frame read failed — camera may have disconnected.")
                        last_read_ok = False
                else:
                    last_read_ok = True
                    # Apply Bird's-Eye View perspective transformation
                    if self._perspective_matrix is not None:
                        frame = cv2.warpPerspective(
                            raw_frame,
                            self._perspective_matrix,
                            (WARPED_WIDTH, WARPED_HEIGHT)
                        )
                    else:
                        frame = raw_frame

            if frame is not None and self._detector:
                position = self._detector.detect(frame)

                # ==========================================
                # HISTORIE & KOMETENSCHWEIF
                # ==========================================
                if position:
                    self._ball_history.appendleft(position)  # Speichere das ganze Objekt!
                elif len(self._ball_history) > 0:
                    self._ball_history.pop()

                for i in range(1, len(self._ball_history)):
                    # Auf x und y zugreifen
                    pt1 = (int(self._ball_history[i - 1].x), int(self._ball_history[i - 1].y))
                    pt2 = (int(self._ball_history[i].x), int(self._ball_history[i].y))

                    thickness = int(np.sqrt(self._ball_history.maxlen / float(i + 1)) * 2.5)
                    fade_factor = 1.0 - (i / self._ball_history.maxlen)
                    color = (0, int(165 * fade_factor), int(255 * fade_factor))
                    cv2.line(frame, pt1, pt2, color, thickness)

                # ==========================================
                # GESCHWINDIGKEITSBERECHNUNG
                # ==========================================
                current_speed_kmh = 0.0

                # Wir brauchen mindestens 5 Frames für eine geglättete Berechnung
                if len(self._ball_history) >= 5:
                    pos_current = self._ball_history[0]
                    pos_old = self._ball_history[4]  # Position vor 4 Frames

                    # 1. Strecke berechnen (Pixel)
                    dist_px = math.hypot(pos_current.x - pos_old.x, pos_current.y - pos_old.y)

                    # Pixel in Meter umrechnen (1000 Pixel = 1 Meter)
                    dist_m = dist_px / self._config.pixels_per_meter

                    # 2. Zeitdifferenz berechnen (Sekunden)
                    time_diff = pos_current.timestamp - pos_old.timestamp

                    if time_diff > 0:
                        # v = s / t (in m/s)
                        speed_ms = dist_m / time_diff
                        # Umrechnung in km/h (* 3.6)
                        current_speed_kmh = speed_ms * 3.6

                    # Text neben den Ball zeichnen (Neon-Gelb)
                    if current_speed_kmh > 2.0:  # Nur anzeigen, wenn er wirklich rollt
                        text_pos = (int(pos_current.x) + 20, int(pos_current.y) - 20)
                        cv2.putText(
                            frame,
                            f"{current_speed_kmh:.1f} km/h",
                            text_pos,
                            cv2.FONT_HERSHEY_DUPLEX,
                            0.7,
                            (0, 255, 255),
                            2
                        )

            goal_event: Optional[GameEvent] = None
            if self._goal_detector:
                goal_event = self._goal_detector.update(position)

            if self._stats:
                self._stats.update(position, goal_event)

            score_left = self._goal_detector.score_left if self._goal_detector else 0
            score_right = self._goal_detector.score_right if self._goal_detector else 0

            try:
                signal.update.emit(frame, position, self._stats, score_left, score_right)
            except RuntimeError:
                logger.warning("GUI signal receiver deleted. Stopping game loop.")
                break

            elapsed = time.monotonic() - t_start
            sleep_time = FRAME_INTERVAL - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        logger.debug("Game loop stopped.")