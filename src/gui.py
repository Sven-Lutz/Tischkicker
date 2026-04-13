"""Kicker GT3 — PyQt6 main window with three screens.

Screen flow:
  StartScreen  →  DashboardScreen  →  SummaryScreen
                                           │
                                           └── "Neues Spiel" → StartScreen

Design tokens (dark cockpit, Porsche-red accent):
  BG      = #0a0a0a
  SURFACE = #111111
  ACCENT  = #C41E3A
  BORDER  = #222222
  TEXT    = #ffffff
  DIM     = #777777
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

import cv2
import numpy as np
from PyQt6.QtCore import QObject, Qt, pyqtSignal
from PyQt6.QtGui import QFont, QImage, QPixmap
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from src.game_events import EventType, GameConfig, GameMode, Team

logger = logging.getLogger(__name__)

# ── Design tokens ──────────────────────────────────────────────────────────────
BG = "#0a0a0a"
SURFACE = "#111111"
SURFACE2 = "#181818"
ACCENT = "#C41E3A"
BORDER = "#222222"
TEXT = "#ffffff"
DIM = "#777777"

_BASE_STYLE = f"""
    QWidget {{ background-color: {BG}; color: {TEXT}; font-family: Arial, sans-serif; }}
    QLabel  {{ color: {TEXT}; }}
    QLineEdit {{
        background-color: {SURFACE};
        color: {TEXT};
        border: 1px solid {BORDER};
        padding: 6px 10px;
        font-size: 14px;
    }}
    QScrollArea, QScrollArea > QWidget > QWidget {{ background-color: {SURFACE}; border: none; }}
    QScrollBar:vertical {{
        background: {SURFACE};
        width: 6px;
    }}
    QScrollBar::handle:vertical {{
        background: {BORDER};
        border-radius: 3px;
    }}
"""


def _btn(
    text: str,
    font_size: int = 13,
    accent: bool = False,
    flat: bool = False,
) -> QPushButton:
    b = QPushButton(text)
    bg = ACCENT if accent else SURFACE
    hover_bg = "#e02442" if accent else SURFACE2
    border = "none" if accent or flat else f"1px solid {BORDER}"
    b.setStyleSheet(f"""
        QPushButton {{
            background-color: {bg};
            color: {TEXT};
            border: {border};
            padding: 10px 20px;
            font-size: {font_size}px;
            font-weight: {'bold' if accent else 'normal'};
        }}
        QPushButton:hover {{
            background-color: {hover_bg};
            {'color: ' + ACCENT + ';' if not accent else ''}
        }}
        QPushButton:pressed {{
            background-color: {ACCENT};
            color: {TEXT};
        }}
    """)
    return b


def _label(text: str, size: int = 13, bold: bool = False, color: str = TEXT, align_center: bool = False) -> QLabel:
    lbl = QLabel(text)
    weight = "bold" if bold else "normal"
    lbl.setStyleSheet(f"font-size: {size}px; font-weight: {weight}; color: {color};")
    if align_center:
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
    return lbl


def _hsep() -> QFrame:
    f = QFrame()
    f.setFrameShape(QFrame.Shape.HLine)
    f.setStyleSheet(f"background: {BORDER}; border: none; max-height: 1px;")
    return f


# ── Frame update signal ────────────────────────────────────────────────────────

class FrameUpdateSignal(QObject):
    """Carries frame data from the background game loop to the main thread."""

    # (frame | None, BallPosition | None, Statistics | None, score_left, score_right)
    update = pyqtSignal(object, object, object, int, int)


# ── Stat tile widget ───────────────────────────────────────────────────────────

class _StatTile(QWidget):
    def __init__(self, label: str) -> None:
        super().__init__()
        self.setStyleSheet(f"background: {SURFACE}; border: 1px solid {BORDER};")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(4)
        self._value = _label("—", size=22, bold=True, align_center=True)
        self._caption = _label(label, size=10, color=DIM, align_center=True)
        layout.addWidget(self._value)
        layout.addWidget(self._caption)

    def set_value(self, text: str) -> None:
        self._value.setText(text)


# ── Score bar ─────────────────────────────────────────────────────────────────

class _ScoreBar(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setStyleSheet(f"background: {SURFACE}; border: 1px solid {BORDER};")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(20, 12, 20, 12)

        self._left_name = _label("Links", size=15, bold=True, color=ACCENT)
        self._left_score = _label("0", size=40, bold=True, align_center=True)
        self._time = _label("00:00", size=20, color=DIM, align_center=True)
        self._right_score = _label("0", size=40, bold=True, align_center=True)
        self._right_name = _label("Rechts", size=15, bold=True, color=DIM)
        self._right_name.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        layout.addWidget(self._left_name, stretch=2)
        layout.addWidget(self._left_score, stretch=1)
        layout.addWidget(self._time, stretch=2)
        layout.addWidget(self._right_score, stretch=1)
        layout.addWidget(self._right_name, stretch=2)

    def update(
        self,
        score_left: int,
        score_right: int,
        elapsed_seconds: float,
        left_name: str = "Links",
        right_name: str = "Rechts",
    ) -> None:
        mins = int(elapsed_seconds // 60)
        secs = int(elapsed_seconds % 60)
        self._left_score.setText(str(score_left))
        self._right_score.setText(str(score_right))
        self._time.setText(f"{mins:02d}:{secs:02d}")
        self._left_name.setText(left_name)
        self._right_name.setText(right_name)


# ── Events list ───────────────────────────────────────────────────────────────

class _EventsWidget(QWidget):
    def __init__(self) -> None:
        super().__init__()
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        container = QWidget()
        container.setStyleSheet(f"background: {SURFACE};")
        self._inner = QVBoxLayout(container)
        self._inner.setContentsMargins(8, 8, 8, 8)
        self._inner.setSpacing(4)
        self._inner.addStretch()

        scroll.setWidget(container)
        outer.addWidget(scroll)
        self._scroll = scroll
        self._shown_ids: set[int] = set()

    def add_event_if_new(self, event, game_seconds: float) -> None:
        eid = id(event)
        if eid in self._shown_ids:
            return
        self._shown_ids.add(eid)

        mins = int(game_seconds // 60)
        secs = int(game_seconds % 60)
        time_str = f"{mins:02d}:{secs:02d}"

        if event.event_type == EventType.GOAL:
            side = "Links" if event.team == Team.LEFT else "Rechts"
            desc = f"Tor {side}  {event.score_left}:{event.score_right}"
            color = ACCENT
        elif event.event_type == EventType.MAX_SPEED:
            desc = f"Höchstg.  {event.value:.1f} m/s" if event.value else "Höchstg."
            color = "#f0a020"
        else:
            desc = event.description or event.event_type.value
            color = DIM

        row = QHBoxLayout()
        row.setSpacing(8)
        t_lbl = _label(time_str, size=10, color=DIM)
        t_lbl.setFixedWidth(36)
        d_lbl = _label(desc, size=11, color=color)
        row.addWidget(t_lbl)
        row.addWidget(d_lbl)
        row.addStretch()

        w = QWidget()
        w.setLayout(row)
        w.setStyleSheet(f"background: {SURFACE2}; border-bottom: 1px solid {BORDER};")

        # Insert before the trailing stretch
        count = self._inner.count()
        self._inner.insertWidget(count - 1, w)

        # Scroll to bottom
        self._scroll.verticalScrollBar().setValue(
            self._scroll.verticalScrollBar().maximum()
        )

    def clear(self) -> None:
        self._shown_ids.clear()
        while self._inner.count() > 1:
            item = self._inner.takeAt(0)
            if item.widget():
                item.widget().deleteLater()


# ── Screen 1: Start ───────────────────────────────────────────────────────────

class _StartScreen(QWidget):
    def __init__(self, on_start: Callable[[GameConfig], None]) -> None:
        super().__init__()
        self._on_start = on_start
        self._mode = GameMode.ONE_VS_ONE
        self._build()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(80, 50, 80, 50)
        root.setSpacing(0)

        # Header
        title = _label("KICKER GT3", size=52, bold=True, align_center=True)
        sub = _label("EDV4 · 2026", size=13, color=DIM, align_center=True)
        root.addWidget(title)
        root.addWidget(sub)
        root.addSpacing(30)
        root.addWidget(_hsep())
        root.addSpacing(24)

        # Mode buttons
        mode_row = QHBoxLayout()
        mode_row.setSpacing(16)
        self._btn_1v1 = _btn("1 gegen 1", font_size=14, accent=True)
        self._btn_2v2 = _btn("2 gegen 2", font_size=14)
        self._btn_1v1.setFixedWidth(160)
        self._btn_2v2.setFixedWidth(160)
        self._btn_1v1.clicked.connect(lambda: self._set_mode(GameMode.ONE_VS_ONE))
        self._btn_2v2.clicked.connect(lambda: self._set_mode(GameMode.TWO_VS_TWO))
        mode_row.addStretch()
        mode_row.addWidget(self._btn_1v1)
        mode_row.addWidget(self._btn_2v2)
        mode_row.addStretch()
        root.addLayout(mode_row)
        root.addSpacing(32)

        # Team name inputs
        teams = QHBoxLayout()
        teams.setSpacing(60)

        left_col = QVBoxLayout()
        left_col.setSpacing(8)
        left_col.addWidget(_label("TEAM LINKS", size=11, bold=True, color=ACCENT, align_center=True))
        self._l1 = QLineEdit()
        self._l1.setPlaceholderText("Spieler 1")
        self._l1.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._l2 = QLineEdit()
        self._l2.setPlaceholderText("Spieler 2")
        self._l2.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._l2.setVisible(False)
        left_col.addWidget(self._l1)
        left_col.addWidget(self._l2)

        vs = _label("vs", size=28, color=DIM, align_center=True)
        vs.setFixedWidth(60)

        right_col = QVBoxLayout()
        right_col.setSpacing(8)
        right_col.addWidget(_label("TEAM RECHTS", size=11, bold=True, color=DIM, align_center=True))
        self._r1 = QLineEdit()
        self._r1.setPlaceholderText("Spieler 1")
        self._r1.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._r2 = QLineEdit()
        self._r2.setPlaceholderText("Spieler 2")
        self._r2.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._r2.setVisible(False)
        right_col.addWidget(self._r1)
        right_col.addWidget(self._r2)

        teams.addStretch()
        teams.addLayout(left_col)
        teams.addWidget(vs)
        teams.addLayout(right_col)
        teams.addStretch()
        root.addLayout(teams)
        root.addStretch()

        # Camera hint
        root.addWidget(_label("Kamera: Index 0  ·  Namen sind optional", size=11, color=DIM, align_center=True))
        root.addSpacing(20)

        # Start button
        start = _btn("SPIEL STARTEN", font_size=18, accent=True)
        start.setFixedWidth(280)
        start.setFixedHeight(52)
        start.clicked.connect(self._fire_start)
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_row.addWidget(start)
        btn_row.addStretch()
        root.addLayout(btn_row)

    def _set_mode(self, mode: GameMode) -> None:
        self._mode = mode
        if mode == GameMode.ONE_VS_ONE:
            self._btn_1v1.setStyleSheet(self._btn_1v1.styleSheet().replace(SURFACE, ACCENT))
            self._btn_2v2.setStyleSheet(self._btn_2v2.styleSheet().replace(ACCENT, SURFACE))
            self._l2.setVisible(False)
            self._r2.setVisible(False)
        else:
            self._btn_2v2.setStyleSheet(self._btn_2v2.styleSheet().replace(SURFACE, ACCENT))
            self._btn_1v1.setStyleSheet(self._btn_1v1.styleSheet().replace(ACCENT, SURFACE))
            self._l2.setVisible(True)
            self._r2.setVisible(True)

    def _fire_start(self) -> None:
        left_names = [self._l1.text().strip() or "Links"]
        right_names = [self._r1.text().strip() or "Rechts"]
        if self._mode == GameMode.TWO_VS_TWO:
            left_names.append(self._l2.text().strip() or "Links 2")
            right_names.append(self._r2.text().strip() or "Rechts 2")
        config = GameConfig(
            mode=self._mode,
            team_left_names=left_names,
            team_right_names=right_names,
        )
        self._on_start(config)


# ── Screen 2: Live Dashboard ──────────────────────────────────────────────────

class _DashboardScreen(QWidget):
    def __init__(self, on_end_game: Callable[[], None]) -> None:
        super().__init__()
        self._on_end_game = on_end_game
        self._config: Optional[GameConfig] = None
        self._events_shown = 0
        self._build()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 14, 16, 14)
        root.setSpacing(10)

        # Header row
        hdr = QHBoxLayout()
        hdr.addWidget(_label("KICKER GT3", size=18, bold=True))
        hdr.addStretch()
        live = _label("● LIVE", size=13, color=ACCENT)
        hdr.addWidget(live)
        root.addLayout(hdr)

        # Score bar
        self._score_bar = _ScoreBar()
        root.addWidget(self._score_bar)

        # Stats tiles
        stats_row = QHBoxLayout()
        stats_row.setSpacing(8)
        self._tile_max_spd = _StatTile("Max Geschw.")
        self._tile_cur_spd = _StatTile("Akt. Geschw.")
        self._tile_rebounds = _StatTile("Abpralle")
        self._tile_shots = _StatTile("Schüsse")
        for tile in (self._tile_max_spd, self._tile_cur_spd, self._tile_rebounds, self._tile_shots):
            stats_row.addWidget(tile)
        root.addLayout(stats_row)

        # Main content: video + side panel
        content = QHBoxLayout()
        content.setSpacing(10)

        # Video display
        self._video_lbl = QLabel()
        self._video_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._video_lbl.setStyleSheet(f"background: #000; border: 1px solid {BORDER};")
        self._video_lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._video_lbl.setMinimumSize(480, 270)
        content.addWidget(self._video_lbl, stretch=3)

        # Side panel
        side = QVBoxLayout()
        side.setSpacing(8)
        side.addWidget(_label("EREIGNISSE", size=10, bold=True, color=DIM))
        self._events = _EventsWidget()
        side.addWidget(self._events, stretch=1)

        end_btn = _btn("SPIEL BEENDEN", font_size=13, accent=True)
        end_btn.setFixedHeight(44)
        end_btn.clicked.connect(self._on_end_game)
        side.addWidget(end_btn)

        side_w = QWidget()
        side_w.setLayout(side)
        side_w.setMaximumWidth(260)
        content.addWidget(side_w, stretch=1)
        root.addLayout(content, stretch=1)

    # ── Slot (called from GUI main thread via queued signal) ──────────────────

    def update_frame(
        self,
        frame,
        position,
        stats,
        score_left: int,
        score_right: int,
    ) -> None:
        # Video frame with overlays
        if frame is not None:
            self._render_video(frame, position, stats)

        # Stats
        if stats is not None:
            elapsed = stats.game_time_seconds
            left_name = " & ".join(self._config.team_left_names) if self._config else "Links"
            right_name = " & ".join(self._config.team_right_names) if self._config else "Rechts"
            self._score_bar.update(score_left, score_right, elapsed, left_name, right_name)
            self._tile_max_spd.set_value(f"{stats.max_speed_ms:.1f} m/s")
            self._tile_cur_spd.set_value(f"{stats.current_speed_ms:.1f} m/s")
            self._tile_rebounds.set_value(str(stats.rebound_count))
            self._tile_shots.set_value(str(stats.shot_count))

            # New events
            events = stats.events
            for i in range(self._events_shown, len(events)):
                self._events.add_event_if_new(events[i], elapsed)
            self._events_shown = len(events)

    def reset(self, config: Optional[GameConfig] = None) -> None:
        self._config = config
        self._events_shown = 0
        self._events.clear()
        self._score_bar.update(0, 0, 0.0)
        for tile in (self._tile_max_spd, self._tile_cur_spd, self._tile_rebounds, self._tile_shots):
            tile.set_value("—")
        self._video_lbl.clear()
        self._video_lbl.setText("Kamera wird geöffnet …")
        self._video_lbl.setStyleSheet(
            f"background: #000; border: 1px solid {BORDER}; color: {DIM}; font-size: 13px;"
        )

    # ── Rendering ─────────────────────────────────────────────────────────────

    def _render_video(self, frame: np.ndarray, position, stats) -> None:
        display = frame.copy()

        # Trajectory
        if stats is not None:
            trajectory = stats.trajectory
            n = len(trajectory)
            for i in range(1, n):
                p1 = (int(trajectory[i - 1].x), int(trajectory[i - 1].y))
                p2 = (int(trajectory[i].x), int(trajectory[i].y))
                alpha = i / n
                r = int(255 * alpha)
                g = int(80 * alpha)
                cv2.line(display, p1, p2, (r, g, 0), 1, cv2.LINE_AA)

        # Ball circle
        if position is not None:
            cx, cy = int(position.x), int(position.y)
            cv2.circle(display, (cx, cy), 14, (255, 107, 0), 2, cv2.LINE_AA)
            cv2.circle(display, (cx, cy), 3, (255, 107, 0), -1, cv2.LINE_AA)

        self._set_pixmap(display)

    def _set_pixmap(self, bgr: np.ndarray) -> None:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        img = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(img)
        label_sz = self._video_lbl.size()
        if label_sz.width() > 0 and label_sz.height() > 0:
            pixmap = pixmap.scaled(
                label_sz,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.FastTransformation,
            )
        self._video_lbl.setPixmap(pixmap)


# ── Screen 3: Summary ─────────────────────────────────────────────────────────

class _SummaryScreen(QWidget):
    def __init__(
        self,
        on_new_game: Callable[[], None],
        on_quit: Callable[[], None],
    ) -> None:
        super().__init__()
        self._on_new_game = on_new_game
        self._on_quit = on_quit
        self._build()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(40, 30, 40, 30)
        root.setSpacing(12)

        # Header
        hdr = QHBoxLayout()
        hdr.addWidget(_label("KICKER GT3", size=18, bold=True))
        hdr.addStretch()
        hdr.addWidget(_label("SPIELZUSAMMENFASSUNG", size=13, color=DIM))
        root.addLayout(hdr)

        # Final score card
        self._score_card = _ScoreBar()
        root.addWidget(self._score_card)

        # Stat tiles
        tiles_row = QHBoxLayout()
        tiles_row.setSpacing(8)
        self._t_maxspd = _StatTile("Max Geschw.")
        self._t_shots = _StatTile("Schüsse")
        self._t_rebounds = _StatTile("Abpralle")
        self._t_time = _StatTile("Spielzeit")
        for t in (self._t_maxspd, self._t_shots, self._t_rebounds, self._t_time):
            tiles_row.addWidget(t)
        root.addLayout(tiles_row)

        # Events log + action buttons
        content = QHBoxLayout()
        content.setSpacing(12)

        log_col = QVBoxLayout()
        log_col.addWidget(_label("TORCHRONIK", size=10, bold=True, color=DIM))
        self._history = _EventsWidget()
        log_col.addWidget(self._history, stretch=1)
        content.addLayout(log_col, stretch=2)

        btn_col = QVBoxLayout()
        btn_col.addStretch()
        new_btn = _btn("NEUES SPIEL", font_size=15, accent=True)
        new_btn.setFixedWidth(200)
        new_btn.setFixedHeight(48)
        new_btn.clicked.connect(self._on_new_game)
        quit_btn = _btn("BEENDEN", font_size=13)
        quit_btn.setFixedWidth(200)
        quit_btn.clicked.connect(self._on_quit)
        btn_col.addWidget(new_btn)
        btn_col.addSpacing(8)
        btn_col.addWidget(quit_btn)
        btn_col.addStretch()
        content.addLayout(btn_col, stretch=1)

        root.addLayout(content, stretch=1)

    def populate(self, stats, config, score_left: int, score_right: int) -> None:
        """Fill the screen with final game data."""
        from src.statistics import Statistics
        from src.game_events import GameConfig

        self._history.clear()

        left_name = " & ".join(config.team_left_names) if config else "Links"
        right_name = " & ".join(config.team_right_names) if config else "Rechts"

        elapsed = stats.game_time_seconds if stats else 0.0
        self._score_card.update(score_left, score_right, elapsed, left_name, right_name)

        if stats:
            mins = int(elapsed // 60)
            secs = int(elapsed % 60)
            self._t_maxspd.set_value(f"{stats.max_speed_ms:.1f} m/s")
            self._t_shots.set_value(str(stats.shot_count))
            self._t_rebounds.set_value(str(stats.rebound_count))
            self._t_time.set_value(f"{mins:02d}:{secs:02d}")

            for evt in stats.events:
                self._history.add_event_if_new(evt, elapsed)
        else:
            for tile in (self._t_maxspd, self._t_shots, self._t_rebounds, self._t_time):
                tile.set_value("—")


# ── Main window ────────────────────────────────────────────────────────────────

class KickerGUI(QMainWindow):
    """Top-level PyQt6 window. Owned by main.py.

    The controller communicates with the GUI exclusively through:
    * ``frame_signal.update`` — emitted from the game-loop thread
    * ``show_dashboard()``, ``show_summary()``, ``show_start_screen()`` — called
      from the controller in the Qt main thread (safe, because they happen before
      or after the game loop, never concurrently).
    """

    def __init__(
        self,
        on_start: Callable[[GameConfig], None],
        on_end_game: Callable[[], None],
        on_new_game: Callable[[], None],
        on_quit: Callable[[], None],
    ) -> None:
        super().__init__()
        self._on_start_cb = on_start
        self._on_end_game_cb = on_end_game
        self._on_new_game_cb = on_new_game
        self._on_quit_cb = on_quit

        # Signal used by the background game loop to push frame data to the GUI.
        self.frame_signal = FrameUpdateSignal()

        self._build_ui()
        self.setStyleSheet(_BASE_STYLE)

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        self.setWindowTitle("Kicker GT3")
        self.setMinimumSize(1100, 700)

        self._stack = QStackedWidget()
        self.setCentralWidget(self._stack)

        self._start = _StartScreen(on_start=self._handle_start)
        self._dashboard = _DashboardScreen(on_end_game=self._on_end_game_cb)
        self._summary = _SummaryScreen(
            on_new_game=self._on_new_game_cb,
            on_quit=self._on_quit_cb,
        )

        self._stack.addWidget(self._start)      # index 0
        self._stack.addWidget(self._dashboard)  # index 1
        self._stack.addWidget(self._summary)    # index 2

        # Cross-thread frame updates → dashboard slot (queued connection is
        # automatic because sender and receiver live in different threads)
        self.frame_signal.update.connect(self._dashboard.update_frame)

    # ── Screen transitions ────────────────────────────────────────────────────

    def show_start_screen(self) -> None:
        self._stack.setCurrentIndex(0)

    def show_dashboard(self, config: Optional[GameConfig] = None) -> None:
        self._dashboard.reset(config)
        self._stack.setCurrentIndex(1)

    def show_summary(
        self,
        stats,
        config,
        score_left: int,
        score_right: int,
    ) -> None:
        self._summary.populate(stats, config, score_left, score_right)
        self._stack.setCurrentIndex(2)

    # ── Internal callbacks ────────────────────────────────────────────────────

    def _handle_start(self, config: GameConfig) -> None:
        # Pass config to dashboard so it can show team names, then call controller
        self._dashboard.reset(config)
        self._on_start_cb(config)
