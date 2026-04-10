"""Kicker GT3 — PyQt6 GUI. Porsche Design Studio aesthetic."""
from __future__ import annotations
import math
from typing import Callable, Optional
import cv2
import numpy as np
from PyQt6.QtCore import QObject, QRect, QSize, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QApplication, QFrame, QHBoxLayout, QLabel, QLineEdit,
    QListWidget, QListWidgetItem, QMainWindow, QProgressBar,
    QPushButton, QScrollArea, QSizePolicy, QStackedWidget,
    QVBoxLayout, QWidget,
)
from src.game_events import EventType, GameConfig, GameEvent, GameMode, Team, Zone
from src.statistics import Statistics

BG_DARK = "#0a0a0a"
BG_SURFACE = "#111111"
BG_ELEVATED = "#161616"
ACCENT = "#C41E3A"
BORDER = "#222222"
TEXT_PRIMARY = "#ffffff"
TEXT_SECONDARY = "#888888"

GLOBAL_QSS = f"""
QMainWindow, QWidget {{ background-color: {BG_DARK}; color: {TEXT_PRIMARY}; font-family: "Inter", "Helvetica Neue", Arial, sans-serif; }}
QLabel {{ background: transparent; color: {TEXT_PRIMARY}; }}
QLabel#score {{ font-size: 72px; font-weight: 900; letter-spacing: -2px; }}
QLabel#metric-value {{ font-size: 36px; font-weight: 700; }}
QLabel#metric-label {{ font-size: 10px; color: {TEXT_SECONDARY}; letter-spacing: 3px; }}
QLabel#section-header {{ font-size: 11px; color: {TEXT_SECONDARY}; letter-spacing: 4px; font-weight: 600; }}
QLabel#app-title {{ font-size: 16px; font-weight: 900; letter-spacing: 4px; }}
QLabel#app-subtitle {{ font-size: 10px; color: {TEXT_SECONDARY}; letter-spacing: 3px; }}
QLabel#winner {{ font-size: 13px; font-weight: 700; letter-spacing: 4px; color: {ACCENT}; }}
QLabel#team-name {{ font-size: 13px; font-weight: 600; letter-spacing: 2px; color: {TEXT_SECONDARY}; }}
QLabel#timer {{ font-size: 18px; font-weight: 700; letter-spacing: 2px; color: {TEXT_SECONDARY}; }}
QLabel#live-dot {{ font-size: 10px; color: {ACCENT}; }}
QFrame#divider {{ background-color: {BORDER}; min-height: 1px; max-height: 1px; border: none; }}
QFrame#surface {{ background-color: {BG_SURFACE}; }}
QPushButton#primary {{ background: transparent; border: 1px solid {ACCENT}; color: {TEXT_PRIMARY}; padding: 12px 32px; letter-spacing: 3px; font-size: 11px; font-weight: 600; }}
QPushButton#primary:hover {{ background-color: {ACCENT}; }}
QPushButton#secondary {{ background: transparent; border: 1px solid {BORDER}; color: {TEXT_SECONDARY}; padding: 12px 32px; letter-spacing: 3px; font-size: 11px; }}
QPushButton#secondary:hover {{ border-color: {TEXT_SECONDARY}; color: {TEXT_PRIMARY}; }}
QPushButton#mode-btn {{ background: transparent; border: 1px solid {BORDER}; color: {TEXT_SECONDARY}; padding: 14px 28px; letter-spacing: 3px; font-size: 11px; font-weight: 600; min-width: 110px; }}
QPushButton#mode-btn:checked {{ border-color: {ACCENT}; color: {TEXT_PRIMARY}; }}
QLineEdit#team-input {{ background: transparent; border: none; border-bottom: 1px solid {BORDER}; color: {TEXT_PRIMARY}; font-size: 16px; font-weight: 600; letter-spacing: 2px; padding: 8px 0; }}
QLineEdit#team-input:focus {{ border-bottom-color: {ACCENT}; }}
QListWidget#event-log {{ background: {BG_SURFACE}; border: none; color: {TEXT_SECONDARY}; font-size: 11px; letter-spacing: 1px; padding: 8px; }}
QListWidget#event-log::item {{ padding: 6px 0; border-bottom: 1px solid {BORDER}; }}
QListWidget#event-log::item:selected {{ background: transparent; color: {TEXT_PRIMARY}; }}
QProgressBar {{ background: {BG_ELEVATED}; border: none; max-height: 4px; text-align: center; color: transparent; }}
QProgressBar::chunk {{ background-color: {ACCENT}; }}
QScrollArea {{ background: transparent; border: none; }}
QScrollBar:vertical {{ background: {BG_DARK}; width: 4px; }}
QScrollBar::handle:vertical {{ background: {BORDER}; }}
"""


def _speed_to_bgr(speed: float, max_speed: float) -> tuple:
    """Blue (slow) -> White (medium) -> Red (fast). Returns BGR for OpenCV."""
    if max_speed <= 0:
        return (180, 180, 180)
    t = min(speed / max_speed, 1.0)
    if t < 0.5:
        s = t * 2.0
        b, g, r = 255, int(144 + 111 * s), int(30 + 225 * s)
    else:
        s = (t - 0.5) * 2.0
        b, g, r = int(255 - 197 * s), int(255 - 225 * s), int(255 - 59 * s)
    return (b, g, r)


def _fmt_time(secs: float) -> str:
    return f"{int(secs // 60):02d}:{int(secs % 60):02d}"


class _Divider(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("divider")
        self.setFrameShape(QFrame.Shape.HLine)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)


class _MetricCard(QWidget):
    def __init__(self, label: str, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(4)
        self._value_lbl = QLabel("—")
        self._value_lbl.setObjectName("metric-value")
        self._value_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label_lbl = QLabel(label.upper())
        self._label_lbl.setObjectName("metric-label")
        self._label_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._value_lbl)
        layout.addWidget(self._label_lbl)
        self.setObjectName("surface")
        self.setFrameShape = lambda *a: None  # suppress unused

    def set_value(self, text: str) -> None:
        self._value_lbl.setText(text)


class _ZoneBar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(24)
        self._pcts = {Zone.ATTACK_LEFT: 33.3, Zone.MIDDLE: 33.4, Zone.ATTACK_RIGHT: 33.3}

    def update_zones(self, pcts: dict) -> None:
        self._pcts = pcts
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        w, h = self.width(), self.height()
        segments = [
            (self._pcts.get(Zone.ATTACK_LEFT, 0) / 100, QColor(30, 100, 220)),
            (self._pcts.get(Zone.MIDDLE, 0) / 100,      QColor(80, 80, 80)),
            (self._pcts.get(Zone.ATTACK_RIGHT, 0) / 100, QColor(196, 30, 58)),
        ]
        x = 0
        for pct, color in segments:
            seg_w = int(w * pct)
            if seg_w < 1:
                continue
            painter.fillRect(x, 0, seg_w, h, color)
            if seg_w > 28:
                painter.setPen(QColor(255, 255, 255, 180))
                painter.setFont(QFont("Arial", 8))
                painter.drawText(QRect(x, 0, seg_w, h),
                                 Qt.AlignmentFlag.AlignCenter, f"{pct*100:.0f}%")
            x += seg_w
        painter.end()


class _ComparisonRow(QWidget):
    def __init__(self, label: str, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 4)
        layout.setSpacing(8)

        self._val_left = QLabel("0")
        self._val_left.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._val_left.setFixedWidth(50)
        self._val_left.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 13px; font-weight: 700;")

        self._bar_left = QProgressBar()
        self._bar_left.setRange(0, 100)
        self._bar_left.setValue(50)
        self._bar_left.setInvertedAppearance(True)
        self._bar_left.setTextVisible(False)

        lbl = QLabel(label.upper())
        lbl.setObjectName("metric-label")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setFixedWidth(90)

        self._bar_right = QProgressBar()
        self._bar_right.setRange(0, 100)
        self._bar_right.setValue(50)
        self._bar_right.setTextVisible(False)
        self._bar_right.setStyleSheet(f"QProgressBar::chunk {{ background-color: {TEXT_SECONDARY}; }}")

        self._val_right = QLabel("0")
        self._val_right.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self._val_right.setFixedWidth(50)
        self._val_right.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 13px; font-weight: 700;")

        layout.addWidget(self._val_left)
        layout.addWidget(self._bar_left)
        layout.addWidget(lbl)
        layout.addWidget(self._bar_right)
        layout.addWidget(self._val_right)

    def update_values(self, left, right, fmt="{:.0f}") -> None:
        total = left + right
        self._val_left.setText(fmt.format(left))
        self._val_right.setText(fmt.format(right))
        if total > 0:
            self._bar_left.setValue(int(left / total * 100))
            self._bar_right.setValue(int(right / total * 100))



class FieldLabel(QLabel):
    """Camera frame with heatmap overlay and speed-coded comet tail trajectory."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 300)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet(f"background: #111111;")
        self._show_placeholder()

    def _show_placeholder(self):
        self.setText("KAMERA NICHT VERBUNDEN")
        self.setStyleSheet(
            f"background: #111111; color: #333333; font-size: 11px; letter-spacing: 3px;"
        )

    def update_frame(self, frame, stats, config) -> None:
        if frame is None or stats is None:
            self._show_placeholder()
            return

        self.setStyleSheet("background: #111111;")
        self.setText("")
        display = frame.copy()
        trajectory = stats.trajectory
        max_speed = stats.max_speed_ms

        # 1. Heatmap overlay (40% opacity)
        hm = stats.heatmap
        if hm is not None and hm.max() > 0:
            hm_uint8 = (hm * 255).astype(np.uint8)
            if len(hm_uint8.shape) == 2:
                hm_color = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_HOT)
                if hm_color.shape[:2] == display.shape[:2]:
                    display = cv2.addWeighted(display, 0.6, hm_color, 0.4, 0)

        # 2. Speed-coded comet tail
        if len(trajectory) >= 2:
            n = len(trajectory)
            for i in range(1, n):
                p1, p2 = trajectory[i - 1], trajectory[i]
                dx, dy = p2.x - p1.x, p2.y - p1.y
                dist = (dx ** 2 + dy ** 2) ** 0.5
                cfg = config
                speed = (dist / cfg.pixels_per_meter) * cfg.fps
                base_color = _speed_to_bgr(speed, max_speed if max_speed > 0 else 1.0)
                alpha = i / n
                color = tuple(int(c * alpha) for c in base_color)
                thickness = max(1, int(3 * alpha))
                cv2.line(display,
                         (int(p1.x), int(p1.y)),
                         (int(p2.x), int(p2.y)),
                         color, thickness, cv2.LINE_AA)

        # 3. Ball marker (newest position)
        if trajectory:
            last = trajectory[-1]
            cx, cy = int(last.x), int(last.y)
            cv2.circle(display, (cx, cy), 9, (0, 107, 255), -1, cv2.LINE_AA)
            cv2.circle(display, (cx, cy), 9, (255, 255, 255), 1, cv2.LINE_AA)

        # 4. Convert BGR -> RGB -> QPixmap
        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        self.setPixmap(
            QPixmap.fromImage(qimg).scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )


class GoalTimelineWidget(QWidget):
    """Horizontal timeline with goal markers painted via QPainter."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(90)
        self._events: list = []
        self._total_seconds: float = 1.0

    def set_events(self, events: list, total_seconds: float) -> None:
        self._events = [e for e in events if e.event_type == EventType.GOAL]
        self._total_seconds = max(total_seconds, 1.0)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        mid_y = h // 2
        pad = 32

        # Background
        painter.fillRect(0, 0, w, h, QColor(17, 17, 17))

        # Timeline base line
        painter.setPen(QPen(QColor(34, 34, 34), 1))
        painter.drawLine(pad, mid_y, w - pad, mid_y)

        # Start / end caps
        painter.setPen(QPen(QColor(85, 85, 85), 1))
        painter.drawLine(pad, mid_y - 6, pad, mid_y + 6)
        painter.drawLine(w - pad, mid_y - 6, w - pad, mid_y + 6)

        # Goal markers
        for e in self._events:
            t = e.timestamp if self._total_seconds > 0 else 0
            # timestamp on event is absolute; use score to reconstruct relative time
            # We store relative time via description or compute from position
            rel = min(t / self._total_seconds, 1.0)
            x = int(pad + rel * (w - 2 * pad))
            is_left = e.team is not None and e.team.value == "Links"
            color = QColor(196, 30, 58) if is_left else QColor(136, 136, 136)

            painter.setPen(QPen(color, 2))
            if is_left:
                painter.drawLine(x, mid_y - 2, x, mid_y - 18)
                pts = [QRect(x - 4, mid_y - 22, 8, 6)]
                painter.setBrush(color)
                painter.drawEllipse(x - 4, mid_y - 24, 8, 8)
                painter.setBrush(Qt.BrushStyle.NoBrush)
            else:
                painter.drawLine(x, mid_y + 2, x, mid_y + 18)
                painter.setBrush(color)
                painter.drawEllipse(x - 4, mid_y + 18, 8, 8)
                painter.setBrush(Qt.BrushStyle.NoBrush)

            # Score label
            painter.setPen(color)
            painter.setFont(QFont("Arial", 7))
            score_str = f"{e.score_left}:{e.score_right}"
            y_lbl = mid_y - 36 if is_left else mid_y + 36
            painter.drawText(QRect(x - 16, y_lbl, 32, 12),
                             Qt.AlignmentFlag.AlignCenter, score_str)

        painter.end()



class StartScreen(QWidget):
    def __init__(self, on_start: Callable, parent=None):
        super().__init__(parent)
        self._on_start = on_start
        self._build()

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(48, 0, 48, 48)
        root.setSpacing(0)

        # ── Header ──
        header = QHBoxLayout()
        title = QLabel("KICKER GT3")
        title.setObjectName("app-title")
        sub = QLabel("EDV4 · 2026")
        sub.setObjectName("app-subtitle")
        header.addWidget(title)
        header.addStretch()
        header.addWidget(sub)
        root.addSpacing(32)
        root.addLayout(header)
        root.addSpacing(8)
        root.addWidget(_Divider())
        root.addSpacing(56)

        # ── Mode selector ──
        mode_lbl = QLabel("SPIELMODUS")
        mode_lbl.setObjectName("section-header")
        root.addWidget(mode_lbl)
        root.addSpacing(12)

        mode_row = QHBoxLayout()
        mode_row.setSpacing(0)
        self._btn_1v1 = QPushButton("1 GEGEN 1")
        self._btn_1v1.setObjectName("mode-btn")
        self._btn_1v1.setCheckable(True)
        self._btn_1v1.setChecked(True)
        self._btn_2v2 = QPushButton("2 GEGEN 2")
        self._btn_2v2.setObjectName("mode-btn")
        self._btn_2v2.setCheckable(True)
        self._btn_1v1.clicked.connect(lambda: self._select_mode(1))
        self._btn_2v2.clicked.connect(lambda: self._select_mode(2))
        mode_row.addWidget(self._btn_1v1)
        mode_row.addWidget(self._btn_2v2)
        mode_row.addStretch()
        root.addLayout(mode_row)
        root.addSpacing(48)

        # ── Team names ──
        teams_lbl = QLabel("TEAMS")
        teams_lbl.setObjectName("section-header")
        root.addWidget(teams_lbl)
        root.addSpacing(12)

        teams_row = QHBoxLayout()
        teams_row.setSpacing(32)

        left_col = QVBoxLayout()
        self._input_left = QLineEdit()
        self._input_left.setObjectName("team-input")
        self._input_left.setPlaceholderText("LINKS")
        self._input_p1_left = QLineEdit()
        self._input_p1_left.setObjectName("team-input")
        self._input_p1_left.setPlaceholderText("SPIELER 1")
        self._input_p2_left = QLineEdit()
        self._input_p2_left.setObjectName("team-input")
        self._input_p2_left.setPlaceholderText("SPIELER 2")
        left_col.addWidget(self._input_left)
        left_col.addWidget(self._input_p1_left)
        left_col.addWidget(self._input_p2_left)
        self._input_p2_left.setVisible(False)

        vs_lbl = QLabel("VS")
        vs_lbl.setObjectName("section-header")
        vs_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        vs_lbl.setFixedWidth(48)

        right_col = QVBoxLayout()
        self._input_right = QLineEdit()
        self._input_right.setObjectName("team-input")
        self._input_right.setPlaceholderText("RECHTS")
        self._input_p1_right = QLineEdit()
        self._input_p1_right.setObjectName("team-input")
        self._input_p1_right.setPlaceholderText("SPIELER 1")
        self._input_p2_right = QLineEdit()
        self._input_p2_right.setObjectName("team-input")
        self._input_p2_right.setPlaceholderText("SPIELER 2")
        right_col.addWidget(self._input_right)
        right_col.addWidget(self._input_p1_right)
        right_col.addWidget(self._input_p2_right)
        self._input_p2_right.setVisible(False)

        teams_row.addLayout(left_col)
        teams_row.addWidget(vs_lbl)
        teams_row.addLayout(right_col)
        root.addLayout(teams_row)
        root.addStretch()

        # ── Start button ──
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        start_btn = QPushButton("SPIEL STARTEN")
        start_btn.setObjectName("primary")
        start_btn.clicked.connect(self._on_start_clicked)
        btn_row.addWidget(start_btn)
        btn_row.addStretch()
        root.addLayout(btn_row)

    def _select_mode(self, mode: int):
        self._btn_1v1.setChecked(mode == 1)
        self._btn_2v2.setChecked(mode == 2)
        self._input_p2_left.setVisible(mode == 2)
        self._input_p2_right.setVisible(mode == 2)

    def _on_start_clicked(self):
        config = self.get_config()
        self._on_start(config)

    def get_config(self) -> GameConfig:
        mode = GameMode.TWO_VS_TWO if self._btn_2v2.isChecked() else GameMode.ONE_VS_ONE
        left_names = [n for n in [
            self._input_left.text().strip() or "Links",
            self._input_p1_left.text().strip() or "Spieler 1",
            self._input_p2_left.text().strip() if mode == GameMode.TWO_VS_TWO else "",
        ] if n]
        right_names = [n for n in [
            self._input_right.text().strip() or "Rechts",
            self._input_p1_right.text().strip() or "Spieler 1",
            self._input_p2_right.text().strip() if mode == GameMode.TWO_VS_TWO else "",
        ] if n]
        return GameConfig(
            mode=mode,
            team_left_names=left_names[:2] if mode == GameMode.TWO_VS_TWO else left_names[:1],
            team_right_names=right_names[:2] if mode == GameMode.TWO_VS_TWO else right_names[:1],
        )



class DashboardScreen(QWidget):
    def __init__(self, on_end_game: Callable, parent=None):
        super().__init__(parent)
        self._on_end_game = on_end_game
        self._build()

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(32, 0, 32, 16)
        root.setSpacing(0)

        # ── Header bar ──
        header = QHBoxLayout()
        self._team_left_lbl = QLabel("LINKS")
        self._team_left_lbl.setObjectName("team-name")
        timer_row = QHBoxLayout()
        timer_row.setSpacing(8)
        self._timer_lbl = QLabel("00:00")
        self._timer_lbl.setObjectName("timer")
        self._timer_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._live_dot = QLabel("●")
        self._live_dot.setObjectName("live-dot")
        timer_row.addWidget(self._timer_lbl)
        timer_row.addWidget(self._live_dot)
        self._team_right_lbl = QLabel("RECHTS")
        self._team_right_lbl.setObjectName("team-name")
        self._team_right_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        header.addWidget(self._team_left_lbl)
        header.addStretch()
        header.addLayout(timer_row)
        header.addStretch()
        header.addWidget(self._team_right_lbl)
        root.addSpacing(16)
        root.addLayout(header)
        root.addSpacing(4)
        root.addWidget(_Divider())

        # ── Score ──
        score_row = QHBoxLayout()
        self._score_left = QLabel("0")
        self._score_left.setObjectName("score")
        self._score_left.setAlignment(Qt.AlignmentFlag.AlignCenter)
        dash = QLabel("—")
        dash.setObjectName("score")
        dash.setAlignment(Qt.AlignmentFlag.AlignCenter)
        dash.setStyleSheet("font-size: 36px; color: #333333;")
        self._score_right = QLabel("0")
        self._score_right.setObjectName("score")
        self._score_right.setAlignment(Qt.AlignmentFlag.AlignCenter)
        score_row.addStretch()
        score_row.addWidget(self._score_left)
        score_row.addWidget(dash)
        score_row.addWidget(self._score_right)
        score_row.addStretch()
        root.addLayout(score_row)
        root.addWidget(_Divider())
        root.addSpacing(16)

        # ── Metric cards ──
        metrics_row = QHBoxLayout()
        metrics_row.setSpacing(8)
        self._card_max = _MetricCard("MAX GESCHW")
        self._card_cur = _MetricCard("AKT GESCHW")
        self._card_reb = _MetricCard("ABPRALLE")
        self._card_sho = _MetricCard("SCHÜSSE")
        for card in [self._card_max, self._card_cur, self._card_reb, self._card_sho]:
            card.setStyleSheet(f"background: {BG_SURFACE};")
            metrics_row.addWidget(card)
        root.addLayout(metrics_row)
        root.addSpacing(16)
        root.addWidget(_Divider())
        root.addSpacing(8)

        # ── Main area: field + events ──
        main_row = QHBoxLayout()
        main_row.setSpacing(16)

        # Left: field + zone bar
        field_col = QVBoxLayout()
        self._field = FieldLabel()
        self._field.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        field_col.addWidget(self._field)
        zone_lbl = QLabel("ZONENVERTEILUNG")
        zone_lbl.setObjectName("metric-label")
        field_col.addWidget(zone_lbl)
        self._zone_bar = _ZoneBar()
        field_col.addWidget(self._zone_bar)
        main_row.addLayout(field_col, stretch=3)

        # Right: event log + end button
        right_col = QVBoxLayout()
        right_col.setSpacing(8)
        events_hdr = QLabel("EREIGNISSE")
        events_hdr.setObjectName("section-header")
        right_col.addWidget(events_hdr)
        self._event_log = QListWidget()
        self._event_log.setObjectName("event-log")
        self._event_log.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        right_col.addWidget(self._event_log)
        end_btn = QPushButton("SPIEL BEENDEN")
        end_btn.setObjectName("primary")
        end_btn.clicked.connect(self._on_end_game)
        right_col.addWidget(end_btn)
        main_row.addLayout(right_col, stretch=1)

        root.addLayout(main_row)

    def update(self, frame, position, stats, score_left: int, score_right: int, config) -> None:
        self._score_left.setText(str(score_left))
        self._score_right.setText(str(score_right))
        if config:
            left_name = config.team_left_names[0].upper() if config.team_left_names else "LINKS"
            right_name = config.team_right_names[0].upper() if config.team_right_names else "RECHTS"
            self._team_left_lbl.setText(left_name)
            self._team_right_lbl.setText(right_name)
        if stats:
            self._card_max.set_value(f"{stats.max_speed_ms:.1f} m/s")
            self._card_cur.set_value(f"{stats.current_speed_ms:.1f} m/s")
            self._card_reb.set_value(str(stats.rebound_count))
            self._card_sho.set_value(str(stats.shot_count))
            self._timer_lbl.setText(_fmt_time(stats.game_time_seconds))
            self._zone_bar.update_zones(stats.zone_percentages)
        self._field.update_frame(frame, stats, config)

    def add_event(self, event: GameEvent) -> None:
        ts = _fmt_time(event.timestamp)
        text = f"{ts}  {event.description}"
        item = QListWidgetItem(text)
        if event.event_type == EventType.GOAL:
            item.setForeground(QColor(196, 30, 58))
        self._event_log.insertItem(0, item)

    def reset(self) -> None:
        self._event_log.clear()
        self._score_left.setText("0")
        self._score_right.setText("0")
        self._timer_lbl.setText("00:00")
        self._card_max.set_value("—")
        self._card_cur.set_value("—")
        self._card_reb.set_value("—")
        self._card_sho.set_value("—")



class SummaryScreen(QWidget):
    def __init__(self, on_new_game: Callable, on_quit: Callable, parent=None):
        super().__init__(parent)
        self._on_new_game = on_new_game
        self._on_quit = on_quit
        self._build()

    def _build(self):
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        container = QWidget()
        root = QVBoxLayout(container)
        root.setContentsMargins(48, 0, 48, 48)
        root.setSpacing(0)

        # ── Header ──
        hdr_row = QHBoxLayout()
        title = QLabel("KICKER GT3")
        title.setObjectName("app-title")
        summ_lbl = QLabel("SPIELZUSAMMENFASSUNG")
        summ_lbl.setObjectName("section-header")
        hdr_row.addWidget(title)
        hdr_row.addStretch()
        hdr_row.addWidget(summ_lbl)
        root.addSpacing(24)
        root.addLayout(hdr_row)
        root.addSpacing(6)
        root.addWidget(_Divider())
        root.addSpacing(24)

        # ── Winner + score ──
        self._winner_lbl = QLabel("—")
        self._winner_lbl.setObjectName("winner")
        self._winner_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(self._winner_lbl)
        root.addSpacing(8)

        score_row = QHBoxLayout()
        self._score_left_lbl = QLabel("0")
        self._score_left_lbl.setObjectName("score")
        self._score_left_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._time_lbl = QLabel("00:00 MIN")
        self._time_lbl.setObjectName("timer")
        self._time_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._score_right_lbl = QLabel("0")
        self._score_right_lbl.setObjectName("score")
        self._score_right_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        score_row.addStretch()
        score_row.addWidget(self._score_left_lbl)
        score_row.addSpacing(24)
        score_row.addWidget(self._time_lbl)
        score_row.addSpacing(24)
        score_row.addWidget(self._score_right_lbl)
        score_row.addStretch()
        root.addLayout(score_row)
        root.addSpacing(8)
        root.addWidget(_Divider())
        root.addSpacing(24)

        # ── Section A: Global metrics ──
        a_lbl = QLabel("STATISTIKEN")
        a_lbl.setObjectName("section-header")
        root.addWidget(a_lbl)
        root.addSpacing(12)
        metrics_row = QHBoxLayout()
        metrics_row.setSpacing(8)
        self._s_max_speed = _MetricCard("MAX GESCHW")
        self._s_shots = _MetricCard("SCHÜSSE")
        self._s_rebounds = _MetricCard("ABPRALLE")
        self._s_time = _MetricCard("SPIELZEIT")
        for c in [self._s_max_speed, self._s_shots, self._s_rebounds, self._s_time]:
            c.setStyleSheet(f"background: {BG_SURFACE};")
            metrics_row.addWidget(c)
        root.addLayout(metrics_row)
        root.addSpacing(24)
        root.addWidget(_Divider())
        root.addSpacing(24)

        # ── Section B: Speed Analyse ──
        b_lbl = QLabel("SPEED ANALYSE")
        b_lbl.setObjectName("section-header")
        root.addWidget(b_lbl)
        root.addSpacing(12)
        speed_row = QHBoxLayout()
        speed_row.setSpacing(0)
        self._sa_max = _MetricCard("MAX SPEED")
        self._sa_avg = _MetricCard("⌀ SPEED")
        self._sa_ts  = _MetricCard("TOP SPEED UM")
        for c in [self._sa_max, self._sa_avg, self._sa_ts]:
            c.setStyleSheet(f"background: {BG_ELEVATED};")
            speed_row.addWidget(c)
        root.addLayout(speed_row)
        root.addSpacing(24)
        root.addWidget(_Divider())
        root.addSpacing(24)

        # ── Section C: Team Vergleich ──
        c_lbl = QLabel("TEAM VERGLEICH")
        c_lbl.setObjectName("section-header")
        root.addWidget(c_lbl)
        root.addSpacing(8)

        self._team_hdr_row = QHBoxLayout()
        self._team_left_hdr = QLabel("LINKS")
        self._team_left_hdr.setStyleSheet(f"color: {ACCENT}; font-size: 11px; font-weight: 700; letter-spacing: 3px;")
        self._team_right_hdr = QLabel("RECHTS")
        self._team_right_hdr.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 11px; font-weight: 700; letter-spacing: 3px;")
        self._team_right_hdr.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._team_hdr_row.addWidget(self._team_left_hdr)
        self._team_hdr_row.addStretch()
        self._team_hdr_row.addWidget(self._team_right_hdr)
        root.addLayout(self._team_hdr_row)
        root.addSpacing(8)

        self._cmp_possession = _ComparisonRow("Ballbesitz %")
        self._cmp_speed      = _ComparisonRow("Max Speed m/s")
        self._cmp_shots      = _ComparisonRow("Schüsse")
        for w in [self._cmp_possession, self._cmp_speed, self._cmp_shots]:
            root.addWidget(w)
        root.addSpacing(24)
        root.addWidget(_Divider())
        root.addSpacing(24)

        # ── Section D: Heatmaps ──
        d_lbl = QLabel("HEATMAP PRO TEAM")
        d_lbl.setObjectName("section-header")
        root.addWidget(d_lbl)
        root.addSpacing(12)
        heatmap_row = QHBoxLayout()
        heatmap_row.setSpacing(16)
        left_hm_col = QVBoxLayout()
        self._hm_left_name = QLabel("LINKS")
        self._hm_left_name.setObjectName("metric-label")
        self._hm_left_name.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._hm_left_lbl = QLabel()
        self._hm_left_lbl.setMinimumHeight(160)
        self._hm_left_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._hm_left_lbl.setStyleSheet(f"background: {BG_SURFACE};")
        left_hm_col.addWidget(self._hm_left_name)
        left_hm_col.addWidget(self._hm_left_lbl)
        right_hm_col = QVBoxLayout()
        self._hm_right_name = QLabel("RECHTS")
        self._hm_right_name.setObjectName("metric-label")
        self._hm_right_name.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._hm_right_lbl = QLabel()
        self._hm_right_lbl.setMinimumHeight(160)
        self._hm_right_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._hm_right_lbl.setStyleSheet(f"background: {BG_SURFACE};")
        right_hm_col.addWidget(self._hm_right_name)
        right_hm_col.addWidget(self._hm_right_lbl)
        heatmap_row.addLayout(left_hm_col)
        heatmap_row.addLayout(right_hm_col)
        root.addLayout(heatmap_row)
        root.addSpacing(24)
        root.addWidget(_Divider())
        root.addSpacing(24)

        # ── Section E: Goal Timeline ──
        e_lbl = QLabel("ZIEL-TIMELINE")
        e_lbl.setObjectName("section-header")
        root.addWidget(e_lbl)
        root.addSpacing(8)
        self._timeline = GoalTimelineWidget()
        root.addWidget(self._timeline)
        root.addSpacing(24)
        root.addWidget(_Divider())
        root.addSpacing(32)

        # ── Buttons ──
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        new_btn = QPushButton("NEUES SPIEL")
        new_btn.setObjectName("primary")
        new_btn.clicked.connect(self._on_new_game)
        quit_btn = QPushButton("BEENDEN")
        quit_btn.setObjectName("secondary")
        quit_btn.clicked.connect(self._on_quit)
        btn_row.addWidget(new_btn)
        btn_row.addSpacing(16)
        btn_row.addWidget(quit_btn)
        btn_row.addStretch()
        root.addLayout(btn_row)

        scroll.setWidget(container)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    def populate(self, stats, config, score_left: int, score_right: int) -> None:
        if stats is None or config is None:
            return

        left_name = config.team_left_names[0].upper() if config.team_left_names else "LINKS"
        right_name = config.team_right_names[0].upper() if config.team_right_names else "RECHTS"

        # Winner
        if score_left > score_right:
            self._winner_lbl.setText(f"{left_name}  ·  SIEGER")
        elif score_right > score_left:
            self._winner_lbl.setText(f"{right_name}  ·  SIEGER")
        else:
            self._winner_lbl.setText("UNENTSCHIEDEN")

        secs = stats.game_time_seconds
        self._score_left_lbl.setText(str(score_left))
        self._score_right_lbl.setText(str(score_right))
        self._time_lbl.setText(f"{_fmt_time(secs)} MIN")

        # Section A
        self._s_max_speed.set_value(f"{stats.max_speed_ms:.1f} m/s")
        self._s_shots.set_value(str(stats.shot_count))
        self._s_rebounds.set_value(str(stats.rebound_count))
        self._s_time.set_value(_fmt_time(secs))

        # Section B
        self._sa_max.set_value(f"{stats.max_speed_ms:.1f} m/s")
        self._sa_avg.set_value(f"{stats.average_speed_ms:.1f} m/s")
        self._sa_ts.set_value(_fmt_time(stats.max_speed_timestamp))

        # Section C
        poss = stats.team_possession_pct
        tm_speed = stats.team_max_speed
        tm_shots = stats.team_shot_counts
        self._cmp_possession.update_values(poss[Team.LEFT], poss[Team.RIGHT], "{:.0f}%")
        self._cmp_speed.update_values(tm_speed[Team.LEFT], tm_speed[Team.RIGHT], "{:.1f}")
        self._cmp_shots.update_values(float(tm_shots[Team.LEFT]), float(tm_shots[Team.RIGHT]))
        self._team_left_hdr.setText(left_name)
        self._team_right_hdr.setText(right_name)
        self._hm_left_name.setText(left_name)
        self._hm_right_name.setText(right_name)

        # Section D: heatmaps
        hms = stats.team_heatmaps
        self._render_heatmap(hms[Team.LEFT], self._hm_left_lbl)
        self._render_heatmap(hms[Team.RIGHT], self._hm_right_lbl)

        # Section E: timeline — use relative timestamps
        events = stats.events
        game_start = secs  # total game seconds
        self._timeline.set_events(events, game_start)

    @staticmethod
    def _render_heatmap(hm: "np.ndarray", label: QLabel) -> None:
        if hm is None or hm.max() == 0:
            label.setText("KEINE DATEN")
            label.setStyleSheet(f"background: {BG_SURFACE}; color: {TEXT_SECONDARY}; font-size: 10px; letter-spacing: 2px;")
            return
        hm_uint8 = (hm * 255).astype(np.uint8)
        colored = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_HOT)
        rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        label.setStyleSheet("")
        label.setText("")
        label.setPixmap(
            QPixmap.fromImage(qimg).scaled(
                label.size() if label.width() > 10 else QSize(300, 160),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )



class FrameUpdateSignal(QObject):
    """Emitted from background thread, delivered to main thread via Qt queued connection."""
    update = pyqtSignal(object, object, object, int, int)


class KickerGUI(QMainWindow):
    def __init__(
        self,
        on_start: Callable,
        on_end_game: Callable,
        on_new_game: Callable,
        on_quit: Callable,
    ) -> None:
        self._app = QApplication.instance() or QApplication([])
        super().__init__()
        self._on_start = on_start
        self._on_end_game = on_end_game
        self._on_new_game = on_new_game
        self._on_quit = on_quit

        self.setWindowTitle("Kicker GT3")
        self.setMinimumSize(1024, 700)
        self._app.setStyleSheet(GLOBAL_QSS)

        # Stacked widget with 3 screens
        self._stack = QStackedWidget()
        self.setCentralWidget(self._stack)

        self._start_screen = StartScreen(on_start=self._handle_start)
        self._dash_screen  = DashboardScreen(on_end_game=self._handle_end_game)
        self._summ_screen  = SummaryScreen(on_new_game=self._handle_new_game,
                                            on_quit=self._handle_quit)

        self._stack.addWidget(self._start_screen)   # index 0
        self._stack.addWidget(self._dash_screen)    # index 1
        self._stack.addWidget(self._summ_screen)    # index 2

        # Signal for thread-safe dashboard updates
        self.frame_signal = FrameUpdateSignal()
        self.frame_signal.update.connect(self._on_frame_update)

        # LIVE dot blink timer
        self._live_timer = QTimer(self)
        self._live_timer.setInterval(800)
        self._live_timer.timeout.connect(self._toggle_live_dot)
        self._live_dot_state = True

        # Current config (needed for dashboard updates)
        self._current_config = None

    # ── Screen navigation ──────────────────────────────────────────────────

    def show_start_screen(self) -> None:
        self._live_timer.stop()
        self._stack.setCurrentIndex(0)

    def show_dashboard(self) -> None:
        self._dash_screen.reset()
        self._stack.setCurrentIndex(1)
        self._live_timer.start()

    def show_summary(self, stats, config, score_left: int, score_right: int) -> None:
        self._live_timer.stop()
        self._summ_screen.populate(stats, config, score_left, score_right)
        self._stack.setCurrentIndex(2)

    # ── Dashboard update (called from signal — main thread) ────────────────

    def update_dashboard(self, frame, position, stats, score_left: int, score_right: int) -> None:
        self.frame_signal.update.emit(frame, position, stats, score_left, score_right)

    def _on_frame_update(self, frame, position, stats, score_left: int, score_right: int) -> None:
        if self._stack.currentIndex() != 1:
            return
        self._dash_screen.update(frame, position, stats, score_left, score_right,
                                  self._current_config)
        # Forward new goal events to event log
        if stats is not None:
            for event in stats.events:
                if event.event_type == EventType.GOAL:
                    pass  # events are shown from controller callback

    # ── Button handlers ───────────────────────────────────────────────────

    def _handle_start(self, config) -> None:
        self._current_config = config
        self._on_start(config)

    def _handle_end_game(self) -> None:
        self._on_end_game()

    def _handle_new_game(self) -> None:
        self._current_config = None
        self._on_new_game()

    def _handle_quit(self) -> None:
        self._on_quit()

    # ── LIVE dot blink ────────────────────────────────────────────────────

    def _toggle_live_dot(self) -> None:
        self._live_dot_state = not self._live_dot_state
        dot = self._dash_screen._live_dot
        dot.setVisible(self._live_dot_state)

    # ── Event forwarding ─────────────────────────────────────────────────

    def add_event(self, event: GameEvent) -> None:
        """Called by controller when a notable event occurs."""
        self._dash_screen.add_event(event)

    # ── Run ───────────────────────────────────────────────────────────────

    def run(self) -> None:
        self._app.exec()
