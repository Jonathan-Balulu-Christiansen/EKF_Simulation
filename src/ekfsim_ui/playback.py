"""the playback clock and its transport bar (THEORY 8.1). the run is computed once, then played back: a
QTimer at about 60 Hz only asks for a new frame, and the sim time advances by the wall time that really
elapsed (QElapsedTimer) times the playback speed, never by counting timer ticks. so a late or skipped
tick changes the frame rate, not the speed of sim time.
t_sim += (wall time since the last tick) x speed - playback clock"""

import math

from PySide6.QtCore import QElapsedTimer, QObject, Qt, QTimer, Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QSlider, QToolButton, QWidget

from ekfsim.config import UIConfig
from ekfsim_ui import fmt, theme
from ekfsim_ui.theme import C
from ekfsim_ui.widgets import icon

SLIDER_STEPS_PER_S = 100  # scrubber resolution: one slider step per 10 ms of sim time [1/s]


# the clock part:

class PlaybackClock(QObject):
    """sim time between 0 and the end of the run, advanced from wall time while playing."""

    timeChanged = Signal(float)
    playingChanged = Signal(bool)
    speedChanged = Signal(float)

    def __init__(self, ui_cfg: UIConfig, parent=None):
        super().__init__(parent)
        self.t = 0.0  # sim time in [s]
        self.t_end = 0.0  # end of the run in [s]
        self.speed = ui_cfg.default_speed  # sim seconds per wall second [-]
        self.playing = False
        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._timer.setInterval(max(1, math.floor(1000.0 * ui_cfg.frame_interval)))  # 16 ms: rounding 16.7 up would give 58.8 Hz
        self._timer.timeout.connect(self._tick)
        self._wall = QElapsedTimer()
        self._last_ns = 0

    def set_range(self, t_end: float) -> None:
        self.t_end = t_end
        self.seek(min(self.t, t_end))

    def _tick(self) -> None:
        now = self._wall.nsecsElapsed()
        dt_wall = (now - self._last_ns) * 1e-9  # wall time since the last tick in [s]
        self._last_ns = now
        t = self.t + dt_wall * self.speed  # t_sim += dt_wall x speed
        if t >= self.t_end:
            t = self.t_end
            self.pause()
        self.t = t
        self.timeChanged.emit(t)

    def play(self) -> None:
        if self.t_end <= 0.0:
            return
        if self.t >= self.t_end:
            self.t = 0.0
        self.playing = True
        self._wall.start()
        self._last_ns = 0
        self._timer.start()
        self.playingChanged.emit(True)

    def pause(self) -> None:
        self._timer.stop()
        if self.playing:
            self.playing = False
            self.playingChanged.emit(False)

    def toggle(self) -> None:
        self.pause() if self.playing else self.play()

    def seek(self, t: float) -> None:
        self.t = min(max(float(t), 0.0), self.t_end)
        if self.playing:
            self._last_ns = self._wall.nsecsElapsed()
        self.timeChanged.emit(self.t)

    def set_speed(self, speed: float) -> None:
        self.speed = float(speed)
        self.speedChanged.emit(self.speed)


# the transport bar part:

def speed_text(speed: float) -> str:
    """a playback speed as a factor: ×5, ×0.25, or ×1/100,000 in the pulse view."""
    return f"×1/{1.0 / speed:,.0f}" if speed < 0.01 else f"×{speed:g}"


class TransportBar(QWidget):
    """play/pause, back to start, one scan back or forward, the time scrubber, the sim time, and the button of
    the equations drawer (the playback speed is set in the Run group of the controls)."""

    stepRequested = Signal(int)  # -1 or +1 scan

    def __init__(self, clock: PlaybackClock, ui_cfg: UIConfig):
        super().__init__()
        self.clock = clock
        self.setObjectName("panel")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        lay = QHBoxLayout(self)
        e = theme.em()
        lay.setContentsMargins(round(0.6 * e), round(0.3 * e), round(0.6 * e), round(0.3 * e))
        lay.setSpacing(round(0.4 * e))

        def button(name, tip, slot):
            b = QToolButton()
            b.setIcon(icon(name, C.INK))
            b.setToolTip(tip)
            b.setAutoRaise(False)
            b.clicked.connect(slot)
            lay.addWidget(b)
            return b

        button("to_start", "back to t = 0 (Home)", lambda: clock.seek(0.0))
        button("step_back", "one scan back (Left)", lambda: self.stepRequested.emit(-1))
        self.play_button = button("play", "play / pause (Space)", clock.toggle)
        button("step_forward", "one scan forward (Right)", lambda: self.stepRequested.emit(+1))

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.setToolTip("scrub the playback time; the run is precomputed, so any time can be shown")
        self.slider.valueChanged.connect(self._slider_moved)
        lay.addWidget(self.slider, 1)

        self.time_label = QLabel()
        self.time_label.setFont(theme.font("display"))
        self.time_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.time_label.setMinimumWidth(round(theme.char_width("display") * 12))
        self.time_label.setToolTip("sim time t since lift-off")
        lay.addWidget(self.time_label)
        unit = QLabel("s")
        unit.setProperty("role", "unit")
        lay.addWidget(unit)

        self.speed_label = QLabel(speed_text(clock.speed))
        self.speed_label.setFont(theme.font("mono"))
        self.speed_label.setProperty("role", "secondary")
        self.speed_label.setToolTip("playback speed (set in the Run group; keys + and −)")
        lay.addWidget(self.speed_label)
        self.pulse_button = QToolButton()
        self.pulse_button.setText("Pulse")
        self.pulse_button.setCheckable(True)
        self.pulse_button.setToolTip("slow-motion pulse view: one radar pulse travels out and back at its true "
                                     "length, with time slowed by the factor shown (P)")
        lay.addWidget(self.pulse_button)
        self.equations_button = QToolButton()
        self.equations_button.setText("Equations")
        self.equations_button.setCheckable(True)
        self.equations_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.equations_button.setIcon(icon("chev_up", C.INK))
        self.equations_button.setToolTip("open or close the equations drawer (E)")
        self.equations_button.toggled.connect(
            lambda on: self.equations_button.setIcon(icon("chev_down" if on else "chev_up", C.INK)))
        lay.addWidget(self.equations_button)

        self.decimals = 2  # digits of the sim time; 7 (0.1 us) in the pulse view
        clock.timeChanged.connect(self._on_time)
        clock.playingChanged.connect(lambda p: self.play_button.setIcon(icon("pause" if p else "play", C.INK)))
        clock.speedChanged.connect(self._on_speed)
        self._on_time(0.0)

    def set_range(self, t_end: float) -> None:
        self.slider.blockSignals(True)
        self.slider.setRange(0, round(t_end * SLIDER_STEPS_PER_S))
        self.slider.setPageStep(10 * SLIDER_STEPS_PER_S)
        self.slider.blockSignals(False)

    def _slider_moved(self, v: int) -> None:
        self.clock.seek(v / SLIDER_STEPS_PER_S)

    def _on_time(self, t: float) -> None:
        self.time_label.setText(fmt.num(t, self.decimals))
        self.slider.blockSignals(True)
        self.slider.setValue(round(t * SLIDER_STEPS_PER_S))
        self.slider.blockSignals(False)

    def _on_speed(self, s: float) -> None:
        self.speed_label.setText(speed_text(s))
