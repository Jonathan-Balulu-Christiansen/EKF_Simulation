"""the main window (THEORY 8.1, 8.2, 8.9): a fixed layout of QSplitters (controls and readout left, the 3D view
with its transport bar in the centre, the plot tabs right, the equations panel at the bottom) and the status
bar. a run is started from a worker thread (QThreadPool) into an immutable timeline; the GUI thread only
renders it at the playback cursor. the computation itself runs in one persistent child process: a Python
worker thread shares the GIL with the GUI thread, and every frame makes dozens of Python paint callbacks that
each had to win the GIL back from the busy worker (measured: 533 ms gaps, 6 Hz, during a 1.6 s run). the
worker thread now only waits for the process, without holding the GIL.

minimum and initial sizes are in em (the body line height), so the layout follows the font and the screen
scaling; the left and right columns keep their width and the 3D view takes the rest."""

import logging
import math
import multiprocessing
import traceback
from concurrent.futures import ProcessPoolExecutor

from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, QTimer, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import QHBoxLayout, QMainWindow, QSplitter, QVBoxLayout, QWidget

from ekfsim.config import UIConfig
from ekfsim_ui import fmt, theme
from ekfsim_ui.controls import ControlPanel
from ekfsim_ui.equations_panel import EquationsPanel
from ekfsim_ui.playback import PlaybackClock, TransportBar, speed_text
from ekfsim_ui.plots import PlotPanel
from ekfsim_ui.readout import ReadoutPanel
from ekfsim_ui.theme import C, MODE_COLOUR, STATUS_COLOUR
from ekfsim_ui.timeline import RunRequest, Timeline, compute
from ekfsim_ui.view3d import Scene3D
from ekfsim_ui.widgets import Chip, icon, label, scroll

log = logging.getLogger("ekfsim")
TITLE = "EKF ballistic target tracker"

# minimum sizes in em (THEORY 8.9)
MIN_LEFT_W, MIN_CENTRE_W, MIN_RIGHT_W = 19.0, 26.0, 22.0
MIN_READOUT_H, MIN_VIEW_H, MIN_PLOTS_H, MIN_EQUATIONS_H = 6.0, 14.0, 14.0, 8.0
SPLIT_W = (0.18, 0.57, 0.25)  # initial share of the width: left, centre (the 3D view dominates, THEORY 8.2), right
DRAWER_MAX = 0.55  # largest share of the height the open equations drawer takes [-]
ADVANCED_SHARE = 0.62  # share of the left column for the controls while "Advanced" is open [-]


# the worker part:

class _JobSignals(QObject):
    done = Signal(object, int)
    failed = Signal(str, str, int)


class RunJob(QRunnable):
    """one run: truth, radar, EKF and the display arrays, computed in the child process (module docstring)
    while this worker thread waits for it."""

    def __init__(self, request: RunRequest, token: int, ui_cfg: UIConfig, executor: ProcessPoolExecutor):
        super().__init__()
        self.request, self.token, self.ui_cfg, self.executor = request, token, ui_cfg, executor
        self.signals = _JobSignals()

    def run(self):
        try:
            tl = self.executor.submit(compute, self.request, self.ui_cfg).result()
        except Exception as e:  # reported to the GUI thread, which shows it
            self.signals.failed.emit(f"{type(e).__name__}: {e}", traceback.format_exc(), self.token)
            return
        self.signals.done.emit(tl, self.token)


# the window part:

class MainWindow(QMainWindow):
    """the whole application window."""

    runFinished = Signal(object)  # Timeline, after the views have taken it

    def __init__(self, ui_cfg: UIConfig | None = None, autoplay: bool = True):
        super().__init__()
        self.ui = ui_cfg or UIConfig()
        self.autoplay = autoplay
        self.setWindowTitle(TITLE)
        self.tl: Timeline | None = None
        self._token = 0
        self._pending: RunRequest | None = None
        self.pulse_on = False
        self.pulse_factor = self.ui.default_pulse_factor  # slow-motion factor of the pulse view [-]
        self._speed_before = self.ui.default_speed
        self._running = False
        self._frame = None
        self.pool = QThreadPool(self)
        self.pool.setMaxThreadCount(1)
        self._executor: ProcessPoolExecutor | None = None

        self.clock = PlaybackClock(self.ui, self)
        self.controls = ControlPanel(self.ui)
        self.readout = ReadoutPanel()
        self.scene = Scene3D(self.ui)
        self.transport = TransportBar(self.clock, self.ui)
        self.plots = PlotPanel(self.ui)
        self.equations = EquationsPanel()
        self._build_layout()
        self._build_status_bar()
        self._connect()

    # the layout part

    def _build_layout(self):
        e = theme.em()
        left = QSplitter(Qt.Orientation.Vertical)
        readout_scroll = scroll(self.readout, horizontal=True)  # scrolls rather than clips (THEORY 8.9)
        readout_scroll.setObjectName("panel")
        left.addWidget(self.controls)
        left.addWidget(readout_scroll)
        readout_scroll.setMinimumHeight(round(MIN_READOUT_H * e))
        left.setMinimumWidth(round(MIN_LEFT_W * e))

        centre = QWidget()
        cl = QVBoxLayout(centre)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(0)
        cl.addWidget(self.scene, 1)
        cl.addWidget(self.transport)
        self.scene.setMinimumHeight(round(MIN_VIEW_H * e))
        # never narrower than the transport bar needs, or the splitter squeezes its buttons on top of each other
        centre.setMinimumWidth(max(round(MIN_CENTRE_W * e), self.transport.minimumSizeHint().width()))
        self.plots.setMinimumSize(round(MIN_RIGHT_W * e), round(MIN_PLOTS_H * e))

        row = QSplitter(Qt.Orientation.Horizontal)
        for w, stretch in ((left, 0), (centre, 1), (self.plots, 0)):
            row.addWidget(w)
            row.setStretchFactor(row.indexOf(w), stretch)
        outer = QSplitter(Qt.Orientation.Vertical)
        outer.addWidget(row)
        outer.addWidget(self.equations)
        outer.setStretchFactor(0, 1)
        outer.setStretchFactor(1, 0)
        self.equations.setMinimumHeight(round(MIN_EQUATIONS_H * e))
        self.equations.hide()  # a drawer, closed by default (THEORY 8.2)
        for s in (left, row, outer):
            s.setChildrenCollapsible(False)
            s.setHandleWidth(max(3, round(0.3 * e)))
        self._splitters = (left, row, outer)
        self.setCentralWidget(outer)
        self._sized = False

    def _initial_split(self):
        """set the initial splitter shares once the window has its real size."""
        left, row, outer = self._splitters
        w = row.width()
        row.setSizes([round(f * w) for f in SPLIT_W])
        self._fit_left(self.controls.advanced.button.isChecked())
        if self.equations.isVisible():
            self._open_drawer_sizes()
        self._sized = True

    def _fit_left(self, advanced: bool) -> None:
        """the controls take their natural height with "Advanced" closed, most of the column when it is open."""
        left = self._splitters[0]
        h = left.height()
        top = round(ADVANCED_SHARE * h) if advanced else min(self.controls.sizeHint().height(), h // 2)
        left.setSizes([top, h - top])

    # the equations drawer part (THEORY 8.2, 8.6)

    def toggle_equations(self, on: bool | None = None) -> None:
        """open or close the equations drawer (button and key E)."""
        on = self.equations.isHidden() if on is None else on
        self.transport.equations_button.blockSignals(True)
        self.transport.equations_button.setChecked(on)
        self.transport.equations_button.blockSignals(False)
        self.transport.equations_button.setIcon(icon("chev_down" if on else "chev_up", C.INK))
        self.equations.setVisible(on)
        if on:
            self._open_drawer_sizes()
            self.render_time(self.clock.t)

    # the pulse view part (THEORY 8.5)

    def toggle_pulse(self, on: bool | None = None) -> None:
        """the slow-motion pulse view: playback pauses at the emission of the current scan's pulse and then runs
        at 1/factor of real time; every object is slowed by the same factor."""
        on = not self.pulse_on if on is None else bool(on)
        if self.tl is None or on == self.pulse_on:
            self._sync_pulse_widgets()
            return
        self.pulse_on = on
        self._sync_pulse_widgets()
        self.transport.decimals = 7 if on else 2
        self.clock.pause()
        if on:
            self._speed_before = self.clock.speed
            self.clock.seek(float(self.tl.t_scan[self.tl.scan_index(self.clock.t)]))
            self.clock.set_speed(1.0 / self.pulse_factor)
        else:
            self.clock.set_speed(self._speed_before)
        self.controls.speed.setEnabled(not on)
        self.scene.set_pulse_mode(on, self.pulse_factor)
        self.clock.seek(self.clock.t)  # redraw every view, the time label with its new digits too

    def _sync_pulse_widgets(self) -> None:
        for w in (self.transport.pulse_button, self.controls.pulse):
            w.blockSignals(True)
            w.setChecked(self.pulse_on)
            w.blockSignals(False)

    def set_pulse_factor(self, factor: float) -> None:
        self.pulse_factor = float(factor)
        if self.pulse_on:
            self.clock.set_speed(1.0 / self.pulse_factor)
            self.scene.set_pulse_mode(True, self.pulse_factor)

    def _open_drawer_sizes(self) -> None:
        outer = self._splitters[2]
        h = outer.height()
        drawer = min(self.equations.preferred_height(), round(DRAWER_MAX * h))
        outer.setSizes([h - drawer, drawer])

    def showEvent(self, event):
        super().showEvent(event)
        if not self._sized:
            QTimer.singleShot(0, self._initial_split)

    # the status bar part

    def _build_status_bar(self):
        sb = self.statusBar()
        sb.setSizeGripEnabled(True)
        e = theme.em()
        box = QWidget()
        h = QHBoxLayout(box)
        h.setContentsMargins(round(0.5 * e), 0, round(0.5 * e), 0)
        h.setSpacing(round(0.6 * e))
        self.sb_time = label("", "value", "mono")
        self.sb_time.setToolTip("sim time")
        self.sb_phase = Chip("—", C.INK_3, "mono_caption")
        self.sb_phase.setToolTip("flight phase as reported by the IR plume sensor (latency L, THEORY 5.7)")
        self.sb_status = Chip("—", C.INK_3, "mono_caption")
        self.sb_status.setToolTip("track status (THEORY 6.7)")
        self.sb_miss = label("", "secondary", "mono")
        self.sb_miss.setToolTip("missed detections while tracking (search scans before the first detection apart)")
        self.sb_net = Chip("—", C.INK_3, "mono_caption")
        self.sb_net.setToolTip("safety-net manoeuvre noise (THEORY 6.6)")
        self.sb_speed = label("", "secondary", "mono")
        self.sb_speed.setToolTip("playback speed")
        self.sb_message = label("", "caption", "caption")
        for w in (self.sb_time, self.sb_phase, self.sb_status, self.sb_miss, self.sb_net, self.sb_speed):
            h.addWidget(w)
        h.addWidget(self.sb_message, 1)
        sb.addWidget(box, 1)

    def _connect(self):
        self.controls.applyRequested.connect(self.start_run)
        self.controls.layerToggled.connect(self._layer)
        self.controls.ellipsoidScaleChanged.connect(self.scene.set_ellipsoid_scale)
        self.scene.ellipsoidScaleChanged.connect(self.controls.set_ellipsoid_scale)
        self.controls.resetViewRequested.connect(self.scene.reset_view)
        self.controls.speedRequested.connect(self.clock.set_speed)
        self.controls.advancedToggled.connect(self._fit_left)
        self.transport.equations_button.toggled.connect(self.toggle_equations)
        self.transport.pulse_button.toggled.connect(self.toggle_pulse)
        self.controls.pulseToggled.connect(self.toggle_pulse)
        self.controls.pulseFactorChanged.connect(self.set_pulse_factor)
        self.clock.timeChanged.connect(self.render_time)
        self.clock.playingChanged.connect(lambda playing: None if playing else self.render_time(self.clock.t))
        self.clock.speedChanged.connect(lambda s: self.sb_speed.setText(speed_text(s)))
        self.clock.speedChanged.connect(self.controls.set_speed)
        self.transport.stepRequested.connect(self.step_scan)
        self.plots.seekRequested.connect(self.clock.seek)
        for keys, slot in (("Space", self.clock.toggle), ("Left", lambda: self.step_scan(-1)),
                           ("Right", lambda: self.step_scan(+1)), ("Home", lambda: self.clock.seek(0.0)),
                           ("End", lambda: self.clock.seek(self.clock.t_end)), ("F11", self.toggle_full_screen),
                           ("Escape", self._leave_full_screen), ("R", self.scene.reset_view),
                           ("E", self.toggle_equations), ("L", self.scene.toggle_legend),
                           ("P", self.toggle_pulse),
                           ("+", lambda: self.step_speed(+1)), ("=", lambda: self.step_speed(+1)),
                           ("-", lambda: self.step_speed(-1))):
            sc = QShortcut(QKeySequence(keys), self)
            sc.setContext(Qt.ShortcutContext.WindowShortcut)
            sc.activated.connect(slot)
        self.sb_speed.setText(speed_text(self.clock.speed))

    # the run part

    def start_run(self, request: RunRequest | None = None) -> None:
        """compute a run in the worker thread. a request that arrives during a run waits for it."""
        request = request or self.controls.request()
        if self._running:
            self._pending = request
            return
        self._running = True
        self._token += 1
        self.controls.set_busy(True)
        self.sb_message.setText("computing truth, radar and EKF…")
        job = RunJob(request, self._token, self.ui, self.executor())
        job.signals.done.connect(self._run_done)
        job.signals.failed.connect(self._run_failed)
        self._job = job
        self.pool.start(job)
        self._requested = request

    def executor(self) -> ProcessPoolExecutor:
        """the compute process, started on first use (spawn: a clean interpreter without Qt)."""
        if self._executor is None:
            self._executor = ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"))
        return self._executor

    def closeEvent(self, event):
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
        super().closeEvent(event)

    def _run_finished_common(self):
        self._running = False
        self.controls.set_busy(False)
        if self._pending is not None:
            req, self._pending = self._pending, None
            self.start_run(req)

    def _run_done(self, tl: Timeline, token: int) -> None:
        if token != self._token:
            return
        first = self.tl is None
        self.tl = tl
        self.controls.mark_applied(tl.request)
        self.controls.set_last_run(tl)
        self.controls.show_error("")
        self.scene.set_timeline(tl)
        self.plots.set_timeline(tl)
        self.equations.set_timeline(tl)
        self.transport.set_range(tl.t_end)
        self.clock.set_range(tl.t_end)
        self._frame = None
        self.sb_message.setText(self._run_message(tl))
        self._log_run(tl)
        self._run_finished_common()
        self.render_time(self.clock.t)
        if first and self.autoplay:
            self.clock.seek(0.0)
            self.clock.play()
        self.runFinished.emit(tl)

    def _run_failed(self, message: str, tb: str, token: int) -> None:
        if token != self._token:
            return
        log.error("run failed: %s\n%s", message, tb)
        if message.startswith("BrokenProcessPool"):
            self._executor = None  # the child process died; the next run starts a new one
        self.controls.show_error(f"run refused: {message}")
        self.sb_message.setText("run refused (see the message under the Run button)")
        self._run_finished_common()

    @staticmethod
    def _run_message(tl: Timeline) -> str:
        if not tl.acquired:
            return "NO ACQUISITION: " + tl.acquisition_reason
        msg = (f"first detection at {fmt.num(tl.t_scan[tl.k0], 1)} s · {tl.n_scans:,} scans · "
               f"flight ends at {fmt.num(tl.t_end, 1)} s")
        if tl.elevation_warning:
            msg += f" · warning: max elevation {math.degrees(tl.max_elevation):.1f}° > 80°"
        return msg

    @staticmethod
    def _log_run(tl: Timeline) -> None:
        log.info("run: dry_mass_delta %.0f kg, seed %d, %d scans, acquired %s, misses search %d / track %d",
                 tl.request.vehicle.dry_mass_delta, tl.request.scenario.seed, tl.n_scans, tl.acquired,
                 int(tl.search_miss_cum[-1]), int(tl.track_miss_cum[-1]))
        for ev in tl.events:
            log.info("  t = %8.2f s  %s -> %s: %s", ev.t, ev.old.value if ev.old else "-", ev.new.value, ev.reason)

    # the frame part

    def step_speed(self, n: int) -> None:
        """the next faster (n = +1) or slower (n = -1) playback speed of the list."""
        speeds = list(self.ui.playback_speeds)
        i = min(range(len(speeds)), key=lambda j: abs(speeds[j] - self.clock.speed))
        self.clock.set_speed(speeds[min(max(i + n, 0), len(speeds) - 1)])

    def step_scan(self, n: int) -> None:
        """move the cursor to the scan n scans away (pauses playback)."""
        if self.tl is None:
            return
        self.clock.pause()
        k = self.tl.scan_index(self.clock.t)
        k = min(max(k + n, 0), self.tl.n_scans - 1)
        self.clock.seek(float(self.tl.t_scan[k]))

    def render_time(self, t: float) -> None:
        """draw every view at sim time t: scan k (filter, held between scans) and truth sample i."""
        tl = self.tl
        if tl is None:
            return
        k, i = tl.scan_index(t), tl.truth_index(t)
        self.scene.update_frame(k, i)
        self.readout.update_cursor(tl, t, k, i, self.clock.speed)
        self.plots.set_cursor(t, float(tl.truth_mach[i]))
        self.equations.show_cursor(k, i, self.clock.playing)
        if self.pulse_on:
            state = tl.pulse_state(t)
            self.scene.update_pulse(state, self.pulse_factor)
            if self.clock.playing and t >= state.done:
                self.clock.pause()  # the echo is back (or the pulse has left the beam): nothing moves until the next pulse
        if (k, i) != self._frame:
            self._status(tl, t, k)
            self._frame = (k, i)
        self.sb_time.setText(f"t {fmt.num(t, 7 if self.pulse_on else 2)} s")

    def _status(self, tl: Timeline, t: float, k: int) -> None:
        flag = bool(tl.engine_flag[k])
        phase = "BOOST" if flag else "BALLISTIC"
        self.sb_phase.set(phase, MODE_COLOUR[phase])
        if tl.tracking(k):
            s = str(tl.status[k])
            self.sb_status.set(s, STATUS_COLOUR.get(s, C.INK_2))
            on = bool(tl.manoeuvre_on[k])
            self.sb_net.set("manoeuvre noise ON" if on else "manoeuvre noise OFF", C.COASTING if on else C.INK_3)
        else:
            self.sb_status.set("NO ACQUISITION" if not tl.acquired else "SEARCH", C.LOST if not tl.acquired else C.INK_2)
            self.sb_net.set("manoeuvre noise —", C.INK_3)
        self.sb_miss.setText(f"misses {int(tl.track_miss_cum[k]):,}")

    def _layer(self, name: str, on: bool) -> None:
        self.scene.set_layer(name, on)
        self.scene._last_k = None
        self.render_time(self.clock.t)

    # the window state part

    def toggle_full_screen(self) -> None:
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def _leave_full_screen(self) -> None:
        if self.isFullScreen():
            self.showNormal()
