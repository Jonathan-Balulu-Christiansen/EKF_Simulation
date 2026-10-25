"""the controls (THEORY 8.7, progressive disclosure). the basic group holds only
what a run is about: the mass (absolute kg), the playback speed, boost mode and safety net on/off, Run and
Reset. the scenario, vehicle details, radar, filter and view settings sit in a collapsed "Advanced" section,
with their derived values shown live. nothing recomputes while a value is being edited: the mass slider
applies on release, every other change waits for Run / Apply (THEORY 8.1). each control maps to one config field;
the controls show display units (km, deg, GHz, ms, %) and the request carries SI (THEORY 0.4).
derived values that come from the last run's truth (geometry, required PRF) carry the "evaluation - uses
truth" tag; the rest is computed from config only."""

import dataclasses
import math

from PySide6.QtCore import QEvent, QObject, Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ekfsim.atmosphere import US76Atmosphere
from ekfsim.config import (
    AtmosphereConfig,
    FilterConfig,
    IRSensorConfig,
    PhysicsConstants,
    RadarConfig,
    ScenarioConfig,
    UIConfig,
    VehicleConfig,
)
from ekfsim.radar import (
    cue_direction,
    doppler_shift,
    required_prf,
    unambiguous_range,
    validate_radar,
    velocity_resolution,
    wavelength,
)
from ekfsim.vehicle import derived_values, pitch_end_time, validate_vehicle
from ekfsim_ui import fmt, theme
from ekfsim_ui.fmt import DASH, RDOT
from ekfsim_ui.theme import C
from ekfsim_ui.timeline import RunRequest, Timeline
from ekfsim_ui.widgets import Collapsible, TextTable, eval_tag, label, scroll, section_header

UPDATE_RATE_STEPS = (5, 10, 20, 25, 50, 100)  # scan interval as a whole multiple n of dt_truth [-] (THEORY 4.6, 5.1)
MASS_STEP = 50  # mass slider step in [kg]. DESIGN CHOICE


def _spin(value, lo, hi, decimals, step, suffix, tip="") -> QDoubleSpinBox:
    s = QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setDecimals(decimals)
    s.setSingleStep(step)
    s.setSuffix(f" {suffix}" if suffix else "")
    s.setValue(value)
    s.setAlignment(Qt.AlignmentFlag.AlignRight)
    s.setKeyboardTracking(False)
    s.setCorrectionMode(QDoubleSpinBox.CorrectionMode.CorrectToNearestValue)
    _no_wheel_unless_focused(s)
    if tip:
        s.setToolTip(tip)
    return s


class _WheelGuard(QObject):
    """lets the panel scroll under the mouse: a spin box or list takes wheel steps only when focused."""

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.Wheel and not obj.hasFocus():
            event.ignore()
            return True
        return False


_WHEEL_GUARD = None


def _no_wheel_unless_focused(w: QWidget) -> None:
    global _WHEEL_GUARD
    if _WHEEL_GUARD is None:
        _WHEEL_GUARD = _WheelGuard()
    w.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
    w.installEventFilter(_WHEEL_GUARD)
    # take the width the form gives (down to 5 em), not the width of the widest value in the range
    w.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
    w.setMinimumWidth(round(5 * theme.em()))


def _kv(name, value, unit=""):
    return (name, [value], unit)


class _Page(QWidget):
    """one control tab: a form with section headers, inside a scroll area."""

    def __init__(self):
        super().__init__()
        self.setObjectName("panel_body")
        self.form = QFormLayout(self)
        e = theme.em()
        self.form.setContentsMargins(round(0.7 * e), round(0.5 * e), round(0.7 * e), round(0.7 * e))
        self.form.setHorizontalSpacing(round(0.6 * e))
        self.form.setVerticalSpacing(round(0.25 * e))
        self.form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.DontWrapRows)

    def header(self, text: str, evaluation: bool = False) -> None:
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, round(0.5 * theme.em()) if self.form.rowCount() else 0, 0, 0)
        head = section_header(text)
        head.setWordWrap(True)
        h.addWidget(head, 1)
        if evaluation:
            h.addWidget(eval_tag(), 0, Qt.AlignmentFlag.AlignTop)
        self.form.addRow(row)

    def row(self, name: str, w: QWidget, tip: str = "") -> QWidget:
        lab = label(name, "secondary", tooltip=tip)
        self.form.addRow(lab, w)
        if tip and not w.toolTip():
            w.setToolTip(tip)
        return w

    def wide(self, w: QWidget) -> QWidget:
        self.form.addRow(w)
        return w

    def note(self, text: str, role: str = "caption"):
        n = label(text, role, "caption")
        n.setWordWrap(True)
        return self.wide(n)


# the panel part:

class ControlPanel(QWidget):
    """all control tabs and the Run / Apply footer."""

    applyRequested = Signal(object)  # RunRequest
    layerToggled = Signal(str, bool)
    ellipsoidScaleChanged = Signal(float)
    resetViewRequested = Signal()
    speedRequested = Signal(float)
    advancedToggled = Signal(bool)
    pulseToggled = Signal(bool)
    pulseFactorChanged = Signal(float)

    def __init__(self, ui_cfg: UIConfig):
        super().__init__()
        self.setObjectName("panel")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.ui = ui_cfg
        self.constants = PhysicsConstants()
        self.atmosphere = US76Atmosphere(AtmosphereConfig(), self.constants)
        self.v0, self.s0, self.r0, self.f0, self.ir0 = (VehicleConfig(), ScenarioConfig(), RadarConfig(),
                                                        FilterConfig(), IRSensorConfig())
        self._applied: RunRequest | None = None
        self._busy = False
        self._last: Timeline | None = None

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(self._basic_group())
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.tabBar().setUsesScrollButtons(True)
        self.tabs.tabBar().setExpanding(False)
        for build, name in ((self._scenario_page, "Scenario"), (self._vehicle_page, "Vehicle"),
                            (self._radar_page, "Radar"), (self._filter_page, "Filter"), (self._view_page, "View")):
            self.tabs.addTab(scroll(build(), horizontal=True), name)
        box = QWidget()
        bl = QVBoxLayout(box)
        e = theme.em()
        bl.setContentsMargins(round(0.6 * e), 0, 0, 0)
        self.advanced = Collapsible("Advanced", self.tabs, expanded=False)
        self.advanced.toggled.connect(self.advancedToggled.emit)
        bl.addWidget(self.advanced)
        lay.addWidget(box, 1)
        self._defaults = self._widget_values()
        self._refresh()

    # the basic group part

    def _basic_group(self) -> QWidget:
        g = QWidget()
        g.setObjectName("panel_body")
        lay = QVBoxLayout(g)
        e = theme.em()
        lay.setContentsMargins(round(0.7 * e), round(0.5 * e), round(0.7 * e), round(0.4 * e))
        lay.setSpacing(round(0.3 * e))
        lay.addWidget(section_header("Run"))
        v = self.v0
        self.dry0 = v.launch_mass - v.propellant_mass  # dry mass at dry_mass_delta = 0 in [kg]
        self.mass = QSlider(Qt.Orientation.Horizontal)
        self.mass.setRange(round(self.dry0 + v.dry_mass_delta_min), round(self.dry0 + v.dry_mass_delta_max))
        self.mass.setSingleStep(MASS_STEP)
        self.mass.setPageStep(10 * MASS_STEP)
        self.mass.setValue(round(self.dry0 + v.dry_mass_delta))
        self.mass.setToolTip("dry (non-propellant) mass; propellant, thrust and Isp stay fixed, so only the mass "
                             "effect is visible (THEORY 4.1). applies on release")
        self.mass.valueChanged.connect(self._mass_moved)
        self.mass.sliderReleased.connect(self._apply)
        lay.addWidget(self.mass)
        self.mass_label = label("", "value", "mono_medium", Qt.AlignmentFlag.AlignRight)
        lay.addWidget(self.mass_label)
        self.vehicle_error = label("", "error", "caption")
        self.vehicle_error.setWordWrap(True)
        lay.addWidget(self.vehicle_error)
        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setHorizontalSpacing(round(0.6 * e))
        form.setVerticalSpacing(round(0.25 * e))
        self.speed = QComboBox()
        for sp in self.ui.playback_speeds:
            self.speed.addItem(f"×{sp:g}", sp)
        self.speed.setCurrentIndex(self.speed.findData(self.ui.default_speed))
        self.speed.setToolTip("playback speed: sim seconds per wall second, slow motion below ×1 (keys + and −)")
        self.speed.currentIndexChanged.connect(lambda i: self.speedRequested.emit(float(self.speed.itemData(i))))
        _no_wheel_unless_focused(self.speed)
        form.addRow(label("playback speed", "secondary"), self.speed)
        lay.addLayout(form)
        f = self.f0
        self.boost = QCheckBox("boost mode (from the IR flag)")
        self.boost.setChecked(f.boost_mode)
        self.boost.setToolTip("BOOST / BALLISTIC from the IR flag; off: ignore the flag and the acceleration states, "
                              "BALLISTIC throughout (THEORY 6.6)")
        self.boost.toggled.connect(self._refresh)
        lay.addWidget(self.boost)
        self.net = QCheckBox("safety net (manoeuvre noise)")
        self.net.setChecked(f.adaptive_q)
        self.net.setToolTip("adaptive process noise switched on by the window NIS (THEORY 6.6)")
        self.net.toggled.connect(self._refresh)
        lay.addWidget(self.net)
        row = QHBoxLayout()
        row.setSpacing(round(0.4 * e))
        self.run_button = QPushButton("Run")
        self.run_button.setProperty("primary", True)
        self.run_button.setToolTip("compute truth, radar and EKF for these settings, then play back")
        self.run_button.clicked.connect(self._apply)
        row.addWidget(self.run_button)
        self.reset_button = QPushButton("Reset")
        self.reset_button.setToolTip("all settings back to their defaults, and run them")
        self.reset_button.clicked.connect(self.reset_to_defaults)
        row.addWidget(self.reset_button)
        self.busy = QProgressBar()
        self.busy.setRange(0, 0)
        self.busy.setTextVisible(False)
        self.busy.hide()
        row.addWidget(self.busy, 1)
        lay.addLayout(row)
        self.state = label("", "caption", "caption")
        self.state.setWordWrap(True)
        lay.addWidget(self.state)
        self.message = label("", "error", "caption")
        self.message.setWordWrap(True)
        self.message.hide()
        lay.addWidget(self.message)
        return g

    def set_speed(self, speed: float) -> None:
        i = self.speed.findData(speed)
        if i >= 0 and i != self.speed.currentIndex():
            self.speed.blockSignals(True)
            self.speed.setCurrentIndex(i)
            self.speed.blockSignals(False)

    def _setting_widgets(self) -> list:
        found = [w for cls in (QDoubleSpinBox, QSpinBox, QComboBox, QCheckBox, QSlider) for w in self.findChildren(cls)]
        return [w for w in found if w is not self.speed and w.property("view") is not True]

    @staticmethod
    def _value_of(w):
        if isinstance(w, QComboBox):
            return w.currentIndex()
        if isinstance(w, QCheckBox):
            return w.isChecked()
        return w.value()

    def _widget_values(self) -> dict:
        return {w: self._value_of(w) for w in self._setting_widgets()}

    def reset_to_defaults(self) -> None:
        """every run setting back to its default, then run (the view settings are left as they are)."""
        for w, v in self._defaults.items():
            w.blockSignals(True)
            if isinstance(w, QComboBox):
                w.setCurrentIndex(v)
            elif isinstance(w, QCheckBox):
                w.setChecked(v)
            else:
                w.setValue(v)
            w.blockSignals(False)
        self._refresh()
        self._apply()

    # the scenario page

    def _scenario_page(self) -> QWidget:
        p = _Page()
        s, r = self.s0, self.r0
        p.header("Launch site and radar")
        self.launch_e = p.row("launch site E", _spin(s.launch_east / 1e3, -800, 800, 1, 5, "km"))
        self.launch_n = p.row("launch site N", _spin(s.launch_north / 1e3, -800, 800, 1, 5, "km"))
        self.launch_az = p.row("launch azimuth", _spin(math.degrees(s.launch_azimuth), 0, 360, 1, 5, "°",
                                                       "compass azimuth of the launch plane, clockwise from North"))
        self.radar_e = p.row("radar E", _spin(r.east / 1e3, -800, 800, 1, 5, "km"))
        self.radar_n = p.row("radar N", _spin(r.north / 1e3, -800, 800, 1, 5, "km"))
        self.seed = QSpinBox()
        self.seed.setRange(0, 2 ** 31 - 1)
        self.seed.setValue(s.seed)
        self.seed.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.seed.setKeyboardTracking(False)
        _no_wheel_unless_focused(self.seed)
        p.row("noise seed", self.seed, "random seed of the measurement noise")
        p.header("Pre-track cue (from config)")
        self.t_cue = p.wide(TextTable(1, None, cell_chars=12))
        p.note("assumed early-warning cue of the launch site: launch-site azimuth, elevation mask + half "
               "beamwidth (THEORY 5.4)")
        p.header("Geometry of the last run", evaluation=True)
        self.t_geom = p.wide(TextTable(1, None, cell_chars=14))
        self.geom_warning = p.note("", "warning")
        for w in (self.launch_e, self.launch_n, self.launch_az, self.radar_e, self.radar_n, self.seed):
            w.valueChanged.connect(self._refresh)
        return p

    # the vehicle page

    def _vehicle_page(self) -> QWidget:
        p = _Page()
        v = self.v0
        p.header("Masses (slider in the Run group)")
        self.t_vehicle = p.wide(TextTable(1, None, cell_chars=12))
        p.header("Fixed (illustrative round values)")
        self.t_fixed = p.wide(TextTable(1, None, cell_chars=12))
        p.header("Pitch program (fixed, not tuned)")
        t_end = pitch_end_time(v)
        t = TextTable(1, None, cell_chars=12)
        t.set_rows([_kv("vertical rise t_vert", fmt.num(v.t_vertical, 1), "s"),
                    _kv("pitch rate", fmt.num(math.degrees(v.pitch_rate), 2), "°/s"),
                    _kv("final pitch", fmt.num(math.degrees(v.pitch_final), 1), "°"),
                    _kv("end of pitch-over", fmt.num(t_end, 1), "s")])
        p.wide(t)
        p.note("a generic illustrative vehicle: round, generic values, not the data of any specific system")
        return p

    # the radar page

    def _radar_page(self) -> QWidget:
        p = _Page()
        r, ir = self.r0, self.ir0
        p.header("Radar")
        p.note("representative values, not those of a specific radar (THEORY 5.1)")
        self.fc = p.row("carrier f_c", _spin(r.carrier_frequency / 1e9, 1, 40, 3, 0.5, "GHz"))
        self.dwell = p.row("dwell T_dwell", _spin(r.dwell_time * 1e3, 1, 100, 2, 1, "ms",
                                                  "coherent dwell per scan; sets the Doppler resolution"))
        self.rate = QComboBox()
        dt = ScenarioConfig().dt_truth
        for n in UPDATE_RATE_STEPS:
            T = n * dt
            self.rate.addItem(f"{1.0 / T:g} Hz  (T = {T:g} s)", T)
        self.rate.setCurrentIndex(max(0, self.rate.findData(r.update_interval)))
        _no_wheel_unless_focused(self.rate)
        self.rate.setToolTip("scan rate; the interval is a whole multiple of dt_truth so every scan lands on a "
                             "truth sample")
        p.row("update rate", self.rate)
        self.beamwidth = p.row("3 dB beamwidth", _spin(math.degrees(r.beamwidth), 0.1, 10, 2, 0.1, "°"))
        self.sigma_r = p.row("σ_r", _spin(r.sigma_range, 0.1, 1000, 1, 1, "m", "range noise std"))
        self.sigma_a = p.row("σ_az = σ_el", _spin(math.degrees(r.sigma_angle), 0.001, 2, 3, 0.01, "°",
                                                         "angle noise std"))
        self.doppler = QCheckBox("measure range-rate (Doppler)")
        self.doppler.setChecked(r.doppler_enabled)
        p.wide(self.doppler)
        self.mask = p.row("elevation mask", _spin(math.degrees(r.elevation_mask), 0, 30, 2, 0.1, "°",
                                                  "no measurement below this elevation (horizon, clutter)"))
        p.header("IR plume sensor")
        self.latency = p.row("flag latency L", _spin(ir.latency, 0, 5, 2, 0.1, "s",
                                                     "the engine on/off flag changes this long after the true event "
                                                     "(a sensor specification, known to the tracker)"))
        p.header("Derived from config")
        self.t_radar = p.wide(TextTable(1, None, cell_chars=12))
        p.header("Doppler ambiguity, information", evaluation=True)
        self.t_prf = p.wide(TextTable(1, None, cell_chars=12))
        p.note(f"from the largest |{RDOT}| of the last run and the f_c above; range-rate is assumed "
               "unambiguous (THEORY 5.3)")
        for w in (self.fc, self.dwell, self.beamwidth, self.sigma_r, self.sigma_a, self.mask, self.latency):
            w.valueChanged.connect(self._refresh)
        self.rate.currentIndexChanged.connect(self._refresh)
        self.doppler.toggled.connect(self._refresh)
        return p

    # the filter page

    def _filter_page(self) -> QWidget:
        p = _Page()
        f = self.f0
        p.header("Process noise (derived in config.py)")
        self.sigma_floor = p.row("σ_floor", _spin(f.sigma_a_floor, 0, 50, 3, 0.01, "m/s²",
                                                       "derived 0: no force left unmodelled in ballistic flight"))
        self.k_drag = p.row("k_drag", _spin(f.k_drag, 0, 2, 3, 0.01, "",
                                            "derived 0: the Cd(Mach) mismatch is covered by the ln beta random walk"))
        self.q_beta = p.row("q_β", _spin(f.q_beta, 0, 1, 5, 0.001, "1/s",
                                              "ln beta random walk, from the lag <= sigma/2 margin rule"))
        self.q_thrust = p.row("q_thrust", _spin(f.q_thrust, 0, 10000, 1, 5, "m²/s⁵",
                                                "thrust-acceleration random walk, from the same margin rule"))
        self.p_fa = p.row("p_fa per flight", _spin(f.p_fa_flight * 100, 0.01, 50, 2, 0.1, "%",
                                                   "false-alarm budget of the safety net per nominal flight"))
        p.header("Initialisation (THEORY 6.5)")
        self.sigma_vt = p.row("σ tangential velocity", _spin(f.sigma_v_tangential, 1, 10000, 0, 50, "m/s"))
        self.sigma_lb = p.row("σ ln β₀", _spin(f.sigma_ln_beta0, 0.01, 5, 2, 0.1, "",
                                                         "prior on ln beta around ln(5,000)"))
        self.sigma_a0 = p.row("σ thrust acceleration", _spin(f.sigma_thrust0, 0, 200, 1, 5, "m/s²"))
        p.note("the defaults are derived in config.py, not tuned; changing them is for demonstration")
        for w in (self.sigma_floor, self.k_drag, self.q_beta, self.q_thrust, self.p_fa, self.sigma_vt,
                  self.sigma_lb, self.sigma_a0):
            w.valueChanged.connect(self._refresh)
        return p

    # the view page

    def _view_page(self) -> QWidget:
        p = _Page()
        p.header("Layers")
        for key, text in (("truth", "truth (evaluation – uses truth)"),
                          ("estimate", "EKF estimate and trajectory"),
                          ("measurements", "measurements (display only)"),
                          ("beam", "radar beam"), ("ellipsoid", "3σ position ellipsoid"),
                          ("atmosphere", "ISA atmosphere layers")):
            cb = QCheckBox(text)
            cb.setProperty("view", True)
            if key == "measurements":
                cb.setToolTip("radar measurements converted to ENU for display only")
            cb.setChecked(True)
            cb.toggled.connect(lambda on, key=key: self.layerToggled.emit(key, on))
            p.wide(cb)
        p.header("Ellipsoid magnification")
        box = QWidget()
        h = QHBoxLayout(box)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(round(0.3 * theme.em()))
        group = QButtonGroup(box)
        group.setExclusive(True)
        self.scale_buttons = {}
        for s in self.ui.ellipsoid_scales:
            b = QToolButton()
            b.setText(f"×{s:g}")
            b.setCheckable(True)
            b.setChecked(s == self.ui.default_ellipsoid_scale)
            b.setFont(theme.font("mono"))
            b.clicked.connect(lambda _=False, s=s: self.ellipsoidScaleChanged.emit(s))
            group.addButton(b)
            h.addWidget(b)
            self.scale_buttons[s] = b
        h.addStretch(1)
        p.wide(box)
        p.note("the factor is printed in the 3D view (click it there to cycle); at ×1 a 50 m ellipsoid is far "
               "below one pixel")
        p.header("Slow-motion pulse view")
        self.pulse = QCheckBox("pulse view (P)")
        self.pulse.setProperty("view", True)
        self.pulse.setToolTip("one radar pulse travels out and back at its true length; time is slowed by the "
                              "factor below, which the 3D view prints")
        self.pulse.toggled.connect(self.pulseToggled.emit)
        p.wide(self.pulse)
        self.pulse_factor = QComboBox()
        self.pulse_factor.setProperty("view", True)
        for f in self.ui.pulse_factors:
            self.pulse_factor.addItem(f"time ×1/{f:,.0f}", f)
        self.pulse_factor.setCurrentIndex(self.pulse_factor.findData(self.ui.default_pulse_factor))
        self.pulse_factor.currentIndexChanged.connect(
            lambda i: self.pulseFactorChanged.emit(float(self.pulse_factor.itemData(i))))
        _no_wheel_unless_focused(self.pulse_factor)
        p.row("slow-motion factor", self.pulse_factor)
        p.note("even a 1,000 km round trip lasts less than 7 ms, so the pulse is only visible in slow motion")
        p.header("Camera")
        reset = QPushButton("Reset view")
        reset.setToolTip("frame the whole trajectory again (R)")
        reset.clicked.connect(self.resetViewRequested.emit)
        p.wide(reset)
        return p

    def set_ellipsoid_scale(self, scale: float) -> None:
        b = self.scale_buttons.get(scale)
        if b is not None:
            b.setChecked(True)

    # the request part

    def request(self) -> RunRequest:
        """the run the controls describe, in SI."""
        v = dataclasses.replace(self.v0, dry_mass_delta=float(self.mass.value() - self.dry0))
        s = dataclasses.replace(self.s0, launch_east=self.launch_e.value() * 1e3, launch_north=self.launch_n.value() * 1e3,
                                launch_azimuth=math.radians(self.launch_az.value() % 360.0), seed=int(self.seed.value()))
        r = dataclasses.replace(self.r0, east=self.radar_e.value() * 1e3, north=self.radar_n.value() * 1e3,
                                carrier_frequency=self.fc.value() * 1e9, update_interval=float(self.rate.currentData()),
                                dwell_time=self.dwell.value() / 1e3, beamwidth=math.radians(self.beamwidth.value()),
                                sigma_range=self.sigma_r.value(), sigma_angle=math.radians(self.sigma_a.value()),
                                elevation_mask=math.radians(self.mask.value()), doppler_enabled=self.doppler.isChecked())
        f = dataclasses.replace(self.f0, sigma_a_floor=self.sigma_floor.value(), k_drag=self.k_drag.value(),
                                q_beta=self.q_beta.value(), q_thrust=self.q_thrust.value(),
                                boost_mode=self.boost.isChecked(), adaptive_q=self.net.isChecked(),
                                p_fa_flight=self.p_fa.value() / 100.0, sigma_v_tangential=self.sigma_vt.value(),
                                sigma_ln_beta0=self.sigma_lb.value(), sigma_thrust0=self.sigma_a0.value())
        ir = dataclasses.replace(self.ir0, latency=self.latency.value())
        return RunRequest(v, s, r, f, ir)

    def validation_error(self, req: RunRequest) -> str:
        """why a request cannot run, or "" when it can."""
        try:
            validate_vehicle(req.vehicle, self.constants, self.atmosphere)
            validate_radar(req.radar)
        except ValueError as e:
            return str(e)
        return ""

    def _apply(self) -> None:
        req = self.request()
        err = self.validation_error(req)
        if err:
            self._refresh()
            return
        self.applyRequested.emit(req)

    def mark_applied(self, req: RunRequest) -> None:
        self._applied = req
        self._refresh()

    def set_busy(self, busy: bool) -> None:
        self._busy = busy
        self.busy.setVisible(busy)
        self.state.setVisible(not busy)
        self._refresh()

    def set_last_run(self, tl: Timeline) -> None:
        self._last = tl
        self._refresh()

    def show_error(self, text: str) -> None:
        self.message.setText(text)
        self.message.setVisible(bool(text))

    # the live derived values part

    def _mass_moved(self, value: int) -> None:
        self._refresh()

    def _refresh(self, *_) -> None:
        req = self.request()
        v = req.vehicle
        d = derived_values(v, self.constants, self.atmosphere)
        self.mass_label.setText(f"dry {d.dry_mass:,.0f} kg  ·  launch {d.launch_mass:,.0f} kg")
        self.t_vehicle.set_rows([_kv("dry mass", fmt.num(d.dry_mass, 0), "kg"),
                                 _kv("launch mass m₀", fmt.num(d.launch_mass, 0), "kg"),
                                 _kv("burnout mass", fmt.num(d.burnout_mass, 0), "kg"),
                                 _kv("initial T/W", fmt.num(d.thrust_to_weight, 3), "−")],
                                cell_colours={(3, 0): C.LOST} if d.thrust_to_weight <= v.min_thrust_to_weight else {})
        self.t_fixed.set_rows([_kv("propellant m_prop", fmt.num(v.propellant_mass, 0), "kg"),
                               _kv("thrust T_sl", fmt.num(v.thrust_sl / 1e3, 1), "kN"),
                               _kv("Isp_sl", fmt.num(v.isp_sl, 0), "s"),
                               _kv("mass flow", fmt.num(d.mass_flow, 2), "kg/s"),
                               _kv("burn time t_burn", fmt.num(d.burn_time, 2), "s"),
                               _kv("reference area A", fmt.num(d.reference_area, 4), "m²")])
        err = self.validation_error(req)
        verr = ""
        try:
            validate_vehicle(v, self.constants, self.atmosphere)
        except ValueError as e:
            verr = str(e)
        self.vehicle_error.setText(verr)
        self.vehicle_error.setVisible(bool(verr))

        az, el = cue_direction(req.radar, req.scenario)
        self.t_cue.set_rows([_kv("cue azimuth", fmt.num(math.degrees(az), 2), "°"),
                             _kv("cue elevation", fmt.num(math.degrees(el), 2), "°")])

        r = req.radar
        lam = wavelength(r.carrier_frequency, self.constants.c)
        sig_rdot = velocity_resolution(lam, r.dwell_time)
        self.t_radar.set_rows([_kv("wavelength λ", fmt.num(lam * 100, 2), "cm"),
                               _kv("Δf = 1/T_dwell", fmt.num(1.0 / r.dwell_time, 1), "Hz"),
                               _kv(f"σ_{RDOT} = λ/(2 T_dwell)", fmt.num(sig_rdot, 3), "m/s"),
                               _kv("pulse length cτ", fmt.num(self.constants.c * r.pulse_width, 1), "m"),
                               _kv("measurement dimension", "4" if r.doppler_enabled else "3", "")])
        tl = self._last
        if tl is not None:
            fd = abs(doppler_shift(tl.max_abs_rdot, lam))  # max |f_d| at the f_c above, in [Hz]
            prf = required_prf(fd)
            self.t_prf.set_rows([_kv(f"max |{RDOT}| (last run)", fmt.num(tl.max_abs_rdot, 1), "m/s"),
                                 _kv("|f_d|max", fmt.num(fd / 1e3, 2), "kHz"),
                                 _kv("PRF needed >", fmt.num(prf / 1e3, 2), "kHz"),
                                 _kv("unambiguous range", fmt.num(unambiguous_range(prf, self.constants.c) / 1e3, 3), "km")])
            lo, hi = tl.slant_range_km
            rows = [_kv("min horizontal distance", fmt.num(tl.min_horizontal_km, 2), "km"),
                    _kv("slant range", f"{fmt.num(lo, 1)}–{fmt.num(hi, 1)}", "km"),
                    _kv("max elevation", fmt.num(math.degrees(tl.max_elevation), 2), "°"),
                    _kv("at t", fmt.num(tl.max_elevation_time, 1), "s")]
            if tl.summary is not None:
                rows += [_kv("apogee", fmt.num(tl.summary.apogee_altitude / 1e3, 2), "km"),
                         _kv("flight time", fmt.num(tl.summary.flight_time, 1), "s")]
            self.t_geom.set_rows(rows, cell_colours={(2, 0): C.COASTING} if tl.elevation_warning else {})
            self.geom_warning.setText(f"maximum elevation above {math.degrees(self.ui.max_elevation_warning):.0f}°: "
                                      "azimuth is poorly conditioned near overhead" if tl.elevation_warning else "")
            self.geom_warning.setVisible(tl.elevation_warning)
        else:
            self.t_prf.set_rows([_kv("after the first run", DASH)])
            self.t_geom.set_rows([_kv("after the first run", DASH)])
            self.geom_warning.hide()

        dirty = self._applied is None or req != self._applied
        self.run_button.setEnabled(not err and not self._busy)
        self.run_button.setText("Apply" if dirty and self._applied is not None else "Run")
        self.run_button.setProperty("dirty", bool(dirty and self._applied is not None))
        self.run_button.style().unpolish(self.run_button)
        self.run_button.style().polish(self.run_button)
        self.state.setText("settings differ from the run shown" if dirty and self._applied is not None
                           else ("computing…" if self._busy else ""))
        self.state.setVisible(bool(self.state.text()))
        self.state.setStyleSheet(f"color: {C.COASTING};" if dirty and self._applied is not None else "")
        self.show_error(err)
