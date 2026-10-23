"""the cursor-linked plots (THEORY 8.4). three core tabs answer "how good is the track now": position error
+-3 sigma, NIS and beta; velocity error, innovations, truth kinematics and the Cd(Mach) table sit under one
"More" tab. every time axis shows the same time range, a vertical line follows the
playback cursor, and a click into a time plot seeks to that time. by default the time axis is a window that
follows the cursor and the y-axes fit the data inside it, so the ~15 m ballistic error is readable after the
boost; "Full" shows the whole flight. the tabs that read ground truth carry the
"evaluation - uses truth" tag. colours follow the 3D view: estimate, covariance (bands from P), truth,
measurements; chi2 bands that do not come from P use the neutral bound colour, and the scans the filter
spent in BOOST mode are shaded in the BOOST colour (from the IR flag, not from truth)."""

import math

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QButtonGroup, QGraphicsItem, QHBoxLayout, QTabWidget, QToolButton, QVBoxLayout, QWidget

from ekfsim.config import UIConfig, VehicleConfig
from ekfsim_ui import theme
from ekfsim_ui.fmt import DEG, RDOT
from ekfsim_ui.theme import C, SANS
from ekfsim_ui.timeline import Timeline
from ekfsim_ui.widgets import FlowLayout, Swatch, eval_tag, label

TIME_LABEL = "time t [s]"


def _pen(colour, width=1.2, alpha=1.0, style=Qt.PenStyle.SolidLine):
    return pg.mkPen(theme.qcolor(colour, alpha), width=width, style=style, cosmetic=True)


def _boost_spans(t: np.ndarray, mode: np.ndarray) -> list[tuple[float, float]]:
    """time spans of consecutive BOOST scans of the filter, in [s]."""
    b = np.asarray(mode) == "BOOST"
    if not np.any(b):
        return []
    edges = np.flatnonzero(np.diff(np.r_[0, b.astype(int), 0]))
    return [(float(t[s]), float(t[min(e, len(t) - 1)])) for s, e in zip(edges[0::2], edges[1::2])]


# the single plot part:

class _Tab(QWidget):
    """one plot tab: a header (title, legend, evaluation tag) above a pyqtgraph layout."""

    def __init__(self, title: str, legend: list, evaluation: bool, note: str = ""):
        super().__init__()
        self.setObjectName("panel")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        lay = QVBoxLayout(self)
        e = theme.em()
        lay.setContentsMargins(round(0.5 * e), round(0.4 * e), round(0.3 * e), 0)
        lay.setSpacing(round(0.2 * e))
        head = QHBoxLayout()
        head.setSpacing(round(0.5 * e))
        title_label = label(title, "secondary", "body_medium")
        title_label.setWordWrap(True)  # a long title wraps in a narrow column instead of running under the tag
        head.addWidget(title_label, 1)
        if evaluation:
            head.addWidget(eval_tag())
        lay.addLayout(head)
        if legend:
            box = QWidget()
            flow = FlowLayout(box, spacing=0.6 * theme.em("caption"))
            flow.setContentsMargins(0, 0, 0, 0)
            for colour, kind, text in legend:
                item = QWidget()
                h = QHBoxLayout(item)
                h.setContentsMargins(0, 0, 0, 0)
                h.setSpacing(round(0.25 * e))
                h.addWidget(Swatch(colour, kind))
                h.addWidget(label(text, "caption", "caption"))
                flow.addWidget(item)
            lay.addWidget(box)
        if note:
            n = label(note, "caption", "caption")
            n.setWordWrap(True)
            lay.addWidget(n)
        self.glw = pg.GraphicsLayoutWidget()
        self.glw.setBackground(C.SURFACE)
        self.glw.ci.setContentsMargins(0, 0, 0, 0)
        self.glw.ci.setSpacing(round(0.4 * e))
        lay.addWidget(self.glw, 1)
        self.plots: list[pg.PlotItem] = []
        self.cursors: list[pg.InfiniteLine] = []
        self.boost_regions: dict[pg.PlotItem, list] = {}

    def add_plot(self, row: int, col: int, ylabel: str, bottom: bool, colspan: int = 1,
                 log_y: bool = False, cursor: bool = True) -> pg.PlotItem:
        p = self.glw.addPlot(row=row, col=col, colspan=colspan)
        p.setMenuEnabled(False)
        p.hideButtons()
        tick_font = theme.font("mono_caption")
        css = {"color": C.INK_2, "font-size": "7.5pt", "font-family": SANS}
        for name in ("left", "bottom"):
            ax = p.getAxis(name)
            ax.setPen(_pen(C.HAIRLINE, 1.0))
            ax.setTextPen(_pen(C.INK_3, 1.0))
            ax.setStyle(tickFont=tick_font, tickTextOffset=4, tickLength=-4)
            ax.enableAutoSIPrefix(False)
        p.getAxis("left").setLabel(ylabel, **css)
        p.getAxis("left").setWidth(round(4.6 * theme.em("caption")))
        if bottom:
            p.getAxis("bottom").setLabel(TIME_LABEL if cursor else "", **css)
        else:
            p.getAxis("bottom").setStyle(showValues=False)
        p.showGrid(x=True, y=True, alpha=0.5)
        if log_y:
            p.setLogMode(y=True)
            p.getAxis("left").setStyle(maxTextLevel=0)  # label the decades only
        p.vb.setAutoVisible(y=True)
        p.setClipToView(True)
        p.setDownsampling(auto=True, mode="peak")
        self.plots.append(p)
        if cursor:
            line = pg.InfiniteLine(angle=90, movable=False, pen=_pen(C.ACCENT, 1.0, 0.85))
            line.setZValue(1000)
            p.addItem(line, ignoreBounds=True)
            self.cursors.append(line)
        return p

    def set_boost(self, spans: list[tuple[float, float]]) -> None:
        for p in self.plots:
            for r in self.boost_regions.get(p, []):
                p.removeItem(r)
            regions = []
            for a, b in spans:
                r = pg.LinearRegionItem(values=(a, b), movable=False, brush=theme.qcolor(C.BOOST, 0.11),
                                        pen=pg.mkPen(None))
                r.setZValue(-100)
                p.addItem(r, ignoreBounds=True)
                regions.append(r)
            self.boost_regions[p] = regions


def _band(p: pg.PlotItem, colour: str, alpha_fill=0.16, alpha_edge=0.55, style=Qt.PenStyle.SolidLine):
    lo = p.plot(pen=_pen(colour, 1.0, alpha_edge, style))
    hi = p.plot(pen=_pen(colour, 1.0, alpha_edge, style))
    fill = pg.FillBetweenItem(lo, hi, brush=theme.qcolor(colour, alpha_fill))
    fill.setZValue(-10)
    p.addItem(fill, ignoreBounds=True)  # no dataBounds: its rect would span the whole flight; the edges bound it
    return lo, hi


def _symmetric_band(p: pg.PlotItem, colour: str, alpha_fill=0.16, alpha_edge=0.55):
    """a band -b ... +b around zero (+-3 sigma, +-3 sqrt S): each edge curve fills to zero itself. this is
    the same band as a fill between the edges, drawn from the curves' own view-clipped data, so a moving time
    window does not rebuild a fill path over the whole flight every frame."""
    brush = theme.qcolor(colour, alpha_fill)
    lo = p.plot(pen=_pen(colour, 1.0, alpha_edge), fillLevel=0.0, fillBrush=brush)
    hi = p.plot(pen=_pen(colour, 1.0, alpha_edge), fillLevel=0.0, fillBrush=brush)
    for c in (lo, hi):
        c.setZValue(-10)
    return lo, hi


def _set_band(band, t, lo, hi) -> None:
    band[0].setData(t, lo, connect="finite")
    band[1].setData(t, hi, connect="finite")


# the panel part:

class PlotPanel(QTabWidget):
    """all plot tabs. set_timeline() loads a run; set_cursor() moves the cursor of the visible tab."""

    seekRequested = Signal(float)

    def __init__(self, ui_cfg: UIConfig | None = None):
        super().__init__()
        self.ui = ui_cfg or UIConfig()
        self.setObjectName("panel")
        self.setDocumentMode(True)
        self.tabBar().setUsesScrollButtons(True)
        self.tabBar().setElideMode(Qt.TextElideMode.ElideNone)
        self._cursor = (0.0, math.nan)
        self.tl: Timeline | None = None
        self.follow = True  # time axis: a window that follows the cursor (True) or the full flight
        self._time_plots: list[pg.PlotItem] = []
        self.more = QTabWidget()
        self.more.setObjectName("panel")
        self.more.setDocumentMode(True)
        self.more.tabBar().setUsesScrollButtons(True)
        self._build_error("pos", "Position error per axis, x̂ − x with ±3σ from P", "m")
        self._build_nis()
        self._build_beta()
        self._build_error("vel", "Velocity error per axis, v̂ − v with ±3σ from P", "m/s", inner=True)
        self._build_innovations()
        self._build_kinematics()
        self._build_aero()
        self.setTabToolTip(self.addTab(self.more, "More"), "velocity error, innovations, truth kinematics, Cd(Mach)")
        self._build_window_switch()
        self._syncing = False
        for p in self._time_plots:
            p.sigXRangeChanged.connect(lambda _vb, rng, p=p: self._sync_x(p, rng))
        self._loaders = {self.nis_tab: self._load_nis, self.pos_tab: lambda *a: self._load_errors("pos", *a),
                         self.vel_tab: lambda *a: self._load_errors("vel", *a), self.beta_tab: self._load_beta,
                         self.innov_tab: self._load_innovations, self.kin_tab: self._load_kinematics,
                         self.aero_tab: self._load_aero}
        self._dirty = set()
        self.currentChanged.connect(self._tab_changed)
        self.more.currentChanged.connect(self._tab_changed)

    def current_tab(self):
        """the plot tab on screen (inside "More" when that is open)."""
        w = self.currentWidget()
        return self.more.currentWidget() if w is self.more else w

    def _tab_changed(self, *_):
        tab = self.current_tab()
        self._load(tab)
        if self.follow:
            self._apply_window(self._cursor[0])
        self._apply_cursor()

    def _build_window_switch(self):
        box = QWidget()
        h = QHBoxLayout(box)
        h.setContentsMargins(0, 0, round(0.3 * theme.em()), 0)
        h.setSpacing(0)
        group = QButtonGroup(box)
        group.setExclusive(True)
        self.window_buttons = {}
        for key, text, tip in ((True, f"{self.ui.plot_window:g} s", "time axis follows the cursor over the last "
                                f"{self.ui.plot_window:g} s; the y-axes fit the data in that window"),
                               (False, "Full", "time axis shows the whole flight")):
            b = QToolButton()
            b.setText(text)
            b.setCheckable(True)
            b.setChecked(key == self.follow)
            b.setToolTip(tip)
            b.setFont(theme.font("caption"))
            b.clicked.connect(lambda _=False, key=key: self.set_follow(key))
            group.addButton(b)
            h.addWidget(b)
            self.window_buttons[key] = b
        self.setCornerWidget(box, Qt.Corner.TopRightCorner)

    def set_follow(self, on: bool) -> None:
        """switch the time axis between the cursor window and the full flight."""
        self.follow = on
        self.window_buttons[on].setChecked(True)
        for tab in self._loaders:
            if tab not in self._dirty:
                self._cache_items(tab)
        if on:
            self._apply_window(self._cursor[0])
        elif self.tl is not None:
            self._time_plots[0].setXRange(0.0, self.tl.t_end, padding=0.01)

    def window_range(self, t: float) -> tuple[float, float]:
        """the time window for cursor time t [s]: it pages forward in steps, like an oscilloscope, so between two
        steps only the cursor moves and the plots keep their cached pixmaps. the cursor stays inside because
        lead W >= step.
        lo = floor((t - (1 - lead) W) / step) step, moved inside [0, t_end - W] - window start
        [lo, lo + W] - time window of width W"""
        W, step = self.ui.plot_window, self.ui.plot_window_step
        lo = math.floor((t - (1.0 - self.ui.plot_window_lead) * W) / step) * step
        lo = min(max(lo, 0.0), max(self.tl.t_end - W, 0.0))
        return lo, lo + W

    def _apply_window(self, t: float) -> None:
        """set the cursor window on the visible tab only (the others take it when shown), and only when it moved."""
        if self.tl is None or not math.isfinite(t):
            return
        lo, hi = self.window_range(t)
        self._syncing = True
        try:
            for p in self.current_tab().plots:
                if p in self._time_plots and p.vb.viewRange()[0] != [lo, hi]:
                    p.setXRange(lo, hi, padding=0.0)
        finally:
            self._syncing = False

    def _sync_x(self, source: pg.PlotItem, rng) -> None:
        """give every time plot the same time range. pyqtgraph's setXLink aligns linked views on screen,
        which shifts the range of plots that sit side by side, so the ranges are copied exactly instead."""
        if self._syncing:
            return
        self._syncing = True
        try:
            for p in self._time_plots:
                if p is not source:
                    p.setXRange(float(rng[0]), float(rng[1]), padding=0.0)
        finally:
            self._syncing = False

    def _register(self, tab: _Tab, name: str, tooltip: str, inner: bool = False) -> None:
        bar = self.more if inner else self
        i = bar.addTab(tab, name)
        bar.setTabToolTip(i, tooltip)
        tab.glw.scene().sigMouseClicked.connect(lambda ev, tab=tab: self._clicked(ev, tab))

    def _clicked(self, ev, tab: _Tab) -> None:
        if ev.button() != Qt.MouseButton.LeftButton or ev.double():
            return
        for p in tab.plots:
            if p in self._time_plots and p.vb.sceneBoundingRect().contains(ev.scenePos()):
                self.seekRequested.emit(float(p.vb.mapSceneToView(ev.scenePos()).x()))
                return

    # the tabs

    def _build_nis(self):
        tab = _Tab("Normalised innovation squared εν = ν^T S^-1 ν", [
            (C.ESTIMATE, "dot", "NIS per update"),
            (C.ESTIMATE, "line", "window mean (N_w updates)"),
            (C.BOUND, "band", "χ²(m) 95 % interval of one NIS"),
            (C.COASTING, "dash", "safety-net threshold of the window mean"),
            (C.BEAM_MISS, "line", "missed detection (tick)"),
            (C.BOOST, "band", "BOOST mode")], evaluation=False)
        tab.glw.ci.layout.setRowStretchFactor(0, 1)
        p = tab.add_plot(0, 0, "NIS [−]", bottom=True)
        self.nis_band = _band(p, C.BOUND, 0.14, 0.6, Qt.PenStyle.DashLine)
        self.nis_net = p.plot(pen=_pen(C.COASTING, 1.0, 0.8, Qt.PenStyle.DashLine))
        self.nis_pts = p.plot(pen=None, symbol="o", symbolSize=3, symbolPen=None,
                              symbolBrush=theme.qcolor(C.ESTIMATE, 0.55))
        self.nis_win = p.plot(pen=_pen(C.ESTIMATE, 1.4))
        self.nis_miss = p.plot(pen=None, symbol="|", symbolSize=7, symbolPen=_pen(C.BEAM_MISS, 1.0),
                               symbolBrush=None)
        p.vb.setAutoVisible(y=False)
        self.nis_plot = p
        self._time_plots.append(p)
        self.nis_tab = tab
        self._register(tab, "NIS", "NIS with its chi2 95 % band; misses marked (THEORY 8.4)")

    def _build_error(self, key, title, unit, inner=False):
        tab = _Tab(title, [(C.ESTIMATE, "line", "error"), (C.COVARIANCE, "band", "±3σ from P_{k|k}"),
                           (C.BOOST, "band", "BOOST mode")], evaluation=True)
        curves, bands = [], []
        for r, axis in enumerate(("E", "N", "U")):
            p = tab.add_plot(r, 0, f"e {axis} [{unit}]", bottom=(r == 2))
            bands.append(_symmetric_band(p, C.COVARIANCE))
            curves.append(p.plot(pen=_pen(C.ESTIMATE, 1.2)))
            p.addItem(pg.InfiniteLine(pos=0.0, angle=0, pen=_pen(C.INK_3, 1.0, 0.5)), ignoreBounds=True)
            self._time_plots.append(p)
        setattr(self, f"{key}_curves", curves)
        setattr(self, f"{key}_bands", bands)
        setattr(self, f"{key}_tab", tab)
        name = "Position" if key == "pos" else "Velocity"
        self._register(tab, name, f"{name.lower()} error per axis with ±3σ from P (evaluation – uses truth)", inner)

    def _build_beta(self):
        tab = _Tab("Ballistic coefficient β, estimate ±3σ against truth", [
            (C.ESTIMATE, "line", "β̂ = exp(ln β)"), (C.COVARIANCE, "band", "exp(ln β ± 3σ)"),
            (C.TRUTH, "dash", "truth β = m / (C_d A)"), (C.BOOST, "band", "BOOST mode: ln β frozen")],
            evaluation=True)
        p = tab.add_plot(0, 0, "β [kg/m²]", bottom=True, log_y=True)
        self.beta_band = _band(p, C.COVARIANCE)
        self.beta_est = p.plot(pen=_pen(C.ESTIMATE, 1.4))
        self.beta_truth = p.plot(pen=_pen(C.TRUTH, 1.3, 0.95, Qt.PenStyle.DashLine))
        self._time_plots.append(p)
        self.beta_tab = tab
        self._register(tab, "β", "beta estimate ±3σ against truth beta(t) (evaluation – uses truth)")

    def _build_innovations(self):
        tab = _Tab("Innovations ν = z − h(x̂_{k|k-1}) with ±3√S_ii", [
            (C.MEASUREMENT, "dot", "innovation per update"), (C.COVARIANCE, "band", "±3√S_ii"),
            (C.BOOST, "band", "BOOST mode")], evaluation=False)
        self.innov = []
        for r, (name, unit) in enumerate((("r", "m"), ("az", "°"), ("el", "°"), (RDOT, "m/s"))):
            p = tab.add_plot(r, 0, f"ν {name} [{unit}]", bottom=(r == 3))
            band = _symmetric_band(p, C.COVARIANCE, 0.12)
            pts = p.plot(pen=None, symbol="o", symbolSize=3, symbolPen=None,
                         symbolBrush=theme.qcolor(C.MEASUREMENT, 0.6))
            self.innov.append((p, band, pts))
            self._time_plots.append(p)
        self.innov_tab = tab
        self._register(tab, "Innovations", "innovations per measurement component with ±3√S_ii", True)

    def _build_kinematics(self):
        tab = _Tab("Truth kinematics", [(C.TRUTH, "line", "truth"), (C.BOOST, "band", "BOOST mode (filter)")],
                   evaluation=True)
        panels = (("altitude h [km]", False), ("speed |v| [m/s]", False), ("Mach [−]", False),
                ("mass m [kg]", False), ("thrust |T| [kN]", False), ("drag |D| [kN]", False),
                ("density ρ [kg/m³]", True))
        self.kin = []
        for j, (ylabel, log) in enumerate(panels):
            row, col = divmod(j, 2)
            last = j == len(panels) - 1
            p = tab.add_plot(row, col, ylabel, bottom=(row >= 2), colspan=2 if last else 1, log_y=log)
            self.kin.append(p.plot(pen=_pen(C.TRUTH, 1.2)))
            self._time_plots.append(p)
        self.kin_tab = tab
        self._register(tab, "Kinematics", "truth altitude, speed, Mach, mass, thrust, drag and density "
                                          "(evaluation – uses truth)", True)

    def _build_aero(self):
        tab = _Tab("Drag coefficient table C_d(Mach), ESTIMATE", [
            (C.TRUTH, "line", "piecewise-linear, held beyond the ends"), (C.TRUTH, "dot", "table points (textbook slender-body shape)"),
            (C.ACCENT, "line", "cursor: current truth Mach")], evaluation=True,
            note="the cursor here is the Mach number at the playback time, not a time; Mach is not monotonic "
                 "in time, so a click does not seek in this plot")
        p = tab.add_plot(0, 0, "C_d [−]", bottom=True, cursor=True)
        p.getAxis("bottom").setLabel("Mach [−]", color=C.INK_2, **{"font-size": "7.5pt", "font-family": SANS})
        p.vb.setAutoVisible(y=False)
        v = VehicleConfig()
        mach = np.array(v.cd_mach)
        self.aero_line = p.plot(np.r_[mach, mach[-1] * 1.25], np.r_[v.cd_values, v.cd_values[-1]],
                                pen=_pen(C.TRUTH, 1.4))
        self.aero_pts = p.plot(mach, v.cd_values, pen=None, symbol="o", symbolSize=5, symbolPen=None,
                               symbolBrush=theme.qcolor(C.TRUTH))
        self.aero_tab = tab
        self._register(tab, "Aero", "the Cd(Mach) table of the truth model (evaluation – uses truth)", True)

    # the data part

    def set_timeline(self, tl: Timeline) -> None:
        """take a new run. only the visible tab is loaded now; the others load when first shown, so swapping
        in a run does not stall playback for the fill paths of all seven tabs."""
        self.tl = tl
        self._dirty = set(self._loaders)
        self._load(self.current_tab())
        if self.follow:
            self._apply_window(self._cursor[0])
        else:
            self._time_plots[0].setXRange(0.0, tl.t_end, padding=0.01)
        self._apply_cursor()

    def _load(self, tab) -> None:
        if self.tl is None or tab not in self._dirty:
            return
        self._dirty.discard(tab)
        tl = self.tl
        t = np.asarray(tl.t_scan)
        k0 = tl.k0 if tl.k0 is not None else len(t)
        if tab is not self.aero_tab:
            tab.set_boost(_boost_spans(t, tl.mode))
        self._loaders[tab](tl, t, k0, t[k0:], np.asarray(tl.updated))
        for p in tab.plots:
            if p is not self.nis_plot and p in self._time_plots:
                p.enableAutoRange(axis="y")
        self._cache_items(tab)

    def _load_nis(self, tl, t, k0, tt, upd):
        lo, hi = tl.nis_bounds
        _set_band(self.nis_band, np.array([0.0, tl.t_end]), np.full(2, lo), np.full(2, hi))
        self.nis_pts.setData(t[upd], np.asarray(tl.nis)[upd])
        self.nis_win.setData(tt, np.asarray(tl.window_nis)[k0:], connect="finite")
        self.nis_net.setData(tt, np.asarray(tl.net_threshold)[k0:], connect="finite")
        miss = ~np.asarray(tl.detected)
        miss[:k0] = False
        self.nis_miss.setData(t[miss], np.zeros(int(miss.sum())))
        self.nis_plot.setYRange(0.0, 2.2 * hi, padding=0.0)

    def _load_errors(self, key, tl, t, k0, tt, upd):
        err, sig = (tl.err_pos, tl.sig_pos) if key == "pos" else (tl.err_vel, tl.sig_vel)
        for a in range(3):
            getattr(self, f"{key}_curves")[a].setData(tt, np.asarray(err)[k0:, a], connect="finite")
            s = 3.0 * np.asarray(sig)[k0:, a]
            _set_band(getattr(self, f"{key}_bands")[a], tt, -s, s)

    def _load_beta(self, tl, t, k0, tt, upd):
        if tl.has_track:
            lb, sb = np.asarray(tl.ln_beta)[k0:], np.asarray(tl.sigma_ln_beta)[k0:]
            _set_band(self.beta_band, tt, np.exp(lb - 3 * sb), np.exp(lb + 3 * sb))
            self.beta_est.setData(tt, np.exp(lb))
            self.beta_truth.setData(tt, np.asarray(tl.truth_beta_scan)[k0:])
        else:
            for c in (*self.beta_band, self.beta_est, self.beta_truth):
                c.setData([], [])

    def _load_innovations(self, tl, t, k0, tt, upd):
        scale = (1.0, DEG, DEG, 1.0)  # display units: m, deg, deg, m/s
        for j, (p, band, pts) in enumerate(self.innov):
            p.setVisible(j < tl.dim)
            if j >= tl.dim:
                continue
            nu = np.asarray(tl.nu)[:, j] * scale[j]
            s3 = 3.0 * np.asarray(tl.sqrt_S)[:, j] * scale[j]
            pts.setData(t[upd], nu[upd])
            _set_band(band, t[upd], -s3[upd], s3[upd])

    def _load_kinematics(self, tl, t, k0, tt, upd):
        series = (tl.truth_alt_km, tl.truth_speed, tl.truth_mach, tl.truth_mass, tl.truth_thrust_kN,
                  tl.truth_drag_kN, tl.truth_rho)
        for c, y in zip(self.kin, series):
            c.setData(np.asarray(tl.truth_t), np.asarray(y))

    def _load_aero(self, tl, t, k0, tt, upd):
        v = tl.request.vehicle
        mach = np.array(v.cd_mach)
        m_hi = max(mach[-1], float(np.max(tl.truth_mach))) * 1.05
        self.aero_line.setData(np.r_[mach, m_hi], np.r_[v.cd_values, v.cd_values[-1]])
        self.aero_pts.setData(mach, np.array(v.cd_values))

    def _cache_items(self, tab) -> None:
        """cache the data items as device pixmaps. the playback cursor moves every frame and would otherwise
        make Qt redraw every curve under it; with the cache only a zoom, a pan, a window step or new data
        redraws them."""
        mode = QGraphicsItem.CacheMode.DeviceCoordinateCache
        for p in tab.plots:
            for name in ("left", "bottom"):
                p.getAxis(name).setCacheMode(mode)  # the axes draw the grid too
            for item in p.vb.addedItems:
                if isinstance(item, pg.InfiniteLine):
                    continue
                for it in [item, *item.childItems()]:
                    it.setCacheMode(mode)

    def set_cursor(self, t: float, mach: float) -> None:
        self._cursor = (t, mach)
        if self.follow:
            self._apply_window(t)
        self._apply_cursor()

    def _apply_cursor(self) -> None:
        t, mach = self._cursor
        tab = self.current_tab()
        if tab is None:
            return
        value = mach if tab is self.aero_tab else t
        if not math.isfinite(value):
            return
        for line in tab.cursors:
            if line.value() != value:
                line.setValue(value)
