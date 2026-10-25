"""the equations panel (THEORY 8.6), the teaching core of the UI, shown as a bottom
drawer. five sections in the order of the computation: truth physics -> radar and Doppler ->
EKF predict -> EKF update -> consistency. every card shows its equations as images (matplotlib mathtext in IBM
Plex Sans, rendered once per pixel ratio, cached, and re-rendered when the window moves to a screen with another
scaling) next to the live values at the cursor: scan k for the filter, truth sample i for the physics.
numbers: 3-4 significant digits in one fixed display unit per quantity type (velocities and range-rate in
m/s, accelerations in m/s^2, absolute angles in degrees, angle innovations and spreads in mrad, positions in km
or m), never a prefix chosen per value. matrices larger than 4 x 4 (Phi, Q, P, H, K) are compact heat maps with
the exact entry and its unit in a tooltip; S shows sigma_i on its diagonal and the correlation off it; "show
values" turns them all into tables of the raw entries. the entries of these matrices have mixed units, so the
colour is a unit-free magnitude in [0, 1]:
rho_ij = P_ij / sqrt(P_ii P_jj) - correlation coefficient of a covariance (P, Q)
|K_ij| sqrt(S_jj) / sigma_i - share of the prior sigma of state i that a 1-sigma innovation j moves
|Phi_ij| sigma_j / sigma_i, |H_ij| sigma_j / sqrt(S_ii) - entries scaled by the spreads of their row and column
the header marks the step of scan k: PREDICT from t_{k-1} to t_k, then UPDATE at t_k, or "no update this scan"
with the reason on a miss, and it says when a record was re-processed at the burnout switch (THEORY 6.2)."""

import functools
import io
import math

import numpy as np
from PySide6.QtCore import QElapsedTimer, QEvent, QObject, Qt
from PySide6.QtGui import QGuiApplication, QPixmap
from PySide6.QtWidgets import QCheckBox, QFrame, QHBoxLayout, QLabel, QTabWidget, QVBoxLayout, QWidget

from ekfsim.config import UIConfig
from ekfsim.ekf import BETA_ROW, N_STATE
from ekfsim_ui import fmt, theme
from ekfsim_ui.fmt import DASH, RDOT
from ekfsim_ui.theme import MODE_COLOUR, C
from ekfsim_ui.timeline import Timeline
from ekfsim_ui.widgets import Chip, MatrixView, TextTable, eval_tag, label, scroll, section_header

EQUATION_PT = 10.0  # equation image text size in [pt] before theme.TEXT_SCALE. DESIGN CHOICE: one step above the 9 pt body for sub/superscripts
STATE_LABELS = ("pE", "pN", "pU", "vE", "vN", "vU", "aE", "aN", "aU", "lnβ")
STATE_UNITS = ("m", "m", "m", "m/s", "m/s", "m/s", "m/s²", "m/s²", "m/s²", "")
MEAS_LABELS = ("r", "az", "el", RDOT)
MEAS_UNITS = ("m", "rad", "rad", "m/s")
STATE_DISPLAY = ("pos", "pos", "pos", "m/s", "m/s", "m/s", "m/s²", "m/s²", "m/s²", "")  # fixed display units
MEAS_ABSOLUTE = ("pos", "°", "°", "m/s")  # display units of z and h(x): absolute angles in degrees
MEAS_SMALL = ("pos", "mrad", "mrad", "m/s")  # display units of innovations and spreads: angles in mrad
_UNIT_POWERS = {"m": {"m": 1}, "m/s": {"m": 1, "s": -1}, "m/s²": {"m": 1, "s": -2}, "rad": {"rad": 1}, "": {}}
_SUPERSCRIPT = {2: "²", 3: "³", 4: "⁴", 5: "⁵"}


# the unit part (for the exact values in the tooltips):

def unit_product(a: str, b: str, sign: int = 1) -> str:
    """the unit of a b (sign +1) or a / b (sign -1), e.g. m and m/s give m²/s."""
    powers = dict(_UNIT_POWERS[a])
    for k, v in _UNIT_POWERS[b].items():
        powers[k] = powers.get(k, 0) + sign * v
    num = "·".join(k + _SUPERSCRIPT.get(v, "") for k, v in sorted(powers.items()) if v > 0)
    den = "·".join(k + _SUPERSCRIPT.get(-v, "") for k, v in sorted(powers.items()) if v < 0)
    if not den:
        return num
    return f"{num or '1'}/{den}"


def unit_grid(rows, cols, sign: int = 1) -> list[list[str]]:
    return [[unit_product(r, c, sign) for c in cols] for r in rows]


# the unit-free magnitude part (heat-map colours, module docstring):

def correlation_magnitude(P) -> np.ndarray:
    """|rho_ij| = |P_ij| / sqrt(P_ii P_jj), NaN where a variance is zero (a fixed state)."""
    P = np.asarray(P, dtype=float)
    d = np.sqrt(np.clip(np.diag(P), 0.0, None))
    with np.errstate(invalid="ignore", divide="ignore"):
        N = np.abs(P) / np.outer(d, d)
    N[(d == 0.0)[:, None] | (d == 0.0)[None, :]] = np.nan
    return N


def scaled_magnitude(M, row_scale, col_scale) -> np.ndarray:
    """|M_ij| col_scale_j / row_scale_i, clipped to [0, 1], NaN where a scale is zero."""
    M = np.asarray(M, dtype=float)
    r, c = np.asarray(row_scale, float), np.asarray(col_scale, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        N = np.abs(M) * c[None, :] / r[:, None]
    N[(r == 0.0)[:, None] | (c == 0.0)[None, :]] = np.nan
    return np.clip(N, 0.0, 1.0)


# the equation image part:

_fonts_registered = False


def _register_fonts() -> None:
    global _fonts_registered
    if not _fonts_registered:
        from matplotlib import font_manager
        for path in sorted(theme.FONT_DIR.glob("IBMPlexSans-*.ttf")):
            font_manager.fontManager.addfont(str(path))
        _fonts_registered = True


@functools.lru_cache(maxsize=512)
def render_equation(tex: str, dpr: float, colour: str, size_pt: float, logical_dpi: float) -> bytes:
    """one equation as a tight PNG with a transparent background, rendered by matplotlib mathtext at the
    screen's pixel density, so 1 pt in the image is 1 pt in the UI.
    dpi = logical dpi x device pixel ratio - render density"""
    import matplotlib
    from matplotlib import mathtext
    from matplotlib.font_manager import FontProperties
    _register_fonts()
    rc = {"mathtext.fontset": "custom", "mathtext.rm": theme.SANS, "mathtext.it": f"{theme.SANS}:italic",
          "mathtext.bf": f"{theme.SANS}:semibold", "mathtext.sf": theme.SANS, "mathtext.fallback": "stixsans",
          "font.family": theme.SANS, "savefig.transparent": True}
    buf = io.BytesIO()
    with matplotlib.rc_context(rc):
        mathtext.math_to_image(tex, buf, prop=FontProperties(family=theme.SANS, size=size_pt),
                               dpi=logical_dpi * dpr, format="png", color=colour)
    return buf.getvalue()


class EquationImage(QLabel):
    """an equation that renders itself when shown and again when its device pixel ratio changes."""

    def __init__(self, tex: str, colour: str = C.INK, size_pt: float | None = None):
        super().__init__()
        self.tex, self.colour = tex, colour
        self.size_pt = size_pt or EQUATION_PT * (theme.TEXT_SCALE or 1.0)  # follows the UI text scale
        self._dpr = None
        self.setToolTip(tex.replace("$", ""))

    def _render(self) -> None:
        dpr = round(self.devicePixelRatioF(), 3)
        if dpr == self._dpr:
            return
        screen = self.screen() or QGuiApplication.primaryScreen()
        png = render_equation(self.tex, dpr, self.colour, self.size_pt, float(screen.logicalDotsPerInch()))
        pm = QPixmap()
        pm.loadFromData(png, "PNG")
        pm.setDevicePixelRatio(dpr)
        self._dpr = dpr
        self.setPixmap(pm)
        self.updateGeometry()

    def showEvent(self, event):
        self._render()
        super().showEvent(event)

    def event(self, e):
        if e.type() == QEvent.Type.DevicePixelRatioChange:
            self._dpr = None
            if self.isVisible():
                self._render()
        return super().event(e)


# the card part:

NOTE_WIDTH = {False: 15.0, True: 13.5}  # width of a note in the left and right column of a card, in em


class _Note(QLabel):
    """a wrapped caption with a fixed width and the height its text needs at that width. a word-wrapped QLabel
    otherwise reports the height of its unwrapped text, and the card would cut its last lines."""

    def __init__(self, width_em: float, role: str = "caption"):
        super().__init__()
        self.setProperty("role", role)
        self.setFont(theme.font("caption"))
        self.setWordWrap(True)
        self.setFixedWidth(round(width_em * theme.em()))

    def setText(self, text: str) -> None:
        super().setText(text)
        self.setFixedHeight(self.heightForWidth(self.width()) if text else 0)


class Card(QFrame):
    """one card: a title, the equations and scalar tables on the left, the vector or matrix on the right."""

    def __init__(self, title: str, equations: list[str], evaluation: bool = False):
        super().__init__()
        self.setProperty("role", "card")
        outer = QVBoxLayout(self)
        e = theme.em()
        outer.setContentsMargins(round(0.7 * e), round(0.45 * e), round(0.7 * e), round(0.55 * e))
        outer.setSpacing(round(0.3 * e))
        head = QHBoxLayout()
        head.setSpacing(round(0.5 * e))
        head.addWidget(label(title, "secondary", "body_medium"))  # not uppercased: h(x) is not H(X)
        head.addStretch(1)
        if evaluation:
            head.addWidget(eval_tag())
        outer.addLayout(head)
        body = QHBoxLayout()
        body.setSpacing(round(0.9 * e))
        self.left = QVBoxLayout()
        self.left.setSpacing(round(0.25 * e))
        self.right = QVBoxLayout()
        self.right.setSpacing(round(0.25 * e))
        for tex in equations:
            self.left.addWidget(EquationImage(tex))
        body.addLayout(self.left)
        body.addLayout(self.right)
        outer.addLayout(body)

    def add(self, w: QWidget, right: bool = False) -> QWidget:
        (self.right if right else self.left).addWidget(w, 0, Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        return w

    def note(self, text: str = "", right: bool = False, role: str = "caption") -> QLabel:
        n = _Note(NOTE_WIDTH[right], role)
        n.setText(text)
        return self.add(n, right)

    def finish(self) -> None:
        self.left.addStretch(1)
        self.right.addStretch(1)


def _kv_table() -> TextTable:
    return TextTable(1, None, cell_chars=8, unit_col=True)


def _kv(name: str, x, unit: str = "", sign: bool = False) -> tuple:
    """a row of a scalar table: the value in its fixed display unit, the unit in the unit column."""
    n, u = fmt.disp_parts(x, unit, sign)
    return (name, [n], u)


def _kt(name: str, text: str, unit: str = "") -> tuple:
    return (name, [text], unit)


def _vector_rows(*cols) -> list:
    return [(STATE_LABELS[i], [fmt.disp(c[i], STATE_DISPLAY[i]) for c in cols], "") for i in range(N_STATE)]


S_NOTE = {False: "diagonal: σ_i = √S_ii; off the diagonal: the correlation ρ_ij = S_ij / (σ_i σ_j)",
          True: "raw entries S_ij with the unit of z_i · z_j (m, rad, rad, m/s)"}


def s_rows(S, raw: bool) -> list:
    """the rows of the innovation covariance table. by default the spreads sigma_i on the diagonal, each in
    its display unit, and the unit-free correlation off it; raw: the entries with their units.
    sigma_i = sqrt(S_ii) - innovation spread of component i
    rho_ij = S_ij / (sigma_i sigma_j) - correlation of components i and j"""
    S = np.asarray(S, dtype=float)
    m = S.shape[0]
    sig = np.sqrt(np.clip(np.diag(S), 0.0, None))
    rows = []
    for a in range(m):
        cells = []
        for b in range(m):
            if raw:
                cells.append(f"{fmt.sci_cell(S[a, b])} {unit_product(MEAS_UNITS[a], MEAS_UNITS[b])}")
            elif a == b:
                cells.append(fmt.disp(sig[a], MEAS_SMALL[a]))
            else:
                cells.append(fmt.num(S[a, b] / (sig[a] * sig[b]), 3, sign=True))
        rows.append((MEAS_LABELS[a], cells, ""))
    return rows


def _sigma(P) -> np.ndarray:
    return np.sqrt(np.clip(np.diag(np.asarray(P, float)), 0.0, None))


# the section part:

class _WheelToHorizontal(QObject):
    """a vertical wheel scrolls the strip of cards sideways when there is nothing to scroll vertically."""

    def __init__(self, area):
        super().__init__(area)
        self.area = area

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.Wheel and not self.area.verticalScrollBar().isVisible():
            bar = self.area.horizontalScrollBar()
            bar.setValue(bar.value() - event.angleDelta().y())
            return True
        return False


class _Section(QWidget):
    """one tab: a horizontal strip of cards that scrolls sideways, so no card is cut at the bottom."""

    def __init__(self):
        super().__init__()
        self.show_values = False  # raw numbers instead of heat maps and sigma / correlation
        self.body = QWidget()
        self.body.setObjectName("panel_body")
        self.row = QHBoxLayout(self.body)
        e = round(0.6 * theme.em())
        self.row.setContentsMargins(e, e, e, e)
        self.row.setSpacing(e)
        self.row.addStretch(1)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.area = scroll(self.body, horizontal=True)
        self.area.viewport().installEventFilter(_WheelToHorizontal(self.area))
        lay.addWidget(self.area)
        self.cards: list[Card] = []

    def card(self, title, equations, evaluation=False) -> Card:
        c = Card(title, equations, evaluation)
        self.row.insertWidget(self.row.count() - 1, c, 0, Qt.AlignmentFlag.AlignTop)
        self.cards.append(c)
        return c

    def content_height(self) -> int:
        """the height the strip needs to show its tallest card in full."""
        m = self.row.contentsMargins()
        return max((c.sizeHint().height() for c in self.cards), default=0) + m.top() + m.bottom()

    def update_cursor(self, tl: Timeline, k: int, i: int) -> None:
        raise NotImplementedError


class TruthSection(_Section):
    def __init__(self):
        super().__init__()
        c = self.card("Equations of motion, 3-DOF point mass", [
            r"$\dot{\mathbf{p}} = \mathbf{v}$",
            r"$\dot{\mathbf{v}} = \frac{T(h)\,\mathbf{u}_{thrust} + \mathbf{D}}{m} + [0,\ 0,\ -g(h)]^T$",
            r"$\dot{m} = -T_{sl}\,/\,(I_{sp}\,g_0)$ while burning, 0 after burnout"], evaluation=True)
        self.t_motion = c.add(_kv_table(), right=True)
        c = self.card("Drag and ballistic coefficient", [
            r"$\mathbf{D} = -\frac{1}{2}\,\rho(h)\,|\mathbf{v}|\,\mathbf{v}\,C_d(M)\,A$",
            r"$M = |\mathbf{v}|\,/\,a(h),\quad a = \sqrt{\gamma\,R\,T(h)}$",
            r"$\beta = \frac{m}{C_d\,A}$"], evaluation=True)
        self.t_drag = c.add(_kv_table(), right=True)
        c = self.card("Thrust and gravity", [
            r"$T(h) = T_{sl} + (p_0 - p(h))\,A_e$",
            r"$g(h) = g_0\,\left(\frac{R_e}{R_e + h}\right)^2$"], evaluation=True)
        self.t_force = c.add(_kv_table(), right=True)
        for c in self.cards:
            c.finish()

    def update_cursor(self, tl, k, i):
        self.t_motion.set_rows([_kv("t", tl.truth_t[i], "s"),
                                _kt("phase", "BOOST" if tl.truth_boost[i] else "BALLISTIC"),
                                _kv("m", tl.truth_mass[i], "kg"),
                                _kv("h", tl.truth_alt_km[i] * 1e3, "pos"),
                                _kv("|v|", tl.truth_speed[i], "m/s")])
        self.t_drag.set_rows([_kv("ρ(h)", tl.truth_rho[i], "kg/m³"),
                              _kv("q = ½ρ|v|²", tl.truth_q[i], "kPa"),
                              _kv("M", tl.truth_mach[i]),
                              _kv("C_d(M)", tl.truth_cd[i]),
                              _kv("|D|", tl.truth_drag_kN[i] * 1e3, "kN"),
                              _kv("β", tl.truth_beta[i], "kg/m²")])
        self.t_force.set_rows([_kv("|T(h)|", tl.truth_thrust_kN[i] * 1e3, "kN"),
                               _kv("g(h)", tl.truth_g[i], "m/s²")])


class RadarSection(_Section):
    def __init__(self):
        super().__init__()
        c = self.card("Measurement function h(x)", [
            r"$\mathbf{d} = \mathbf{p} - \mathbf{p}_{radar},\quad r = |\mathbf{d}|$",
            r"$az = \mathrm{atan2}(d_E,\ d_N),\quad el = \mathrm{atan2}(d_U,\ \sqrt{d_E^2 + d_N^2})$",
            r"$\dot{r} = \mathbf{d}\cdot\mathbf{v}\,/\,r,\quad \mathbf{z} = h(\mathbf{x}) + \mathbf{w}$"])
        self.t_z = c.add(_kv_table(), right=True)
        self.n_z = c.note(right=True)
        c = self.card("Measurement noise", [
            r"$\mathbf{w} \sim N(0,\ R)$",
            r"$R = \mathrm{diag}(\sigma_r^2,\ \sigma_{az}^2,\ \sigma_{el}^2,\ \sigma_{\dot{r}}^2)$"])
        self.t_r = c.add(_kv_table(), right=True)
        c.note("R is fixed: no SNR dependence (THEORY 1)")
        c = self.card("Doppler physics", [
            r"$\lambda = c\,/\,f_c,\quad f_d = \frac{2\,v_r\,f_c}{c},\quad v_r = -\dot{r}$",
            r"$\Delta f = 1\,/\,T_{dwell}$",
            r"$\sigma_{\dot{r}} := \Delta v = \frac{\lambda}{2\,T_{dwell}}$"])
        self.t_dop = c.add(_kv_table(), right=True)
        c.note("closing is positive; one resolution cell is conservative")
        c = self.card("Pulse timing", [r"$t_{rx} = t_{tx} + 2\,r\,/\,c$", r"pulse length $c\,\tau$"])
        self.t_pulse = c.add(_kv_table(), right=True)
        c = self.card("Range and Doppler ambiguity, information", [
            r"$PRF > 2\,|f_d|_{max}$", r"$R_{ua} = \frac{c}{2\,PRF}$"], evaluation=True)
        self.t_amb = c.add(_kv_table(), right=True)
        c.note("range-rate is assumed unambiguous (THEORY 5.3); the PRF this needs leaves an unambiguous range far "
               "shorter than the target range")
        for c in self.cards:
            c.finish()

    def update_cursor(self, tl, k, i):
        cfg = tl.request.radar
        j = int(tl.last_detection[k])
        z = tl.z[j] if j >= 0 else np.full(4, np.nan)
        self.t_z.set_rows([_kv("r", z[0], "pos"), _kv("az", z[1], "°"), _kv("el", z[2], "°"),
                           _kv(RDOT, z[3], "m/s", True)])
        self.n_z.setText(f"latest detection: scan {j:,}, t = {fmt.disp(tl.t_scan[j], 's')}" if j >= 0
                         else "no detection yet")
        radar = tl.result.radar
        self.t_r.set_rows([_kv("σ_r", cfg.sigma_range, "m"), _kv("σ_az = σ_el", cfg.sigma_angle, "mrad"),
                           _kv(f"σ_{RDOT} (derived)", radar.sigma_rdot, "m/s"), _kt("dimension m", str(tl.dim))])
        self.t_dop.set_rows([_kv("f_c", cfg.carrier_frequency, "GHz"), _kv("λ", radar.wavelength, "cm"),
                             _kv("T_dwell", cfg.dwell_time, "ms"), _kv("Δf", 1.0 / cfg.dwell_time, "Hz"),
                             _kv(f"σ_{RDOT}", radar.sigma_rdot, "m/s"),
                             _kv("f_d (latest)", tl.echo_fd[j] if j >= 0 else math.nan, "kHz", True)])
        self.t_pulse.set_rows([_kv("t_tx", tl.t_scan[j] if j >= 0 else math.nan, "s"),
                               _kv("2r/c", tl.echo_delay[j] if j >= 0 else math.nan, "ms"),
                               _kt("t_rx", fmt.num(tl.echo_t_rx[j], 6) if j >= 0 else DASH, "s"),
                               _kv("c τ", radar.pulse_length, "m")])
        lo, hi = tl.slant_range_km
        self.t_amb.set_rows([_kv(f"max |{RDOT}| of the run", tl.max_abs_rdot, "m/s"),
                             _kv("|f_d|max", tl.max_abs_fd, "kHz"), _kv("PRF needed >", tl.prf_required, "kHz"),
                             _kv("R_ua at that PRF", tl.unambiguous_range, "km"),
                             _kt("target slant range", f"{fmt.num(lo, 1)}–{fmt.num(hi, 1)}", "km")])


class PredictSection(_Section):
    def __init__(self):
        super().__init__()
        c = self.card("Filter model f(x), mode from the IR flag", [
            r"$\dot{\mathbf{p}} = \mathbf{v},\quad \dot{\mathbf{v}} = -\frac{\rho(h)\,|\mathbf{v}|\,\mathbf{v}}{2\,\beta} + [0,\ 0,\ -g(h)]^T + \mathbf{a}$",
            r"BOOST: $\dot{\mathbf{a}} = 0$ + random walk $q_{thrust}$, $\ln\beta$ frozen",
            r"BALLISTIC: $\mathbf{a} = 0$, $\dot{(\ln\beta)} = 0$ + random walk $q_\beta$"])
        self.t_model = c.add(_kv_table())
        c = self.card("State prediction", [
            r"$\hat{\mathbf{x}}_{k|k-1} = \mathrm{RK4}\left[f\right]$ from $\hat{\mathbf{x}}_{k-1|k-1}$ over $\Delta t$"])
        self.t_x = c.add(TextTable(2, ("x̂ k−1|k−1", "x̂ k|k−1"), cell_chars=12, unit_col=False), right=True)
        self.n_x = c.note()
        c = self.card("Transition matrix", [
            r"$\Phi_k = \prod_i\left(I + A_i h + \frac{1}{2}(A_i h)^2\right)$", r"$A = \partial f / \partial \mathbf{x}$"])
        self.m_phi = c.add(MatrixView(STATE_LABELS, STATE_LABELS), right=True)
        c.note("chained over the RK4 sub-steps h. colour: |Φ_ij| σ_j / σ_i (σ from P k|k−1), unit-free")
        c = self.card("Process noise", [
            r"$Q_{axis} = (q_a + q_m)\,[[\Delta t^3/3,\ \Delta t^2/2],\ [\Delta t^2/2,\ \Delta t]]$",
            r"$Q_{\ln\beta} = q_\beta\,\Delta t,\quad Q_{a} = q_{thrust}\,\Delta t$ (BOOST)",
            r"$q_m = \sigma_{m}^2\,\Delta t$ while the safety net is on"])
        self.t_qpar = c.add(_kv_table())
        self.m_q = c.add(MatrixView(STATE_LABELS, STATE_LABELS, C.COVARIANCE), right=True)
        c.note("colour: |correlation| of Q", right=True)
        c = self.card("Covariance prediction", [r"$P_{k|k-1} = \Phi_k\,P_{k-1|k-1}\,\Phi_k^T + Q_k$"])
        self.m_p = c.add(MatrixView(STATE_LABELS, STATE_LABELS, C.COVARIANCE), right=True)
        c.note("colour: |correlation| ρ_ij = P_ij / √(P_ii P_jj); the exact entry with its unit is in the tooltip")
        for c in self.cards:
            c.finish()

    def update_cursor(self, tl, k, i):
        F = tl.result.filter
        prior = F is not None and tl.tracking(k) and k > tl.k0 and np.all(np.isfinite(F.x_prior[k]))
        cfg = tl.request.filter
        if not prior:
            self.t_model.set_rows([_kt("mode", DASH), _kt("Δt", DASH, "s")])
            self.t_x.set_rows([])
            self.t_qpar.set_rows([])
            for m in (self.m_phi, self.m_q, self.m_p):
                m.set_data(None, None)
            self.n_x.setText("initialisation scan: the track starts from the first detection (THEORY 6.5)"
                             if tl.tracking(k) else "no track yet: the beam stays at the cue")
            return
        boost = tl.mode[k] == "BOOST"
        x_pr, P_pr = F.x_prior[k], F.P_prior[k]
        self.t_model.set_rows([_kt("mode", str(tl.mode[k])), _kv("Δt", tl.T, "ms"),
                               _kv("|â_drag| (filter)", tl.filter_model.drag_acceleration(x_pr), "m/s²"),
                               _kv("|â| thrust", float(np.linalg.norm(x_pr[6:9])) if boost else 0.0, "m/s²")])
        dim = () if boost else (6, 7, 8)
        base = np.full(N_STATE, np.nan) if tl.reprocessed[k] else F.x_post[k - 1]
        self.t_x.set_rows(_vector_rows(base, x_pr), dim_rows=dim)
        self.n_x.setText("re-processed record: restored at the last scan at or before t_flag − L − T and "
                         f"replayed {int(tl.reprocessed[k])} scans in BALLISTIC (THEORY 6.2); this scan's beam was "
                         "pointed from the earlier BOOST prediction" if tl.reprocessed[k] else "")
        s = _sigma(P_pr)
        self.m_phi.set_data(F.Phi[k], scaled_magnitude(F.Phi[k], s, s), unit_grid(STATE_UNITS, STATE_UNITS, -1),
                            dim_rows=dim)
        q_m = cfg.sigma_manoeuvre ** 2 * tl.T if tl.manoeuvre_on[k] else 0.0
        self.t_qpar.set_rows([_kv("q_a", F.q_a[k], "m²/s³"), _kv("q_m", q_m, "m²/s³"),
                              _kv("q_β", cfg.q_beta, "1/s"),
                              _kv("q_thrust", cfg.q_thrust, "m²/s⁵") if boost else _kt("q_thrust", "off", "")])
        cov_units = unit_grid(STATE_UNITS, STATE_UNITS)
        self.m_q.set_data(F.Q[k], correlation_magnitude(F.Q[k]), cov_units, dim_rows=dim)
        self.m_p.set_data(P_pr, correlation_magnitude(P_pr), cov_units,
                          lambda a, b: f"correlation {fmt.num(P_pr[a, b] / (s[a] * s[b]), 3)}"
                          if s[a] > 0 and s[b] > 0 else "fixed state (zero variance)", dim_rows=dim)


class UpdateSection(_Section):
    def __init__(self):
        super().__init__()
        c = self.card("Innovation", [
            r"$\nu_k = \mathbf{z}_k - h(\hat{\mathbf{x}}_{k|k-1})$", r"angle rows wrapped to $[-\pi,\ \pi)$"])
        self.t_nu = c.add(TextTable(3, ("z", "h(x̂)", "ν"), cell_chars=11, unit_col=False), right=True)
        self.n_nu = c.note()
        c = self.card("Measurement Jacobian", [
            r"$H_k = \partial h / \partial \mathbf{x}$ at $\hat{\mathbf{x}}_{k|k-1}$",
            r"$\partial\dot{r}/\partial\mathbf{p} = (\mathbf{v} - \dot{r}\,\mathbf{d}/r)^T / r$",
            r"$\partial\dot{r}/\partial\mathbf{v} = \mathbf{d}^T / r$"])
        self.m_h = c.add(MatrixView(MEAS_LABELS, STATE_LABELS), right=True)
        c.note("colour: |H_ij| σ_j / √S_ii, unit-free", right=True)
        c = self.card("Innovation covariance", [r"$S_k = H_k\,P_{k|k-1}\,H_k^T + R$"])
        self.t_s = c.add(TextTable(4, MEAS_LABELS, cell_chars=7, unit_col=False, diagonal=C.COVARIANCE), right=True)
        self.n_s = c.note(S_NOTE[False], right=True)
        c = self.card("Kalman gain", [
            r"$K_k = P_{k|k-1}\,H_k^T\,S_k^{-1}$", r"linear solve, never an inverse",
            r"BOOST: $K_k[\ln\beta,\ :] = 0$ (consider update)"])
        self.m_k = c.add(MatrixView(STATE_LABELS, MEAS_LABELS), right=True)
        c.note("colour: |K_ij| √S_jj / σ_i, the share of the prior σ of state i that a 1σ innovation j moves",
               right=True)
        c = self.card("State update", [r"$\hat{\mathbf{x}}_{k|k} = \hat{\mathbf{x}}_{k|k-1} + K_k\,\nu_k$"])
        self.t_x = c.add(TextTable(3, ("x̂ k|k−1", "K ν", "x̂ k|k"), cell_chars=11, unit_col=False), right=True)
        c = self.card("Covariance update, Joseph form", [
            r"$P_{k|k} = (I - K_k H_k)\,P_{k|k-1}\,(I - K_k H_k)^T + K_k\,R\,K_k^T$"])
        self.m_p = c.add(MatrixView(STATE_LABELS, STATE_LABELS, C.COVARIANCE), right=True)
        c.note("symmetric and positive definite under rounding, and exact for the frozen ln β row (a non-optimal "
               "gain). colour: |correlation|")
        for c in self.cards:
            c.finish()

    def update_cursor(self, tl, k, i):
        F = tl.result.filter
        m = tl.dim
        upd = F is not None and tl.tracking(k) and bool(tl.updated[k])
        if not upd:
            for t in (self.t_nu, self.t_s, self.t_x):
                t.set_rows([])
            for mv in (self.m_h, self.m_k, self.m_p):
                mv.set_data(None, None)
            self.n_nu.setText(("no update this scan: " + tl.miss_reason(k) + "; x̂ k|k = x̂ k|k−1 and "
                               "P k|k = P k|k−1") if tl.tracking(k) and k > tl.k0 else
                              "no update: " + ("initialisation scan" if tl.tracking(k) else "no track yet"))
            return
        radar = tl.result.radar
        x_pr, P_pr = F.x_prior[k], F.P_prior[k]
        z_pred = radar.h(x_pr[0:3], x_pr[3:6])  # h(x_{k|k-1}), the filter's own prediction of z
        H = np.zeros((m, N_STATE))
        H[:, 0:6] = radar.jacobian(x_pr[0:3], x_pr[3:6])  # H_k, zero a and ln beta columns
        nu, S, K = F.nu[k], F.S[k], F.K[k]
        self.t_nu.set_rows([(MEAS_LABELS[j], [fmt.disp(tl.z[k, j], MEAS_ABSOLUTE[j]),
                                              fmt.disp(z_pred[j], MEAS_ABSOLUTE[j]),
                                              fmt.disp(nu[j], MEAS_SMALL[j], True)], "") for j in range(m)])
        self.n_nu.setText(f"NIS = {fmt.disp(F.nis[k])}")
        s = _sigma(P_pr)
        sS = np.sqrt(np.diag(S))
        frozen = {BETA_ROW} if tl.mode[k] == "BOOST" else set()
        dim = frozen | ({6, 7, 8} if tl.mode[k] != "BOOST" else set())
        self.m_h.set_data(H, scaled_magnitude(H, sS, s), unit_grid(MEAS_UNITS[:m], STATE_UNITS, -1))
        self.t_s.set_rows(s_rows(S, self.show_values))
        self.n_s.setText(S_NOTE[self.show_values])
        self.m_k.set_data(K, scaled_magnitude(K, s, sS), unit_grid(STATE_UNITS, MEAS_UNITS[:m], -1), dim_rows=dim)
        self.t_x.set_rows(_vector_rows(x_pr, K @ nu, F.x_post[k]), dim_rows=dim)
        P = F.P_post[k]
        sp = _sigma(P)
        self.m_p.set_data(P, correlation_magnitude(P), unit_grid(STATE_UNITS, STATE_UNITS),
                          lambda a, b: f"correlation {fmt.num(P[a, b] / (sp[a] * sp[b]), 3)}"
                          if sp[a] > 0 and sp[b] > 0 else "fixed state (zero variance)", dim_rows=dim)


class ConsistencySection(_Section):
    def __init__(self):
        super().__init__()
        c = self.card("Normalised innovation squared", [r"$\varepsilon_\nu = \nu_k^T\,S_k^{-1}\,\nu_k \sim \chi^2(m)$"])
        self.t_nis = c.add(_kv_table(), right=True)
        c = self.card("Window NIS and the safety net", [
            r"$\bar{\varepsilon} = \frac{1}{n}\sum \varepsilon_\nu$ over the last $n \leq N_w$ updates",
            r"manoeuvre noise on while $\bar{\varepsilon} > \chi^{2}_{1-\alpha}(n\,m)\,/\,n$",
            r"$\alpha = p_{fa}\,/\,N_{updates}$ (Bonferroni)"])
        self.t_win = c.add(_kv_table(), right=True)
        c = self.card("Track status rules", [
            r"LOCKED: $\bar{\varepsilon}$ inside its 95 % interval and $\sigma_{pos} < \sigma_{lock}$ for $M$ updates",
            r"LOST: $\sigma_{pos} > \sigma_{lost}$, or $\varepsilon_\nu > \chi^2_{0.999}(m)$ for $K$ updates"])
        self.t_status = c.add(_kv_table(), right=True)
        c = self.card("Normalised estimation error squared", [
            r"$\varepsilon = \mathbf{e}^T P_{k|k}^{-1}\,\mathbf{e},\quad \mathbf{e} = \hat{\mathbf{x}}_{k|k} - \mathbf{x}$",
            r"6 kinematic states, $\varepsilon \sim \chi^2(6)$"], evaluation=True)
        self.t_nees = c.add(_kv_table(), right=True)
        for c in self.cards:
            c.finish()

    def update_cursor(self, tl, k, i):
        cfg = tl.request.filter
        lo, hi = tl.nis_bounds
        nis = float(tl.nis[k])
        inside = lo <= nis <= hi if math.isfinite(nis) else None
        self.t_nis.set_rows([_kv("ε_ν", nis), _kt("m", str(tl.dim)),
                             _kt("95 % interval", f"{fmt.disp(lo)} – {fmt.disp(hi)}"),
                             _kt("verdict", DASH if inside is None else ("inside" if inside else "outside"))],
                            cell_colours={} if inside in (None, True) else {(3, 0): C.COASTING})
        w = float(tl.window_nis[k])
        n_win = min(int(np.sum(tl.updated[:k + 1])), cfg.nis_window)
        alpha = cfg.p_fa_flight / cfg.nominal_updates
        on = bool(tl.manoeuvre_on[k]) if tl.tracking(k) else False
        self.t_win.set_rows([_kv("ε̄", w), _kt("n", str(n_win)),
                             _kt("95 % interval", f"{fmt.disp(tl.window_lo[k])} – {fmt.disp(tl.window_hi[k])}"),
                             _kv("net threshold", tl.net_threshold[k]), _kt("α", fmt.sci_cell(alpha)),
                             _kt("manoeuvre noise", "ON" if on else "OFF")],
                            cell_colours={(5, 0): C.COASTING} if on else {})
        status = str(tl.status[k]) if tl.tracking(k) else DASH
        self.t_status.set_rows([_kt("status", status), _kv("σ_pos", tl.sigma_pos_max[k], "m"),
                                _kv("σ_lock", cfg.lock_sigma_pos, "m"), _kv("σ_lost", cfg.lost_sigma_pos, "pos"),
                                _kv("NIS LOST bound", tl.nis_lost),
                                _kt("M / K", f"{cfg.lock_updates} / {cfg.lost_updates}")],
                               cell_colours={(0, 0): theme.STATUS_COLOUR.get(status, C.INK)})
        nlo, nhi = tl.nees_bounds
        ne = float(tl.nees[k])
        self.t_nees.set_rows([_kv("ε", ne), _kt("95 % interval", f"{fmt.disp(nlo)} – {fmt.disp(nhi)}")],
                             cell_colours={(0, 0): C.COASTING} if math.isfinite(ne) and ne > nhi else {})


# the panel part:

def _colour(w: QLabel, colour: str) -> None:
    if w.property("_colour") != colour:
        w.setProperty("_colour", colour)
        w.setStyleSheet(f"color: {colour};")


class EquationsPanel(QWidget):
    """the step header and the five section tabs. only the visible section is refreshed."""

    def __init__(self):
        super().__init__()
        self.setObjectName("panel")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        lay = QVBoxLayout(self)
        e = theme.em()
        lay.setContentsMargins(0, round(0.3 * e), 0, 0)
        lay.setSpacing(0)
        head = QHBoxLayout()
        head.setContentsMargins(round(0.7 * e), 0, round(0.7 * e), round(0.3 * e))
        head.setSpacing(round(0.5 * e))
        head.addWidget(section_header("Equations"))
        self.chip_predict = Chip("PREDICT", C.INK_3, "mono_medium")
        self.chip_update = Chip("UPDATE", C.INK_3, "mono_medium")
        head.addWidget(self.chip_predict)
        head.addWidget(label("→", "caption"))
        head.addWidget(self.chip_update)
        self.info = label("", "secondary", "mono")
        head.addWidget(self.info)
        self.message = label("", "caption", "caption")
        self.message.setWordWrap(True)
        head.addWidget(self.message, 1)
        self.values = QCheckBox("show values")
        self.values.setToolTip("show every heat map as its full numeric table")
        self.values.toggled.connect(self.set_show_values)
        head.addWidget(self.values)
        lay.addLayout(head)
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.tabBar().setUsesScrollButtons(True)
        self.sections = [TruthSection(), RadarSection(), PredictSection(), UpdateSection(), ConsistencySection()]
        names = ("1  Truth physics", "2  Radar and Doppler", "3  EKF predict", "4  EKF update", "5  Consistency")
        for s, n in zip(self.sections, names):
            self.tabs.addTab(s, n)
        self.tabs.setTabToolTip(0, "evaluation – uses truth")
        self.tabs.setCurrentIndex(2)
        self.tabs.currentChanged.connect(lambda _: self._refresh(force=True))
        lay.addWidget(self.tabs, 1)
        self.tl: Timeline | None = None
        self._cursor = None
        self._shown = None
        self.refresh_interval = UIConfig().text_refresh_interval  # in [s]
        self._clock = QElapsedTimer()
        self._stale = False

    def set_show_values(self, on: bool) -> None:
        for m in self.findChildren(MatrixView):
            m.set_show_values(on)
        for sec in self.sections:
            sec.show_values = on
        self._refresh(force=True)

    def preferred_height(self) -> int:
        """the drawer height that shows the tallest card of the current section in full."""
        head = self.tabs.tabBar().sizeHint().height() + round(2.2 * theme.em())
        bar = self.style().pixelMetric(self.style().PixelMetric.PM_ScrollBarExtent)
        return self.tabs.currentWidget().content_height() + head + bar

    def set_timeline(self, tl: Timeline) -> None:
        self.tl = tl
        self._shown = None

    def show_cursor(self, k: int, i: int, playing: bool = False) -> None:
        """show scan k and truth sample i. while playing, the text refreshes at most every
        UIConfig.text_refresh_interval of wall time: at x20 a new scan arrives every frame, faster than
        numbers can be read. paused, scrubbed or stepped, it refreshes at once. a closed drawer is not refreshed."""
        self._cursor = (k, i)
        if not self.isVisible():
            return
        if playing and self._clock.isValid() and self._clock.elapsed() < 1000.0 * self.refresh_interval:
            self._stale = True
            return
        self._clock.start()
        self._stale = False
        self._refresh()

    def showEvent(self, event):
        super().showEvent(event)
        self._refresh(force=True)

    def _refresh(self, force: bool = False) -> None:
        tl = self.tl
        if tl is None or self._cursor is None:
            return
        k, i = self._cursor
        sec = self.tabs.currentWidget()
        key = (k, i if isinstance(sec, TruthSection) else -1, self.tabs.currentIndex())
        if key == self._shown and not force:
            return
        self._shown = key
        self._header(tl, k)
        sec.update_cursor(tl, k, i)

    def _header(self, tl: Timeline, k: int) -> None:
        mode = str(tl.mode[k]) if tl.tracking(k) else ""
        self.info.setText(f"scan k = {k:,}   t_k = {fmt.num(tl.t_scan[k], 2)} s" + (f"   {mode}" if mode else ""))
        if mode:
            _colour(self.info, MODE_COLOUR.get(mode, C.INK_2))
        if not tl.tracking(k):
            self.chip_predict.set("PREDICT", C.INK_3)
            self.chip_update.set("UPDATE", C.INK_3)
            self.message.setText("no track yet: cued search, the beam stays at the launch-site cue" if tl.acquired
                                 else "NO ACQUISITION: " + tl.acquisition_reason)
            _colour(self.message, C.INK_3 if tl.acquired else C.LOST)
            return
        if k == tl.k0:
            self.chip_predict.set("INITIALISE", C.ACCENT)
            self.chip_update.set("UPDATE", C.INK_3)
            self.message.setText("track initialised from the first detection (THEORY 6.5)")
            _colour(self.message, C.INK_2)
            return
        self.chip_predict.set("PREDICT", C.ACCENT)
        if tl.updated[k]:
            self.chip_update.set("UPDATE", C.ACCENT)
            msg, colour = "", C.INK_2
        else:
            self.chip_update.set("UPDATE", C.INK_3)
            msg, colour = "no update this scan: " + tl.miss_reason(k), C.COASTING
        if tl.reprocessed[k]:
            msg = (msg + "; " if msg else "") + (f"re-processed from t_flag − L − T ({int(tl.reprocessed[k])} "
                                                 "scans replayed in BALLISTIC, THEORY 6.2)")
            colour = C.INK_2 if tl.updated[k] else colour
        self.message.setText(msg)
        _colour(self.message, colour)
