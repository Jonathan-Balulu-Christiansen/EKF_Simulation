"""the live readout panel (THEORY 8.8, progressive disclosure). it follows the playback
cursor every frame. the top answers "what is happening now" with about eight values, filter first: track
status, mode, position uncertainty (3 sigma, largest axis), beta +-1 sigma, NIS against its bound, then the
estimated slant range, altitude and speed. below it, two collapsed groups: details of the radar and the filter,
and the truth and error values (evaluation - uses truth). numbers are right-aligned in IBM Plex Mono with their
unit; a value outside its bound is drawn in the warning colour, status and mode use their reserved colours."""

import math

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QGridLayout, QLabel, QVBoxLayout, QWidget

from ekfsim_ui import fmt, theme
from ekfsim_ui.fmt import DASH
from ekfsim_ui.theme import C, MODE_COLOUR, STATUS_COLOUR
from ekfsim_ui.timeline import Timeline
from ekfsim_ui.widgets import Chip, Collapsible, eval_tag, label, section_header

NAME_COLUMN = 7.5  # width of the name column in em, the same in every group so the values line up


class _Value(QLabel):
    """a numeric value: mono, right-aligned, re-set only when its text or colour changes."""

    def __init__(self):
        super().__init__(DASH)
        self.setFont(theme.font("mono"))
        self.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._state = (DASH, None)

    def show_value(self, text: str, colour: str | None = None) -> None:
        old_text, old_colour = self._state
        if text != old_text:
            self.setText(text)
        if colour != old_colour:
            self.setStyleSheet(f"color: {colour};" if colour else "")
        self._state = (text, colour)


class _Rows(QWidget):
    """one group of readout rows: name, value, unit."""

    def __init__(self, v: dict, chips: dict):
        super().__init__()
        self.v, self.chips = v, chips
        self.grid = QGridLayout(self)
        e = theme.em()
        self.grid.setContentsMargins(0, round(0.15 * e), 0, round(0.2 * e))
        self.grid.setHorizontalSpacing(round(0.5 * e))
        self.grid.setVerticalSpacing(round(0.12 * e))
        self.grid.setColumnMinimumWidth(0, round(NAME_COLUMN * e))
        self.grid.setColumnStretch(1, 1)
        self.row = 0

    def _name(self, text: str, tooltip: str) -> None:
        self.grid.addWidget(label(text, "secondary", tooltip=tooltip), self.row, 0)

    def value(self, key: str, name: str, unit: str, tooltip: str = "") -> None:
        self._name(name, tooltip)
        v = _Value()
        self.grid.addWidget(v, self.row, 1)
        self.grid.addWidget(label(unit, "unit", "caption"), self.row, 2)
        self.v[key] = v
        self.row += 1

    def chip(self, key: str, name: str, tooltip: str) -> None:
        self._name(name, tooltip)
        chip = Chip(DASH, C.INK_3, "mono_medium")
        self.grid.addWidget(chip, self.row, 1, 1, 2, Qt.AlignmentFlag.AlignRight)
        self.chips[key] = chip
        self.row += 1

    def note(self) -> QLabel:
        n = label("", "caption", "caption")
        n.setWordWrap(True)
        self.grid.addWidget(n, self.row, 0, 1, 3)
        self.row += 1
        return n


class ReadoutPanel(QWidget):
    """the readout groups; update_cursor() takes the cursor state from the timeline."""

    def __init__(self):
        super().__init__()
        self.setObjectName("panel_body")
        lay = QVBoxLayout(self)
        e = theme.em()
        lay.setContentsMargins(round(0.7 * e), round(0.5 * e), round(0.7 * e), round(0.7 * e))
        lay.setSpacing(round(0.3 * e))
        self.v: dict[str, _Value] = {}
        self.chips: dict[str, Chip] = {}

        lay.addWidget(section_header("Track"))
        now = _Rows(self.v, self.chips)
        now.chip("status", "track status", "THEORY 6.7, from the filter's own information only")
        now.chip("mode", "mode", "BOOST or BALLISTIC; the filter mode follows the IR flag only (THEORY 6.2)")
        now.value("pos3s", "position 3σ", "m", "3 sigma along the largest axis of the position covariance "
                                               "(3 sqrt of the largest eigenvalue of P_pp)")
        now.value("beta", "β̂ ±1σ", "kg/m²", "beta estimate exp(ln beta)")
        self.beta_note = now.note()
        now.value("nis", "NIS / 95 % bound", "", "normalised innovation squared of this scan against the upper "
                                                 "bound of its 95 % interval; amber outside the interval")
        now.value("range", "slant range", "km", "estimated range from the radar, |p_hat - p_radar|")
        now.value("alt", "altitude", "km", "estimated altitude p_hat_U")
        now.value("speed", "speed", "m/s", "estimated speed |v_hat|")
        lay.addWidget(now)

        det = _Rows(self.v, self.chips)
        det.chip("phase", "phase (IR flag)", "as reported by the IR plume sensor, with its latency (THEORY 5.7)")
        det.chip("scan", "this scan", "detection, or a missed detection with its reason")
        self.scan_reason = det.note()
        det.value("r", "measured range", "km", "slant range of the latest detection")
        det.value("az", "azimuth", "°", "compass azimuth of the latest detection")
        det.value("el", "elevation", "°", "of the latest detection")
        det.value("rdot", f"range-rate {fmt.RDOT}", "m/s", "of the latest detection")
        det.value("age", "age of that detection", "s")
        det.value("miss", "misses search/track", "", "scans without an echo before and after the first detection")
        det.value("wnis", "window NIS", "", "mean of the last N_w NIS values; amber outside its 95 % interval")
        det.value("wnis_b", "  95 % interval", "", "chi2(n m) / n interval of the window mean")
        det.chip("net", "manoeuvre noise", "the safety net of THEORY 6.6")
        det.value("a_hat", "|â| thrust", "m/s²", "estimated thrust acceleration (BOOST mode)")
        det.value("pE", "position E", "km")
        det.value("pN", "position N", "km")
        det.value("pU", "position U", "km")
        det.chip("pd", "P (Cholesky test)", "P_{k|k} positive definite on the active states (THEORY 6.4)")
        self.details = Collapsible("Details", det, expanded=False)
        lay.addWidget(self.details)

        ev = _Rows(self.v, self.chips)
        ev.value("t_alt", "altitude h", "km")
        ev.value("t_vel", "speed |v|", "m/s")
        ev.value("mach", "Mach", "")
        ev.value("mass", "mass m", "kg")
        ev.value("thrust", "thrust |T|", "kN")
        ev.value("drag", "drag |D|", "kN")
        ev.value("rho", "density ρ", "kg/m³")
        ev.value("beta_t", "β truth", "kg/m²")
        ev.value("phase_t", "true phase", "")
        ev.value("ep", "|position error|", "m")
        ev.value("ev", "|velocity error|", "m/s")
        ev.value("nees", "NEES", "", "e^T P^-1 e on the 6 kinematic states; amber above its 95 % interval")
        ev.value("nees_b", "  95 % interval", "", "two-sided chi2(6) interval of one NEES")
        self.evaluation = Collapsible("Truth and errors", ev, expanded=False, tag=eval_tag())
        lay.addWidget(self.evaluation)
        lay.addStretch(1)
        self._last = None
        self.details.toggled.connect(lambda _: self._refresh_last())
        self.evaluation.toggled.connect(lambda _: self._refresh_last())

    def _refresh_last(self) -> None:
        if self._last is not None:
            self.update_cursor(*self._last)

    # the update part

    def update_cursor(self, tl: Timeline, t: float, k: int, i: int, speed: float) -> None:
        self._last = (tl, t, k, i, speed)
        V, q = self.v, fmt.q
        tracking = tl.tracking(k)
        flag = bool(tl.engine_flag[k])

        status = str(tl.status[k]) if tracking else ("NO ACQUISITION" if not tl.acquired else "SEARCH")
        self.chips["status"].set(status, STATUS_COLOUR.get(status, C.LOST if not tl.acquired else C.INK_2))
        mode = str(tl.mode[k]) if tracking else DASH
        self.chips["mode"].set(mode,
                               MODE_COLOUR.get(mode, C.INK_3))
        V["pos3s"].show_value(q(3.0 * tl.sigma_pos_max[k], "m"))
        if tracking:
            b, b_lo, b_hi = tl.beta_interval(k, 1.0)
            V["beta"].show_value(q(b, "beta"))
            note = f"1σ interval {q(b_lo, 'beta')} – {q(b_hi, 'beta')} kg/m² (ln β is Gaussian)"
        else:
            V["beta"].show_value(DASH)
            note = ""
        if self.beta_note.text() != note:
            self.beta_note.setText(note)
            self.beta_note.setVisible(bool(note))
        lo, hi = tl.nis_bounds
        nis = float(tl.nis[k])
        out = math.isfinite(nis) and not lo <= nis <= hi
        V["nis"].show_value(f"{fmt.num(nis, 2)} / {fmt.num(hi, 2)}" if math.isfinite(nis)
                            else (f"no update / {fmt.num(hi, 2)}" if tracking else DASH), C.COASTING if out else None)
        if tracking:
            p = tl.result.filter.x_post[k, 0:3]
            V["range"].show_value(q(float(np.linalg.norm(p - tl.result.radar.position)), "km"))
            V["alt"].show_value(q(p[2], "km"))
            V["speed"].show_value(q(tl.est_speed[k], "speed"))
        else:
            for key in ("range", "alt", "speed"):
                V[key].show_value(DASH)

        if self.details.body.isVisible():
            self._details(tl, t, k, tracking, flag)
        if self.evaluation.body.isVisible():
            self._evaluation(tl, k, i, tracking)

    def _details(self, tl, t, k, tracking, flag):
        V, q = self.v, fmt.q
        self.chips["phase"].set("BOOST" if flag else "BALLISTIC", MODE_COLOUR["BOOST" if flag else "BALLISTIC"])
        if tl.detected[k]:
            self.chips["scan"].set("detection", C.BEAM)
            reason = ""
        else:
            self.chips["scan"].set("miss", C.INK_2)
            reason = "no echo: " + tl.miss_reason(k)
        if self.scan_reason.text() != reason:
            self.scan_reason.setText(reason)
            self.scan_reason.setVisible(bool(reason))
        j = int(tl.last_detection[k])
        z = tl.z[j] if j >= 0 else np.full(4, np.nan)
        V["r"].show_value(q(z[0], "km"))
        V["az"].show_value(q(z[1], "deg"))
        V["el"].show_value(q(z[2], "deg"))
        V["rdot"].show_value(q(z[3], "rdot", sign=True))
        V["age"].show_value(fmt.num(t - tl.t_scan[j], 2) if j >= 0 else DASH)
        V["miss"].show_value(f"{int(tl.search_miss_cum[k]):,} / {int(tl.track_miss_cum[k]):,}")
        w, wlo, whi = float(tl.window_nis[k]), float(tl.window_lo[k]), float(tl.window_hi[k])
        V["wnis"].show_value(fmt.num(w, 2), C.COASTING if math.isfinite(w) and not wlo <= w <= whi else None)
        V["wnis_b"].show_value(f"{fmt.num(wlo, 2)} – {fmt.num(whi, 2)}" if math.isfinite(w) else DASH)
        net = bool(tl.manoeuvre_on[k]) if tracking else False
        self.chips["net"].set("ON" if net else ("OFF" if tracking else DASH), C.COASTING if net else C.INK_3)
        a = float(np.linalg.norm(tl.est_acc[k])) if tracking and tl.mode[k] == "BOOST" else math.nan
        V["a_hat"].show_value(q(a, "accel"))
        p = tl.est_pos_km[k] * 1e3 if tracking else (math.nan,) * 3
        V["pE"].show_value(q(p[0], "km", sign=True))
        V["pN"].show_value(q(p[1], "km", sign=True))
        V["pU"].show_value(q(p[2], "km", sign=True))
        pd_ok = bool(tl.pd_ok[k])
        self.chips["pd"].set(("positive definite" if pd_ok else "NOT positive definite") if tracking else DASH,
                             C.INK_3 if pd_ok else C.LOST)

    def _evaluation(self, tl, k, i, tracking):
        V, q = self.v, fmt.q
        V["t_alt"].show_value(q(tl.truth_alt_km[i] * 1e3, "km"))
        V["t_vel"].show_value(q(tl.truth_speed[i], "speed"))
        V["mach"].show_value(q(tl.truth_mach[i], "mach"))
        V["mass"].show_value(q(tl.truth_mass[i], "mass"))
        V["thrust"].show_value(q(tl.truth_thrust_kN[i] * 1e3, "kN"))
        V["drag"].show_value(q(tl.truth_drag_kN[i] * 1e3, "kN"))
        V["rho"].show_value(fmt.sig(tl.truth_rho[i], 3))
        V["beta_t"].show_value(q(tl.truth_beta[i], "beta"))
        V["phase_t"].show_value("BOOST" if tl.truth_boost[i] else "BALLISTIC")
        ep = float(np.linalg.norm(tl.err_pos[k])) if tracking else math.nan
        ev = float(np.linalg.norm(tl.err_vel[k])) if tracking else math.nan
        V["ep"].show_value(q(ep, "m"))
        V["ev"].show_value(q(ev, "rdot"))
        ne, (nlo, nhi) = float(tl.nees[k]), tl.nees_bounds
        V["nees"].show_value(fmt.num(ne, 2), C.COASTING if math.isfinite(ne) and ne > nhi else None)
        V["nees_b"].show_value(f"{fmt.num(nlo, 2)} – {fmt.num(nhi, 2)}")
