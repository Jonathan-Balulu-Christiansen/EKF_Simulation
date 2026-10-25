"""here I turn one ScenarioResult into the arrays the UI draws (THEORY 8.1: compute, then play back). this
module has no Qt import, so it runs and is tested headless; it is built once per run in the worker thread,
and the GUI thread only indexes these arrays at the playback cursor.
two kinds of arrays live here, kept apart by name:
- filter and sensor arrays (estimate, covariance, NIS, beam, IR flag, status): the tracker's own
  information, what an operator of the real system would see.
- arrays named truth_* or err_*, nees and the geometry/PRF summary: ground truth, used only by views
  labelled "evaluation - uses truth" (THEORY 0.2).
time base: scans at t_k = k T. between two scans every filter quantity holds the value of the last scan
at or before the cursor, because the filter is discrete; nothing is interpolated or extrapolated. the
truth holds its last sample of the 0.01 s truth grid."""

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import chi2

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
from ekfsim.ekf import ACC, BETA_ROW, FilterModel, position_sigma_max
from ekfsim.metrics import chi2_interval, nees
from ekfsim.radar import cue_direction, doppler_shift, pointing_vector, required_prf, unambiguous_range
from ekfsim.scenario import ScenarioResult, run_scenario, truth_summary

KM = 1_000.0  # metres per kilometre, the scene unit of the 3D view [m/km]


def _ro(a) -> np.ndarray:
    a = np.array(a)
    a.setflags(write=False)
    return a


# the run request part:

@dataclass(frozen=True)
class RunRequest:
    """everything one run needs, as the controls set it. frozen, so two requests compare by value."""

    vehicle: VehicleConfig = field(default_factory=VehicleConfig)
    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)
    radar: RadarConfig = field(default_factory=RadarConfig)
    filter: FilterConfig = field(default_factory=FilterConfig)
    ir: IRSensorConfig = field(default_factory=IRSensorConfig)


def compute(request: RunRequest, ui_cfg: UIConfig | None = None) -> "Timeline":
    """here I run truth, radar and EKF for one request and prepare the display arrays. this is what the
    worker thread calls; it raises ValueError with the reason for a configuration that cannot run."""
    res = run_scenario(request.vehicle, request.scenario, request.radar, request.filter, ir_cfg=request.ir)
    return Timeline(res, request, ui_cfg or UIConfig())


# the display geometry part:

def rotation_from_z(u) -> np.ndarray:
    """the rotation matrix that turns the +z axis into the unit vector u, used to point the beam cone
    (built along +z) at the beam pointing. Rodrigues' formula about the axis z x u.
    k = z x u / |z x u|, c = z . u, s = |z x u| - rotation axis, cosine and sine of the angle
    R = I + s [k]_x + (1 - c) [k]_x^2 - Rodrigues rotation"""
    u = np.asarray(u, dtype=float)
    u = u / np.linalg.norm(u)
    c = float(u[2])  # cos of the angle between +z and u
    axis = np.array([-u[1], u[0], 0.0])  # z x u
    s = float(np.linalg.norm(axis))  # sin of that angle
    if s < 1e-12:
        return np.eye(3) if c > 0.0 else np.diag([1.0, -1.0, -1.0])  # u = +z or a half turn about x
    k = axis / s
    kx = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])  # [k]_x, cross-product matrix
    return np.eye(3) + s * kx + (1.0 - c) * kx @ kx


def ellipsoid_axes(P_pp, n_sigma: float, scale: float) -> tuple[np.ndarray, np.ndarray]:
    """the principal axes of the n-sigma position ellipsoid of a 3x3 covariance P_pp [m^2], magnified by
    the display factor scale. returns the rotation (columns = unit axes, right-handed) and the semi-axes
    in [km].
    P_pp = V diag(lambda) V^T - eigen-decomposition
    a_i = n_sigma scale sqrt(lambda_i) - semi-axis i of the displayed ellipsoid"""
    lam, V = np.linalg.eigh(np.asarray(P_pp, dtype=float))
    if np.linalg.det(V) < 0.0:
        V[:, 2] = -V[:, 2]  # keep a proper rotation, so the mesh is not mirrored
    radii = n_sigma * scale * np.sqrt(np.clip(lam, 0.0, None)) / KM  # a_i in [km]
    return V, radii


# the pulse part (slow-motion pulse view, THEORY 8.5):

@dataclass(frozen=True)
class PulseState:
    """where the transmitted pulse of scan k is at one instant, from the stored pulse timing (THEORY 5.5)."""

    k: int  # scan index
    t_tx: float  # emission time in [s]
    elapsed: float  # t - t_tx in [s]
    segments: tuple[tuple[float, float], ...]  # (from, to) distance from the antenna along u of each pulse part, in [m]
    u: np.ndarray  # unit direction the pulse travels (the measured line of sight, or the beam axis on a miss)
    r: float  # reflection range c delay / 2 in [m], NaN on a miss
    phase: str  # outgoing / reflecting at the target / echo returning / echo received / no echo
    done: float  # time the last wavefront is back at the antenna (or has left the beam) in [s]
    detected: bool


def pulse_segments(elapsed: float, tau: float, c: float, r: float | None, end: float) -> tuple:
    """the parts of a pulse of width tau [s] that left the antenna elapsed [s] ago, as (from, to) distances in
    [m]. a wavefront emitted at offset s in [0, tau] has travelled the path c (elapsed - s); it goes out while the
    path is below r and comes back after the reflection, so near the target the pulse folds into two parts.
    without a reflection (a miss, r None) it travels out until the end of the beam, end [m].
    path_head = c elapsed, path_tail = c (elapsed - tau) - first and last wavefront, clipped to [0, 2r]
    d = path for path <= r, d = 2r - path for r < path <= 2r - distance from the antenna"""
    head = max(c * elapsed, 0.0)  # path of the first wavefront in [m]
    tail = max(c * (elapsed - tau), 0.0)  # path of the last wavefront in [m]
    if r is None:
        a, b = min(tail, end), min(head, end)
        return ((a, b),) if b > a else ()
    parts = []
    a, b = tail, min(head, r)  # the part still going out
    if b > a:
        parts.append((a, b))
    a, b = max(tail, r), min(head, 2.0 * r)  # the part already reflected, coming back
    if b > a:
        parts.append((2.0 * r - b, 2.0 * r - a))
    return tuple(parts)


# the timeline part:

class Timeline:
    """the display arrays of one run (module docstring). index k is the scan, index i the truth sample."""

    def __init__(self, result: ScenarioResult, request: RunRequest, ui_cfg: UIConfig):
        self.result = result
        self.request = request
        self.ui = ui_cfg
        truth, plan, radar, F = result.truth, result.plan, result.radar, result.filter
        constants = PhysicsConstants()
        self.filter_model = FilterModel(request.filter, constants, AtmosphereConfig())  # the filter's own model
        self.T = radar.cfg.update_interval  # scan interval T in [s]
        self.tol = request.scenario.time_tol  # time tolerance in [s]
        self.t_scan = plan.t  # t_k in [s]
        self.n_scans = len(plan.t)
        self.t_end = float(truth.t[-1])  # last truth sample (ground contact) in [s]
        self.dim = radar.dim  # measurement dimension m
        self.c = constants.c  # speed of light in [m/s]
        self.pulse_width = radar.cfg.pulse_width  # tau in [s]
        self.atmosphere_boundaries = US76Atmosphere(AtmosphereConfig(), constants).boundaries()  # the ISA layers (model data)
        self.radar_km = _ro(radar.position / KM)  # antenna position in [km]
        self.launch_km = _ro(truth.position[0] / KM)  # launch site in [km]

        # the truth part (evaluation - uses truth):
        self.truth_t = truth.t
        self.truth_pos_km = _ro(truth.position / KM)
        self.truth_alt_km = _ro(truth.altitude / KM)
        self.truth_speed = truth.speed
        self.truth_mach = truth.mach
        self.truth_mass = truth.mass
        self.truth_thrust_kN = _ro(np.linalg.norm(truth.thrust, axis=1) / KM)  # |T| in [kN]
        self.truth_drag_kN = _ro(np.linalg.norm(truth.drag, axis=1) / KM)  # |D| in [kN]
        self.truth_rho = truth.density
        self.truth_beta = truth.beta
        self.truth_cd = truth.cd
        self.truth_q = truth.dynamic_pressure
        self.truth_g = truth.gravity
        self.truth_boost = truth.boost
        n_truth = len(truth.t)
        stride = max(1, math.ceil(n_truth / ui_cfg.trail_max_points))  # trail decimation, THEORY 8.1 cap
        self._truth_trail_idx = np.arange(0, n_truth, stride)
        idx = plan.truth_index
        x_true = np.hstack([truth.position[idx], truth.velocity[idx]])  # truth state at each scan

        # the sensor part: scans, beam, IR flag
        scans = result.scans
        self.detected = _ro([s.detected for s in scans])
        self.scan_status = _ro([s.status.value for s in scans])
        self.engine_flag = plan.engine_flag  # IR plume flag reported with each scan (THEORY 5.7)
        self.beam_u = _ro([s.pointing for s in scans])  # beam axis u_k in ENU
        z = np.full((self.n_scans, 4), np.nan)
        for s in scans:
            if s.detected:
                z[s.k, :self.dim] = s.z
        self.z = _ro(z)  # measurements [r, az, el, rdot] in [m, rad, rad, m/s], NaN on misses
        echo = np.full((self.n_scans, 3), np.nan)
        for s in scans:
            if s.detected:
                echo[s.k] = (s.delay, s.t_rx, s.doppler_shift)
        self.echo_delay, self.echo_t_rx, self.echo_fd = (_ro(echo[:, j]) for j in range(3))  # [s], [s], [Hz]
        meas = np.full((self.n_scans, 3), np.nan)
        det = self.detected
        if np.any(det):
            meas[det] = radar.position_from_measurement(z[det, 0:3])  # converted for display only
        self._meas_det_km = _ro(meas[det] / KM)  # detections only, in scan order, in [km]
        self.det_cum = _ro(np.cumsum(det))  # number of detections at or before scan k
        last = np.where(det, np.arange(self.n_scans), -1)
        self.last_detection = _ro(np.maximum.accumulate(last))  # latest detected scan at or before k, -1 none

        acq = result.acquisition
        self.acquired = acq.acquired
        self.acquisition_reason = acq.reason
        self.cue_az, self.cue_el = cue_direction(radar.cfg, request.scenario)  # in [rad]
        d_launch = truth.position[0, 0:2] - radar.position[0:2]
        cue_len = float(np.hypot(*d_launch)) / math.cos(self.cue_el)  # slant length of the cue beam in [m]
        k0 = F.track_start if F is not None else self.n_scans
        self.k0 = k0 if F is not None else None  # scan of the first detection (track start)
        beam_len = np.full(self.n_scans, cue_len)
        if F is not None and k0 + 1 < self.n_scans:
            d = F.x_prior[k0 + 1:, 0:3] - radar.position
            beam_len[k0 + 1:] = np.linalg.norm(d, axis=1)  # to the predicted range |p_{k|k-1} - p_radar|
        self.beam_len_km = _ro(beam_len / KM)
        search = np.zeros(self.n_scans, bool)
        search[:min(k0, self.n_scans)] = True
        self.search_miss_cum = _ro(np.cumsum(search & ~det))  # misses of the cued search before the track
        self.track_miss_cum = _ro(np.cumsum(~search & ~det))  # misses while tracking

        # the filter part (the tracker's own information):
        n = self.n_scans
        nan3 = np.full((n, 3), np.nan)
        self.has_track = F is not None
        if F is None:
            self.status = _ro(np.full(n, "", dtype="<U9"))
            self.mode = _ro(np.full(n, "", dtype="<U9"))
            for name in ("est_pos_km", "est_vel", "est_acc", "sig_pos", "sig_vel", "err_pos", "err_vel"):
                setattr(self, name, _ro(nan3))
            for name in ("ln_beta", "sigma_ln_beta", "sigma_pos_max", "nis", "window_nis", "window_lo",
                         "window_hi", "net_threshold", "nees", "truth_beta_scan", "est_speed"):
                setattr(self, name, _ro(np.full(n, np.nan)))
            self.updated = self.manoeuvre_on = self.skipped_singular = _ro(np.zeros(n, bool))
            self.pd_ok = _ro(np.ones(n, bool))
            self.reprocessed = _ro(np.zeros(n, int))
            self.nu = self.sqrt_S = _ro(np.full((n, self.dim), np.nan))
            self.events = ()
            self._est_trail_idx = np.arange(0)
        else:
            self.status, self.mode = F.status, F.mode
            self.updated, self.manoeuvre_on = F.updated, F.manoeuvre_on
            self.skipped_singular, self.pd_ok, self.reprocessed = F.skipped_singular, F.pd_ok, F.reprocessed
            self.est_pos_km = _ro(F.x_post[:, 0:3] / KM)
            self.est_vel = _ro(F.x_post[:, 3:6])
            self.est_speed = _ro(np.linalg.norm(F.x_post[:, 3:6], axis=1))
            self.est_acc = _ro(F.x_post[:, ACC])  # a_hat, thrust acceleration estimate in [m/s^2]
            self.ln_beta = _ro(F.x_post[:, BETA_ROW])
            self.sigma_ln_beta = _ro(np.sqrt(F.P_post[:, BETA_ROW, BETA_ROW]))
            diag = np.diagonal(F.P_post, axis1=1, axis2=2)
            self.sig_pos = _ro(np.sqrt(diag[:, 0:3]))  # per-axis position std in [m]
            self.sig_vel = _ro(np.sqrt(diag[:, 3:6]))  # per-axis velocity std in [m/s]
            sig_max = np.full(n, np.nan)
            sig_max[k0:] = [position_sigma_max(P) for P in F.P_post[k0:]]  # THEORY 6.7 sigma_pos
            self.sigma_pos_max = _ro(sig_max)
            self.nis, self.window_nis = F.nis, F.window_nis
            self.nu = F.nu
            with np.errstate(invalid="ignore"):
                self.sqrt_S = _ro(np.sqrt(np.diagonal(F.S, axis1=1, axis2=2)))  # sqrt(S_ii)
            self.events = F.events
            m, cfg = self.dim, request.filter
            n_win = np.minimum(np.cumsum(F.updated), cfg.nis_window)  # NIS values in the window at scan k
            lo, hi, net = (np.full(n, np.nan) for _ in range(3))
            alpha = cfg.p_fa_flight / cfg.nominal_updates  # Bonferroni alpha per window test (THEORY 6.6)
            for w in range(1, cfg.nis_window + 1):
                sel = n_win == w
                a = 1.0 - cfg.confidence
                lo[sel] = chi2.ppf(0.5 * a, w * m) / w  # lower 95 % bound of the window mean
                hi[sel] = chi2.ppf(1.0 - 0.5 * a, w * m) / w  # upper 95 % bound of the window mean
                net[sel] = chi2.ppf(1.0 - alpha, w * m) / w  # safety-net switch-on threshold
            self.window_lo, self.window_hi, self.net_threshold = _ro(lo), _ro(hi), _ro(net)
            track = np.arange(k0, n)
            stride = max(1, math.ceil(len(track) / ui_cfg.trail_max_points))
            self._est_trail_idx = track[::stride]

            # the evaluation part (uses truth):
            err_p, err_v, ne = nan3.copy(), nan3.copy(), np.full(n, np.nan)
            err_p[k0:] = F.x_post[k0:, 0:3] - x_true[k0:, 0:3]  # e_p = p_{k|k} - p_true in [m]
            err_v[k0:] = F.x_post[k0:, 3:6] - x_true[k0:, 3:6]  # e_v = v_{k|k} - v_true in [m/s]
            ne[k0:] = nees(F.x_post[k0:], F.P_post[k0:], x_true[k0:])  # NEES on the 6 kinematic states
            self.err_pos, self.err_vel, self.nees = _ro(err_p), _ro(err_v), _ro(ne)
            self.truth_beta_scan = _ro(truth.beta[idx])
        self.nis_bounds = chi2_interval(self.dim, request.filter.confidence)  # single-NIS 95 % interval
        self.nees_bounds = chi2_interval(6, request.filter.confidence)  # single-run NEES 95 % interval
        self.nis_lost = float(chi2.ppf(request.filter.lost_confidence, self.dim))  # LOST NIS bound

        # the geometry and PRF summary part (evaluation - uses truth, THEORY 9 and 5.3):
        d = truth.position - radar.position  # line of sight from the antenna in [m]
        rho_h = np.hypot(d[:, 0], d[:, 1])  # horizontal distance in [m]
        slant = np.linalg.norm(d, axis=1)  # slant range in [m]
        el = np.arctan2(d[:, 2], rho_h)  # elevation in [rad]
        rdot = np.sum(d * truth.velocity, axis=1) / slant  # truth range-rate in [m/s]
        self.min_horizontal_km = float(rho_h.min() / KM)
        self.slant_range_km = (float(slant.min() / KM), float(slant.max() / KM))
        i_el = int(np.argmax(el))
        self.max_elevation = float(el[i_el])  # in [rad]
        self.max_elevation_time = float(truth.t[i_el])  # in [s]
        self.elevation_warning = self.max_elevation > ui_cfg.max_elevation_warning
        self.max_abs_rdot = float(np.max(np.abs(rdot)))  # max |rdot| of the run in [m/s]
        self.max_abs_fd = float(abs(doppler_shift(self.max_abs_rdot, radar.wavelength)))  # max |f_d| in [Hz]
        self.prf_required = required_prf(self.max_abs_fd)  # PRF > 2 |f_d|max in [Hz]
        self.unambiguous_range = unambiguous_range(self.prf_required, constants.c)  # c / (2 PRF) in [m]
        try:
            self.summary = truth_summary(truth, tuple(radar.position))
        except ValueError:
            self.summary = None

    # the cursor part

    def scan_index(self, t: float) -> int:
        """k of the last scan at or before t [s], clipped to the run.
        k = floor((t + tol) / T) - last scan index"""
        return int(min(max(math.floor((t + self.tol) / self.T), 0), self.n_scans - 1))

    def truth_index(self, t: float) -> int:
        """i of the last truth sample at or before t [s]."""
        i = int(np.searchsorted(self.truth_t, t + self.tol, side="right")) - 1
        return min(max(i, 0), len(self.truth_t) - 1)

    def tracking(self, k: int) -> bool:
        """True once the track exists at scan k (the first detection initialises it)."""
        return self.k0 is not None and k >= self.k0

    # the trail part (each trail is capped at UIConfig.trail_max_points, THEORY 8.1)

    def truth_trail(self, i: int) -> np.ndarray:
        """truth trajectory in [km] up to truth sample i, ending exactly at sample i."""
        n = int(np.searchsorted(self._truth_trail_idx, i, side="right"))
        return np.vstack([self.truth_pos_km[self._truth_trail_idx[:n]], self.truth_pos_km[i:i + 1]])

    def estimate_trail(self, k: int) -> np.ndarray:
        """posterior positions x_{k|k} in [km] from the track start up to scan k, ending at scan k."""
        if not self.tracking(k):
            return np.zeros((0, 3))
        n = int(np.searchsorted(self._est_trail_idx, k, side="right"))
        return np.vstack([self.est_pos_km[self._est_trail_idx[:n]], self.est_pos_km[k:k + 1]])

    def measurements_upto(self, k: int) -> np.ndarray:
        """the detections up to scan k, converted to ENU in [km] for display only."""
        return self._meas_det_km[:int(self.det_cum[k])]

    # the per-scan helpers

    def position_covariance(self, k: int) -> np.ndarray:
        """P_pp of x_{k|k}, the 3x3 position block, in [m^2]."""
        return self.result.filter.P_post[k, 0:3, 0:3]

    def beta_interval(self, k: int, n_sigma: float = 1.0) -> tuple[float, float, float]:
        """beta_hat and its n-sigma interval in [kg/m^2]. ln beta is Gaussian in the filter, so beta is
        lognormal and its interval is asymmetric.
        beta_hat = exp(ln beta), [exp(ln beta - n sigma), exp(ln beta + n sigma)] - interval"""
        lb, s = float(self.ln_beta[k]), float(self.sigma_ln_beta[k])
        return math.exp(lb), math.exp(lb - n_sigma * s), math.exp(lb + n_sigma * s)

    def pulse_state(self, t: float) -> PulseState:
        """the pulse of the scan at or before t [s]. the timing is the stored two-way delay 2r/c of the
        detection (THEORY 5.5); the direction is the measured line of sight, or the beam axis on a miss, so the
        view needs no truth position.
        r = c (2r/c) / 2 - reflection range from the stored delay
        t_done = t_tx + 2r/c + tau (detection), t_tx + L/c + tau (miss, L the beam length) - end of the pulse"""
        k = self.scan_index(t)
        t_tx = float(self.t_scan[k])
        e = t - t_tx  # elapsed since emission in [s]
        c, tau = self.c, self.pulse_width
        if self.detected[k]:
            r = 0.5 * c * float(self.echo_delay[k])  # reflection range in [m]
            u = pointing_vector(float(self.z[k, 1]), float(self.z[k, 2]))  # measured line of sight
            segs = pulse_segments(e, tau, c, r, 2.0 * r)
            done = t_tx + 2.0 * r / c + tau
            head, tail = c * e, c * (e - tau)
            phase = ("outgoing" if head < r else "reflecting at the target" if tail < r else
                     "echo returning" if tail < 2.0 * r else "echo received")
            return PulseState(k, t_tx, e, segs, u, r, phase, done, True)
        end = float(self.beam_len_km[k]) * KM  # beam length L in [m]
        u = np.asarray(self.beam_u[k], dtype=float)
        segs = pulse_segments(e, tau, c, None, end)
        phase = "no echo (" + self.miss_reason(k) + ")" if c * (e - tau) < end else "left the beam, no echo"
        return PulseState(k, t_tx, e, segs, u, math.nan, phase, t_tx + end / c + tau, False)

    def miss_reason(self, k: int) -> str:
        """why scan k gave no update: the scan's miss status, or the singularity guard."""
        if self.skipped_singular[k]:
            return "update skipped: target near the vertical"
        return str(self.scan_status[k]).replace("miss: ", "")
