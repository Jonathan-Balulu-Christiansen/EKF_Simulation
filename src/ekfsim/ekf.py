"""here I build the Extended Kalman Filter (THEORY 6): the filter model f with its analytic Jacobian in the
two modes BOOST and BALLISTIC, the discretisation (Phi, Q), predict and update, the initialisation from
the first detection, the adaptive process noise (a safety net) and the track status.
the tracker never reads truth and never imports the truth model (vehicle.py) or the scenario. its mode
comes only from the IR engine flag (THEORY 5.7). it uses the same ISA density as the truth (known
atmosphere); the deliberate model mismatch left is a constant beta against Cd(Mach) and the changing
mass, and an estimated acceleration instead of a thrust model (no inverse crime, THEORY 6.2). the
measurement model h(x) and its Jacobian come from radar.py, as a real tracker uses its radar's equations."""

import dataclasses
import math
from collections import deque
from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy.stats import chi2

from ekfsim.atmosphere import US76Atmosphere
from ekfsim.config import AtmosphereConfig, FilterConfig, PhysicsConstants, ScenarioConfig
from ekfsim.integrators import rk4_step
from ekfsim.radar import Radar, Scan, wrap_to_pi

N_STATE = 10  # x = [pE, pN, pU, vE, vN, vU, aE, aN, aU, ln beta]
ANGLE_ROWS = (1, 2)  # az and el rows of z; their innovations are wrapped to [-pi, pi)
ACC = slice(6, 9)  # thrust acceleration a in x
BETA_ROW = 9  # index of ln beta in x
ACTIVE_BALLISTIC = np.r_[0:6, BETA_ROW]  # states with variance in BALLISTIC mode (a is fixed at 0)


class FilterMode(Enum):
    """the filter mode, set only by the IR engine flag (THEORY 6.2)."""

    BOOST = "BOOST"
    BALLISTIC = "BALLISTIC"


# the filter dynamics model part:

class FilterModel:
    """here I build the filter's dynamics model (THEORY 6.2), units SI, with two modes. in BOOST the
    unknown thrust acceleration a is estimated as a random walk, because the filter has no thrust model
    (thrust/mass grows as propellant burns and turns with the pitch-over). in BALLISTIC a is 0. I
    estimate ln beta instead of beta: it keeps beta positive and makes its scale well conditioned.
    the density is the known ISA (US76) density; outside the atmosphere rho is about 0, so drag drops out
    of f and of A by itself and beta simply gets no information there: no extra mode is needed.
    dp/dt = v - kinematics
    dv/dt = -rho(h) |v| v / (2 beta) + [0, 0, -g(h)] + a - drag, gravity and thrust acceleration
    da/dt = 0, plus random-walk noise q_thrust (BOOST); a = 0 (BALLISTIC) - thrust acceleration
    d(ln beta)/dt = 0, plus random-walk noise q_beta - nearly constant ballistic coefficient
    g(h) = g0 (Re / (Re + h))^2 - gravity"""

    n = N_STATE

    def __init__(self, cfg: FilterConfig, constants: PhysicsConstants, atmosphere_cfg: AtmosphereConfig):
        self._atm = US76Atmosphere(atmosphere_cfg, constants)
        self._g0 = constants.g0  # g0 in [m/s^2]
        self._Re = constants.earth_radius  # Re in [m]
        self._substep = cfg.propagation_substep  # longest sub-step in [s]
        self._sigma_floor = cfg.sigma_a_floor  # sigma_floor in [m/s^2]
        self._k_drag = cfg.k_drag  # k_drag [-]
        self._q_beta = cfg.q_beta  # q_beta in [1/s]
        self._q_thrust = cfg.q_thrust  # q_thrust in [m^2/s^5]

    def f(self, x: np.ndarray, boost: bool) -> np.ndarray:
        """the filter right-hand side dx/dt (class docstring equations), units [m/s, m/s^2, m/s^3, 1/s]."""
        _, _, h, vE, vN, vU, aE, aN, aU, ln_beta = x.tolist()
        speed = math.sqrt(vE * vE + vN * vN + vU * vU)  # |v| in [m/s]
        c = self._atm.density(h) * speed / (2.0 * math.exp(ln_beta))  # rho |v| / (2 beta) in [1/s]
        g = self._g0 * (self._Re / (self._Re + h)) ** 2  # g(h) in [m/s^2]
        if not boost:
            aE = aN = aU = 0.0
        return np.array([vE, vN, vU, -c * vE + aE, -c * vN + aN, -c * vU - g + aU, 0.0, 0.0, 0.0, 0.0])

    def jacobian(self, x: np.ndarray, boost: bool) -> np.ndarray:
        """this is the continuous Jacobian A = df/dx (10 x 10), derived analytically. the position
        columns enter only through h = p_U, via the ISA gradient drho/dh and dg/dh = -2 g / (Re + h).
        dp'/dv = I - kinematics block
        da/dv = -rho / (2 beta) (|v| I + v v^T / |v|) - drag vs velocity
        da/dp_U = -(drho/dh) |v| v / (2 beta) + [0, 0, 2 g / (Re + h)] - drag and gravity vs altitude
        da/d(ln beta) = +rho |v| v / (2 beta) - drag vs ln beta
        dv'/da = I in BOOST, 0 in BALLISTIC - thrust acceleration"""
        _, _, h, vE, vN, vU, _, _, _, ln_beta = x.tolist()
        v = np.array([vE, vN, vU])
        speed = math.sqrt(vE * vE + vN * vN + vU * vU)  # |v| in [m/s]
        rho, drho_dh = self._atm.density_and_gradient(h)  # rho [kg/m^3], drho/dh [kg/m^4]
        two_beta = 2.0 * math.exp(ln_beta)  # 2 beta in [kg/m^2]
        k = rho / two_beta  # rho / (2 beta) in [1/m]
        g = self._g0 * (self._Re / (self._Re + h)) ** 2  # g(h) in [m/s^2]
        A = np.zeros((N_STATE, N_STATE))
        A[0:3, 3:6] = np.eye(3)  # dp'/dv
        if speed > 0.0:
            A[3:6, 3:6] = -k * (speed * np.eye(3) + np.outer(v, v) / speed)  # da/dv
        A[3:6, 2] = -(drho_dh / two_beta) * speed * v  # da/dp_U from drho/dh
        A[5, 2] += 2.0 * g / (self._Re + h)  # da_U/dp_U from dg/dh
        A[3:6, BETA_ROW] = k * speed * v  # da/d(ln beta)
        if boost:
            A[3:6, ACC] = np.eye(3)  # dv'/da
        return A

    def propagate(self, x: np.ndarray, dt: float, boost: bool) -> tuple[np.ndarray, np.ndarray]:
        """here I predict the state over dt [s] with RK4 sub-steps and build the transition matrix over
        the same sub-steps, so a fast drag change inside one scan interval is still followed.
        x_{k|k-1} = RK4 sub-steps of f from x_{k-1|k-1} - state prediction
        Phi_i = I + A_i h + (A_i h)^2 / 2, A_i = A(x_i) - second-order transition matrix per sub-step
        Phi_k = Phi_n ... Phi_2 Phi_1 - transition matrix over dt"""
        n = max(1, math.ceil(dt / self._substep))  # number of sub-steps
        h = dt / n  # sub-step length in [s]
        Phi = np.eye(N_STATE)
        for _ in range(n):
            Ah = self.jacobian(x, boost) * h  # A_i h
            Phi = (np.eye(N_STATE) + Ah + 0.5 * Ah @ Ah) @ Phi  # Phi_i Phi
            x = rk4_step(lambda t, s, seg: self.f(s, boost), 0.0, x, h, 0)
        return x, Phi

    def drag_acceleration(self, x: np.ndarray) -> float:
        """the filter's OWN estimate of the drag deceleration magnitude at state x, in [m/s^2] (no truth).
        a_drag_hat = rho(h) |v|^2 / (2 beta) - estimated drag acceleration"""
        _, _, h, vE, vN, vU, _, _, _, ln_beta = x.tolist()
        return self._atm.density(h) * (vE * vE + vN * vN + vU * vU) / (2.0 * math.exp(ln_beta))

    def spectral_density(self, x: np.ndarray, dt: float) -> float:
        """here I schedule the process noise on the filter's own drag estimate (THEORY 6.3). in the coast
        the filter model matches the truth almost exactly, so only a small floor is needed; in re-entry the
        model error is mostly in drag (Cd(Mach) against a constant beta), so the unmodelled acceleration
        scales with the drag itself. sigma_a is turned into the spectral density with the scan interval
        dt [s], so that the velocity spread added over one scan equals (sigma_a dt)^2.
        sigma_a = sigma_floor + k_drag |a_drag_hat| - drag-scheduled acceleration noise std
        q_a = sigma_a^2 dt - spectral density of the white acceleration, in [m^2/s^3]"""
        sigma_a = self._sigma_floor + self._k_drag * self.drag_acceleration(x)  # sigma_a in [m/s^2]
        return sigma_a * sigma_a * dt  # q_a in [m^2/s^3]

    def process_noise(self, dt: float, q_a: float, boost: bool, q_m: float = 0.0) -> np.ndarray:
        """the discrete process noise Q (10 x 10) over dt [s]: continuous white acceleration per axis with
        spectral density q_a plus, while the safety net is on, the manoeuvre density q_m (THEORY 6.6); a
        random walk on ln beta; and in BOOST a random walk on each thrust-acceleration component.
        Q_axis = (q_a + q_m) [[dt^3/3, dt^2/2], [dt^2/2, dt]] - white-acceleration block of (p_i, v_i)
        Q_a = q_thrust dt per axis (BOOST only) - thrust-acceleration random walk
        Q_beta = q_beta dt - ln beta random walk"""
        q = q_a + q_m  # white-acceleration spectral density in [m^2/s^3]
        Q = np.zeros((N_STATE, N_STATE))
        for i in range(3):
            Q[i, i] = q * dt ** 3 / 3.0  # position variance growth in [m^2]
            Q[i, i + 3] = Q[i + 3, i] = q * dt ** 2 / 2.0  # position-velocity covariance in [m^2/s]
            Q[i + 3, i + 3] = q * dt  # velocity variance growth in [m^2/s^2]
            if boost:
                Q[6 + i, 6 + i] = self._q_thrust * dt  # thrust-acceleration variance growth in [m^2/s^4]
        Q[BETA_ROW, BETA_ROW] = self._q_beta * dt  # ln beta variance growth [-]
        return Q


# the EKF update part:

@dataclass(frozen=True)
class UpdateResult:
    """the quantities of one measurement update, kept for plots and the equations panel."""

    x: np.ndarray  # x_{k|k}
    P: np.ndarray  # P_{k|k}
    nu: np.ndarray  # innovation y_k
    S: np.ndarray  # innovation covariance S_k
    K: np.ndarray  # Kalman gain K_k
    nis: float  # NIS = y_k^T S_k^-1 y_k


def kalman_update(x_pred: np.ndarray, P_pred: np.ndarray, z: np.ndarray, z_pred: np.ndarray,
                  H: np.ndarray, R: np.ndarray, angle_rows=(), frozen_rows=()) -> UpdateResult:
    """this is the EKF update step. because h is nonlinear the innovation uses the full nonlinear
    h(x_{k|k-1}) (passed in as z_pred), and H is the Jacobian at x_{k|k-1}. the gain comes from a linear
    solve, never from an inverse. I use the Joseph form because it keeps P symmetric and positive
    definite under rounding, and because it stays exact for a gain that is not optimal: that is what
    makes frozen rows possible. a frozen row of K is set to zero (a Schmidt "consider" update): that state
    is not updated, its variance does not collapse, and its cross-covariances stay correct.
    y_k = z - h(x_{k|k-1}), angle rows wrapped to [-pi, pi) - innovation
    S_k = H_k P_{k|k-1} H_k^T + R - innovation covariance
    K_k = P_{k|k-1} H_k^T S_k^-1 - Kalman gain
    K_k[i, :] = 0 for frozen rows i - consider update, no learning of state i
    x_{k|k} = x_{k|k-1} + K_k y_k - state update
    P_{k|k} = (I - K_k H_k) P_{k|k-1} (I - K_k H_k)^T + K_k R K_k^T - Joseph form, then symmetrised
    NIS = y_k^T S_k^-1 y_k - normalised innovation squared"""
    nu = np.asarray(z, dtype=float) - z_pred  # y_k, innovation
    for i in angle_rows:
        nu[i] = wrap_to_pi(nu[i])  # keep the angle part of y_k in [-pi, pi)
    S = H @ P_pred @ H.T + R  # S_k, innovation covariance
    S = 0.5 * (S + S.T)
    K = np.linalg.solve(S, H @ P_pred).T  # K_k = P H^T S^-1, from S K^T = H P (S and P symmetric)
    for i in frozen_rows:
        K[i, :] = 0.0  # consider update: state i is not learned from this measurement
    x = x_pred + K @ nu  # x_{k|k}, state update
    IKH = np.eye(len(x_pred)) - K @ H  # I - K_k H_k
    P = IKH @ P_pred @ IKH.T + K @ R @ K.T  # P_{k|k}, Joseph form
    P = 0.5 * (P + P.T)
    nis = float(nu @ np.linalg.solve(S, nu))  # NIS
    return UpdateResult(x, P, nu, S, K, nis)


def is_positive_definite(P: np.ndarray) -> bool:
    """P is positive definite exactly when its Cholesky factorisation exists (THEORY 6.4)."""
    try:
        np.linalg.cholesky(P)
        return True
    except np.linalg.LinAlgError:
        return False


def active_states(mode: FilterMode) -> np.ndarray:
    """the states that carry variance in a mode. in BALLISTIC a is fixed at 0 with zero variance, so P is
    singular there by construction and the positive-definite check (THEORY 6.4) runs on the other states."""
    return np.arange(N_STATE) if mode is FilterMode.BOOST else ACTIVE_BALLISTIC


def position_sigma_max(P: np.ndarray) -> float:
    """sigma_pos = sqrt(max eig(P_pp)) - largest position std in [m] (THEORY 6.7)"""
    return math.sqrt(max(float(np.linalg.eigvalsh(P[0:3, 0:3])[-1]), 0.0))


# the initialisation part:

def initial_estimate(radar: Radar, z: np.ndarray, cfg: FilterConfig,
                     mode: FilterMode) -> tuple[np.ndarray, np.ndarray]:
    """here I initialise the filter from the first detection only, no truth (THEORY 6.5).
    p0 = p_radar + r [cos(el) sin(az), cos(el) cos(az), sin(el)] - inverse of h
    J = dp/d(r, az, el), P_pp = J diag(sigma_r^2, sigma_az^2, sigma_el^2) J^T - linearised conversion of R
    v0 = rdot u_r, u_r = (p0 - p_radar) / r - radial velocity from the range-rate
    P_vv = sigma_rdot^2 u_r u_r^T + sigma_t^2 (I - u_r u_r^T) - radial part measured, tangential unknown
    a0 = 0, P_aa = sigma_thrust0^2 I in BOOST, 0 in BALLISTIC - unknown thrust acceleration
    ln beta0 = ln(beta_0), sigma = sigma_ln_beta0 - wide prior on the ballistic coefficient"""
    r, az, el = float(z[0]), float(z[1]), float(z[2])
    ce, se, ca, sa = math.cos(el), math.sin(el), math.cos(az), math.sin(az)
    u_r = np.array([ce * sa, ce * ca, se])  # u_r, unit line of sight
    J = np.column_stack([u_r,  # dp/dr
                         r * np.array([ce * ca, -ce * sa, 0.0]),  # dp/daz
                         r * np.array([-se * sa, -se * ca, ce])])  # dp/del
    sigma_t = cfg.sigma_v_tangential  # sigma_t in [m/s]
    x0 = np.zeros(N_STATE)
    P0 = np.zeros((N_STATE, N_STATE))
    x0[0:3] = radar.position + r * u_r  # p0 in [m]
    P0[0:3, 0:3] = J @ radar.R[0:3, 0:3] @ J.T  # P_pp in [m^2]
    if radar.dim == 4:
        x0[3:6] = float(z[3]) * u_r  # v0 = rdot u_r in [m/s]
        P0[3:6, 3:6] = (radar.R[3, 3] * np.outer(u_r, u_r)
                        + sigma_t ** 2 * (np.eye(3) - np.outer(u_r, u_r)))  # P_vv in [m^2/s^2]
    else:
        P0[3:6, 3:6] = sigma_t ** 2 * np.eye(3)  # no range-rate: all velocity components unknown
    if mode is FilterMode.BOOST:
        P0[ACC, ACC] = cfg.sigma_thrust0 ** 2 * np.eye(3)  # P_aa in [m^2/s^4]
    x0[BETA_ROW] = math.log(cfg.beta0)  # ln beta0
    P0[BETA_ROW, BETA_ROW] = cfg.sigma_ln_beta0 ** 2  # var(ln beta0)
    return x0, P0


# the track status part:

class TrackStatus(Enum):
    """track status from the filter's own information only (THEORY 6.7)."""

    ACQUIRING = "ACQUIRING"
    LOCKED = "LOCKED"
    COASTING = "COASTING"
    LOST = "LOST"


@dataclass(frozen=True)
class StatusEvent:
    """one logged event: a status transition, a mode switch, an ignored flag or a P failure."""

    t: float  # time in [s]
    old: TrackStatus | None
    new: TrackStatus
    reason: str


@dataclass(frozen=True)
class StepRecord:
    """everything about one filter step (one scan), kept for plots and the equations panel. prior
    quantities are None at the initialisation step; update quantities are None without an update."""

    t: float  # scan time in [s]
    x_prior: np.ndarray | None  # x_{k|k-1}
    P_prior: np.ndarray | None  # P_{k|k-1}
    Phi: np.ndarray | None  # Phi_k
    Q: np.ndarray | None  # Q_k
    update: UpdateResult | None  # y_k, S_k, K_k, NIS
    x_post: np.ndarray  # x_{k|k} (equals x_{k|k-1} without an update)
    P_post: np.ndarray  # P_{k|k}
    q_a: float  # drag-scheduled spectral density q_a used in Q_k in [m^2/s^3], NaN at the init step
    manoeuvre_on: bool  # the safety net's manoeuvre noise was in Q_k (THEORY 6.6)
    mode: FilterMode  # filter mode after this step
    engine_flag: bool  # IR engine flag received with this scan
    beta_frozen: bool  # ln beta was not updated this step (BOOST mode, THEORY 6.2)
    window_nis: float  # mean NIS over the window after this step, NaN before the first update
    status: TrackStatus
    skipped_singular: bool  # update skipped by the singularity guard (THEORY 5.6)
    pd_ok: bool  # P_{k|k} passed the Cholesky test on the active states
    reprocessed: int  # number of scans re-processed in BALLISTIC at this step (THEORY 6.2), else 0


@dataclass(frozen=True)
class _Snapshot:
    """the full tracker state after one scan, kept in the short re-processing buffer (THEORY 6.2)."""

    t: float
    x: np.ndarray
    P: np.ndarray
    mode: FilterMode
    nis: tuple
    manoeuvre_on: bool
    status: TrackStatus
    good: int
    bad: int
    locked: bool
    scan: Scan | None  # the scan processed at t (None at the initialisation)
    engine_flag: bool


class Tracker:
    """here I run the EKF track: initialise from the first detection, predict to every scan, point the
    beam at the prediction, update with detections, follow the IR flag between BOOST and BALLISTIC, run
    the safety net and keep the track status. it reads no truth: its inputs are the radar's measurements,
    the IR flag with its specified latency, and its own configuration.
    modes (THEORY 6.2): the mode follows only the IR flag. in BOOST ln beta is frozen, because thrust and
    drag act along nearly the same line and the radar sees only their sum, so beta is unobservable while
    the engine burns: the ln beta row of K is zero and its variance grows by q_beta dt.
    burnout re-processing (THEORY 6.2): the flag reports with the known latency L (a sensor specification,
    like R) and goes off at the first scan with t_k - L >= t_burnout, so it places the burnout only within
    one scan interval T, and the filter has integrated its thrust estimate past the burnout. I keep a short
    buffer of filter states and scans; on the flag change I restore the filter at the last scan at or before
    t_flag - L - T, which is never after the burnout, switch it to BALLISTIC (a = 0, rows and columns of a
    in P zeroed) and re-process the buffered scans and the current one. a restore point after the burnout
    would carry thrust that never happened; the BALLISTIC model can explain that only by more drag (a lower
    ln beta), where the EKF linearisation of exp(-ln beta) collapses sigma(ln beta) onto a wrong value.
    restoring before the burnout misses at most one scan of real thrust instead. the scans in between already
    corrected part of the error, so re-processing them is exact where an analytic correction (subtracting
    a_hat L) would over-correct. the records of the earlier scans keep what the filter reported at their
    time (causal); the current scan's record has the re-processed result.
    t_flag - L - T < t_burnout <= t_flag - L - burnout bracket from the flag
    t_restore = last buffered scan time <= t_flag - L - T - restore point, never after the burnout
    x = x(t_restore), a = 0, P[a, :] = P[:, a] = 0, then predict/update the later scans - re-processing
    this target class burns once, so a flag that turns on again in BALLISTIC is logged as a warning and
    ignored. with boost_mode off the tracker stays in BALLISTIC and ignores the flag (THEORY 6.6 toggle).
    safety net (THEORY 6.6, two-level adjustable process noise, Bar-Shalom et al.): when the window NIS
    exceeds a threshold set by a false-alarm budget per flight, an additional white acceleration with std
    sigma_manoeuvre is switched on for the next predictions; it is switched off when the window NIS is back
    inside. it is independent of q_a, which may be zero. the window test runs after every update, so a
    nominal flight holds N_updates tests; counting every one of them (Bonferroni) bounds the per-flight
    false-alarm probability by p_fa_flight, whatever the correlation between overlapping windows.
    alpha = p_fa_flight / N_updates - false-alarm probability per window test (Bonferroni)
    P(at least one false alarm in a flight) <= N_updates alpha = p_fa_flight - Bonferroni bound
    eps_bar = (1/n) sum of the last n <= N_w NIS values - window NIS
    eps_bar > chi2^-1(1 - alpha, n m) / n -> q_m = sigma_manoeuvre^2 dt, else q_m = 0 - manoeuvre noise"""

    def __init__(self, cfg: FilterConfig, radar: Radar, model: FilterModel, ir_latency: float = 0.0,
                 time_tol: float = ScenarioConfig.time_tol):
        self.cfg = cfg
        self.radar = radar
        self.model = model
        self.latency = ir_latency  # L, specified reporting latency of the IR flag in [s]
        self.scan_interval = radar.cfg.update_interval  # T, time between scans in [s]
        self.time_tol = time_tol  # two times closer than this are one instant (scan grid k T in floating point), in [s]
        self.m = radar.dim  # measurement dimension
        a = 1.0 - cfg.confidence
        n_w = range(1, cfg.nis_window + 1)
        self._window_lo = [chi2.ppf(0.5 * a, n * self.m) / n for n in n_w]  # lower window bounds
        self._window_hi = [chi2.ppf(1.0 - 0.5 * a, n * self.m) / n for n in n_w]  # upper window bounds
        alpha = cfg.p_fa_flight / cfg.nominal_updates  # alpha, false-alarm probability per window test (Bonferroni)
        self._net_hi = [chi2.ppf(1.0 - alpha, n * self.m) / n for n in n_w]  # net switch-on thresholds
        self.nis_lost = float(chi2.ppf(cfg.lost_confidence, self.m))  # single-NIS LOST bound
        self._nis = deque(maxlen=cfg.nis_window)
        self.x = self.P = None
        self.t = None
        self.mode = FilterMode.BALLISTIC
        self.manoeuvre_on = False  # the safety net's state
        self.status = None
        self.events: list[StatusEvent] = []
        self._pending = None
        self._good = 0  # consecutive updates that meet the lock test
        self._bad = 0  # consecutive updates above the LOST NIS bound
        self._locked = False
        self._warned_reignition = False
        self._buffer: deque[_Snapshot] = deque()  # re-processing buffer, about L seconds long

    # status bookkeeping

    def _set_status(self, new: TrackStatus, reason: str) -> None:
        if new is not self.status:
            self.events.append(StatusEvent(self.t, self.status, new, reason))
            self.status = new

    def _log(self, reason: str) -> None:
        self.events.append(StatusEvent(self.t, self.status, self.status, reason))

    def window_bounds(self) -> tuple[float, float]:
        """the 95 % interval of the window NIS for the current window length."""
        n = len(self._nis)
        return self._window_lo[n - 1], self._window_hi[n - 1]

    # the buffer part

    def _snapshot(self, scan: Scan | None, engine_flag: bool) -> _Snapshot:
        return _Snapshot(self.t, self.x.copy(), self.P.copy(), self.mode, tuple(self._nis), self.manoeuvre_on,
                         self.status, self._good, self._bad, self._locked, scan, bool(engine_flag))

    def _restore_time(self, t_flag: float) -> float:
        """the latest time a restore point may have for a flag change at t_flag [s] (class docstring).
        t_restore <= t_flag - L - T - never after the burnout"""
        return t_flag - self.latency - self.scan_interval

    def _remember(self, scan: Scan | None, engine_flag: bool) -> None:
        """keep the state after this scan; drop entries older than the one a flag change now would need."""
        self._buffer.append(self._snapshot(scan, engine_flag))
        while len(self._buffer) >= 2 and self._buffer[1].t <= self._restore_time(self.t) + self.time_tol:
            self._buffer.popleft()

    def _restore(self, snap: _Snapshot) -> None:
        self.t, self.x, self.P, self.mode = snap.t, snap.x.copy(), snap.P.copy(), snap.mode
        self._nis = deque(snap.nis, maxlen=self.cfg.nis_window)
        self.manoeuvre_on, self.status = snap.manoeuvre_on, snap.status
        self._good, self._bad, self._locked = snap.good, snap.bad, snap.locked

    # the mode part

    def _mode_from_flag(self, engine_flag: bool) -> FilterMode:
        return FilterMode.BOOST if (self.cfg.boost_mode and engine_flag) else FilterMode.BALLISTIC

    @staticmethod
    def _to_ballistic(x: np.ndarray, P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x, P = x.copy(), P.copy()
        x[ACC] = 0.0  # a = 0 after burnout
        P[ACC, :] = 0.0
        P[:, ACC] = 0.0
        return x, P

    # the filter steps

    def initialise(self, t: float, z: np.ndarray, engine_flag: bool) -> StepRecord:
        """start the track at the first detection (THEORY 6.5), in the mode the IR flag gives."""
        self.t = t
        self.mode = self._mode_from_flag(engine_flag)
        self.x, self.P = initial_estimate(self.radar, z, self.cfg, self.mode)
        self._set_status(TrackStatus.ACQUIRING, f"track initialised from the first detection, mode {self.mode.value}")
        self._remember(None, engine_flag)
        idx = active_states(self.mode)
        return StepRecord(t, None, None, None, None, None, self.x.copy(), self.P.copy(), math.nan,
                          self.manoeuvre_on, self.mode, bool(engine_flag), False, math.nan, self.status, False,
                          is_positive_definite(self.P[np.ix_(idx, idx)]), 0)

    def predict(self, t: float) -> np.ndarray:
        """EKF prediction from the last estimate to scan time t [s] in the current mode. it returns the
        predicted position, where the beam points for this scan (THEORY 5.4).
        x_{k|k-1} = RK4 propagation of x_{k-1|k-1} - state prediction
        q_a = (sigma_floor + k_drag |a_drag_hat(x_{k-1|k-1})|)^2 dt - drag-scheduled spectral density
        q_m = sigma_manoeuvre^2 dt while the safety net is on, else 0 - manoeuvre noise
        P_{k|k-1} = Phi_k P_{k-1|k-1} Phi_k^T + Q_k(q_a + q_m) - covariance prediction"""
        dt = t - self.t  # scan interval in [s]
        boost = self.mode is FilterMode.BOOST
        x_pred, Phi = self.model.propagate(self.x, dt, boost)  # x_{k|k-1}, Phi_k
        q_a = self.model.spectral_density(self.x, dt)  # q_a from the filter's own drag estimate
        q_m = self.cfg.sigma_manoeuvre ** 2 * dt if self.manoeuvre_on else 0.0  # q_m in [m^2/s^3]
        Q = self.model.process_noise(dt, q_a, boost, q_m)  # Q_k
        P_pred = Phi @ self.P @ Phi.T + Q  # P_{k|k-1}
        P_pred = 0.5 * (P_pred + P_pred.T)
        self._pending = (t, x_pred, P_pred, Phi, Q, q_a, self.manoeuvre_on)
        return x_pred[0:3].copy()

    def pointing(self) -> np.ndarray:
        """the beam axis for the coming scan: from the antenna towards the predicted position.
        u = (p_{k|k-1} - p_radar) / |p_{k|k-1} - p_radar| - beam pointing"""
        d = self._pending[1][0:3] - self.radar.position
        return d / np.linalg.norm(d)

    def process(self, scan: Scan, engine_flag: bool) -> StepRecord:
        """take this scan's IR flag (mode switch), then update with the detection or coast on a miss,
        then run the safety net and the status."""
        wanted = self._mode_from_flag(engine_flag)
        if self.mode is FilterMode.BOOST and wanted is FilterMode.BALLISTIC:
            if self.cfg.reprocess_burnout:
                return self._reprocess(scan, engine_flag)
            t, x_pred, P_pred, *rest = self._pending
            x_pred, P_pred = self._to_ballistic(x_pred, P_pred)
            self._pending = (t, x_pred, P_pred, *rest)
            self.mode = FilterMode.BALLISTIC
            self.t = t
            self._log("mode BOOST -> BALLISTIC: IR flag reports engine off (no re-processing)")
        elif self.mode is FilterMode.BALLISTIC and wanted is FilterMode.BOOST and not self._warned_reignition:
            self._warned_reignition = True
            self._log("warning: IR flag reports engine on in BALLISTIC; ignored (single-burn target)")
        return self._step(scan, engine_flag)

    def _reprocess(self, scan: Scan, engine_flag: bool) -> StepRecord:
        """the burnout re-processing of THEORY 6.2 (class docstring)."""
        t_flag = self._pending[0]  # time of the scan whose flag is off, in [s]
        t_restore = self._restore_time(t_flag)  # t_restore, at or before the burnout, in [s]
        base = next((b for b in reversed(self._buffer) if b.t <= t_restore + self.time_tol), self._buffer[0])
        later = [b for b in self._buffer if b.t > base.t]
        self._restore(base)
        self.x, self.P = self._to_ballistic(self.x, self.P)
        self.mode = FilterMode.BALLISTIC
        self._log(f"mode BOOST -> BALLISTIC: IR flag off at {t_flag:.2f} s, latency {self.latency:.2f} s; "
                  f"restored at {base.t:.2f} s and re-processed {len(later) + 1} scans")
        self._buffer = deque([self._snapshot(base.scan, base.engine_flag)])
        for b in later:
            self.predict(b.t)
            self._step(b.scan, b.engine_flag)
        self.predict(t_flag)
        rec = self._step(scan, engine_flag)
        return dataclasses.replace(rec, reprocessed=len(later) + 1)

    def _step(self, scan: Scan, engine_flag: bool) -> StepRecord:
        """update with the scan's detection, or coast on a miss, in the current mode."""
        t, x_pred, P_pred, Phi, Q, q_a, net_used = self._pending
        self._pending = None
        self.t = t
        frozen = self.mode is FilterMode.BOOST  # beta unobservable while the engine burns
        upd, skipped = None, False
        if scan.detected and self.radar.is_singular(x_pred[0:3]):
            skipped = True  # THEORY 5.6: near the vertical, skip the update and flag it
        elif scan.detected:
            z_pred = self.radar.h(x_pred[0:3], x_pred[3:6])  # h(x_{k|k-1})
            H = np.zeros((self.m, N_STATE))
            H[:, 0:6] = self.radar.jacobian(x_pred[0:3], x_pred[3:6])  # H_k, zero a and ln beta columns
            upd = kalman_update(x_pred, P_pred, scan.z, z_pred, H, self.radar.R, ANGLE_ROWS,
                                frozen_rows=(BETA_ROW,) if frozen else ())
        if upd is None:
            self.x, self.P = x_pred, P_pred
        else:
            self.x, self.P = upd.x, upd.P
            self._nis.append(upd.nis)
        idx = active_states(self.mode)
        pd_ok = is_positive_definite(self.P[np.ix_(idx, idx)])
        if not pd_ok:
            self._log("P is not positive definite")
        window_nis = float(np.mean(self._nis)) if self._nis else math.nan
        if upd is not None:
            self._safety_net(window_nis)
        self._update_status(upd, skipped, scan, window_nis)
        self._remember(scan, engine_flag)
        return StepRecord(t, x_pred, P_pred, Phi, Q, upd, self.x.copy(), self.P.copy(), q_a, net_used,
                          self.mode, bool(engine_flag), frozen and upd is not None, window_nis,
                          self.status, skipped, pd_ok, 0)

    def _safety_net(self, window_nis: float) -> None:
        """the two-level adjustable process noise (class docstring)."""
        if not self.cfg.adaptive_q:
            self.manoeuvre_on = False
            return
        on = window_nis > self._net_hi[len(self._nis) - 1]
        if on != self.manoeuvre_on:
            self._log("manoeuvre noise ON: window NIS above its false-alarm-budget threshold" if on
                      else "manoeuvre noise OFF: window NIS back inside")
        self.manoeuvre_on = on

    def _update_status(self, upd: UpdateResult | None, skipped: bool, scan: Scan, window_nis: float) -> None:
        """the status rules of THEORY 6.7, from the filter's own information only.
        LOCKED: window NIS inside its 95 % interval and sigma_pos < lock bound for M consecutive updates
        COASTING: the last scan gave no update
        LOST: sigma_pos > lost bound, or NIS > chi2^-1(0.999, m) for K consecutive updates"""
        cfg = self.cfg
        sigma = position_sigma_max(self.P)  # sigma_pos in [m]
        if upd is not None:
            lo, hi = self.window_bounds()
            self._good = self._good + 1 if (lo <= window_nis <= hi and sigma < cfg.lock_sigma_pos) else 0
            self._bad = self._bad + 1 if upd.nis > self.nis_lost else 0
        if sigma > cfg.lost_sigma_pos:
            self._locked, self._good = False, 0
            self._set_status(TrackStatus.LOST, f"position sigma {sigma:.0f} m above {cfg.lost_sigma_pos:.0f} m")
        elif self._bad >= cfg.lost_updates:
            self._locked, self._good = False, 0
            self._set_status(TrackStatus.LOST, f"NIS above the {cfg.lost_confidence:.1%} bound for "
                                               f"{self._bad} consecutive updates")
        elif upd is None:
            if self.status is not TrackStatus.LOST:
                why = "update skipped: target near the vertical" if skipped else f"no update this scan ({scan.status.value})"
                self._set_status(TrackStatus.COASTING, why)
        elif self._good >= cfg.lock_updates:
            reason = "re-locked" if self.status is TrackStatus.LOST else (
                f"window NIS inside its 95 % interval and sigma_pos < {cfg.lock_sigma_pos:.0f} m "
                f"for {cfg.lock_updates} updates")
            self._locked = True
            self._set_status(TrackStatus.LOCKED, reason)
        elif self._locked:
            self._set_status(TrackStatus.LOCKED, "update after coasting")
        elif self.status is not TrackStatus.LOST:
            self._set_status(TrackStatus.ACQUIRING, "update, lock test not yet met")
