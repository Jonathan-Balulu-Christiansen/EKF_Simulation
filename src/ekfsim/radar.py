"""here I build the radar (THEORY 5): geometry, the measurement function h(x), the noise, the Doppler
physics, the beam and the pulse timing, plus the IR plume flag of the separate engine sensor (THEORY 5.7). this module is the measurement generator, one of the two
places allowed to read truth (THEORY 0.2). the beam pointing it receives must never come from truth:
it comes from the pre-track cue or, once a track exists, from the tracker's prediction.
assumptions, all documented:
- h(x) is evaluated at the transmit time t_tx and light time is neglected. the target moves at most
  about 1 m during r/c = 0.5 ms at 160 km, against sigma_r = 10 m.
- range-rate is unambiguous (THEORY 5.3). the PRF that would need is computed for information only.
- no clutter, no false alarms, Pd = 1 inside the beam and above the elevation mask (THEORY 5.4).
- R is fixed (no SNR dependence, THEORY 1)."""

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np

from ekfsim.config import PhysicsConstants, RadarConfig, ScenarioConfig

TWO_PI = 2.0 * math.pi


# the angle wrapping part:

def wrap_to_pi(a):
    """this function wraps an angle [rad] (scalar or array) to [-pi, pi). I use it for angle
    innovations and residuals, so a difference across North never looks like a 360 deg jump.
    a_w = ((a + pi) mod 2 pi) - pi - angle wrapped to [-pi, pi)"""
    w = np.mod(np.asarray(a, dtype=float) + math.pi, TWO_PI)  # (a + pi) mod 2 pi
    w = np.where(w >= TWO_PI, w - TWO_PI, w)  # mod can round up to exactly 2 pi; that is 0
    w = w - math.pi  # a_w in [-pi, pi)
    return float(w) if w.ndim == 0 else w


def wrap_to_2pi(a):
    """this function wraps an angle [rad] (scalar or array) to [0, 2 pi), the compass range of the
    azimuth (THEORY 3).
    a_w = a mod 2 pi - angle wrapped to [0, 2 pi)"""
    w = np.mod(np.asarray(a, dtype=float), TWO_PI)  # a mod 2 pi
    w = np.where(w >= TWO_PI, w - TWO_PI, w)  # mod can round up to exactly 2 pi; that is 0
    return float(w) if w.ndim == 0 else w


# the Doppler physics part:

def wavelength(f_c: float, c: float) -> float:
    """carrier wavelength in [m] from f_c [Hz] and c [m/s].
    lambda = c / f_c - wavelength"""
    return c / f_c  # lambda in [m]


def doppler_shift(rdot, lam: float):
    """Doppler shift [Hz] of the echo for range-rate rdot [m/s] and wavelength lam [m]. sign convention:
    a closing target (rdot < 0) gives a positive f_d.
    v_r = -rdot - closing (radial) speed
    f_d = 2 v_r / lambda = 2 v_r f_c / c - two-way Doppler shift"""
    return -2.0 * rdot / lam  # f_d in [Hz], works for floats and numpy arrays


def velocity_resolution(lam: float, dwell_time: float) -> float:
    """the range-rate resolution of one coherent dwell, which I use as the range-rate noise std. this
    is conservative: real frequency estimators do better than one resolution cell at high SNR.
    delta_f = 1 / T_dwell - Doppler frequency resolution
    delta_v = lambda / 2 * delta_f = lambda / (2 T_dwell) - range-rate resolution
    sigma_rdot := delta_v - range-rate noise std"""
    delta_f = 1.0 / dwell_time  # delta_f is the frequency resolution in [Hz]
    return 0.5 * lam * delta_f  # delta_v = sigma_rdot in [m/s]


def required_prf(max_abs_fd: float) -> float:
    """the PRF [Hz] needed for unambiguous signed Doppler, from the largest |f_d| [Hz] of a run.
    rdot changes sign during the flight, and with I/Q sampling the unambiguous Doppler interval is
    [-PRF/2, PRF/2), so the PRF must cover twice the largest shift (information only, THEORY 5.3).
    PRF > 2 |f_d|_max - unambiguous signed Doppler"""
    return 2.0 * max_abs_fd  # lower bound of the PRF in [Hz]


def unambiguous_range(prf: float, c: float) -> float:
    """the longest range [m] whose echo returns before the next pulse leaves, at pulse rate prf [Hz].
    R_ua = c / (2 PRF) - unambiguous range"""
    return c / (2.0 * prf)  # R_ua in [m]


# the beam part:

def pointing_vector(az: float, el: float) -> np.ndarray:
    """unit vector in ENU for compass azimuth az [rad] and elevation el [rad].
    u = [cos(el) sin(az), cos(el) cos(az), sin(el)] - beam axis direction"""
    return np.array([math.cos(el) * math.sin(az), math.cos(el) * math.cos(az), math.sin(el)])


def cue_direction(radar_cfg: RadarConfig, scenario_cfg: ScenarioConfig) -> tuple[float, float]:
    """here I compute the pre-track cue of THEORY 5.4, in [rad]: an assumed early-warning cue of the
    launch site. it uses config only, never truth. at ground level the beam would cover only
    elevations below the mask and nothing could ever be detected, so the lower beam edge sits on the mask.
    az_cue = atan2(E_launch - E_radar, N_launch - N_radar) - launch-site azimuth from the radar
    el_cue = el_mask + beamwidth / 2 - cue elevation"""
    az = wrap_to_2pi(math.atan2(scenario_cfg.launch_east - radar_cfg.east,
                                scenario_cfg.launch_north - radar_cfg.north))  # az_cue in [rad]
    el = radar_cfg.elevation_mask + 0.5 * radar_cfg.beamwidth  # el_cue in [rad]
    return az, el


def validate_radar(cfg: RadarConfig) -> None:
    """this function refuses radar settings that make no physical sense and says why (ValueError)."""
    if cfg.carrier_frequency <= 0.0 or cfg.pulse_width <= 0.0 or cfg.dwell_time <= 0.0:
        raise ValueError("f_c, pulse width and dwell time must be positive")
    if cfg.dwell_time > cfg.update_interval:
        raise ValueError(f"the dwell ({cfg.dwell_time * 1e3:.1f} ms) cannot be longer than the time "
                         f"between scans ({cfg.update_interval * 1e3:.1f} ms)")
    if cfg.pulse_width >= cfg.dwell_time:
        raise ValueError("the pulse width must be shorter than the dwell")
    if not 0.0 < cfg.beamwidth < math.pi:
        raise ValueError("the beamwidth must lie between 0 and 180 deg")
    if not 0.0 <= cfg.elevation_mask < 0.5 * math.pi:
        raise ValueError("the elevation mask must lie in [0, 90) deg")
    if cfg.sigma_range <= 0.0 or cfg.sigma_angle <= 0.0 or cfg.rho_min <= 0.0:
        raise ValueError("sigma_r, sigma_angle and rho_min must be positive")



# the IR plume sensor part:

def engine_flags(t_scan, t_ignition: float, t_burnout: float, latency: float) -> np.ndarray:
    """here I generate the IR plume flag of THEORY 5.7, one per radar scan, from the true engine times
    (this is the measurement layer, which may read truth; the filter only ever sees the flag). the
    sensor reports with a delay, so the flag changes latency [s] after the true ignition and burnout.
    the target class has a single burn (THEORY 6.2), so the engine is on exactly in [t_ignition, t_burnout).
    flag_k = (t_ignition <= t_k - latency < t_burnout) - delayed engine on/off flag"""
    t_seen = np.asarray(t_scan, dtype=float) - latency  # time the reported plume state belongs to, in [s]
    return (t_seen >= t_ignition) & (t_seen < t_burnout)


# the scan result part:

class ScanStatus(Enum):
    """the outcome of one scan."""

    DETECTED = "detection"
    OUTSIDE_BEAM = "miss: target outside beam"
    BELOW_MASK = "miss: target below elevation mask"


@dataclass(frozen=True)
class Scan:
    """one scan (one dwell). a miss carries no measurement and no echo timing: no echo comes back.
    the timing quantities drive the pulse view (THEORY 5.5)."""

    k: int  # scan index, emission time t_tx = k T_update
    t_tx: float  # pulse emission time in [s]
    pointing: tuple[float, float, float]  # beam axis unit vector u in ENU
    status: ScanStatus
    z: np.ndarray | None  # measurement [r, az, el(, rdot)] in [m, rad, rad, m/s], read-only
    delay: float | None  # two-way delay 2r/c in [s]
    t_rx: float | None  # echo arrival time t_tx + 2r/c in [s]
    doppler_shift: float | None  # f_d of the echo in [Hz], closing positive

    @property
    def detected(self) -> bool:
        return self.status is ScanStatus.DETECTED


# the radar and measurement function part:

class Radar:
    """here I hold one radar: its antenna position and everything derived from RadarConfig (SI units).
    R = diag(sigma_r^2, sigma_az^2, sigma_el^2, sigma_rdot^2) - fixed measurement noise covariance"""

    def __init__(self, cfg: RadarConfig, constants: PhysicsConstants):
        validate_radar(cfg)
        self.cfg = cfg
        self.c = constants.c  # speed of light in [m/s]
        self.position = np.array([cfg.east, cfg.north, cfg.antenna_height])  # p_radar, antenna phase centre in [m]
        self.wavelength = wavelength(cfg.carrier_frequency, constants.c)  # lambda in [m]
        self.sigma_rdot = velocity_resolution(self.wavelength, cfg.dwell_time)  # sigma_rdot in [m/s]
        self.pulse_length = constants.c * cfg.pulse_width  # c tau, physical pulse length in [m]
        self.half_beamwidth = 0.5 * cfg.beamwidth  # cone half-angle in [rad]
        self.dim = 4 if cfg.doppler_enabled else 3  # measurement dimension
        self._sigma4 = np.array([cfg.sigma_range, cfg.sigma_angle, cfg.sigma_angle, self.sigma_rdot])
        self.R = np.diag(self._sigma4[:self.dim] ** 2)  # R, measurement noise covariance

    def h(self, p, v) -> np.ndarray:
        """this is the measurement function h(x) of THEORY 5.2, without noise. p [m] and v [m/s] are
        (3,) or (N, 3); the result is [r, az, el, rdot] (or [r, az, el] without Doppler) per row.
        d = p - p_radar - line of sight from the antenna
        r = |d| - slant range
        az = atan2(dE, dN) wrapped to [0, 2 pi) - compass azimuth, clockwise from North
        el = atan2(dU, sqrt(dE^2 + dN^2)) - elevation above the horizon
        rdot = (d . v) / r - range-rate, positive when opening"""
        d = np.asarray(p, dtype=float) - self.position  # d is the line of sight in [m]
        r = np.linalg.norm(d, axis=-1)  # r is the slant range in [m]
        az = wrap_to_2pi(np.arctan2(d[..., 0], d[..., 1]))  # az in [rad]
        el = np.arctan2(d[..., 2], np.hypot(d[..., 0], d[..., 1]))  # el in [rad]
        cols = [r, az, el]
        if self.dim == 4:
            cols.append(np.sum(d * np.asarray(v, dtype=float), axis=-1) / r)  # rdot in [m/s]
        return np.stack(cols, axis=-1)

    def jacobian(self, p, v) -> np.ndarray:
        """this is H = dh/d[p, v], shape (dim, 6), at position p [m] and velocity v [m/s] (THEORY 6.4).
        the filter pads it with a zero column for ln beta. the range-rate row is the only one with
        velocity columns. it needs rho > 0; the singularity guard (THEORY 5.6) keeps the filter away from rho = 0.
        dr/dp = d^T / r - range row
        daz/dp = [dN, -dE, 0] / rho^2 - azimuth row
        del/dp = [-dE dU, -dN dU, rho^2] / (r^2 rho) - elevation row
        drdot/dp = (v - rdot d / r)^T / r - range-rate row, position part
        drdot/dv = d^T / r - range-rate row, velocity part"""
        d = np.asarray(p, dtype=float) - self.position  # d is the line of sight in [m]
        v = np.asarray(v, dtype=float)
        dE, dN, dU = d.tolist()
        rho2 = dE * dE + dN * dN  # rho^2, squared horizontal range in [m^2]
        rho = math.sqrt(rho2)  # rho in [m]
        r2 = rho2 + dU * dU  # r^2 in [m^2]
        r = math.sqrt(r2)  # r in [m]
        H = np.zeros((self.dim, 6))
        H[0, 0:3] = d / r  # dr/dp
        H[1, 0:3] = (dN / rho2, -dE / rho2, 0.0)  # daz/dp
        H[2, 0:3] = (-dE * dU / (r2 * rho), -dN * dU / (r2 * rho), rho / r2)  # del/dp
        if self.dim == 4:
            rdot = float(d @ v) / r  # rdot in [m/s]
            H[3, 0:3] = (v - rdot * d / r) / r  # drdot/dp
            H[3, 3:6] = d / r  # drdot/dv
        return H

    def position_from_measurement(self, z) -> np.ndarray:
        """this is the inverse of the position part of h: it turns [r, az, el, ...] ((M,) or (N, M))
        into an ENU position [m]. used for the initialisation (THEORY 6.5) and to draw measurements in
        3D ("converted for display only").
        d = r [cos(el) sin(az), cos(el) cos(az), sin(el)] - spherical to ENU
        p = p_radar + d - position"""
        z = np.asarray(z, dtype=float)
        r, az, el = z[..., 0], z[..., 1], z[..., 2]
        d = np.stack([r * np.cos(el) * np.sin(az), r * np.cos(el) * np.cos(az), r * np.sin(el)],
                     axis=-1)  # d is the line of sight in [m]
        return self.position + d

    def horizontal_range(self, p) -> float:
        """rho = sqrt(dE^2 + dN^2) - horizontal distance from the antenna in [m]"""
        d = np.asarray(p, dtype=float) - self.position
        return float(math.hypot(d[0], d[1]))

    def is_singular(self, p) -> bool:
        """the singularity guard of THEORY 5.6. near the vertical, az and the 1/rho terms of the Jacobian
        blow up, so an update there is skipped. the filter applies this to its own predicted state.
        rho < rho_min - skip the update"""
        return self.horizontal_range(p) < self.cfg.rho_min

    def off_axis_angle(self, p, u) -> float:
        """the angle [rad] between the line of sight to p [m] and the beam axis u (unit vector).
        atan2 of cross and dot products stays accurate near 0, unlike acos.
        psi = atan2(|d x u|, d . u) - off-axis angle"""
        d = np.asarray(p, dtype=float) - self.position
        return math.atan2(float(np.linalg.norm(np.cross(d, u))), float(np.dot(d, u)))  # psi in [rad]

    def noise(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """n draws of the measurement noise, one row per scan, shape (n, 4) in [m, rad, rad, m/s].
        all four components are drawn even without Doppler, so one seed gives the same r, az, el noise
        with Doppler on or off.
        w_k ~ N(0, diag(sigma_r^2, sigma_az^2, sigma_el^2, sigma_rdot^2)) - white Gaussian noise"""
        return rng.standard_normal((n, 4)) * self._sigma4

    def scan(self, k: int, t_tx: float, p, v, pointing, w) -> Scan:
        """this is one scan of the measurement generator (THEORY 5.2, 5.4, 5.5). p [m] and v [m/s] are
        the TRUE target state at t_tx, which is allowed here; pointing is the beam axis from the cue
        or the tracker, never from truth; w is this scan's noise row.
        detection: psi <= beamwidth / 2 and el > el_mask - beam and mask test
        z = h(x) + w, az wrapped to [0, 2 pi) - noisy measurement
        2 r / c - two-way delay
        t_rx = t_tx + 2 r / c - echo arrival time
        f_d = -2 rdot / lambda - Doppler shift of the echo"""
        u = tuple(float(c) for c in pointing)
        z_true = self.h(p, v)
        if z_true[2] <= self.cfg.elevation_mask:
            return Scan(k, t_tx, u, ScanStatus.BELOW_MASK, None, None, None, None)
        if self.off_axis_angle(p, u) > self.half_beamwidth:
            return Scan(k, t_tx, u, ScanStatus.OUTSIDE_BEAM, None, None, None, None)
        z = z_true + np.asarray(w, dtype=float)[:self.dim]  # z = h(x) + w
        z[1] = wrap_to_2pi(z[1])  # keep the measured azimuth in [0, 2 pi)
        z.setflags(write=False)
        r = float(z_true[0])
        delay = 2.0 * r / self.c  # two-way delay in [s]
        rdot = float(np.dot(np.asarray(p, float) - self.position, np.asarray(v, float))) / r  # true rdot in [m/s]
        return Scan(k, t_tx, u, ScanStatus.DETECTED, z, delay, t_tx + delay,
                    float(doppler_shift(rdot, self.wavelength)))
