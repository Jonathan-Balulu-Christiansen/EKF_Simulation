"""checks of the radar (THEORY 5): angle wrapping, h(x) and its inverse, Doppler physics and
sigma_rdot, beam containment, elevation mask, noise statistics, pulse timing, the cue and the first
detection for the slider cases."""

import dataclasses
import math

import numpy as np
import pytest

from ekfsim.config import PhysicsConstants, RadarConfig, ScenarioConfig, VehicleConfig
from ekfsim.radar import (Radar, ScanStatus, cue_direction, doppler_shift, engine_flags, pointing_vector,
                          required_prf, unambiguous_range, velocity_resolution, wavelength,
                          wrap_to_2pi, wrap_to_pi)
from ekfsim.scenario import acquire, plan_scans, simulate_truth

CONST, RCFG, SCN = PhysicsConstants(), RadarConfig(), ScenarioConfig()
RADAR = Radar(RCFG, CONST)
ZERO_NOISE = np.zeros(4)


def target_at(radar, az, el, r):
    """a position at compass azimuth az, elevation el [rad] and slant range r [m] from the antenna."""
    return radar.position + r * pointing_vector(az, el)


def off_axis_target(radar, u, psi, roll, r=100_000.0):
    """a position at off-axis angle psi from the beam axis u, rotated by roll around the axis."""
    e1 = np.cross(u, [0.0, 0.0, 1.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(u, e1)
    direction = math.cos(psi) * u + math.sin(psi) * (math.cos(roll) * e1 + math.sin(roll) * e2)
    return radar.position + r * direction


# the angle wrapping part:

@pytest.mark.parametrize("a, expected", [
    (0.0, 0.0), (math.pi, -math.pi), (-math.pi, -math.pi), (3 * math.pi, -math.pi),
    (math.pi / 2 + 4 * math.pi, math.pi / 2), (-3 * math.pi / 2, math.pi / 2), (math.radians(359), math.radians(-1)),
])
def test_wrap_to_pi(a, expected):
    assert wrap_to_pi(a) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("a, expected", [
    (0.0, 0.0), (2 * math.pi, 0.0), (-math.pi / 2, 3 * math.pi / 2), (5 * math.pi, math.pi),
    (math.radians(-1), math.radians(359)),
])
def test_wrap_to_2pi(a, expected):
    assert wrap_to_2pi(a) == pytest.approx(expected, abs=1e-12)


def test_wrap_ranges_hold_at_rounding_edges():
    """tiny negative angles: mod rounds to exactly 2 pi, which must come back as 0 (and -pi, not pi)."""
    tiny = np.array([-1e-17, -1e-300, 0.0, 1e-17])
    w2 = wrap_to_2pi(tiny)
    assert np.all((w2 >= 0.0) & (w2 < 2 * math.pi))
    w1 = wrap_to_pi(tiny - math.pi)
    assert np.all((w1 >= -math.pi) & (w1 < math.pi))


def test_wrap_to_pi_on_arrays():
    a = np.linspace(-20.0, 20.0, 1001)
    w = wrap_to_pi(a)
    assert np.all((w >= -math.pi) & (w < math.pi))
    assert np.allclose(np.sin(w), np.sin(a)) and np.allclose(np.cos(w), np.cos(a))


# the measurement function part:

@pytest.mark.parametrize("dE, dN, az_deg", [(0, 1, 0), (1, 0, 90), (0, -1, 180), (-1, 0, 270),
                                            (1, 1, 45), (-1, -1, 225)])
def test_compass_azimuth(dE, dN, az_deg):
    p = RADAR.position + 50_000.0 * np.array([dE, dN, 0.2])
    assert math.degrees(RADAR.h(p, np.zeros(3))[1]) == pytest.approx(az_deg, abs=1e-12)


def test_h_then_inverse_returns_the_position():
    rng = np.random.default_rng(1)
    p = np.column_stack([rng.uniform(-300e3, 300e3, 500), rng.uniform(-300e3, 300e3, 500),
                         rng.uniform(0.0, 150e3, 500)])
    v = rng.normal(0.0, 1_000.0, (500, 3))
    p_back = RADAR.position_from_measurement(RADAR.h(p, v))
    assert np.max(np.abs(p_back - p)) < 1e-8 * 300e3
    # one row at a time gives the same answer
    assert np.allclose(RADAR.h(p[7], v[7]), RADAR.h(p, v)[7], rtol=0, atol=1e-12)


def test_range_rate_is_derivative_of_range():
    p, v, dt = np.array([-120e3, -60e3, 40e3]), np.array([900.0, 50.0, -300.0]), 1e-3
    r = lambda q: RADAR.h(q, v)[0]  # noqa: E731
    rdot_fd = (r(p + v * dt) - r(p - v * dt)) / (2 * dt)  # central difference
    assert RADAR.h(p, v)[3] == pytest.approx(rdot_fd, rel=1e-8)


@pytest.mark.parametrize("az_target_deg, az_meas_deg, expected_deg", [
    (1.0, 359.0, -2.0),   # target just east of North, measurement just west: -2 deg, not +358 deg
    (359.0, 1.0, 2.0),    # the mirror case
    (0.5, 359.5, -1.0),
])
def test_azimuth_innovation_across_north(az_target_deg, az_meas_deg, expected_deg):
    """h(x) gives compass azimuths on both sides of North, and the wrapped difference is the short way."""
    p = target_at(RADAR, math.radians(az_target_deg), math.radians(20.0), 100e3)
    az_pred = RADAR.h(p, np.zeros(3))[1]
    assert math.degrees(az_pred) == pytest.approx(az_target_deg, abs=1e-9)
    nu = wrap_to_pi(math.radians(az_meas_deg) - az_pred)
    assert math.degrees(nu) == pytest.approx(expected_deg, abs=1e-9)


def test_h_is_continuous_across_north():
    """a target crossing North: h jumps from 359.99 to 0.01 deg, the wrapped step stays 0.02 deg."""
    dE = np.array([-35.0, 35.0])  # [m] west and east of the North axis at 100 km
    p = np.column_stack([dE, [100e3, 100e3], [20e3, 20e3]]) + RADAR.position
    az = RADAR.h(p, np.zeros((2, 3)))[:, 1]
    assert math.degrees(az[0]) > 359.9 and math.degrees(az[1]) < 0.1
    assert abs(wrap_to_pi(az[1] - az[0])) < math.radians(0.05)


def test_measurement_jacobian_matches_finite_differences():
    """THEORY 6.3: analytic H against central differences at random states, row-relative error < 1e-5."""
    rng = np.random.default_rng(4)
    for _ in range(200):
        p = np.array([rng.uniform(-2e5, 2e5), rng.uniform(-2e5, 2e5), rng.uniform(1e3, 1.5e5)])
        v = rng.uniform(-2_000.0, 2_000.0, 3)
        x = np.r_[p, v]
        H = RADAR.jacobian(p, v)
        step = 1e-6 * np.maximum(np.abs(x), 1.0)
        cols = []
        for j in range(6):
            e = np.zeros(6)
            e[j] = step[j]
            d = RADAR.h((x + e)[:3], (x + e)[3:]) - RADAR.h((x - e)[:3], (x - e)[3:])
            d[1] = wrap_to_pi(d[1])
            cols.append(d / (2 * step[j]))
        H_fd = np.column_stack(cols)
        for i in range(4):
            assert np.max(np.abs(H[i] - H_fd[i])) / np.max(np.abs(H_fd[i])) < 1e-5


def test_doppler_off_gives_three_components():
    radar3 = Radar(dataclasses.replace(RCFG, doppler_enabled=False), CONST)
    assert radar3.dim == 3 and radar3.R.shape == (3, 3)
    assert radar3.h(np.array([1e4, 2e4, 3e3]), np.ones(3)).shape == (3,)


# the Doppler physics part:

def test_sigma_rdot_derivation():
    lam = wavelength(RCFG.carrier_frequency, CONST.c)
    assert lam == pytest.approx(0.03, rel=1e-3)  # THEORY 5.1: 3.0 cm
    assert velocity_resolution(lam, RCFG.dwell_time) == pytest.approx(lam / (2 * RCFG.dwell_time), rel=1e-15)
    assert RADAR.sigma_rdot == pytest.approx(1.5, rel=1e-3)  # THEORY 5.3 default
    assert np.allclose(np.diag(RADAR.R), [10.0 ** 2, math.radians(0.1) ** 2, math.radians(0.1) ** 2,
                                         RADAR.sigma_rdot ** 2], rtol=1e-14)


def test_doppler_sign_closing_is_positive():
    lam = RADAR.wavelength
    assert doppler_shift(-100.0, lam) > 0.0 and doppler_shift(100.0, lam) < 0.0
    # f_d = 2 v_r f_c / c with v_r = -rdot
    assert doppler_shift(-100.0, lam) == pytest.approx(2 * 100.0 * RCFG.carrier_frequency / CONST.c, rel=1e-14)


def test_prf_and_unambiguous_range():
    assert required_prf(100e3) == 200e3
    assert unambiguous_range(200e3, CONST.c) == pytest.approx(749.48, abs=0.01)


# the beam part:

@pytest.mark.parametrize("roll", np.linspace(0.0, 2 * math.pi, 7)[:-1])
def test_beam_containment(roll):
    u = pointing_vector(math.radians(200.0), math.radians(20.0))
    half = RADAR.half_beamwidth
    p_in = off_axis_target(RADAR, u, 0.99 * half, roll)
    p_out = off_axis_target(RADAR, u, 1.01 * half, roll)
    assert RADAR.off_axis_angle(p_in, u) == pytest.approx(0.99 * half, rel=1e-9)
    assert RADAR.scan(0, 0.0, p_in, np.zeros(3), u, ZERO_NOISE).status is ScanStatus.DETECTED
    assert RADAR.scan(0, 0.0, p_out, np.zeros(3), u, ZERO_NOISE).status is ScanStatus.OUTSIDE_BEAM


def test_elevation_mask():
    """the beam points straight at the target; only the mask decides."""
    mask = RCFG.elevation_mask
    for el, status in ((mask - math.radians(0.1), ScanStatus.BELOW_MASK),
                       (mask + math.radians(0.1), ScanStatus.DETECTED)):
        p = target_at(RADAR, math.radians(250.0), el, 150e3)
        u = pointing_vector(math.radians(250.0), el)
        s = RADAR.scan(0, 0.0, p, np.zeros(3), u, ZERO_NOISE)
        assert s.status is status
        assert (s.z is None) == (status is not ScanStatus.DETECTED)


def test_detection_timing_and_noise_free_measurement():
    u = pointing_vector(1.0, 0.3)  # the beam is set first, the target is placed on its axis
    p, v = RADAR.position + 160e3 * u, np.array([800.0, -200.0, -100.0])
    s = RADAR.scan(5, 0.5, p, v, u, ZERO_NOISE)
    z_true = RADAR.h(p, v)
    assert np.allclose(s.z, z_true, rtol=0, atol=1e-9)
    assert s.delay == pytest.approx(2 * 160e3 / CONST.c, rel=1e-12)  # about 1.07 ms
    assert s.t_rx == s.t_tx + s.delay
    assert s.doppler_shift == pytest.approx(doppler_shift(z_true[3], RADAR.wavelength), rel=1e-12)
    with pytest.raises(ValueError):
        s.z[0] = 0.0  # measurements are read-only


def test_noise_statistics():
    """measurement residuals have zero mean and the std of R (fixed seed, so not flaky)."""
    n = 20_000
    u = pointing_vector(math.radians(359.9), math.radians(30.0))  # beam first, target on its axis
    p, v = RADAR.position + 120e3 * u, np.array([500.0, 0.0, 0.0])
    w = RADAR.noise(n, np.random.default_rng(3))
    z = np.array([RADAR.scan(k, 0.0, p, v, u, w[k]).z for k in range(n)])
    res = z - RADAR.h(p, v)
    res[:, 1] = wrap_to_pi(res[:, 1])  # the azimuth straddles North here
    sig = np.sqrt(np.diag(RADAR.R))
    assert np.all(np.abs(res.mean(axis=0)) < 4 * sig / math.sqrt(n))
    assert np.allclose(res.std(axis=0), sig, rtol=0.03)


def test_singularity_guard():
    assert RADAR.is_singular(RADAR.position + np.array([0.3, 0.4, 50e3]))  # 0.5 m off the vertical
    assert not RADAR.is_singular(RADAR.position + np.array([1_000.0, 0.0, 50e3]))


def test_invalid_radar_settings_are_refused():
    with pytest.raises(ValueError, match="dwell"):
        Radar(dataclasses.replace(RCFG, dwell_time=0.2), CONST)


# the cue and acquisition part:

def test_cue_direction():
    az, el = cue_direction(RCFG, SCN)
    assert math.degrees(az) == pytest.approx(math.degrees(math.atan2(-150.0, -60.0)) + 360.0, abs=1e-12)
    assert el - 0.5 * RCFG.beamwidth == pytest.approx(RCFG.elevation_mask, abs=1e-15)  # lower edge on the mask


@pytest.mark.parametrize("delta", [-1_000.0, 0.0, 3_000.0])
def test_first_detection_for_slider_cases(delta):
    """THEORY 5.4: with the cue, a first detection occurs for all three slider cases."""
    truth = simulate_truth(dataclasses.replace(VehicleConfig(), dry_mass_delta=delta))
    plan = plan_scans(truth, RADAR, SCN)
    acq = acquire(truth, RADAR, plan, RCFG, SCN)
    assert acq.acquired and acq.reason == ""
    first = acq.first_detection
    assert all(not s.detected for s in acq.scans[:-1]) and acq.scans[-1] is first
    u_cue = pointing_vector(acq.cue_azimuth, acq.cue_elevation)
    assert all(np.allclose(s.pointing, u_cue) for s in acq.scans)
    i = plan.truth_index[first.k]
    assert RADAR.off_axis_angle(truth.position[i], u_cue) <= RADAR.half_beamwidth
    assert RADAR.h(truth.position[i], truth.velocity[i])[2] > RCFG.elevation_mask


def test_no_acquisition_gives_reason():
    truth = simulate_truth(dataclasses.replace(VehicleConfig(), dry_mass_delta=3_000.0))  # max el about 5 deg
    rcfg = dataclasses.replace(RCFG, elevation_mask=math.radians(10.0))
    radar = Radar(rcfg, CONST)
    acq = acquire(truth, radar, plan_scans(truth, radar, SCN), rcfg, SCN)
    assert not acq.acquired and "never entered the cued beam" in acq.reason
    assert len(acq.scans) == len(plan_scans(truth, radar, SCN).t)


def test_scan_times_stay_on_grid_around_split_truth_steps():
    """the truth integrator splits steps at the pitch corners (5 s, 50 s) and at burnout (62.926 s).
    the scans around those events must stay exactly on the k T_update grid and read the truth sample
    of that very instant, never the extra event sample."""
    truth = simulate_truth()
    plan = plan_scans(truth, RADAR, SCN)
    T = RCFG.update_interval
    assert np.array_equal(plan.t, np.arange(len(plan.t)) * T)  # t_k = k T exactly, no drift
    for t_e, i_e in zip(truth.event_times, truth.event_index):
        near = np.nonzero(np.abs(plan.t - t_e) <= 3 * T)[0]
        assert len(near) >= 6
        for k in near:
            i = plan.truth_index[k]
            assert abs(truth.t[i] - plan.t[k]) < 1e-12
            on_grid = abs(t_e - round(t_e / T) * T) < 1e-12
            assert i != i_e or on_grid  # an off-grid event sample is never read by a scan
        # the scan just before burnout sees thrust, the one just after sees none
        if t_e == truth.event_times[-1]:
            k_before = int(np.floor(t_e / T))
            assert truth.boost[plan.truth_index[k_before]] and not truth.boost[plan.truth_index[k_before + 1]]


def test_scan_plan_on_truth_grid():
    truth = simulate_truth()
    plan = plan_scans(truth, RADAR, SCN)
    assert np.allclose(truth.t[plan.truth_index], plan.t, rtol=0, atol=SCN.time_tol)
    assert plan.t[-1] <= truth.t[-1] < plan.t[-1] + RCFG.update_interval
    assert plan.noise.shape == (len(plan.t), 4)
    same = plan_scans(truth, RADAR, SCN)
    assert np.array_equal(same.noise, plan.noise)  # reproducible from the seed
    with pytest.raises(ValueError, match="whole multiple"):
        plan_scans(truth, Radar(dataclasses.replace(RCFG, update_interval=0.015), CONST), SCN)


# the IR plume sensor part:

def test_engine_flag_is_the_delayed_engine_state():
    """THEORY 5.7: one flag per scan, the true engine state latency earlier (single burn)."""
    t = np.arange(0, 1000) * 0.1
    flags = engine_flags(t, 0.0, 62.926, 0.5)
    assert not flags[t < 0.5].any() and flags[(t >= 0.5) & (t < 63.426)].all()
    assert not flags[t >= 63.426 + 1e-9].any()
    assert flags[np.searchsorted(t, 63.4 - 1e-9)] and not flags[np.searchsorted(t, 63.5 - 1e-9)]


def test_scan_plan_carries_the_ir_flag():
    truth = simulate_truth()
    plan = plan_scans(truth, RADAR, SCN)
    t_off = truth.t[truth.burnout_index] + 0.5
    assert np.array_equal(plan.engine_flag, (plan.t >= 0.5) & (plan.t < t_off))
