"""checks of the EKF (THEORY 6): analytic Jacobian A in both modes, discretisation, the
linear case where the EKF must reduce to the standard KF, the update with azimuth crossing North, the
BOOST/BALLISTIC modes driven by the IR flag, the initialisation, adaptive Q, the track status rules, the
singularity guard, and consistency on matched models."""

import ast
import dataclasses
import math
import pathlib

import numpy as np
import pytest
from scipy.linalg import expm
from scipy.stats import chi2

from ekfsim.atmosphere import US76Atmosphere
from ekfsim.config import (
    AtmosphereConfig,
    FilterConfig,
    PhysicsConstants,
    RadarConfig,
    ScenarioConfig,
    VehicleConfig,
)
from ekfsim.ekf import (
    ACC,
    ACTIVE_BALLISTIC,
    ANGLE_ROWS,
    BETA_ROW,
    N_STATE,
    FilterMode,
    FilterModel,
    Tracker,
    TrackStatus,
    initial_estimate,
    is_positive_definite,
    kalman_update,
)
from ekfsim.integrators import rk4_step
from ekfsim.metrics import average_interval, nees
from ekfsim.radar import Radar, Scan, ScanStatus, pointing_vector, wrap_to_pi
from ekfsim.scenario import simulate_truth
from ekfsim.vehicle import TruthModel

CONST, ATM, FCFG, RCFG = PhysicsConstants(), AtmosphereConfig(), FilterConfig(), RadarConfig()
MODEL = FilterModel(FCFG, CONST, ATM)
RADAR = Radar(RCFG, CONST)
US76 = US76Atmosphere(ATM, CONST)
SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "ekfsim"


def random_state(rng):
    return np.array([rng.uniform(-2e5, 2e5), rng.uniform(-2e5, 2e5), rng.uniform(2e3, 8e4),
                     *rng.uniform(-2_000.0, 2_000.0, 3), *rng.uniform(-50.0, 50.0, 3),
                     math.log(rng.uniform(1e3, 1e5))])


def detection(z, k=1, t=0.1):
    z = np.asarray(z, dtype=float)
    z.setflags(write=False)
    return Scan(k, t, (0.0, 0.0, 1.0), ScanStatus.DETECTED, z, None, None, None)


MISS = Scan(1, 0.1, (0.0, 0.0, 1.0), ScanStatus.OUTSIDE_BEAM, None, None, None, None)


def state(p, v, a=(0.0, 0.0, 0.0), beta=5_000.0):
    return np.r_[p, v, a, math.log(beta)]


# the filter model part:

@pytest.mark.parametrize("boost", [True, False])
def test_filter_jacobian_matches_finite_differences(boost):
    """THEORY 6.3: analytic A against central differences, max column-relative error < 1e-5, in both modes.
    columns that are zero by construction (pE, pN on a flat Earth; a in BALLISTIC) must be exactly zero."""
    rng = np.random.default_rng(0)
    for _ in range(200):
        x = random_state(rng)
        A = MODEL.jacobian(x, boost)
        step = 1e-6 * np.maximum(np.abs(x), 1.0)
        A_fd = np.column_stack([(MODEL.f(x + s * e, boost) - MODEL.f(x - s * e, boost)) / (2 * s)
                                for s, e in zip(step, np.eye(N_STATE))])
        for j in range(N_STATE):
            scale = np.max(np.abs(A_fd[:, j]))
            if scale == 0.0:
                assert np.all(A[:, j] == 0.0)
            else:
                assert np.max(np.abs(A[:, j] - A_fd[:, j])) / scale < 1e-5
    x = random_state(rng)
    assert np.array_equal(MODEL.jacobian(x, True)[3:6, ACC], np.eye(3))  # dv'/da = I in BOOST
    assert np.all(MODEL.jacobian(x, False)[:, ACC] == 0.0)


def test_isa_density_gradient():
    """the filter's density gradient (ISA, THEORY 6.2) against central differences away from the kinks."""
    for h in np.linspace(-200.0, 120e3, 400):
        rho, g = US76.density_and_gradient(h)
        assert rho == US76.density(h)
        fd = (US76.density(h + 0.01) - US76.density(h - 0.01)) / 0.02
        assert g == pytest.approx(fd, rel=1e-6)


def test_transition_matrix_is_second_order():
    """Phi from I + A dt + (A dt)^2 / 2 matches the matrix exponential to third order in dt."""
    x = state([-50e3, -60e3, 15e3], [900.0, 0.0, -700.0], [20.0, 0.0, 15.0], 8_000.0)
    errs = []
    for dt in (0.1, 0.05):
        model = FilterModel(dataclasses.replace(FCFG, propagation_substep=dt), CONST, ATM)
        _, Phi = model.propagate(x, dt, True)
        errs.append(np.max(np.abs(Phi - expm(MODEL.jacobian(x, True) * dt))))
    assert errs[0] < 1e-3 and errs[1] / errs[0] < 0.2  # third order up to the change of A along the step


def test_propagation_substeps_converge():
    """the state prediction is converged at the default sub-step; Phi converges as the sub-step
    shrinks (A is taken at the start of each sub-step) and is within 0.2 % at the default."""
    x = state([-50e3, -60e3, 15e3], [900.0, 0.0, -700.0], beta=8_000.0)
    fine_x, fine_Phi = FilterModel(dataclasses.replace(FCFG, propagation_substep=0.001),
                                   CONST, ATM).propagate(x, 0.1, False)
    assert np.allclose(MODEL.propagate(x, 0.1, False)[0], fine_x, rtol=0, atol=1e-6)
    scale = np.max(np.abs(fine_Phi - np.eye(N_STATE)))
    diffs = [np.max(np.abs(FilterModel(dataclasses.replace(FCFG, propagation_substep=s), CONST,
                                       ATM).propagate(x, 0.1, False)[1] - fine_Phi)) / scale
             for s in (0.1, FCFG.propagation_substep)]
    assert diffs[1] < diffs[0] and diffs[1] < 2e-3


def test_filter_uses_the_isa_density():
    """THEORY 6.2: the filter's drag uses the same US76 density as the truth, and a enters dv/dt in
    BOOST only."""
    x = state([0.0, 0.0, 10e3], [600.0, 0.0, -800.0], [5.0, -2.0, 30.0], 8_000.0)
    g = CONST.g0 * (CONST.earth_radius / (CONST.earth_radius + 10e3)) ** 2
    a_drag = US76.density(10e3) * (600.0 ** 2 + 800.0 ** 2) / (2 * 8_000.0)  # rho |v|^2 / (2 beta)
    f_ball, f_boost = MODEL.f(x, False), MODEL.f(x, True)
    assert np.linalg.norm(f_ball[3:6] + np.array([0.0, 0.0, g])) == pytest.approx(a_drag, rel=1e-12)
    assert np.allclose(f_boost[3:6] - f_ball[3:6], x[ACC], rtol=0, atol=1e-12)
    assert np.all(f_boost[6:] == 0.0)


def test_drag_scheduled_spectral_density():
    """THEORY 6.3: sigma_a = sigma_floor + k_drag |a_drag_hat| from the filter's own state, q_a = sigma_a^2 dt.
    checked with non-zero values; the derived defaults are both 0 (config.py)."""
    dt = 0.1
    fc = dataclasses.replace(FCFG, sigma_a_floor=0.05, k_drag=0.25)
    model = FilterModel(fc, CONST, ATM)
    high = state([0.0, 0.0, 200e3], [1_000.0, 0.0, 0.0])  # almost no air: almost no drag
    assert model.drag_acceleration(high) < 1e-6
    assert model.spectral_density(high, dt) == pytest.approx(0.05 ** 2 * dt, rel=1e-3)
    low = state([0.0, 0.0, 10e3], [600.0, 0.0, -800.0], beta=8_000.0)
    a_drag = US76.density(10e3) * (600.0 ** 2 + 800.0 ** 2) / (2 * 8_000.0)
    assert model.drag_acceleration(low) == pytest.approx(a_drag, rel=1e-12)
    assert model.spectral_density(low, dt) == pytest.approx((0.05 + 0.25 * a_drag) ** 2 * dt, rel=1e-12)
    assert MODEL.spectral_density(low, dt) == (FCFG.sigma_a_floor + FCFG.k_drag * a_drag) ** 2 * dt


@pytest.fixture(scope="module")
def class_flights():
    """the truth flights of the vehicle config range with the scan index of their first detection
    (design analysis, evaluation - uses truth)."""
    from ekfsim.scenario import acquire, plan_scans
    flights = []
    for dm in (-1_000.0, 0.0, 1_000.0, 2_000.0, 3_000.0):
        truth = simulate_truth(VehicleConfig(dry_mass_delta=dm))
        plan = plan_scans(truth, RADAR, ScenarioConfig())
        acq = acquire(truth, RADAR, plan, RCFG, ScenarioConfig())
        flights.append((truth, plan, acq.first_detection.k))
    return flights


def test_q_beta_meets_the_margin_rule_over_the_config_range(class_flights):
    """design check behind k_drag = 0 and the q_beta derivation (config.py, evaluation - uses truth): over
    every ballistic scan with drag >= 0.05 g of the config range, the ln beta random walk's lag behind the
    Cd(Mach) ramp in the cross-range channel is at most half of its own steady-state spread (THEORY 6.3 margin
    rule), so no drag mismatch is left for k_drag. fails if q_beta or the Cd table change enough."""
    for truth, plan, _ in class_flights:
        idx, t = plan.truth_index, plan.t
        a_drag = np.linalg.norm(truth.drag[idx], axis=1) / truth.mass[idx]  # |a_D| in [m/s^2]
        sel = ~np.asarray(truth.boost[idx]) & (a_drag >= 0.05 * CONST.g0)
        ramp = np.abs(np.gradient(np.log(truth.beta[idx]), t))[sel]  # |d ln beta / dt| in [1/s]
        rng = np.linalg.norm(truth.position[idx] - RADAR.position, axis=1)[sel]
        r = (rng * RCFG.sigma_angle) ** 2 * RCFG.update_interval  # cross-range noise density
        w = (FCFG.q_beta * a_drag[sel] ** 2 / r) ** (1 / 6)  # steady-state bandwidth of the ln beta channel
        assert np.all(ramp / w <= 0.5 * np.sqrt(FCFG.q_beta / (2 * w)))  # lag <= spread / 2


def test_q_thrust_meets_the_margin_rule_over_the_config_range(class_flights):
    """design check behind the q_thrust derivation (config.py, evaluation - uses truth): over every tracked
    boost scan before burnout of the config range, the lag of the a random walk behind what it must follow
    (T/m plus the drag mismatch at the frozen prior beta) in the cross-range channel is at most half of
    its own steady-state spread (THEORY 6.3 margin rule)."""
    for truth, plan, k0 in class_flights:
        ks = np.array([k for k in range(k0, len(plan.t))
                       if truth.boost[plan.truth_index[k]] and plan.t[k] < truth.derived.burn_time - 0.05])
        t, ii = plan.t[ks], plan.truth_index[ks]
        need = np.zeros((len(ks), 3))
        for j, i in enumerate(ii):
            x = state(truth.position[i], truth.velocity[i], beta=FCFG.beta0)
            drag_filter = MODEL.f(x, False)[3:6] + np.array([0.0, 0.0, truth.gravity[i]])  # filter drag at beta0
            need[j] = (truth.thrust[i] + truth.drag[i]) / truth.mass[i] - drag_filter  # a the filter must follow
        jerk = np.linalg.norm(np.gradient(need, t, axis=0), axis=1)  # |J| in [m/s^3]
        rng = np.linalg.norm(truth.position[ii] - RADAR.position, axis=1)
        r = (rng * RCFG.sigma_angle) ** 2 * RCFG.update_interval  # cross-range noise density
        w = (FCFG.q_thrust / r) ** (1 / 6)  # steady-state bandwidth of the a channel
        assert np.all(jerk / w <= 0.5 * np.sqrt(FCFG.q_thrust / (2 * w)))  # lag <= spread / 2


@pytest.mark.parametrize("boost", [True, False])
def test_process_noise_structure(boost):
    dt, q_a, q_m = 0.1, 0.02, 0.5
    Q = MODEL.process_noise(dt, q_a, boost, q_m)
    q = q_a + q_m  # drag-scheduled density plus the safety net's manoeuvre density
    for i in range(3):
        assert Q[i, i] == pytest.approx(q * dt ** 3 / 3)
        assert Q[i, i + 3] == Q[i + 3, i] == pytest.approx(q * dt ** 2 / 2)
        assert Q[i + 3, i + 3] == pytest.approx(q * dt)
        assert Q[6 + i, 6 + i] == pytest.approx(FCFG.q_thrust * dt if boost else 0.0)  # thrust random walk
    assert Q[BETA_ROW, BETA_ROW] == pytest.approx(FCFG.q_beta * dt)  # the manoeuvre noise acts on v only
    assert np.allclose(Q, Q.T) and np.all(np.linalg.eigvalsh(Q) >= -1e-15)
    assert np.count_nonzero(Q) == 3 * 4 + 1 + (3 if boost else 0)  # no coupling between axes


# the update part:

def test_ekf_reduces_to_standard_kf_on_a_linear_system():
    """constant-velocity model with a linear position measurement: the EKF steps (RK4 propagation,
    second-order Phi, Joseph update with a solved gain) give exactly the textbook KF (closed-form F,
    gain by inverse, standard covariance update)."""
    A = np.zeros((4, 4))
    A[0:2, 2:4] = np.eye(2)
    rng = np.random.default_rng(2)
    dt, q, r = 0.1, 0.5, 4.0
    F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
    Q = q * np.kron(np.array([[dt ** 3 / 3, dt ** 2 / 2], [dt ** 2 / 2, dt]]), np.eye(2))
    H = np.hstack([np.eye(2), np.zeros((2, 2))])
    R = r * np.eye(2)
    x_e = x_k = np.array([0.0, 0.0, 10.0, -5.0])
    P_e = P_k = np.diag([100.0, 100.0, 25.0, 25.0])
    for _ in range(50):
        z = rng.normal(size=2) * 3.0
        x_pred = rk4_step(lambda t, s, seg: A @ s, 0.0, x_e, dt, 0)
        Phi = np.eye(4) + A * dt + 0.5 * (A * dt) @ (A * dt)
        u = kalman_update(x_pred, Phi @ P_e @ Phi.T + Q, z, H @ x_pred, H, R)
        x_e, P_e = u.x, u.P
        x_k, P_k = F @ x_k, F @ P_k @ F.T + Q  # textbook KF
        K = P_k @ H.T @ np.linalg.inv(H @ P_k @ H.T + R)
        x_k = x_k + K @ (z - H @ x_k)
        P_k = (np.eye(4) - K @ H) @ P_k
        assert np.allclose(Phi, F, rtol=0, atol=1e-15)
        assert np.allclose(x_e, x_k, rtol=1e-12, atol=1e-10)
        assert np.allclose(P_e, P_k, rtol=1e-10, atol=1e-10)


def h_jacobian(x):
    H = np.zeros((4, N_STATE))
    H[:, 0:6] = RADAR.jacobian(x[0:3], x[3:6])
    return H


@pytest.mark.parametrize("az_pred_deg, az_meas_deg, nu_deg", [(1.0, 359.0, -2.0), (359.0, 1.0, 2.0)])
def test_update_with_azimuth_across_north(az_pred_deg, az_meas_deg, nu_deg):
    """the predicted target is on one side of North and the measurement on the other: the azimuth
    innovation must be the short way (-2 deg, not 358 deg), and the update must move the estimate
    by about that angle, not by a full turn."""
    r, el = 100e3, math.radians(20.0)
    p = RADAR.position + r * pointing_vector(math.radians(az_pred_deg), el)
    x_pred = state(p, [0.0, 300.0, -100.0])
    P_pred = np.diag([300.0 ** 2] * 3 + [30.0 ** 2] * 3 + [1.0] * 3 + [1.0])
    z_pred = RADAR.h(p, x_pred[3:6])
    z = z_pred.copy()
    z[1] = math.radians(az_meas_deg)
    u = kalman_update(x_pred, P_pred, z, z_pred, h_jacobian(x_pred), RADAR.R, ANGLE_ROWS)
    assert math.degrees(u.nu[1]) == pytest.approx(nu_deg, abs=1e-9)
    az_new = RADAR.h(u.x[0:3], u.x[3:6])[1]
    moved = math.degrees(wrap_to_pi(az_new - z_pred[1]))
    assert np.sign(moved) == np.sign(nu_deg) and 0.5 * abs(nu_deg) < abs(moved) <= abs(nu_deg)
    assert np.linalg.norm(u.x[0:3] - p) < 2 * r * math.radians(abs(nu_deg))  # a few km, not a full turn
    assert u.nis < 100.0


def test_frozen_row_is_a_consider_update():
    """with the ln beta row of K at zero, ln beta and its variance are untouched by the update, the other
    states are still corrected, and the Joseph form gives the exact covariance of that suboptimal
    estimate (checked against a direct Monte Carlo of the update)."""
    rng = np.random.default_rng(8)
    x = state([-60e3, -60e3, 20e3], [900.0, 0.0, -500.0], [20.0, 0.0, 20.0])
    L = rng.normal(size=(N_STATE, N_STATE)) * 0.1
    P = np.diag([200.0 ** 2] * 3 + [20.0 ** 2] * 3 + [10.0 ** 2] * 3 + [1.5 ** 2]) + L @ L.T
    P[0:6, BETA_ROW] = P[BETA_ROW, 0:6] = np.sqrt(np.diag(P)[0:6]) * 1.5 * 0.3  # correlated with ln beta
    H = h_jacobian(x)
    z = RADAR.h(x[0:3], x[3:6]) + np.array([30.0, 1e-3, -1e-3, 2.0])
    u = kalman_update(x, P, z, RADAR.h(x[0:3], x[3:6]), H, RADAR.R, ANGLE_ROWS, frozen_rows=(BETA_ROW,))
    assert u.x[BETA_ROW] == x[BETA_ROW] and u.P[BETA_ROW, BETA_ROW] == pytest.approx(P[BETA_ROW, BETA_ROW], rel=1e-12)
    assert np.all(u.K[BETA_ROW] == 0.0) and np.linalg.norm(u.x[0:3] - x[0:3]) > 0.0
    assert is_positive_definite(u.P)
    n = 200_000  # Monte Carlo of e+ = (I - K H) e- - K w for e- ~ N(0, P), w ~ N(0, R)
    e = rng.multivariate_normal(np.zeros(N_STATE), P, n)
    w = rng.multivariate_normal(np.zeros(4), RADAR.R, n)
    e_post = e @ (np.eye(N_STATE) - u.K @ H).T - w @ u.K.T
    Lp = np.linalg.cholesky(u.P)
    whitened = np.linalg.solve(Lp, np.linalg.solve(Lp, np.cov(e_post.T)).T)
    assert np.allclose(whitened, np.eye(N_STATE), atol=0.02)


def test_joseph_update_keeps_P_symmetric_positive_definite():
    rng = np.random.default_rng(5)
    for _ in range(100):
        x = random_state(rng)
        L = rng.normal(size=(N_STATE, N_STATE))
        P = L @ L.T + 1e-3 * np.eye(N_STATE)
        z = RADAR.h(x[0:3], x[3:6]) + rng.normal(size=4) * np.sqrt(np.diag(RADAR.R))
        u = kalman_update(x, P, z, RADAR.h(x[0:3], x[3:6]), h_jacobian(x), RADAR.R, ANGLE_ROWS)
        assert np.array_equal(u.P, u.P.T) and is_positive_definite(u.P)
        assert u.nis == pytest.approx(float(u.nu @ np.linalg.inv(u.S) @ u.nu), rel=1e-8)


# the mode part:

def boost_track(**kw):
    """a tracker started in BOOST on a target climbing under thrust (no truth involved: z from h)."""
    tr = Tracker(dataclasses.replace(FCFG, **kw), RADAR, MODEL)
    p = RADAR.position + 120e3 * pointing_vector(math.radians(240.0), math.radians(10.0))
    v = np.array([300.0, 0.0, 500.0])
    tr.initialise(0.0, RADAR.h(p, v), engine_flag=True)
    return tr, p, v


def test_beta_frozen_in_boost_and_learned_after_burnout():
    """THEORY 6.2: in BOOST ln beta is not updated (zero K row) and its variance grows by q_beta dt; when the
    IR flag goes off the track switches to BALLISTIC, a is set to 0 with zero rows and columns in P, and
    ln beta is learned from that update on."""
    tr, p, v = boost_track()
    assert tr.mode is FilterMode.BOOST
    var0 = tr.P[BETA_ROW, BETA_ROW]
    for k in range(1, 6):
        tr.predict(0.1 * k)
        lnb_before = tr._pending[1][BETA_ROW]
        rec = tr.process(detection(RADAR.h(p + v * 0.1 * k, v), k, 0.1 * k), engine_flag=True)
        assert rec.beta_frozen and rec.x_post[BETA_ROW] == lnb_before and rec.mode is FilterMode.BOOST
    assert tr.P[BETA_ROW, BETA_ROW] == pytest.approx(var0 + 5 * FCFG.q_beta * 0.1, rel=1e-9)
    tr.predict(0.6)
    rec = tr.process(detection(RADAR.h(p + v * 0.6, v), 6, 0.6), engine_flag=False)
    assert rec.mode is FilterMode.BALLISTIC and not rec.beta_frozen
    assert np.all(rec.x_post[ACC] == 0.0)
    assert np.all(rec.P_post[ACC, :] == 0.0) and np.all(rec.P_post[:, ACC] == 0.0)
    assert rec.pd_ok and is_positive_definite(rec.P_post[np.ix_(ACTIVE_BALLISTIC, ACTIVE_BALLISTIC)])
    assert "BOOST -> BALLISTIC" in tr.events[-1].reason
    tr.predict(0.7)
    assert np.all(tr._pending[1][ACC] == 0.0) and np.all(tr._pending[2][ACC, :] == 0.0)  # stays fixed at 0


def test_reignition_flag_is_ignored_with_a_warning():
    tr, p, v = boost_track()
    tr.predict(0.1)
    tr.process(detection(RADAR.h(p + v * 0.1, v)), engine_flag=False)
    for k in (2, 3):
        tr.predict(0.1 * k)
        rec = tr.process(detection(RADAR.h(p + v * 0.1 * k, v), k, 0.1 * k), engine_flag=True)
        assert rec.mode is FilterMode.BALLISTIC
    warnings = [e for e in tr.events if "ignored" in e.reason]
    assert len(warnings) == 1  # logged once


def test_boost_mode_off_ignores_the_flag():
    """THEORY 6.6 toggle: with boost mode off the tracker stays in BALLISTIC (a fixed at 0, beta learned)."""
    tr, p, v = boost_track(boost_mode=False)
    assert tr.mode is FilterMode.BALLISTIC and np.all(tr.P[ACC, :] == 0.0)
    tr.predict(0.1)
    rec = tr.process(detection(RADAR.h(p + v * 0.1, v)), engine_flag=True)
    assert rec.mode is FilterMode.BALLISTIC and not rec.beta_frozen
    assert not any("ignored" in e.reason for e in tr.events)


# the initialisation part:

@pytest.mark.parametrize("mode", [FilterMode.BOOST, FilterMode.BALLISTIC])
def test_initialisation_from_one_detection(mode):
    """position from the inverse of h, its covariance against a Monte Carlo of the conversion; velocity
    with only the radial part measured; a = 0 with sigma_thrust0 in BOOST, fixed at 0 in BALLISTIC."""
    p = RADAR.position + 150e3 * pointing_vector(math.radians(250.0), math.radians(5.0))
    v = np.array([300.0, 50.0, 400.0])
    z = RADAR.h(p, v)
    x0, P0 = initial_estimate(RADAR, z, FCFG, mode)
    assert np.allclose(x0[0:3], p, rtol=0, atol=1e-6)
    u_r = (p - RADAR.position) / 150e3
    assert x0[3:6] @ u_r == pytest.approx(z[3]) and np.allclose(np.cross(x0[3:6], u_r), 0.0, atol=1e-9)
    assert u_r @ P0[3:6, 3:6] @ u_r == pytest.approx(RADAR.sigma_rdot ** 2, rel=1e-9)
    t1 = np.cross(u_r, [0.0, 0.0, 1.0])
    t1 /= np.linalg.norm(t1)
    assert t1 @ P0[3:6, 3:6] @ t1 == pytest.approx(FCFG.sigma_v_tangential ** 2, rel=1e-9)
    assert np.all(x0[ACC] == 0.0)
    expected_aa = FCFG.sigma_thrust0 ** 2 * np.eye(3) if mode is FilterMode.BOOST else np.zeros((3, 3))
    assert np.array_equal(P0[ACC, ACC], expected_aa)
    assert x0[BETA_ROW] == pytest.approx(math.log(FCFG.beta0))
    assert P0[BETA_ROW, BETA_ROW] == pytest.approx(FCFG.sigma_ln_beta0 ** 2)
    rng = np.random.default_rng(6)
    samples = RADAR.position_from_measurement(z[0:3] + RADAR.noise(50_000, rng)[:, 0:3])
    L = np.linalg.cholesky(P0[0:3, 0:3])
    whitened = np.linalg.solve(L, np.linalg.solve(L, np.cov(samples.T)).T)  # L^-1 Sigma L^-T, should be I
    assert np.allclose(whitened, np.eye(3), rtol=0, atol=0.03)
    idx = np.arange(N_STATE) if mode is FilterMode.BOOST else ACTIVE_BALLISTIC
    assert is_positive_definite(P0[np.ix_(idx, idx)])


# the adaptive Q and status part:

def make_tracker(**kw):
    return Tracker(dataclasses.replace(FCFG, **kw), RADAR, MODEL)


def fake_step(tr, nis_value, detected=True, sigma_pos=50.0):
    """drive the status and safety-net logic with a chosen NIS, without a real measurement."""
    tr.P = np.diag([sigma_pos ** 2] * 3 + [10.0] * 3 + [0.0] * 3 + [0.1])
    tr._nis.append(nis_value) if detected else None
    window = float(np.mean(tr._nis)) if tr._nis else math.nan
    upd = type("U", (), {"nis": nis_value})() if detected else None
    if detected:
        tr._safety_net(window)
    tr._update_status(upd, False, detection(np.zeros(4)) if detected else MISS, window)


def test_safety_net_switches_on_and_off():
    """THEORY 6.6: two levels. the manoeuvre noise switches on while the window NIS is above its
    false-alarm-budget threshold and off once it is back inside; with the net off it never switches on."""
    tr = make_tracker()
    tr.status = TrackStatus.ACQUIRING
    for _ in range(10):
        fake_step(tr, 200.0)
    assert tr.manoeuvre_on and "manoeuvre noise ON" in tr.events[-2].reason + tr.events[-1].reason
    tr._nis.clear()
    for _ in range(10):
        fake_step(tr, 4.0)
    assert not tr.manoeuvre_on
    off = make_tracker(adaptive_q=False)
    off.status = TrackStatus.ACQUIRING
    for _ in range(10):
        fake_step(off, 200.0)
    assert not off.manoeuvre_on


def test_manoeuvre_noise_is_independent_of_q_a():
    """the net adds sigma_manoeuvre^2 dt to the white-acceleration density even where q_a is 0 (the
    derived sigma_floor = k_drag = 0, THEORY 6.3)."""
    tr, p, v = boost_track(boost_mode=False)
    assert FCFG.sigma_a_floor == 0.0 and FCFG.k_drag == 0.0
    tr.predict(0.1)
    Q_off = tr._pending[4]
    tr.process(detection(RADAR.h(p + v * 0.1, v)), engine_flag=False)
    tr.manoeuvre_on = True
    tr.predict(0.2)
    Q_on = tr._pending[4]
    q_m = FCFG.sigma_manoeuvre ** 2 * 0.1
    assert Q_off[3, 3] == pytest.approx(0.0, abs=1e-12)
    assert Q_on[3, 3] == pytest.approx(q_m * 0.1, rel=1e-12) and Q_on[0, 0] == pytest.approx(q_m * 0.1 ** 3 / 3, rel=1e-12)


def test_sigma_manoeuvre_is_the_largest_thrust_acceleration_at_burnout():
    """THEORY 6.6: sigma_manoeuvre = vacuum thrust over the smallest burnout mass of the vehicle config range."""
    from ekfsim.vehicle import derived_values, thrust_magnitude
    vc = VehicleConfig()
    lightest = derived_values(dataclasses.replace(vc, dry_mass_delta=vc.dry_mass_delta_min), CONST, US76)
    a_max = thrust_magnitude(vc, ATM.p0, 0.0) / lightest.burnout_mass  # T_vac / m_bo,min in [m/s^2]
    assert FCFG.sigma_manoeuvre == pytest.approx(a_max, abs=0.05)


def test_net_threshold_from_the_false_alarm_budget():
    """THEORY 6.6 (Bonferroni): every window test of a nominal flight counts, so the per-test false-alarm
    probability is alpha = p_fa_flight / N_updates and the sum over the flight is p_fa_flight; for a full
    window the threshold is 9.32."""
    tr = make_tracker()
    alpha = FCFG.p_fa_flight / FCFG.nominal_updates
    assert FCFG.nominal_updates * alpha == pytest.approx(FCFG.p_fa_flight, rel=1e-12)
    for n in (1, 5, FCFG.nis_window):
        assert tr._net_hi[n - 1] == pytest.approx(chi2.ppf(1.0 - alpha, n * RADAR.dim) / n, rel=1e-12)
    assert tr._net_hi[-1] == pytest.approx(9.32, abs=0.005)


def test_net_false_alarm_rate_per_flight_meets_the_budget():
    """THEORY 6.6: iid chi2(m) NIS values (a consistent filter) fed through the implemented window test fire
    in fewer than p_fa_flight of nominal flights, with a margin of 3 standard errors of the measured rate.
    a vectorised copy of the test runs the many flights; it is first checked against the tracker's own
    switching on one sequence with spikes, so the copy is the implemented test.
    eps_bar_k = (1/n_k) sum of the last n_k = min(k, N_w) NIS values - window NIS, growing at the start
    fired = any_k eps_bar_k > threshold(n_k) - at least one false alarm in the flight
    rate + 3 sqrt(rate (1 - rate) / flights) < p_fa_flight - budget met, not by chance"""
    m, n_w = RADAR.dim, FCFG.nis_window
    tr = make_tracker()
    hi = np.array(tr._net_hi)

    def window_nis(nis):
        """(flights, updates) NIS values -> window NIS and the window length n_k after every update."""
        c = np.cumsum(nis, axis=1)
        c[:, n_w:] -= c[:, :-n_w].copy()
        n = np.minimum(np.arange(1, nis.shape[1] + 1), n_w)
        return c / n, n

    rng = np.random.default_rng(0)
    seq = rng.chisquare(m, size=(1, 300))
    seq[0, [50, 51, 52, 200]] = [60.0, 50.0, 40.0, 120.0]  # spikes, so the net switches on and off
    eps, n = window_nis(seq)
    tr.status = TrackStatus.ACQUIRING
    switching = []
    for v in seq[0]:
        fake_step(tr, float(v))
        switching.append(tr.manoeuvre_on)
    assert np.any(switching) and np.array_equal(switching, eps[0] > hi[n - 1])

    flights, chunk, fired = 20_000, 1_000, 0
    for _ in range(flights // chunk):
        eps, n = window_nis(rng.chisquare(m, size=(chunk, FCFG.nominal_updates)))
        fired += int(np.sum(np.any(eps > hi[n - 1], axis=1)))
    rate = fired / flights  # measured false-alarm probability per nominal flight
    assert rate + 3.0 * math.sqrt(rate * (1.0 - rate) / flights) < FCFG.p_fa_flight


def test_burnout_reprocessing_restores_and_replays():
    """THEORY 6.2: when the flag goes off at t_flag, the filter restores the state of the last scan at or
    before t_flag - L - T (never after the burnout, which the flag places in (t_flag - L - T, t_flag - L]),
    switches to BALLISTIC and re-processes the buffered scans; the result equals a filter that had switched
    to BALLISTIC at that scan."""
    latency, dt = 0.3, RCFG.update_interval  # L in [s]; T = 0.1 s
    p0 = RADAR.position + 120e3 * pointing_vector(math.radians(240.0), math.radians(10.0))
    v = np.array([300.0, 0.0, 500.0])
    zs = [RADAR.h(p0 + v * dt * k, v) + np.array([5.0, 1e-4, -1e-4, 0.5]) * (-1) ** k for k in range(10)]
    flags = [True] * 7 + [False] * 3  # flag off at scan 7 (t = 0.7 s), L = 0.3 s, T = 0.1 s -> restore at 0.3 s
    tr = Tracker(FCFG, RADAR, MODEL, ir_latency=latency)
    tr.initialise(0.0, zs[0], engine_flag=True)
    recs = []
    for k in range(1, 10):
        tr.predict(dt * k)
        recs.append(tr.process(detection(zs[k], k, dt * k), engine_flag=flags[k]))
    assert recs[6].reprocessed == 4 and recs[6].mode is FilterMode.BALLISTIC  # buffered 0.4-0.6 s, then 0.7 s
    assert any("restored at 0.30 s" in e.reason for e in tr.events)
    ref = Tracker(FCFG, RADAR, MODEL, ir_latency=latency)  # reference: switch to BALLISTIC right after 0.3 s
    ref.initialise(0.0, zs[0], engine_flag=True)
    for k in range(1, 4):
        ref.predict(dt * k)
        ref.process(detection(zs[k], k, dt * k), engine_flag=True)
    ref.x, ref.P = Tracker._to_ballistic(ref.x, ref.P)
    ref.mode = FilterMode.BALLISTIC
    for k in range(4, 10):
        ref.predict(dt * k)
        rec_ref = ref.process(detection(zs[k], k, dt * k), engine_flag=False)
        if k >= 7:
            assert np.allclose(recs[k - 1].x_post, rec_ref.x_post, rtol=0, atol=1e-9)
            assert np.allclose(recs[k - 1].P_post, rec_ref.P_post, rtol=0, atol=1e-9)
    assert recs[4].mode is FilterMode.BOOST  # the record of a scan inside the latency stays causal


def test_track_status_rules():
    tr = make_tracker(lock_updates=5, lost_updates=3)
    tr.t = 0.0
    tr._set_status(TrackStatus.ACQUIRING, "init")
    for _ in range(4):
        fake_step(tr, 4.0)
    assert tr.status is TrackStatus.ACQUIRING
    fake_step(tr, 4.0)
    assert tr.status is TrackStatus.LOCKED  # M = 5 good updates
    fake_step(tr, 4.0, detected=False)
    assert tr.status is TrackStatus.COASTING
    fake_step(tr, 4.0)
    assert tr.status is TrackStatus.LOCKED  # back after coasting
    fake_step(tr, 4.0, sigma_pos=FCFG.lost_sigma_pos * 1.1)
    assert tr.status is TrackStatus.LOST
    tr2 = make_tracker(lock_updates=5, lost_updates=3)
    tr2.t = 0.0
    tr2._set_status(TrackStatus.ACQUIRING, "init")
    for _ in range(3):
        fake_step(tr2, 1_000.0)
    assert tr2.status is TrackStatus.LOST and "NIS above" in tr2.events[-1].reason
    reasons = [e.reason for e in tr.events]
    assert all(reasons) and len(tr.events) == 5  # every transition logged with a reason


def test_singularity_guard_skips_the_update():
    tr = make_tracker()
    p_overhead = RADAR.position + np.array([0.2, 0.3, 50e3])
    z = RADAR.h(p_overhead + 5.0, np.zeros(3))
    tr.initialise(0.0, RADAR.h(p_overhead + np.array([0.0, 50.0, 0.0]), np.zeros(3)), engine_flag=False)
    tr.x[0:3] = p_overhead
    tr.x[3:6] = 0.0
    tr.predict(0.1)
    rec = tr.process(detection(z), engine_flag=False)
    assert rec.skipped_singular and rec.update is None
    assert tr.status is TrackStatus.COASTING and "vertical" in tr.events[-1].reason


def test_filter_never_imports_truth():
    """no inverse crime and no truth access: the filter module imports neither the truth model nor
    the scenario (which holds the truth). the shared ISA atmosphere module is allowed (THEORY 6.2)."""
    tree = ast.parse((SRC / "ekf.py").read_text(encoding="utf-8"))
    imported = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    imported |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert not imported & {"ekfsim.vehicle", "ekfsim.scenario", "ekfsim.aero", "ekfsim.montecarlo"}


# the consistency part (evaluation - uses truth):

def test_consistent_on_matched_ballistic_model_with_weak_nonlinearity():
    """implementation check: the truth follows the BALLISTIC filter model exactly (no thrust, the same
    ISA density, constant Cd so constant beta), starts high (weak drag), and the initial estimate is
    drawn from P0: the ANEES of 25 runs stays inside its 95 % bounds most of the time."""
    vc, sc = VehicleConfig(), ScenarioConfig()
    fc = dataclasses.replace(FCFG, sigma_a_floor=math.sqrt(1e-6 / 0.1), k_drag=0.0, q_beta=1e-8,
                             adaptive_q=False, boost_mode=False)  # the plain EKF
    truth_model = TruthModel(vc, sc, US76, lambda mach: 0.3, CONST, thrust_on=False)
    truth = simulate_truth(model=truth_model, initial_state=np.array(
        [-100e3, -60e3, 80e3, 1_800.0, 0.0, -600.0, 2_500.0]))
    n_runs, dt, n_steps = 25, 0.1, 400  # 40 s, from 80 km down to about 50 km
    P0 = np.diag([100.0 ** 2] * 3 + [20.0 ** 2] * 3 + [0.0] * 3 + [0.3 ** 2])
    sel = ACTIVE_BALLISTIC
    rng = np.random.default_rng(7)
    all_nees = np.zeros((n_runs, n_steps - 1))
    for run in range(n_runs):
        tr = Tracker(fc, RADAR, FilterModel(fc, CONST, ATM))
        x_true0 = state(truth.position[0], truth.velocity[0], beta=truth.beta[0])
        tr.t, tr.P, tr.status, tr.mode = 0.0, P0.copy(), TrackStatus.ACQUIRING, FilterMode.BALLISTIC
        tr.x = x_true0.copy()
        tr.x[sel] += np.linalg.cholesky(P0[np.ix_(sel, sel)]) @ rng.standard_normal(len(sel))
        w = RADAR.noise(n_steps, rng)
        for k in range(1, n_steps):
            tr.predict(k * dt)
            i = 10 * k  # truth grid 0.01 s
            rec = tr.process(RADAR.scan(k, k * dt, truth.position[i], truth.velocity[i], tr.pointing(), w[k]), False)
            all_nees[run, k - 1] = nees(rec.x_post, rec.P_post, np.r_[truth.position[i], truth.velocity[i]])
    anees = all_nees.mean(axis=0)
    lo, hi = average_interval(n_runs, 6)
    assert np.mean((anees >= lo) & (anees <= hi)) > 0.85
    assert 5.0 < anees.mean() < 7.0


def test_boost_mode_follows_a_thrust_ramp_consistently():
    """the q_thrust derivation (config.py): the truth is the filter's own BOOST model with a thrust
    acceleration that ramps at 1 m/s^3, the realistic one-detection initialisation is used, and after the
    first 10 s the thrust-acceleration error stays consistent with its covariance (mean eps_a well below
    the 99 % bound of its 3 degrees of freedom) and ANEES stays near 6."""
    fc = dataclasses.replace(FCFG, adaptive_q=False)
    model = FilterModel(fc, CONST, ATM)
    u = np.array([0.6, 0.0, 0.8])  # thrust direction
    x = state([-140e3, -60e3, 3e3], [150.0, 0.0, 300.0], 25.0 * u, 17_000.0)
    X = [x.copy()]
    for _ in range(300):  # 30 s at 0.1 s, truth integrated at 0.01 s
        for _ in range(10):
            x = rk4_step(lambda t, s, seg: model.f(s, True), 0.0, x, 0.01, 0)
            x[ACC] += 1.0 * 0.01 * u  # jerk 1 m/s^3 along u
        X.append(x.copy())
    X = np.array(X)
    rng = np.random.default_rng(1)
    n_runs, eps_a, ne = 15, [], []
    for _ in range(n_runs):
        tr = Tracker(fc, RADAR, model)
        tr.initialise(0.0, RADAR.h(X[0, :3], X[0, 3:6]) + RADAR.noise(1, rng)[0], engine_flag=True)
        tr.x[BETA_ROW], tr.P[BETA_ROW, BETA_ROW] = X[0, BETA_ROW], 1e-6
        w = RADAR.noise(len(X), rng)
        ea, nn = [], []
        for k in range(1, len(X)):
            tr.predict(0.1 * k)
            rec = tr.process(RADAR.scan(k, 0.1 * k, X[k, :3], X[k, 3:6], tr.pointing(), w[k]), True)
            e = rec.x_post[ACC] - X[k, ACC]
            ea.append(float(e @ np.linalg.solve(rec.P_post[ACC, ACC], e)))
            nn.append(float(nees(rec.x_post, rec.P_post, X[k])))
        eps_a.append(ea)
        ne.append(nn)
    eps_a, ne = np.mean(eps_a, axis=0)[100:], np.mean(ne, axis=0)[100:]
    assert eps_a.mean() < chi2.ppf(0.99, 3)
    assert 4.0 < ne.mean() < 8.0
