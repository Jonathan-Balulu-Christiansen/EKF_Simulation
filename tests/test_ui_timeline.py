"""the display arrays of the UI (no Qt). they must show exactly what the run computed: the filter
quantities of the last scan at or before the cursor, the truth sample at or before it, and the derived
geometry (beam, ellipsoid) from the filter's own state."""

import dataclasses
import math

import numpy as np
import pytest

from ekfsim.config import AtmosphereConfig, FilterConfig, PhysicsConstants, RadarConfig, UIConfig
from ekfsim.ekf import FilterModel, Tracker
from ekfsim.radar import Radar
from ekfsim_ui import fmt
from ekfsim_ui.timeline import KM, RunRequest, compute, ellipsoid_axes, rotation_from_z


def test_cursor_indices_hold_the_last_scan_and_truth_sample(default_timeline):
    tl = default_timeline
    T = tl.T
    assert tl.scan_index(0.0) == 0
    assert tl.scan_index(150 * T) == 150  # exactly on a scan
    assert tl.scan_index(150 * T + 0.999 * T) == 150  # held until the next scan
    assert tl.scan_index(-5.0) == 0 and tl.scan_index(1e6) == tl.n_scans - 1
    t = 123.456
    i = tl.truth_index(t)
    assert tl.truth_t[i] <= t + 1e-9 < tl.truth_t[i + 1]
    assert tl.truth_index(tl.t_end + 10.0) == len(tl.truth_t) - 1


def test_trails_end_at_the_cursor_and_are_capped(default_timeline):
    tl = default_timeline
    cap = UIConfig().trail_max_points
    for t in (30.0, 150.0, tl.t_end):
        i, k = tl.truth_index(t), tl.scan_index(t)
        tr = tl.truth_trail(i)
        assert np.allclose(tr[-1], tl.truth_pos_km[i]) and len(tr) <= cap + 1
        assert np.allclose(tr[0], tl.launch_km)
        est = tl.estimate_trail(k)
        assert np.allclose(est[-1], tl.result.filter.x_post[k, 0:3] / KM) and len(est) <= cap + 1
    assert tl.estimate_trail(tl.k0 - 1).shape == (0, 3)  # no estimate before the track


def test_measurements_are_the_detections_converted_for_display(default_timeline):
    tl = default_timeline
    k = tl.scan_index(150.0)
    m = tl.measurements_upto(k)
    det = [s for s in tl.result.scans[:k + 1] if s.detected]
    assert len(m) == len(det)
    p = tl.result.radar.position_from_measurement(det[-1].z) / KM
    assert np.allclose(m[-1], p)


def test_beam_follows_the_cue_then_the_prediction(default_timeline):
    tl = default_timeline
    F, radar = tl.result.filter, tl.result.radar
    k0 = tl.k0
    cue_len = math.hypot(*(tl.result.truth.position[0, 0:2] - radar.position[0:2])) / math.cos(tl.cue_el) / KM
    assert tl.beam_len_km[0] == pytest.approx(cue_len) and tl.beam_len_km[k0] == pytest.approx(cue_len)
    for k in (k0 + 1, k0 + 500, tl.n_scans - 1):
        assert tl.beam_len_km[k] == pytest.approx(np.linalg.norm(F.x_prior[k, 0:3] - radar.position) / KM)
        assert np.allclose(tl.beam_u[k], tl.result.scans[k].pointing)


def test_window_thresholds_match_the_tracker():
    """the UI recomputes the window bounds and the safety-net threshold; they must be the tracker's own."""
    cfg = FilterConfig()
    radar = Radar(RadarConfig(), PhysicsConstants())
    tr = Tracker(cfg, radar, FilterModel(cfg, PhysicsConstants(), AtmosphereConfig()))
    tl = compute(RunRequest())
    n_win = np.minimum(np.cumsum(tl.updated), cfg.nis_window)
    for w in range(1, cfg.nis_window + 1):
        sel = np.flatnonzero(n_win == w)
        if len(sel) == 0:
            continue
        k = sel[0]
        assert tl.window_lo[k] == pytest.approx(tr._window_lo[w - 1])
        assert tl.window_hi[k] == pytest.approx(tr._window_hi[w - 1])
        assert tl.net_threshold[k] == pytest.approx(tr._net_hi[w - 1])


def test_errors_and_summary_are_evaluation_values(default_timeline):
    tl = default_timeline
    F, truth, idx = tl.result.filter, tl.result.truth, tl.result.plan.truth_index
    k = tl.scan_index(200.0)
    assert np.allclose(tl.err_pos[k], F.x_post[k, 0:3] - truth.position[idx[k]])
    assert np.allclose(tl.err_vel[k], F.x_post[k, 3:6] - truth.velocity[idx[k]])
    assert np.all(np.isnan(tl.err_pos[:tl.k0]))
    # the Doppler numbers of the default run: PRF > 85.0 kHz, R_ua 1.76 km
    assert tl.prf_required / 1e3 == pytest.approx(85.0, abs=0.1)
    assert tl.unambiguous_range / 1e3 == pytest.approx(1.76, abs=0.01)
    assert tl.min_horizontal_km == pytest.approx(60.0, abs=0.01)
    assert not tl.elevation_warning


def test_no_acquisition_gives_an_empty_filter_and_the_reason():
    req = RunRequest(radar=dataclasses.replace(RadarConfig(), elevation_mask=math.radians(10.0)))
    tl = compute(req)
    assert not tl.acquired and not tl.has_track and tl.k0 is None
    assert "elevation mask" in tl.acquisition_reason
    k = tl.scan_index(100.0)
    assert not tl.tracking(k) and np.isnan(tl.nis[k]) and tl.estimate_trail(k).shape == (0, 3)
    assert tl.track_miss_cum[-1] == 0 and tl.search_miss_cum[-1] == tl.n_scans


def test_rotation_from_z_points_the_cone():
    rng = np.random.default_rng(1)
    for u in list(rng.standard_normal((50, 3))) + [np.array([0, 0, 1.0]), np.array([0, 0, -1.0])]:
        u = u / np.linalg.norm(u)
        R = rotation_from_z(u)
        assert np.allclose(R @ [0, 0, 1.0], u, atol=1e-12)
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-12) and np.linalg.det(R) == pytest.approx(1.0)


def test_ellipsoid_axes_rebuild_the_covariance():
    rng = np.random.default_rng(2)
    for _ in range(20):
        A = rng.standard_normal((3, 3)) * 30.0
        P = A @ A.T + np.eye(3)
        V, radii = ellipsoid_axes(P, 3.0, 10.0)
        assert np.linalg.det(V) == pytest.approx(1.0)
        lam = (radii * KM / 30.0) ** 2  # a_i = 3 x 10 x sqrt(lambda_i) / 1000
        assert np.allclose(V @ np.diag(lam) @ V.T, P)


def test_fixed_display_units_of_the_equations_panel():
    """one fixed display unit per quantity type, never a prefix chosen per value; only a position switches
    between km and m."""
    assert fmt.disp(149_739.0, "pos") == "149.7 km" and fmt.disp(-7.57, "pos") == "−7.57 m"
    assert fmt.disp(0.07897, "m/s") == "0.0790 m/s" and fmt.disp(877.1, "m/s") == "877.1 m/s"  # never mm/s
    assert fmt.disp(0.3647, "m/s", sign=True) == "+0.365 m/s"
    assert fmt.disp(0.00156, "m/s²") == "0.00156 m/s²"
    assert fmt.disp(math.radians(245.8), "°") == "245.8°"  # absolute angles in degrees
    assert fmt.disp(-3.434e-3, "mrad") == "−3.43 mrad"  # angle innovations in mrad
    assert fmt.disp(0.7239e-3, "ms") == "0.724 ms" and fmt.disp(10e9, "GHz") == "10.00 GHz"
    assert fmt.disp(9.05e-5, "kg/m³") == "0.0000905 kg/m³"
    assert fmt.sci_cell(-1.32e-5) == "−1.32e−5" and fmt.sci_cell(0.0) == "·"
    assert fmt.sci_cell(102.0) == "102" and fmt.sci_cell(0.342) == "0.342"


def test_pulse_segments_fold_at_the_target():
    """the pulse goes out, folds at the reflection and comes back; its total length is c tau until it is home."""
    from ekfsim_ui.timeline import pulse_segments
    c, tau, r = 3e8, 1e-6, 90_000.0
    assert pulse_segments(-1e-6, tau, c, r, 2 * r) == ()
    (a, b), = pulse_segments(0.5 * r / c, tau, c, r, 2 * r)
    assert b == pytest.approx(0.5 * r) and b - a == pytest.approx(c * tau)
    parts = pulse_segments(r / c + 0.5 * tau, tau, c, r, 2 * r)  # half of it already reflected
    assert len(parts) == 2 and all(p[1] == pytest.approx(r) for p in parts)
    assert sum(b - a for a, b in parts) == pytest.approx(c * tau)
    (a, b), = pulse_segments(1.5 * r / c, tau, c, r, 2 * r)  # coming back: d = 2r - path
    assert a == pytest.approx(0.5 * r) and b == pytest.approx(0.5 * r + c * tau)
    assert pulse_segments(2 * r / c + 2 * tau, tau, c, r, 2 * r) == ()  # the echo is back at the antenna
    (a, b), = pulse_segments(1e-4, tau, c, None, 50_000.0)  # a miss: it only goes out
    assert (a, b) == (pytest.approx(30_000.0 - c * tau), pytest.approx(30_000.0))
    assert pulse_segments(1e-3, tau, c, None, 50_000.0) == ()  # gone past the end of the beam


def test_pulse_state_uses_the_stored_timing(default_timeline):
    tl = default_timeline
    k = 1500
    t_tx, delay = float(tl.t_scan[k]), float(tl.echo_delay[k])
    st = tl.pulse_state(t_tx + 0.25 * delay)
    assert st.k == k and st.detected and st.phase == "outgoing"
    assert st.r == pytest.approx(0.5 * tl.c * delay)
    assert st.segments[0][1] == pytest.approx(0.5 * st.r, rel=1e-6)
    assert st.done == pytest.approx(t_tx + delay + tl.pulse_width)
    assert tl.pulse_state(t_tx + 0.75 * delay).phase == "echo returning"
    assert tl.pulse_state(st.done + 1e-7).phase == "echo received"
    miss = int(np.flatnonzero(~np.asarray(tl.detected))[0])
    sm = tl.pulse_state(float(tl.t_scan[miss]) + 1e-4)
    assert not sm.detected and math.isnan(sm.r) and sm.phase.startswith("no echo")


def test_number_policy():
    assert fmt.num(-1234.5, 1) == "−1,234.5"
    assert fmt.num(float("nan"), 2) == fmt.DASH and fmt.num(None, 2) == fmt.DASH
    assert fmt.qu(161_555.0, "km") == "161.555 km"
    assert fmt.q(math.radians(45.302), "deg") == "45.302"
    assert fmt.sci(0.0).strip() == "·" and len(fmt.sci(-1.234e-5)) == 9
    assert fmt.sci(-1.234e-5) == "−1.23e−05"
    assert fmt.sig(9.05e-5) == "9.05e−05" and fmt.sig(0.0377) == "0.0377"
