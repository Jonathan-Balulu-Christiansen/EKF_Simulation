"""checks of the full scenario (THEORY 5.4, 5.7, 6, 8.1): radar and EKF run interleaved, the beam
follows the tracker's prediction (never truth), the IR flag switches the filter from BOOST to BALLISTIC,
the track locks and holds, the earlier design without boost mode fails in boost, and a flight whose
azimuth crosses North is tracked."""

import math

import numpy as np
import pytest

from ekfsim.config import FilterConfig, IRSensorConfig, RadarConfig, ScenarioConfig, VehicleConfig
from ekfsim.scenario import run_scenario, simulate_truth


@pytest.fixture(scope="module")
def default_run():
    return run_scenario()


def tracked_position_error(res):
    """evaluation - uses truth: |p_{k|k} - p_true(t_k)| over the tracked scans."""
    F, idx = res.filter, res.plan.truth_index
    k0 = F.track_start
    return np.linalg.norm(F.x_post[k0:, 0:3] - res.truth.position[idx[k0:]], axis=1)


def test_beam_follows_the_tracker_prediction(default_run):
    """before the track: the cue; after it: exactly the direction of x_{k|k-1}. no other source. the one
    scan where the IR flag goes off is skipped: its beam used the prediction made before the flag arrived,
    while its record holds the re-processed BALLISTIC prior (THEORY 6.2)."""
    res = default_run
    F, k0 = res.filter, res.filter.track_start
    cue = res.acquisition.scans[0].pointing
    assert all(s.pointing == cue for s in res.scans[:k0 + 1])
    assert np.count_nonzero(F.reprocessed) == 1
    for k in range(k0 + 1, len(res.scans)):
        if F.reprocessed[k]:
            continue
        d = F.x_prior[k, 0:3] - res.radar.position
        assert np.allclose(res.scans[k].pointing, d / np.linalg.norm(d), rtol=0, atol=1e-15)


def test_default_run_locks_and_holds(default_run):
    res = default_run
    F, k0 = res.filter, res.filter.track_start
    statuses = [e.new.value for e in F.events]
    assert statuses[0] == "ACQUIRING" and "LOCKED" in statuses and "LOST" not in statuses
    tracked = res.scans[k0 + 1:]
    assert sum(s.detected for s in tracked) / len(tracked) > 0.99  # misses only below the mask at the end
    assert np.all(F.pd_ok[k0:])
    err = tracked_position_error(res)
    burnout = res.truth.derived.burn_time
    assert np.max(err[res.plan.t[k0:] > burnout + 10.0]) < 300.0  # evaluation - uses truth
    boost = res.plan.t < res.truth.derived.burn_time
    assert np.mean(F.manoeuvre_on[k0:][boost[k0:]]) < 0.05  # the safety net rarely acts in boost (THEORY 6.6)
    assert len(res.scans) == len(res.plan.t)


def test_ir_flag_switches_boost_to_ballistic(default_run):
    """THEORY 5.7 and 6.2: the track starts in BOOST (engine on), and switches to BALLISTIC at the first scan
    whose IR flag is off, latency after the true burnout; a is 0 from there on, and beta is frozen exactly
    while in BOOST."""
    res = default_run
    F, k0, t = res.filter, res.filter.track_start, res.plan.t
    latency = IRSensorConfig().latency
    t_off = res.truth.t[res.truth.burnout_index] + latency
    assert F.mode[k0] == "BOOST" and F.engine_flag[k0]
    k_switch = int(np.argmax(F.mode == "BALLISTIC"))
    assert t[k_switch - 1] < t_off <= t[k_switch]
    assert not F.engine_flag[k_switch] and F.engine_flag[k_switch - 1]
    assert np.all(F.mode[k_switch:] == "BALLISTIC") and np.all(F.x_post[k_switch:, 6:9] == 0.0)
    assert np.all(F.beta_frozen[k0 + 1:k_switch][F.updated[k0 + 1:k_switch]])
    assert not np.any(F.beta_frozen[k_switch:])
    assert any("BOOST -> BALLISTIC" in e.reason for e in F.events)


def test_without_boost_mode_and_adaptive_q_the_boost_breaks_the_track():
    """THEORY 6.6: the earlier design (no boost mode) without adaptation loses the track in boost, while the
    boost mode holds it even without the adaptive Q."""
    res = run_scenario(filter_cfg=FilterConfig(boost_mode=False, adaptive_q=False))
    lost = [e for e in res.filter.events if e.new.value == "LOST"]
    assert lost and lost[0].t < res.truth.derived.burn_time
    assert not np.any(res.filter.manoeuvre_on)
    held = run_scenario(filter_cfg=FilterConfig(boost_mode=True, adaptive_q=False), truth=res.truth)
    assert not any(e.new.value == "LOST" for e in held.filter.events)
    netted = run_scenario(filter_cfg=FilterConfig(boost_mode=False, adaptive_q=True), truth=res.truth)
    assert not any(e.new.value == "LOST" for e in netted.filter.events)  # the net alone keeps the track
    assert np.any(netted.filter.manoeuvre_on)


def test_burnout_reprocessing_removes_the_latency_error():
    """THEORY 6.2 (evaluation - uses truth): during the IR latency the BOOST model keeps integrating the
    thrust estimate after the true burnout; re-processing from t_flag - L - T removes that error. over 20 runs
    with the same noise seeds, in the first second after the flag change (10 scans):
    (a) with re-processing the filter is not over-confident: the mean NEES (200 samples) is below the 95 %
        upper bound of such a mean (like the acceptance rule of THEORY 7, only over-confidence counts);
    (b) re-processing reduces the true position error and the true velocity error: the paired mean
        reduction of the mean-square error is more than two standard errors above zero (about 97 %
        one-sided), so it is measurable and not a noise effect.
    whether the filter without re-processing is over-confident depends on q_thrust (a wide thrust random
    walk makes P wide enough to cover the latency error), so that is not asserted. the scans inside the
    latency itself (before the flag arrives) cannot be corrected causally and are left out.
    e_k = x_{k|k} - x_true(t_k) - estimation error
    d_i = mean_k |e_k|^2 (no re-processing) - mean_k |e_k|^2 (re-processing) - paired reduction in run i
    mean(d) > 2 std(d) / sqrt(N) - measurable reduction over N runs"""
    from ekfsim.metrics import average_interval, nees
    truth = simulate_truth()
    anees, mse = {}, {}
    for reprocess in (False, True):
        values, pos2, vel2 = [], [], []
        for seed in range(20):
            res = run_scenario(scenario_cfg=ScenarioConfig(seed=seed), truth=truth,
                               filter_cfg=FilterConfig(reprocess_burnout=reprocess))
            F, idx, t = res.filter, res.plan.truth_index, res.plan.t
            k_flag = int(np.argmax(F.mode == "BALLISTIC"))
            ks = np.arange(k_flag, k_flag + 10)  # the first second after the flag change
            x_true = np.hstack([truth.position[idx[ks]], truth.velocity[idx[ks]]])
            e = F.x_post[ks, 0:6] - x_true  # e_k
            values.append(nees(F.x_post[ks], F.P_post[ks], x_true))
            pos2.append(np.mean(np.sum(e[:, 0:3] ** 2, axis=1)))
            vel2.append(np.mean(np.sum(e[:, 3:6] ** 2, axis=1)))
            if reprocess and seed == 0:
                assert t[k_flag] > truth.derived.burn_time
                T = RadarConfig().update_interval
                n_replayed = round((IRSensorConfig().latency + T) / T)
                assert F.reprocessed[k_flag] == n_replayed  # 6 scans re-processed, from the 62.9 s restore point
        anees[reprocess] = float(np.mean(values))
        mse[reprocess] = np.array([pos2, vel2])  # rows: position, velocity; one column per run
    _, hi = average_interval(200, 6)
    assert anees[True] < hi  # (a) not over-confident after the flag
    d = mse[False] - mse[True]  # d_i for position and velocity
    se = d.std(axis=1, ddof=1) / np.sqrt(d.shape[1])  # standard error of mean(d)
    assert np.all(d.mean(axis=1) > 2.0 * se)  # (b) measurable reduction in both


def test_track_through_north():
    """a ground track at N = +60 km: the azimuth runs from about 292 deg through 0 to about 40 deg,
    so the innovations cross the 0/360 wrap during the flight. the tracker must hold the lock and
    show no 360 deg innovation jumps."""
    scn = ScenarioConfig(launch_north=60_000.0)
    res = run_scenario(scenario_cfg=scn)
    F, k0 = res.filter, res.filter.track_start
    az_true = res.radar.h(res.truth.position[res.plan.truth_index], res.truth.velocity[res.plan.truth_index])[:, 1]
    tracked = np.arange(len(az_true)) > k0
    assert np.any(np.degrees(az_true[tracked & F.updated]) > 350.0)
    assert np.any(np.degrees(az_true[tracked & F.updated]) < 10.0)
    crossings = np.nonzero(np.abs(np.diff(np.degrees(az_true))) > 180.0)[0]
    assert len(crossings) == 1 and F.updated[crossings[0]] and F.updated[crossings[0] + 1]
    nu_az = F.nu[F.updated, 1]
    assert np.max(np.abs(nu_az)) < 6 * res.radar.cfg.sigma_angle
    assert not any(e.new.value == "LOST" for e in F.events)
    k_cross = crossings[0]
    assert np.all(F.nis[k_cross - 5:k_cross + 6][F.updated[k_cross - 5:k_cross + 6]] < 20.0)
    assert np.max(tracked_position_error(res)[res.plan.t[k0:] > 80.0]) < 300.0


def test_no_acquisition_gives_no_filter():
    res = run_scenario(VehicleConfig(dry_mass_delta=3_000.0),
                       radar_cfg=RadarConfig(elevation_mask=math.radians(10.0)))
    assert res.filter is None and not res.acquisition.acquired and res.acquisition.reason


def test_result_is_read_only(default_run):
    with pytest.raises(ValueError):
        default_run.filter.x_post[0, 0] = 1.0


def test_scenario_is_deterministic_per_seed():
    truth = simulate_truth()
    a = run_scenario(scenario_cfg=ScenarioConfig(seed=3), truth=truth)
    b = run_scenario(scenario_cfg=ScenarioConfig(seed=3), truth=truth)
    c = run_scenario(scenario_cfg=ScenarioConfig(seed=4), truth=truth)
    assert np.array_equal(a.filter.x_post, b.filter.x_post, equal_nan=True)
    assert not np.array_equal(a.filter.x_post, c.filter.x_post, equal_nan=True)
