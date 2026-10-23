"""checks of the consistency measures (THEORY 7)."""

import numpy as np
import pytest

from ekfsim.metrics import average_interval, chi2_interval, nees, normalised_squared, rmse
from ekfsim.montecarlo import OVERCONFIDENT_MAX, aggregate, flight_phases, mode_summary
from ekfsim.config import FilterConfig, RadarConfig, PhysicsConstants, ScenarioConfig
from ekfsim.radar import Radar
from ekfsim.scenario import plan_scans, simulate_truth


def test_chi2_interval_reference_values():
    lo, hi = chi2_interval(4, 0.95)
    assert lo == pytest.approx(0.4844, abs=1e-4) and hi == pytest.approx(11.1433, abs=1e-4)
    lo6, hi6 = chi2_interval(6, 0.95)
    assert lo6 == pytest.approx(1.2373, abs=1e-4) and hi6 == pytest.approx(14.4494, abs=1e-4)


def test_average_interval_shrinks_with_runs():
    lo1, hi1 = average_interval(1, 6)
    assert (float(lo1), float(hi1)) == pytest.approx(chi2_interval(6))
    lo, hi = average_interval(np.array([0, 100]), 6)
    assert np.isnan(lo[0]) and np.isnan(hi[0])  # no runs, no bound
    assert lo[1] == pytest.approx(5.34, abs=0.01) and hi[1] == pytest.approx(6.70, abs=0.01)


def test_normalised_squared_and_nees():
    e = np.array([1.0, 2.0, 2.0])
    assert float(normalised_squared(e, np.eye(3))) == pytest.approx(9.0)
    assert float(normalised_squared(e, 4 * np.eye(3))) == pytest.approx(9.0 / 4)
    E = np.tile(e, (5, 1))
    P = np.tile(np.eye(3), (5, 1, 1))
    assert np.allclose(normalised_squared(E, P), 9.0)
    x_hat = np.r_[1.0, 0, 0, 0, 0, 0, 99.0]  # the ln beta component is left out of NEES
    assert float(nees(x_hat, np.eye(7), np.zeros(7))) == pytest.approx(1.0)


def test_nees_is_chi2_distributed_for_a_consistent_estimator():
    rng = np.random.default_rng(0)
    P = np.diag([4.0, 9.0, 1.0, 0.25, 0.5, 2.0, 1.0])
    e = rng.standard_normal((20_000, 7)) * np.sqrt(np.diag(P))
    eps = nees(e, np.tile(P, (20_000, 1, 1)), np.zeros(7))
    assert eps.mean() == pytest.approx(6.0, rel=0.03)


def test_rmse_ignores_missing_runs():
    err = np.array([[[3.0, 4.0], [np.nan, np.nan]], [[0.0, 0.0], [1.0, 0.0]]])  # (runs, time, dim)
    assert np.allclose(rmse(err), [np.sqrt(12.5), 1.0])


def test_flight_phases_from_truth():
    """evaluation labels: boost while thrust is on, then coast, then re-entry from the first descending
    scan with truth drag >= 0.05 g0, until impact."""
    truth = simulate_truth()
    plan = plan_scans(truth, Radar(RadarConfig(), PhysicsConstants()), ScenarioConfig())
    ph = flight_phases(truth, plan)
    idx = plan.truth_index
    assert np.array_equal(ph == "boost", np.asarray(truth.boost[idx]))
    k_re = int(np.argmax(ph == "reentry"))
    assert np.all(ph[k_re:] == "reentry") and np.all(ph[:k_re][~truth.boost[idx][:k_re]] == "coast")
    assert truth.velocity[idx[k_re], 2] < 0.0
    a_drag = np.linalg.norm(truth.drag[idx[k_re]]) / truth.mass[idx[k_re]]
    assert a_drag >= 0.05 * 9.80665
    assert 200.0 < plan.t[k_re] < 270.0  # baseline: during the descent, well after apogee


def test_overconfidence_criterion():
    """the acceptance rule: ANEES above the upper bound counts as over-confident, below as conservative; the
    verdict uses only the over-confident share of the ballistic scans."""
    n_scans = 100
    phases = np.array(["boost"] * 20 + ["coast"] * 50 + ["reentry"] * 30)
    lo, hi = average_interval(50, 6)
    def runs_with(anees_value):
        return [{"nees": np.full(n_scans, anees_value), "nis": np.full(n_scans, 4.0),
                 "pos_err": np.zeros((n_scans, 3)), "vel_err": np.zeros((n_scans, 3)),
                 "ln_beta_err": np.zeros(n_scans), "sigma_ln_beta": np.ones(n_scans),
                 "net_on": np.zeros(n_scans), "thrust_err": np.full((n_scans, 3), np.nan),
                 "lock_time": 1.0, "lost": False, "track_start": 0}
                for _ in range(50)]
    for value, accepted, key in ((0.5 * float(lo), True, "below"), (2.0 * float(hi), False, "above"),
                                 (6.0, True, "inside")):
        a = aggregate(runs_with(value), phases, 4)
        s = mode_summary(a, runs_with(value), phases, FilterConfig())
        assert s["consistency_shares"]["anees"]["ballistic"][key] == 1.0
        assert s["acceptance_overconfident_max_10pct"] is accepted
    assert OVERCONFIDENT_MAX == 0.10
