"""acceptance tests of the truth simulator (THEORY 4.7): RK4 order on a smooth configuration,
full-model convergence, energy conservation, mass bookkeeping, plus flight-structure checks."""

import dataclasses
import math
from itertools import pairwise

import numpy as np
import pytest

from ekfsim.atmosphere import ExponentialAtmosphere, US76Atmosphere
from ekfsim.config import AtmosphereConfig, PhysicsConstants, ScenarioConfig, VehicleConfig
from ekfsim.scenario import simulate_truth
from ekfsim.vehicle import TruthModel

VEH, SCN, ATM_CFG, CONST = VehicleConfig(), ScenarioConfig(), AtmosphereConfig(), PhysicsConstants()


def ballistic_start(model, speed=1_500.0, gamma_deg=45.0):
    """a launch-site state already moving at speed [m/s] and flight-path angle gamma in the launch
    plane, for the configurations without thrust."""
    g, psi = math.radians(gamma_deg), SCN.launch_azimuth
    v = (speed * math.cos(g) * math.sin(psi), speed * math.cos(g) * math.cos(psi), speed * math.sin(g))
    return np.array([SCN.launch_east, SCN.launch_north, 0.0, *v, model.derived.burnout_mass])


def run(dt, **kw):
    return simulate_truth(scenario_cfg=dataclasses.replace(SCN, dt_truth=dt, **kw.pop("scn", {})), **kw)


@pytest.fixture(scope="module")
def baseline():
    return simulate_truth()


def test_rk4_order_on_smooth_configuration():
    """constant Cd, exponential atmosphere, no thrust: each halving of dt cuts the impact-point error
    (against a dt/64 reference) by about 1/16."""
    atm = ExponentialAtmosphere(1.225, 7_200.0, ATM_CFG, CONST)  # THEORY 6.2 values
    model = TruthModel(VEH, SCN, atm, lambda mach: 0.3, CONST, thrust_on=False)
    x0 = ballistic_start(model)
    tight = {"time_tol": 1e-11}
    ref = run(1.0 / 64, model=model, initial_state=x0, scn=tight).impact_point
    err = [np.linalg.norm(run(dt, model=model, initial_state=x0, scn=tight).impact_point - ref)
           for dt in (1.0, 0.5, 0.25)]
    for e_coarse, e_fine in pairwise(err):
        assert e_fine / e_coarse == pytest.approx(1 / 16, rel=0.2)


def test_full_model_converges():
    """full model (thrust, Cd(Mach), US76): the ground-contact point moves by < 1 m when dt halves."""
    p1 = run(0.01).impact_point
    p2 = run(0.005).impact_point
    assert np.linalg.norm(p1 - p2) < 1.0


def test_energy_conserved_without_thrust_and_drag():
    """thrust off, drag off: E = 1/2 |v|^2 + g0 Re h / (Re + h) stays constant, with the potential
    consistent with g(h) = g0 (Re / (Re + h))^2."""
    atm = US76Atmosphere(ATM_CFG, CONST)
    model = TruthModel(VEH, SCN, atm, lambda mach: 0.3, CONST, thrust_on=False, drag_on=False)
    tr = simulate_truth(model=model, initial_state=ballistic_start(model))
    h, Re, g0 = tr.altitude, CONST.earth_radius, CONST.g0
    E = 0.5 * np.sum(tr.velocity ** 2, axis=1) + g0 * Re * h / (Re + h)
    assert tr.impacted and tr.t[-1] > 100.0
    assert np.max(np.abs(E - E[0])) / abs(E[0]) < 1e-6


def test_mass_bookkeeping(baseline):
    tr, d = baseline, baseline.derived
    i = tr.burnout_index
    assert tr.t[i] == d.burn_time
    consumed = tr.mass[0] - tr.mass[i]
    assert consumed == pytest.approx(d.mass_flow * d.burn_time, rel=1e-12)
    assert consumed == pytest.approx(VEH.propellant_mass, rel=1e-12)
    assert np.all(tr.mass[i:] == tr.mass[i])  # exactly constant after burnout
    assert np.all(np.diff(tr.mass[:i + 1]) < 0.0)


def test_flight_structure(baseline):
    tr = baseline
    i_bo, i_imp = tr.burnout_index, tr.impact_index
    assert np.array_equal(tr.position[0], [SCN.launch_east, SCN.launch_north, 0.0])
    assert np.all(tr.velocity[0] == 0.0)
    assert [tr.t[i] for i in tr.event_index] == list(tr.event_times)
    assert np.all(tr.boost[:i_bo + 1]) and not np.any(tr.boost[i_bo + 1:])
    assert np.all(np.linalg.norm(tr.thrust[i_bo + 1:], axis=1) == 0.0)
    assert tr.altitude.max() > tr.altitude[i_bo] > 0.0  # coast to apogee after burnout
    assert i_imp == len(tr.t) - 1 and -1e-5 < tr.altitude[i_imp] <= 0.0
    assert np.all(tr.altitude[1:-1] > 0.0)


def test_motion_stays_in_launch_plane(baseline):
    """thrust, drag and gravity all lie in the vertical plane of the launch azimuth (due East)."""
    assert np.max(np.abs(baseline.position[:, 1] - SCN.launch_north)) < 1e-6


def test_truth_beta_varies_with_mass_and_cd(baseline):
    tr = baseline
    beta = tr.mass / (tr.cd * tr.derived.reference_area)
    assert np.allclose(tr.beta, beta, rtol=1e-14)
    assert tr.beta.max() / tr.beta.min() > 2.0  # far from the constant beta the filter assumes


def test_result_is_read_only(baseline):
    with pytest.raises(ValueError):
        baseline.position[0, 0] = 1.0


@pytest.mark.parametrize("delta", [-1_000.0, 3_000.0])
def test_slider_extremes_fly(delta):
    tr = simulate_truth(dataclasses.replace(VEH, dry_mass_delta=delta))
    assert tr.impacted and tr.burnout_index is not None
    assert tr.mass[-1] == pytest.approx(tr.derived.burnout_mass, rel=1e-12)
