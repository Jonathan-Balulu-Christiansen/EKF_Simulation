"""checks of the generic vehicle (THEORY 4.1, 4.5): derived values, thrust equation, pitch
program and the configuration refusal."""

import dataclasses
import math

import pytest

from ekfsim.atmosphere import US76Atmosphere
from ekfsim.config import AtmosphereConfig, PhysicsConstants, VehicleConfig
from ekfsim.vehicle import (derived_values, pitch_angle, pitch_end_time, thrust_direction,
                            thrust_magnitude, validate_vehicle)

CFG = VehicleConfig()
CONST = PhysicsConstants()
ATM = US76Atmosphere(AtmosphereConfig(), CONST)


def test_derived_values():
    d = derived_values(CFG, CONST, ATM)
    assert d.reference_area == pytest.approx(math.pi * 0.45 ** 2, rel=1e-15)
    assert d.mass_flow == pytest.approx(55.6, abs=0.05)
    assert d.burn_time == pytest.approx(63.0, abs=0.1)
    assert d.mass_flow * d.burn_time == pytest.approx(CFG.propellant_mass, rel=1e-14)
    assert d.launch_mass == 6_000.0 and d.burnout_mass == 2_500.0
    assert d.thrust_to_weight == pytest.approx(2.0, abs=0.05)


@pytest.mark.parametrize("delta", [-1_000.0, 0.0, 3_000.0])
def test_slider_changes_only_the_mass(delta):
    d0 = derived_values(CFG, CONST, ATM)
    d = validate_vehicle(dataclasses.replace(CFG, dry_mass_delta=delta), CONST, ATM)
    assert d.launch_mass == d0.launch_mass + delta
    assert d.burnout_mass == d0.burnout_mass + delta
    assert (d.mass_flow, d.burn_time) == (d0.mass_flow, d0.burn_time)
    assert d.thrust_to_weight > CFG.min_thrust_to_weight


def test_thrust_equation():
    p0 = ATM.properties(0.0).pressure
    assert thrust_magnitude(CFG, p0, p0) == CFG.thrust_sl
    for h in (5_000.0, 20_000.0, 60_000.0):
        p = ATM.properties(h).pressure
        assert thrust_magnitude(CFG, p0, p) == pytest.approx(
            CFG.thrust_sl + (p0 - p) * CFG.nozzle_exit_area, rel=1e-15)
    vacuum = thrust_magnitude(CFG, p0, 0.0)
    assert vacuum == pytest.approx(CFG.thrust_sl + p0 * CFG.nozzle_exit_area, rel=1e-15)


@pytest.mark.parametrize("t, theta_deg", [(0.0, 90.0), (5.0, 90.0), (27.5, 67.5), (50.0, 45.0),
                                          (60.0, 45.0)])
def test_pitch_program(t, theta_deg):
    assert math.degrees(pitch_angle(t, CFG)) == pytest.approx(theta_deg, abs=1e-12)


def test_pitch_corners_come_before_burnout():
    t_burn = derived_values(CFG, CONST, ATM).burn_time
    assert CFG.t_vertical < pitch_end_time(CFG) < t_burn
    assert pitch_end_time(CFG) == pytest.approx(50.0, abs=1e-12)


def test_thrust_direction_in_launch_plane():
    psi = math.radians(90.0)
    for theta in (math.radians(90.0), math.radians(60.0), math.radians(45.0)):
        u = thrust_direction(theta, psi)
        assert math.fsum(c * c for c in u) == pytest.approx(1.0, abs=1e-15)
        assert u[1] == pytest.approx(0.0, abs=1e-15)  # due East: no North component
        assert u[2] == pytest.approx(math.sin(theta), abs=1e-15)


def test_refuses_low_thrust_to_weight():
    heavy = dataclasses.replace(CFG, dry_mass_delta=6_000.0, dry_mass_delta_max=10_000.0)
    with pytest.raises(ValueError, match="thrust-to-weight"):
        validate_vehicle(heavy, CONST, ATM)


def test_refuses_slider_out_of_range():
    with pytest.raises(ValueError, match="slider range"):
        validate_vehicle(dataclasses.replace(CFG, dry_mass_delta=3_001.0), CONST, ATM)
