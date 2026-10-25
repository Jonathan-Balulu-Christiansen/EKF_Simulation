"""checks of the atmosphere models (THEORY 4.3): US76 reference values at geometric altitudes,
continuity at the layer boundaries and at the top, and the exponential model."""

import math
from itertools import pairwise

import pytest

from ekfsim.atmosphere import (
    ExponentialAtmosphere,
    US76Atmosphere,
    geometric_to_geopotential,
    geopotential_to_geometric,
)
from ekfsim.config import AtmosphereConfig, PhysicsConstants

CFG = AtmosphereConfig()
CONST = PhysicsConstants()


@pytest.fixture(scope="module")
def us76():
    return US76Atmosphere(CFG, CONST)


@pytest.mark.parametrize("h, rho_ref", [
    (0.0, 1.2250),        # US76 sea level
    (11_000.0, 0.3648),   # US76 table, 11 km geometric
    (20_000.0, 0.0889),   # US76 table, 20 km geometric
])
def test_density_reference_values(us76, h, rho_ref):
    assert us76.properties(h).density == pytest.approx(rho_ref, rel=5e-3)


def test_sea_level_temperature_and_pressure(us76):
    air = us76.properties(0.0)
    assert air.temperature == pytest.approx(288.15, abs=1e-12)
    assert air.pressure == pytest.approx(101_325.0, abs=1e-9)
    assert air.speed_of_sound == pytest.approx(340.29, abs=0.01)  # US76 sea-level a


def test_tropopause_layer_base(us76):
    """the computed base of the 11 km geopotential layer matches the US76 table."""
    H_b, T_b, p_b = us76.layer_bases[1]
    assert H_b == 11_000.0
    assert T_b == pytest.approx(216.65, abs=1e-9)
    assert p_b == pytest.approx(22_632.06, rel=1e-5)


def test_continuous_at_layer_boundaries_and_top(us76):
    for H_b, _, _ in us76.layer_bases[1:]:
        h_b = geopotential_to_geometric(H_b, CFG.r0)
        below, above = us76.properties(h_b - 1e-6), us76.properties(h_b + 1e-6)
        for a, b in zip(below, above):
            assert a == pytest.approx(b, rel=1e-8)


def test_above_top_is_isothermal_exponential(us76):
    h_top = geopotential_to_geometric(CFG.H_top, CFG.r0)
    top = us76.properties(h_top)
    assert top.density == pytest.approx(6.958e-6, rel=0.01)  # US76 table, 86 km geometric
    higher = us76.properties(h_top + 10_000.0)
    assert higher.temperature == top.temperature
    assert higher.density < top.density


def test_density_decreases_with_altitude(us76):
    rho = [us76.properties(h).density for h in range(-500, 150_001, 500)]
    assert all(b < a for a, b in pairwise(rho))


@pytest.mark.parametrize("h", [-500.0, 0.0, 11_000.0, 50_000.0, 120_000.0])
def test_geopotential_round_trip(h):
    H = geometric_to_geopotential(h, CFG.r0)
    assert geopotential_to_geometric(H, CFG.r0) == pytest.approx(h, abs=1e-8)


def test_exponential_atmosphere():
    rho0, H = 1.225, 7_200.0  # THEORY 6.2 values
    atm = ExponentialAtmosphere(rho0, H, CFG, CONST)
    assert atm.properties(0.0).density == rho0
    assert atm.properties(H).density == pytest.approx(rho0 / math.e, rel=1e-14)
    air = atm.properties(5_000.0)
    # isothermal hydrostatics: the scale height of the temperature it reports is H itself
    assert CFG.R_air * air.temperature / CONST.g0 == pytest.approx(H, rel=1e-14)
    assert air.pressure == pytest.approx(air.density * CFG.R_air * air.temperature, rel=1e-14)


def test_layer_boundaries_for_the_3d_view():
    """the ISA layer boundaries the 3D view draws come from the model: geometric altitude of each layer base,
    the density there, and the layer names (US76 nomenclature)."""
    from ekfsim.atmosphere import geopotential_to_geometric
    cfg = AtmosphereConfig()
    atm = US76Atmosphere(cfg, PhysicsConstants())
    b = atm.boundaries()
    assert len(b) == 6
    assert [round(x.altitude) for x in b] == [round(geopotential_to_geometric(H, cfg.r0)) for H in cfg.layer_base_H[1:]]
    assert b[0].altitude == pytest.approx(11_019.0, abs=1.0) and b[0].density == pytest.approx(0.3639, rel=5e-3)
    assert b[1].density == pytest.approx(0.0880, rel=5e-3)
    assert (b[0].below, b[0].above, b[3].above, b[4].above) == ("troposphere", "stratosphere", "stratopause", "mesosphere")
    assert all(x.density == atm.density(x.altitude) for x in b)
