"""checks of the Cd(Mach) table (THEORY 4.4)."""

import pytest

from ekfsim.aero import CdTable
from ekfsim.config import VehicleConfig

CFG = VehicleConfig()


@pytest.fixture(scope="module")
def cd():
    return CdTable(CFG.cd_mach, CFG.cd_values)


def test_exact_at_table_points(cd):
    for M, C in zip(CFG.cd_mach, CFG.cd_values):
        assert cd(M) == C


def test_linear_between_points(cd):
    for (M0, C0), (M1, C1) in zip(zip(CFG.cd_mach, CFG.cd_values),
                                  zip(CFG.cd_mach[1:], CFG.cd_values[1:])):
        assert cd(0.5 * (M0 + M1)) == pytest.approx(0.5 * (C0 + C1), rel=1e-14)


def test_held_beyond_the_ends(cd):
    assert cd(-1.0) == CFG.cd_values[0]
    assert cd(CFG.cd_mach[-1] + 10.0) == CFG.cd_values[-1]


def test_textbook_shape(cd):
    """peak in the transonic region near Mach 1-1.2, flat subsonic, slow supersonic decline."""
    peak_mach = CFG.cd_mach[CFG.cd_values.index(max(CFG.cd_values))]
    assert 1.0 <= peak_mach <= 1.2
    assert cd(0.3) < cd(1.1) and cd(3.0) < cd(1.5) < cd(1.2)


def test_rejects_bad_tables():
    with pytest.raises(ValueError):
        CdTable((0.0, 1.0, 1.0), (0.2, 0.3, 0.3))
    with pytest.raises(ValueError):
        CdTable((0.0, 1.0), (0.2,))
    with pytest.raises(ValueError):
        CdTable((0.0, 1.0), (0.2, 0.0))
