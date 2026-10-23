"""checks of the RK4 integrator (THEORY 4.6): order, exact event splitting, segment-dependent
right-hand sides and the ground-impact bisection."""

import math

import numpy as np
import pytest

from ekfsim.integrators import integrate, rk4_step

NEVER = lambda x: 1.0  # noqa: E731  ground function that never reaches the ground


def test_rk4_is_fourth_order():
    """x' = x on [0, 1]: halving the step cuts the error of x(1) = e by 16."""
    errors = []
    for n in (8, 16, 32):
        x, h = np.array([1.0]), 1.0 / n
        for k in range(n):
            x = rk4_step(lambda t, x, s: x, k * h, x, h, 0)
        errors.append(abs(x[0] - math.e))
    assert errors[1] / errors[0] == pytest.approx(1 / 16, rel=0.1)
    assert errors[2] / errors[1] == pytest.approx(1 / 16, rel=0.1)


def test_event_inside_step_is_hit_exactly():
    """x' = 1 before the event and 0 after it: x ends exactly at the event time, which needs the
    split step and the segment index passed to every stage."""
    t_event = 0.123
    f = lambda t, x, seg: np.array([1.0 if seg == 0 else 0.0])  # noqa: E731
    sol = integrate(f, np.array([0.0]), 0.0, 0.1, (t_event,), NEVER, t_max=0.5, time_tol=1e-9)
    i = sol.event_index[0]
    assert sol.t[i] == t_event and sol.segment[i] == 0
    assert np.all(sol.segment[i + 1:] == 1)
    assert sol.x[-1, 0] == pytest.approx(t_event, abs=1e-15)
    # the uniform grid is kept around the event
    grid = np.delete(sol.t, i)
    assert np.allclose(grid, 0.1 * np.arange(len(grid)), atol=1e-15)


def test_event_on_grid_point_replaces_it():
    f = lambda t, x, seg: np.array([1.0])  # noqa: E731
    sol = integrate(f, np.array([0.0]), 0.0, 0.1, (0.3 + 1e-12,), NEVER, t_max=0.5, time_tol=1e-9)
    assert len(sol.t) == 6  # 0.0 ... 0.5, no near-duplicate sample
    assert sol.t[sol.event_index[0]] == 0.3 + 1e-12


def test_impact_by_bisection_on_parabola():
    """constant gravity: the trajectory is a parabola, which RK4 integrates exactly. the ground-contact point
    must therefore match the analytic one to the bisection tolerance; linear interpolation of the
    last step would miss it by metres."""
    g, vx, vz = 9.81, 50.0, 100.0
    f = lambda t, x, seg: np.array([x[2], x[3], 0.0, -g])  # noqa: E731  x = [px, pz, vx, vz]
    sol = integrate(f, np.array([0.0, 0.0, vx, vz]), 0.0, 0.7, (), lambda x: x[1],
                    t_max=100.0, time_tol=1e-10)
    t_imp = 2.0 * vz / g  # analytic flight time in [s]
    i = sol.impact_index
    assert i == len(sol.t) - 1
    assert sol.t[i] == pytest.approx(t_imp, abs=2e-10)
    assert sol.x[i, 0] == pytest.approx(vx * t_imp, abs=1e-7)
    assert -1e-7 < sol.x[i, 1] <= 0.0


def test_t_max_without_impact():
    f = lambda t, x, seg: np.array([1.0])  # noqa: E731
    sol = integrate(f, np.array([0.0]), 0.0, 0.25, (), NEVER, t_max=1.0, time_tol=1e-9)
    assert sol.impact_index is None and sol.t[-1] == 1.0
