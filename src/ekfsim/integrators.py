"""here I build the fixed-step RK4 integrator with event handling (THEORY 4.6). it knows no physics:
it takes a right-hand side f(t, x, seg), a list of known event times and a ground function.
- known-time events (burnout, pitch-program corners) split the step exactly at the event.
- ground impact is found by bisection on the step length, re-integrating the partial RK4 step
  from the last state above ground (not by linear interpolation, which is only 2nd order).
- there are deliberately no events for Cd(Mach) table points or ISA layer boundaries."""

from typing import Callable, NamedTuple

import numpy as np

Rhs = Callable[[float, np.ndarray, int], np.ndarray]


class IntegrationResult(NamedTuple):
    """the sampled solution of one integration."""

    t: np.ndarray  # sample times, shape (N,), in [s]
    x: np.ndarray  # states at the sample times, shape (N, n)
    segment: np.ndarray  # segment index of each sample, shape (N,); an event sample carries the segment that ends there
    event_index: tuple[int, ...]  # sample index of each known event that was reached
    impact_index: int | None  # sample index of the ground impact, None if t_max came first


# the RK4 step part:

def rk4_step(f: Rhs, t: float, x: np.ndarray, h: float, seg: int) -> np.ndarray:
    """this is one classic 4th-order Runge-Kutta step of length h [s] inside segment seg. passing seg
    explicitly means a stage evaluated exactly at a segment end still uses that segment's formula,
    so a jump at the end (thrust at burnout) is integrated exactly.
    k1 = f(t, x) - slope at the start
    k2 = f(t + h/2, x + h/2 k1) - first midpoint slope
    k3 = f(t + h/2, x + h/2 k2) - second midpoint slope
    k4 = f(t + h, x + h k3) - slope at the end
    x(t + h) = x + h/6 (k1 + 2 k2 + 2 k3 + k4) - RK4 update"""
    k1 = f(t, x, seg)  # k1 is the slope at the start of the step
    k2 = f(t + 0.5 * h, x + 0.5 * h * k1, seg)  # k2 is the first midpoint slope
    k3 = f(t + 0.5 * h, x + 0.5 * h * k2, seg)  # k3 is the second midpoint slope
    k4 = f(t + h, x + h * k3, seg)  # k4 is the slope at the end of the step
    return x + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)  # x(t+h) from the weighted slopes


# the ground impact part:

def _bisect_impact(f: Rhs, t: float, x: np.ndarray, h: float, seg: int,
                   ground: Callable[[np.ndarray], float], time_tol: float) -> tuple[float, np.ndarray]:
    """this function finds the step length s in (0, h] at which ground(x(t + s)) = 0, units [s].
    every trial re-integrates one RK4 step of length s from the last state above ground, so the impact
    point keeps the 4th-order accuracy of the integrator.
    g(s) = ground(rk4_step(t, x, s)) - height above ground after a partial step
    s_lo: g > 0, s_hi: g <= 0, halve [s_lo, s_hi] until s_hi - s_lo < time_tol - bisection
    returns s_hi, so the impact state is at or a hair below the ground"""
    s_lo, s_hi = 0.0, h  # bracket of the impact step length in [s]
    x_hi = rk4_step(f, t, x, s_hi, seg)
    while s_hi - s_lo > time_tol:
        s_mid = 0.5 * (s_lo + s_hi)  # trial step length in [s]
        x_mid = rk4_step(f, t, x, s_mid, seg)
        if ground(x_mid) > 0.0:
            s_lo = s_mid
        else:
            s_hi, x_hi = s_mid, x_mid
    return s_hi, x_hi


# the integration loop part:

def integrate(f: Rhs, x0: np.ndarray, t0: float, dt: float, event_times: tuple[float, ...],
              ground: Callable[[np.ndarray], float], t_max: float,
              time_tol: float) -> IntegrationResult:
    """here I integrate from t0 [s] with the fixed step dt [s] until ground impact or t_max [s].
    the output grid is uniform, t_k = t0 + k dt, computed by multiplication so no rounding builds up.
    a known event that falls inside a step splits it: one sub-step to the event (stored as an extra
    sample), then the rest of the step. an event within time_tol [s] of a grid point replaces that
    grid point, so no near-duplicate samples appear. the segment index counts the events passed.
    t_k = t0 + k dt - uniform output grid
    impact: ground(x) > 0 before the step and ground(x) <= 0 after it - impact test"""
    events = sorted(e for e in event_times if e > t0)
    ts, xs, segs, event_index = [t0], [np.asarray(x0, dtype=float)], [0], []
    t, x, seg, k, i_ev = t0, xs[0], 0, 0, 0
    impact_index = None
    while True:
        t_grid = t0 + (k + 1) * dt  # next grid time t_{k+1} in [s]
        is_event = i_ev < len(events) and events[i_ev] <= t_grid + time_tol
        if is_event:
            t_end = events[i_ev]  # the step ends exactly at the event time in [s]
            on_grid = t_end >= t_grid - time_tol
        else:
            t_end, on_grid = t_grid, True
        if t_end > t_max:
            break
        h = t_end - t  # length of this (sub-)step in [s]
        x_new = rk4_step(f, t, x, h, seg)
        if ground(x) > 0.0 and ground(x_new) <= 0.0:
            s, x_imp = _bisect_impact(f, t, x, h, seg, ground, time_tol)
            ts.append(t + s)
            xs.append(x_imp)
            segs.append(seg)
            impact_index = len(ts) - 1
            break
        ts.append(t_end)
        xs.append(x_new)
        segs.append(seg)
        if is_event:
            event_index.append(len(ts) - 1)
            seg += 1
            i_ev += 1
        if on_grid:
            k += 1
        t, x = t_end, x_new
    return IntegrationResult(np.array(ts), np.vstack(xs), np.array(segs, dtype=int),
                             tuple(event_index), impact_index)
