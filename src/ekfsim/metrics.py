"""here I compute the consistency and accuracy measures (THEORY 7): chi2 bounds, NIS, NEES and RMSE.
NIS needs only the filter's own quantities and runs online. NEES and RMSE need the true state, so they
are evaluation only ("evaluation - uses truth") and never feed back into the tracker."""

import warnings

import numpy as np
from scipy.stats import chi2


# the chi2 bounds part:

def chi2_interval(dof: int, confidence: float = 0.95) -> tuple[float, float]:
    """the two-sided chi2 interval with dof degrees of freedom that holds the given probability.
    [chi2^-1((1 - c)/2, dof), chi2^-1((1 + c)/2, dof)] - two-sided interval"""
    a = 1.0 - confidence
    return float(chi2.ppf(0.5 * a, dof)), float(chi2.ppf(1.0 - 0.5 * a, dof))


def average_interval(n_runs, dim: int, confidence: float = 0.95):
    """bounds of an average of n_runs independent chi2(dim) values, e.g. ANEES or ANIS over N Monte
    Carlo runs (Bar-Shalom et al.). n_runs can be an array (runs that contribute at each time step).
    N eps_bar ~ chi2(N n) -> bounds = chi2^-1(., N n) / N - N-run average bounds"""
    n = np.asarray(n_runs, dtype=float)
    a = 1.0 - confidence
    with np.errstate(invalid="ignore", divide="ignore"):
        lo = np.where(n > 0, chi2.ppf(0.5 * a, n * dim) / n, np.nan)
        hi = np.where(n > 0, chi2.ppf(1.0 - 0.5 * a, n * dim) / n, np.nan)
    return lo, hi


# the normalised error part:

def normalised_squared(e: np.ndarray, P: np.ndarray) -> np.ndarray:
    """eps = e^T P^-1 e for one vector (n,) or a stack (K, n) with covariances (K, n, n), from a linear
    solve, never an inverse. this is NIS with (y, S) and NEES with (x_hat - x, P).
    eps = e^T P^-1 e - normalised squared error"""
    e = np.asarray(e, dtype=float)
    P = np.asarray(P, dtype=float)
    if e.ndim == 1:
        return np.array(float(e @ np.linalg.solve(P, e)))
    return np.einsum("ki,ki->k", e, np.linalg.solve(P, e[..., None])[..., 0])


def nees(x_hat: np.ndarray, P: np.ndarray, x_true: np.ndarray) -> np.ndarray:
    """NEES on the 6 kinematic states (evaluation - uses truth). beta is left out because the truth
    beta varies with time and the filter models it as nearly constant; its error is reported apart.
    e = x_hat[0:6] - x_true[0:6] - kinematic estimation error
    NEES = e^T P_kin^-1 e - normalised estimation error squared"""
    e = np.asarray(x_hat)[..., 0:6] - np.asarray(x_true)[..., 0:6]
    return normalised_squared(e, np.asarray(P)[..., 0:6, 0:6])


def rmse(errors: np.ndarray, axis: int = 0) -> np.ndarray:
    """root mean square of error vectors over runs (axis), ignoring runs without a value (NaN).
    RMSE = sqrt(mean over runs of |e|^2) - root mean square error"""
    sq = np.sum(np.asarray(errors, dtype=float) ** 2, axis=-1)  # |e|^2 per run and time
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # time steps where no run has a value give NaN
        return np.sqrt(np.nanmean(sq, axis=axis))
