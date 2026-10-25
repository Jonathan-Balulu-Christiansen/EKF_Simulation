"""here I keep the drag coefficient Cd(Mach) as a piecewise-linear table and interpolate it (THEORY 4.4).
the table itself lives in VehicleConfig and is an ESTIMATE of a textbook slender-body shape."""

from bisect import bisect_right
from itertools import pairwise

# the Cd(Mach) table part:

class CdTable:
    """here I interpolate Cd linearly between the table points and hold it constant beyond both ends.
    the truth beta = m / (Cd A) therefore varies with Mach, which is intentional: the filter assumes
    beta is nearly constant (THEORY 4.4, 6.2).
    Cd(M) = Cd_i + (Cd_{i+1} - Cd_i) (M - M_i) / (M_{i+1} - M_i), M_i <= M <= M_{i+1} - linear interpolation
    Cd(M) = Cd_0 for M < M_0, Cd(M) = Cd_last for M > M_last - held end values"""

    def __init__(self, mach_points: tuple[float, ...], cd_values: tuple[float, ...]):
        if len(mach_points) != len(cd_values) or len(mach_points) < 2:
            raise ValueError("the Cd table needs at least two points and one Cd per Mach point")
        if any(b <= a for a, b in pairwise(mach_points)):
            raise ValueError("the Mach points of the Cd table must be strictly increasing")
        if any(cd <= 0.0 for cd in cd_values):
            raise ValueError("every Cd in the table must be positive")
        self.mach_points = tuple(mach_points)  # M_i, table Mach numbers [-]
        self.cd_values = tuple(cd_values)  # Cd_i, table drag coefficients [-]

    def __call__(self, mach: float) -> float:
        """this function returns Cd [-] at Mach number mach [-] (class docstring equations)."""
        M, C = self.mach_points, self.cd_values
        if mach <= M[0]:
            return C[0]
        if mach >= M[-1]:
            return C[-1]
        i = bisect_right(M, mach) - 1  # index of the table interval that contains mach
        return C[i] + (C[i + 1] - C[i]) * (mach - M[i]) / (M[i + 1] - M[i])  # linear Cd between points
