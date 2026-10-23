"""here I build the atmosphere models (THEORY 4.3, 4.7 and 6.2). the US Standard Atmosphere 1976 is the
one atmosphere of the project: the truth simulator uses it, and the filter uses its
density too (known atmosphere), with the analytic altitude gradient its Jacobian needs. the simple
exponential atmosphere remains only for the smooth configuration of the RK4 order test (THEORY 4.7).
both take the geometric altitude h in [m] and return T [K], p [Pa], rho [kg/m^3] and a [m/s].
scalar math only, because the RK4 right-hand sides call this four times per step."""

import math
from bisect import bisect_right
from typing import NamedTuple

from ekfsim.config import AtmosphereConfig, PhysicsConstants


class LayerBoundary(NamedTuple):
    """one boundary between two US76 layers, for display."""

    altitude: float  # geometric altitude h of the boundary in [m]
    density: float  # rho at the boundary in [kg/m^3]
    below: str  # name of the layer below
    above: str  # name of the layer above


class AtmosphereState(NamedTuple):
    """the air state at one altitude."""

    temperature: float  # T in [K]
    pressure: float  # p in [Pa]
    density: float  # rho in [kg/m^3]
    speed_of_sound: float  # a in [m/s]


# the altitude conversion part:

def geometric_to_geopotential(h: float, r0: float) -> float:
    """here I convert geometric altitude h [m] to geopotential altitude H [m']. US76 defines its layers
    in geopotential altitude, which folds the decrease of g with height into the height scale, so the
    hydrostatic equation keeps the constant g0.
    H = r0 h / (r0 + h) - geopotential altitude"""
    return r0 * h / (r0 + h)  # H is the geopotential altitude in [m']


def geopotential_to_geometric(H: float, r0: float) -> float:
    """this is the inverse conversion, geopotential H [m'] to geometric h [m].
    h = r0 H / (r0 - H) - geometric altitude"""
    return r0 * H / (r0 - H)  # h is the geometric altitude in [m]


# the US Standard Atmosphere 1976 part:

class US76Atmosphere:
    """here I build the US Standard Atmosphere 1976 (truth model): 7 layers up to H = 84.852 km
    geopotential, each with a constant temperature gradient L_b, plus a continuation above.
    inside a layer the hydrostatic equation dp/dH = -rho g0 with p = rho R T integrates in closed form:
    T = T_b + L_b (H - H_b) - layer temperature
    p = p_b (T_b / T)^(g0 / (R L_b)), L_b != 0 - pressure in a gradient layer
    p = p_b exp(-g0 (H - H_b) / (R T_b)), L_b = 0 - pressure in an isothermal layer
    rho = p / (R T) - ideal-gas density
    a = sqrt(gamma R T) - speed of sound
    simplifications, all documented:
    - I use the molecular-scale temperature as the temperature. US76 corrects it by M/M0 between 80
      and 86 km; that factor stays >= 0.9996, so T is off by at most 0.04 % there.
    - above the top (84.852 km geopotential, about 86 km geometric) I hold T at the top value. the
      isothermal hydrostatic solution is an exponential with scale height R T_top / g0 (about 5.5 km),
      matched in value at the boundary. rho is about 7e-6 kg/m^3 there, so drag is negligible anyway.
    - below sea level (reached only by RK4 stages and the impact bisection) the first layer continues.
    - r0 (US76 geopotential radius) and Re (gravity in the equations of motion) are separate constants
      of separate models."""

    def __init__(self, cfg: AtmosphereConfig, constants: PhysicsConstants):
        self._r0 = cfg.r0  # Earth radius of the geopotential conversion in [m]
        self._R = cfg.R_air  # specific gas constant R in [J/(kg K)]
        self._gamma = cfg.gamma  # ratio of specific heats [-]
        self._g0 = constants.g0  # standard gravity in [m/s^2]
        self._H_b = tuple(cfg.layer_base_H) + (cfg.H_top,)  # layer bases H_b plus the top in [m']
        self._L_b = tuple(cfg.layer_lapse_rate)  # layer gradients L_b in [K/m']
        self._names = tuple(cfg.layer_names)  # layer names, for display
        # here I integrate layer by layer from sea level, so the base T_b and p_b of every layer
        # follow from T0, p0 and the gradients instead of being typed in from a table
        T_b, p_b = [cfg.T0], [cfg.p0]
        for i, L in enumerate(self._L_b):
            T, p = self._in_layer(T_b[i], p_b[i], L, self._H_b[i + 1] - self._H_b[i])
            T_b.append(T)
            p_b.append(p)
        self._T_b = tuple(T_b)  # base temperature of each layer, last entry = top, in [K]
        self._p_b = tuple(p_b)  # base pressure of each layer, last entry = top, in [Pa]

    def _in_layer(self, T_b: float, p_b: float, L: float, dH: float) -> tuple[float, float]:
        """this function steps from a layer base up by dH [m'] inside one layer, units [K], [Pa].
        T = T_b + L dH - layer temperature
        p = p_b (T_b / T)^(g0 / (R L)) - gradient layer
        p = p_b exp(-g0 dH / (R T_b)) - isothermal layer"""
        if L == 0.0:
            return T_b, p_b * math.exp(-self._g0 * dH / (self._R * T_b))  # isothermal p in [Pa]
        T = T_b + L * dH  # T is the layer temperature in [K]
        return T, p_b * (T_b / T) ** (self._g0 / (self._R * L))  # gradient-layer p in [Pa]

    def _layer_state(self, h: float) -> tuple[float, float, float]:
        """T [K], p [Pa] and the layer gradient L [K/m'] at geometric altitude h [m] (class docstring)."""
        H = geometric_to_geopotential(h, self._r0)  # H is the geopotential altitude in [m']
        n = len(self._L_b)
        if H >= self._H_b[n]:
            # isothermal continuation above the top, exponential in p and rho
            T, p = self._in_layer(self._T_b[n], self._p_b[n], 0.0, H - self._H_b[n])
            return T, p, 0.0
        i = max(bisect_right(self._H_b, H) - 1, 0)  # index of the layer that contains H
        T, p = self._in_layer(self._T_b[i], self._p_b[i], self._L_b[i], H - self._H_b[i])
        return T, p, self._L_b[i]

    def properties(self, h: float) -> AtmosphereState:
        """this function returns the US76 air state at geometric altitude h [m].
        H = r0 h / (r0 + h) - geopotential altitude
        T, p from the layer that contains H (class docstring) - layer equations
        rho = p / (R T) - ideal-gas density
        a = sqrt(gamma R T) - speed of sound"""
        T, p, _ = self._layer_state(h)
        rho = p / (self._R * T)  # rho is the air density in [kg/m^3]
        a = math.sqrt(self._gamma * self._R * T)  # a is the speed of sound in [m/s]
        return AtmosphereState(T, p, rho, a)

    def density(self, h: float) -> float:
        """rho(h) = p / (R T) - US76 density in [kg/m^3] at geometric altitude h [m]"""
        T, p, _ = self._layer_state(h)
        return p / (self._R * T)

    def density_and_gradient(self, h: float) -> tuple[float, float]:
        """the density [kg/m^3] and its altitude derivative [kg/m^4] that the filter Jacobian needs. inside
        a layer T = T_b + L dH and p follows the hydrostatic equation, so the log-derivative is closed form;
        the isothermal layers and the continuation above the top are the case L = 0. the derivative jumps
        at layer boundaries (T has a kink there), which is part of the model and needs no smoothing.
        d ln p / dH = -g0 / (R T) - hydrostatic equation
        d ln T / dH = L / T - layer temperature gradient
        d ln rho / dH = -(g0 / (R T) + L / T) - from rho = p / (R T)
        dH/dh = (r0 / (r0 + h))^2 - geopotential vs geometric altitude
        drho/dh = rho (d ln rho / dH) (dH/dh) - density gradient"""
        T, p, L = self._layer_state(h)
        rho = p / (self._R * T)  # rho in [kg/m^3]
        dlnrho_dH = -(self._g0 / (self._R * T) + L / T)  # d ln rho / dH in [1/m']
        dH_dh = (self._r0 / (self._r0 + h)) ** 2  # dH/dh [-]
        return rho, rho * dlnrho_dH * dH_dh  # rho, drho/dh

    def boundaries(self) -> tuple[LayerBoundary, ...]:
        """the six boundaries between the seven layers at their geometric altitude, with the density there
        and the names of the layers on both sides (the 3D view draws them).
        h_b = r0 H_b / (r0 - H_b) - geometric altitude of the layer base H_b"""
        out = []
        for i in range(1, len(self._L_b)):
            h = geopotential_to_geometric(self._H_b[i], self._r0)  # h_b in [m]
            out.append(LayerBoundary(h, self.density(h), self._names[i - 1], self._names[i]))
        return tuple(out)

    @property
    def layer_bases(self) -> tuple[tuple[float, float, float], ...]:
        """the computed (H_b [m'], T_b [K], p_b [Pa]) of every layer base and the top, for tests and display."""
        return tuple(zip(self._H_b, self._T_b, self._p_b))


# the exponential atmosphere part:

class ExponentialAtmosphere:
    """here I build an exponential density model. it is the smooth atmosphere of the RK4 order test
    (THEORY 4.7), because it has no layer kinks; the filter does not use it (it uses the ISA density, THEORY 6.2).
    an exponential density with a constant scale height is exactly the hydrostatic
    solution of an isothermal atmosphere, so I give it the matching constant temperature. that makes
    it a complete, self-consistent atmosphere and not only a density.
    rho(h) = rho0 exp(-h / H) - exponential density
    T = g0 H / R - isothermal temperature that has scale height H
    p(h) = rho(h) R T - ideal-gas pressure
    a = sqrt(gamma R T) - speed of sound"""

    def __init__(self, rho0: float, scale_height: float, cfg: AtmosphereConfig,
                 constants: PhysicsConstants):
        self._rho0 = rho0  # rho0, density at h = 0 in [kg/m^3]
        self._H = scale_height  # H, density scale height in [m]
        self._R = cfg.R_air  # specific gas constant R in [J/(kg K)]
        self._T = constants.g0 * scale_height / cfg.R_air  # T is the isothermal temperature in [K]
        self._a = math.sqrt(cfg.gamma * cfg.R_air * self._T)  # a is the speed of sound in [m/s]

    def properties(self, h: float) -> AtmosphereState:
        """this function returns the air state at geometric altitude h [m] (class docstring equations)."""
        rho = self.density(h)  # rho is the exponential density in [kg/m^3]
        return AtmosphereState(self._T, rho * self._R * self._T, rho, self._a)

    def density(self, h: float) -> float:
        """rho(h) = rho0 exp(-h / H) - exponential density in [kg/m^3] at geometric altitude h [m]"""
        return self._rho0 * math.exp(-h / self._H)
