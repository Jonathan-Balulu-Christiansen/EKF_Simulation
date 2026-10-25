"""here I build the generic illustrative vehicle: the derived mass-flow values, thrust vs altitude,
the pitch program and the truth 3-DOF point-mass equations of motion (THEORY 4.1, 4.2, 4.5).
this module is truth only. the filter never imports it (no inverse crime, THEORY 6.2).
simplifications: flat Earth, no Coriolis or centrifugal terms (small at these speeds and flight
times), velocity relative to still air (no wind)."""

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import NamedTuple, Protocol

import numpy as np

from ekfsim.atmosphere import AtmosphereState
from ekfsim.config import PhysicsConstants, ScenarioConfig, VehicleConfig


class Atmosphere(Protocol):
    """any atmosphere model that returns the air state at geometric altitude h [m]."""

    def properties(self, h: float) -> AtmosphereState: ...


# the derived vehicle values part:

@dataclass(frozen=True)
class VehicleDerived:
    """values that follow from VehicleConfig. computed, never typed in (THEORY 4.1)."""

    reference_area: float  # A, body cross-section in [m^2]
    mass_flow: float  # mdot, constant propellant mass flow in [kg/s]
    burn_time: float  # t_burn in [s]
    dry_mass: float  # mass without propellant, including dry_mass_delta, in [kg]
    launch_mass: float  # m0 in [kg]
    burnout_mass: float  # mass after burnout in [kg]
    thrust_to_weight: float  # initial thrust-to-weight ratio at the launch site [-]


def thrust_magnitude(cfg: VehicleConfig, p0: float, p: float) -> float:
    """this is the rocket thrust equation with the exit-pressure term (Sutton & Biblarz). mdot stays
    constant, so only the pressure term changes with altitude. p0 and p in [Pa], result in [N].
    T(h) = T_sl + (p0 - p(h)) A_e - thrust vs altitude"""
    return cfg.thrust_sl + (p0 - p) * cfg.nozzle_exit_area  # T(h) is the thrust in [N]


def derived_values(cfg: VehicleConfig, constants: PhysicsConstants,
                   atmosphere: Atmosphere) -> VehicleDerived:
    """here I compute the derived vehicle values of THEORY 4.1. the launch site is at h = 0.
    A = pi (d/2)^2 - reference area
    mdot = T_sl / (Isp_sl g0) - mass flow
    t_burn = m_prop / mdot - burn time
    m_dry = m_launch - m_prop + dry_mass_delta - dry mass
    m0 = m_dry + m_prop - launch mass
    m_bo = m0 - m_prop - burnout mass
    T/W = T(0) / (m0 g0) - initial thrust-to-weight"""
    A = math.pi * (0.5 * cfg.diameter) ** 2  # A is the reference area in [m^2]
    mdot = cfg.thrust_sl / (cfg.isp_sl * constants.g0)  # mdot is the mass flow in [kg/s]
    t_burn = cfg.propellant_mass / mdot  # t_burn is the burn time in [s]
    m_dry = cfg.launch_mass - cfg.propellant_mass + cfg.dry_mass_delta  # dry mass in [kg]
    m0 = m_dry + cfg.propellant_mass  # m0 is the launch mass in [kg]
    m_bo = m0 - cfg.propellant_mass  # m_bo is the burnout mass in [kg]
    p0 = atmosphere.properties(0.0).pressure  # pressure at the launch site in [Pa]
    tw = thrust_magnitude(cfg, p0, p0) / (m0 * constants.g0)  # T/W at lift-off [-]
    return VehicleDerived(A, mdot, t_burn, m_dry, m0, m_bo, tw)


def validate_vehicle(cfg: VehicleConfig, constants: PhysicsConstants,
                     atmosphere: Atmosphere) -> VehicleDerived:
    """this function refuses vehicle configurations that cannot fly as specified and says why
    (ValueError with the reason). it returns the derived values when the configuration is valid.
    T/W = T(0) / (m0 g0) > T/W_min - lift-off condition"""
    if not cfg.dry_mass_delta_min <= cfg.dry_mass_delta <= cfg.dry_mass_delta_max:
        raise ValueError(
            f"dry_mass_delta = {cfg.dry_mass_delta:.0f} kg is outside the slider range "
            f"[{cfg.dry_mass_delta_min:.0f}, {cfg.dry_mass_delta_max:.0f}] kg")
    if not 0.0 < cfg.pitch_final <= 0.5 * math.pi or cfg.pitch_rate <= 0.0 or cfg.t_vertical < 0.0:
        raise ValueError("the pitch program needs 0 < pitch_final <= 90 deg, pitch_rate > 0 and t_vert >= 0")
    d = derived_values(cfg, constants, atmosphere)
    if d.dry_mass <= 0.0:
        raise ValueError(f"the dry mass {d.dry_mass:.0f} kg must be positive")
    if d.thrust_to_weight <= cfg.min_thrust_to_weight:
        raise ValueError(
            f"initial thrust-to-weight T/W = {d.thrust_to_weight:.3f} is not above "
            f"{cfg.min_thrust_to_weight:.2f}: the vehicle would barely lift off "
            f"(launch mass {d.launch_mass:.0f} kg, thrust {cfg.thrust_sl / 1000:.0f} kN)")
    return d


# the pitch program part:

def pitch_angle(t: float, cfg: VehicleConfig) -> float:
    """this is the fixed pitch program (THEORY 4.5): the pitch angle above the horizon, in [rad] at time
    t [s]. vertical first, then a linear pitch-over, then a hold until burnout.
    theta(t) = pi/2, t <= t_vert - vertical rise
    theta(t) = pi/2 - pitch_rate (t - t_vert) - linear pitch-over
    theta(t) = pitch_final once the pitch-over reaches it - hold"""
    return max(0.5 * math.pi - cfg.pitch_rate * max(t - cfg.t_vertical, 0.0), cfg.pitch_final)


def pitch_end_time(cfg: VehicleConfig) -> float:
    """the time [s] at which the pitch-over reaches pitch_final.
    t_pitch_end = t_vert + (pi/2 - pitch_final) / pitch_rate - end of the pitch-over"""
    return cfg.t_vertical + (0.5 * math.pi - cfg.pitch_final) / cfg.pitch_rate


def thrust_direction(theta: float, psi: float) -> tuple[float, float, float]:
    """the unit thrust vector in ENU for pitch angle theta [rad] in the vertical plane of the compass
    azimuth psi [rad] (clockwise from North).
    u_thrust = [cos(theta) sin(psi), cos(theta) cos(psi), sin(theta)] - thrust direction"""
    c = math.cos(theta)
    return (c * math.sin(psi), c * math.cos(psi), math.sin(theta))


# the truth equations of motion part:

class TruthForces(NamedTuple):
    """everything the truth right-hand side computes at one state, kept for the plots."""

    thrust: tuple[float, float, float]  # thrust vector T u_thrust in [N]
    drag: tuple[float, float, float]  # drag vector D in [N]
    gravity: float  # g(h), magnitude of the gravity acceleration in [m/s^2]
    density: float  # rho(h) in [kg/m^3]
    speed_of_sound: float  # a(h) in [m/s]
    mach: float  # Mach number [-]
    cd: float  # Cd(Mach) [-]
    dynamic_pressure: float  # q in [Pa]
    beta: float  # truth ballistic coefficient beta = m / (Cd A) in [kg/m^2]
    mass_flow: float  # mdot in this segment, 0 after burnout, in [kg/s]


class TruthModel:
    """here I build the truth 3-DOF point-mass model of THEORY 4.2. state X = [p (3), v (3), m] in
    [m, m/s, kg], ENU frame, h = p_U above the flat ground. the flags switch thrust and drag off for
    the energy and RK4-order tests (THEORY 4.7); the default is the full model.
    segments: the known event times (pitch-program corners, burnout) split the flight into segments,
    numbered from 0. the integrator passes the segment index, and thrust is on in every segment
    before burnout."""

    def __init__(self, vehicle_cfg: VehicleConfig, scenario_cfg: ScenarioConfig,
                 atmosphere: Atmosphere, cd_of_mach: Callable[[float], float],
                 constants: PhysicsConstants, thrust_on: bool = True, drag_on: bool = True):
        self.vehicle_cfg = vehicle_cfg
        self.derived = derived_values(vehicle_cfg, constants, atmosphere)
        self.thrust_on = thrust_on  # False: no thrust and no mass flow at all
        self.drag_on = drag_on  # False: no drag force
        self._atm = atmosphere
        self._cd = cd_of_mach
        self._g0 = constants.g0  # standard gravity in [m/s^2]
        self._Re = constants.earth_radius  # Re in [m]
        self._A = self.derived.reference_area  # A in [m^2]
        self._p0 = atmosphere.properties(0.0).pressure  # p0 of the thrust equation in [Pa]
        self._psi = scenario_cfg.launch_azimuth  # psi, launch azimuth in [rad]
        self._events = self._known_events()

    def _known_events(self) -> tuple[float, ...]:
        """the known event times [s]: the pitch-program corners that come before burnout, then burnout."""
        if not self.thrust_on:
            return ()
        t_burn = self.derived.burn_time
        corners = [self.vehicle_cfg.t_vertical, pitch_end_time(self.vehicle_cfg)]
        events = []
        for e in corners:
            if 0.0 < e < t_burn and (not events or e > events[-1]):
                events.append(e)
        return tuple(events) + (t_burn,)

    def event_times(self) -> tuple[float, ...]:
        return self._events

    def is_boost(self, seg: int) -> bool:
        """thrust is on in every segment before the burnout event."""
        return self.thrust_on and seg < len(self._events)

    def forces(self, t: float, X: np.ndarray, seg: int) -> TruthForces:
        """this function computes all forces at time t [s], state X, segment seg (THEORY 4.2).
        h = p_U - altitude above the flat ground
        Mach = |v| / a(h) - Mach number
        q = 1/2 rho(h) |v|^2 - dynamic pressure
        D = -1/2 rho(h) |v| v Cd(Mach) A - drag vector
        T(h) = T_sl + (p0 - p(h)) A_e - thrust with the exit-pressure term
        u_thrust = [cos(theta) sin(psi), cos(theta) cos(psi), sin(theta)] - thrust direction
        g(h) = g0 (Re / (Re + h))^2 - inverse-square gravity
        beta = m / (Cd A) - truth ballistic coefficient"""
        _, _, h, vE, vN, vU, m = X.tolist()
        air = self._atm.properties(h)
        speed = math.sqrt(vE * vE + vN * vN + vU * vU)  # |v| in [m/s]
        mach = speed / air.speed_of_sound  # Mach number [-]
        cd = self._cd(mach)  # Cd(Mach) [-]
        q = 0.5 * air.density * speed * speed  # q is the dynamic pressure in [Pa]
        if self.drag_on:
            k = -0.5 * air.density * speed * cd * self._A  # D = k v, k in [kg/m]
            drag = (k * vE, k * vN, k * vU)  # D is the drag vector in [N]
        else:
            drag = (0.0, 0.0, 0.0)
        if self.is_boost(seg):
            T = thrust_magnitude(self.vehicle_cfg, self._p0, air.pressure)  # T(h) in [N]
            u = thrust_direction(pitch_angle(t, self.vehicle_cfg), self._psi)  # u_thrust [-]
            thrust = (T * u[0], T * u[1], T * u[2])  # thrust vector in [N]
            mdot = self.derived.mass_flow  # mdot in [kg/s]
        else:
            thrust, mdot = (0.0, 0.0, 0.0), 0.0
        g = self._g0 * (self._Re / (self._Re + h)) ** 2  # g(h) in [m/s^2]
        beta = m / (cd * self._A)  # beta is the truth ballistic coefficient in [kg/m^2]
        return TruthForces(thrust, drag, g, air.density, air.speed_of_sound, mach, cd, q, beta, mdot)

    def rhs(self, t: float, X: np.ndarray, seg: int) -> np.ndarray:
        """this is the truth right-hand side dX/dt (THEORY 4.2), units [m/s, m/s^2, kg/s].
        dp/dt = v - kinematics
        dv/dt = (T u_thrust + D) / m + [0, 0, -g(h)] - Newton's second law for a point mass
        dm/dt = -mdot (t < t_burn), 0 afterwards - propellant consumption"""
        F = self.forces(t, X, seg)
        _, _, _, vE, vN, vU, m = X.tolist()
        aE = (F.thrust[0] + F.drag[0]) / m  # East acceleration in [m/s^2]
        aN = (F.thrust[1] + F.drag[1]) / m  # North acceleration in [m/s^2]
        aU = (F.thrust[2] + F.drag[2]) / m - F.gravity  # Up acceleration with gravity in [m/s^2]
        return np.array([vE, vN, vU, aE, aN, aU, -F.mass_flow])

    def launch_state(self, scenario_cfg: ScenarioConfig) -> np.ndarray:
        """the state at lift-off: at the launch site on the ground, at rest, with the full launch mass.
        X0 = [E_launch, N_launch, 0, 0, 0, 0, m0] - initial state"""
        return np.array([scenario_cfg.launch_east, scenario_cfg.launch_north, 0.0,
                         0.0, 0.0, 0.0, self.derived.launch_mass])
