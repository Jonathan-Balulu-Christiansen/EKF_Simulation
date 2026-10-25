"""here I run one full scenario and return it as an immutable result (THEORY 8.1). the truth flight is
computed first and on its own, because it does not depend on the radar or the tracker. then the radar
scans it: cued until the first detection, after that scan by scan with the EKF pointing the beam at
its own prediction, so radar and filter run interleaved (THEORY 5.4). the IR plume flag (THEORY 5.7) comes
with every scan and sets the filter mode.
everything in TruthResult is ground truth: only the measurement generator (radar.scan) and evaluation
views (labelled "evaluation - uses truth") may read it (THEORY 0.2). the beam is never pointed from truth."""

import math
from dataclasses import dataclass

import numpy as np

from ekfsim.aero import CdTable
from ekfsim.atmosphere import US76Atmosphere
from ekfsim.config import (
    AtmosphereConfig,
    FilterConfig,
    IRSensorConfig,
    PhysicsConstants,
    RadarConfig,
    ScenarioConfig,
    VehicleConfig,
)
from ekfsim.ekf import N_STATE, FilterModel, StatusEvent, StepRecord, Tracker
from ekfsim.integrators import integrate
from ekfsim.radar import Radar, Scan, cue_direction, engine_flags, pointing_vector
from ekfsim.vehicle import TruthModel, VehicleDerived, validate_vehicle


def _frozen(a) -> np.ndarray:
    a = np.array(a)
    a.setflags(write=False)
    return a


# the truth result part:

@dataclass(frozen=True)
class TruthResult:
    """the full truth time series of one flight (THEORY 4.6), all arrays read-only. samples lie on the
    uniform grid t_k = k dt_truth, plus one sample at each known event and one at ground impact.
    an event sample carries the segment that ends there (the left limit), so the burnout sample still
    shows full thrust and the next sample shows none."""

    t: np.ndarray  # sample times, (N,), in [s]
    position: np.ndarray  # p = [E, N, U], (N, 3), in [m]
    velocity: np.ndarray  # v = [vE, vN, vU], (N, 3), in [m/s]
    mass: np.ndarray  # m, (N,), in [kg]
    segment: np.ndarray  # segment index, (N,)
    boost: np.ndarray  # True while thrust is on (phase BOOST), else BALLISTIC, (N,)
    speed: np.ndarray  # |v|, (N,), in [m/s]
    mach: np.ndarray  # Mach number, (N,) [-]
    density: np.ndarray  # rho(h), (N,), in [kg/m^3]
    cd: np.ndarray  # Cd(Mach), (N,) [-]
    beta: np.ndarray  # truth beta = m / (Cd A), (N,), in [kg/m^2]
    thrust: np.ndarray  # thrust vector, (N, 3), in [N]
    drag: np.ndarray  # drag vector, (N, 3), in [N]
    dynamic_pressure: np.ndarray  # q, (N,), in [Pa]
    gravity: np.ndarray  # g(h), (N,), in [m/s^2]
    event_times: tuple[float, ...]  # known event times (pitch corners, burnout) in [s]
    event_index: tuple[int, ...]  # sample index of each reached event
    burnout_index: int | None  # sample index of burnout, None without thrust
    impact_index: int | None  # sample index of ground impact, None if t_max came first
    derived: VehicleDerived  # derived vehicle values of this run
    dt: float  # integration step dt_truth in [s]

    @property
    def altitude(self) -> np.ndarray:
        """h = p_U, altitude above the flat ground in [m]."""
        return self.position[:, 2]

    @property
    def impacted(self) -> bool:
        return self.impact_index is not None

    @property
    def impact_point(self) -> np.ndarray:
        """ground-impact position [E, N, U] in [m]."""
        if self.impact_index is None:
            raise ValueError("this flight reached t_max without ground impact")
        return self.position[self.impact_index]


def build_truth_model(vehicle_cfg: VehicleConfig, scenario_cfg: ScenarioConfig,
                      atmosphere_cfg: AtmosphereConfig, constants: PhysicsConstants) -> TruthModel:
    """the full truth model: US76 atmosphere, the Cd(Mach) table, thrust and drag on."""
    atmosphere = US76Atmosphere(atmosphere_cfg, constants)
    validate_vehicle(vehicle_cfg, constants, atmosphere)
    cd = CdTable(vehicle_cfg.cd_mach, vehicle_cfg.cd_values)
    return TruthModel(vehicle_cfg, scenario_cfg, atmosphere, cd, constants)


def simulate_truth(vehicle_cfg: VehicleConfig | None = None,
                   scenario_cfg: ScenarioConfig | None = None,
                   atmosphere_cfg: AtmosphereConfig | None = None,
                   constants: PhysicsConstants | None = None, *,
                   model: TruthModel | None = None,
                   initial_state: np.ndarray | None = None) -> TruthResult:
    """here I run the truth flight: build (or take) the model, start from the launch state, integrate
    with RK4 at dt_truth until ground impact, then record the forces at every sample. the forces come
    from the same TruthModel.forces() that the right-hand side uses, so plots and dynamics never differ.
    tests pass their own model (thrust or drag off, smooth atmosphere) and initial state."""
    vehicle_cfg = vehicle_cfg or VehicleConfig()
    scenario_cfg = scenario_cfg or ScenarioConfig()
    atmosphere_cfg = atmosphere_cfg or AtmosphereConfig()
    constants = constants or PhysicsConstants()
    if model is None:
        model = build_truth_model(vehicle_cfg, scenario_cfg, atmosphere_cfg, constants)
    x0 = model.launch_state(scenario_cfg) if initial_state is None else np.asarray(initial_state, float)

    sol = integrate(model.rhs, x0, 0.0, scenario_cfg.dt_truth, model.event_times(),
                    ground=lambda x: x[2], t_max=scenario_cfg.t_max,
                    time_tol=scenario_cfg.time_tol)

    forces = [model.forces(t, x, s) for t, x, s in zip(sol.t.tolist(), sol.x, sol.segment.tolist())]
    n_events = len(model.event_times())
    burnout_index = sol.event_index[-1] if model.thrust_on and len(sol.event_index) == n_events else None
    return TruthResult(
        t=_frozen(sol.t),
        position=_frozen(sol.x[:, 0:3]),
        velocity=_frozen(sol.x[:, 3:6]),
        mass=_frozen(sol.x[:, 6]),
        segment=_frozen(sol.segment),
        boost=_frozen([model.is_boost(s) for s in sol.segment.tolist()]),
        speed=_frozen(np.linalg.norm(sol.x[:, 3:6], axis=1)),
        mach=_frozen([f.mach for f in forces]),
        density=_frozen([f.density for f in forces]),
        cd=_frozen([f.cd for f in forces]),
        beta=_frozen([f.beta for f in forces]),
        thrust=_frozen([f.thrust for f in forces]),
        drag=_frozen([f.drag for f in forces]),
        dynamic_pressure=_frozen([f.dynamic_pressure for f in forces]),
        gravity=_frozen([f.gravity for f in forces]),
        event_times=model.event_times(),
        event_index=sol.event_index,
        burnout_index=burnout_index,
        impact_index=sol.impact_index,
        derived=model.derived,
        dt=scenario_cfg.dt_truth,
    )


# the flight summary part (evaluation - uses truth):

@dataclass(frozen=True)
class TruthSummary:
    """key numbers of one truth flight, for the plausibility report (THEORY 4.7) and the geometry check
    of THEORY 9. information only, no pass/fail."""

    apogee_altitude: float  # max h in [m]
    apogee_time: float  # time of max h in [s]
    flight_time: float  # time of ground impact in [s]
    burnout_time: float  # t_burn in [s]
    burnout_speed: float  # |v| at burnout in [m/s]
    burnout_altitude: float  # h at burnout in [m]
    burnout_flight_path_angle: float  # gamma = atan2(vU, |v_horizontal|) at burnout in [rad]
    max_mach: float  # max Mach number [-]
    max_mach_time: float  # time of max Mach in [s]
    max_dynamic_pressure: float  # max q in [Pa]
    max_dynamic_pressure_time: float  # time of max q in [s]
    min_horizontal_distance: float  # min horizontal distance to the reference point in [m]
    max_elevation: float  # max elevation seen from the reference point in [rad]
    max_elevation_time: float  # time of max elevation in [s]


def truth_summary(truth: TruthResult,
                  reference_point: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> TruthSummary:
    """here I extract the plausibility numbers of one flight. geometry is seen from reference_point
    [m] (default: the ENU origin, the radar ground position of THEORY 9).
    gamma = atan2(vU, sqrt(vE^2 + vN^2)) - flight-path angle
    rho_h = sqrt(dE^2 + dN^2) - horizontal distance to the reference point
    el = atan2(dU, rho_h) - elevation (THEORY 3)"""
    if truth.burnout_index is None or truth.impact_index is None:
        raise ValueError("the summary needs a powered flight that reaches the ground")
    i_bo, i_imp = truth.burnout_index, truth.impact_index
    h = truth.altitude
    d = truth.position - np.asarray(reference_point)  # d = p - p_ref in [m]
    rho_h = np.hypot(d[:, 0], d[:, 1])  # horizontal distance in [m]
    el = np.arctan2(d[:, 2], rho_h)  # elevation in [rad]
    v_bo = truth.velocity[i_bo]
    i_apo, i_mach, i_q, i_el = (int(np.argmax(a)) for a in (h, truth.mach, truth.dynamic_pressure, el))
    return TruthSummary(
        apogee_altitude=float(h[i_apo]),
        apogee_time=float(truth.t[i_apo]),
        flight_time=float(truth.t[i_imp]),
        burnout_time=float(truth.t[i_bo]),
        burnout_speed=float(truth.speed[i_bo]),
        burnout_altitude=float(h[i_bo]),
        burnout_flight_path_angle=math.atan2(v_bo[2], math.hypot(v_bo[0], v_bo[1])),
        max_mach=float(truth.mach[i_mach]),
        max_mach_time=float(truth.t[i_mach]),
        max_dynamic_pressure=float(truth.dynamic_pressure[i_q]),
        max_dynamic_pressure_time=float(truth.t[i_q]),
        min_horizontal_distance=float(rho_h.min()),
        max_elevation=float(el[i_el]),
        max_elevation_time=float(truth.t[i_el]),
    )


# the scan plan part:

@dataclass(frozen=True)
class ScanPlan:
    """the scan schedule of one run: emission times, the truth sample at each of them, the noise drawn
    up front from the run's seed, and the IR engine flag reported with each scan (THEORY 5.7). drawing one noise row per scan, detected or not, keeps the
    noise of scan k the same whatever the detection pattern (common random numbers between runs
    that differ only in pointing or filter settings)."""

    t: np.ndarray  # scan emission times t_k = k T_update, (K,), in [s]
    truth_index: np.ndarray  # index of the truth sample at each t_k, (K,)
    noise: np.ndarray  # noise rows w_k, (K, 4), in [m, rad, rad, m/s]
    engine_flag: np.ndarray  # IR plume flag reported at t_k (engine on, delayed by the sensor latency), (K,)


def plan_scans(truth: TruthResult, radar: Radar, scenario_cfg: ScenarioConfig,
               ir_cfg: IRSensorConfig | None = None) -> ScanPlan:
    """here I schedule the scans from t = 0 until ground impact. every scan time must coincide with a
    truth sample, so the measurement generator reads the truth state exactly, without interpolation.
    that holds when T_update is a whole multiple of dt_truth (10 x 0.01 s by default).
    t_k = k T_update, k = 0, 1, ... while t_k <= t_impact - scan emission times"""
    T = radar.cfg.update_interval  # T_update in [s]
    tol = scenario_cfg.time_tol  # two times closer than this are one instant, in [s]
    n = math.floor((truth.t[-1] + tol) / T) + 1  # number of scans up to the last truth sample
    t = np.arange(n) * T  # t_k in [s]
    idx = np.searchsorted(truth.t, t - tol)  # first truth sample at or after t_k - tol
    if np.any(idx >= len(truth.t)) or np.any(np.abs(truth.t[np.minimum(idx, len(truth.t) - 1)] - t) > tol):
        raise ValueError(f"update interval {T} s is not a whole multiple of dt_truth {truth.dt} s: "
                         "scan times must land on truth samples")
    rng = np.random.default_rng(scenario_cfg.seed)
    ir_cfg = ir_cfg or IRSensorConfig()
    t_off = truth.t[truth.burnout_index] if truth.burnout_index is not None else 0.0  # true burnout in [s]
    flags = engine_flags(t, 0.0, t_off, ir_cfg.latency) if truth.burnout_index is not None else np.zeros(n, bool)
    return ScanPlan(_frozen(t), _frozen(idx), _frozen(radar.noise(n, rng)), _frozen(flags))


# the cued acquisition part:

@dataclass(frozen=True)
class Acquisition:
    """the search before the first track (THEORY 5.4): the beam stays at the cue until the first detection."""

    cue_azimuth: float  # az_cue in [rad]
    cue_elevation: float  # el_cue in [rad]
    scans: tuple[Scan, ...]  # every cued scan, up to and including the first detection
    first_detection: Scan | None  # None: NO ACQUISITION
    reason: str  # why there is no acquisition; empty when acquired

    @property
    def acquired(self) -> bool:
        return self.first_detection is not None


def acquire(truth: TruthResult, radar: Radar, plan: ScanPlan, radar_cfg: RadarConfig,
            scenario_cfg: ScenarioConfig) -> Acquisition:
    """here I run the pre-track scans of THEORY 5.4. the pointing is the cue, computed from config only
    (no truth). the truth state enters only the measurement generator, radar.scan(). the search stops
    at the first detection; from there on the tracker points the beam. if the target never
    enters the cued beam above the mask, the result carries the reason for the "NO ACQUISITION" message.
    u_cue = [cos(el_cue) sin(az_cue), cos(el_cue) cos(az_cue), sin(el_cue)] - cue beam axis"""
    az, el = cue_direction(radar_cfg, scenario_cfg)
    u = pointing_vector(az, el)  # u_cue, unit vector in ENU
    scans = []
    for k, (t_k, i) in enumerate(zip(plan.t.tolist(), plan.truth_index.tolist())):
        s = radar.scan(k, t_k, truth.position[i], truth.velocity[i], u, plan.noise[k])
        scans.append(s)
        if s.detected:
            return Acquisition(az, el, tuple(scans), s, "")
    reason = (f"the target never entered the cued beam (az {math.degrees(az):.1f} deg, el "
              f"{math.degrees(el):.2f} deg, half-angle {math.degrees(radar.half_beamwidth):.2f} deg) "
              f"above the {math.degrees(radar_cfg.elevation_mask):.1f} deg elevation mask")
    return Acquisition(az, el, tuple(scans), None, reason)


# the filter history part:

@dataclass(frozen=True)
class FilterHistory:
    """the per-scan EKF history over all K scans of the run, read-only. rows before the track starts
    are NaN; prior and update rows are NaN where they do not exist (init step, misses). this is what
    the plots and the equations panel show (THEORY 6.4)."""

    x_prior: np.ndarray  # x_{k|k-1}, (K, 7)
    P_prior: np.ndarray  # P_{k|k-1}, (K, 7, 7)
    Phi: np.ndarray  # Phi_k, (K, 7, 7)
    Q: np.ndarray  # Q_k, (K, 7, 7)
    nu: np.ndarray  # innovation y_k, (K, m)
    S: np.ndarray  # S_k, (K, m, m)
    K: np.ndarray  # Kalman gain K_k, (K, 7, m)
    nis: np.ndarray  # NIS, (K,)
    x_post: np.ndarray  # x_{k|k}, (K, 7)
    P_post: np.ndarray  # P_{k|k}, (K, 7, 7)
    updated: np.ndarray  # True where a measurement update happened, (K,)
    skipped_singular: np.ndarray  # update skipped by the singularity guard, (K,)
    pd_ok: np.ndarray  # P_{k|k} passed the Cholesky test, (K,)
    q_a: np.ndarray  # drag-scheduled spectral density q_a used in Q_k in [m^2/s^3], (K,)
    manoeuvre_on: np.ndarray  # the safety net's manoeuvre noise was in Q_k (THEORY 6.6), (K,)
    reprocessed: np.ndarray  # number of scans re-processed in BALLISTIC at this scan (THEORY 6.2), (K,)
    beta_frozen: np.ndarray  # ln beta not updated this scan (BOOST mode, THEORY 6.2), (K,)
    mode: np.ndarray  # filter mode name per scan (BOOST / BALLISTIC), "" before the track, (K,)
    engine_flag: np.ndarray  # IR engine flag received with each tracked scan, (K,)
    window_nis: np.ndarray  # window-average NIS after the step, (K,)
    status: np.ndarray  # track status name per scan, "" before the track, (K,)
    events: tuple[StatusEvent, ...]  # logged status transitions and P failures
    track_start: int  # scan index of the first detection (initialisation)


def _history(records: dict[int, StepRecord], n_scans: int, m: int, events, k0: int) -> FilterHistory:
    n = N_STATE
    a = {name: np.full(shape, np.nan) for name, shape in (
        ("x_prior", (n_scans, n)), ("P_prior", (n_scans, n, n)), ("Phi", (n_scans, n, n)),
        ("Q", (n_scans, n, n)), ("nu", (n_scans, m)), ("S", (n_scans, m, m)), ("K", (n_scans, n, m)),
        ("nis", (n_scans,)), ("x_post", (n_scans, n)), ("P_post", (n_scans, n, n)),
        ("q_a", (n_scans,)), ("window_nis", (n_scans,)))}
    updated = np.zeros(n_scans, bool)
    skipped = np.zeros(n_scans, bool)
    frozen = np.zeros(n_scans, bool)
    net = np.zeros(n_scans, bool)
    reproc = np.zeros(n_scans, int)
    pd_ok = np.ones(n_scans, bool)
    status = np.full(n_scans, "", dtype="<U9")
    mode = np.full(n_scans, "", dtype="<U9")
    flag = np.zeros(n_scans, bool)
    for k, r in records.items():
        if r.x_prior is not None:
            a["x_prior"][k], a["P_prior"][k], a["Phi"][k], a["Q"][k] = r.x_prior, r.P_prior, r.Phi, r.Q
        if r.update is not None:
            u = r.update
            a["nu"][k], a["S"][k], a["K"][k], a["nis"][k] = u.nu, u.S, u.K, u.nis
            updated[k] = True
        a["x_post"][k], a["P_post"][k] = r.x_post, r.P_post
        a["q_a"][k], a["window_nis"][k] = r.q_a, r.window_nis
        net[k], reproc[k] = r.manoeuvre_on, r.reprocessed
        frozen[k] = r.beta_frozen
        mode[k], flag[k] = r.mode.value, r.engine_flag
        skipped[k], pd_ok[k], status[k] = r.skipped_singular, r.pd_ok, r.status.value
    return FilterHistory(**{k: _frozen(v) for k, v in a.items()}, updated=_frozen(updated),
                         skipped_singular=_frozen(skipped), beta_frozen=_frozen(frozen), mode=_frozen(mode),
                         engine_flag=_frozen(flag), pd_ok=_frozen(pd_ok), manoeuvre_on=_frozen(net),
                         reprocessed=_frozen(reproc),
                         status=_frozen(status),
                         events=tuple(events), track_start=k0)


# the full scenario part:

@dataclass(frozen=True)
class ScenarioResult:
    """one full run: truth, every scan, the acquisition and the filter history (THEORY 8.1)."""

    truth: TruthResult  # ground truth: evaluation and measurement generation only
    plan: ScanPlan
    scans: tuple[Scan, ...]  # one per scan time, cued and tracked
    acquisition: Acquisition
    filter: FilterHistory | None  # None: NO ACQUISITION
    radar: Radar
    filter_cfg: FilterConfig


def run_scenario(vehicle_cfg: VehicleConfig | None = None,
                 scenario_cfg: ScenarioConfig | None = None,
                 radar_cfg: RadarConfig | None = None,
                 filter_cfg: FilterConfig | None = None,
                 atmosphere_cfg: AtmosphereConfig | None = None,
                 constants: PhysicsConstants | None = None, *,
                 truth: TruthResult | None = None,
                 ir_cfg: IRSensorConfig | None = None) -> ScenarioResult:
    """here I run truth, radar and EKF for one scenario. a precomputed truth can be passed in (Monte
    Carlo runs share it). the loop is strictly causal: at scan k the tracker predicts to t_k, the beam
    points at that prediction, the radar scans the truth with that pointing, and the tracker updates
    or coasts. truth enters only radar.scan(), the measurement generator.
    u_k = (p_{k|k-1} - p_radar) / |p_{k|k-1} - p_radar| - beam pointing from the tracker's prediction"""
    vehicle_cfg = vehicle_cfg or VehicleConfig()
    scenario_cfg = scenario_cfg or ScenarioConfig()
    radar_cfg = radar_cfg or RadarConfig()
    filter_cfg = filter_cfg or FilterConfig()
    atmosphere_cfg = atmosphere_cfg or AtmosphereConfig()
    constants = constants or PhysicsConstants()
    if truth is None:
        truth = simulate_truth(vehicle_cfg, scenario_cfg, atmosphere_cfg, constants)
    radar = Radar(radar_cfg, constants)
    plan = plan_scans(truth, radar, scenario_cfg, ir_cfg)
    acq = acquire(truth, radar, plan, radar_cfg, scenario_cfg)
    scans = list(acq.scans)
    history = None
    if acq.acquired:
        ir_latency = (ir_cfg or IRSensorConfig()).latency  # the specified flag latency L, known to the tracker
        tracker = Tracker(filter_cfg, radar, FilterModel(filter_cfg, constants, atmosphere_cfg), ir_latency,
                          scenario_cfg.time_tol)
        k0 = acq.first_detection.k
        records = {k0: tracker.initialise(float(plan.t[k0]), acq.first_detection.z, bool(plan.engine_flag[k0]))}
        for k in range(k0 + 1, len(plan.t)):
            t_k = float(plan.t[k])
            tracker.predict(t_k)
            u = tracker.pointing()  # from x_{k|k-1}, never from truth
            i = plan.truth_index[k]
            scan = radar.scan(k, t_k, truth.position[i], truth.velocity[i], u, plan.noise[k])
            scans.append(scan)
            records[k] = tracker.process(scan, bool(plan.engine_flag[k]))  # the IR flag of this scan
        history = _history(records, len(plan.t), radar.dim, tracker.events, k0)
    return ScenarioResult(truth, plan, tuple(scans), acq, history, radar, filter_cfg)
