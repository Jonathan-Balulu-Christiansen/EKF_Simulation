"""here I keep every number of the simulation in typed dataclasses (THEORY section 2).
every parameter gets its unit in brackets and a SOURCE / ESTIMATE / DESIGN CHOICE comment.
angles are stored in [rad] (converted from degrees right here), so the physics code only sees SI.
PhysicsConstants, AtmosphereConfig, VehicleConfig, ScenarioConfig, RadarConfig, IRSensorConfig,
FilterConfig and UIConfig."""

import math
from dataclasses import dataclass


# the physical constants part:

@dataclass(frozen=True)
class PhysicsConstants:
    """constants shared by several models. g0 appears in the US76 hydrostatics, in Isp -> mdot and
    in g(h), so it lives in one place."""

    g0: float = 9.80665  # standard gravity at sea level in [m/s^2]. SOURCE: CGPM 1901, exact by definition
    earth_radius: float = 6_371_000.0  # Re, mean Earth radius used in g(h) in [m]. SOURCE: IUGG mean Earth radius (THEORY 4.2)
    c: float = 299_792_458.0  # speed of light in vacuum in [m/s]. SOURCE: SI definition, exact


# the atmosphere part:

@dataclass(frozen=True)
class AtmosphereConfig:
    """the US Standard Atmosphere 1976 constants (THEORY 4.3). altitudes in this table are geopotential,
    in geopotential metres [m'], because that is how US76 defines its layers."""

    T0: float = 288.15  # sea-level temperature in [K]. SOURCE: US Standard Atmosphere 1976
    p0: float = 101_325.0  # sea-level pressure in [Pa]. SOURCE: US76
    R_air: float = 287.053  # specific gas constant of air R*/M0 = 8.31432/0.0289644 in [J/(kg K)]. SOURCE: US76
    gamma: float = 1.4  # ratio of specific heats of air [-]. SOURCE: US76
    r0: float = 6_356_766.0  # Earth radius of the geopotential conversion in [m]. SOURCE: US76
    layer_base_H: tuple[float, ...] = (
        0.0, 11_000.0, 20_000.0, 32_000.0, 47_000.0, 51_000.0, 71_000.0,
    )  # base geopotential altitude H_b of the 7 layers in [m']. SOURCE: US76 table 4
    layer_lapse_rate: tuple[float, ...] = (
        -6.5e-3, 0.0, 1.0e-3, 2.8e-3, 0.0, -2.8e-3, -2.0e-3,
    )  # temperature gradient L_b = dT/dH of the 7 layers in [K/m']. SOURCE: US76 table 4
    H_top: float = 84_852.0  # top of the 7-layer model, geopotential altitude in [m']. SOURCE: US76
    layer_names: tuple[str, ...] = (
        "troposphere", "stratosphere", "stratosphere", "stratosphere", "stratopause", "mesosphere", "mesosphere",
    )  # name of each of the 7 layers, for display only. SOURCE: US76 nomenclature (the 11-20 km' isothermal layer is the lower stratosphere)


# the vehicle part:

@dataclass(frozen=True)
class VehicleConfig:
    """the generic illustrative vehicle (THEORY 4.1, 4.4, 4.5). the values are round numbers chosen so
    that T/W > 1, the burn lasts about a minute and the flight shows boost, coast and re-entry.
    they are round, generic values, not the data of any specific system, and not tuned towards any
    performance figure. derived values (A, mdot, t_burn,
    T/W) are computed in vehicle.py, never stored here."""

    launch_mass: float = 6_000.0  # launch mass at dry_mass_delta = 0 in [kg]. DESIGN CHOICE: illustrative
    propellant_mass: float = 3_500.0  # m_prop, fixed by the slider rule in [kg]. DESIGN CHOICE: illustrative
    diameter: float = 0.90  # body diameter d, sets the reference area in [m]. DESIGN CHOICE: illustrative
    length: float = 11.0  # body length in [m], display only, unused in 3-DOF physics. DESIGN CHOICE: illustrative
    thrust_sl: float = 120_000.0  # T_sl, thrust at sea level in [N]. DESIGN CHOICE: illustrative
    isp_sl: float = 220.0  # Isp_sl, specific impulse at sea level in [s]. DESIGN CHOICE: illustrative
    nozzle_exit_area: float = 0.10  # A_e, nozzle exit area in [m^2]. ESTIMATE: illustrative

    dry_mass_delta: float = 0.0  # mass slider, added to the dry mass in [kg]. DESIGN CHOICE: baseline
    dry_mass_delta_min: float = -1_000.0  # lower slider bound in [kg]. DESIGN CHOICE: THEORY 4.1
    dry_mass_delta_max: float = 3_000.0  # upper slider bound in [kg]. DESIGN CHOICE: THEORY 4.1
    min_thrust_to_weight: float = 1.05  # initial T/W at or below this is refused [-]. DESIGN CHOICE: THEORY 4.1

    t_vertical: float = 5.0  # t_vert, vertical rise before the pitch-over in [s]. DESIGN CHOICE: fixed, not tuned
    pitch_rate: float = math.radians(1.0)  # decrease rate of the pitch angle in [rad/s] (1 deg/s). DESIGN CHOICE: fixed, not tuned
    pitch_final: float = math.radians(45.0)  # pitch angle held until burnout in [rad] (45 deg). DESIGN CHOICE: fixed, not tuned

    # ESTIMATE: textbook shape of a slender body (flat subsonic, sharp transonic rise with the peak
    # near Mach 1.1, slow supersonic decline). illustrative, not measured on any vehicle.
    cd_mach: tuple[float, ...] = (
        0.0, 0.6, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0,
    )  # Mach numbers of the Cd table points [-]. ESTIMATE
    cd_values: tuple[float, ...] = (
        0.20, 0.20, 0.22, 0.28, 0.40, 0.45, 0.43, 0.38, 0.32, 0.26, 0.23, 0.21, 0.20,
    )  # drag coefficient Cd at those Mach numbers, held constant beyond both ends [-]. ESTIMATE


# the scenario part:

@dataclass(frozen=True)
class ScenarioConfig:
    """geometry and integration settings of one flight (THEORY 3, 4.6, 9). positions are in the radar's
    flat-Earth ENU frame (x = East, y = North, z = Up), origin at the radar's ground position."""

    launch_east: float = -150_000.0  # launch-site East coordinate in [m]. DESIGN CHOICE: THEORY 9
    launch_north: float = -60_000.0  # launch-site North coordinate in [m]. DESIGN CHOICE: THEORY 9
    launch_azimuth: float = math.radians(90.0)  # psi, compass azimuth of the launch plane, clockwise from North, in [rad] (090 deg). DESIGN CHOICE: THEORY 9

    dt_truth: float = 0.01  # fixed RK4 step of the truth simulator in [s]. DESIGN CHOICE: THEORY 4.6
    t_max: float = 3_600.0  # guard: the truth run stops here if no impact happened, in [s]. DESIGN CHOICE
    time_tol: float = 1e-9  # two times closer than this are one instant (event/grid merge, impact bisection, scan lookup) in [s]. DESIGN CHOICE
    seed: int = 0  # random seed of the measurement noise [-]. DESIGN CHOICE


# the radar part:

@dataclass(frozen=True)
class RadarConfig:
    """a generic X-band tracking radar (THEORY 5.1). these are representative values, not those of any
    specific radar. sigma_rdot is not a parameter: it is derived from f_c and T_dwell in radar.py."""

    east: float = 0.0  # radar ground position E in [m]. DESIGN CHOICE: THEORY 9 (ENU origin)
    north: float = 0.0  # radar ground position N in [m]. DESIGN CHOICE: THEORY 9 (ENU origin)
    antenna_height: float = 10.0  # height of the antenna phase centre above ground in [m]. DESIGN CHOICE: THEORY 5.1
    carrier_frequency: float = 10.0e9  # f_c in [Hz] (X-band). DESIGN CHOICE: representative value, THEORY 5.1
    update_interval: float = 0.1  # T_update, time between scans in [s] (10 Hz). DESIGN CHOICE: representative value, THEORY 5.1
    dwell_time: float = 10.0e-3  # T_dwell, coherent dwell per scan in [s]. DESIGN CHOICE: representative value, THEORY 5.1
    pulse_width: float = 1.0e-6  # tau, transmitted pulse width in [s]. DESIGN CHOICE: representative value, THEORY 5.1
    beamwidth: float = math.radians(1.5)  # full 3 dB beamwidth in [rad] (1.5 deg). DESIGN CHOICE: representative value, THEORY 5.1
    sigma_range: float = 10.0  # sigma_r, range noise std in [m]. DESIGN CHOICE: THEORY 5.1
    sigma_angle: float = math.radians(0.1)  # sigma_az = sigma_el, angle noise std in [rad] (0.1 deg). DESIGN CHOICE: THEORY 5.1
    elevation_mask: float = math.radians(1.0)  # no measurement below this elevation in [rad] (1 deg). DESIGN CHOICE: THEORY 5.1
    doppler_enabled: bool = True  # measure range-rate (dim 4) or not (dim 3). DESIGN CHOICE: THEORY 7, 8.7
    rho_min: float = 1.0  # singularity guard: no update below this horizontal range in [m]. DESIGN CHOICE: THEORY 5.6



# the IR plume sensor part:

@dataclass(frozen=True)
class IRSensorConfig:
    """the IR plume sensor (THEORY 5.7): one engine on/off flag per radar scan, a cue only, no kinematics."""

    latency: float = 0.5  # reporting latency: the flag changes this long after the true ignition or burnout, in [s]. DESIGN CHOICE: THEORY 5.7


# the filter part:

@dataclass(frozen=True)
class FilterConfig:
    """the EKF settings (THEORY 6). the filter has two modes set only by the IR engine flag (THEORY 6.2):
    BOOST estimates a thrust acceleration a with ln beta frozen, BALLISTIC has a = 0 and learns ln beta.
    it uses the same ISA density model as the truth (known atmosphere); the remaining model mismatch
    is the constant beta against Cd(Mach) and the changing mass."""

    propagation_substep: float = 0.025  # longest RK4 sub-step of the filter prediction in [s]. DESIGN CHOICE
    sigma_a_floor: float = 0.0  # sigma_floor, std of the accelerations the truth has and the filter does not model, in [m/s^2]. DESIGN CHOICE: derived below
    k_drag: float = 0.0  # k_drag, share of the drag that the filter's model cannot represent [-]. DESIGN CHOICE: derived below
    # derivation (THEORY 6.3). sigma_floor: in ballistic flight the truth and the filter share gravity g(h),
    # the US76 density, the flat Earth and still air, and the mass is constant after burnout, so no force is
    # left unmodelled; RK4 at 0.01 s (truth) and at 0.025 s sub-steps (filter) differ by about 1e-11 m/s^2
    # equivalent acceleration, and P stays positive definite through the updates without a floor. so 0.
    # k_drag: the drag mismatch left is Cd(Mach) against a constant beta, d ln beta/dt = -(d ln Cd/dM)(dM/dt).
    # ln beta enters the acceleration with gain a_D, so its random walk acts like an acceleration random walk
    # of intensity a_D^2 q_beta. seen through the cross-range channel (r = (R sigma_angle)^2 T) the steady-state
    # bandwidth is w = (q_beta a_D^2 / r)^(1/6), the lag of ln beta behind the Cd(Mach) ramp J is J / w and its
    # own spread is sqrt(q_beta / (2 w)) (the same theory as q_thrust). where lag <= spread the mismatch is
    # inside P(ln beta) and reaches the velocity through A[v, ln beta]; only lag - spread would be unmodelled.
    # default re-entry (Mach 4.39 -> 2.98, Cd 0.222-0.261, |d ln beta/dt| <= 0.029/s, a_D 0.5-41 m/s^2,
    # R 73-81 km), pointwise along the trajectory: lag/spread <= 0.33, the q_beta needed is at most 0.0043
    # (0.0282 is used). the unrepresented share is 0, so k_drag = 0 (lag/spread <= 0.50 for every mass of the sweep).
    q_beta: float = 0.0282  # q_beta, random-walk spectral density of ln beta in [1/s]. DESIGN CHOICE: derived below (THEORY 6.3 margin rule)
    # ln beta must follow the Cd(Mach) ramp d ln beta/dt = -(d ln Cd/dM)(dM/dt) of this target class. its random walk
    # acts like an acceleration random walk of intensity a_D^2 q_beta; through the cross-range channel
    # (r = (R sigma_angle)^2 T) the steady-state bandwidth is w = (q_beta a_D^2 / r)^(1/6), the lag J/w and the
    # spread sqrt(q_beta/(2w)) (the q_thrust theory). lag <= spread needs q_beta >= (2 J^2)^(6/7) (r/a_D^2)^(1/7).
    # over the class (dry_mass_delta -1,000 ... +3,000 kg, every ballistic scan with drag >= 0.05 g) the largest
    # need is 0.0086 /s (-1,000 kg, end of re-entry: a_D = 94 m/s^2, |d ln beta/dt| = 0.040/s, R = 317 km; the
    # default scenario alone needs 0.0043). margin: a lag of one sigma doubles the mean-square error of ln beta
    # in that direction; I require lag <= sigma/2, so the bias adds at most 25 % to it. lag/spread scales as
    # q^(-7/12), so this needs q x 2^(12/7) = x3.28: q_beta = 3.281 * 0.00858 = 0.0282 /s (not rounded down, so the
    # margin holds; a test checks lag <= spread/2 over the config range).
    boost_mode: bool = True  # BOOST/BALLISTIC modes from the IR flag; off = BALLISTIC model throughout (THEORY 6.6 toggle). DESIGN CHOICE: THEORY 6.2, 6.6
    q_thrust: float = 135.0  # q_thrust, random-walk spectral density of the thrust acceleration a per axis in [m^2/s^5]. DESIGN CHOICE: derived below (THEORY 6.3 margin rule)
    # in BOOST, a must follow everything the filter does not model: the thrust acceleration T/m, which ramps as
    # propellant burns and turns with the pitch-over, plus the drag mismatch at the frozen prior beta. a random walk
    # follows such a persistent ramp only if its own spread covers the lag the ramp causes (the per-scan rule
    # q = J^2 T holds only for a zero-mean jerk). for a random-walk acceleration seen through a position channel
    # with noise density r = sigma_m^2 T, the steady-state bandwidth is w = (q/r)^(1/6) (third-order Kalman-Bucy
    # filter), the lag J/w and the spread sqrt(q/(2w)); lag <= spread needs q >= (2 J^2)^(6/7) r^(1/7), with the
    # cross-range channel sigma_m = R sigma_angle as the weakest one. THEORY 6.3 margin rule: lag <= spread/2, so the
    # bias adds at most 25 % to the mean-square error; lag/spread scales as q^(-7/12), so q is multiplied by
    # 2^(12/7) = 3.28. over the config range (dry_mass_delta -1,000 ... +3,000 kg, every tracked boost scan before
    # burnout, pointwise) the largest need is at -1,000 kg just before burnout: |J| = 2.96 m/s^3 (T mdot / m^2 at
    # the 1,500 kg burnout mass; thrust alone 3.18 m/s^3) at R = 147 km, so q >= 41.0 m^2/s^5 for lag <= spread and
    # q_thrust = 3.28 * 41.0 = 134.5, rounded up to 135 m^2/s^5. the default vehicle alone would need 22.5.

    sigma_v_tangential: float = 1_000.0  # initial std of the unmeasured velocity components in [m/s]. DESIGN CHOICE: THEORY 6.5
    beta0: float = 5_000.0  # initial beta estimate in [kg/m^2]. DESIGN CHOICE: THEORY 6.5
    sigma_ln_beta0: float = 1.5  # initial std of ln beta [-]. DESIGN CHOICE: THEORY 6.5
    sigma_thrust0: float = 30.0  # initial std of each thrust-acceleration component when the track starts in BOOST in [m/s^2]. DESIGN CHOICE: THEORY 6.5

    adaptive_q: bool = True  # the safety net (two-level adjustable process noise) on/off (THEORY 6.6). DESIGN CHOICE
    nis_window: int = 10  # N_w, updates in the sliding NIS window [-]. DESIGN CHOICE
    sigma_manoeuvre: float = 86.8  # std of the additional white acceleration the safety net switches on, in [m/s^2]. DESIGN CHOICE: derived below
    # the net must cover the largest acceleration this target class can have that the filter does not model: the
    # full thrust acceleration at burnout (a burn the IR flag has not reported yet, or boost mode off). from the
    # vehicle config ranges (THEORY 4.1), the largest thrust is the vacuum thrust T_sl + p0 A_e = 120,000 + 101,325 *
    # 0.10 = 130,133 N and the smallest burnout mass is m_launch - m_prop + dry_mass_delta_min = 6,000 - 3,500 -
    # 1,000 = 1,500 kg, so a_max = 130,133 / 1,500 = 86.8 m/s^2 (8.8 g). the noise std is set to a_max itself, the
    # upper end of the usual range 0.5-1 a_max (Bar-Shalom et al.), and q_m = sigma_manoeuvre^2 dt like q_a.
    # a test recomputes a_max from VehicleConfig, so this stays consistent with the vehicle.
    p_fa_flight: float = 0.01  # probability of at least one false safety-net alarm in a whole nominal flight [-]. DESIGN CHOICE: THEORY 6.6 (1 %)
    nominal_updates: int = 2_574  # updates in a nominal flight: default scenario (THEORY 9), first detection at 22.2 s, last before the mask at 279.6 s, 10 Hz [-]. ESTIMATE
    # threshold derivation (THEORY 6.6, Bonferroni): the window test runs after every update, so a nominal flight holds
    # N = nominal_updates = 2,574 tests. the probability of at least one false alarm is at most the sum of the
    # per-test probabilities, whatever the overlap between consecutive windows, so alpha = p_fa_flight / N =
    # 0.01 / 2,574 = 3.9e-6 per test keeps a nominal flight within p_fa_flight. the net switches on when the window
    # mean exceeds chi2^-1(1 - alpha, n m) / n (n = updates in the window, m = measurement dimension), 9.32 for a
    # full window of 10 four-dimensional updates. iid chi2(4) NIS fed through the implemented test (a test repeats
    # this) fire in 0.59 % of nominal flights (200,000 simulated flights): below the budget, because overlapping
    # windows are positively correlated. a count of only N / N_w = 257 independent windows would assume that a window
    # above the threshold stays above for about N_w updates; for chi2 terms the excess is spread over the window
    # and an exceedance lasts only a few updates, so that count is too low and gives about 5 % per flight. the longest
    # flight of the config range (-1,000 kg, 3,806 updates) has the bound 3,806 alpha = 1.5 % and measures 0.85 %.
    reprocess_burnout: bool = True  # at the IR flag change, restore the filter at the last scan at or before t_flag - L - T and re-process in BALLISTIC (THEORY 6.2). DESIGN CHOICE: THEORY 6.2

    confidence: float = 0.95  # two-sided chi2 interval for the NIS window and the lock test [-]. DESIGN CHOICE: THEORY 6.6, 6.7
    lost_confidence: float = 0.999  # single-update NIS bound of the LOST test [-]. DESIGN CHOICE: THEORY 6.7
    lock_updates: int = 10  # M, consecutive good updates for LOCKED [-]. DESIGN CHOICE
    lock_sigma_pos: float = 300.0  # largest position std for LOCKED in [m]. DESIGN CHOICE
    lost_sigma_pos: float = 2_000.0  # largest position std before LOST, about the beam half-width at 150 km, in [m]. DESIGN CHOICE
    lost_updates: int = 10  # K, consecutive updates above the lost NIS bound for LOST [-]. DESIGN CHOICE


# the user interface part:

@dataclass(frozen=True)
class UIConfig:
    """display settings of the desktop UI (THEORY 8). nothing here reaches the physics, the radar or the
    filter. every exaggeration listed here is shown on screen with its factor (THEORY 0.1)."""

    frame_interval: float = 1.0 / 60.0  # period of the playback timer in [s] (about 60 Hz). DESIGN CHOICE: THEORY 8.1
    playback_speeds: tuple[float, ...] = (0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0)  # sim seconds per wall second [-]. DESIGN CHOICE: THEORY 8.1 (x0.1 ... x20)
    default_speed: float = 5.0  # playback speed after a run [-]. DESIGN CHOICE: a 280 s flight plays in about a minute
    ellipsoid_sigma: float = 3.0  # size of the covariance ellipsoid in standard deviations [-]. DESIGN CHOICE: THEORY 8.3
    ellipsoid_scales: tuple[float, ...] = (1.0, 10.0, 100.0)  # display magnification of the ellipsoid [-]. DESIGN CHOICE: THEORY 8.3
    default_ellipsoid_scale: float = 10.0  # magnification after start [-]. DESIGN CHOICE: a 50 m ellipsoid is invisible at 1:1 in a 200 km scene
    range_ring_spacing: float = 50_000.0  # distance between the range rings around the radar in [m]. DESIGN CHOICE: THEORY 8.3
    axis_length: float = 20_000.0  # length of the E/N/U axes drawn at the radar in [m]. DESIGN CHOICE
    grid_spacing: float = 10_000.0  # ground grid cell in [m]. DESIGN CHOICE
    marker_size: float = 9.0  # screen-space symbol of the true target in [logical px], not to scale. DESIGN CHOICE: THEORY 8.3
    estimate_marker_size: float = 14.0  # screen-space symbol of the estimate, drawn behind the truth symbol, in [logical px]. DESIGN CHOICE
    plume_marker_size: float = 24.0  # screen-space halo while the IR flag reports engine on in [logical px]. DESIGN CHOICE: THEORY 5.7
    measurement_size: float = 2.5  # screen-space dot of one converted measurement in [logical px]. DESIGN CHOICE
    site_marker_size: float = 8.0  # screen-space symbol of the radar and the launch site in [logical px]. DESIGN CHOICE
    trail_max_points: int = 4_000  # cap on the points of one trajectory trail (decimated beyond) [-]. DESIGN CHOICE: THEORY 8.1
    beam_segments: int = 48  # facets around the beam cone [-]. DESIGN CHOICE
    ellipsoid_rows: int = 18  # latitude bands of the ellipsoid mesh [-]. DESIGN CHOICE
    ellipsoid_cols: int = 32  # longitude segments of the ellipsoid mesh [-]. DESIGN CHOICE
    plot_window: float = 60.0  # width of the time window that follows the cursor in the plots in [s]. DESIGN CHOICE: one flight phase at a time, so each phase's error scale is readable
    plot_window_lead: float = 0.2  # share of that window kept free after the cursor [-]. DESIGN CHOICE: must be >= plot_window_step / plot_window, so the cursor stays inside
    plot_window_step: float = 10.0  # the window pages forward in steps of this length in [s]. DESIGN CHOICE: between steps only the cursor moves and the plots are not redrawn
    text_refresh_interval: float = 0.05  # shortest wall time between two refreshes of the equations text while playing in [s]. DESIGN CHOICE: 20 Hz, faster than numbers can be read
    pulse_factors: tuple[float, ...] = (1e3, 1e4, 1e5, 1e6)  # slow-motion factors of the pulse view [-]. DESIGN CHOICE: the default 1e5 is THEORY 8.5
    default_pulse_factor: float = 1e5  # time runs 1/100,000 as fast in the pulse view [-]. DESIGN CHOICE: THEORY 8.5
    pulse_marker_size: float = 12.0  # screen-space marker at the pulse centre in [logical px], not to scale. DESIGN CHOICE
    max_elevation_warning: float = math.radians(80.0)  # warn above this maximum target elevation in [rad] (80 deg). DESIGN CHOICE: THEORY 9
