# Theory: the model and the filter, step by step

This document explains what the simulator computes and why, in the order the computation runs: the truth
model of the flight, the radar measurement model, the Extended Kalman Filter, and the consistency tests
that check it. Section numbers match the references in the code comments (for example `(THEORY 6.3)` in
`ekf.py` points to section 6.3 below).

Notation follows the code: `x_{k|k-1}` is the prediction of the state at scan k from the data up to scan
k−1, `P_{k|k}` the covariance after the update at scan k, `^T` a transpose, `^-1` an inverse. Units are SI
everywhere inside the code (m, s, kg, rad); km, degrees and kN appear only on screen.

---

## 0. Principles

**0.1 Faithful visuals.** Every visual is a computed quantity. Every visual magnification is printed on
screen with its factor: the covariance ellipsoid magnification (×1 / ×10 / ×100), the screen-space markers
("symbols not to scale"), and the time factor of the pulse view ("TIME ×1/100,000").

**0.2 Truth is for evaluation only.** The ground truth of the flight is read by two parts of the code only:
the measurement generator (the radar, which must observe the real target) and the evaluation views and
metrics, which carry the tag "evaluation – uses truth". Nothing in the tracker reads it: not the beam
pointing, not the track status, not the noise adaptation, not the initialization.

**0.3 No inverse crime.** The truth simulator and the filter use different models on purpose (6.2). A
filter that shares the generating model exactly is tested only against itself.

**0.4 Units.** SI inside, display units only at display time, and every parameter in `config.py` carries its
unit and its origin: `SOURCE` (an external reference), `ESTIMATE` (an illustrative textbook shape) or
`DESIGN CHOICE`.

## 1. Scope

A generic, illustrative ballistic vehicle is flown by a 3-DOF point-mass simulator, observed by one ground
radar (range, azimuth, elevation, range-rate) and one infrared plume sensor (engine on/off), and tracked by
an EKF that also estimates the ballistic coefficient. The vehicle parameters are round numbers chosen only so
that the flight shows a boost, a coast and an atmospheric re-entry; they are round, generic values, not the
data of any specific system. Out of scope: other filters (UKF, IMM), clutter and false alarms, several radars, a round or rotating
Earth, 6-DOF dynamics.

## 2. Code map

| Module | Contents |
|---|---|
| `ekfsim/config.py` | every parameter, with unit and origin |
| `ekfsim/atmosphere.py` | US Standard Atmosphere 1976 (4.3) |
| `ekfsim/aero.py` | the Cd(Mach) table (4.4) |
| `ekfsim/vehicle.py` | thrust, mass flow, pitch program, truth equations of motion (4.1, 4.2, 4.5) |
| `ekfsim/integrators.py` | RK4 with exact events and ground contact (4.6) |
| `ekfsim/radar.py` | h(x), its Jacobian, noise, Doppler, beam, pulse timing, IR flag (5) |
| `ekfsim/ekf.py` | filter model, Jacobian, predict/update, modes, safety net, status (6) |
| `ekfsim/scenario.py` | one full run: truth, then radar and EKF interleaved |
| `ekfsim/metrics.py`, `ekfsim/montecarlo.py` | NIS, NEES, chi2 bounds, Monte Carlo (7) |
| `ekfsim_ui/` | the desktop UI (8); the only package that imports Qt |

## 3. Frames and conventions

A flat-Earth local **ENU** frame: x East, y North, z Up, origin at the radar's ground position; the antenna
is 10 m above it.

```
az = atan2(dE, dN)                       compass azimuth, clockwise from North, shown in [0, 360) deg
el = atan2(dU, sqrt(dE^2 + dN^2))        elevation, 0 at the horizon
```

Angle *differences* (innovations) are wrapped to [−π, π), so a difference across North is −2° and not 358°.

Flat Earth is a simplification: at ground distance d the real surface drops about d²/(2R) below the plane,
about 3 km at 200 km, which matters for the radar horizon of a real system.

---

## 4. Truth simulator

### 4.1 The vehicle

| Parameter | Value |
|---|---|
| launch mass (slider at 0) | 6,000 kg |
| propellant | 3,500 kg (fixed) |
| diameter | 0.90 m, so A = π (d/2)² = 0.636 m² |
| sea-level thrust T_sl | 120 kN |
| sea-level specific impulse Isp | 220 s |
| nozzle exit area A_e | 0.10 m² (ESTIMATE) |

Derived in code, never typed in:

```
mdot   = T_sl / (Isp g0)            = 55.62 kg/s       mass flow
t_burn = m_prop / mdot              = 62.93 s          burn time
T/W    = T(0) / (m0 g0)             = 2.04             initial thrust-to-weight
```

A slider adds −1,000 … +3,000 kg to the dry (non-propellant) mass; propellant, thrust and Isp stay fixed,
so only the mass effect shows. A configuration with T/W ≤ 1.05 is refused with the reason.

### 4.2 Equations of motion (3-DOF point mass)

State `X = [p, v, m]`.

```
dp/dt = v
dv/dt = (T(h) u_thrust + D) / m + [0, 0, −g(h)]
dm/dt = −mdot while the engine burns, 0 after burnout
D     = −½ ρ(h) |v| v Cd(Mach) A                drag
g(h)  = g0 (Re / (Re + h))²                     inverse-square gravity, Re = 6,371 km
Mach  = |v| / a(h),  a = sqrt(γ R_air T(h))     speed of sound from the local temperature
T(h)  = T_sl + (p0 − p(h)) A_e                  thrust with the exit-pressure term (Sutton & Biblarz)
β     = m / (Cd A)                              truth ballistic coefficient
```

The truth β changes with time, because both m and Cd(Mach) change. This is deliberate: the filter assumes a
nearly constant β (6.2), so the model mismatch is real. No Coriolis or centrifugal terms (small for these
speeds and flight times); the air is still.

### 4.3 Atmosphere: US Standard Atmosphere 1976

Seven layers up to 84.852 km geopotential altitude, each with a constant temperature gradient L_b. Inside a
layer the hydrostatic equation `dp/dH = −ρ g0` with `p = ρ R T` integrates in closed form:

```
H   = r0 h / (r0 + h)                      geopotential altitude from geometric altitude h
T   = T_b + L_b (H − H_b)                  layer temperature
p   = p_b (T_b / T)^(g0 / (R L_b))         gradient layer (L_b ≠ 0)
p   = p_b exp(−g0 (H − H_b) / (R T_b))     isothermal layer (L_b = 0)
ρ   = p / (R T)
```

The base values T_b, p_b of every layer are integrated from the sea-level values, not typed in from a
table. Above the top the temperature is held, which gives an exponential continuation matched at the
boundary (ρ ≈ 7·10⁻⁶ kg/m³ there). Tests check ρ(0) = 1.2250, ρ(11 km) = 0.3648, ρ(20 km) = 0.0889 kg/m³.

The 3D view draws the layer boundaries that the flight passes through, labeled with the geometric altitude
and the density these equations give there (11.0 km, ρ 0.364 kg/m³; 20.1 km, 0.0880 kg/m³; …).

### 4.4 Drag coefficient Cd(Mach)

A piecewise-linear table with the textbook shape of a slender body (ESTIMATE): flat at 0.20 below Mach 0.6,
a sharp transonic rise to a peak of 0.45 at Mach 1.1, then a slow decline to 0.20 at Mach 6, held beyond both
ends.

### 4.5 Launch and pitch program

Vertical for 5 s, then the thrust direction pitches over at 1°/s in the vertical plane of the launch azimuth
until 45°, held until burnout; after burnout the flight is purely ballistic. These three values are fixed
and not tuned towards any performance figure.

### 4.6 Integration

Fixed-step RK4 at 0.01 s. The step is split *exactly* at the known discontinuities: the two pitch-program
corners (5 s, 50 s) and burnout (62.93 s), so the jump of thrust and mass flow is integrated exactly. Ground
contact (z = 0) is found by bisection on the step length, re-integrating the partial RK4 step (not a linear
interpolation). The Cd table points and the atmosphere layer boundaries are kinks, not events.

### 4.7 Verification of the truth simulator

- RK4 order on a smooth configuration (constant Cd, exponential atmosphere, no thrust): halving dt divides
  the end-point error by 16.4–16.9 (theory: 16).
- Full model: the end point changes by 8·10⁻⁶ m between dt = 0.01 s and 0.005 s.
- Energy: without thrust and drag, ½|v|² plus the potential of g(h) is conserved to 10⁻⁶ relative.
- Mass: the propellant consumed equals mdot · t_burn to 10⁻¹²; the mass is constant after burnout.

---

## 5. Radar and the IR sensor

### 5.1 Parameters (representative values, not a specific radar)

| Parameter | Value |
|---|---|
| carrier f_c | 10 GHz, so λ = c / f_c = 3.0 cm |
| scan interval | 0.1 s (10 Hz) |
| coherent dwell T_dwell | 10 ms |
| pulse width τ | 1 µs, so the pulse is c τ = 300 m long |
| 3 dB beamwidth | 1.5° |
| σ_r, σ_az = σ_el | 10 m, 0.1° |
| elevation mask | 1.0° |

### 5.2 Measurement function

With the line of sight `d = p − p_radar`:

```
r    = |d|                                      slant range
az   = atan2(dE, dN)                            compass azimuth
el   = atan2(dU, sqrt(dE² + dN²))               elevation
ṙ    = (d · v) / r                              range-rate, positive when opening
z    = h(x) + w,   w ~ N(0, R),   R = diag(σ_r², σ_az², σ_el², σ_ṙ²)
```

Its Jacobian, used by the filter, is analytic:

```
∂r/∂p  = dᵀ / r
∂az/∂p = [dN, −dE, 0] / ρ_h²                    ρ_h = horizontal range
∂el/∂p = [−dE dU, −dN dU, ρ_h²] / (r² ρ_h)
∂ṙ/∂p  = (v − ṙ d / r)ᵀ / r,   ∂ṙ/∂v = dᵀ / r   only the range-rate row depends on the velocity
```

### 5.3 Doppler noise from the radar parameters

```
f_d  = 2 v_r f_c / c,  v_r = −ṙ                 Doppler shift, closing positive
Δf   = 1 / T_dwell                              frequency resolution of one dwell
Δv   = λ Δf / 2 = λ / (2 T_dwell)               range-rate resolution
σ_ṙ  := Δv = 1.50 m/s                           (one resolution cell: conservative)
```

Range-rate is assumed unambiguous. For information the UI shows what that would cost: ṙ changes sign during
the flight, so unambiguous signed Doppler needs `PRF > 2 |f_d|max` = 85.0 kHz for this run, and at that PRF
the unambiguous range `c / (2 PRF)` is only 1.76 km, against target ranges of 73–162 km. That is the
classic range/Doppler ambiguity trade-off of pulse radar; resolving it is out of scope.

### 5.4 Beam, cue and detections

The beam is a cone of half-angle 0.75° around the pointing direction. A detection needs the target inside
the cone and above the elevation mask:

```
ψ = atan2(|d × u|, d · u) ≤ beamwidth / 2  and  el > el_mask      (atan2 stays accurate near 0, acos does not)
```

Once a track exists, the beam points at the filter's own prediction `x_{k|k-1}`, as a real tracking radar
does. Before that it is cued at the launch-site azimuth, at elevation `mask + beamwidth/2` (an assumed
early-warning cue; at ground level the whole beam would lie below the mask). A scan without a detection is a
miss: the filter predicts and does not update.

### 5.5 Pulse timing

Every detection stores its emission time `t_tx`, the two-way delay `2r/c` and the echo arrival
`t_rx = t_tx + 2r/c`. These, and only these, drive the slow-motion pulse view (8.5).

### 5.6 Singularity guard

Near the vertical the azimuth and the `1/ρ_h` terms of the Jacobian blow up. If the predicted horizontal
range is below 1 m the update is skipped and flagged; no random epsilon is added.

### 5.7 IR plume sensor

A separate sensor reports one engine on/off flag per scan, with a specified latency L = 0.5 s:

```
flag_k = (t_ignition ≤ t_k − L < t_burnout)
```

It carries no position or velocity information. It is a cue that sets the filter mode (6.2).

---

## 6. Extended Kalman Filter

### 6.1 State

```
x = [pE, pN, pU, vE, vN, vU, aE, aN, aU, ln β]       10 states
```

`a` is the unknown thrust acceleration, used only while the engine burns. The filter estimates `ln β`
rather than β: it keeps β positive and makes a parameter that spans two orders of magnitude well
conditioned (its uncertainty is relative).

### 6.2 Filter model and modes

```
dp/dt = v
dv/dt = −ρ(h) |v| v / (2 β) + [0, 0, −g(h)] + a
BOOST:      da/dt = 0 + random walk (q_thrust),   ln β frozen
BALLISTIC:  a = 0 (fixed, zero variance),         d(ln β)/dt = 0 + random walk (q_β)
```

The filter uses the same ISA density as the truth model (a known atmosphere). The deliberate mismatch is
elsewhere: a constant β against the truth's Cd(Mach) and changing mass, and an estimated acceleration
against the truth's thrust model.

**The mode comes only from the IR flag.** In BOOST, thrust and drag act along nearly the same line and the
radar sees only their sum, so β is unobservable. Its row of the gain is set to zero (a Schmidt *consider*
update): ln β is not learned, its variance keeps growing by `q_β Δt`, and its cross-covariances stay
correct.

**Burnout re-processing.** The flag reports burnout with the latency L, and it locates burnout only within
one scan interval T:

```
t_flag − L − T < t_burnout ≤ t_flag − L
```

For those scans the filter has integrated a thrust that no longer existed. At the flag change the tracker
restores its state at the last scan at or before `t_flag − L − T` (never after burnout), sets `a = 0`
(rows and columns of `a` in P zeroed), switches to BALLISTIC and re-processes the buffered scans. In the
default run it restores 62.9 s at the flag (63.5 s) and replays 6 scans. Restoring *after* burnout would
carry thrust that never happened; the BALLISTIC model could explain it only by more drag, and the EKF
linearization of `exp(−ln β)` would then collapse σ(ln β) onto a wrong value.

### 6.3 Discretization and process noise

The state is propagated with RK4 sub-steps (at most 25 ms). The continuous Jacobian `A = ∂f/∂x` is analytic,
including the altitude terms through `dρ/dh` and `dg/dh`:

```
∂v'/∂v      = −ρ/(2β) (|v| I + v vᵀ / |v|)
∂v'/∂p_U    = −(dρ/dh) |v| v / (2β) + [0, 0, 2g / (Re + h)]
∂v'/∂(ln β) = +ρ |v| v / (2β)
∂v'/∂a      = I (BOOST), 0 (BALLISTIC)
```

The transition matrix is composed over the same sub-steps:

```
Φ_i = I + A_i h + (A_i h)² / 2,     Φ_k = Φ_n … Φ_2 Φ_1
```

Process noise per axis is white acceleration, plus random walks on `a` and `ln β`:

```
Q_axis = q [[Δt³/3, Δt²/2], [Δt²/2, Δt]]      on (p_i, v_i)
Q_a    = q_thrust Δt                          on each a_i (BOOST)
Q_lnβ  = q_β Δt
```

**The noise levels are derived, not tuned.**

- *Kinematic floor and drag share: both 0.* In ballistic flight the truth and the filter share gravity, the
  density, the flat Earth and still air, and the mass is constant, so no force is left unmodeled (the
  integrator difference is about 10⁻¹¹ m/s²). The remaining drag mismatch, Cd(Mach) against a constant β,
  is carried by the ln β random walk (next point); adding a drag-proportional term on top would count it
  twice. A test keeps this justified if the table or q_β change.
- *One margin rule for every random walk.* A random walk follows a persistent ramp J only with a lag. For a
  random-walk acceleration seen through a position channel with noise density `r = σ_m² T` (the weakest one
  is cross-range, `σ_m = R σ_angle`), steady-state Kalman–Bucy theory gives

  ```
  w = (q / r)^(1/6)            bandwidth
  lag = J / w                  bias behind the ramp
  spread = sqrt(q / (2 w))     the filter's own sigma of that state
  lag ≤ spread / 2   ⇔   q ≥ 2^(12/7) (2 J²)^(6/7) r^(1/7)
  ```

  The margin `lag ≤ spread/2` lets the bias add at most 25 % to the mean-square error. It is applied over
  the whole slider range of the vehicle:
  - `q_thrust = 135 m²/s⁵`: the thrust ramp T·mdot/m² is steepest for the lightest vehicle just before
    burnout (2.96 m/s³ at 147 km range).
  - `q_β = 0.0282 /s`: ln β follows the Cd(Mach) ramp `d ln β/dt = −(d ln Cd/dM)(dM/dt)`; its random walk
    acts like an acceleration random walk of intensity `a_D² q_β`, so the same theory applies with r
    divided by the drag acceleration squared.

### 6.4 Update

```
y_k = z − h(x_{k|k-1})                         innovation, angle rows wrapped to [−π, π)
S_k = H_k P_{k|k-1} H_kᵀ + R                   innovation covariance
K_k = P_{k|k-1} H_kᵀ S_k⁻¹                     gain, from a linear solve (never an explicit inverse)
x_{k|k} = x_{k|k-1} + K_k y_k
P_{k|k} = (I − K_k H_k) P_{k|k-1} (I − K_k H_k)ᵀ + K_k R K_kᵀ      Joseph form, then symmetrized
```

The Joseph form keeps P symmetric and positive definite under rounding, and it stays exact for a gain that
is not optimal. That is what makes the frozen ln β row of the consider update possible. Every step checks
that P is positive definite (Cholesky) on the states that carry variance; in BALLISTIC `a` is fixed at zero,
so P is singular there by construction.

Tests: the analytic A and H against central finite differences (relative error below 10⁻⁵ at random
states), and a linear constant-velocity case in which the EKF reproduces the textbook Kalman filter to 10⁻¹⁰.

### 6.5 Initialization (from the first detection only)

```
p0   = p_radar + r [cos el sin az, cos el cos az, sin el]          inverse of h
P_pp = J diag(σ_r², σ_az², σ_el²) Jᵀ,  J = ∂p/∂(r, az, el)         linearized conversion of R
v0   = ṙ u_r,   P_vv = σ_ṙ² u_r u_rᵀ + σ_t² (I − u_r u_rᵀ)         radial part measured, σ_t = 1 km/s
ln β0 = ln 5,000 with σ = 1.5;  a0 = 0 with σ = 30 m/s² per axis (BOOST)
```

### 6.6 Safety net (two-level adjustable process noise)

With an explicit thrust model in BOOST the noise adaptation should rarely act; it is kept as a safety net
(Bar-Shalom et al.). When the window mean of the last n ≤ N_w = 10 NIS values exceeds a threshold, an
additional white acceleration with σ_m = 86.8 m/s² is switched on; it is switched off when the mean is
back. σ_m is the largest acceleration this class can have unmodeled: the vacuum thrust over the smallest
burnout mass.

The threshold comes from a false-alarm budget **per flight**, not per test. The test runs after every update,
so a nominal flight holds N = 2,574 tests; by the Bonferroni bound

```
α = p_fa / N = 0.01 / 2,574 = 3.9·10⁻⁶        per test
P(at least one false alarm in a flight) ≤ N α = 1 %
on while  ε̄ > χ²_{1−α}(n m) / n               9.32 for a full window
```

Fed with independent χ²(4) values through the implemented test, it fires in 0.59 % of nominal flights
(200,000 simulated flights): below the budget, because overlapping windows are positively correlated.

### 6.7 Track status (from the filter's own information only)

```
LOCKED    window NIS inside its 95 % interval and σ_pos < 300 m, for 10 consecutive updates
COASTING  the last scan gave no update
LOST      σ_pos > 2 km, or NIS above its 99.9 % bound for 10 consecutive updates
σ_pos = sqrt(largest eigenvalue of the position block of P)
```

---

## 7. Consistency and evaluation

```
NIS   ε_ν = νᵀ S⁻¹ ν ~ χ²(m)                   online, the filter's own check (m = 4)
NEES  ε = eᵀ P⁻¹ e ~ χ²(6),  e = x̂ − x          offline, on the 6 kinematic states (evaluation – uses truth)
N-run average:  N ε̄ ~ χ²(N n)  →  bounds χ²(N n) / N
```

ln β is left out of NEES because its truth varies with time and the filter models it as nearly constant;
its error is reported apart. The Monte Carlo runner repeats the same flight with N noise seeds and splits
the time inside, above (over-confident) and below (conservative) the N-run bounds by flight phase: boost,
coast, and re-entry (from the first descending scan with drag ≥ 0.05 g). The acceptance rule counts
over-confidence only (at most 10 % of the ballistic scans): a truth without process noise penalizes a
conservative filter by construction, so conservatism is reported but not counted as a failure.

---

## 8. Desktop UI

**8.1 Compute, then play back.** A run is computed once into an immutable result, in a separate process (a
Python worker thread would compete with the GUI thread for the interpreter lock), and then played back. The
playback clock advances by real elapsed time × speed, never by counting timer ticks.

**8.2 Layout.** The main screen shows the 3D view, a short readout (filter first), three core plots and the
basic controls; everything else is one click away (an Advanced section, a "More" plot tab, collapsed readout
groups, the equations drawer).

**8.3 3D view.** The scene is 1:1 in km. It shows the ground grid, range rings, the E/N/U axes, the truth
and estimate trajectories, the measurements (converted to ENU for display only), screen-space markers, the
3σ position ellipsoid (eigen-decomposition of the position block of P), the beam cone (gray on a miss), and
the ISA layer boundaries (4.3).

**8.4 Plots.** Position error ±3σ, NIS with its χ² band, β against truth, and under "More": velocity
error, innovations ±3√S_ii, truth kinematics and the Cd table. By default the time axis is a 60 s window
that pages with the cursor, so each flight phase is shown at its own error scale.

**8.5 Pulse view.** One pulse is shown at its true length c τ = 300 m, traveling at c, reflecting at the
target and returning. A wavefront emitted at offset s ∈ [0, τ] has traveled the path c (t − t_tx − s):

```
d = path              while path ≤ r          going out
d = 2r − path         for r < path ≤ 2r        coming back after the reflection
```

so near the target the pulse folds into an outgoing and a returning part. The timing is the stored delay
(5.5). Why slow motion, and why not a smaller scene? Even a 1,000 km round trip lasts 2r/c = 6.7 ms;
shrinking distances does not slow the pulse down. So time is slowed by a factor printed on screen (×1/100,000
by default), and every object is slowed by the same factor: the target is nearly frozen, moving about 1 m
during the round trip.

**8.6 Equations panel.** Each step of the computation, from the truth model to the consistency tests,
shows its equations next to the live numbers of the scan under the cursor. Each quantity type has one
fixed display unit. Large matrices are heat maps colored by a unit-free magnitude, because their entries
mix m, m/s, m/s² and ln β:

```
|ρ_ij| = |P_ij| / sqrt(P_ii P_jj)        covariances P, Q (correlation)
|K_ij| sqrt(S_jj) / σ_i                  gain: share of the prior σ of state i a 1σ innovation moves
|Φ_ij| σ_j / σ_i,   |H_ij| σ_j / √S_ii    transition and measurement Jacobians
```

**8.7–8.9** Controls, readout and visual design: one color per meaning (truth, estimate, covariance,
measurements, beam, modes; green/amber/red reserved for status), text contrast at least 4.5:1, bundled
fonts, sizes in font units so the layout follows the screen scaling.

## 9. Default scenario

Radar at the ENU origin; launch site at (E −150 km, N −60 km), launch azimuth 090°. The ground track runs
60 km south of the radar, so the target never passes overhead. For the default vehicle the flight reaches
69.7 km and lasts 281 s; the maximum elevation seen from the radar is 45°.
