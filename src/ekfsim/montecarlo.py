"""here I run many noise realisations of the same scenario and report accuracy and consistency
(THEORY 7): RMSE(t) of position and velocity, ANEES and ANIS with their N-run chi2 bounds, the share
of time inside, above and below the bounds per flight phase (boost, coast, re-entry), the lock-time
distribution, boost mode and safety net on/off (THEORY 6.6), how often the safety net still switches on,
the thrust-acceleration estimate, and the mass sweep of THEORY 7. all of it is evaluation and
uses truth. the truth flight is
the same for every run (it does not depend on noise); each run gets its own noise seed, and because
the beam follows each run's own filter, misses can differ between runs.
acceptance rule: ANEES above its upper bound (over-confidence) on at most 10 % of
the ballistic-phase scans. scans below the lower bound (conservative) are reported, not counted as a
failure, because a truth without process noise penalises conservatism by construction. the original
THEORY 7 criterion (ANEES and ANIS inside on >= 90 %) is still reported for comparison.
command line: python -m ekfsim.montecarlo --runs 100 --seed 0 [--workers N] [--variants a,b,...]
                 [--dry-mass-delta KG] [--q-beta Q] [--q-thrust Q] [--mass-sweep]
variants: boost_on_net_on (default filter), boost_off_net_on, boost_off_net_off, boost_on_net_off
outputs in results/ under the current working directory (or --out): mc_summary.json, mc_timeseries.csv,
mc_consistency.png, mc_rmse.png, mc_lock_times.png, mc_run_detail.png, mc_thrust.png; with --mass-sweep:
mass_sweep.json, mass_sweep.txt"""

import argparse
import csv
import dataclasses
import json
import math
import os
import pathlib
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from ekfsim.config import FilterConfig, IRSensorConfig, PhysicsConstants, ScenarioConfig, VehicleConfig
from ekfsim.ekf import ACC, BETA_ROW
from ekfsim.metrics import average_interval, nees, rmse
from ekfsim.scenario import ScanPlan, ScenarioResult, TruthResult, run_scenario, simulate_truth

REENTRY_DRAG_G = 0.05  # re-entry starts at the first descending scan with truth drag >= this many g0 [-]. DESIGN CHOICE: evaluation only
OVERCONFIDENT_MAX = 0.10  # largest share of ballistic scans with ANEES above its upper bound [-]. DESIGN CHOICE: the acceptance rule (THEORY 7)
PHASES = ("boost", "coast", "reentry")
MASS_SWEEP = (-1_000.0, 0.0, 1_000.0, 2_000.0, 3_000.0)  # dry_mass_delta values of the sweep in [kg]. DESIGN CHOICE: THEORY 7
_TRUTH = None  # the shared truth flight, built once per worker process


def _worker_init(dry_mass_delta: float = 0.0):
    global _TRUTH
    _TRUTH = simulate_truth(VehicleConfig(dry_mass_delta=dry_mass_delta))


# the flight phase part (evaluation - uses truth):

def flight_phases(truth: TruthResult, plan: ScanPlan, constants: PhysicsConstants | None = None,
                  reentry_drag_g: float = REENTRY_DRAG_G) -> np.ndarray:
    """here I label every scan as boost, coast or re-entry, from the truth (evaluation only). re-entry
    starts at the first scan where the target descends and the truth drag deceleration reaches
    reentry_drag_g g0; from there on it lasts until impact.
    a_D = |D| / m - truth drag deceleration in [m/s^2]
    re-entry from the first k with v_U < 0 and a_D >= reentry_drag_g g0 - start of re-entry"""
    g0 = (constants or PhysicsConstants()).g0
    idx = plan.truth_index
    a_drag = np.linalg.norm(truth.drag[idx], axis=1) / truth.mass[idx]  # a_D in [m/s^2]
    boost = np.asarray(truth.boost[idx])
    start = (~boost) & (truth.velocity[idx, 2] < 0.0) & (a_drag >= reentry_drag_g * g0)
    labels = np.where(boost, "boost", "coast").astype("<U7")
    if np.any(start):
        labels[int(np.argmax(start)):] = "reentry"
    return labels


# the single-run part:

def run_summary(res: ScenarioResult) -> dict:
    """the per-run series the Monte Carlo statistics need (evaluation - uses truth). NaN where the
    run has no track yet (NEES, errors) or no update (NIS).
    e_k = x_{k|k} - x_true(t_k) - estimation error at each scan"""
    truth, F, idx = res.truth, res.filter, res.plan.truth_index
    n_scans = len(res.plan.t)
    out = {"nees": np.full(n_scans, np.nan), "nis": np.full(n_scans, np.nan),
           "pos_err": np.full((n_scans, 3), np.nan), "vel_err": np.full((n_scans, 3), np.nan),
           "ln_beta_err": np.full(n_scans, np.nan), "sigma_ln_beta": np.full(n_scans, np.nan),
           "net_on": np.full(n_scans, np.nan), "thrust_err": np.full((n_scans, 3), np.nan),
           "lock_time": math.nan, "lost": False, "track_start": -1}
    if F is None:
        return out
    k0 = F.track_start
    x_true = np.hstack([truth.position[idx], truth.velocity[idx]])
    out["nees"][k0:] = nees(F.x_post[k0:], F.P_post[k0:], x_true[k0:])
    out["nis"] = np.array(F.nis)
    out["pos_err"][k0:] = F.x_post[k0:, 0:3] - x_true[k0:, 0:3]
    out["vel_err"][k0:] = F.x_post[k0:, 3:6] - x_true[k0:, 3:6]
    out["ln_beta_err"][k0:] = F.x_post[k0:, BETA_ROW] - np.log(truth.beta[idx][k0:])
    out["sigma_ln_beta"][k0:] = np.sqrt(F.P_post[k0:, BETA_ROW, BETA_ROW])
    out["net_on"][k0:] = F.manoeuvre_on[k0:]  # 1 where the manoeuvre noise was in Q_k
    boost_scans = F.mode == "BOOST"  # scans the filter spent in BOOST mode
    a_true = truth.thrust[idx] / truth.mass[idx, None]  # true thrust acceleration T/m in [m/s^2]
    out["thrust_err"][boost_scans] = F.x_post[boost_scans, ACC] - a_true[boost_scans]
    locks = [e.t for e in F.events if e.new.value == "LOCKED"]
    out["lock_time"] = locks[0] if locks else math.nan
    out["lost"] = any(e.new.value == "LOST" for e in F.events)
    out["track_start"] = k0
    return out


def _run(job: tuple[int, FilterConfig]) -> dict:
    seed, filter_cfg = job
    return run_summary(run_scenario(scenario_cfg=ScenarioConfig(seed=seed), filter_cfg=filter_cfg, truth=_TRUTH))


def run_batch(filter_cfg: FilterConfig, seeds, pool: ProcessPoolExecutor, workers: int) -> list[dict]:
    """all seeds of one filter configuration, in parallel worker processes."""
    jobs = [(s, filter_cfg) for s in seeds]
    return list(pool.map(_run, jobs, chunksize=max(1, len(jobs) // (4 * workers))))


# the statistics part:

def _shares(v, n, lo, hi, sel) -> dict | None:
    """share of the selected scans (with at least one run) inside, above and below the bounds."""
    ok = (n > 0) & sel
    if not np.any(ok):
        return None
    above = float(np.mean(v[ok] > hi[ok]))
    below = float(np.mean(v[ok] < lo[ok]))
    return {"inside": 1.0 - above - below, "above": above, "below": below, "scans": int(ok.sum())}


def aggregate(runs: list[dict], phases: np.ndarray, m: int, confidence: float = 0.95) -> dict:
    """ANEES, ANIS and RMSE per scan, with the N-run bounds, and per phase the share of scans inside,
    above (over-confident) and below (conservative) the bounds.
    ANEES_k = (1/N_k) sum_i NEES_i,k, bounds chi2^-1(., 6 N_k) / N_k - averaged NEES (6 states)
    ANIS_k = (1/M_k) sum_i NIS_i,k, bounds chi2^-1(., m M_k) / M_k - averaged NIS (runs with an update)
    RMSE_k = sqrt((1/N_k) sum_i |e_i,k|^2) - root mean square error over runs"""
    nees_all = np.array([r["nees"] for r in runs])
    nis_all = np.array([r["nis"] for r in runs])
    n_nees = np.sum(~np.isnan(nees_all), axis=0)  # N_k, runs with a track at scan k
    n_nis = np.sum(~np.isnan(nis_all), axis=0)  # M_k, runs with an update at scan k
    anees = np.nansum(nees_all, axis=0) / np.where(n_nees > 0, n_nees, np.nan)  # ANEES_k
    anis = np.nansum(nis_all, axis=0) / np.where(n_nis > 0, n_nis, np.nan)  # ANIS_k
    anees_lo, anees_hi = average_interval(n_nees, 6, confidence)  # NaN where no run has a track
    anis_lo, anis_hi = average_interval(n_nis, m, confidence)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        lnb = np.array([r["ln_beta_err"] for r in runs])
        mean_sigma_lnb = np.nanmean(np.array([r["sigma_ln_beta"] for r in runs]), axis=0)
    out = {"anees": anees, "anees_lo": anees_lo, "anees_hi": anees_hi, "n_nees": n_nees,
           "anis": anis, "anis_lo": anis_lo, "anis_hi": anis_hi, "n_nis": n_nis,
           "rmse_pos": rmse(np.array([r["pos_err"] for r in runs])),
           "rmse_vel": rmse(np.array([r["vel_err"] for r in runs])),
           "rms_ln_beta": rmse(lnb[..., None]), "mean_sigma_ln_beta": mean_sigma_lnb}
    selections = {p: phases == p for p in PHASES}
    selections["ballistic"] = phases != "boost"
    out["shares"] = {name: {p: _shares(out[name], n, lo, hi, sel) for p, sel in selections.items()}
                     for name, n, lo, hi in (("anees", n_nees, anees_lo, anees_hi),
                                             ("anis", n_nis, anis_lo, anis_hi))}
    return out


def lock_stats(runs: list[dict]) -> dict:
    lt = np.array([r["lock_time"] for r in runs])
    ok = lt[~np.isnan(lt)]
    return {"runs_locked": int(ok.size), "runs_lost": int(sum(r["lost"] for r in runs)),
            "lock_time_mean": float(ok.mean()) if ok.size else None,
            "lock_time_min": float(ok.min()) if ok.size else None,
            "lock_time_max": float(ok.max()) if ok.size else None,
            "lock_time_p50": float(np.percentile(ok, 50)) if ok.size else None}


def _nanstat(f, a) -> float | None:
    """a NaN-aware statistic of a series, None when the series has no value at all."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        v = float(f(a))
    return None if math.isnan(v) else v


def mode_summary(a: dict, runs: list[dict], phases: np.ndarray, filter_cfg: FilterConfig,
                 t: np.ndarray | None = None) -> dict:
    """the summary numbers of one configuration, with the old and the new acceptance verdicts."""
    bal = phases != "boost"
    boost = ~bal
    sh = a["shares"]
    old_ok = all(sh[x]["ballistic"] is not None and sh[x]["ballistic"]["inside"] >= 0.9 for x in ("anees", "anis"))
    over = sh["anees"]["ballistic"]["above"] if sh["anees"]["ballistic"] else None
    return {
        "filter_config": dataclasses.asdict(filter_cfg),
        "consistency_shares": sh,
        "acceptance_overconfident_max_10pct": bool(over is not None and over <= OVERCONFIDENT_MAX),
        "acceptance_original_inside_90pct": bool(old_ok),
        "mean_anees": {p: _nanstat(np.nanmean, a["anees"][phases == p]) for p in PHASES},
        "mean_anis": {p: _nanstat(np.nanmean, a["anis"][phases == p]) for p in PHASES},
        "rmse_pos_m": {"boost_max": _nanstat(np.nanmax, a["rmse_pos"][boost]),
                       "boost_median": _nanstat(np.nanmedian, a["rmse_pos"][boost]),
                       "ballistic_max": _nanstat(np.nanmax, a["rmse_pos"][bal]),
                       "ballistic_median": _nanstat(np.nanmedian, a["rmse_pos"][bal]),
                       "final": _nanstat(float, a["rmse_pos"][-1])},
        "rmse_vel_mps": {"boost_max": _nanstat(np.nanmax, a["rmse_vel"][boost]),
                         "boost_median": _nanstat(np.nanmedian, a["rmse_vel"][boost]),
                         "ballistic_max": _nanstat(np.nanmax, a["rmse_vel"][bal]),
                         "ballistic_median": _nanstat(np.nanmedian, a["rmse_vel"][bal]),
                         "final": _nanstat(float, a["rmse_vel"][-1])},
        "rms_ln_beta_err": {"reentry_start": _nanstat(float, a["rms_ln_beta"][np.argmax(phases == "reentry")]),
                            "final": _nanstat(float, a["rms_ln_beta"][-1])},
        "safety_net": net_stats(runs, phases, t),
        "thrust_rms_error_boost_mps2": _nanstat(float, rmse(np.array([r["thrust_err"] for r in runs]).reshape(-1, 1, 3))[0])
        if any(np.any(~np.isnan(r["thrust_err"])) for r in runs) else None,
        "lock": lock_stats(runs),
    }


def net_stats(runs: list[dict], phases: np.ndarray, t: np.ndarray | None = None) -> dict:
    """how often the safety net is on (THEORY 6.6): the share of tracked scans whose Q held the manoeuvre
    noise, per phase over all runs, the number of runs in which it switched on, and when it first did."""
    q = np.array([r["net_on"] for r in runs])  # 1 / 0 per run and scan, NaN before the track
    tracked = ~np.isnan(q)
    out = {}
    for p in PHASES:
        sel = tracked & (phases == p)[None, :]
        out[p] = float(np.mean(q[sel] > 0.5)) if np.any(sel) else None
    on = np.where(tracked, q, 0.0) > 0.5
    out["runs_with_net_on"] = int(np.sum(np.any(on, axis=1)))
    if t is not None and np.any(on):
        first = np.array([t[int(np.argmax(row))] for row in on if row.any()])  # first scan with the net on
        out["first_on_time_s"] = {"p10": float(np.percentile(first, 10)), "median": float(np.median(first)),
                                  "p90": float(np.percentile(first, 90))}
    return out


# the plot part:

PALETTE = {"ON": "#2a78d6", "OFF": "#eb6834", "THIRD": "#1baf7a", "INK": "#0b0b0b", "INK2": "#52514e",
           "MUTED": "#898781", "GRID": "#e1e0d9", "AXIS": "#c3c2b7", "SURFACE": "#fcfcfb", "SHADE": "#f0efec",
           "BAND": "#e1e0d9"}


def plot_style(plt):
    P = PALETTE
    plt.rcParams.update({
        "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 9, "axes.edgecolor": P["AXIS"],
        "axes.labelcolor": P["INK2"], "axes.titlecolor": P["INK"], "axes.titlesize": 10, "xtick.color": P["INK2"],
        "ytick.color": P["INK2"], "axes.grid": True, "grid.color": P["GRID"], "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": P["SURFACE"],
        "axes.facecolor": P["SURFACE"]})


def mark_phases(ax, t, phases):
    """dotted lines at burnout and at the start of re-entry, labelled (evaluation - uses truth)."""
    for name, label in (("coast", "burnout"), ("reentry", "re-entry")):
        if np.any(phases == name):
            t0 = t[int(np.argmax(phases == name))]
            ax.axvline(t0, color=PALETTE["MUTED"], lw=0.8, ls=":")
            ax.annotate(label, (t0, 1.0), xycoords=("data", "axes fraction"), xytext=(3, -10),
                        textcoords="offset points", color=PALETTE["MUTED"], fontsize=8)


def beta_panel(ax, res: ScenarioResult, color: str, title: str):
    """beta estimate +-3 sigma vs truth beta over time for one run (evaluation - uses truth); grey
    ticks mark the scans where ln beta was frozen (BOOST mode)."""
    P = PALETTE
    F, idx = res.filter, res.plan.truth_index
    k0 = F.track_start
    t = res.plan.t[k0:]
    lnb, sb = F.x_post[k0:, BETA_ROW], np.sqrt(F.P_post[k0:, BETA_ROW, BETA_ROW])
    ax.fill_between(t, np.exp(lnb - 3 * sb), np.exp(lnb + 3 * sb), color=P["BAND"], lw=0, label="+-3 sigma")
    ax.plot(t, np.exp(lnb), color=color, lw=1.6, label="estimate exp(ln beta)")
    ax.plot(t, res.truth.beta[idx][k0:], color=P["INK"], lw=1.2, ls="--", label="truth m/(Cd A)")
    frozen = F.beta_frozen[k0:]
    if np.any(frozen):
        ax.plot(t[frozen], np.full(frozen.sum(), 1.5e2), "|", color=P["MUTED"], ms=5, label="ln beta frozen (BOOST mode)")
    ax.set_yscale("log")
    ax.set_ylim(1e2, 1e9)
    ax.set(title=title, xlabel="Time t [s]", ylabel="beta [kg/m^2]")
    ax.legend(loc="upper center", frameon=True, framealpha=0.9, edgecolor="none", fontsize=8, ncol=2)


VARIANTS = {"boost_on_net_on": (True, True), "boost_off_net_on": (False, True),
            "boost_off_net_off": (False, False), "boost_on_net_off": (True, False)}  # (boost mode, safety net)
VARIANT_STYLE = {"boost_on_net_on": ("ON", "boost mode on, net on"), "boost_off_net_on": ("THIRD", "boost mode off, net on"),
                 "boost_off_net_off": ("OFF", "boost mode off, net off"), "boost_on_net_off": ("MUTED", "boost mode on, net off")}


def _plots(t, phases, agg, runs, detail: ScenarioResult | None, out_dir: pathlib.Path, n_runs: int, seed: int):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_style(plt)
    P = PALETTE
    colors = {v: P[VARIANT_STYLE[v][0]] for v in agg}
    labels = {v: VARIANT_STYLE[v][1] for v in agg}

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True)
    for ax, name in ((axes[0], "anees"), (axes[1], "anis")):
        for mode, a in agg.items():
            ax.plot(t, a[name], color=colors[mode], lw=1.6 if mode == "boost_on_net_on" else 1.2, label=labels[mode])
        ref = agg["boost_on_net_on"] if "boost_on_net_on" in agg else next(iter(agg.values()))
        ax.fill_between(t, ref[name + "_lo"], ref[name + "_hi"], color=P["BAND"], lw=0,
                        label=f"95 % bounds, {n_runs} runs")
        ax.set_yscale("log")
        mark_phases(ax, t, phases)
        ax.set(xlabel="Time t [s]", title=("ANEES, 6 kinematic states (evaluation - uses truth): above the band = "
                                           "over-confident, below = conservative" if name == "anees"
                                           else "ANIS (filter's own innovations)"))
        ax.legend(loc="upper right", frameon=False)
    fig.suptitle(f"Monte Carlo consistency, {n_runs} runs per mode", color=P["INK"])
    fig.savefig(out_dir / "mc_consistency.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), constrained_layout=True)
    for ax, name, unit in ((axes[0], "rmse_pos", "m"), (axes[1], "rmse_vel", "m/s")):
        for mode, a in agg.items():
            ax.plot(t, a[name], color=colors[mode], lw=1.6, label=labels[mode])
        ax.set_yscale("log")
        mark_phases(ax, t, phases)
        ax.set(xlabel="Time t [s]", title=f"{'Position' if name == 'rmse_pos' else 'Velocity'} RMSE [{unit}] "
                                          "(evaluation - uses truth)")
        ax.legend(loc="upper right", frameon=False)
    fig.savefig(out_dir / "mc_rmse.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    for mode, rr in runs.items():
        lt = np.array([r["lock_time"] for r in rr])
        lt = lt[~np.isnan(lt)]
        if lt.size:
            ax.hist(lt, bins=20, color=colors[mode], alpha=0.8, label=f"{labels[mode]} ({lt.size} runs locked)")
    ax.set(xlabel="Time of first LOCKED [s]", ylabel="Runs", title="Lock time distribution")
    ax.legend(frameon=False)
    fig.savefig(out_dir / "mc_lock_times.png", dpi=130)
    plt.close(fig)

    if detail is not None:
        _detail_plot(detail, out_dir / "mc_run_detail.png", plt, seed)
        if detail.filter_cfg.boost_mode:
            _thrust_plot(detail, out_dir / "mc_thrust.png", plt, seed)


def _thrust_plot(res: ScenarioResult, path, plt, seed: int):
    """estimated vs true thrust acceleration per axis for one run (evaluation - uses truth). besides the
    true T/m I draw the acceleration the filter's a actually has to supply: the true thrust plus the drag
    mismatch at the filter's frozen beta (truth drag minus the filter's drag at the true state)."""
    from ekfsim.config import AtmosphereConfig
    from ekfsim.ekf import FilterModel
    P = PALETTE
    truth, F, idx = res.truth, res.filter, res.plan.truth_index
    k0 = F.track_start
    boost = np.nonzero(F.mode == "BOOST")[0]
    k1 = int(boost[-1]) + 30 if boost.size else k0 + 30  # 3 s past the mode switch
    ks = np.arange(k0, min(k1, len(idx)))
    t = res.plan.t[ks]
    a_hat = F.x_post[ks][:, ACC]
    sig = np.sqrt(np.diagonal(F.P_post[ks], axis1=1, axis2=2))[:, ACC]
    a_true = truth.thrust[idx[ks]] / truth.mass[idx[ks], None]  # T/m in [m/s^2]
    model = FilterModel(res.filter_cfg, PhysicsConstants(), AtmosphereConfig())
    needed = np.full_like(a_true, np.nan)  # only meaningful in BOOST (a is fixed at 0 afterwards)
    for j, k in enumerate(ks):
        if F.mode[k] != "BOOST":
            continue
        i = idx[k]
        x_true = np.r_[truth.position[i], truth.velocity[i], 0.0, 0.0, 0.0, F.x_post[k, BETA_ROW]]
        drag_filter = model.f(x_true, False)[3:6] + np.array([0.0, 0.0, truth.gravity[i]])  # filter drag at the true state
        needed[j] = a_true[j] + truth.drag[i] / truth.mass[i] - drag_filter
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), constrained_layout=True, sharex=True)
    for j, name in enumerate(("East", "North", "Up")):
        ax = axes[j]
        ax.fill_between(t, a_hat[:, j] - 3 * sig[:, j], a_hat[:, j] + 3 * sig[:, j], color=P["BAND"], lw=0,
                        label="+-3 sigma from P")
        ax.plot(t, a_true[:, j], color=P["INK"], lw=1.2, ls="--", label="true thrust acceleration T/m")
        ax.plot(t, needed[:, j], color=P["OFF"], lw=1.2, ls=":", label="T/m + drag mismatch at the frozen beta")
        ax.plot(t, a_hat[:, j], color=P["ON"], lw=1.6, label="estimate a_hat")
        switch = res.plan.t[boost[-1] + 1] if boost.size and boost[-1] + 1 < len(res.plan.t) else None
        if switch is not None:
            ax.axvline(switch, color=P["MUTED"], lw=0.8, ls=":")
            ax.annotate("IR flag off: BALLISTIC", (switch, 1.0), xycoords=("data", "axes fraction"),
                        xytext=(3, -10), textcoords="offset points", color=P["MUTED"], fontsize=8)
        ax.set(title=f"Thrust acceleration, {name} [m/s^2] (evaluation - uses truth)")
    axes[0].legend(loc="upper left", frameon=True, framealpha=0.9, edgecolor="none", fontsize=8)
    axes[-1].set_xlabel("Time t [s]")
    fig.suptitle(f"BOOST mode: estimated vs true thrust acceleration, seed {seed}", color=P["INK"])
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _detail_plot(res: ScenarioResult, path, plt, seed: int):
    """one run in detail: errors with +-3 sigma, NIS, beta and the Q scale (evaluation - uses truth)."""
    from ekfsim.metrics import chi2_interval
    P = PALETTE
    truth, F, idx = res.truth, res.filter, res.plan.truth_index
    k0 = F.track_start
    t = res.plan.t[k0:]
    x, Pk = F.x_post[k0:], F.P_post[k0:]
    x_true = np.hstack([truth.position[idx], truth.velocity[idx]])[k0:]
    err = x[:, 0:6] - x_true
    sig = np.sqrt(np.diagonal(Pk, axis1=1, axis2=2))
    fig, axes = plt.subplots(4, 2, figsize=(12, 12), constrained_layout=True)
    names = ["E", "N", "U"]
    for j in range(3):
        for col, off, unit in ((0, 0, "m"), (1, 3, "m/s")):
            ax = axes[j, col]
            ax.fill_between(t, -3 * sig[:, j + off], 3 * sig[:, j + off], color=P["BAND"], lw=0,
                            label="+-3 sigma from P")
            ax.plot(t, err[:, j + off], color=P["ON"], lw=1.0, label="error")
            ax.set_yscale("symlog", linthresh=10.0 if off == 0 else 1.0)
            ax.set(title=f"{'Position' if off == 0 else 'Velocity'} error {names[j]} [{unit}] (evaluation - uses truth)",
                   xlabel="Time t [s]")
    axes[0, 0].legend(loc="upper right", frameon=False)
    ax = axes[3, 0]
    lo, hi = chi2_interval(res.radar.dim)
    upd = F.updated[k0:]
    ax.axhspan(lo, hi, color=P["BAND"], lw=0, label="95 % chi2 interval")
    ax.plot(t[upd], F.nis[k0:][upd], ".", ms=2, color=P["ON"], label="NIS")
    net = F.manoeuvre_on[k0:]
    if np.any(net):
        ax.plot(t[net], np.full(net.sum(), lo * 0.3), "|", color=P["MUTED"], ms=6, label="manoeuvre noise ON")
    ax.set_yscale("log")
    ax.set(title=f"NIS (dim {res.radar.dim}), filter's own, and the safety net", xlabel="Time t [s]")
    ax.legend(loc="upper right", frameon=True, framealpha=0.9, edgecolor="none", markerscale=3)
    beta_panel(axes[3, 1], res, P["ON"], "beta +-3 sigma vs truth (evaluation - uses truth)")
    fig.suptitle(f"Single run in detail, seed {seed}, boost mode {'on' if res.filter_cfg.boost_mode else 'off'}, "
                 f"safety net {'on' if res.filter_cfg.adaptive_q else 'off'}", color=P["INK"])
    fig.savefig(path, dpi=130)
    plt.close(fig)


# the command line part:

def run_variants(dry_mass_delta: float, cfgs: dict, seeds, workers: int):
    """all seeds of every filter configuration on the truth of one vehicle mass."""
    truth = simulate_truth(VehicleConfig(dry_mass_delta=dry_mass_delta))
    first = next(iter(cfgs.values()))
    detail = run_scenario(VehicleConfig(dry_mass_delta=dry_mass_delta), ScenarioConfig(seed=seeds[0]),
                          filter_cfg=first, truth=truth)
    phases = flight_phases(truth, detail.plan)
    with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init, initargs=(dry_mass_delta,)) as pool:
        runs = {v: run_batch(cfg, seeds, pool, workers) for v, cfg in cfgs.items()}
    agg = {v: aggregate(runs[v], phases, detail.radar.dim) for v in cfgs}
    return truth, detail, phases, runs, agg


def _share_text(d) -> str:
    return "n/a" if d is None else f"{100 * d['inside']:5.1f} / {100 * d['above']:5.1f} / {100 * d['below']:5.1f}"


def _filter_cfg(variant: str, q_beta: float | None = None, q_thrust: float | None = None) -> FilterConfig:
    boost, net = VARIANTS[variant]
    cfg = FilterConfig(boost_mode=boost, adaptive_q=net)
    if q_beta is not None:
        cfg = dataclasses.replace(cfg, q_beta=q_beta)
    if q_thrust is not None:
        cfg = dataclasses.replace(cfg, q_thrust=q_thrust)
    return cfg


def mass_sweep(args) -> int:
    """the mass sweep of THEORY 7: the same filter settings for every dry_mass_delta, no per-mass tuning."""
    cfg = _filter_cfg("boost_on_net_on", args.q_beta, args.q_thrust)
    seeds = list(range(args.seed, args.seed + args.runs))
    out, lines = {}, [f"Mass sweep (THEORY 7, information): {args.runs} runs per mass, boost mode on, safety net on, "
                      f"q_beta {cfg.q_beta}, q_thrust {cfg.q_thrust}, the same filter settings for every mass "
                      f"(evaluation - uses truth)",
                      "ANEES shares are inside / above (over-confident) / below (conservative) the 95 % bounds, in %",
                      f"pass: over-confident in at most {100 * OVERCONFIDENT_MAX:.0f} % of the ballistic scans "
                      f"(the acceptance rule of THEORY 7)", ""]
    head = (f"{'dry_mass_delta':>15}{'ballistic':>22}{'coast':>22}{'re-entry':>22}{'pass':>6}{'lost':>6}"
            f"{'pos RMSE med':>14}{'ln beta re-entry':>18}{'net on boost/coast/re':>24}")
    lines.append(head)
    for dm in MASS_SWEEP:
        truth, detail, phases, runs, agg = run_variants(dm, {"boost_on_net_on": cfg}, seeds, args.workers)
        s = mode_summary(agg["boost_on_net_on"], runs["boost_on_net_on"], phases, cfg, detail.plan.t)
        s["ir_switch_s"] = float(truth.t[truth.burnout_index] + IRSensorConfig().latency)
        s["apogee_km"] = float(truth.altitude.max() / 1e3)
        s["flight_time_s"] = float(truth.t[-1])
        s["reentry_start_s"] = float(detail.plan.t[int(np.argmax(phases == "reentry"))]) if np.any(phases == "reentry") else None
        out[f"{dm:+.0f}"] = s
        sh, trig = s["consistency_shares"]["anees"], s["safety_net"]
        fmt = lambda v: "n/a" if v is None else f"{100 * v:.0f}"  # noqa: E731
        lines.append(f"{dm:>+15,.0f}{_share_text(sh['ballistic']):>22}{_share_text(sh['coast']):>22}"
                     f"{_share_text(sh['reentry']):>22}{'PASS' if s['acceptance_overconfident_max_10pct'] else 'FAIL':>6}"
                     f"{s['lock']['runs_lost']:>6}{s['rmse_pos_m']['ballistic_median'] or float('nan'):>12.1f} m"
                     f"{s['rms_ln_beta_err']['reentry_start'] or float('nan'):>18.2f}"
                     f"{fmt(trig['boost']) + ' / ' + fmt(trig['coast']) + ' / ' + fmt(trig['reentry']) + ' %':>24}")
        print(lines[-1], flush=True)
    (args.out / "mass_sweep.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    (args.out / "mass_sweep.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ekfsim.montecarlo", description=__doc__.split("\n")[0])
    ap.add_argument("--runs", type=int, default=100, help="noise realisations per variant")
    ap.add_argument("--seed", type=int, default=0, help="seed of the first run; run i uses seed + i")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1, help="parallel processes")
    ap.add_argument("--variants", default="boost_on_net_on,boost_off_net_on,boost_off_net_off",
                    help="comma-separated filter variants: " + ", ".join(VARIANTS))
    ap.add_argument("--dry-mass-delta", type=float, default=0.0, help="vehicle mass slider in kg")
    ap.add_argument("--q-beta", type=float, default=None, help="override q_beta (effect studies)")
    ap.add_argument("--q-thrust", type=float, default=None, help="override q_thrust (effect studies)")
    ap.add_argument("--mass-sweep", action="store_true", help="run the THEORY 7 mass sweep instead")
    ap.add_argument("--out", type=pathlib.Path, default=None,
                    help="output folder (default: results/ in the current working directory)")
    args = ap.parse_args(argv)
    args.out = (args.out or pathlib.Path.cwd() / "results").resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.mass_sweep:
        return mass_sweep(args)
    variants = tuple(v.strip() for v in args.variants.split(",") if v.strip())
    cfgs = {v: _filter_cfg(v, args.q_beta, args.q_thrust) for v in variants}

    t_start = time.perf_counter()
    seeds = list(range(args.seed, args.seed + args.runs))
    truth, detail, phases, runs, agg = run_variants(args.dry_mass_delta, cfgs, seeds, args.workers)
    plan = detail.plan
    summary = {"runs_per_variant": args.runs, "first_seed": args.seed, "dry_mass_delta": args.dry_mass_delta,
               "q_beta": next(iter(cfgs.values())).q_beta, "q_thrust": next(iter(cfgs.values())).q_thrust,
               "measurement_dim": detail.radar.dim, "confidence": 0.95,
               "reentry_start_s": float(plan.t[int(np.argmax(phases == "reentry"))]),
               "burnout_s": float(truth.derived.burn_time),
               "elapsed_s": round(time.perf_counter() - t_start, 1),
               "ir_switch_s": float(truth.t[truth.burnout_index] + IRSensorConfig().latency),
               "variants": {v: mode_summary(agg[v], runs[v], phases, cfgs[v], plan.t) for v in variants}}
    (args.out / "mc_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    with open(args.out / "mc_timeseries.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        cols = ["anees", "anees_lo", "anees_hi", "anis", "anis_lo", "anis_hi", "rmse_pos", "rmse_vel",
                "rms_ln_beta", "mean_sigma_ln_beta"]
        w.writerow(["k", "t_s", "phase"] + [f"{c}_{v}" for v in variants for c in cols])
        for k in range(len(plan.t)):
            w.writerow([k, f"{plan.t[k]:.2f}", phases[k].upper()]
                       + [f"{agg[v][c][k]:.6g}" for v in variants for c in cols])

    _plots(plan.t, phases, agg, runs, detail if detail.filter is not None else None, args.out, args.runs, args.seed)
    print(json.dumps(summary, indent=2))
    print(f"wrote mc_summary.json, mc_timeseries.csv and PNGs to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
