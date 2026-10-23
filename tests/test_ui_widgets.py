"""the UI widgets on Qt's offscreen platform (no display needed). the controls must map to the
config exactly, the clock must follow wall time, every equation must render, and every view must accept
every kind of scan (search, initialisation, BOOST, miss, re-processed switch, BALLISTIC, last scan)."""

import dataclasses
import math

import numpy as np
import pytest

from ekfsim.config import FilterConfig, IRSensorConfig, RadarConfig, ScenarioConfig, UIConfig, VehicleConfig


def _close(a, b):
    """two configs are equal up to the rounding of the display units (deg <-> rad, ms <-> s)."""
    for f in dataclasses.fields(a):
        x, y = getattr(a, f.name), getattr(b, f.name)
        if isinstance(x, float):
            assert math.isclose(x, y, rel_tol=1e-12, abs_tol=1e-15), f.name
        else:
            assert x == y, f.name


def _special_scans(tl):
    """one scan index of each kind the views must handle."""
    k0 = tl.k0
    after = np.arange(tl.n_scans) > k0
    miss = np.flatnonzero(~np.asarray(tl.detected) & after)
    reproc = np.flatnonzero(np.asarray(tl.reprocessed) > 0)
    boost = np.flatnonzero((np.asarray(tl.mode) == "BOOST") & after)
    return {"search": 5, "init": k0, "boost": int(boost[10]), "miss": int(miss[0]), "switch": int(reproc[0]),
            "ballistic": tl.scan_index(150.0), "last": tl.n_scans - 1}


def test_controls_default_request_is_the_default_config(qapp):
    from ekfsim_ui.controls import ControlPanel
    req = ControlPanel(UIConfig()).request()
    for got, want in ((req.vehicle, VehicleConfig()), (req.scenario, ScenarioConfig()), (req.radar, RadarConfig()),
                      (req.filter, FilterConfig()), (req.ir, IRSensorConfig())):
        _close(got, want)


def test_mass_slider_shows_absolute_masses(qapp):
    from ekfsim_ui.controls import ControlPanel
    c = ControlPanel(UIConfig())
    assert (c.mass.minimum(), c.mass.maximum()) == (1_500, 5_500)  # dry mass in [kg]
    c.mass.setValue(5_500)
    assert c.request().vehicle.dry_mass_delta == 3_000.0
    assert "launch 9,000 kg" in c.mass_label.text() and "dry 5,500 kg" in c.mass_label.text()
    c.mass.setValue(1_500)
    assert c.request().vehicle.dry_mass_delta == -1_000.0 and "launch 5,000 kg" in c.mass_label.text()


def test_apply_state_and_validation(qapp):
    from ekfsim_ui.controls import ControlPanel
    c = ControlPanel(UIConfig())
    c.mark_applied(c.request())
    assert c.run_button.text() == "Run" and c.run_button.isEnabled()
    c.sigma_r.setValue(20.0)
    assert c.run_button.text() == "Apply"
    c.rate.setCurrentIndex(c.rate.findData(0.05))
    c.dwell.setValue(60.0)  # a 60 ms dwell is longer than the 0.05 s scan interval
    assert not c.run_button.isEnabled() and "dwell" in c.message.text()


def test_playback_clock_follows_wall_time_not_ticks(qapp):
    from ekfsim_ui.playback import PlaybackClock

    class FakeWall:
        ns = 0

        def start(self):
            pass

        def nsecsElapsed(self):
            return self.ns

    clock = PlaybackClock(UIConfig())
    clock.set_range(100.0)
    clock.set_speed(20.0)
    clock.play()
    clock._timer.stop()
    clock._wall = FakeWall()
    clock._last_ns = 0
    for ns in (10_000_000, 10_000_000, 50_000_000, 250_000_000):  # irregular ticks; one repeated
        clock._wall.ns = ns
        clock._tick()
    assert clock.t == pytest.approx(0.25 * 20.0)  # 0.25 s of wall time at x20, whatever the tick count
    clock._wall.ns = 10_000_000_000
    clock._tick()
    assert clock.t == 100.0 and not clock.playing  # stops at the end of the run
    clock.seek(-3.0)
    assert clock.t == 0.0


def test_every_equation_renders(qapp):
    from PySide6.QtGui import QPixmap
    from ekfsim_ui.equations_panel import EquationImage, EquationsPanel, render_equation
    panel = EquationsPanel()
    images = panel.findChildren(EquationImage)
    assert len(images) > 30
    for img in images:
        for dpr in (1.0, 1.5):
            png = render_equation(img.tex, dpr, "#E4E9EF", 10.0, 96.0)
            pm = QPixmap()
            assert pm.loadFromData(png, "PNG") and pm.width() > 0, img.tex
    a = QPixmap()
    a.loadFromData(render_equation(images[0].tex, 1.0, "#E4E9EF", 10.0, 96.0), "PNG")
    b = QPixmap()
    b.loadFromData(render_equation(images[0].tex, 1.5, "#E4E9EF", 10.0, 96.0), "PNG")
    assert b.width() / a.width() == pytest.approx(1.5, rel=0.08)  # rendered at the pixel ratio, not scaled


def test_equation_sections_take_every_kind_of_scan(qapp, default_timeline):
    from ekfsim_ui.equations_panel import EquationsPanel
    tl = default_timeline
    panel = EquationsPanel()
    panel.set_timeline(tl)
    for name, k in _special_scans(tl).items():
        i = tl.truth_index(float(tl.t_scan[k]))
        for j, sec in enumerate(panel.sections):
            panel.tabs.setCurrentIndex(j)
            panel.show_cursor(k, i)
            sec.update_cursor(tl, k, i)
        text = panel.message.text()
        if name == "miss":
            assert text.startswith("no update this scan")
        if name == "switch":
            assert "re-processed" in text
        if name == "search":
            assert "no track yet" in text


def test_readout_takes_every_kind_of_scan(qapp, default_timeline):
    from ekfsim_ui.readout import ReadoutPanel
    tl = default_timeline
    r = ReadoutPanel()
    for name, k in _special_scans(tl).items():
        t = float(tl.t_scan[k])
        r.update_cursor(tl, t, k, tl.truth_index(t), 5.0)
        if name == "search":
            assert r.chips["status"].text() == "SEARCH"
        if name == "ballistic":
            assert r.chips["status"].text() == "LOCKED" and r.v["nis"].text() != "—"
        if name == "boost":
            assert r.chips["mode"].text().startswith("BOOST")
    # a NIS outside its 95 % interval is drawn in the warning colour
    lo, hi = tl.nis_bounds
    k = int(np.flatnonzero(np.asarray(tl.nis) > hi)[0])
    r.update_cursor(tl, float(tl.t_scan[k]), k, tl.truth_index(float(tl.t_scan[k])), 1.0)
    from ekfsim_ui.theme import C
    assert C.COASTING in r.v["nis"].styleSheet()


def test_plots_share_one_time_range(qapp, default_timeline):
    from ekfsim_ui.plots import PlotPanel
    p = PlotPanel()
    p.set_timeline(default_timeline)
    p._time_plots[5].setXRange(50.0, 120.0, padding=0.0)
    for plot in p._time_plots:
        lo, hi = plot.vb.viewRange()[0]
        assert lo == pytest.approx(50.0) and hi == pytest.approx(120.0)
    seeks = []
    p.seekRequested.connect(seeks.append)
    p.set_cursor(77.0, 2.0)
    assert all(line.value() == 77.0 for line in p.currentWidget().cursors)


def test_main_window_shows_a_run(qapp, default_timeline):
    from ekfsim_ui.main_window import MainWindow
    w = MainWindow(autoplay=False)
    w._run_done(default_timeline, w._token)
    for t in (0.0, 10.0, 22.2, 50.0, 63.5, 150.0, default_timeline.t_end):
        w.clock.seek(t)
    assert w.sb_time.text().startswith("t ")
    w.step_scan(-1)
    assert w.clock.t == pytest.approx(float(default_timeline.t_scan[-2]))


def test_a_run_computes_in_the_child_process_without_blocking(qapp):
    """start_run() computes in the spawned process; the GUI thread keeps its event loop running."""
    from PySide6.QtCore import QElapsedTimer, QEventLoop, QTimer
    from ekfsim_ui.main_window import MainWindow
    w = MainWindow(autoplay=False)
    done, ticks = [], []
    w.runFinished.connect(done.append)
    beat = QTimer()
    beat.setInterval(10)
    clock = QElapsedTimer()
    clock.start()
    beat.timeout.connect(lambda: ticks.append(clock.elapsed()))
    beat.start()
    w.start_run()
    loop = QEventLoop()
    w.runFinished.connect(lambda _: loop.quit())
    QTimer.singleShot(90_000, loop.quit)
    loop.exec()
    beat.stop()
    w.close()
    assert done and done[0].acquired and done[0].n_scans == 2_813
    gaps = np.diff(ticks)
    assert len(gaps) > 50 and np.max(gaps) < 250  # the event loop never stalled while the run computed


# progressive disclosure, display units, pulse view, atmosphere

def test_heat_map_magnitudes_are_unit_free(qapp):
    from ekfsim_ui.equations_panel import correlation_magnitude, scaled_magnitude, unit_product
    rng = np.random.default_rng(3)
    A = rng.standard_normal((5, 5)) * np.array([1e3, 1.0, 1e-3, 10.0, 0.1])  # mixed scales
    P = A @ A.T
    N = correlation_magnitude(P)
    assert np.allclose(np.diag(N), 1.0) and np.all(N <= 1.0 + 1e-12)
    D = np.diag([1e-3, 1.0, 1e3, 7.0, 0.2])
    assert np.allclose(correlation_magnitude(D @ P @ D), N)  # the same colours in any units
    P[2, :] = P[:, 2] = 0.0
    assert np.all(np.isnan(correlation_magnitude(P)[2]))  # a fixed state has no colour
    K = rng.standard_normal((4, 3))
    s_state, s_meas = np.array([1.0, 2.0, 0.0, 4.0]), np.array([0.5, 1.0, 2.0])
    M = scaled_magnitude(K, s_state, s_meas)
    assert M[0, 2] == pytest.approx(min(abs(K[0, 2]) * 2.0 / 1.0, 1.0)) and np.all(np.isnan(M[2]))
    assert unit_product("m", "m/s") == "m²/s" and unit_product("m/s", "m", -1) == "1/s"
    assert unit_product("m", "rad", -1) == "m/rad" and unit_product("", "m/s²") == "m/s²"


def test_progressive_disclosure_defaults(qapp, default_timeline):
    """THEORY 8.2: the main screen shows the 3D view, a short readout, three core plots and the basic controls."""
    from ekfsim_ui.main_window import MainWindow
    w = MainWindow(autoplay=False)
    w._run_done(default_timeline, w._token)
    assert not w.equations.isVisibleTo(w)  # the drawer is closed
    tabs = [w.plots.tabText(i) for i in range(w.plots.count())]
    more = [w.plots.more.tabText(i) for i in range(w.plots.more.count())]
    assert tabs == ["Position", "NIS", "β", "More"] and more == ["Velocity", "Innovations", "Kinematics", "Aero"]
    assert not w.controls.advanced.button.isChecked()
    assert not w.readout.details.button.isChecked() and not w.readout.evaluation.button.isChecked()
    for key in ("status", "mode"):
        assert key in w.readout.chips
    for key in ("pos3s", "beta", "nis", "range", "alt", "speed"):
        assert key in w.readout.v
    w.clock.seek(150.0)
    assert w.readout.v["pos3s"].text() == fmt_q(3.0 * default_timeline.sigma_pos_max[1500])
    w.toggle_equations(True)
    assert w.equations.isVisibleTo(w) and w.transport.equations_button.isChecked()
    w.toggle_equations()
    assert not w.equations.isVisibleTo(w)


def fmt_q(x):
    from ekfsim_ui import fmt
    return fmt.q(x, "m")


def test_plot_window_pages_with_the_cursor(qapp, default_timeline):
    """THEORY 8.4: the time axis is a window that follows the cursor, so the ballistic error is readable."""
    from ekfsim_ui.plots import PlotPanel
    p = PlotPanel(UIConfig())
    p.resize(500, 700)
    p.show()  # pyqtgraph updates an auto-range when the view is shown
    p.set_timeline(default_timeline)
    W = UIConfig().plot_window
    for t in np.linspace(0.0, default_timeline.t_end, 400):
        lo, hi = p.window_range(float(t))
        assert lo - 1e-9 <= t <= hi + 1e-9 and hi - lo == pytest.approx(W)
    p.set_cursor(150.0, 3.0)
    qapp.processEvents()
    lo, hi = p.window_range(150.0)
    plot_e = p.pos_tab.plots[0]
    assert plot_e.vb.viewRange()[0] == pytest.approx([lo, hi])
    y_lo, y_hi = plot_e.vb.viewRange()[1]
    assert y_hi - y_lo < 200.0  # fits the ~15 m ballistic error and its band, not the boost-phase scale
    p.set_follow(False)
    lo, hi = plot_e.vb.viewRange()[0]
    assert lo <= 0.0 and hi >= default_timeline.t_end


def test_reset_restores_every_default(qapp):
    from ekfsim_ui.controls import ControlPanel
    c = ControlPanel(UIConfig())
    default = c.request()
    sent = []
    c.applyRequested.connect(sent.append)
    c.sigma_r.setValue(42.0)
    c.mass.setValue(4_000)
    c.boost.setChecked(False)
    c.rate.setCurrentIndex(0)
    c.reset_to_defaults()
    assert c.request() == default and sent and sent[-1] == default


def test_camera_frames_the_trajectory_and_the_legend_folds(qapp, default_timeline):
    """the auto-framed camera keeps every trajectory point inside the viewport margin; the legend folds to its
    header when it would cover more than a small corner (THEORY 8.3)."""
    from ekfsim_ui.view3d import FRAME_NDC, Scene3D
    s = Scene3D(UIConfig())
    s.resize(900, 600)
    s.show()
    qapp.processEvents()
    s.set_timeline(default_timeline)
    v = s.view
    w, h = v.width(), v.height()
    m = np.array((v.projectionMatrix((0, 0, w, h), (0, 0, w, h)) * v.viewMatrix()).data()).reshape(4, 4).T
    q = np.c_[default_timeline.truth_pos_km, np.ones(len(default_timeline.truth_pos_km))] @ m.T
    ndc = q[:, 0:2] / q[:, 3:4]
    assert np.all(q[:, 3] > 0) and np.all(np.abs(ndc[:, 0]) <= FRAME_NDC[0] + 1e-6)
    assert np.all(np.abs(ndc[:, 1]) <= FRAME_NDC[1] + 1e-6)
    assert s.legend.body.isVisibleTo(s)
    s.resize(320, 220)
    qapp.processEvents()
    s._place_overlays()
    assert not s.legend.body.isVisibleTo(s)  # folded in a small view
    s.labels.place()
    assert all(lab.parent() is v for _, lab in s.labels.items)  # plain child widgets, kept by QWidget.grab


def test_matrix_view_heat_map_is_compact(qapp):
    from ekfsim_ui.equations_panel import STATE_LABELS
    from ekfsim_ui.widgets import MatrixView
    m = MatrixView(STATE_LABELS, STATE_LABELS)
    m.set_data(np.eye(10), np.eye(10))
    heat = m.sizeHint()
    m.set_show_values(True)
    table = m.sizeHint()
    assert heat.width() < table.width() and heat.width() < 18 * __import__("ekfsim_ui.theme").theme.em()



def test_innovation_covariance_shows_sigma_and_correlation(qapp):
    """the S table: sigma_i in its display unit on the diagonal, the correlation off it; raw entries on request."""
    from ekfsim_ui.equations_panel import s_rows
    S = np.array([[100.0, 0.5e-3], [0.5e-3, (1.75e-3) ** 2]])
    S = np.block([[S, np.zeros((2, 2))], [np.zeros((2, 2)), np.diag([(1.75e-3) ** 2, 2.25])]])
    rows = s_rows(S, raw=False)
    assert rows[0][1][0] == "10.00 m" and rows[1][1][1] == "1.75 mrad" and rows[3][1][3] == "1.50 m/s"
    rho = 0.5e-3 / (10.0 * 1.75e-3)
    assert rows[0][1][1] == f"+{rho:.3f}" and rows[0][1][2] == "+0.000"
    raw = s_rows(S, raw=True)
    assert raw[0][1][0] == "100 m²" and raw[0][1][1].endswith("m·rad")


def test_pulse_view_slows_time_and_stops_at_the_echo(qapp, default_timeline):
    from ekfsim_ui.main_window import MainWindow
    from ekfsim_ui.playback import speed_text
    w = MainWindow(autoplay=False)
    w._run_done(default_timeline, w._token)
    w.clock.set_speed(5.0)
    w.clock.seek(150.04)
    w.toggle_pulse(True)
    assert w.clock.t == pytest.approx(150.0) and w.clock.speed == pytest.approx(1e-5)  # x1/100,000 at t_tx
    assert speed_text(w.clock.speed) == "×1/100,000" and w.scene.pulse_factor.text() == "TIME ×1/100,000"
    st = default_timeline.pulse_state(150.0)
    w.clock.play()
    w.clock.seek(st.done + 1e-9)
    assert not w.clock.playing  # nothing moves after the echo is back
    w.set_pulse_factor(1e3)
    assert w.clock.speed == pytest.approx(1e-3)
    w.toggle_pulse(False)
    assert w.clock.speed == 5.0 and not w.scene.pulse_box.isVisibleTo(w.scene)


def test_atmosphere_layers_are_labelled_from_the_model(qapp, default_timeline):
    from ekfsim_ui import fmt
    from ekfsim_ui.view3d import Scene3D
    s = Scene3D(UIConfig())
    s.set_timeline(default_timeline)
    texts = [lab.text() for _, lab in s.atmo_labels.items]
    top = float(np.max(default_timeline.truth_alt_km))
    shown = [b for b in default_timeline.atmosphere_boundaries if b.altitude / 1e3 < top]
    for b in shown:
        assert f"{fmt.disp(b.altitude, 'km')} · ρ {fmt.disp(b.density, 'kg/m³')}" in texts
    assert "troposphere" in texts and "stratosphere" in texts and len(s.atmo_items) == 2 * len(shown)
    s.set_layer("atmosphere", False)
    assert not any(it.visible() for it in s.atmo_items)


def test_the_transport_bar_never_overlaps_at_the_smallest_window(qapp, default_timeline):
    """THEORY 8.9: nothing clipped or overlapping at any size. at the window's minimum width the centre column
    still holds the whole transport bar, so its buttons and labels never sit on top of each other."""
    from PySide6.QtWidgets import QSlider, QWidget
    from ekfsim_ui.main_window import MainWindow
    w = MainWindow(autoplay=False)
    w._run_done(default_timeline, w._token)
    w.show()
    w.resize(w.minimumSizeHint().width(), w.height())
    qapp.processEvents()
    kids = [c for c in w.transport.findChildren(QWidget) if c.isVisible() and c.parent() is w.transport]
    for i, a in enumerate(kids):
        assert a.width() >= a.minimumSizeHint().width() or isinstance(a, QSlider), type(a).__name__
        for b in kids[i + 1:]:
            assert not a.geometry().intersects(b.geometry()), (type(a).__name__, type(b).__name__)
    w.close()


def test_atmosphere_labels_stay_clear_of_the_trajectory(qapp, default_timeline):
    """THEORY 8.3, 8.9: after a view reset the column of atmosphere labels sits where, from that camera, it covers
    no trajectory point and no other scene label, and stays inside the view (a cost of zero). before, it was
    pinned behind the launch site, where the rising arc crossed it."""
    from ekfsim_ui.main_window import MainWindow
    w = MainWindow(autoplay=False)
    w._run_done(default_timeline, w._token)
    w.show()
    w.resize(1600, 900)
    qapp.processEvents()
    w.scene.reset_view()
    costs = w.scene._atmo_costs
    chosen = w.scene.atmo_labels.items[0][0][:2]
    assert min(costs) == 0.0
    assert costs[w.scene.atmo_anchors.index(tuple(chosen))] == 0.0
    w.close()
