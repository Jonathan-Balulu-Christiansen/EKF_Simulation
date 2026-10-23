"""the 3D view (THEORY 8.3): the flat-Earth ENU scene in km around the radar, with the ground grid, range
rings, the E/N/U axes with the north arrow, the radar and launch-site markers, the truth and estimate
trajectories, the measurements (converted to ENU for display only), screen-space markers, the 3-sigma
position ellipsoid and the radar beam. one orbit camera, auto-framed after each run.
the ISA layer boundaries the flight passes through are faint planes labelled with their altitude and
density from the atmosphere module. the slow-motion pulse view (THEORY 8.5) shows the transmitted pulse at its
true length c tau, slowed by a factor printed on screen.
honesty rules (THEORY 0.1): the scene is 1:1 in km with no vertical exaggeration; markers are screen-space
symbols, not to scale, and say so; the ellipsoid magnification is printed on screen. the truth items are
evaluation views and are labelled as such. every item is created once and updated with setData or
setTransform; nothing is recreated per frame (THEORY 8.1). the scene labels are plain child widgets placed
at the projection of their 3D anchors, so a window grab keeps them.
OpenGL state: every item gets explicit glOptions. the translucent meshes (ellipsoid, beam) are drawn after
the opaque items with depth test on and depth writes off, so they never hide each other by draw order;
the screen-space markers and labels are drawn last without depth test, so they stay visible."""

import math

import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from OpenGL import GL
from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtGui import QMatrix4x4, QVector3D, QVector4D
from PySide6.QtWidgets import QFrame, QGridLayout, QHBoxLayout, QLabel, QToolButton, QVBoxLayout, QWidget

from ekfsim.config import UIConfig
from ekfsim_ui import fmt, theme
from ekfsim_ui.theme import C, rgba
from ekfsim_ui.timeline import KM, Timeline, ellipsoid_axes, rotation_from_z
from ekfsim_ui.widgets import Swatch, icon, label

_BLEND = ("glBlendFuncSeparate", (GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA, GL.GL_ONE,
                                  GL.GL_ONE_MINUS_SRC_ALPHA))  # keeps the framebuffer alpha at 1 for the widget compositor
GL_LINES = {GL.GL_DEPTH_TEST: True, GL.GL_BLEND: True, GL.GL_CULL_FACE: False, _BLEND[0]: _BLEND[1],
            "glDepthMask": (GL.GL_TRUE,)}  # antialiased lines, depth tested
GL_TRANSLUCENT = {GL.GL_DEPTH_TEST: True, GL.GL_BLEND: True, GL.GL_CULL_FACE: False, _BLEND[0]: _BLEND[1],
                  "glDepthMask": (GL.GL_FALSE,)}  # translucent meshes: tested, no depth writes
GL_OVERLAY = {GL.GL_DEPTH_TEST: False, GL.GL_BLEND: True, GL.GL_CULL_FACE: False, _BLEND[0]: _BLEND[1],
              "glDepthMask": (GL.GL_TRUE,)}  # screen-space markers and labels on top

FOV = 60.0  # horizontal field of view of the orbit camera in [deg]. DESIGN CHOICE: pyqtgraph default
CAMERA_ELEVATION = 22.0  # camera elevation after auto-framing in [deg]. DESIGN CHOICE
FRAME_NDC = (0.88, 0.72)  # the framed points stay inside this share of the half-width and half-height [-]. DESIGN
# CHOICE: room for the legend and factor at the top and the notes at the bottom
ATMO_LABEL_SPOTS = 11  # evenly spaced candidate spots of the atmosphere label column along each plane edge [-]. DESIGN CHOICE
CAMERA_SWING = 25.0  # camera azimuth offset from the perpendicular to the ground track, towards the launch end, in [deg]. DESIGN CHOICE


def _matrix(M: np.ndarray) -> QMatrix4x4:
    return QMatrix4x4(*np.asarray(M, dtype=float).ravel().tolist())


def _ring_segments(radius: float, n: int = 180) -> np.ndarray:
    """one ring of radius [km] around the origin as line-segment pairs (GL lines mode)."""
    a = np.linspace(0.0, 2.0 * math.pi, n + 1)
    pts = np.column_stack([radius * np.sin(a), radius * np.cos(a), np.zeros_like(a)])
    return np.repeat(pts, 2, axis=0)[1:-1]


def cone_mesh(half_angle: float, segments: int) -> gl.MeshData:
    """the beam cone with its apex at the origin along +z, length 1 and half-angle half_angle [rad]. a
    uniform scale by the length keeps the angle.
    rim radius = tan(half_angle) at z = 1 - cone geometry"""
    rim = math.tan(half_angle)
    th = np.linspace(0.0, 2.0 * math.pi, segments, endpoint=False)
    verts = np.vstack([[0.0, 0.0, 0.0], np.column_stack([rim * np.cos(th), rim * np.sin(th), np.ones(segments)])])
    i = np.arange(segments)
    faces = np.column_stack([np.zeros(segments, int), 1 + i, 1 + (i + 1) % segments])
    return gl.MeshData(vertexes=verts, faces=faces)


# the overlay part (labels, legend, factor, banner):

class _View(gl.GLViewWidget):
    """the orbit-camera GL view; it signals after each paint, so the anchored labels can follow the camera."""

    painted = Signal()

    def paintGL(self, *args, **kwargs):
        super().paintGL(*args, **kwargs)
        self.painted.emit()


class _Overlay(QFrame):
    def __init__(self, parent):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setObjectName("overlay")
        self.setStyleSheet(f"QFrame#overlay {{ background: {theme.css_rgba(C.SCENE, 0.82)}; "
                           f"border: 1px solid {C.HAIRLINE}; border-radius: 2px; }}")


class AnchoredLabels:
    """scene labels as ordinary child widgets of the GL view, placed at the screen projection of their 3D
    anchor. they replace pyqtgraph's GLTextItem, whose QPainter text inside the OpenGL frame is lost by
    QWidget.grab (the screenshots grab the application window only). Qt composites child
    widgets over the GL frame in both cases. a label behind the camera is hidden.
    x_clip = P V [p, 1] - clip coordinates of the anchor p
    px = (x/w + 1) W / 2, py = (1 - y/w) H / 2 - logical pixel position (w > 0: in front of the camera)"""

    def __init__(self, view: gl.GLViewWidget):
        self.view = view
        self.items: list[tuple[np.ndarray, QLabel]] = []
        self._key = None
        self.enabled = True

    def set_enabled(self, on: bool) -> None:
        self.enabled = on
        self._key = None
        if not on:
            for _, w in self.items:
                w.hide()
        else:
            self.place()

    def clear(self) -> None:
        for _, w in self.items:
            w.deleteLater()
        self.items = []
        self._key = None

    def add(self, pos, text: str, colour: str = C.INK_3, font_role: str = "caption") -> None:
        w = QLabel(text, self.view)
        w.setFont(theme.font(font_role))
        w.setStyleSheet(f"color: {colour}; background: transparent;")
        w.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        w.adjustSize()
        w.show()
        self.items.append((np.asarray(pos, float), w))
        self._key = None

    def place(self) -> None:
        """re-project the anchors when the camera or the view size changed."""
        v = self.view
        o = v.opts
        c = o["center"]
        key = (c.x(), c.y(), c.z(), o["distance"], o["elevation"], o["azimuth"], o["fov"], v.width(), v.height())
        if key == self._key or not self.items or not self.enabled:
            return
        self._key = key
        w, h = max(v.width(), 1), max(v.height(), 1)
        mvp = v.projectionMatrix((0, 0, w, h), (0, 0, w, h)) * v.viewMatrix()
        for p, lab in self.items:
            q = mvp.map(QVector4D(float(p[0]), float(p[1]), float(p[2]), 1.0))
            if q.w() <= 0.0:
                lab.hide()
                continue
            px = (q.x() / q.w() + 1.0) * 0.5 * w  # logical x of the anchor
            py = (1.0 - q.y() / q.w()) * 0.5 * h  # logical y of the anchor
            lab.move(round(px), round(py - lab.height()))  # text above-right of the anchor, like GLTextItem
            lab.setVisible(-lab.width() < px < w and 0 < py < h + lab.height())


class Legend(_Overlay):
    """the compact 3D legend: a header that collapses it to one line, and one short
    row per drawn quantity in the colours used everywhere else."""

    def __init__(self, parent, on_toggle):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        e = theme.em("caption")
        lay.setContentsMargins(round(0.35 * e), round(0.15 * e), round(0.55 * e), round(0.35 * e))
        lay.setSpacing(0)
        self.button = QToolButton()
        self.button.setObjectName("overlay_button")
        self.button.setText("Legend")
        self.button.setFont(theme.font("caption"))
        self.button.setCheckable(True)
        self.button.setChecked(True)
        self.button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.button.setToolTip("show or hide the legend (L)")
        lay.addWidget(self.button)
        self.body = QWidget()
        grid = QGridLayout(self.body)
        grid.setContentsMargins(round(0.25 * e), 0, 0, 0)
        grid.setHorizontalSpacing(round(0.4 * e))
        grid.setVerticalSpacing(round(0.05 * e))
        self.ellipsoid_text = None
        rows = (((C.TRUTH, "dot"), "truth (evaluation)", True), ((C.ESTIMATE, "dot"), "EKF estimate", False),
                ((C.MEASUREMENT, "small_dot"), "measurements (display only)", False),
                ((C.BEAM, "cone"), "beam, echo", False), ((C.BEAM_MISS, "cone"), "beam, no echo (miss)", False),
                ((C.COVARIANCE, "ellipse"), "3σ ellipsoid", False), ((C.BOOST, "halo"), "IR: engine on", False))
        for r, ((colour, kind), text, evaluation) in enumerate(rows):
            grid.addWidget(Swatch(colour, kind, C.TRUTH if kind == "halo" else None), r, 0)
            t = label(text, "secondary", "caption")
            if evaluation:
                t.setToolTip("evaluation – uses truth")
                t.setStyleSheet(f"color: {C.TRUTH};")
            grid.addWidget(t, r, 1)
            if kind == "ellipse":
                self.ellipsoid_text = t
        lay.addWidget(self.body)
        self.button.toggled.connect(on_toggle)
        self.set_open(True)

    def set_open(self, on: bool) -> None:
        self.button.setIcon(icon("chev_down" if on else "chev_right", C.INK_3))
        self.body.setVisible(on)
        self.adjustSize()


class Scene3D(QWidget):
    """the 3D view with its overlays."""

    ellipsoidScaleChanged = Signal(float)

    def __init__(self, ui_cfg: UIConfig):
        super().__init__()
        self.ui = ui_cfg
        self.tl: Timeline | None = None
        self.scale = ui_cfg.default_ellipsoid_scale  # ellipsoid display magnification [-]
        self.layers = {"truth": True, "estimate": True, "measurements": True, "beam": True, "ellipsoid": True,
                       "atmosphere": True}
        self.atmo_items: list = []
        self.pulse_on = False
        self._last_k = None
        self._last_det = None
        self._dpr = 1.0
        self._legend_wanted = True

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.view = _View()
        self.view.setBackgroundColor(pg.mkColor(C.SCENE))
        self.view.opts["fov"] = FOV
        lay.addWidget(self.view)
        self.labels = AnchoredLabels(self.view)
        self.atmo_labels = AnchoredLabels(self.view)
        self.atmo_anchors: list[tuple[float, float]] = []  # candidate (E, N) spots of the atmosphere label column
        self.view.painted.connect(self.labels.place)
        self.view.painted.connect(self.atmo_labels.place)
        self._build_items()
        self._build_overlays()
        self.view.installEventFilter(self)

    # the scene items

    def _build_items(self):
        v = self.view
        self.grid = gl.GLGridItem(color=pg.mkColor(C.GRID), glOptions=GL_LINES)
        self.rings = gl.GLLinePlotItem(mode="lines", color=rgba(C.HAIRLINE, 1.0), width=1.0, antialias=True,
                                       glOptions=GL_LINES)
        self.axes = gl.GLLinePlotItem(mode="lines", color=rgba(C.INK_3, 1.0), width=1.0, antialias=True,
                                      glOptions=GL_LINES)
        self.truth_line = gl.GLLinePlotItem(color=rgba(C.TRUTH, 0.95), width=1.5, antialias=True, glOptions=GL_LINES)
        self.est_line = gl.GLLinePlotItem(color=rgba(C.ESTIMATE, 0.95), width=1.5, antialias=True, glOptions=GL_LINES)
        self.meas = gl.GLScatterPlotItem(color=rgba(C.MEASUREMENT, 0.55), pxMode=True, glOptions=GL_TRANSLUCENT)  # no depth writes: the trails stay on top
        self.ellipsoid = gl.GLMeshItem(meshdata=gl.MeshData.sphere(self.ui.ellipsoid_rows, self.ui.ellipsoid_cols),
                                       smooth=True, color=rgba(C.COVARIANCE, 0.55), shader="balloon",
                                       glOptions=GL_TRANSLUCENT)
        self.beam = gl.GLMeshItem(meshdata=cone_mesh(math.radians(0.75), self.ui.beam_segments), smooth=True,
                                  color=rgba(C.BEAM, 0.16), shader="balloon", glOptions=GL_TRANSLUCENT)
        self.sites = gl.GLScatterPlotItem(color=rgba(C.INK_2, 1.0), pxMode=True, glOptions=GL_OVERLAY)
        self.halo = gl.GLScatterPlotItem(color=rgba(C.BOOST, 0.45), pxMode=True, glOptions=GL_OVERLAY)
        self.est_marker = gl.GLScatterPlotItem(color=rgba(C.ESTIMATE, 0.95), pxMode=True, glOptions=GL_OVERLAY)
        self.truth_marker = gl.GLScatterPlotItem(color=rgba(C.TRUTH, 1.0), pxMode=True, glOptions=GL_OVERLAY)
        self.pulse_line = gl.GLLinePlotItem(mode="lines", color=rgba(C.BEAM, 1.0), width=3.0, antialias=True,
                                            glOptions=GL_OVERLAY)
        self.pulse_marker = gl.GLScatterPlotItem(color=rgba(C.BEAM, 0.9), pxMode=True, glOptions=GL_OVERLAY)
        self.pulse_line.setVisible(False)
        self.pulse_marker.setVisible(False)
        for item, depth in ((self.grid, -10), (self.rings, -9), (self.axes, -8), (self.truth_line, 0),
                            (self.est_line, 1), (self.ellipsoid, 10), (self.beam, 11), (self.meas, -1),
                            (self.sites, 21), (self.halo, 22), (self.est_marker, 23), (self.truth_marker, 24),
                            (self.pulse_line, 25), (self.pulse_marker, 26)):
            item.setDepthValue(depth)
            v.addItem(item)

    def _build_overlays(self):
        e = theme.em("caption")
        self.legend = Legend(self.view, self._legend_toggled)
        self.factor = _Overlay(self.view)
        fl = QHBoxLayout(self.factor)
        fl.setContentsMargins(round(0.3 * e), 0, round(0.3 * e), 0)
        self.factor_button = QToolButton()
        self.factor_button.setObjectName("overlay_button")
        self.factor_button.setFont(theme.font("mono_medium"))
        self.factor_button.setStyleSheet(f"color: {C.COVARIANCE};")
        self.factor_button.setToolTip("3σ ellipsoid magnification, printed because the ellipsoid is not to "
                                      "scale; click to cycle ×1 / ×10 / ×100")
        self.factor_button.clicked.connect(self._cycle_scale)
        fl.addWidget(self.factor_button)
        self.banner = _Overlay(self.view)
        bl = QHBoxLayout(self.banner)
        bl.setContentsMargins(round(0.8 * e), round(0.5 * e), round(0.8 * e), round(0.5 * e))
        self.banner_label = label("", "value", "body_medium")
        self.banner_label.setWordWrap(True)
        bl.addWidget(self.banner_label)
        self.banner.hide()
        self.pulse_box = _Overlay(self.view)
        pl = QVBoxLayout(self.pulse_box)
        pl.setContentsMargins(round(0.9 * e), round(0.4 * e), round(0.9 * e), round(0.5 * e))
        pl.setSpacing(round(0.15 * e))
        self.pulse_factor = label("", "value", "display_large", Qt.AlignmentFlag.AlignHCenter)
        self.pulse_factor.setStyleSheet(f"color: {C.BEAM};")
        self.pulse_phase = label("", "value", "mono", Qt.AlignmentFlag.AlignHCenter)
        self.pulse_info = label("", "caption", "caption", Qt.AlignmentFlag.AlignHCenter)
        for w in (self.pulse_factor, self.pulse_phase, self.pulse_info):
            pl.addWidget(w)
        self.pulse_box.setToolTip(
            "why slow motion: light needs 2r/c = 6.7 ms for a 1,000 km round trip, far too fast to see. making the "
            "scene smaller does not slow the pulse down; only slowing time makes it visible. every object is slowed "
            "by the same factor, so the target is nearly frozen (it moves about 1 m during the round trip).")
        self.pulse_box.hide()
        self.note = label("symbols not to scale · scene 1:1 in km, no vertical exaggeration", "caption", "caption")
        self.note.setParent(self.view)
        self.hint = label("drag: orbit · wheel: zoom · Ctrl+drag: pan", "caption", "caption")
        self.hint.setParent(self.view)
        self._update_factor()

    def eventFilter(self, obj, event):
        if obj is self.view and event.type() in (QEvent.Type.Resize, QEvent.Type.Show):
            self._place_overlays()
        if obj is self.view and event.type() == QEvent.Type.DevicePixelRatioChange:
            self._apply_sizes()
        return super().eventFilter(obj, event)

    def _legend_toggled(self, on: bool) -> None:
        self._legend_wanted = on
        self._place_overlays()

    def toggle_legend(self) -> None:
        self.legend.button.toggle()

    def _place_overlays(self):
        """legend top-left, factor top-right, notes at the bottom. the legend folds to its header by itself
        when it would cover more than a small corner: a third of the width or half of the height."""
        w, h = self.view.width(), self.view.height()
        e = round(0.5 * theme.em())
        self.legend.set_open(self._legend_wanted)
        if self._legend_wanted and (self.legend.width() > w / 3 or self.legend.height() > h / 2):
            self.legend.set_open(False)
        self.legend.move(e, e)
        self.factor.adjustSize()
        self.factor.move(w - self.factor.width() - e, e)
        self.factor.setVisible(self.layers["ellipsoid"])
        self.pulse_box.adjustSize()
        self.pulse_box.move((w - self.pulse_box.width()) // 2, e)
        self.banner.setMaximumWidth(max(int(0.7 * w), 10))
        self.banner.adjustSize()
        self.banner.move((w - self.banner.width()) // 2, h - self.banner.height() - 3 * e)
        self.note.adjustSize()
        self.note.move(e, h - self.note.height() - e // 2)
        self.hint.adjustSize()
        self.hint.move(w - self.hint.width() - e, h - self.hint.height() - e // 2)
        self.hint.setVisible(w > self.note.width() + self.hint.width() + 4 * e)

    def _update_factor(self):
        self.factor_button.setText(f"3σ ×{self.scale:g}")
        if self.legend.ellipsoid_text is not None:
            self.legend.ellipsoid_text.setText(f"3σ ellipsoid, magnified ×{self.scale:g}")
        self._place_overlays()

    def _cycle_scale(self) -> None:
        scales = list(self.ui.ellipsoid_scales)
        i = scales.index(self.scale) if self.scale in scales else -1
        self.set_ellipsoid_scale(scales[(i + 1) % len(scales)])
        self.ellipsoidScaleChanged.emit(self.scale)

    def show_banner(self, text: str, colour: str) -> None:
        if not text:
            self.banner.hide()
            return
        self.banner_label.setText(text)
        self.banner_label.setStyleSheet(f"color: {colour};")
        self.banner.setStyleSheet(f"QFrame#overlay {{ background: {theme.css_rgba(C.SCENE, 0.9)}; "
                                  f"border: 1px solid {theme.css_rgba(colour, 0.7)}; border-radius: 2px; }}")
        self.banner.show()
        self._place_overlays()

    # the per-run part

    def set_timeline(self, tl: Timeline) -> None:
        """rebuild the scene geometry that depends on the run (extent, rings, beam angle) and frame it."""
        self.tl = tl
        self._last_k = self._last_det = None
        self.labels.clear()
        ui = self.ui
        radar, launch = tl.radar_km, tl.launch_km
        pts = np.vstack([tl.truth_pos_km, radar[None, :], launch[None, :]])
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        sp = ui.grid_spacing / KM  # grid cell in [km]
        g_lo = np.floor((lo[0:2] - 2 * sp) / sp) * sp
        g_hi = np.ceil((hi[0:2] + 2 * sp) / sp) * sp
        size = g_hi - g_lo
        self.grid.resetTransform()
        self.grid.setSize(x=float(size[0]), y=float(size[1]))
        self.grid.setSpacing(x=sp, y=sp)
        self.grid.translate(float((g_lo[0] + g_hi[0]) / 2), float((g_lo[1] + g_hi[1]) / 2), 0.0)

        ring = ui.range_ring_spacing / KM  # ring spacing in [km]
        r_max = float(np.max(np.hypot(*(np.asarray(tl.truth_pos_km[:, 0:2]) - radar[0:2]).T)))
        n_rings = max(1, math.ceil(r_max / ring))
        segs = [_ring_segments(ring * (j + 1)) for j in range(n_rings)]
        self.rings.setData(pos=np.vstack(segs) + np.array([radar[0], radar[1], 0.0]))
        label_az = tl.cue_az + math.pi  # ring labels on the bearing away from the launch site [rad]
        for j in range(n_rings):
            r = ring * (j + 1)
            self.labels.add([radar[0] + r * math.sin(label_az), radar[1] + r * math.cos(label_az), 0.0], f"{r:g} km")

        a = ui.axis_length / KM  # axis length in [km]
        o = np.array([radar[0], radar[1], 0.0])
        head = 0.18 * a
        axes = [o, o + [a, 0, 0], o, o + [0, a, 0], o, o + [0, 0, a],
                o + [0, a, 0], o + [-0.4 * head, a - head, 0], o + [0, a, 0], o + [0.4 * head, a - head, 0]]
        self.axes.setData(pos=np.array(axes))
        self.labels.add(o + [1.08 * a, 0, 0], "E")
        self.labels.add(o + [0, 1.08 * a, 0], "N (north)", C.INK_2, "body_medium")
        self.labels.add(o + [0, 0, 1.08 * a], "U")
        self.labels.add(radar + [0.5 * head, -1.2 * head, 0], "radar", C.INK_2)
        self.labels.add(launch + [0.5 * head, -1.2 * head, 0], "launch site", C.INK_2)
        self.sites.setData(pos=np.array([radar, launch]))
        self._build_atmosphere(tl, g_lo, g_hi)

        self.beam.setMeshData(meshdata=cone_mesh(0.5 * tl.request.radar.beamwidth, ui.beam_segments))
        self._apply_sizes()
        self.reset_view()
        if not tl.acquired:
            self.show_banner(f"NO ACQUISITION — {tl.acquisition_reason}", C.LOST)
        elif tl.elevation_warning:
            self.show_banner(f"warning: maximum target elevation {math.degrees(tl.max_elevation):.1f}° is above "
                             f"{math.degrees(ui.max_elevation_warning):.0f}° (azimuth poorly conditioned near "
                             "overhead) — evaluation – uses truth", C.COASTING)
        else:
            self.show_banner("", C.INK)

    def _build_atmosphere(self, tl: Timeline, g_lo, g_hi) -> None:
        """faint planes at the ISA layer boundaries below the highest point of the flight, labelled with the
        altitude and the density the atmosphere module computes there, and the layer names in one column. the
        column starts at the north-west corner of the flight; reset_view moves it to the candidate spot where it
        covers the least of the trajectory from the new camera (_place_atmosphere_labels)."""
        for it in self.atmo_items:
            self.view.removeItem(it)
        self.atmo_items = []
        self.atmo_labels.clear()
        top = float(np.max(tl.truth_alt_km))  # highest point of the flight in [km]
        x0, x1, y0, y1 = float(g_lo[0]), float(g_hi[0]), float(g_lo[1]), float(g_hi[1])
        framed = np.vstack([tl.truth_pos_km, tl.radar_km[None, :], tl.launch_km[None, :]])
        ax, ay = float(framed[:, 0].min()), float(framed[:, 1].max())  # label column: north-west of the framed scene
        x_apex = float(tl.truth_pos_km[int(np.argmax(tl.truth_alt_km)), 0])  # East of the apex in [km]
        # candidate (E, N) spots of the column on the planes, the north-west corner first (kept on a tie)
        xs = sorted({*np.linspace(x0, x1, ATMO_LABEL_SPOTS).tolist(), ax, x_apex, float(framed[:, 0].max())})
        self.atmo_anchors = [(ax, ay)] + [(x, y) for y in (y1, y0) for x in xs if (x, y) != (ax, ay)]
        shown = [b for b in tl.atmosphere_boundaries if b.altitude / KM < top]
        for b in shown:
            z = b.altitude / KM  # boundary altitude in [km]
            corners = np.array([[x0, y0, z], [x1, y0, z], [x1, y1, z], [x0, y1, z]])
            plane = gl.GLMeshItem(meshdata=gl.MeshData(vertexes=corners, faces=np.array([[0, 1, 2], [0, 2, 3]])),
                                  color=rgba(C.ATMOSPHERE, 0.035), smooth=False, glOptions=GL_TRANSLUCENT)
            edge = gl.GLLinePlotItem(pos=np.vstack([corners, corners[:1]]), color=rgba(C.ATMOSPHERE, 0.28), width=1.0,
                                     antialias=True, glOptions=GL_LINES)
            plane.setDepthValue(5)
            edge.setDepthValue(-7)
            for it in (plane, edge):
                self.view.addItem(it)
                self.atmo_items.append(it)
            self.atmo_labels.add([ax, ay, z], f"{fmt.disp(b.altitude, 'km')} · ρ {fmt.disp(b.density, 'kg/m³')}")
        # the layer names: contiguous layers with the same name form one band
        edges = [0.0] + [b.altitude for b in tl.atmosphere_boundaries] + [math.inf]
        names = [tl.atmosphere_boundaries[0].below] + [b.above for b in tl.atmosphere_boundaries]
        bands = []
        for lo, hi, name in zip(edges[:-1], edges[1:], names):
            if bands and bands[-1][2] == name:
                bands[-1][1] = hi
            else:
                bands.append([lo, hi, name])
        drawn = [b.altitude / KM for b in shown]
        for lo, hi, name in bands:
            lo_km, hi_km = lo / KM, min(hi / KM, top)
            if hi_km - lo_km > 6.0:  # a band thinner than 6 km in the flight gets no name (it would collide with its boundary labels)
                # the name sits in the middle of the widest gap between the boundary labels inside its band, so it
                # never crowds one of them (the stratosphere spans three boundary labels)
                cuts = [lo_km] + [z for z in drawn if lo_km < z < hi_km] + [hi_km]
                a, b = max(zip(cuts[:-1], cuts[1:]), key=lambda g: g[1] - g[0])
                self.atmo_labels.add([ax, ay, 0.5 * (a + b)], name, C.INK_2)
        self._show_atmosphere()

    def _show_atmosphere(self) -> None:
        on = self.layers["atmosphere"]
        for it in self.atmo_items:
            it.setVisible(on)
        self.atmo_labels.set_enabled(on)

    # the pulse view part (THEORY 8.5)

    def set_pulse_mode(self, on: bool, factor: float) -> None:
        """show or hide the pulse, its marker and the slow-motion factor."""
        self.pulse_on = on
        self.pulse_box.setVisible(on)
        self.pulse_line.setVisible(on)
        self.pulse_marker.setVisible(on)
        self.pulse_marker.setData(size=self.ui.pulse_marker_size * self._dpr)
        self.pulse_factor.setText(f"TIME ×1/{factor:,.0f}")
        self._place_overlays()

    def update_pulse(self, state, factor: float) -> None:
        """draw the pulse parts at their true length (c tau, about 300 m) and a screen-space marker at the pulse
        centre, which is not to scale. a part is the segment p_radar + u [d_from, d_to].
        p = p_radar + u d - position of a wavefront at distance d [m]"""
        tl = self.tl
        if tl is None or not self.pulse_on:
            return
        radar = np.asarray(tl.result.radar.position, dtype=float)
        pts, centres = [], []
        for a, b in state.segments:
            pa, pb = (radar + state.u * a) / KM, (radar + state.u * b) / KM
            pts += [pa, pb]
            centres.append(0.5 * (pa + pb))
        self.pulse_line.setData(pos=np.array(pts) if pts else np.zeros((2, 3)))
        self.pulse_line.setVisible(bool(pts))
        self.pulse_marker.setVisible(bool(centres))
        if centres:
            self.pulse_marker.setData(pos=np.array(centres[-1:]))
        head = max((b for _, b in state.segments), default=math.nan)
        self.pulse_phase.setText(f"{state.phase}   ·   t − t_tx = {fmt.disp(state.elapsed, 'ms')}"
                                 + (f"   ·   pulse at {fmt.disp(head, 'pos')}" if state.segments else ""))
        info = f"pulse length cτ = {fmt.disp(tl.c * tl.pulse_width, 'm')}, drawn to scale · marker not to scale"
        if state.detected:
            info += f" · round trip 2r/c = {fmt.disp(2.0 * state.r / tl.c, 'ms')}"
        self.pulse_info.setText(info)
        self._place_overlays()

    def _apply_sizes(self):
        """screen-space sizes are given in logical pixels; the point shader takes device pixels."""
        self._dpr = theme.device_pixel_ratio(self.view)
        d, ui = self._dpr, self.ui
        self.meas.setData(size=ui.measurement_size * d)
        self.sites.setData(size=ui.site_marker_size * d)
        self.halo.setData(size=ui.plume_marker_size * d)
        self.est_marker.setData(size=ui.estimate_marker_size * d)
        self.truth_marker.setData(size=ui.marker_size * d)

    def reset_view(self, elevation: float = CAMERA_ELEVATION, swing: float = CAMERA_SWING) -> None:
        """frame the whole trajectory, the radar and the launch site with the orbit camera. the camera stands
        on the side of the ground track from which the flight runs left to right (East to the right for the
        default eastward launch, like a map seen from the south), swung towards the launch end by swing [deg]
        and raised by elevation [deg].
        right = (-sin az, cos az) - screen-right direction of a camera at azimuth az looking at the centre
        az = atan2(-t_E, t_N) - camera azimuth that puts the track direction t on the right"""
        tl = self.tl
        if tl is None:
            return
        pts = np.vstack([tl.truth_pos_km, tl.radar_km[None, :], tl.launch_km[None, :]])
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        centre = 0.5 * (lo + hi)
        radius = 0.5 * float(np.linalg.norm(hi - lo))
        track = tl.truth_pos_km[-1, 0:2] - tl.launch_km[0:2]
        if np.linalg.norm(track) < 1e-9:
            track = np.array([1.0, 0.0])
        az = math.degrees(math.atan2(-track[0], track[1])) - swing
        self.view.setCameraPosition(pos=QVector3D(*centre.tolist()), elevation=elevation, azimuth=az)
        # the closest distance at which every framed point projects inside FRAME_NDC of the viewport, by
        # bisection: the projected extent shrinks monotonically as the camera backs away
        sample = pts[:: max(1, len(pts) // 400)]
        w, h = max(self.view.width(), 1), max(self.view.height(), 1)

        def fits(d: float) -> bool:
            self.view.opts["distance"] = d
            mvp = self.view.projectionMatrix((0, 0, w, h), (0, 0, w, h)) * self.view.viewMatrix()
            m = np.array(mvp.data(), dtype=float).reshape(4, 4).T  # QMatrix4x4.data() is column-major
            q = np.c_[sample, np.ones(len(sample))] @ m.T  # clip coordinates of every point
            if np.any(q[:, 3] <= 0.0):
                return False
            ndc = q[:, 0:2] / q[:, 3:4]
            return bool(np.all(np.abs(ndc[:, 0]) <= FRAME_NDC[0]) and np.all(np.abs(ndc[:, 1]) <= FRAME_NDC[1]))

        lo_d, hi_d = 0.2 * radius, 20.0 * radius
        for _ in range(40):
            mid = 0.5 * (lo_d + hi_d)
            lo_d, hi_d = (lo_d, mid) if fits(mid) else (mid, hi_d)
        self.view.setCameraPosition(distance=hi_d)
        self._place_atmosphere_labels()

    def _place_atmosphere_labels(self) -> None:
        """move the column of atmosphere labels to the candidate spot where, from the current camera, the labels
        stay inside the view, cover the fewest trajectory points and do not sit on the scene's other labels
        (radar, launch site, axes, range rings). a label is drawn above-right of its anchor,
        so its screen box is [px, px + w] x [py - h, py] (THEORY 8.3; AnchoredLabels gives px, py)."""
        tl, items = self.tl, self.atmo_labels.items
        if tl is None or not items or not self.atmo_anchors:
            return
        w, h = max(self.view.width(), 1), max(self.view.height(), 1)
        mvp = self.view.projectionMatrix((0, 0, w, h), (0, 0, w, h)) * self.view.viewMatrix()
        m = np.array(mvp.data(), dtype=float).reshape(4, 4).T  # QMatrix4x4.data() is column-major

        def to_px(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            q = np.c_[p, np.ones(len(p))] @ m.T
            ok = q[:, 3] > 0.0
            q[~ok, 3] = 1.0
            return np.c_[(q[:, 0] / q[:, 3] + 1.0) * 0.5 * w, (1.0 - q[:, 1] / q[:, 3]) * 0.5 * h], ok

        track, _ = to_px(tl.truth_pos_km[:: max(1, len(tl.truth_pos_km) // 600)])
        zs = np.array([p[2] for p, _ in items])
        sizes = np.array([(lab.width(), lab.height()) for _, lab in items], dtype=float)
        pad = 3.0  # clearance between a label and the trajectory [px]
        self.labels._key = None
        self.labels.place()  # the other scene labels at the new camera, as obstacles
        others = np.array([[g.left(), g.right(), g.top(), g.bottom()] for g in
                           (lab.geometry() for _, lab in self.labels.items if lab.isVisible())],
                          dtype=float).reshape(-1, 4)
        label_clash = 50.0  # cost of one atmosphere label on another scene label, in covered trajectory points [-]

        def cost(x: float, y: float) -> float:
            px, ok = to_px(np.c_[np.full(len(zs), x), np.full(len(zs), y), zs])
            l, r = px[:, 0] - pad, px[:, 0] + sizes[:, 0] + pad
            t, b = px[:, 1] - sizes[:, 1] - pad, px[:, 1] + pad
            outside = np.sum(~ok | (l < 0) | (r > w) | (t < 0) | (b > h))
            inside = ((track[:, 0][None, :] > l[:, None]) & (track[:, 0][None, :] < r[:, None])
                      & (track[:, 1][None, :] > t[:, None]) & (track[:, 1][None, :] < b[:, None]))
            clash = ((l[:, None] < others[None, :, 1] + pad) & (r[:, None] > others[None, :, 0] - pad)
                     & (t[:, None] < others[None, :, 3] + pad) & (b[:, None] > others[None, :, 2] - pad))
            return 1000.0 * outside + float(inside.sum()) + label_clash * float(clash.sum())

        costs = [cost(x, y) for x, y in self.atmo_anchors]
        self._atmo_costs = costs  # kept for inspection and the tests
        x, y = self.atmo_anchors[int(np.argmin(costs))]  # argmin keeps the first (north-west) spot on a tie
        self.atmo_labels.items = [(np.array([x, y, z]), lab) for z, (_, lab) in zip(zs, items)]
        self.atmo_labels._key = None
        self.atmo_labels.place()

    # the per-frame part

    def set_layer(self, name: str, on: bool) -> None:
        self.layers[name] = on
        self._last_k = self._last_det = None
        if name == "atmosphere":
            self._show_atmosphere()
        self._place_overlays()

    def set_ellipsoid_scale(self, scale: float) -> None:
        self.scale = float(scale)
        self._last_k = None
        self._update_factor()

    def update_frame(self, k: int, i: int) -> None:
        """draw the cursor state: truth at sample i (evaluation), the filter at scan k (held between scans)."""
        tl = self.tl
        if tl is None:
            return
        L = self.layers
        self.truth_line.setVisible(L["truth"])
        self.truth_marker.setVisible(L["truth"])
        if L["truth"]:
            self.truth_line.setData(pos=tl.truth_trail(i))
            self.truth_marker.setData(pos=tl.truth_pos_km[i:i + 1])
        flag = bool(tl.engine_flag[k])
        self.halo.setVisible(flag and L["truth"])
        if flag:
            self.halo.setData(pos=tl.truth_pos_km[i:i + 1])

        tracking = tl.tracking(k)
        if k != self._last_k:
            self.est_line.setVisible(L["estimate"] and tracking)
            self.est_marker.setVisible(L["estimate"] and tracking)
            if tracking and L["estimate"]:
                self.est_line.setData(pos=tl.estimate_trail(k))
                self.est_marker.setData(pos=tl.est_pos_km[k:k + 1])
            self._update_ellipsoid(k, tracking)
            self._update_beam(k)
            self._last_k = k
        n_det = int(tl.det_cum[k])
        if n_det != self._last_det:
            self.meas.setVisible(L["measurements"] and n_det > 0)
            if n_det > 0:
                self.meas.setData(pos=tl.measurements_upto(k))
            self._last_det = n_det

    def _update_ellipsoid(self, k: int, tracking: bool) -> None:
        """the n-sigma ellipsoid of the position block of P_{k|k}, magnified by the display factor.
        M = T(p_{k|k}) V diag(a_1, a_2, a_3) - unit sphere to ellipsoid"""
        tl = self.tl
        show = tracking and self.layers["ellipsoid"]
        self.ellipsoid.setVisible(show)
        if not show:
            return
        V, radii = ellipsoid_axes(tl.position_covariance(k), self.ui.ellipsoid_sigma, self.scale)
        M = np.eye(4)
        M[0:3, 0:3] = V @ np.diag(np.maximum(radii, 1e-6))
        M[0:3, 3] = tl.est_pos_km[k]
        self.ellipsoid.setTransform(_matrix(M))

    def _update_beam(self, k: int) -> None:
        """the beam cone of scan k along its pointing u_k, from the antenna to the predicted range.
        M = T(p_radar) R(z -> u_k) diag(L, L, L) - unit cone to beam"""
        tl = self.tl
        show = self.layers["beam"]
        self.beam.setVisible(show)
        if not show:
            return
        L = float(tl.beam_len_km[k])
        M = np.eye(4)
        M[0:3, 0:3] = rotation_from_z(tl.beam_u[k]) * L
        M[0:3, 3] = tl.radar_km
        self.beam.setTransform(_matrix(M))
        hit = bool(tl.detected[k])
        self.beam.setColor(rgba(C.BEAM, 0.16) if hit else rgba(C.BEAM_MISS, 0.22))
