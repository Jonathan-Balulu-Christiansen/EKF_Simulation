"""here I define the look of the UI once (THEORY 8.9): the colour tokens, the bundled fonts with their type
scale, the Qt stylesheet and the pyqtgraph defaults. every view takes its colours from here, so a colour
means the same thing in the controls, the plots and the 3D view.
direction: a calibrated engineering instrument. a deep blue-grey slate (not black), hairline borders
instead of cards and shadows, colour only where it carries meaning. data hues (truth, estimate,
covariance, measurements, beam, modes) avoid green, amber and red, which are reserved for status.
every text colour reaches the WCAG AA contrast of 4.5:1 on every surface (a test recomputes it).
sizes are in points and em (the line height of the body font), never in fixed pixels, so they follow
the font and the screen scaling."""

import pathlib
from dataclasses import dataclass, fields

from PySide6.QtGui import QColor, QFont, QFontDatabase, QFontMetricsF, QGuiApplication, QPalette

FONT_DIR = pathlib.Path(__file__).resolve().parent / "assets" / "fonts"
ICON_DIR = pathlib.Path(__file__).resolve().parent / "assets" / "icons"
SANS = "IBM Plex Sans"
MONO = "IBM Plex Mono"


# the colour token part:

@dataclass(frozen=True)
class Palette:
    """the colour tokens with what each one means (hex sRGB)."""

    BG: str = "#11171E"  # window background, deep blue-grey slate
    SURFACE: str = "#161E27"  # panels: controls, readout, plots, equations
    RAISED: str = "#1E2833"  # inputs, tab bars, equation cards
    SCENE: str = "#0C1117"  # 3D view background, the deepest tone
    HAIRLINE: str = "#2B3644"  # borders, splitter handles, separators
    GRID: str = "#222C38"  # plot grid and 3D ground grid
    INK: str = "#E4E9EF"  # primary text and numbers
    INK_2: str = "#AAB6C3"  # secondary text: labels, axis titles
    INK_3: str = "#8794A4"  # tertiary text: units, captions, hints
    ACCENT: str = "#D6E0EA"  # UI only: focus, active tab, primary button, playback cursor
    TRUTH: str = "#E6DCC5"  # ground truth and the "evaluation - uses truth" tag
    ESTIMATE: str = "#45B0F0"  # EKF estimate and NIS
    COVARIANCE: str = "#8E97FF"  # uncertainty from P: ellipsoid, +-3 sigma bands
    MEASUREMENT: str = "#D886E8"  # radar measurements and innovations
    BEAM: str = "#5ED1C4"  # radar beam with an echo
    BEAM_MISS: str = "#6E7A8C"  # radar beam without an echo (graphics only)
    BOOST: str = "#F2955E"  # BOOST mode, engine on (IR plume marker)
    BALLISTIC: str = "#9FB1C6"  # BALLISTIC mode
    BOUND: str = "#7F8C9C"  # chi2 bounds that do not come from P (graphics only)
    ATMOSPHERE: str = "#6D86A3"  # ISA layer boundaries in the 3D view, drawn faint (graphics only)
    LOCKED: str = "#5BC98A"  # status LOCKED
    ACQUIRING: str = "#9FB0C2"  # status ACQUIRING
    COASTING: str = "#E8C25A"  # status COASTING, manoeuvre noise ON, values outside a bound
    LOST: str = "#F2676F"  # status LOST, NO ACQUISITION, errors


C = Palette()
TEXT_TOKENS = ("INK", "INK_2", "INK_3", "ACCENT", "TRUTH", "ESTIMATE", "COVARIANCE", "MEASUREMENT", "BEAM",
               "BOOST", "BALLISTIC", "LOCKED", "ACQUIRING", "COASTING", "LOST")  # used as text: AA required
GRAPHIC_TOKENS = ("BEAM_MISS", "BOUND", "ATMOSPHERE")  # graphics only: 3:1 required (WCAG 1.4.11)
SURFACE_TOKENS = ("BG", "SURFACE", "RAISED", "SCENE")
STATUS_COLOUR = {"LOCKED": C.LOCKED, "ACQUIRING": C.ACQUIRING, "COASTING": C.COASTING, "LOST": C.LOST}
MODE_COLOUR = {"BOOST": C.BOOST, "BALLISTIC": C.BALLISTIC}


def tokens() -> dict[str, str]:
    return {f.name: getattr(C, f.name) for f in fields(Palette)}


def qcolor(hex_colour: str, alpha: float = 1.0) -> QColor:
    c = QColor(hex_colour)
    c.setAlphaF(alpha)
    return c


def rgba(hex_colour: str, alpha: float = 1.0) -> tuple[float, float, float, float]:
    """the colour as an (r, g, b, a) tuple in [0, 1], the form the OpenGL items take."""
    c = QColor(hex_colour)
    return (c.redF(), c.greenF(), c.blueF(), alpha)


def css_rgba(hex_colour: str, alpha: float) -> str:
    c = QColor(hex_colour)
    return f"rgba({c.red()}, {c.green()}, {c.blue()}, {alpha:.3f})"


def contrast_ratio(a: str, b: str) -> float:
    """the WCAG 2.1 contrast ratio between two sRGB colours.
    c_lin = c / 12.92 (c <= 0.04045), ((c + 0.055) / 1.055)^2.4 otherwise - linearised channel
    L = 0.2126 R + 0.7152 G + 0.0722 B - relative luminance
    CR = (L_light + 0.05) / (L_dark + 0.05) - contrast ratio"""
    def lum(h: str) -> float:
        h = h.lstrip("#")
        ch = [int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]
        ch = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in ch]
        return 0.2126 * ch[0] + 0.7152 * ch[1] + 0.0722 * ch[2]
    hi, lo = sorted((lum(a), lum(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


# the type part:

# one factor on every text size, so the whole UI reads larger or smaller together; sizes in em follow it.
# the layout is checked at the smallest supported window (THEORY 8.9); everything in it is in em, so at scale s
# it needs s times that size. the text grows one step only on a screen with that much room.
TEXT_SCALES = (1.0, 1.2)  # allowed multipliers on the point sizes below [-]. DESIGN CHOICE: one step of the type scale
MIN_LAYOUT = (1366, 768)  # smallest window the layout is checked at, in logical [px]. DESIGN CHOICE: THEORY 8.9 acceptance
TEXT_SCALE: float | None = None  # the chosen multiplier [-]; None until apply() picks it from the screen
# role -> (family, point size before TEXT_SCALE, weight, uppercase tracking in %). a step of about x1.2 between sizes
TYPE_SCALE = {
    "caption": (SANS, 7.5, QFont.Weight.Normal, 0.0),
    "section": (SANS, 7.5, QFont.Weight.DemiBold, 6.0),
    "body": (SANS, 9.0, QFont.Weight.Normal, 0.0),
    "body_medium": (SANS, 9.0, QFont.Weight.Medium, 0.0),
    "mono": (MONO, 9.0, QFont.Weight.Normal, 0.0),
    "mono_medium": (MONO, 9.0, QFont.Weight.Medium, 0.0),
    "mono_small": (MONO, 8.0, QFont.Weight.Normal, 0.0),
    "mono_caption": (MONO, 7.5, QFont.Weight.Normal, 0.0),
    "display": (MONO, 13.0, QFont.Weight.Medium, 0.0),
    "display_large": (MONO, 18.0, QFont.Weight.Medium, 0.0),
}
_loaded = False


def load_fonts() -> None:
    """register the bundled IBM Plex fonts (SIL OFL 1.1) with Qt, once."""
    global _loaded
    if _loaded:
        return
    for path in sorted(FONT_DIR.glob("*.ttf")):
        if QFontDatabase.addApplicationFont(str(path)) < 0:
            raise RuntimeError(f"could not load the bundled font {path.name}")
    _loaded = True


def font(role: str) -> QFont:
    family, size, weight, tracking = TYPE_SCALE[role]
    f = QFont(family)
    f.setPointSizeF(size * (TEXT_SCALE or 1.0))
    f.setWeight(weight)
    f.setHintingPreference(QFont.HintingPreference.PreferNoHinting)
    if tracking:
        f.setLetterSpacing(QFont.SpacingType.PercentageSpacing, 100.0 + tracking)
    return f


def em(role: str = "body") -> float:
    """1 em: the line height of a type role in logical pixels (15.6 px for the 9 pt body at text scale 1)."""
    return QFontMetricsF(font(role)).height()


def char_width(role: str = "mono") -> float:
    """the advance of one digit in a type role in logical pixels."""
    return QFontMetricsF(font(role)).horizontalAdvance("0")


# the stylesheet part:

def stylesheet() -> str:
    """the Qt stylesheet. lengths are in em and pt, so they scale with the font and the DPI."""
    c = C
    check = (ICON_DIR / "check.svg").as_posix()
    return f"""
QMainWindow, QSplitter, QStatusBar {{ background: {c.BG}; }}
QWidget {{ color: {c.INK}; }}
QWidget#panel, QScrollArea#panel, QWidget#panel_body, QTabWidget#panel > QStackedWidget > QWidget {{
    background: {c.SURFACE}; }}
QScrollArea {{ border: none; background: transparent; }}
QScrollArea > QWidget > QWidget {{ background: transparent; }}
QSplitter::handle {{ background: {c.BG}; }}
QSplitter::handle:hover {{ background: {c.HAIRLINE}; }}
QToolTip {{ background: {c.RAISED}; color: {c.INK}; border: 1px solid {c.HAIRLINE}; padding: 0.3em 0.5em; }}

QTabWidget::pane {{ border: none; border-top: 1px solid {c.HAIRLINE}; background: {c.SURFACE}; }}
QTabBar {{ background: transparent; qproperty-drawBase: 0; }}
QTabBar::tab {{ background: transparent; color: {c.INK_2}; padding: 0.35em 0.75em 0.3em 0.75em;
    border: none; border-bottom: 2px solid transparent; }}
QTabBar::tab:hover {{ color: {c.INK}; }}
QTabBar::tab:selected {{ color: {c.INK}; border-bottom: 2px solid {c.ACCENT}; }}
QTabBar QToolButton {{ background: {c.SURFACE}; border: none; padding: 0; }}

QPushButton, QToolButton {{ background: {c.RAISED}; color: {c.INK}; border: 1px solid {c.HAIRLINE};
    border-radius: 2px; padding: 0.25em 0.8em; }}
QPushButton:hover, QToolButton:hover {{ border-color: {c.INK_3}; }}
QPushButton:pressed, QToolButton:pressed {{ background: {c.HAIRLINE}; }}
QPushButton:disabled, QToolButton:disabled {{ color: {c.INK_3}; border-color: {c.GRID}; }}
QPushButton:focus, QToolButton:focus {{ border-color: {c.ACCENT}; }}
QToolButton:checked {{ background: {c.HAIRLINE}; border-color: {c.ACCENT}; color: {c.INK}; }}
QToolButton#collapse, QToolButton#collapse:checked {{ background: transparent; border: none; color: {c.INK_3};
    padding: 0.25em 0.1em; }}
QToolButton#collapse:hover {{ color: {c.INK}; }}
QToolButton#overlay_button, QToolButton#overlay_button:checked {{ background: transparent; border: none;
    color: {c.INK_2}; padding: 0.1em 0.2em; }}
QToolButton#overlay_button:hover {{ color: {c.INK}; }}
QPushButton[primary="true"] {{ border-color: {c.ACCENT}; color: {c.ACCENT}; font-weight: 500; }}
QPushButton[primary="true"][dirty="true"] {{ background: {c.ACCENT}; color: {c.BG}; }}
QPushButton[primary="true"]:disabled {{ border-color: {c.HAIRLINE}; color: {c.INK_3}; background: {c.RAISED}; }}

QAbstractSpinBox, QLineEdit, QComboBox {{ background: {c.RAISED}; color: {c.INK}; border: 1px solid {c.HAIRLINE};
    border-radius: 2px; padding: 0.1em 0.3em; font-family: "{MONO}"; selection-background-color: {c.HAIRLINE}; }}
QAbstractSpinBox:focus, QLineEdit:focus, QComboBox:focus {{ border-color: {c.ACCENT}; }}
QAbstractSpinBox:disabled, QComboBox:disabled {{ color: {c.INK_3}; }}
QAbstractSpinBox::up-button, QAbstractSpinBox::down-button {{ background: transparent; border: none; width: 1.0em; }}
QAbstractSpinBox::up-button:hover, QAbstractSpinBox::down-button:hover {{ background: {c.HAIRLINE}; }}
QComboBox::drop-down {{ border: none; width: 1.2em; }}
QComboBox QAbstractItemView {{ background: {c.RAISED}; color: {c.INK}; border: 1px solid {c.HAIRLINE};
    selection-background-color: {c.HAIRLINE}; outline: none; }}

QCheckBox {{ spacing: 0.5em; color: {c.INK}; }}
QCheckBox::indicator {{ width: 0.8em; height: 0.8em; border: 1px solid {c.INK_3}; background: {c.RAISED};
    border-radius: 1px; }}
QCheckBox::indicator:checked {{ background: {c.ACCENT}; border-color: {c.ACCENT}; image: url({check}); }}
QCheckBox::indicator:hover {{ border-color: {c.INK}; }}
QCheckBox:disabled {{ color: {c.INK_3}; }}

QSlider::groove:horizontal {{ height: 0.2em; background: {c.HAIRLINE}; border-radius: 1px; }}
QSlider::sub-page:horizontal {{ background: {c.INK_3}; border-radius: 1px; }}
QSlider::handle:horizontal {{ width: 0.75em; height: 0.75em; margin: -0.3em 0; background: {c.ACCENT};
    border-radius: 0.375em; }}
QSlider::handle:horizontal:hover {{ background: {c.INK}; }}

QScrollBar:vertical {{ background: transparent; width: 0.6em; margin: 0; }}
QScrollBar:horizontal {{ background: transparent; height: 0.6em; margin: 0; }}
QScrollBar::handle {{ background: {c.HAIRLINE}; border-radius: 0.3em; min-height: 2em; min-width: 2em; }}
QScrollBar::handle:hover {{ background: {c.INK_3}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

QProgressBar {{ background: {c.HAIRLINE}; border: none; border-radius: 1px; max-height: 0.25em; }}
QProgressBar::chunk {{ background: {c.ACCENT}; }}

QStatusBar {{ border-top: 1px solid {c.HAIRLINE}; color: {c.INK_2}; }}
QStatusBar::item {{ border: none; }}

QLabel[role="caption"] {{ color: {c.INK_3}; }}
QLabel[role="secondary"] {{ color: {c.INK_2}; }}
QLabel[role="section"] {{ color: {c.INK_3}; }}
QLabel[role="value"] {{ color: {c.INK}; }}
QLabel[role="unit"] {{ color: {c.INK_3}; }}
QLabel[role="error"] {{ color: {c.LOST}; }}
QLabel[role="warning"] {{ color: {c.COASTING}; }}
QLabel[role="eval"] {{ color: {c.TRUTH}; border: 1px dashed {css_rgba(c.TRUTH, 0.55)}; border-radius: 2px;
    padding: 0 0.35em; }}
QFrame[role="card"] {{ background: {c.RAISED}; border: 1px solid {c.HAIRLINE}; border-radius: 2px; }}
"""


def configure_pyqtgraph() -> None:
    import pyqtgraph as pg
    pg.setConfigOptions(antialias=True, background=C.SURFACE, foreground=C.INK_2)


def text_scale_for(width: float, height: float) -> float:
    """the largest text scale whose layout still fits a screen of width x height logical px.
    s fits if width >= s W_min and height >= s H_min - room for the layout scaled by s"""
    fits = [s for s in TEXT_SCALES if width >= s * MIN_LAYOUT[0] and height >= s * MIN_LAYOUT[1]]
    return max(fits, default=min(TEXT_SCALES))


def apply(app) -> None:
    """load the fonts, pick the text scale from the primary screen (unless a caller set it), and apply the
    theme to the whole application."""
    global TEXT_SCALE
    load_fonts()
    if TEXT_SCALE is None:
        screen = app.primaryScreen()
        g = screen.availableGeometry() if screen else None
        TEXT_SCALE = text_scale_for(g.width(), g.height()) if g else min(TEXT_SCALES)
    app.setFont(font("body"))
    pal = QPalette()
    for role, colour in ((QPalette.ColorRole.Window, C.BG), (QPalette.ColorRole.Base, C.RAISED),
                         (QPalette.ColorRole.AlternateBase, C.SURFACE), (QPalette.ColorRole.Text, C.INK),
                         (QPalette.ColorRole.WindowText, C.INK), (QPalette.ColorRole.Button, C.RAISED),
                         (QPalette.ColorRole.ButtonText, C.INK), (QPalette.ColorRole.Highlight, C.HAIRLINE),
                         (QPalette.ColorRole.HighlightedText, C.INK), (QPalette.ColorRole.ToolTipBase, C.RAISED),
                         (QPalette.ColorRole.ToolTipText, C.INK), (QPalette.ColorRole.PlaceholderText, C.INK_3)):
        pal.setColor(role, QColor(colour))
    app.setPalette(pal)
    app.setStyleSheet(stylesheet())
    configure_pyqtgraph()


def device_pixel_ratio(widget=None) -> float:
    if widget is not None:
        return float(widget.devicePixelRatioF())
    screen = QGuiApplication.primaryScreen()
    return float(screen.devicePixelRatio()) if screen else 1.0
