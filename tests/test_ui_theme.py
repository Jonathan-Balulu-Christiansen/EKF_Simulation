"""THEORY 8.9: the colour tokens reach WCAG AA, colours carry one meaning each, and the bundled fonts
cover every character the UI writes."""

import pathlib

from fontTools.ttLib import TTFont

from ekfsim_ui import theme
from ekfsim_ui.theme import C

UI_DIR = pathlib.Path(theme.__file__).resolve().parent


def test_text_tokens_reach_wcag_aa_on_every_surface():
    for name in theme.TEXT_TOKENS:
        for surface in theme.SURFACE_TOKENS:
            cr = theme.contrast_ratio(getattr(C, name), getattr(C, surface))
            assert cr >= 4.5, f"{name} on {surface}: {cr:.2f}"


def test_graphic_tokens_reach_the_non_text_contrast():
    for name in theme.GRAPHIC_TOKENS:
        for surface in theme.SURFACE_TOKENS:
            assert theme.contrast_ratio(getattr(C, name), getattr(C, surface)) >= 3.0


def test_contrast_ratio_reference_values():
    assert theme.contrast_ratio("#000000", "#FFFFFF") == 21.0
    assert abs(theme.contrast_ratio("#777777", "#FFFFFF") - 4.48) < 0.01


def test_each_colour_has_one_meaning():
    """data, mode and status colours are all different, so no colour means two things."""
    meaning = ["TRUTH", "ESTIMATE", "COVARIANCE", "MEASUREMENT", "BEAM", "BEAM_MISS", "BOOST", "BALLISTIC",
               "BOUND", "LOCKED", "ACQUIRING", "COASTING", "LOST", "ACCENT"]
    values = [getattr(C, n).lower() for n in meaning]
    assert len(set(values)) == len(values)
    assert theme.STATUS_COLOUR == {"LOCKED": C.LOCKED, "ACQUIRING": C.ACQUIRING, "COASTING": C.COASTING,
                                   "LOST": C.LOST}


def test_fonts_are_bundled_with_their_licence():
    names = {p.name for p in theme.FONT_DIR.iterdir()}
    for f in ("IBMPlexSans-Regular.ttf", "IBMPlexSans-SemiBold.ttf", "IBMPlexMono-Regular.ttf",
              "IBMPlexMono-Medium.ttf", "OFL.txt"):
        assert f in names
    assert "SIL Open Font License" in (theme.FONT_DIR / "OFL.txt").read_text(encoding="utf-8")


def test_the_fonts_cover_every_character_of_the_ui():
    """labels use Plex Sans, numbers Plex Mono; a missing glyph would fall back to another font."""
    sans = TTFont(theme.FONT_DIR / "IBMPlexSans-Regular.ttf").getBestCmap()
    mono = TTFont(theme.FONT_DIR / "IBMPlexMono-Regular.ttf").getBestCmap()
    used = set()
    for path in UI_DIR.glob("*.py"):
        used |= {ord(c) for c in path.read_text(encoding="utf-8") if ord(c) > 127}
    missing = sorted(hex(o) for o in used if o not in sans)
    assert missing == []
    number_chars = {ord(c) for c in (theme_number_chars())}
    assert sorted(hex(o) for o in number_chars if o not in mono) == []


def theme_number_chars() -> str:
    from ekfsim_ui import fmt
    return fmt.MINUS + fmt.DASH + "·×,.+0123456789e"


def test_text_scale_grows_only_where_the_layout_fits():
    """THEORY 8.9: the layout is in em, so at text scale s it needs s times the smallest checked window
    (1366 x 768). the text grows one step on a screen with that room and keeps its size on a small one.
    s fits if width >= s W_min and height >= s H_min - room for the layout scaled by s"""
    assert theme.text_scale_for(1366, 768) == 1.0  # the smallest checked window itself
    assert theme.text_scale_for(1280, 688) == 1.0  # 1920 x 1080 at 150 % (taskbar excluded): smaller than that
    assert theme.text_scale_for(1.2 * 1366, 1.2 * 768) == 1.2  # exactly the room the larger text needs
    assert theme.text_scale_for(1920, 1032) == 1.2  # 1920 x 1080 at 100 %
    assert theme.text_scale_for(3115, 1256) == 1.2  # a wide screen: one step only, never more
