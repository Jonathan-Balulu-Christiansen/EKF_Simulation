"""here I keep the number policy of the UI (THEORY 8.9): every quantity has one fixed resolution that matches
what it means, numbers are right-aligned in fixed-width columns and always carry a unit. SI is converted
to display units (km, deg, kN, kHz) only here, at display time (THEORY 0.4). no Qt import.
- thousands are separated by commas, the minus sign is the true minus U+2212 (the same width as a digit
  in IBM Plex Mono, so columns stay aligned), and a missing value is an em dash.
- matrices use 3 significant digits in scientific form; exact zeros print as a centred dot, so the
  structure of P (for example the zero rows of a in BALLISTIC) stays visible."""

import math

MINUS = "−"
DASH = "—"
DEG = 180.0 / math.pi  # degrees per radian
RDOT = "r" + chr(0x0307)  # r with a combining dot above: IBM Plex has no precomposed U+1E59

# quantity name -> (factor from SI, decimals, unit). the resolution is the smallest step that matters
QUANTITIES = {
    "time": (1.0, 2, "s"),  # 10 ms, a tenth of the scan interval
    "km": (1e-3, 3, "km"),  # 1 m, a tenth of sigma_r
    "m": (1.0, 1, "m"),
    "m_int": (1.0, 0, "m"),
    "speed": (1.0, 1, "m/s"),
    "rdot": (1.0, 2, "m/s"),  # a hundredth of sigma_rdot (1.5 m/s)
    "accel": (1.0, 2, "m/s²"),
    "mach": (1.0, 2, ""),
    "mass": (1.0, 0, "kg"),
    "kN": (1e-3, 2, "kN"),
    "beta": (1.0, 0, "kg/m²"),
    "deg": (DEG, 3, "°"),  # a hundredth of sigma_angle (0.1 deg)
    "deg1": (DEG, 1, "°"),
    "nis": (1.0, 2, ""),
    "ratio": (1.0, 3, ""),
    "kHz": (1e-3, 1, "kHz"),
    "cm": (100.0, 2, "cm"),
    "ms": (1e3, 3, "ms"),
    "us": (1e6, 1, "µs"),
    "percent": (100.0, 1, "%"),
}


def _minus(s: str) -> str:
    return s.replace("-", MINUS)


def num(x, decimals: int, sign: bool = False, thousands: bool = True) -> str:
    """x with a fixed number of decimals, or an em dash when x is missing (None or NaN)."""
    if x is None or not math.isfinite(float(x)):
        return DASH
    pattern = ("+" if sign else "") + ("," if thousands else "") + f".{decimals}f"
    return _minus(format(float(x), pattern))


def q(x, name: str, sign: bool = False) -> str:
    """x in SI, shown as the named quantity without its unit (see QUANTITIES)."""
    factor, decimals, _ = QUANTITIES[name]
    return num(None if x is None else float(x) * factor, decimals, sign)


def unit(name: str) -> str:
    return QUANTITIES[name][2]


def qu(x, name: str, sign: bool = False) -> str:
    """x in SI with its display unit, e.g. '161.555 km'."""
    u = unit(name)
    return q(x, name, sign) + (f" {u}" if u and u != "°" else u)


def sci(x, sig: int = 3, width: int = 9) -> str:
    """x in scientific form with sig significant digits, right-aligned to width characters; exact zero
    is a centred dot and a missing value an em dash."""
    if x is None or not math.isfinite(float(x)):
        return DASH.rjust(width)
    if x == 0.0:
        return "·".center(width)
    return _minus(format(float(x), f"+.{sig - 1}e")).rjust(width)


# the fixed display unit part (equations panel):

# one fixed display unit per quantity type, never a prefix chosen per value: a column of velocities must not mix
# m/s and mm/s. display unit -> factor from SI. only a position switches between km and m (see pos)
DISPLAY_UNITS = {
    "km": 1e-3, "m": 1.0, "cm": 100.0, "m/s": 1.0, "m/s²": 1.0, "s": 1.0, "ms": 1e3, "GHz": 1e-9,
    "kHz": 1e-3, "Hz": 1.0, "°": DEG, "mrad": 1e3, "kg": 1.0, "kg/m²": 1.0, "kg/m³": 1.0, "kN": 1e-3,
    "kPa": 1e-3, "m²/s³": 1.0, "m²/s⁵": 1.0, "1/s": 1.0, "": 1.0,
}


def _plain(x: float) -> str:
    """3-4 significant digits without a prefix, with thousands separators: 22,700, 149.7, 9.43, 0.263."""
    a = abs(x)
    if a >= 1000.0:
        return num(round(x, -(math.floor(math.log10(a)) - 3)), 0)
    if a >= 100.0:
        return num(x, 1)
    if a >= 1.0:
        return num(x, 2)
    return num(x, max(2 - math.floor(math.log10(a)), 0))


def disp_parts(x, unit: str = "", sign: bool = False) -> tuple[str, str]:
    """x in SI shown in a fixed display unit (DISPLAY_UNITS) with 3-4 significant digits, as (number, unit).
    the unit "pos" is a position or a distance: km from 1 km on, m below, the one exception to the rule."""
    if x is None or not math.isfinite(float(x)):
        return DASH, ("km" if unit == "pos" else unit)
    x = float(x)
    if unit == "pos":
        unit = "km" if abs(x) >= 1000.0 else "m"
    v = x * DISPLAY_UNITS[unit]
    if v == 0.0:
        return "0", unit
    s = _plain(v)
    return ("+" + s if sign and v > 0 else s), unit


def disp(x, unit: str = "", sign: bool = False) -> str:
    """x in SI with its fixed display unit, e.g. '149.7 km', '0.0790 m/s', '245.8°', '−3.43 mrad'."""
    n, u = disp_parts(x, unit, sign)
    if not u:
        return n
    return n + u if u == "°" else f"{n} {u}"


def sci_cell(x) -> str:
    """a raw matrix entry for the 'show values' tables: 3 significant digits, plain between 0.01 and 1000,
    otherwise in a compact scientific form with the true minus ('−1.32e−5'); exact zero is a centred dot."""
    if x is None or not math.isfinite(float(x)):
        return DASH
    x = float(x)
    if x == 0.0:
        return "\u00b7"
    a = abs(x)
    if 0.01 <= a < 1000.0:
        return num(x, 0 if a >= 100.0 else (1 if a >= 10.0 else (2 if a >= 1.0 else 3)))
    e = math.floor(math.log10(a))
    m = x / 10.0 ** e
    if round(abs(m), 2) >= 10.0:
        m, e = m / 10.0, e + 1
    return f"{num(m, 2)}e{_minus(str(e))}"


def exact(x, unit: str = "") -> str:
    """the exact value for a tooltip: 10 significant digits."""
    if x is None or not math.isfinite(float(x)):
        return DASH
    return _minus(format(float(x), ".10g")) + (f" {unit}" if unit else "")


def sig(x, digits: int = 3) -> str:
    """x with a fixed number of significant digits in plain or scientific form, for values whose range
    spans decades (density, q, spectral densities)."""
    if x is None or not math.isfinite(float(x)):
        return DASH
    if x == 0.0:
        return "0"
    e = math.floor(math.log10(abs(x)))
    if -3 <= e < 5:
        return num(x, max(digits - 1 - e, 0))
    return _minus(format(float(x), f".{digits - 1}e"))
