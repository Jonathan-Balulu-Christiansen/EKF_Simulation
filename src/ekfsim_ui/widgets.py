"""small shared widgets of the UI: section headers, the "evaluation - uses truth" tag, status chips,
legend swatches, vector icons (drawn with QPainter, so they stay sharp at every screen scaling, and the
bundled fonts have no geometric glyphs) and a flow layout that wraps cards to the available width."""

import math

import numpy as np
from PySide6.QtCore import QEvent, QPointF, QRect, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QIcon, QIconEngine, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import (QFrame, QHBoxLayout, QLabel, QLayout, QScrollArea, QSizePolicy, QToolButton, QToolTip,
                               QVBoxLayout, QWidget)

from ekfsim_ui import theme
from ekfsim_ui.theme import C

EVAL_TEXT = "evaluation – uses truth"


def label(text: str = "", role: str | None = None, font_role: str | None = None,
          align=None, tooltip: str = "") -> QLabel:
    w = QLabel(text)
    if role:
        w.setProperty("role", role)
    if font_role:
        w.setFont(theme.font(font_role))
    if align is not None:
        w.setAlignment(align)
    if tooltip:
        w.setToolTip(tooltip)
    return w


def section_header(text: str) -> QLabel:
    return label(text.upper(), "section", "section")


def eval_tag(text: str = EVAL_TEXT) -> QLabel:
    """the tag every truth-based view carries (THEORY 0.2), drawn in the truth colour."""
    w = label(text, "eval", "caption", tooltip="this view reads ground truth; the tracker never does")
    w.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
    return w


def scroll(widget: QWidget, horizontal: bool = False) -> QScrollArea:
    """a frameless scroll area: panels scroll instead of clipping when the window is small (THEORY 8.9)."""
    s = QScrollArea()
    s.setWidget(widget)
    s.setWidgetResizable(True)
    s.setFrameShape(QFrame.Shape.NoFrame)
    s.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded if horizontal
                                   else Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    return s


class Chip(QLabel):
    """a status or mode chip: coloured text on a faint fill of the same colour, with a hairline border."""

    def __init__(self, text: str = "", colour: str = C.INK_3, font_role: str = "mono_medium"):
        super().__init__()
        self.setFont(theme.font(font_role))
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._state = None
        self.set(text, colour)

    def set(self, text: str, colour: str) -> None:
        old = self._state or (None, None)
        self._state = (text, colour)
        if text != old[0]:
            self.setText(text)
        if colour == old[1]:
            return
        self.setStyleSheet(f"QLabel {{ color: {colour}; background: {theme.css_rgba(colour, 0.12)}; "
                           f"border: 1px solid {theme.css_rgba(colour, 0.45)}; border-radius: 2px; "
                           f"padding: 0 0.45em; }}")


class Swatch(QWidget):
    """a legend glyph painted in a data colour: dot, ring, line, dashed line, band or cone."""

    def __init__(self, colour: str, kind: str = "dot", colour2: str | None = None):
        super().__init__()
        self.colour, self.kind, self.colour2 = colour, kind, colour2
        e = theme.em("caption")
        self.setFixedSize(QSize(round(1.6 * e), round(0.9 * e)))

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        c = QColor(self.colour)
        cy, cx = r.center().y(), r.center().x()
        h = r.height()
        if self.kind == "dot":
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(c)
            p.drawEllipse(QPointF(cx, cy), 0.28 * h, 0.28 * h)
        elif self.kind == "small_dot":
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(c)
            for dx in (-0.35, 0.0, 0.35):
                p.drawEllipse(QPointF(cx + dx * r.width(), cy + 0.1 * h * math.sin(dx * 9)), 0.12 * h, 0.12 * h)
        elif self.kind == "halo":
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(theme.qcolor(self.colour, 0.45))
            p.drawEllipse(QPointF(cx, cy), 0.48 * h, 0.48 * h)
            p.setBrush(QColor(self.colour2 or C.TRUTH))
            p.drawEllipse(QPointF(cx, cy), 0.2 * h, 0.2 * h)
        elif self.kind in ("line", "dash"):
            pen = QPen(c, max(1.5, 0.12 * h))
            if self.kind == "dash":
                pen.setStyle(Qt.PenStyle.DashLine)
            p.setPen(pen)
            p.drawLine(QPointF(r.left(), cy), QPointF(r.right(), cy))
        elif self.kind == "band":
            p.setPen(QPen(c, 1))
            p.setBrush(theme.qcolor(self.colour, 0.25))
            p.drawRect(r.adjusted(0, 0.2 * h, 0, -0.2 * h))
        elif self.kind == "cone":
            path = QPainterPath(QPointF(r.left(), cy))
            path.lineTo(QPointF(r.right(), r.top()))
            path.lineTo(QPointF(r.right(), r.bottom()))
            path.closeSubpath()
            p.setPen(QPen(c, 1))
            p.setBrush(theme.qcolor(self.colour, 0.35))
            p.drawPath(path)
        elif self.kind == "ellipse":
            p.setPen(QPen(c, 1.2))
            p.setBrush(theme.qcolor(self.colour, 0.3))
            p.drawEllipse(QPointF(cx, cy), 0.45 * r.width(), 0.4 * h)
        p.end()


# the icon part:

class _PainterIcon(QIconEngine):
    """a vector icon drawn at whatever size and pixel ratio Qt asks for."""

    def __init__(self, name: str, colour: str):
        super().__init__()
        self.name, self.colour = name, colour

    def clone(self):
        return _PainterIcon(self.name, self.colour)

    def paint(self, painter: QPainter, rect: QRect, mode, state):
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        colour = QColor(C.INK_3 if mode == QIcon.Mode.Disabled else self.colour)
        s = min(rect.width(), rect.height())
        r = QRectF(rect.center().x() - s / 2, rect.center().y() - s / 2, s, s).adjusted(s * .2, s * .2, -s * .2, -s * .2)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(colour)
        w, h, x, y = r.width(), r.height(), r.left(), r.top()

        def tri(x0, pointing_right=True, width=None):
            width = width or w
            path = QPainterPath()
            if pointing_right:
                path.moveTo(x0, y)
                path.lineTo(x0 + width, y + h / 2)
                path.lineTo(x0, y + h)
            else:
                path.moveTo(x0 + width, y)
                path.lineTo(x0, y + h / 2)
                path.lineTo(x0 + width, y + h)
            path.closeSubpath()
            painter.drawPath(path)

        if self.name == "play":
            tri(x + 0.08 * w, True, 0.84 * w)
        elif self.name == "pause":
            painter.drawRect(QRectF(x + 0.12 * w, y, 0.26 * w, h))
            painter.drawRect(QRectF(x + 0.62 * w, y, 0.26 * w, h))
        elif self.name == "step_back":
            painter.drawRect(QRectF(x, y, 0.16 * w, h))
            tri(x + 0.22 * w, False, 0.78 * w)
        elif self.name == "step_forward":
            tri(x, True, 0.78 * w)
            painter.drawRect(QRectF(x + 0.84 * w, y, 0.16 * w, h))
        elif self.name == "to_start":
            painter.drawRect(QRectF(x, y, 0.14 * w, h))
            tri(x + 0.16 * w, False, 0.42 * w)
            tri(x + 0.58 * w, False, 0.42 * w)
        elif self.name in ("chev_right", "chev_down", "chev_up"):
            painter.setBrush(Qt.BrushStyle.NoBrush)
            pen = QPen(colour, max(1.2, 0.14 * w))
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            painter.setPen(pen)
            c = r.center()
            q = 0.28 * w
            if self.name == "chev_right":
                pts = [QPointF(c.x() - 0.5 * q, c.y() - q), QPointF(c.x() + 0.5 * q, c.y()), QPointF(c.x() - 0.5 * q, c.y() + q)]
            elif self.name == "chev_down":
                pts = [QPointF(c.x() - q, c.y() - 0.5 * q), QPointF(c.x(), c.y() + 0.5 * q), QPointF(c.x() + q, c.y() - 0.5 * q)]
            else:
                pts = [QPointF(c.x() - q, c.y() + 0.5 * q), QPointF(c.x(), c.y() - 0.5 * q), QPointF(c.x() + q, c.y() + 0.5 * q)]
            painter.drawPolyline(pts)
        painter.restore()

    def pixmap(self, size: QSize, mode, state):
        pm = QPixmap(size)
        pm.fill(Qt.GlobalColor.transparent)
        p = QPainter(pm)
        self.paint(p, QRect(0, 0, size.width(), size.height()), mode, state)
        p.end()
        return pm


def icon(name: str, colour: str = C.INK) -> QIcon:
    return QIcon(_PainterIcon(name, colour))


# the painted table part:

class TextTable(QWidget):
    """a table of labelled rows, painted directly: row labels and headers in the sans font (it has the
    Greek letters), cells in the mono font, right-aligned, and a unit column. painting avoids one QLabel per
    cell, so a 10 x 10 matrix updates at the playback rate. rows can be dimmed (frozen or zero rows) and
    diagonal cells tinted (covariances)."""

    def __init__(self, n_cols: int, col_headers=None, cell_chars: int = 9, unit_col: bool = True,
                 diagonal: str | None = None):
        super().__init__()
        self.n_cols = n_cols
        self.col_headers = list(col_headers) if col_headers else None
        self.cell_chars = cell_chars
        self.unit_col = unit_col
        self.diagonal = diagonal  # tint colour of the diagonal cells, or None
        self.rows: list[tuple[str, list[str], str]] = []
        self.dim_rows: set[int] = set()
        self.cell_colours: dict[tuple[int, int], str] = {}
        self._f_label, self._f_cell, self._f_head = theme.font("caption"), theme.font("mono_small"), theme.font("caption")
        self._widest = 0.0  # widest cell text so far in [px]
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

    def set_rows(self, rows, dim_rows=(), cell_colours=None) -> None:
        old_shape = ([(r[0], len(r[1]), r[2]) for r in self.rows], self._widest)
        self.rows = [(str(a), [str(c) for c in b], str(u)) for a, b, u in rows]
        self.dim_rows = set(dim_rows)
        self.cell_colours = dict(cell_colours or {})
        from PySide6.QtGui import QFontMetricsF
        fc = QFontMetricsF(self._f_cell)
        widest = max([fc.horizontalAdvance(c) for r in self.rows for c in r[1]] + [0.0])
        self._widest = max(self._widest, widest)  # grows only, so the table does not jitter while values change
        if ([(r[0], len(r[1]), r[2]) for r in self.rows], self._widest) != old_shape:
            self.updateGeometry()
        self.update()

    def _metrics(self):
        from PySide6.QtGui import QFontMetricsF
        fl, fc, fh = (QFontMetricsF(f) for f in (self._f_label, self._f_cell, self._f_head))
        label_w = max([fl.horizontalAdvance(r[0]) for r in self.rows] + [0.0]) + fl.horizontalAdvance("  ")
        cell_w = max(fc.horizontalAdvance("0") * self.cell_chars, self._widest,
                     max([fh.horizontalAdvance(h) for h in self.col_headers or []] + [0.0])) + fc.horizontalAdvance(" ")
        unit_w = (max([fl.horizontalAdvance(r[2]) for r in self.rows] + [0.0]) + fl.horizontalAdvance("  ")
                  if self.unit_col else 0.0)
        row_h = max(fc.height(), fl.height()) * 1.02
        head_h = fh.height() * 1.1 if self.col_headers else 0.0
        return label_w, cell_w, unit_w, row_h, head_h

    def sizeHint(self):
        label_w, cell_w, unit_w, row_h, head_h = self._metrics()
        return QSize(math.ceil(label_w + self.n_cols * cell_w + unit_w), math.ceil(head_h + len(self.rows) * row_h) + 2)

    def minimumSizeHint(self):
        return self.sizeHint()

    def paintEvent(self, event):
        label_w, cell_w, unit_w, row_h, head_h = self._metrics()
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        if self.col_headers:
            p.setFont(self._f_head)
            p.setPen(QColor(C.INK_3))
            for j, h in enumerate(self.col_headers):
                p.drawText(QRectF(label_w + j * cell_w, 0, cell_w, head_h),
                           int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter), h)
        for i, (name, cells, unit) in enumerate(self.rows):
            y = head_h + i * row_h
            dim = i in self.dim_rows
            if self.diagonal and i < self.n_cols:
                p.fillRect(QRectF(label_w + i * cell_w, y, cell_w, row_h), theme.qcolor(self.diagonal, 0.13))
            p.setFont(self._f_label)
            p.setPen(QColor(C.INK_3 if dim else C.INK_2))
            p.drawText(QRectF(0, y, label_w, row_h), int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter), name)
            p.setFont(self._f_cell)
            for j, cell in enumerate(cells):
                colour = self.cell_colours.get((i, j), C.INK_3 if dim or cell.strip() in ("·", "—") else C.INK)
                p.setPen(QColor(colour))
                p.drawText(QRectF(label_w + j * cell_w, y, cell_w - 2, row_h),
                           int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter), cell)
            if self.unit_col and unit:
                p.setFont(self._f_label)
                p.setPen(QColor(C.INK_3))
                p.drawText(QRectF(label_w + self.n_cols * cell_w + 4, y, unit_w, row_h),
                           int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter), unit)
        p.end()


# the collapsible section part:

class Collapsible(QWidget):
    """a section with a header button that shows or hides its body (progressive disclosure)."""

    toggled = Signal(bool)

    def __init__(self, title: str, body: QWidget, expanded: bool = False, tag: QWidget | None = None):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.setSpacing(round(0.4 * theme.em()))
        self.button = QToolButton()
        self.button.setObjectName("collapse")
        self.button.setText(title.upper())
        self.button.setFont(theme.font("section"))
        self.button.setCheckable(True)
        self.button.setChecked(expanded)
        self.button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.button.setAutoRaise(True)
        self.button.toggled.connect(self._toggle)
        head.addWidget(self.button)
        head.addStretch(1)
        if tag is not None:
            head.addWidget(tag)
        lay.addLayout(head)
        self.body = body
        lay.addWidget(body)
        self._toggle(expanded)

    def _toggle(self, on: bool) -> None:
        self.button.setIcon(icon("chev_down" if on else "chev_right", C.INK_3))
        self.body.setVisible(on)
        self.toggled.emit(on)


# the heat-map matrix part (equations panel):

class MatrixView(QWidget):
    """a matrix as a compact heat map, or as a numeric table when values are shown. the colour is a
    unit-free magnitude in [0, 1] computed by the caller (for a covariance |correlation|; for K, Phi and H the
    entry scaled by the standard deviations of its row and column), because the entries of P, K and Phi
    have mixed units and a raw |entry| would depend on the units chosen. the tooltip gives the exact entry
    with its unit. exact zeros stay empty, so the structure of the matrix shows."""

    def __init__(self, row_labels, col_labels, colour: str = C.ACCENT, show_values: bool = False):
        super().__init__()
        self.row_labels, self.col_labels = list(row_labels), list(col_labels)
        self.colour = colour
        self.show_values = show_values
        self.M = None
        self.N = None
        self.units = None  # (n_rows, n_cols) unit strings for the tooltip
        self.extra = None  # optional per-cell tooltip addition
        self.dim_rows: set[int] = set()
        self._f_label, self._f_cell = theme.font("caption"), theme.font("mono_small")
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.setMouseTracking(True)

    def set_data(self, M, N, units=None, extra=None, dim_rows=()) -> None:
        shape_changed = (self.M is None) != (M is None)
        self.M = None if M is None else np.asarray(M, dtype=float)
        self.N = None if N is None else np.asarray(N, dtype=float)
        self.units, self.extra, self.dim_rows = units, extra, set(dim_rows)
        if shape_changed:
            self.updateGeometry()
        self.update()

    def set_show_values(self, on: bool) -> None:
        self.show_values = on
        self.updateGeometry()
        self.update()

    def _metrics(self):
        from PySide6.QtGui import QFontMetricsF
        fl, fc = QFontMetricsF(self._f_label), QFontMetricsF(self._f_cell)
        label_w = max(fl.horizontalAdvance(t) for t in self.row_labels) + fl.horizontalAdvance("  ")
        if self.show_values:
            cell_w = fc.horizontalAdvance("0") * 9.0
            cell_h = fc.height() * 1.05
        else:
            cell_w = cell_h = 1.35 * theme.em("caption")
        head_h = fl.height() * 1.15
        return label_w, cell_w, cell_h, head_h

    def sizeHint(self):
        label_w, cell_w, cell_h, head_h = self._metrics()
        return QSize(math.ceil(label_w + len(self.col_labels) * cell_w) + 2,
                     math.ceil(head_h + len(self.row_labels) * cell_h) + 2)

    def minimumSizeHint(self):
        return self.sizeHint()

    def _cell_at(self, pos):
        label_w, cell_w, cell_h, head_h = self._metrics()
        j = math.floor((pos.x() - label_w) / cell_w)
        i = math.floor((pos.y() - head_h) / cell_h)
        if 0 <= i < len(self.row_labels) and 0 <= j < len(self.col_labels):
            return i, j
        return None

    def event(self, e):
        if e.type() == QEvent.Type.ToolTip and self.M is not None:
            cell = self._cell_at(e.position() if hasattr(e, "position") else e.pos())
            if cell is not None:
                from ekfsim_ui import fmt
                i, j = cell
                unit = self.units[i][j] if self.units is not None else ""
                text = f"[{self.row_labels[i]}, {self.col_labels[j]}] = {fmt.exact(self.M[i, j], unit)}"
                if self.extra is not None:
                    text += "\n" + self.extra(i, j)
                QToolTip.showText(e.globalPos(), text, self)
                return True
            QToolTip.hideText()
            e.ignore()
            return True
        return super().event(e)

    def paintEvent(self, event):
        from ekfsim_ui import fmt
        label_w, cell_w, cell_h, head_h = self._metrics()
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        p.setFont(self._f_label)
        p.setPen(QColor(C.INK_3))
        for j, t in enumerate(self.col_labels):
            p.drawText(QRectF(label_w + j * cell_w, 0, cell_w, head_h),
                       int(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignBottom), t)
        for i, t in enumerate(self.row_labels):
            p.setPen(QColor(C.INK_3 if i in self.dim_rows else C.INK_2))
            p.drawText(QRectF(0, head_h + i * cell_h, label_w, cell_h),
                       int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter), t)
        if self.M is None:
            p.end()
            return
        grid = QPen(theme.qcolor(C.HAIRLINE, 1.0), 1)
        p.setFont(self._f_cell)
        for i in range(self.M.shape[0]):
            for j in range(self.M.shape[1]):
                r = QRectF(label_w + j * cell_w, head_h + i * cell_h, cell_w - 1, cell_h - 1)
                v, n = self.M[i, j], (self.N[i, j] if self.N is not None else math.nan)
                if self.show_values:
                    p.setPen(QColor(C.INK_3 if (i in self.dim_rows or v == 0.0) else C.INK))
                    p.drawText(r.adjusted(0, 0, -3, 0), int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter),
                               fmt.sci_cell(v))
                    continue
                p.setPen(grid)
                if v != 0.0 and math.isfinite(n):
                    p.setBrush(theme.qcolor(self.colour, 0.06 + 0.88 * min(max(n, 0.0), 1.0)))
                else:
                    p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawRect(r)
        p.end()


# the flow layout part:

class FlowLayout(QLayout):
    """a layout that places its items left to right and wraps them to the next row when the width runs
    out (the Qt flow-layout example). cards keep their natural size, so nothing is squeezed."""

    def __init__(self, parent=None, spacing: float = 8.0):
        super().__init__(parent)
        self._items = []
        self._spacing = spacing

    def addItem(self, item):
        self._items.append(item)

    def count(self):
        return len(self._items)

    def itemAt(self, i):
        return self._items[i] if 0 <= i < len(self._items) else None

    def takeAt(self, i):
        return self._items.pop(i) if 0 <= i < len(self._items) else None

    def expandingDirections(self):
        return Qt.Orientation(0)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self._place(QRect(0, 0, width, 0), apply=False)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._place(rect, apply=True)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        size = QSize()
        for it in self._items:
            size = size.expandedTo(it.minimumSize())
        m = self.contentsMargins()
        return size + QSize(m.left() + m.right(), m.top() + m.bottom())

    def _place(self, rect: QRect, apply: bool) -> int:
        m = self.contentsMargins()
        r = rect.adjusted(m.left(), m.top(), -m.right(), -m.bottom())
        x, y, row_h = r.x(), r.y(), 0
        gap = round(self._spacing)
        for it in self._items:
            if it.widget() is not None and it.widget().isHidden():
                continue
            hint = it.sizeHint()
            if x + hint.width() > r.right() + 1 and row_h > 0:
                x, y, row_h = r.x(), y + row_h + gap, 0
            if apply:
                it.setGeometry(QRect(x, y, hint.width(), hint.height()))
            x += hint.width() + gap
            row_h = max(row_h, hint.height())
        return y + row_h - rect.y() + m.bottom()
