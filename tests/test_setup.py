"""setup checks: every module imports, the core stays free of Qt, and the UI stack is installed."""

import importlib
import os
import pathlib
import re
import subprocess
import sys

import pytest

SRC_DIR = pathlib.Path(__file__).resolve().parents[1] / "src"
CORE_MODULES = [
    "ekfsim", "ekfsim.config", "ekfsim.atmosphere", "ekfsim.aero", "ekfsim.vehicle",
    "ekfsim.integrators", "ekfsim.radar", "ekfsim.ekf", "ekfsim.scenario",
    "ekfsim.metrics", "ekfsim.montecarlo",
]
UI_MODULES = [
    "ekfsim_ui", "ekfsim_ui.app", "ekfsim_ui.main_window", "ekfsim_ui.view3d",
    "ekfsim_ui.plots", "ekfsim_ui.equations_panel", "ekfsim_ui.controls", "ekfsim_ui.playback",
    "ekfsim_ui.theme", "ekfsim_ui.timeline", "ekfsim_ui.fmt", "ekfsim_ui.widgets", "ekfsim_ui.readout",
]
QT_PACKAGES = ("PySide6", "shiboken6", "PyQt5", "PyQt6", "pyqtgraph")


def test_python_version():
    assert sys.version_info >= (3, 12)


@pytest.mark.parametrize("name", CORE_MODULES + UI_MODULES)
def test_module_imports(name):
    importlib.import_module(name)


def test_core_loads_no_qt_at_runtime():
    """import the whole core in a fresh interpreter and check that no Qt module got loaded,
    also not indirectly through another library."""
    code = (
        "import importlib, sys\n"
        f"for m in {CORE_MODULES!r}: importlib.import_module(m)\n"
        f"print(sorted(m for m in sys.modules if m.split('.')[0] in {QT_PACKAGES!r}))"
    )
    env = {**os.environ, "PYTHONPATH": str(SRC_DIR)}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         check=True, env=env)
    assert out.stdout.strip() == "[]"


def test_core_source_has_no_qt_import():
    """static check that also catches imports hidden inside functions."""
    pattern = re.compile(r"^\s*(import|from)\s+(" + "|".join(QT_PACKAGES) + r")\b", re.M)
    offenders = [p.name for p in (SRC_DIR / "ekfsim").rglob("*.py")
                 if pattern.search(p.read_text(encoding="utf-8"))]
    assert offenders == []


def test_ui_stack_installed():
    import OpenGL.GL  # noqa: F401
    import pyqtgraph
    import pyqtgraph.opengl  # noqa: F401
    from PySide6 import QtWidgets  # noqa: F401

    assert pyqtgraph.__version__.startswith("0.14.")
