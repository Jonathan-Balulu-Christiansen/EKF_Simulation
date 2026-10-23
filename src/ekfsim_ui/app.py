"""entry point of the desktop UI (THEORY 8.1): python -m ekfsim_ui. it requests 4x MSAA for the OpenGL view
before the QApplication exists, keeps fractional screen scaling exact (150 % stays 1.5, not rounded to 2),
writes logs/ekfsim.log under the current working directory, and installs a global exception hook that logs
the traceback and shows it in a dialog instead of letting the window die silently."""

import logging
import pathlib
import sys
import traceback


def setup_logging() -> pathlib.Path:
    """the log file logs/ekfsim.log in the current working directory, not in the installed package."""
    log_dir = pathlib.Path.cwd() / "logs"
    log_dir.mkdir(exist_ok=True)
    path = log_dir / "ekfsim.log"
    logger = logging.getLogger("ekfsim")
    if not logger.handlers:
        h = logging.FileHandler(path, encoding="utf-8")
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
    return path


def install_exception_hook(log_path: pathlib.Path) -> None:
    from PySide6.QtWidgets import QApplication, QMessageBox

    def hook(exc_type, exc, tb):
        text = "".join(traceback.format_exception(exc_type, exc, tb))
        logging.getLogger("ekfsim").error("uncaught exception\n%s", text)
        if QApplication.instance() is not None:
            box = QMessageBox(QMessageBox.Icon.Critical, "EKF tracker: unexpected error",
                              f"{exc_type.__name__}: {exc}\n\nThe full traceback is in {log_path}.")
            box.setDetailedText(text)
            box.exec()
        else:
            sys.__excepthook__(exc_type, exc, tb)

    sys.excepthook = hook


def create_application(argv=None):
    """the QApplication with the surface format, the DPI policy and the theme (also used by scripts)."""
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QGuiApplication, QSurfaceFormat
    from PySide6.QtWidgets import QApplication

    from ekfsim_ui import theme

    fmt = QSurfaceFormat()
    fmt.setSamples(4)  # MSAA, set before the QApplication exists (THEORY 8.1)
    fmt.setDepthBufferSize(24)
    QSurfaceFormat.setDefaultFormat(fmt)
    QGuiApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication.instance() or QApplication(list(argv or sys.argv))
    app.setApplicationName("EKF ballistic target tracker")
    theme.apply(app)
    return app


def main(argv=None) -> int:
    log_path = setup_logging()
    install_exception_hook(log_path)
    app = create_application(argv)
    from ekfsim_ui.main_window import MainWindow

    window = MainWindow()
    screen = app.primaryScreen().availableGeometry()
    window.resize(round(0.85 * screen.width()), round(0.85 * screen.height()))
    window.show()
    window.start_run()
    return app.exec()
