"""shared fixtures. the UI tests run Qt on the offscreen platform, so they need no display."""

import os

import pytest


@pytest.fixture(scope="session")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from ekfsim_ui.app import create_application
    return create_application(["pytest"])


@pytest.fixture(scope="session")
def default_timeline():
    from ekfsim_ui.timeline import RunRequest, compute
    return compute(RunRequest())
