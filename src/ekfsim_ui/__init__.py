"""ekfsim_ui is the PySide6 + pyqtgraph desktop UI. it is the only package that imports Qt."""

import OpenGL

# PyOpenGL checks glGetError after every call by default; that cost a fifth of the frame time while no UI
# logic depends on OpenGL error codes, so it is switched off (the documented production setting). this must
# happen before OpenGL.GL is imported anywhere.
OpenGL.ERROR_CHECKING = False
