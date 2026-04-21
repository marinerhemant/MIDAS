"""Image Tools module — launches gui/imageManipulation.py in a separate process.

imageManipulation.py is a Tkinter app and cannot share a Qt event loop, so it
runs as a subprocess. The user gets a launch button and a list of running
instances.
"""

from __future__ import annotations
import os

from PyQt5 import QtWidgets

from ..widgets.external_launcher import ExternalLauncher

_GUI_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SCRIPT = os.path.join(_GUI_DIR, 'imageManipulation.py')


class ImageToolsModule(QtWidgets.QWidget):
    def __init__(self, theme: str = 'light', parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        launcher = ExternalLauncher(
            title="Image Tools — imageManipulation.py",
            description=(
                "Multi-detector GE1–GE5 image processor with dark subtraction, "
                "bad-pixel masking, and method-1 vs method-2 reconstruction "
                "comparison. The script is a Tkinter app and cannot share a Qt "
                "event loop, so it runs in a separate process. You can launch "
                "multiple instances independently."),
            script_path=_SCRIPT,
            args=[],  # imageManipulation.py reads everything from its own UI
            parent=self,
        )
        layout.addWidget(launcher)
