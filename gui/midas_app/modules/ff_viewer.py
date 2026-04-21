"""FF Viewer module — embeds the full ``ff_asym_qt.FFViewer`` as-is.

No functionality is removed. The legacy QMainWindow lives inside the unified
launcher's stacked widget; its menus/toolbars/status bar render inside the
tab area.
"""

from __future__ import annotations
from PyQt5 import QtCore, QtWidgets

# ff_asym_qt lives in gui/ which midas_gui.py adds to sys.path.
import ff_asym_qt as _ff_asym_qt  # noqa: E402


class FFViewerModule(QtWidgets.QWidget):
    def __init__(self, theme: str = 'light', parent=None):
        super().__init__(parent)
        self._viewer = _ff_asym_qt.FFViewer(theme=theme)
        # Strip the title-bar window flags so the embedded QMainWindow plays
        # nicely as a child widget.
        self._viewer.setWindowFlags(QtCore.Qt.Widget)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._viewer)

    def open_directory(self, path: str) -> None:
        """Forward an Open Directory request to the underlying viewer."""
        v = self._viewer
        if hasattr(v, 'folder'):
            v.folder = path.rstrip('/') + '/'
        # Re-run auto-detection if the viewer exposes a hook for it.
        for hook in ('_start_auto_detect', '_auto_detect', '_run_auto_detect'):
            if hasattr(v, hook):
                try:
                    getattr(v, hook)()
                except Exception as e:
                    print(f"[FFViewer] {hook}() failed: {e}")
                break

    @property
    def viewer(self):
        return self._viewer
