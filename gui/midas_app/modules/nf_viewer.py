"""NF Viewer module — embeds the full ``nf_qt.NFViewer`` as-is.

No functionality is removed. The legacy QMainWindow lives inside the unified
launcher's stacked widget.
"""

from __future__ import annotations
from PyQt5 import QtCore, QtWidgets

import nf_qt as _nf_qt  # noqa: E402


class NFViewerModule(QtWidgets.QWidget):
    def __init__(self, theme: str = 'light', parent=None):
        super().__init__(parent)
        self._viewer = _nf_qt.NFViewer(theme=theme)
        self._viewer.setWindowFlags(QtCore.Qt.Widget)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._viewer)

    def open_directory(self, path: str) -> None:
        v = self._viewer
        if hasattr(v, 'folder'):
            v.folder = path
        for hook in ('_start_auto_detect', '_auto_detect', '_run_auto_detect'):
            if hasattr(v, hook):
                try:
                    getattr(v, hook)()
                except Exception as e:
                    print(f"[NFViewer] {hook}() failed: {e}")
                break

    @property
    def viewer(self):
        return self._viewer
