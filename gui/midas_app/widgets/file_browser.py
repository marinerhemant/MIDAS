"""Directory file-list sidebar with extension filtering (PyQt5)."""

from __future__ import annotations
import os
from typing import Iterable, Optional

from PyQt5 import QtCore, QtWidgets


class FileBrowser(QtWidgets.QWidget):
    fileSelected = QtCore.pyqtSignal(str)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None,
                 patterns: Optional[Iterable[str]] = None):
        super().__init__(parent)
        self._dir: Optional[str] = None
        self._patterns = list(patterns) if patterns else None

        self._dir_label = QtWidgets.QLabel("(no directory)")
        self._dir_label.setStyleSheet("font-style: italic; color: #888;")
        self._dir_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self._dir_label.setWordWrap(True)

        browse = QtWidgets.QPushButton("Choose folder…")
        browse.clicked.connect(self._on_browse)

        refresh = QtWidgets.QPushButton("⟳")
        refresh.setFixedWidth(32)
        refresh.setToolTip("Refresh listing")
        refresh.clicked.connect(self.refresh)

        self._list = QtWidgets.QListWidget()
        self._list.itemActivated.connect(self._on_item_activated)
        self._list.itemDoubleClicked.connect(self._on_item_activated)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        top = QtWidgets.QHBoxLayout()
        top.addWidget(browse)
        top.addWidget(refresh)
        layout.addLayout(top)
        layout.addWidget(self._dir_label)
        layout.addWidget(self._list, stretch=1)

    def set_directory(self, path: str):
        if not path or not os.path.isdir(path):
            return
        self._dir = path
        self._dir_label.setText(path)
        self.refresh()

    def set_patterns(self, patterns: Iterable[str]):
        self._patterns = list(patterns)
        self.refresh()

    def refresh(self):
        self._list.clear()
        if not self._dir:
            return
        try:
            entries = sorted(os.listdir(self._dir))
        except OSError as e:
            self._list.addItem(f"<error: {e}>")
            return
        for name in entries:
            full = os.path.join(self._dir, name)
            if os.path.isdir(full):
                continue
            if self._patterns and not _matches_any(name, self._patterns):
                continue
            self._list.addItem(name)

    def current_path(self) -> Optional[str]:
        item = self._list.currentItem()
        return os.path.join(self._dir, item.text()) if item and self._dir else None

    def _on_browse(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose folder", self._dir or "")
        if d:
            self.set_directory(d)

    def _on_item_activated(self, item: QtWidgets.QListWidgetItem):
        if self._dir is None:
            return
        self.fileSelected.emit(os.path.join(self._dir, item.text()))


def _matches_any(name: str, patterns: Iterable[str]) -> bool:
    import fnmatch
    return any(fnmatch.fnmatch(name, p) for p in patterns)
