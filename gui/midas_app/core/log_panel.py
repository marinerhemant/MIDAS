"""Collapsible log dock that captures stdout (PyQt5)."""

import sys
from PyQt5 import QtCore, QtGui, QtWidgets


class LogPanel(QtWidgets.QDockWidget):
    def __init__(self, parent=None, title='Log'):
        super().__init__(title, parent)
        self.setAllowedAreas(QtCore.Qt.BottomDockWidgetArea)

        self._text = QtWidgets.QPlainTextEdit()
        self._text.setReadOnly(True)
        self._text.setMaximumBlockCount(5000)
        font = QtGui.QFont('Menlo', 10)
        font.setStyleHint(QtGui.QFont.Monospace)
        self._text.setFont(font)
        self.setWidget(self._text)

        self._original_stdout = None

    def install_redirect(self):
        self._original_stdout = sys.stdout
        sys.stdout = self

    def uninstall_redirect(self):
        if self._original_stdout:
            sys.stdout = self._original_stdout
            self._original_stdout = None

    def write(self, text):
        if text.strip():
            self._text.appendPlainText(text.rstrip('\n'))
        if self._original_stdout:
            self._original_stdout.write(text)

    def flush(self):
        if self._original_stdout:
            self._original_stdout.flush()
