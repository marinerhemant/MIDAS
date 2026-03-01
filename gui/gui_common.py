#!/usr/bin/env python3
"""
MIDAS GUI Common Utilities
Shared components for PyQt5 + PyQtGraph based viewers.

Provides:
  - apply_theme()       : Dark/light palette for Qt + PyQtGraph
  - MIDASImageView      : ImageView subclass with crosshair, colormap, status, wheel-nav
  - AsyncWorker         : QThread wrapper for background tasks
  - LogPanel            : Collapsible log output widget
"""

import sys
import os
import time
import numpy as np

from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg

# ── Constants ──────────────────────────────────────────────────────────
COLORMAPS = ['viridis', 'inferno', 'plasma', 'magma', 'turbo',
             'gray', 'gray_r', 'hot', 'cool', 'bone']


# ── Theme ──────────────────────────────────────────────────────────────
def apply_theme(app, theme='light'):
    """Apply dark or light theme to the Qt application and PyQtGraph."""
    app.setStyle('Fusion')
    if theme == 'dark':
        pal = QtGui.QPalette()
        pal.setColor(QtGui.QPalette.Window, QtGui.QColor(30, 30, 30))
        pal.setColor(QtGui.QPalette.WindowText, QtGui.QColor(220, 220, 220))
        pal.setColor(QtGui.QPalette.Base, QtGui.QColor(25, 25, 25))
        pal.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(40, 40, 40))
        pal.setColor(QtGui.QPalette.Text, QtGui.QColor(220, 220, 220))
        pal.setColor(QtGui.QPalette.Button, QtGui.QColor(50, 50, 50))
        pal.setColor(QtGui.QPalette.ButtonText, QtGui.QColor(220, 220, 220))
        pal.setColor(QtGui.QPalette.Highlight, QtGui.QColor(42, 130, 218))
        app.setPalette(pal)
        pg.setConfigOptions(background='k', foreground='w')
    else:
        app.setPalette(app.style().standardPalette())
        pg.setConfigOptions(background='w', foreground='k')


def get_colormap(name):
    """Get a pyqtgraph ColorMap by name, with matplotlib fallback."""
    try:
        return pg.colormap.get(name)
    except Exception:
        try:
            return pg.colormap.getFromMatplotlib(name)
        except Exception:
            return pg.colormap.get('viridis')


# ── MIDASImageView ─────────────────────────────────────────────────────
class MIDASImageView(pg.ImageView):
    """
    Enhanced ImageView with:
      - Crosshair overlay with coordinate tracking
      - Colormap dropdown
      - Mouse-wheel frame navigation signal
      - Log-scale display
      - Export-to-PNG action
      - Status bar signal for cursor position + pixel value
    """

    # Emitted when cursor moves: (x, y, pixel_value)
    cursorMoved = QtCore.pyqtSignal(float, float, float)
    # Emitted when mouse wheel changes frame: delta (+1 or -1)
    frameScrolled = QtCore.pyqtSignal(int)
    # Emitted when image data changes: (min, max, p2, p98)
    dataStatsUpdated = QtCore.pyqtSignal(float, float, float, float)

    def __init__(self, parent=None, name='MIDASImageView', **kwargs):
        super().__init__(parent=parent, name=name, view=pg.PlotItem(), **kwargs)

        self._raw_data = None
        self._log_mode = False

        # Remove the default ROI and Norm buttons for a cleaner look
        self.ui.roiBtn.hide()
        self.ui.menuBtn.hide()

        # ── Crosshair ──
        self._vline = pg.InfiniteLine(angle=90, movable=False,
                                       pen=pg.mkPen('y', width=1, style=QtCore.Qt.DashLine))
        self._hline = pg.InfiniteLine(angle=0, movable=False,
                                       pen=pg.mkPen('y', width=1, style=QtCore.Qt.DashLine))
        self._vline.setZValue(1000)
        self._hline.setZValue(1000)
        self.addItem(self._vline)
        self.addItem(self._hline)
        self._crosshair_visible = True

        # Track mouse
        self._proxy = pg.SignalProxy(self.scene.sigMouseMoved, rateLimit=60,
                                      slot=self._on_mouse_moved)

        # ── Overlay items (rings, annotations, etc.) ──
        self._overlay_items = []

    # ── Public API ──────────────────────────────────────────────────

    def set_image_data(self, data, auto_levels=True, levels=None):
        """Set image data with smart percentile-based auto-levels."""
        self._raw_data = data
        display = self._apply_log(data) if self._log_mode else data

        # Compute stats
        finite = display[np.isfinite(display)]
        if finite.size > 0:
            dmin, dmax = float(finite.min()), float(finite.max())
            p2 = float(np.percentile(finite, 2))
            p98 = float(np.percentile(finite, 98))
        else:
            dmin = dmax = p2 = p98 = 0.0
        self.dataStatsUpdated.emit(dmin, dmax, p2, p98)

        if levels is not None:
            if self._log_mode:
                levels = (np.log10(max(levels[0], 1e-10)),
                          np.log10(max(levels[1], 1e-10)))
            self.setImage(display, autoLevels=False, levels=levels)
        elif auto_levels:
            # Smart levels: use 2nd-98th percentile to avoid hot pixel domination
            self.setImage(display, autoLevels=False, levels=(p2, p98))
        else:
            self.setImage(display, autoLevels=False)

        # Match matplotlib's invert_yaxis(): row 0 at top visually,
        # but axis labels read 0 at bottom, increasing upward.
        self.getView().invertY(True)

    def set_log_mode(self, enabled):
        """Toggle log10 display."""
        self._log_mode = enabled
        if self._raw_data is not None:
            self.set_image_data(self._raw_data)

    def set_colormap(self, name):
        """Apply a named colormap."""
        cmap = get_colormap(name)
        lut = cmap.getLookupTable(nPts=256)
        self.imageItem.setLookupTable(lut)

    def set_crosshair_visible(self, visible):
        """Show or hide crosshair."""
        self._crosshair_visible = visible
        self._vline.setVisible(visible)
        self._hline.setVisible(visible)

    def add_overlay(self, item):
        """Add a PlotItem overlay (rings, markers, etc.)."""
        self.addItem(item)
        self._overlay_items.append(item)

    def clear_overlays(self):
        """Remove all overlay items."""
        for item in self._overlay_items:
            self.removeItem(item)
        self._overlay_items.clear()

    def export_png(self, filename=None):
        """Export current view to PNG."""
        if filename is None:
            filename, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, 'Export Image', '', 'PNG Files (*.png);;All Files (*)')
        if filename:
            exporter = pg.exporters.ImageExporter(self.scene)
            exporter.export(filename)

    # ── Internal ────────────────────────────────────────────────────

    def _apply_log(self, data):
        """Apply log10 to data for display."""
        return np.log10(np.clip(data.astype(np.float64), 1e-10, None))

    def _on_mouse_moved(self, evt):
        pos = evt[0]
        vb = self.getView()
        vbox = vb.getViewBox()
        if vb.sceneBoundingRect().contains(pos):
            mouse_point = vbox.mapSceneToView(pos)
            x, y = mouse_point.x(), mouse_point.y()
            self._vline.setPos(x)
            self._hline.setPos(y)

            # Get pixel value
            val = 0.0
            if self._raw_data is not None:
                ix, iy = int(x + 0.5), int(y + 0.5)
                h, w = self._raw_data.shape[:2]
                if 0 <= iy < h and 0 <= ix < w:
                    val = float(self._raw_data[iy, ix])
            self.cursorMoved.emit(x, y, val)

    def wheelEvent(self, ev):
        """Emit frame scroll signal on Ctrl+wheel, else default zoom."""
        if ev.modifiers() & QtCore.Qt.ControlModifier:
            delta = 1 if ev.angleDelta().y() > 0 else -1
            self.frameScrolled.emit(delta)
            ev.accept()
        else:
            super().wheelEvent(ev)


# ── AsyncWorker ────────────────────────────────────────────────────────
class AsyncWorker(QtCore.QThread):
    """
    Generic background worker thread.

    Usage:
        worker = AsyncWorker(target=my_function, args=(arg1, arg2))
        worker.finished_signal.connect(on_done)
        worker.start()
    """
    finished_signal = QtCore.pyqtSignal(object)
    error_signal = QtCore.pyqtSignal(str)
    progress_signal = QtCore.pyqtSignal(int, int)  # current, total

    def __init__(self, target=None, args=(), parent=None):
        super().__init__(parent)
        self._target = target
        self._args = args

    def run(self):
        try:
            result = self._target(*self._args)
            self.finished_signal.emit(result)
        except Exception as e:
            self.error_signal.emit(str(e))


# ── LogPanel ───────────────────────────────────────────────────────────
class LogPanel(QtWidgets.QDockWidget):
    """
    Collapsible log output panel. Captures print statements
    when installed as sys.stdout redirect.
    """

    def __init__(self, parent=None, title='Log'):
        super().__init__(title, parent)
        self.setAllowedAreas(QtCore.Qt.BottomDockWidgetArea)

        self._text = QtWidgets.QPlainTextEdit()
        self._text.setReadOnly(True)
        self._text.setMaximumBlockCount(5000)
        font = QtGui.QFont('Consolas', 10)
        font.setStyleHint(QtGui.QFont.Monospace)
        self._text.setFont(font)
        self.setWidget(self._text)

        self._original_stdout = None

    def install_redirect(self):
        """Redirect sys.stdout to this panel (keeps terminal output too)."""
        self._original_stdout = sys.stdout
        sys.stdout = self

    def uninstall_redirect(self):
        """Restore original stdout."""
        if self._original_stdout:
            sys.stdout = self._original_stdout

    def write(self, text):
        if text.strip():
            self._text.appendPlainText(text.rstrip('\n'))
        if self._original_stdout:
            self._original_stdout.write(text)

    def flush(self):
        if self._original_stdout:
            self._original_stdout.flush()


# ── Keyboard Shortcut Helper ──────────────────────────────────────────
def add_shortcut(parent, key, callback, context=QtCore.Qt.WindowShortcut):
    """Add a keyboard shortcut to a widget."""
    shortcut = QtWidgets.QShortcut(QtGui.QKeySequence(key), parent)
    shortcut.setContext(context)
    shortcut.activated.connect(callback)
    return shortcut
