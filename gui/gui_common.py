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
class MIDASImageView(QtWidgets.QWidget):
    """
    Enhanced ImageView wrapper with:
      - Crosshair overlay with coordinate tracking
      - Colormap dropdown
      - Mouse-wheel frame navigation signal
      - Log-scale display
      - Export-to-PNG action
      - Navigation toolbar (Home, Back, Forward, Pan, Zoom-to-rect)
      - Status bar signal for cursor position + pixel value
    """

    # Emitted when cursor moves: (x, y, pixel_value)
    cursorMoved = QtCore.pyqtSignal(float, float, float)
    # Emitted when mouse wheel changes frame: delta (+1 or -1)
    frameScrolled = QtCore.pyqtSignal(int)
    # Emitted when image data changes: (min, max, p2, p98)
    dataStatsUpdated = QtCore.pyqtSignal(float, float, float, float)
    # Emitted for movie mode: advance one frame
    movieFrameAdvance = QtCore.pyqtSignal()
    # Emitted when a file is dropped onto the viewer
    fileDropped = QtCore.pyqtSignal(str)

    def __init__(self, parent=None, name='MIDASImageView', origin='bl', **kwargs):
        super().__init__(parent)

        self._raw_data = None
        self._log_mode = False
        self._origin = origin  # 'bl' = bottom-left, 'br' = bottom-right

        # ── Internal ImageView ──
        self._iv = pg.ImageView(parent=self, name=name, view=pg.PlotItem(), **kwargs)
        self._iv.ui.roiBtn.hide()
        self._iv.ui.menuBtn.hide()

        # ── Navigation state ──
        self._nav_mode = 'pointer'  # 'pointer', 'pan', 'zoom'
        self._view_history = []
        self._view_index = -1

        # ── Crosshair ──
        self._vline = pg.InfiniteLine(angle=90, movable=False,
                                       pen=pg.mkPen('y', width=1, style=QtCore.Qt.DashLine))
        self._hline = pg.InfiniteLine(angle=0, movable=False,
                                       pen=pg.mkPen('y', width=1, style=QtCore.Qt.DashLine))
        self._vline.setZValue(1000)
        self._hline.setZValue(1000)
        self._iv.addItem(self._vline)
        self._iv.addItem(self._hline)
        self._crosshair_visible = True

        # Track mouse for crosshair
        self._proxy = pg.SignalProxy(self._iv.scene.sigMouseMoved, rateLimit=60,
                                      slot=self._on_mouse_moved)

        # ── Overlay items (rings, annotations, etc.) ──
        self._overlay_items = []

        # ── Navigation Toolbar ──
        self._nav_bar = self._build_nav_bar()

        # ── Layout: image view + nav bar ──
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._iv, stretch=1)
        layout.addWidget(self._nav_bar)

        # Disable mouse interaction by default (avoids expensive redraws over SSH)
        # Pan/Zoom only enabled via toolbar buttons
        vb = self._get_viewbox()
        vb.setMouseEnabled(x=False, y=False)
        vb.enableAutoRange(False)
        # Disable scroll-to-zoom (only rectangle-drag zoom via toolbar)
        vb.wheelScaleFactor = 0

        # Install event filter for zoom-rect and pan
        self._iv.scene.sigMouseClicked.connect(self._on_scene_clicked)

        # Enable drag-and-drop
        self.setAcceptDrops(True)

    # ── Navigation Toolbar ─────────────────────────────────────────

    def _build_nav_bar(self):
        bar = QtWidgets.QToolBar()
        bar.setIconSize(QtCore.QSize(20, 20))
        bar.setMovable(False)
        bar.setStyleSheet("QToolBar { spacing: 4px; padding: 2px; }")

        style = QtWidgets.QApplication.style()

        # Home
        self._home_btn = QtWidgets.QToolButton()
        self._home_btn.setIcon(style.standardIcon(QtWidgets.QStyle.SP_DirHomeIcon))
        self._home_btn.setToolTip("Home – Reset to full view")
        self._home_btn.clicked.connect(self._nav_home)
        bar.addWidget(self._home_btn)

        # Back
        self._back_btn = QtWidgets.QToolButton()
        self._back_btn.setIcon(style.standardIcon(QtWidgets.QStyle.SP_ArrowBack))
        self._back_btn.setToolTip("Back – Previous view")
        self._back_btn.clicked.connect(self._nav_back)
        self._back_btn.setEnabled(False)
        bar.addWidget(self._back_btn)

        # Forward
        self._fwd_btn = QtWidgets.QToolButton()
        self._fwd_btn.setIcon(style.standardIcon(QtWidgets.QStyle.SP_ArrowForward))
        self._fwd_btn.setToolTip("Forward – Next view")
        self._fwd_btn.clicked.connect(self._nav_forward)
        self._fwd_btn.setEnabled(False)
        bar.addWidget(self._fwd_btn)

        bar.addSeparator()

        # Pan
        self._pan_btn = QtWidgets.QToolButton()
        self._pan_btn.setIcon(style.standardIcon(QtWidgets.QStyle.SP_FileDialogDetailedView))
        self._pan_btn.setToolTip("Pan – Drag to move view")
        self._pan_btn.setCheckable(True)
        self._pan_btn.clicked.connect(lambda: self._set_nav_mode('pan'))
        bar.addWidget(self._pan_btn)

        # Zoom
        self._zoom_btn = QtWidgets.QToolButton()
        self._zoom_btn.setIcon(style.standardIcon(QtWidgets.QStyle.SP_FileDialogContentsView))
        self._zoom_btn.setToolTip("Zoom – Drag rectangle to zoom")
        self._zoom_btn.setCheckable(True)
        self._zoom_btn.clicked.connect(lambda: self._set_nav_mode('zoom'))
        bar.addWidget(self._zoom_btn)

        bar.addSeparator()

        # ── Movie controls ──
        self._play_btn = QtWidgets.QToolButton()
        self._play_btn.setIcon(style.standardIcon(QtWidgets.QStyle.SP_MediaPlay))
        self._play_btn.setToolTip("Play – Animate frames")
        self._play_btn.clicked.connect(self._movie_play)
        bar.addWidget(self._play_btn)

        self._pause_btn = QtWidgets.QToolButton()
        self._pause_btn.setIcon(style.standardIcon(QtWidgets.QStyle.SP_MediaPause))
        self._pause_btn.setToolTip("Pause – Pause animation")
        self._pause_btn.clicked.connect(self._movie_pause)
        self._pause_btn.setEnabled(False)
        bar.addWidget(self._pause_btn)

        self._stop_btn = QtWidgets.QToolButton()
        self._stop_btn.setIcon(style.standardIcon(QtWidgets.QStyle.SP_MediaStop))
        self._stop_btn.setToolTip("Stop – Stop and reset to first frame")
        self._stop_btn.clicked.connect(self._movie_stop)
        self._stop_btn.setEnabled(False)
        bar.addWidget(self._stop_btn)

        fps_label = QtWidgets.QLabel("  FPS:")
        bar.addWidget(fps_label)
        self._fps_spin = QtWidgets.QSpinBox()
        self._fps_spin.setRange(1, 30)
        self._fps_spin.setValue(5)
        self._fps_spin.setToolTip("Frames per second for animation")
        self._fps_spin.valueChanged.connect(self._update_movie_timer)
        self._fps_spin.setFixedWidth(50)
        bar.addWidget(self._fps_spin)

        bar.addSeparator()

        # Mode label
        self._mode_label = QtWidgets.QLabel("  Mode: Pointer")
        bar.addWidget(self._mode_label)

        # ── Movie timer ──
        self._movie_timer = QtCore.QTimer(self)
        self._movie_timer.timeout.connect(self._movie_tick)

        return bar

    def _set_nav_mode(self, mode):
        """Set navigation mode: 'pointer', 'pan', or 'zoom'."""
        if self._nav_mode == mode:
            mode = 'pointer'  # Toggle off

        self._nav_mode = mode
        vb = self._get_viewbox()

        self._pan_btn.setChecked(mode == 'pan')
        self._zoom_btn.setChecked(mode == 'zoom')

        if mode == 'pan':
            vb.setMouseEnabled(x=True, y=True)
            vb.setMouseMode(pg.ViewBox.PanMode)
            self._mode_label.setText("  Mode: Pan")
        elif mode == 'zoom':
            vb.setMouseEnabled(x=True, y=True)
            vb.setMouseMode(pg.ViewBox.RectMode)
            self._mode_label.setText("  Mode: Zoom")
        else:
            vb.setMouseEnabled(x=False, y=False)
            self._mode_label.setText("  Mode: Pointer")

    def _push_view(self):
        """Save current view range to history stack."""
        vb = self._get_viewbox()
        xr = vb.viewRange()[0]
        yr = vb.viewRange()[1]
        entry = (list(xr), list(yr))
        # Trim forward history
        if self._view_index < len(self._view_history) - 1:
            self._view_history = self._view_history[:self._view_index + 1]
        self._view_history.append(entry)
        self._view_index = len(self._view_history) - 1
        self._update_nav_buttons()

    def _nav_home(self):
        """Reset view to full image extent."""
        self._push_view()
        vb = self._get_viewbox()
        vb.autoRange()
        self._push_view()

    def _nav_back(self):
        if self._view_index > 0:
            self._view_index -= 1
            xr, yr = self._view_history[self._view_index]
            vb = self._get_viewbox()
            vb.setRange(xRange=xr, yRange=yr, padding=0)
            self._update_nav_buttons()

    def _nav_forward(self):
        if self._view_index < len(self._view_history) - 1:
            self._view_index += 1
            xr, yr = self._view_history[self._view_index]
            vb = self._get_viewbox()
            vb.setRange(xRange=xr, yRange=yr, padding=0)
            self._update_nav_buttons()

    def _update_nav_buttons(self):
        self._back_btn.setEnabled(self._view_index > 0)
        self._fwd_btn.setEnabled(self._view_index < len(self._view_history) - 1)

    def _on_scene_clicked(self, ev):
        """Double-click resets zoom to full view."""
        if ev.double():
            self._nav_home()

    # ── Movie Controls ─────────────────────────────────────────────

    def _movie_play(self):
        """Start frame animation."""
        fps = self._fps_spin.value()
        self._movie_timer.start(int(1000 / fps))
        self._play_btn.setEnabled(False)
        self._pause_btn.setEnabled(True)
        self._stop_btn.setEnabled(True)
        self._mode_label.setText(f"  ▶ Playing ({fps} fps)")

    def _movie_pause(self):
        """Pause frame animation."""
        self._movie_timer.stop()
        self._play_btn.setEnabled(True)
        self._pause_btn.setEnabled(False)
        self._mode_label.setText("  ⏸ Paused")

    def _movie_stop(self):
        """Stop frame animation."""
        self._movie_timer.stop()
        self._play_btn.setEnabled(True)
        self._pause_btn.setEnabled(False)
        self._stop_btn.setEnabled(False)
        self._mode_label.setText("  Mode: Pointer")

    def _movie_tick(self):
        """Advance one frame in movie mode."""
        self.movieFrameAdvance.emit()

    def _update_movie_timer(self, fps):
        """Update timer interval when FPS changes."""
        if self._movie_timer.isActive():
            self._movie_timer.setInterval(int(1000 / max(1, fps)))
            self._mode_label.setText(f"  ▶ Playing ({fps} fps)")

    # ── Drag-and-Drop ──────────────────────────────────────────────

    def dragEnterEvent(self, ev):
        if ev.mimeData().hasUrls():
            ev.acceptProposedAction()
        else:
            ev.ignore()

    def dropEvent(self, ev):
        urls = ev.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            self.fileDropped.emit(path)

    # ── Public API ──────────────────────────────────────────────────

    def set_image_data(self, data, auto_levels=True, levels=None):
        """Set image data with smart percentile-based auto-levels."""
        self._raw_data = data
        # Transpose: PyQtGraph maps axis-0→X, axis-1→Y, but numpy images
        # are (rows, cols). Transpose so rows→Y (vertical), cols→X (horizontal).
        display = self._apply_log(data.T) if self._log_mode else data.T

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
            self._iv.setImage(display, autoLevels=False, levels=levels)
        elif auto_levels:
            self._iv.setImage(display, autoLevels=False, levels=(p2, p98))
        else:
            self._iv.setImage(display, autoLevels=False)

        # Force origin position AFTER setImage (which may reset axes)
        vb = self._iv.getView()
        if self._origin == 'bl':
            vb.invertY(False)
            vb.invertX(False)
        elif self._origin == 'br':
            vb.invertY(False)
            vb.invertX(True)

        # Push initial view to history
        self._push_view()

    def set_log_mode(self, enabled):
        """Toggle log10 display, preserving current view range and intensity levels."""
        self._log_mode = enabled
        if self._raw_data is not None:
            # Save current view range and levels before re-applying
            vb = self._iv.getView()
            saved_range = vb.viewRange()  # [[xmin, xmax], [ymin, ymax]]
            current_levels = self._iv.getLevels()
            self.set_image_data(self._raw_data, auto_levels=False)
            # Restore view range (prevents zoom reset)
            vb.setRange(xRange=saved_range[0], yRange=saved_range[1], padding=0)
            # Re-apply levels (converted to/from log space)
            if current_levels is not None:
                lo, hi = current_levels
                if enabled:
                    # Convert linear levels to log space
                    lo = np.log10(max(lo, 1e-10))
                    hi = np.log10(max(hi, 1e-10))
                else:
                    # Convert log levels back to linear space
                    lo = 10.0 ** lo
                    hi = 10.0 ** hi
                self._iv.setLevels(lo, hi)

    def set_colormap(self, name):
        """Apply a named colormap."""
        cmap = get_colormap(name)
        lut = cmap.getLookupTable(nPts=256)
        self._iv.imageItem.setLookupTable(lut)

    def set_crosshair_visible(self, visible):
        """Show or hide crosshair."""
        self._crosshair_visible = visible
        self._vline.setVisible(visible)
        self._hline.setVisible(visible)

    def setLevels(self, lo, hi):
        """Set intensity levels (proxied to internal ImageView)."""
        self._iv.setLevels(lo, hi)

    def add_overlay(self, item):
        """Add a PlotItem overlay (rings, markers, etc.)."""
        self._iv.addItem(item)
        self._overlay_items.append(item)

    def clear_overlays(self):
        """Remove all overlay items."""
        for item in self._overlay_items:
            self._iv.removeItem(item)
        self._overlay_items.clear()

    def export_png(self, filename=None):
        """Export current view to PNG."""
        if filename is None:
            filename, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, 'Export Image', '', 'PNG Files (*.png);;All Files (*)')
        if filename:
            exporter = pg.exporters.ImageExporter(self._iv.scene)
            exporter.export(filename)

    def addItem(self, item):
        """Proxy addItem to internal ImageView."""
        self._iv.addItem(item)

    def removeItem(self, item):
        """Proxy removeItem to internal ImageView."""
        self._iv.removeItem(item)

    def getView(self):
        """Proxy getView to internal ImageView."""
        return self._iv.getView()

    def getViewBox(self):
        """Get the ViewBox."""
        return self._get_viewbox()

    @property
    def imageItem(self):
        return self._iv.imageItem

    @property
    def scene(self):
        return self._iv.scene

    # ── Internal ────────────────────────────────────────────────────

    def _get_viewbox(self):
        """Get the ViewBox from the PlotItem."""
        return self._iv.getView().getViewBox()

    def _apply_log(self, data):
        """Apply log10 to data for display."""
        return np.log10(np.clip(data.astype(np.float64), 1e-10, None))

    def _on_mouse_moved(self, evt):
        pos = evt[0]
        vb = self._iv.getView()
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
        """Ctrl+wheel scrolls frames; plain wheel ignored (no zoom over SSH)."""
        if ev.modifiers() & QtCore.Qt.ControlModifier:
            delta = 1 if ev.angleDelta().y() > 0 else -1
            self.frameScrolled.emit(delta)
            ev.accept()
        else:
            ev.ignore()

    def mousePressEvent(self, ev):
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev):
        super().mouseReleaseEvent(ev)


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
        font = QtGui.QFont('Menlo', 10)
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
