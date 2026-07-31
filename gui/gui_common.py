#!/usr/bin/env python3
"""
MIDAS GUI Common Utilities
Shared components for PyQt5 + PyQtGraph based viewers.

Provides:
  - apply_theme()       : Dark/light palette for Qt + PyQtGraph
  - MIDASImageView      : ImageView subclass with crosshair, colormap, status, wheel-nav
  - AsyncWorker         : QThread wrapper for background tasks
  - LogPanel            : Collapsible log output widget
  - export_frame_movie(): Save a frame range as MP4 / GIF / PNG sequence
"""

import sys
import os
import time
import numpy as np

from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg
import pyqtgraph.exporters  # noqa: F401 — populates pg.exporters for export_png

# ── Constants ──────────────────────────────────────────────────────────
COLORMAPS = ['viridis', 'inferno', 'plasma', 'magma', 'turbo',
             'gray', 'gray_r', 'hot', 'cool', 'bone']


# ── Theme ──────────────────────────────────────────────────────────────
def apply_theme(app, theme='light'):
    """Apply dark or light theme to the Qt application and PyQtGraph.

    Switching themes is idempotent: every palette role is set explicitly for
    both light and dark, and top-level widgets are unpolished + repolished
    so the new palette actually propagates (Fusion on Linux otherwise leaves
    LineEdit/Button backgrounds at their previous theme's values).
    """
    app.setStyle('Fusion')
    pal = QtGui.QPalette()

    if theme == 'dark':
        win, win_text   = QtGui.QColor(45, 45, 45),   QtGui.QColor(220, 220, 220)
        base, alt_base  = QtGui.QColor(30, 30, 30),   QtGui.QColor(50, 50, 50)
        text            = QtGui.QColor(220, 220, 220)
        tip_bg, tip_fg  = QtGui.QColor(45, 45, 45),   QtGui.QColor(220, 220, 220)
        btn, btn_text   = QtGui.QColor(60, 60, 60),   QtGui.QColor(220, 220, 220)
        bright          = QtGui.QColor(255, 80, 80)
        hl, hl_text     = QtGui.QColor(42, 130, 218), QtGui.QColor(255, 255, 255)
        placeholder     = QtGui.QColor(150, 150, 150)
        link            = QtGui.QColor(80, 160, 240)
        disabled_text   = QtGui.QColor(127, 127, 127)
        pg_bg, pg_fg    = 'k', 'w'
    else:
        win, win_text   = QtGui.QColor(240, 240, 240), QtGui.QColor(0, 0, 0)
        base, alt_base  = QtGui.QColor(255, 255, 255), QtGui.QColor(245, 245, 245)
        text            = QtGui.QColor(0, 0, 0)
        tip_bg, tip_fg  = QtGui.QColor(255, 255, 220), QtGui.QColor(0, 0, 0)
        btn, btn_text   = QtGui.QColor(240, 240, 240), QtGui.QColor(0, 0, 0)
        bright          = QtGui.QColor(255, 0, 0)
        hl, hl_text     = QtGui.QColor(48, 140, 198),  QtGui.QColor(255, 255, 255)
        placeholder     = QtGui.QColor(120, 120, 120)
        link            = QtGui.QColor(0, 0, 255)
        disabled_text   = QtGui.QColor(160, 160, 160)
        pg_bg, pg_fg    = 'w', 'k'

    pal.setColor(QtGui.QPalette.Window,           win)
    pal.setColor(QtGui.QPalette.WindowText,       win_text)
    pal.setColor(QtGui.QPalette.Base,             base)
    pal.setColor(QtGui.QPalette.AlternateBase,    alt_base)
    pal.setColor(QtGui.QPalette.Text,             text)
    pal.setColor(QtGui.QPalette.ToolTipBase,      tip_bg)
    pal.setColor(QtGui.QPalette.ToolTipText,      tip_fg)
    pal.setColor(QtGui.QPalette.Button,           btn)
    pal.setColor(QtGui.QPalette.ButtonText,       btn_text)
    pal.setColor(QtGui.QPalette.BrightText,       bright)
    pal.setColor(QtGui.QPalette.Highlight,        hl)
    pal.setColor(QtGui.QPalette.HighlightedText,  hl_text)
    pal.setColor(QtGui.QPalette.PlaceholderText,  placeholder)
    pal.setColor(QtGui.QPalette.Link,             link)
    pal.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.Text,       disabled_text)
    pal.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.ButtonText, disabled_text)
    pal.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText, disabled_text)
    app.setPalette(pal)

    pg.setConfigOptions(background=pg_bg, foreground=pg_fg)

    # Force every existing widget to repaint with the new palette. Without
    # this, Fusion on Linux leaves QLineEdit/QPushButton/QSpinBox painted in
    # the old palette's Base/Button colors until they are interacted with.
    style = app.style()
    for top in app.topLevelWidgets():
        for w in [top] + top.findChildren(QtWidgets.QWidget):
            style.unpolish(w)
            style.polish(w)
            w.update()


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
    # Emitted when image data changes: (min, max, p2, p98).
    # Always linear-space, like levelsChanged, even when log display is on.
    dataStatsUpdated = QtCore.pyqtSignal(float, float, float, float)
    # Emitted for movie mode: advance one frame
    movieFrameAdvance = QtCore.pyqtSignal()
    # Emitted when the record button is pressed — host runs export_frame_movie
    movieSaveRequested = QtCore.pyqtSignal()
    # Emitted when a file is dropped onto the viewer
    fileDropped = QtCore.pyqtSignal(str)
    # Emitted when the font size spinbox changes
    fontSizeChanged = QtCore.pyqtSignal(int)
    # Emitted when intensity levels change via the histogram region drag.
    # Always reports linear-space (lo, hi) regardless of log display mode,
    # so consumers can populate text fields without unit conversion.
    levelsChanged = QtCore.pyqtSignal(float, float)

    def __init__(self, parent=None, name='MIDASImageView', origin='bl', **kwargs):
        super().__init__(parent)

        self._raw_data = None
        self._log_mode = False
        self._origin = origin  # 'bl' = bottom-left, 'br' = bottom-right

        # ── Internal ImageView ──
        self._iv = pg.ImageView(parent=self, name=name, view=pg.PlotItem(), **kwargs)
        self._iv.ui.roiBtn.hide()
        self._iv.ui.menuBtn.hide()

        # Force the histogram axis to integer tick labels. Diffraction
        # intensities are always counts; the default '%.2g' formatter
        # prints things like '1.2e+03' which are unreadable at narrow widths.
        try:
            hist_axis = self._iv.ui.histogram.axis
            hist_axis.tickStrings = lambda values, scale, spacing: [
                f'{int(round(v * scale))}' for v in values]
        except Exception:
            pass

        # Bidirectional level sync: forward histogram region drags out as
        # levelsChanged. setLevels() raises _suppress_levels_signal so
        # programmatic region updates (from text-field edits / setImage with
        # explicit levels) don't echo back and overwrite the source field.
        self._suppress_levels_signal = False
        hist = getattr(getattr(self._iv, 'ui', None), 'histogram', None)
        if hist is not None and hasattr(hist, 'sigLevelsChanged'):
            hist.sigLevelsChanged.connect(self._on_hist_levels_changed)

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

        # Record: hand off to the host, which knows how to walk its own frame
        # index (multi-file spans, aggregation modes, …). See export_frame_movie.
        self._rec_btn = QtWidgets.QToolButton()
        self._rec_btn.setIcon(style.standardIcon(QtWidgets.QStyle.SP_DialogSaveButton))
        self._rec_btn.setText("REC")
        self._rec_btn.setToolTipDuration(20000)
        self._rec_btn.setToolTip(
            "Save Movie – write a frame range to MP4 / GIF / PNG sequence.\n"
            "Captures the rendered view (colormap, levels, log scale, zoom\n"
            "and every overlay), so the movie matches what Play shows.")
        self._rec_btn.clicked.connect(self.movieSaveRequested)
        bar.addWidget(self._rec_btn)

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

        bar.addSeparator()

        # Font size control
        bar.addWidget(QtWidgets.QLabel("Font:"))
        self._font_spin = QtWidgets.QSpinBox()
        self._font_spin.setRange(8, 36)
        self._font_spin.setValue(14)
        self._font_spin.setFixedWidth(50)
        self._font_spin.setToolTip("Viewer font size (pt)")
        self._font_spin.valueChanged.connect(self.fontSizeChanged)
        bar.addWidget(self._font_spin)

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
        """Reset view to full image extent (image bounds only, no overlays)."""
        self._push_view()
        self._fit_to_image()
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
        prev_shape = None if self._raw_data is None else self._raw_data.shape
        self._raw_data = data
        # Transpose: PyQtGraph maps axis-0→X, axis-1→Y, but numpy images
        # are (rows, cols). Transpose so rows→Y (vertical), cols→X (horizontal).
        display = self._apply_log(data.T) if self._log_mode else data.T

        # Compute stats. These are display-space (log10'd when log mode is on),
        # which is what the auto-level branch below needs.
        finite = display[np.isfinite(display)]
        if finite.size > 0:
            dmin, dmax = float(finite.min()), float(finite.max())
            p2 = float(np.percentile(finite, 2))
            p98 = float(np.percentile(finite, 98))
        else:
            dmin = dmax = p2 = p98 = 0.0
        # dataStatsUpdated always reports LINEAR space — same convention as
        # levelsChanged — so a consumer can drop the values straight into a
        # MinI/MaxI field (which setLevels re-log's) without knowing whether
        # log display is on.
        if self._log_mode:
            stats_out = tuple(10.0 ** v for v in (dmin, dmax, p2, p98))
        else:
            stats_out = (dmin, dmax, p2, p98)
        self.dataStatsUpdated.emit(*stats_out)

        # autoRange=False: keep the user's pan/zoom across frame changes
        # and stop pyqtgraph from inflating the view to fit off-image
        # overlays (rings outside the detector, lab-axis labels, …).
        # We manually set a tight range below for fresh / shape-changed
        # displays.
        #
        # _suppress_levels_signal: setImage makes the histogram emit
        # sigLevelsChanged, which would echo out as levelsChanged and let a
        # frame change rewrite the host's MinI/MaxI fields. Only a user drag
        # should do that, so gate the whole redraw.
        self._suppress_levels_signal = True
        try:
            if levels is not None:
                if self._log_mode:
                    levels = (np.log10(max(levels[0], 1e-10)),
                              np.log10(max(levels[1], 1e-10)))
                self._iv.setImage(display, autoLevels=False, levels=levels,
                                  autoRange=False)
                _hist_levels = levels
            elif auto_levels:
                self._iv.setImage(display, autoLevels=False, levels=(p2, p98),
                                  autoRange=False)
                _hist_levels = (p2, p98)
            else:
                self._iv.setImage(display, autoLevels=False, autoRange=False)
                _hist_levels = None
        finally:
            self._suppress_levels_signal = False

        if _hist_levels is not None:
            try:
                self._iv.ui.histogram.setHistogramRange(
                    _hist_levels[0], _hist_levels[1], padding=0.05)
            except Exception:
                pass

        # Force origin position AFTER setImage (which may reset axes)
        self._apply_origin()

        # Tight initial view: fit to image bounds with no padding so the
        # data fills the canvas. Skip on same-shape redraws so the user's
        # pan/zoom is preserved across frame changes.
        if prev_shape is None or prev_shape != data.shape:
            self._fit_to_image()

        # Push initial view to history
        self._push_view()

    def _fit_to_image(self):
        """Set view range to the current image bounds with no padding."""
        if self._raw_data is None:
            return
        vb = self._iv.getView()
        ny, nz = self._raw_data.shape  # data is (rows, cols) = (Y, X)
        # display = data.T → (X, Y) = (nz, ny). View X spans [0, nz], Y [0, ny].
        vb.setRange(xRange=(0, nz), yRange=(0, ny), padding=0)

    def _apply_origin(self):
        vb = self._iv.getView()
        vb.invertY(False)
        vb.invertX(self._origin == 'br')

    def set_origin(self, origin):
        """Switch display origin between 'bl' (no flip) and 'br' (mirror X).

        Use 'bl' for single-panel data already in physical chirality and
        'br' for the HYDRA composite, whose stitching introduces an X-axis
        flip that needs cancelling at display time. Overlays branch on
        ``_origin`` so they follow automatically on the next redraw.
        """
        if origin not in ('bl', 'br'):
            raise ValueError(f"origin must be 'bl' or 'br', got {origin!r}")
        if origin == self._origin:
            return
        self._origin = origin
        if self._raw_data is not None:
            self._apply_origin()

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
        """Apply a named colormap to both the image and the histogram LUT strip."""
        cmap = get_colormap(name)
        lut = cmap.getLookupTable(nPts=256)
        self._iv.imageItem.setLookupTable(lut)
        # Sync the histogram gradient bar on the right side.
        hist = getattr(getattr(self._iv, 'ui', None), 'histogram', None)
        if hist is not None:
            grad = getattr(hist, 'gradient', None)
            if grad is not None and hasattr(grad, 'setColorMap'):
                try:
                    grad.setColorMap(cmap)
                except Exception:
                    pass

    def set_crosshair_visible(self, visible):
        """Show or hide crosshair."""
        self._crosshair_visible = visible
        self._vline.setVisible(visible)
        self._hline.setVisible(visible)

    def setLevels(self, lo, hi):
        """Set intensity levels (accepts linear-space values) and zoom the
        histogram axis to match.

        - Log conversion: set_image_data feeds the ImageView log10-converted
          data when _log_mode is on, so levels also have to be log10'd or
          the region sits at linear y-values off the log axis.
        - _suppress_levels_signal: stops the resulting sigLevelsChanged from
          looping back through _on_hist_levels_changed → levelsChanged →
          MinI/MaxI text field, which would overwrite whatever the user
          just typed.
        - setHistogramRange: pyqtgraph's axis otherwise spans the full data
          range; in HYDRA composite mode the level lines collapse to a
          tiny sliver. Zoom to the levels with a small padding instead.
        """
        if self._log_mode:
            lo = np.log10(max(lo, 1e-10))
            hi = np.log10(max(hi, 1e-10))
        self._suppress_levels_signal = True
        try:
            self._iv.setLevels(lo, hi)
            try:
                self._iv.ui.histogram.setHistogramRange(lo, hi, padding=0.05)
            except Exception:
                pass
        finally:
            self._suppress_levels_signal = False

    def _on_hist_levels_changed(self, *_args):
        """Forward histogram region drags to listeners as linear-space levels."""
        if self._suppress_levels_signal:
            return
        hist = getattr(getattr(self._iv, 'ui', None), 'histogram', None)
        if hist is None:
            return
        try:
            lo, hi = hist.getLevels()
        except Exception:
            return
        if lo is None or hi is None:
            return
        if self._log_mode:
            lo = 10.0 ** float(lo)
            hi = 10.0 ** float(hi)
        self.levelsChanged.emit(float(lo), float(hi))

    def add_overlay(self, item, category='default'):
        """Add a PlotItem overlay (rings, markers, etc.).

        ``category`` is a free-form tag enabling selective clearing via
        :meth:`clear_overlays`. Items added without a category go to ``'default'``.
        """
        self._iv.addItem(item)
        self._overlay_items.append((category, item))

    def clear_overlays(self, category=None):
        """Remove overlay items.

        ``category=None`` (default) removes everything. Pass a specific tag
        (e.g. ``'rings'``, ``'axes'``) to remove only items in that category.
        """
        keep = []
        for entry in self._overlay_items:
            # Backward compat: legacy entries may be bare items (no tuple)
            if isinstance(entry, tuple) and len(entry) == 2:
                cat, item = entry
            else:
                cat, item = 'default', entry
            if category is None or cat == category:
                self._iv.removeItem(item)
            else:
                keep.append((cat, item))
        self._overlay_items = keep

    def export_png(self, filename=None):
        """Export current view to PNG."""
        if filename is None:
            filename, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, 'Export Image', '', 'PNG Files (*.png);;All Files (*)')
        if filename:
            exporter = pg.exporters.ImageExporter(self._iv.scene)
            exporter.export(filename)

    def scene_exporter(self, even_dims=False):
        """Build an ImageExporter bound to this view's scene.

        Same object ``export_png`` uses, but returned so a caller can hold one
        exporter across many frames: ImageExporter freezes width/height at
        construction, which is what keeps every captured frame the same size —
        a hard requirement for video encoders.

        ``even_dims`` rounds the capture size down to even width/height, which
        several MP4 codecs require.
        """
        exporter = pg.exporters.ImageExporter(self._iv.scene)
        if even_dims:
            w = int(exporter.params['width'])
            h = int(exporter.params['height'])
            # Assign height last: widthChanged() rewrites height to preserve
            # the aspect ratio, so setting width first would undo it.
            exporter.params.param('width').setValue(max(w - (w % 2), 2))
            exporter.params.param('height').setValue(max(h - (h % 2), 2))
        return exporter

    def grab_scene_rgb(self, exporter=None):
        """Render the current view to an (H, W, 3) uint8 RGB array.

        Renders the scene offscreen rather than grabbing the widget, so the
        capture is correct even when a modal progress dialog covers the window.
        """
        if exporter is None:
            exporter = self.scene_exporter()
        return qimage_to_rgb_array(exporter.export(toBytes=True))

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

    def set_image_rect(self, x, y, w, h):
        """Position the displayed image item at scene rectangle (x, y, w, h).

        Used by the Tx-rotation path to display an expanded-canvas rotated
        image at its original-scene-coord location, so ring overlays, lab-axes
        overlay, and cursor R/η stay aligned with where the data actually is.

        Auto-fits the viewport only when the rect *changes* — frame navigation
        with a constant rect preserves the user's zoom; a Tx-toggle that moves
        the image to a new region triggers a one-shot ``autoRange`` so the
        new image is visible.
        """
        new_rect = (float(x), float(y), float(w), float(h))
        rect_changed = new_rect != getattr(self, '_last_image_rect', None)
        self._iv.imageItem.setRect(QtCore.QRectF(*new_rect))
        self._last_image_rect = new_rect
        if rect_changed:
            # Fit to the image rect itself, not autoRange — overlays
            # (rings outside the detector, lab-axis labels) would
            # otherwise inflate the view and reintroduce blank margins.
            vb = self._get_viewbox()
            vb.setRange(xRange=(x, x + w), yRange=(y, y + h), padding=0)

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


# ── Movie export ───────────────────────────────────────────────────────

def qimage_to_rgb_array(img):
    """QImage → (H, W, 3) uint8 RGB array.

    Goes through Format_RGB888 so the result is byte-order independent (the
    native ARGB32 buffer is BGRA on little-endian, ARGB on big-endian) and
    honours ``bytesPerLine``, which Qt pads to a 4-byte boundary.
    """
    img = img.convertToFormat(QtGui.QImage.Format_RGB888)
    w, h, bpl = img.width(), img.height(), img.bytesPerLine()
    ptr = img.constBits()
    ptr.setsize(bpl * h)
    flat = np.frombuffer(bytes(ptr), dtype=np.uint8).reshape(h, bpl)
    return flat[:, :w * 3].reshape(h, w, 3).copy()


class MovieWriter:
    """Frame sink for :func:`export_frame_movie`, dispatched on extension.

      ``.mp4`` / ``.mov`` / ``.avi`` / ``.mkv``  → OpenCV ``VideoWriter``
      ``.gif``                                   → Pillow animated GIF
      ``.png`` / ``.jpg`` / ``.tif``             → numbered still sequence

    Backends are imported lazily; a missing one raises ``RuntimeError`` naming
    the package and a working alternative, so the GUI can show it verbatim
    instead of a traceback.

    Frames must all be the same size for the video path; anything off-size is
    cropped (or zero-padded) to the first frame's dimensions.
    """

    VIDEO_EXTS = ('.mp4', '.mov', '.avi', '.mkv')
    STILL_EXTS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')

    def __init__(self, path, fps):
        self.path = path
        self.fps = max(1, int(round(fps)))
        self.ext = os.path.splitext(path)[1].lower()
        self.count = 0
        self._size = None       # (h, w) locked in by the first frame
        self._video = None
        self._gif_frames = None

        if self.ext in self.VIDEO_EXTS:
            self.kind = 'video'
            try:
                import cv2  # noqa: F401
            except ImportError:
                raise RuntimeError(
                    "Video export needs OpenCV.\n\n"
                    "    pip install opencv-python-headless\n\n"
                    "Or save as .gif / .png instead — those need no extra package.")
        elif self.ext == '.gif':
            self.kind = 'gif'
            try:
                from PIL import Image  # noqa: F401
            except ImportError:
                raise RuntimeError(
                    "GIF export needs Pillow.\n\n    pip install Pillow")
            self._gif_frames = []
        elif self.ext in self.STILL_EXTS:
            self.kind = 'stills'
            try:
                from PIL import Image  # noqa: F401
            except ImportError:
                raise RuntimeError(
                    "Image-sequence export needs Pillow.\n\n    pip install Pillow")
            self._stem, self._suffix = os.path.splitext(path)
        else:
            raise RuntimeError(
                f"Don't know how to write '{self.ext or path}'.\n"
                "Use .mp4, .gif, or .png (numbered sequence).")

    @property
    def needs_even_dims(self):
        """MP4 codecs reject odd width/height; stills and GIF don't care."""
        return self.kind == 'video'

    def _fit(self, rgb):
        """Force *rgb* to the locked-in frame size (crop, then zero-pad)."""
        if self._size is None:
            self._size = rgb.shape[:2]
            return rgb
        h, w = self._size
        if rgb.shape[:2] == (h, w):
            return rgb
        out = np.zeros((h, w, 3), dtype=np.uint8)
        ch, cw = min(h, rgb.shape[0]), min(w, rgb.shape[1])
        out[:ch, :cw] = rgb[:ch, :cw]
        return out

    def append(self, rgb):
        rgb = self._fit(np.ascontiguousarray(rgb, dtype=np.uint8))
        if self.kind == 'video':
            import cv2
            if self._video is None:
                h, w = self._size
                fourcc = cv2.VideoWriter_fourcc(*('MJPG' if self.ext == '.avi'
                                                  else 'mp4v'))
                self._video = cv2.VideoWriter(self.path, fourcc, self.fps, (w, h))
                if not self._video.isOpened():
                    self._video = None
                    raise RuntimeError(
                        f"OpenCV could not open '{os.path.basename(self.path)}' "
                        f"for writing at {w}×{h}.\nTry .gif or .png instead.")
            self._video.write(rgb[:, :, ::-1])          # cv2 wants BGR
        elif self.kind == 'gif':
            from PIL import Image
            # Palettize per frame rather than buffering RGB: a few hundred
            # captures of a 1500×900 view is ~4 GB in RGB, ~1.3 GB as 'P'.
            self._gif_frames.append(
                Image.fromarray(rgb).convert('P', palette=Image.ADAPTIVE))
        else:
            from PIL import Image
            Image.fromarray(rgb).save(f"{self._stem}_{self.count:05d}{self._suffix}")
        self.count += 1

    def close(self):
        """Finalise and return the path actually written (None if no frames)."""
        if self.kind == 'video':
            if self._video is not None:
                self._video.release()
                self._video = None
        elif self.kind == 'gif' and self._gif_frames:
            self._gif_frames[0].save(
                self.path, save_all=True, append_images=self._gif_frames[1:],
                duration=int(round(1000.0 / self.fps)), loop=0)
            self._gif_frames = []
        if self.count == 0:
            return None
        if self.kind == 'stills':
            return f"{self._stem}_00000{self._suffix} … +{self.count - 1} more"
        return self.path


class MovieExportDialog(QtWidgets.QDialog):
    """Ask for start frame, frame count, step, fps, and an output file."""

    def __init__(self, parent=None, current_frame=0, n_frames_total=None,
                 default_path=''):
        super().__init__(parent)
        self.setWindowTitle("Save Movie")
        self._n_total = n_frames_total

        form = QtWidgets.QFormLayout()

        self.start_spin = QtWidgets.QSpinBox()
        self.start_spin.setRange(0, 9999999)
        self.start_spin.setValue(int(current_frame))
        self.start_spin.setToolTip("Frame index the movie starts at "
                                   "(same numbering as Display Frame).")
        form.addRow("Start frame", self.start_spin)

        self.count_spin = QtWidgets.QSpinBox()
        self.count_spin.setRange(1, 9999999)
        if n_frames_total:
            self.count_spin.setValue(max(1, int(n_frames_total) - int(current_frame)))
        else:
            self.count_spin.setValue(100)
        self.count_spin.setToolTip("How many frames to capture.")
        form.addRow("# frames", self.count_spin)

        self.step_spin = QtWidgets.QSpinBox()
        self.step_spin.setRange(1, 10000)
        self.step_spin.setValue(1)
        self.step_spin.setToolTip("Capture every Nth frame (1 = every frame).")
        form.addRow("Step", self.step_spin)

        self.fps_spin = QtWidgets.QSpinBox()
        self.fps_spin.setRange(1, 60)
        self.fps_spin.setValue(10)
        self.fps_spin.setToolTip("Playback rate of the written movie.")
        form.addRow("FPS", self.fps_spin)

        path_row = QtWidgets.QHBoxLayout()
        self.path_edit = QtWidgets.QLineEdit(default_path)
        self.path_edit.setMinimumWidth(360)
        self.path_edit.setToolTip(
            ".mp4 / .mov / .avi → video (needs OpenCV)\n"
            ".gif               → animated GIF (needs Pillow)\n"
            ".png / .tif        → numbered still sequence")
        path_row.addWidget(self.path_edit, 1)
        browse = QtWidgets.QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        path_row.addWidget(browse)
        path_w = QtWidgets.QWidget()
        path_w.setLayout(path_row)
        form.addRow("Output", path_w)

        self._summary = QtWidgets.QLabel("")
        self._summary.setStyleSheet("color: gray;")
        form.addRow("", self._summary)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Save | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        lay = QtWidgets.QVBoxLayout(self)
        note = QtWidgets.QLabel(
            "Captures the rendered view — current colormap, intensity levels,\n"
            "log scale, zoom, and all overlays — one capture per frame.")
        note.setStyleSheet("color: gray;")
        lay.addWidget(note)
        lay.addLayout(form)
        lay.addWidget(buttons)

        for spin in (self.start_spin, self.count_spin, self.step_spin, self.fps_spin):
            spin.valueChanged.connect(self._update_summary)
        self._update_summary()

    def _browse(self):
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, 'Save Movie', self.path_edit.text(),
            'MP4 video (*.mp4);;Animated GIF (*.gif);;'
            'PNG sequence (*.png);;TIFF sequence (*.tif);;All Files (*)')
        if fn:
            self.path_edit.setText(fn)

    def _update_summary(self):
        vals = self.values()
        last = vals['start'] + (vals['count'] - 1) * vals['step']
        secs = vals['count'] / float(vals['fps'])
        txt = (f"frames {vals['start']}–{last} "
               f"({vals['count']} captures) → {secs:.1f} s at {vals['fps']} fps")
        if self._n_total and last > self._n_total - 1:
            txt += f"\n⚠ last frame {last} is past the known end ({self._n_total - 1})"
        self._summary.setText(txt)

    def values(self):
        return dict(start=self.start_spin.value(),
                    count=self.count_spin.value(),
                    step=self.step_spin.value(),
                    fps=self.fps_spin.value(),
                    path=self.path_edit.text().strip())


def export_frame_movie(parent, view, set_frame, current_frame=0,
                       n_frames_total=None, wait_for_render=None,
                       default_path=''):
    """Save a range of frames as a movie, capturing the rendered view.

    Drives the host's own frame navigation rather than duplicating it:

    ``set_frame(i)``       host callback that displays frame *i*.
    ``wait_for_render()``  optional host callback that blocks (pumping the
                           event loop) until the frame is actually on screen —
                           required wherever the display path is threaded, e.g.
                           the HYDRA composite and the Max/Sum/Median workers.

    Runs synchronously behind a cancellable progress dialog rather than reusing
    the play timer, so every frame is written exactly once regardless of how
    long a frame takes to load. Restores the original frame when done.
    Returns the written path, or None if cancelled before any frame.
    """
    dlg = MovieExportDialog(parent, current_frame=current_frame,
                            n_frames_total=n_frames_total,
                            default_path=default_path)
    if dlg.exec_() != QtWidgets.QDialog.Accepted:
        return None
    opts = dlg.values()
    if not opts['path']:
        QtWidgets.QMessageBox.warning(parent, "Save Movie", "No output file given.")
        return None

    try:
        writer = MovieWriter(opts['path'], opts['fps'])
    except RuntimeError as e:
        QtWidgets.QMessageBox.critical(parent, "Save Movie", str(e))
        return None

    frame_ids = [opts['start'] + i * opts['step'] for i in range(opts['count'])]
    exporter = view.scene_exporter(even_dims=writer.needs_even_dims)

    prog = QtWidgets.QProgressDialog("Capturing frames…", "Cancel",
                                     0, len(frame_ids), parent)
    prog.setWindowTitle("Save Movie")
    prog.setWindowModality(QtCore.Qt.WindowModal)
    prog.setMinimumDuration(0)
    prog.setValue(0)

    err = None
    try:
        for i, f in enumerate(frame_ids):
            if prog.wasCanceled():
                break
            prog.setLabelText(f"Frame {f}   ({i + 1} / {len(frame_ids)})")
            prog.setValue(i)
            QtWidgets.QApplication.processEvents()
            set_frame(f)
            if wait_for_render is not None:
                wait_for_render()
            QtWidgets.QApplication.processEvents()
            writer.append(view.grab_scene_rgb(exporter))
        prog.setValue(len(frame_ids))
    except Exception as e:                    # noqa: BLE001 — reported to the user
        err = e
    finally:
        prog.close()
        written = writer.close()
        # Put the viewer back where the user left it.
        try:
            set_frame(current_frame)
            if wait_for_render is not None:
                wait_for_render()
        except Exception:
            pass

    if err is not None:
        QtWidgets.QMessageBox.critical(
            parent, "Save Movie",
            f"Wrote {writer.count} frame(s), then failed:\n{err}")
        return written
    if written is None:
        QtWidgets.QMessageBox.information(
            parent, "Save Movie", "Cancelled — no frames written.")
        return None
    QtWidgets.QMessageBox.information(
        parent, "Save Movie",
        f"Wrote {writer.count} frame(s) at {writer.fps} fps:\n{written}")
    return written


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


def draw_lab_frame_axes(image_view, bc_y, bc_z, ny, nz,
                         category='axes',
                         xl_color='#FF3B30',
                         yl_color='#34C759',
                         zl_color='#0A84FF',
                         eta_tick_color='#FFA500',
                         font_size=12):
    """Draw MIDAS lab-frame axes anchored at (bc_y, bc_z).

    Convention (independent of detector readout / display origin):
      X_Lab (MIDAS Y_MIDAS, red)   → display LEFT
      Y_Lab (MIDAS Z_MIDAS, green) → display UP
      Z_Lab (MIDAS X_MIDAS, blue)  → INTO page (⊗ at BC)
      η = 0 toward Y_Lab/Z_MIDAS (top), +90° on −Y_lab side (display-right),
      ±180° at bottom, −90° on X_Lab/+Y_MIDAS side (display-left).
      An arc from η=0 to η=+45° with an arrowhead shows the η-sweep
      direction (yellow/orange, matching the η cardinal labels).
      Each axis label shows both its lab name and the MIDAS-native
      axis it corresponds to, e.g. ``X_Lab (Y_MIDAS)``.

    Compatible with both `'bl'` and `'br'` :class:`MIDASImageView` origins
    by reading ``image_view._origin`` and flipping the data-X sign.

    All overlay items are tagged with ``category`` (default ``'axes'``)
    so they can be cleared independently via
    ``image_view.clear_overlays(category)``.

    Parameters
    ----------
    image_view : MIDASImageView
        Target view to draw the overlay on.
    bc_y, bc_z : float
        Beam-center pixel coordinates in display data space (the same
        coordinate system the cursor reports).
    ny, nz : int
        Image dimensions; used only to size the arrows sensibly.
    """
    import math
    image_view.clear_overlays(category)

    # Arrow length: 15% of the smaller image dim, clamped to a sensible range.
    L = max(60.0, min(400.0, 0.15 * min(ny, nz)))
    head = max(15.0, L * 0.20)

    origin = getattr(image_view, '_origin', 'bl')
    # Sign of data-x that visually appears on display-LEFT:
    #   'bl': display-left = pixel −x  → y_sign = −1
    #   'br': display-left = pixel +x  → y_sign = +1
    y_sign = +1.0 if origin == 'br' else -1.0

    text_pen  = pg.mkPen('w')
    text_fill = pg.mkBrush(0, 0, 0, 200)
    xl_pen    = pg.mkPen(xl_color, width=3.5)
    yl_pen    = pg.mkPen(yl_color, width=3.5)
    arc_pen   = pg.mkPen(eta_tick_color, width=2.5)

    # Fonts scale off the caller-supplied label size. ⊗ uses a larger size so
    # the beam-direction glyph reads clearly; η ticks match the label size.
    label_font = QtGui.QFont(); label_font.setPointSize(int(font_size)); label_font.setBold(True)
    glyph_font = QtGui.QFont(); glyph_font.setPointSize(max(int(font_size * 1.5), int(font_size) + 4)); glyph_font.setBold(True)
    eta_font   = QtGui.QFont(); eta_font.setPointSize(int(font_size)); eta_font.setBold(True)

    # Size of one screen pixel in data coords at the *current* zoom/pan —
    # lets us convert a desired on-screen clearance (which the fixed-size
    # text always needs) into data units that stay correct whether the view
    # is zoomed all the way out or magnified in.
    px_w = px_h = 1.0
    try:
        pw, ph = image_view.getViewBox().viewPixelSize()
        if pw and ph and pw > 0 and ph > 0:
            px_w, px_h = pw, ph
    except Exception:
        pass
    # Text-clearance conversions use the *isotropic* pixel scale (geometric
    # mean of px_w/px_h) rather than the raw per-axis value. If the user has
    # zoomed/panned to a non-square view (common with a rectangular
    # drag-zoom), px_w and px_h can differ hugely; using the raw axis value
    # would blow up just the X_Lab (or just the η=±90°) offset while Y_Lab
    # stayed put, making one label shoot out far past the others.
    px_iso = math.sqrt(px_w * px_h) if (px_w > 0 and px_h > 0) else 1.0

    def shaft_with_head(x0, y0, x1, y1):
        """Single polyline = shaft + open V-shaped arrowhead at (x1,y1)."""
        dx, dy = x1 - x0, y1 - y0
        length = math.hypot(dx, dy)
        if length < 1e-9:
            return [x0, x1], [y0, y1]
        ux, uy = dx / length, dy / length
        nx, ny_ = -uy, ux
        base_x = x1 - ux * head
        base_y = y1 - uy * head
        wing = head * 0.55
        p1x, p1y = base_x + nx * wing, base_y + ny_ * wing
        p2x, p2y = base_x - nx * wing, base_y - ny_ * wing
        return ([x0, x1, p1x, x1, p2x],
                [y0, y1, p1y, y1, p2y])

    # X_Lab arrow (MIDAS-native Y_MIDAS, visually display-LEFT)
    xs, ys = shaft_with_head(bc_y, bc_z, bc_y + y_sign * L, bc_z)
    image_view.add_overlay(pg.PlotDataItem(xs, ys, pen=xl_pen, connect='all'),
                            category)
    # Y_Lab arrow (MIDAS-native Z_MIDAS, visually display-UP)
    xs, ys = shaft_with_head(bc_y, bc_z, bc_y, bc_z + L)
    image_view.add_overlay(pg.PlotDataItem(xs, ys, pen=yl_pen, connect='all'),
                            category)

    # X_Lab / Y_Lab labels at arrow tips, showing the MIDAS-native axis too.
    # "Lab" / "MIDAS" render as true subscripts via Qt rich text, and each
    # axis is prefixed with an explicit "+" since only the positive
    # direction is ever drawn. Color is embedded as an inline
    # <span style="color:..."> rather than passed via TextItem's `color=`
    # kwarg: pyqtgraph's TextItem only applies `color` in its plain-text
    # branch and silently ignores it whenever `html` is given, so a bare
    # `color=` here would leave the text un-tinted.
    #
    # Clearance beyond the arrow tip is sized from the label's *actual*
    # rendered footprint (via QFontMetrics), not a fixed multiple of `head`:
    # a flat multiplier stays constant even when the label text is long, so
    # for a horizontally-anchored box (its near edge — not center — sits at
    # the offset point) it doesn't guarantee the box clears the arrowhead.
    # The pixel->data conversion is capped at a multiple of L: on a very
    # zoomed-out (e.g. multi-tile hydra) view, one screen pixel covers a lot
    # of data, so an uncapped conversion pushes the label absurdly far from
    # the arrow just to preserve an exact on-screen gap. Capping trades a
    # little precision at extreme zoom-out for keeping the label visually
    # anchored near its arrow.
    fm = QtGui.QFontMetrics(label_font)
    margin_px = 10.0
    label_specs = (
            ('h', '+X<sub>Lab</sub> (+Y<sub>MIDAS</sub>)', '+X_Lab (+Y_MIDAS)', xl_color),
            ('v', '+Y<sub>Lab</sub> (+Z<sub>MIDAS</sub>)', '+Y_Lab (+Z_MIDAS)', yl_color))
    arrow_label_R_h = arrow_label_R_v = L + head * 1.6
    for axis_kind, html_body, plain_text, axis_color in label_specs:
        html = '<span style="color:{};">{}</span>'.format(axis_color, html_body)
        if axis_kind == 'h':
            # Anchored box extends fully backward from the offset point, so
            # the offset itself must clear the tip by the box's full width.
            text_extent = min((fm.boundingRect(plain_text).width() + margin_px) * px_iso,
                               0.8 * L)
            arrow_label_R_h = L + max(head * 1.6, text_extent)
            # Nudge below the shaft's y so the red line stays visible instead
            # of running straight through the middle of the label box.
            dx, dy = y_sign * arrow_label_R_h, -head * 0.9
            anchor = (0.0 if dx > 0 else 1.0, 0.5)
        else:
            # Centered box only extends half its height back toward the tip.
            text_extent = min((fm.height() / 2.0 + margin_px) * px_iso,
                               0.8 * L)
            arrow_label_R_v = L + max(head * 1.6, text_extent)
            dx, dy = 0.0, arrow_label_R_v
            anchor = (0.5, 0.5)
        lbl = pg.TextItem(html=html, anchor=anchor,
                          border=text_pen, fill=text_fill)
        lbl.setFont(label_font)
        lbl.setPos(bc_y + dx, bc_z + dy)
        image_view.add_overlay(lbl, category)

    # ⊗ glyph at BC + label — Z_Lab (MIDAS-native X_MIDAS), the beam direction.
    glyph = pg.TextItem('⊗', color=zl_color, anchor=(0.5, 0.5),
                        border=text_pen, fill=text_fill)
    glyph.setFont(glyph_font)
    glyph.setPos(bc_y, bc_z)
    image_view.add_overlay(glyph, category)

    beam_html = '<span style="color:{};">+Z<sub>Lab</sub> (+X<sub>MIDAS</sub>, beam)</span>'.format(zl_color)
    beam_anchor_x = 0.0 if y_sign > 0 else 1.0
    x_lbl = pg.TextItem(html=beam_html, anchor=(beam_anchor_x, 0.5),
                        border=text_pen, fill=text_fill)
    x_lbl.setFont(label_font)
    x_lbl.setPos(bc_y + y_sign * head * 1.4, bc_z - head * 1.4)
    image_view.add_overlay(x_lbl, category)

    # η sweep arc from 0° to +45°, plus an arrowhead at the +45° end.
    # Position in data coords: (bc_y + (-y_sign) * R * sin(η),  bc_z + R * cos(η))
    # so that η=+90° maps to display-right (= −Y_lab side) on either origin.
    R_arc = L * 0.85
    n_pts = 24
    eta_deg = np.linspace(0.0, 45.0, n_pts)
    eta_rad = np.deg2rad(eta_deg)
    arc_x = bc_y + (-y_sign) * R_arc * np.sin(eta_rad)
    arc_y = bc_z + R_arc * np.cos(eta_rad)
    image_view.add_overlay(pg.PlotDataItem(arc_x, arc_y, pen=arc_pen),
                            category)

    # Arrowhead-only at arc end, tangent in direction of increasing η.
    end = math.radians(45.0)
    tan_x = (-y_sign) * math.cos(end)
    tan_y = -math.sin(end)
    head_size = head * 0.9
    tip_x, tip_y = float(arc_x[-1]), float(arc_y[-1])
    # base point for the V
    bx = tip_x - tan_x * head_size
    by = tip_y - tan_y * head_size
    # perpendicular for the wings
    nx_, ny_ = -tan_y, tan_x
    wing = head_size * 0.55
    p1x, p1y = bx + nx_ * wing, by + ny_ * wing
    p2x, p2y = bx - nx_ * wing, by - ny_ * wing
    image_view.add_overlay(
        pg.PlotDataItem([p1x, tip_x, p2x], [p1y, tip_y, p2y],
                         pen=arc_pen, connect='all'),
        category)

    # Tiny radial tick at η=0 (just outside the arc) so η=0 has its own marker
    # independent of the +Z arrow.
    tick_inner = R_arc * 1.04
    tick_outer = R_arc * 1.18
    image_view.add_overlay(
        pg.PlotDataItem([bc_y, bc_y], [bc_z + tick_inner, bc_z + tick_outer],
                         pen=arc_pen),
        category)

    # η cardinal labels — placed beyond the X_Lab/Y_Lab arrow-tip labels so
    # they never overlap. A purely data-unit gap is zoom-dependent: it
    # shrinks to nothing when zoomed out to fit a large hydra composite,
    # while the labels themselves render at a fixed screen-pixel size. So
    # convert a minimum on-screen clearance into data units using the
    # view's *current* isotropic pixel scale (px_iso, computed above), which
    # keeps the spacing visually correct whether the view is fully zoomed
    # out or magnified in. Base each pair off the arrow-tip labels' *actual*
    # offsets (arrow_label_R_h/_v), since those already grow to fit long
    # label text — using a flat estimate here could under-space the eta
    # labels whenever the arrow-tip labels had to grow past it. As with the
    # arrow-tip labels, the pixel->data conversion is capped at a multiple
    # of L so a very zoomed-out view can't push these off toward the edges
    # of a multi-tile composite.
    base_gap = max(30.0, 0.25 * L)
    min_gap_v = min((2.5 * font_size + 15.0) * px_iso, 0.6 * L)
    min_gap_h = min((8.0 * font_size + 40.0) * px_iso, 0.6 * L)
    R_eta_v = arrow_label_R_v + max(base_gap, min_gap_v)
    R_eta_h = arrow_label_R_h + max(base_gap, min_gap_h)
    for dx, dy, txt in (
            ( 0.0,             +R_eta_v, 'η=0°'),
            (-y_sign*R_eta_h,   0.0,     'η=+90°'),
            ( 0.0,             -R_eta_v, 'η=±180°'),
            ( y_sign*R_eta_h,   0.0,     'η=−90°')):
        tick = pg.TextItem(txt, color=eta_tick_color, anchor=(0.5, 0.5),
                           border=text_pen, fill=text_fill)
        tick.setFont(eta_font)
        tick.setPos(bc_y + dx, bc_z + dy)
        image_view.add_overlay(tick, category)


def draw_caking_overlay(image_view, bc_y, bc_z, detectors,
                        category='caking'):
    """Draw caking sector overlays for one or more GE detectors.

    Each entry in *detectors* is a tuple:
      ``(color, R_MIN, R_MAX, ETA_MIN, ETA_MAX, ETA_STEP)``
    where R values are in detector pixels from the beam center and ETA values
    are in degrees using the MIDAS convention (η=0 → +Z/top, +90 → display-right).
    ETA_MIN/MAX may use the 0-360 branch (e.g. 125→200) or the ±180 branch.

    Draws for each detector:
      - Inner arc at R_MIN from ETA_MIN to ETA_MAX
      - Outer arc at R_MAX from ETA_MIN to ETA_MAX
      - Radial lines at ETA_MIN, ETA_MIN+STEP, …, ETA_MAX (R_MIN→R_MAX)
    No intermediate R circles are drawn (R_STEP is typically too fine).
    """
    import math
    image_view.clear_overlays(category)
    if not detectors:
        return

    origin = getattr(image_view, '_origin', 'bl')
    y_sign = +1.0 if origin == 'br' else -1.0

    for color, r_min, r_max, eta_min, eta_max, eta_step in detectors:
        pen = pg.mkPen(color, width=1.5)

        # Arc resolution: ~0.5° per point, minimum 120 points
        n_pts = max(120, int(abs(eta_max - eta_min) / 0.5))
        eta_arc = np.linspace(eta_min, eta_max, n_pts)
        eta_rad = np.deg2rad(eta_arc)
        sin_e = np.sin(eta_rad)
        cos_e = np.cos(eta_rad)

        # Inner arc at R_MIN
        arc_y = bc_y + (-y_sign) * r_min * sin_e
        arc_z = bc_z + r_min * cos_e
        image_view.add_overlay(
            pg.PlotDataItem(arc_y, arc_z, pen=pen), category)

        # Outer arc at R_MAX
        arc_y = bc_y + (-y_sign) * r_max * sin_e
        arc_z = bc_z + r_max * cos_e
        image_view.add_overlay(
            pg.PlotDataItem(arc_y, arc_z, pen=pen), category)

        # Radial lines at each ETA_STEP boundary (including ETA_MIN and ETA_MAX)
        if eta_step > 0:
            n_steps = int(round((eta_max - eta_min) / eta_step))
            eta_lines = [eta_min + i * eta_step for i in range(n_steps + 1)]
            # Always include the exact max boundary
            if abs(eta_lines[-1] - eta_max) > 0.01:
                eta_lines.append(eta_max)
        else:
            eta_lines = [eta_min, eta_max]

        for eta in eta_lines:
            er = math.radians(eta)
            s, c = math.sin(er), math.cos(er)
            y0 = bc_y + (-y_sign) * r_min * s
            z0 = bc_z + r_min * c
            y1 = bc_y + (-y_sign) * r_max * s
            z1 = bc_z + r_max * c
            image_view.add_overlay(
                pg.PlotDataItem([y0, y1], [z0, z1], pen=pen), category)
