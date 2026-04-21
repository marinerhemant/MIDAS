"""MIDASImageView (PyQt5).

Wraps ``pyqtgraph.ImageView`` with crosshair, colormap, log toggle, nav toolbar,
movie playback, drag-drop, and a wheel-event signal so callers can implement
frame stepping. Mouse zoom/pan is off by default (cheaper over SSH).

Kept in sync with ``gui/gui_common.py`` — this module lets the unified launcher
reuse the same widget without importing the legacy module directly.
"""

from __future__ import annotations
from typing import Optional, Sequence

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
import pyqtgraph.exporters  # noqa: F401

from ..core.theme import get_colormap


class MIDASImageView(QtWidgets.QWidget):
    cursorMoved = QtCore.pyqtSignal(float, float, float)
    frameScrolled = QtCore.pyqtSignal(int)
    dataStatsUpdated = QtCore.pyqtSignal(float, float, float, float)
    movieFrameAdvance = QtCore.pyqtSignal()
    fileDropped = QtCore.pyqtSignal(str)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None,
                 name: str = 'MIDASImageView', origin: str = 'bl', **kwargs):
        super().__init__(parent)
        self._raw_data: Optional[np.ndarray] = None
        self._log_mode = False
        self._origin = origin

        self._iv = pg.ImageView(parent=self, name=name, view=pg.PlotItem(), **kwargs)
        self._iv.ui.roiBtn.hide()
        self._iv.ui.menuBtn.hide()

        self._nav_mode = 'pointer'
        self._view_history: list = []
        self._view_index = -1

        self._vline = pg.InfiniteLine(angle=90, movable=False,
                                      pen=pg.mkPen('y', width=1, style=QtCore.Qt.DashLine))
        self._hline = pg.InfiniteLine(angle=0, movable=False,
                                      pen=pg.mkPen('y', width=1, style=QtCore.Qt.DashLine))
        self._vline.setZValue(1000)
        self._hline.setZValue(1000)
        self._iv.addItem(self._vline)
        self._iv.addItem(self._hline)
        self._crosshair_visible = True

        self._proxy = pg.SignalProxy(self._iv.scene.sigMouseMoved,
                                     rateLimit=60, slot=self._on_mouse_moved)

        self._overlay_items: list = []
        self._nav_bar = self._build_nav_bar()

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._iv, stretch=1)
        layout.addWidget(self._nav_bar)

        vb = self._get_viewbox()
        vb.setMouseEnabled(x=False, y=False)
        vb.enableAutoRange(False)
        vb.wheelScaleFactor = 0
        self._iv.scene.sigMouseClicked.connect(self._on_scene_clicked)
        self.setAcceptDrops(True)

    # ── Nav toolbar ──────────────────────────────────────────────

    def _build_nav_bar(self) -> QtWidgets.QToolBar:
        bar = QtWidgets.QToolBar()
        bar.setIconSize(QtCore.QSize(20, 20))
        bar.setMovable(False)
        bar.setStyleSheet("QToolBar { spacing: 4px; padding: 2px; }")

        style = QtWidgets.QApplication.style()

        def add_btn(icon, tip, slot, checkable=False):
            b = QtWidgets.QToolButton()
            b.setIcon(style.standardIcon(icon))
            b.setToolTip(tip)
            if checkable:
                b.setCheckable(True)
            b.clicked.connect(slot)
            bar.addWidget(b)
            return b

        self._home_btn = add_btn(QtWidgets.QStyle.SP_DirHomeIcon, 'Home — reset view', self._nav_home)
        self._back_btn = add_btn(QtWidgets.QStyle.SP_ArrowBack, 'Back — previous view', self._nav_back)
        self._fwd_btn = add_btn(QtWidgets.QStyle.SP_ArrowForward, 'Forward — next view', self._nav_forward)
        self._back_btn.setEnabled(False)
        self._fwd_btn.setEnabled(False)
        bar.addSeparator()

        self._pan_btn = add_btn(QtWidgets.QStyle.SP_FileDialogDetailedView, 'Pan — drag to move',
                                lambda: self._set_nav_mode('pan'), checkable=True)
        self._zoom_btn = add_btn(QtWidgets.QStyle.SP_FileDialogContentsView, 'Zoom — drag rectangle',
                                 lambda: self._set_nav_mode('zoom'), checkable=True)
        bar.addSeparator()

        self._play_btn = add_btn(QtWidgets.QStyle.SP_MediaPlay, 'Play', self._movie_play)
        self._pause_btn = add_btn(QtWidgets.QStyle.SP_MediaPause, 'Pause', self._movie_pause)
        self._stop_btn = add_btn(QtWidgets.QStyle.SP_MediaStop, 'Stop', self._movie_stop)
        self._pause_btn.setEnabled(False)
        self._stop_btn.setEnabled(False)

        bar.addWidget(QtWidgets.QLabel("  FPS:"))
        self._fps_spin = QtWidgets.QSpinBox()
        self._fps_spin.setRange(1, 30)
        self._fps_spin.setValue(5)
        self._fps_spin.valueChanged.connect(self._update_movie_timer)
        self._fps_spin.setFixedWidth(50)
        bar.addWidget(self._fps_spin)

        bar.addSeparator()
        self._mode_label = QtWidgets.QLabel("  Mode: Pointer")
        bar.addWidget(self._mode_label)

        self._movie_timer = QtCore.QTimer(self)
        self._movie_timer.timeout.connect(lambda: self.movieFrameAdvance.emit())
        return bar

    def _set_nav_mode(self, mode: str):
        if self._nav_mode == mode:
            mode = 'pointer'
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
        vb = self._get_viewbox()
        entry = (list(vb.viewRange()[0]), list(vb.viewRange()[1]))
        if self._view_index < len(self._view_history) - 1:
            self._view_history = self._view_history[:self._view_index + 1]
        self._view_history.append(entry)
        self._view_index = len(self._view_history) - 1
        self._update_nav_buttons()

    def _nav_home(self):
        self._push_view()
        self._get_viewbox().autoRange()
        self._push_view()

    def _nav_back(self):
        if self._view_index > 0:
            self._view_index -= 1
            xr, yr = self._view_history[self._view_index]
            self._get_viewbox().setRange(xRange=xr, yRange=yr, padding=0)
            self._update_nav_buttons()

    def _nav_forward(self):
        if self._view_index < len(self._view_history) - 1:
            self._view_index += 1
            xr, yr = self._view_history[self._view_index]
            self._get_viewbox().setRange(xRange=xr, yRange=yr, padding=0)
            self._update_nav_buttons()

    def _update_nav_buttons(self):
        self._back_btn.setEnabled(self._view_index > 0)
        self._fwd_btn.setEnabled(self._view_index < len(self._view_history) - 1)

    def _on_scene_clicked(self, ev):
        if ev.double():
            self._nav_home()

    # ── Movie ────────────────────────────────────────────────────

    def _movie_play(self):
        fps = self._fps_spin.value()
        self._movie_timer.start(int(1000 / fps))
        self._play_btn.setEnabled(False)
        self._pause_btn.setEnabled(True)
        self._stop_btn.setEnabled(True)
        self._mode_label.setText(f"  ▶ Playing ({fps} fps)")

    def _movie_pause(self):
        self._movie_timer.stop()
        self._play_btn.setEnabled(True)
        self._pause_btn.setEnabled(False)
        self._mode_label.setText("  ⏸ Paused")

    def _movie_stop(self):
        self._movie_timer.stop()
        self._play_btn.setEnabled(True)
        self._pause_btn.setEnabled(False)
        self._stop_btn.setEnabled(False)
        self._mode_label.setText("  Mode: Pointer")

    def _update_movie_timer(self, fps: int):
        if self._movie_timer.isActive():
            self._movie_timer.setInterval(int(1000 / max(1, fps)))
            self._mode_label.setText(f"  ▶ Playing ({fps} fps)")

    # ── Drag-and-drop ────────────────────────────────────────────

    def dragEnterEvent(self, ev):
        if ev.mimeData().hasUrls():
            ev.acceptProposedAction()
        else:
            ev.ignore()

    def dropEvent(self, ev):
        urls = ev.mimeData().urls()
        if urls:
            self.fileDropped.emit(urls[0].toLocalFile())

    # ── Public API ───────────────────────────────────────────────

    def set_image_data(self, data: np.ndarray, auto_levels: bool = True,
                       levels: Optional[Sequence[float]] = None):
        self._raw_data = data
        display = self._apply_log(data.T) if self._log_mode else data.T

        finite = display[np.isfinite(display)]
        if finite.size:
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
            self._iv.setImage(display, autoLevels=False, levels=tuple(levels))
        elif auto_levels:
            self._iv.setImage(display, autoLevels=False, levels=(p2, p98))
        else:
            self._iv.setImage(display, autoLevels=False)

        vb = self._iv.getView()
        if self._origin == 'bl':
            vb.invertY(False); vb.invertX(False)
        elif self._origin == 'br':
            vb.invertY(False); vb.invertX(True)
        self._push_view()

    def set_log_mode(self, enabled: bool):
        self._log_mode = enabled
        if self._raw_data is not None:
            vb = self._iv.getView()
            saved = vb.viewRange()
            current = self._iv.getLevels()
            self.set_image_data(self._raw_data, auto_levels=False)
            vb.setRange(xRange=saved[0], yRange=saved[1], padding=0)
            if current is not None:
                lo, hi = current
                if enabled:
                    lo = np.log10(max(lo, 1e-10)); hi = np.log10(max(hi, 1e-10))
                else:
                    lo = 10.0 ** lo; hi = 10.0 ** hi
                self._iv.setLevels(lo, hi)

    def set_colormap(self, name: str):
        cmap = get_colormap(name)
        lut = cmap.getLookupTable(nPts=256)
        self._iv.imageItem.setLookupTable(lut)

    def set_crosshair_visible(self, visible: bool):
        self._crosshair_visible = visible
        self._vline.setVisible(visible)
        self._hline.setVisible(visible)

    def setLevels(self, lo: float, hi: float):
        self._iv.setLevels(lo, hi)

    def add_overlay(self, item):
        self._iv.addItem(item)
        self._overlay_items.append(item)

    def clear_overlays(self):
        for item in self._overlay_items:
            self._iv.removeItem(item)
        self._overlay_items.clear()

    def export_png(self, filename: Optional[str] = None):
        if filename is None:
            filename, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, 'Export Image', '', 'PNG Files (*.png);;All Files (*)')
        if filename:
            exporter = pg.exporters.ImageExporter(self._iv.scene)
            exporter.export(filename)

    def addItem(self, item):
        self._iv.addItem(item)

    def removeItem(self, item):
        self._iv.removeItem(item)

    def getView(self):
        return self._iv.getView()

    def getViewBox(self):
        return self._get_viewbox()

    @property
    def imageItem(self):
        return self._iv.imageItem

    @property
    def scene(self):
        return self._iv.scene

    # ── Internals ───────────────────────────────────────────────

    def _get_viewbox(self):
        return self._iv.getView().getViewBox()

    @staticmethod
    def _apply_log(data: np.ndarray) -> np.ndarray:
        return np.log10(np.clip(data.astype(np.float64), 1e-10, None))

    def _on_mouse_moved(self, evt):
        pos = evt[0]
        vb = self._iv.getView()
        vbox = vb.getViewBox()
        if vb.sceneBoundingRect().contains(pos):
            mp = vbox.mapSceneToView(pos)
            x, y = mp.x(), mp.y()
            self._vline.setPos(x); self._hline.setPos(y)
            val = 0.0
            if self._raw_data is not None:
                ix, iy = int(x + 0.5), int(y + 0.5)
                h, w = self._raw_data.shape[:2]
                if 0 <= iy < h and 0 <= ix < w:
                    val = float(self._raw_data[iy, ix])
            self.cursorMoved.emit(x, y, val)

    def wheelEvent(self, ev):
        if ev.modifiers() & QtCore.Qt.ControlModifier:
            delta = 1 if ev.angleDelta().y() > 0 else -1
            self.frameScrolled.emit(delta)
            ev.accept()
        else:
            ev.ignore()


def add_shortcut(parent, key, callback, context=QtCore.Qt.WindowShortcut):
    sc = QtWidgets.QShortcut(QtGui.QKeySequence(key), parent)
    sc.setContext(context)
    sc.activated.connect(callback)
    return sc
