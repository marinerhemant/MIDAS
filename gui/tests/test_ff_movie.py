#!/usr/bin/env python3
"""Drive the real FF viewer offscreen: freeze/autoscale + full movie export.

Run:  QT_QPA_PLATFORM=offscreen python gui/tests/test_ff_movie.py
Builds a synthetic .ge3 stack whose brightness ramps hard across frames, so
frozen vs. unfrozen intensity scaling is unmistakable, then exports a movie
through the real _save_movie path with the dialog auto-accepted.
"""
import os, sys, tempfile
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PyQt5 import QtWidgets, QtCore

_tmp = tempfile.TemporaryDirectory(prefix='midas_ff_movie_test_')
OUT = _tmp.name
DATA = os.path.join(OUT, 'ffdata')
os.makedirs(DATA, exist_ok=True)

NY, NZ, NFR = 128, 128, 12
yy, xx = np.mgrid[0:NY, 0:NZ]
r = np.hypot(yy - NY / 2, xx - NZ / 2)

# One GE-style raw file: NFR frames, uint16, no header.
# Brightness ramps hard across frames so freeze-vs-autoscale is unmistakable.
stack = np.empty((NFR, NY, NZ), np.uint16)
for i in range(NFR):
    ring = 3000 * np.exp(-((r - 20) ** 2) / 8.0)
    stack[i] = np.clip(ring * (0.2 + 0.8 * i / (NFR - 1)) + 10 * i, 0, 65535)
raw = os.path.join(DATA, 'synth_000001.ge3')
stack.tofile(raw)

import ff_asym_qt as ff
from gui_common import MovieExportDialog

app = QtWidgets.QApplication([])
ff.apply_theme(app, 'light')
win = ff.FFViewer()
win.resize(1000, 750)
win.show()
app.processEvents()

# Point it at the synthetic stack the same way _open_single_file would.
win.folder = DATA + '/'
win.file_stem = 'synth'
win.ext = '.ge3'
win.padding = 6
win.det_nr = 3
win.first_file_nr = 1
win.header_size = 0
win.bytes_per_pixel = 2
win.ny, win.nz = NY, NZ
win.n_frames_per_file = NFR
win.nframes_edit.setText(str(NFR))
win.ny_edit.setText(str(NY))
win.nz_edit.setText(str(NZ))
win.header_edit.setText('0')
win.bpp_edit.setText('2')
win.frame_spin.setValue(0)
win._load_and_display()
app.processEvents()
print('loaded frame0 max =', float(win.bdata.max()))
assert win.bdata.max() > 0, 'no data loaded'

def levels():
    return (float(win.min_intensity_edit.text()), float(win.max_intensity_edit.text()))

# ── FREEZE ON (default): levels must not move across frames ──
assert win.freeze_levels_check.isChecked(), 'Freeze should default to on'
win.min_intensity_edit.setText('0')
win.max_intensity_edit.setText('1500')
win._apply_intensity_levels()
seen = set()
for f in range(NFR):
    win.frame_spin.setValue(f)
    app.processEvents()
    seen.add(levels())
print('frozen levels across frames:', seen)
assert seen == {(0.0, 1500.0)}, seen

# ── FREEZE OFF: levels must track each frame ──
win.freeze_levels_check.setChecked(False)
app.processEvents()
unfrozen = []
for f in range(NFR):
    win.frame_spin.setValue(f)
    app.processEvents()
    unfrozen.append(levels()[1])
print('unfrozen MaxI per frame   :', unfrozen)
assert len(set(unfrozen)) > 1, 'levels did not follow the data'
assert unfrozen[-1] > unfrozen[0], 'MaxI should rise with the ramp'

# ── Autoscale button while frozen ──
win.freeze_levels_check.setChecked(True)
win.min_intensity_edit.setText('0')
win.max_intensity_edit.setText('99999')
win._apply_intensity_levels()
win.frame_spin.setValue(NFR - 1)
app.processEvents()
before = levels()
win._autoscale_levels()
after = levels()
print('autoscale: %s -> %s' % (before, after))
assert after != before and after[1] < 99999, (before, after)
assert win.freeze_levels_check.isChecked(), 'Autoscale must not unfreeze'

# ── log mode: stats must arrive linear, not log10 ──
win.log_check.setChecked(True)
win.freeze_levels_check.setChecked(False)
win.frame_spin.setValue(NFR - 1)
app.processEvents()
log_max = levels()[1]
win.log_check.setChecked(False)
win.frame_spin.setValue(NFR - 1)
app.processEvents()
lin_max = levels()[1]
print('log-mode MaxI=%s   linear-mode MaxI=%s' % (log_max, lin_max))
assert log_max > 100, 'log mode leaked log10 values into MinI/MaxI'
win.freeze_levels_check.setChecked(True)

# ── Full movie export through the real code path (dialog auto-accepted) ──
mp4 = os.path.join(OUT, 'ff_movie.mp4')
orig_exec = MovieExportDialog.exec_
def fake_exec(self):
    self.start_spin.setValue(2)
    self.count_spin.setValue(8)
    self.step_spin.setValue(1)
    self.fps_spin.setValue(5)
    self.path_edit.setText(mp4)
    print('dialog summary:', self._summary.text().replace('\n', ' | '))
    return QtWidgets.QDialog.Accepted
MovieExportDialog.exec_ = fake_exec
QtWidgets.QMessageBox.information = staticmethod(lambda *a, **k: print('  [info]', a[2].replace('\n', ' ')))
QtWidgets.QMessageBox.critical = staticmethod(lambda *a, **k: print('  [ERR ]', a[2].replace('\n', ' ')))

win.frame_spin.setValue(5)
app.processEvents()
win._save_movie()
MovieExportDialog.exec_ = orig_exec

assert os.path.exists(mp4), 'no movie written'
import cv2
cap = cv2.VideoCapture(mp4)
frames = []
while True:
    ok, fr = cap.read()
    if not ok:
        break
    frames.append(fr)
cap.release()
print('mp4: %d frames, %s, fps=%.1f' % (len(frames), frames[0].shape,
                                        cv2.VideoCapture(mp4).get(cv2.CAP_PROP_FPS)))
assert len(frames) == 8, len(frames)

# Frames must differ from one another (i.e. the viewer really advanced)
diffs = [float(np.abs(frames[i].astype(int) - frames[i - 1].astype(int)).mean())
         for i in range(1, len(frames))]
print('mean |frame diff|:', [round(d, 2) for d in diffs])
assert all(d > 0 for d in diffs), 'captured the same frame repeatedly'

# Viewer restored to where it was
print('frame after export:', win.frame_spin.value())
assert win.frame_spin.value() == 5, win.frame_spin.value()

print('\nALL FF CHECKS PASSED')
