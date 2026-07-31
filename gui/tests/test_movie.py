#!/usr/bin/env python3
"""Offscreen exercise of the gui_common movie-export machinery.

Run:  QT_QPA_PLATFORM=offscreen python gui/tests/test_movie.py
Covers capture-size stability, even-dim rounding for MP4, all three writer
backends, off-size frame fitting, and the unsupported-extension error path.
"""
import os, sys, tempfile
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PyQt5 import QtWidgets
from gui_common import (MIDASImageView, MovieWriter, qimage_to_rgb_array,
                        apply_theme)

_tmp = tempfile.TemporaryDirectory(prefix='midas_movie_test_')
OUT = _tmp.name

app = QtWidgets.QApplication([])
apply_theme(app, 'light')

view = MIDASImageView()
view.resize(801, 603)          # deliberately odd dims -> exercise even_dims
view.show()
app.processEvents()

ny, nz = 120, 160
yy, xx = np.mgrid[0:ny, 0:nz]


def frame(i):
    r = np.hypot(yy - ny / 2, xx - nz / 2)
    return (1000 * np.exp(-((r - 10 - 2 * i) ** 2) / 20.0) + 5 * i).astype(np.float32)


# ── grab_scene_rgb: shape stability + even dims ──
exp_odd = view.scene_exporter()
exp_even = view.scene_exporter(even_dims=True)
shapes = set()
for i in range(5):
    view.set_image_data(frame(i), auto_levels=False, levels=(0, 1000))
    app.processEvents()
    shapes.add(view.grab_scene_rgb(exp_even).shape)
print('captured shapes  :', shapes)
assert len(shapes) == 1, 'capture size drifted between frames'
h, w, c = shapes.pop()
assert c == 3 and h % 2 == 0 and w % 2 == 0, (h, w, c)
print('even_dims OK     :', (h, w))
print('odd exporter dims:', exp_odd.params['width'], exp_odd.params['height'])

# capture must not be uniformly blank
img = view.grab_scene_rgb(exp_even)
print('capture dtype/rng:', img.dtype, img.min(), img.max(), 'unique>1:', len(np.unique(img)) > 1)
assert len(np.unique(img)) > 1, 'capture is a flat image'

# ── writers ──
for name in ('t.mp4', 't.gif', 'seq.png'):
    path = os.path.join(OUT, name)
    wtr = MovieWriter(path, fps=10)
    for i in range(6):
        view.set_image_data(frame(i), auto_levels=False, levels=(0, 1000))
        app.processEvents()
        wtr.append(view.grab_scene_rgb(exp_even if wtr.needs_even_dims else exp_odd))
    written = wtr.close()
    if wtr.kind == 'stills':
        n = len([f for f in os.listdir(OUT) if f.startswith('seq_') and f.endswith('.png')])
        print(f'{name:10s} kind={wtr.kind:7s} frames={wtr.count} files={n}')
        assert n == 6, n
    else:
        sz = os.path.getsize(path)
        print(f'{name:10s} kind={wtr.kind:7s} frames={wtr.count} bytes={sz}')
        assert sz > 0

# ── mixed frame sizes get fitted, not crashed ──
wtr = MovieWriter(os.path.join(OUT, 'fit.mp4'), fps=10)
wtr.append(np.zeros((100, 200, 3), np.uint8))
wtr.append(np.full((80, 260, 3), 255, np.uint8))   # different shape
print('size-fit OK      : frames=', wtr.count, 'path=', wtr.close())

# ── bad extension is a clean RuntimeError, not a traceback ──
try:
    MovieWriter(os.path.join(OUT, 'x.wav'), 10)
except RuntimeError as e:
    print('bad ext handled  :', str(e).splitlines()[0])

# ── read the mp4 back and confirm frame count + size ──
import cv2
cap = cv2.VideoCapture(os.path.join(OUT, 't.mp4'))
n = 0
while True:
    ok, fr = cap.read()
    if not ok:
        break
    n += 1
    last = fr.shape
cap.release()
print('mp4 readback     : frames=', n, 'shape=', last, 'fps=', 10)
assert n == 6, n

print('\nALL CHECKS PASSED')
