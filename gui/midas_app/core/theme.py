"""Theme + colormap helpers (PyQt5)."""

from PyQt5 import QtGui, QtWidgets
import pyqtgraph as pg

COLORMAPS = ['viridis', 'inferno', 'plasma', 'magma', 'turbo',
             'gray', 'gray_r', 'hot', 'cool', 'bone']


def apply_theme(app: QtWidgets.QApplication, theme: str = 'light') -> None:
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


def get_colormap(name: str):
    try:
        return pg.colormap.get(name)
    except Exception:
        try:
            return pg.colormap.getFromMatplotlib(name)
        except Exception:
            return pg.colormap.get('viridis')
