"""Generic background worker thread (PyQt5)."""

from PyQt5 import QtCore


class AsyncWorker(QtCore.QThread):
    finished_signal = QtCore.pyqtSignal(object)
    error_signal = QtCore.pyqtSignal(str)
    progress_signal = QtCore.pyqtSignal(int, int)

    def __init__(self, target=None, args=(), parent=None):
        super().__init__(parent)
        self._target = target
        self._args = args

    def run(self):
        try:
            result = self._target(*self._args)
            self.finished_signal.emit(result)
        except Exception as e:
            import traceback
            self.error_signal.emit(f"{e}\n{traceback.format_exc()}")
