"""Sortable peak parameter table with row-selection signal (PyQt5)."""

from __future__ import annotations
from typing import List, Optional

import pandas as pd
from PyQt5 import QtCore, QtWidgets


class PeakTable(QtWidgets.QTableWidget):
    rowSelected = QtCore.pyqtSignal(int)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.setAlternatingRowColors(True)
        self.setSortingEnabled(True)
        self.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.itemSelectionChanged.connect(self._on_selection_changed)
        self._df: Optional[pd.DataFrame] = None

    def set_dataframe(self, df: pd.DataFrame, columns: Optional[List[str]] = None):
        cols = columns if columns else list(df.columns)
        cols = [c for c in cols if c in df.columns]
        self._df = df[cols].reset_index(drop=True)

        self.setSortingEnabled(False)
        self.clear()
        self.setRowCount(len(self._df))
        self.setColumnCount(len(cols))
        self.setHorizontalHeaderLabels(cols)

        for r in range(len(self._df)):
            for c, col in enumerate(cols):
                v = self._df.iloc[r, c]
                item = _NumericItem(v) if _is_numeric(v) else QtWidgets.QTableWidgetItem(str(v))
                self.setItem(r, c, item)

        self.resizeColumnsToContents()
        self.setSortingEnabled(True)

    def selected_row(self) -> Optional[int]:
        rows = self.selectionModel().selectedRows()
        if not rows:
            return None
        return rows[0].row()

    def _on_selection_changed(self):
        r = self.selected_row()
        if r is not None:
            self.rowSelected.emit(r)


class _NumericItem(QtWidgets.QTableWidgetItem):
    def __init__(self, value):
        try:
            f = float(value)
        except (TypeError, ValueError):
            f = 0.0
        super().__init__(_fmt(f))
        self._value = f

    def __lt__(self, other):
        if isinstance(other, _NumericItem):
            return self._value < other._value
        return super().__lt__(other)


def _is_numeric(v) -> bool:
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


def _fmt(f: float) -> str:
    if f == 0:
        return "0"
    a = abs(f)
    if a < 1e-3 or a >= 1e6:
        return f"{f:.4e}"
    if a >= 100:
        return f"{f:.2f}"
    return f"{f:.4f}"
