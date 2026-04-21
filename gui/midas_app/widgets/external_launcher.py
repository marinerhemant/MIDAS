"""Generic widget that launches a script in an external process.

Used for tools that don't embed cleanly in a Qt event loop:
  - Tkinter scripts (imageManipulation.py, dt.py)
  - Dash apps (dig_tw.py, pfIntensityViewer.py, interactiveFFplotting.py)
  - Plotly-HTML generators (plotFFSpots3d.py, plotGrains3d.py)

The widget shows a description, optional argument fields, a Launch button,
and a live process-state line. Multiple launches are tracked so users can
spawn several instances and stop them individually.
"""

from __future__ import annotations
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from PyQt5 import QtCore, QtWidgets


@dataclass
class ArgSpec:
    """Description of one parameter to surface in the launch form."""
    label: str
    flag: str                          # e.g. "--lineout", "--port"; "" for positional
    default: str = ""
    placeholder: str = ""
    kind: str = "text"                 # "text" | "file" | "dir" | "int"
    file_filter: str = "All files (*)"
    required: bool = False
    tooltip: str = ""


@dataclass
class _RunningProcess:
    proc: subprocess.Popen
    label: str
    started_at: float = field(default_factory=lambda: __import__('time').time())


class ExternalLauncher(QtWidgets.QWidget):
    """Form + Launch button for a single subprocess-style script.

    Parameters
    ----------
    title       Heading shown at the top.
    description Multi-line context shown under the heading.
    script_path Absolute path to the script to invoke.
    args        Sequence of ArgSpec describing the form fields.
    interpreter argv[0] for the subprocess. Defaults to ``sys.executable``.
    open_url    Optional URL to open after launch (e.g. http://localhost:port).
                Use ``"{<flag>}"`` placeholders to substitute argument values.
    """

    def __init__(self, title: str, description: str, script_path: str,
                 args: Optional[List[ArgSpec]] = None,
                 interpreter: Optional[str] = None,
                 open_url: Optional[str] = None,
                 parent=None):
        super().__init__(parent)
        self._script = script_path
        self._args = args or []
        self._interpreter = interpreter or sys.executable
        self._open_url = open_url
        self._running: List[_RunningProcess] = []
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(2000)
        self._timer.timeout.connect(self._poll_processes)
        self._timer.start()

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(40, 30, 40, 30)
        layout.setSpacing(10)

        layout.addWidget(QtWidgets.QLabel(f"<h3>{title}</h3>"))

        if description:
            d = QtWidgets.QLabel(description)
            d.setWordWrap(True)
            d.setStyleSheet("color: #555; padding-bottom: 8px;")
            layout.addWidget(d)

        # Script path row
        script_row = QtWidgets.QHBoxLayout()
        script_row.addWidget(QtWidgets.QLabel("<b>Script:</b>"))
        path_lbl = QtWidgets.QLabel(f"<tt>{script_path}</tt>")
        path_lbl.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        script_row.addWidget(path_lbl, stretch=1)
        layout.addLayout(script_row)

        # Argument form
        self._field_widgets: Dict[str, QtWidgets.QWidget] = {}
        if self._args:
            form = QtWidgets.QFormLayout()
            form.setHorizontalSpacing(12)
            form.setVerticalSpacing(6)
            for spec in self._args:
                row = QtWidgets.QHBoxLayout()
                row.setContentsMargins(0, 0, 0, 0)
                if spec.kind == "int":
                    w = QtWidgets.QSpinBox()
                    w.setRange(0, 1_000_000)
                    if spec.default:
                        try:
                            w.setValue(int(spec.default))
                        except ValueError:
                            pass
                    row.addWidget(w)
                else:
                    w = QtWidgets.QLineEdit(spec.default)
                    if spec.placeholder:
                        w.setPlaceholderText(spec.placeholder)
                    row.addWidget(w)
                    if spec.kind in ("file", "dir"):
                        b = QtWidgets.QPushButton("Browse…")
                        b.setFixedWidth(90)
                        if spec.kind == "file":
                            b.clicked.connect(
                                lambda _checked, edit=w, s=spec: self._pick_file(edit, s))
                        else:
                            b.clicked.connect(
                                lambda _checked, edit=w, s=spec: self._pick_dir(edit, s))
                        row.addWidget(b)
                if spec.tooltip:
                    w.setToolTip(spec.tooltip)
                container = QtWidgets.QWidget()
                container.setLayout(row)
                lbl = spec.label + (" *" if spec.required else "")
                form.addRow(lbl, container)
                self._field_widgets[spec.label] = w
            layout.addLayout(form)

        # Buttons
        btn_row = QtWidgets.QHBoxLayout()
        self._launch_btn = QtWidgets.QPushButton("▶ Launch")
        self._launch_btn.setFixedWidth(140)
        self._launch_btn.clicked.connect(self._on_launch)
        btn_row.addWidget(self._launch_btn)

        self._stop_all_btn = QtWidgets.QPushButton("Stop all")
        self._stop_all_btn.setFixedWidth(110)
        self._stop_all_btn.clicked.connect(self._stop_all)
        btn_row.addWidget(self._stop_all_btn)
        btn_row.addStretch(1)

        if self._open_url:
            self._open_btn = QtWidgets.QPushButton("Open in browser")
            self._open_btn.setFixedWidth(150)
            self._open_btn.clicked.connect(self._open_url_in_browser)
            btn_row.addWidget(self._open_btn)

        layout.addLayout(btn_row)

        # Running processes list
        layout.addWidget(QtWidgets.QLabel("<b>Running processes:</b>"))
        self._proc_list = QtWidgets.QListWidget()
        self._proc_list.setStyleSheet("font-family: Menlo, monospace; font-size: 11px;")
        layout.addWidget(self._proc_list, stretch=1)

        self._status = QtWidgets.QLabel(" ")
        self._status.setStyleSheet("color: #555; padding: 4px;")
        layout.addWidget(self._status)

    # ── Form helpers ─────────────────────────────────────────────

    def _pick_file(self, edit: QtWidgets.QLineEdit, spec: ArgSpec):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, f"Choose {spec.label}", edit.text() or "", spec.file_filter)
        if path:
            edit.setText(path)

    def _pick_dir(self, edit: QtWidgets.QLineEdit, spec: ArgSpec):
        d = QtWidgets.QFileDialog.getExistingDirectory(
            self, f"Choose {spec.label}", edit.text() or "")
        if d:
            edit.setText(d)

    def _collect_args(self) -> Tuple[List[str], Dict[str, str]]:
        argv: List[str] = []
        values: Dict[str, str] = {}
        for spec in self._args:
            w = self._field_widgets[spec.label]
            val = str(w.value()) if isinstance(w, QtWidgets.QSpinBox) else w.text().strip()
            values[spec.flag.lstrip('-')] = val
            if not val:
                if spec.required:
                    raise ValueError(f"{spec.label} is required")
                continue
            if spec.flag:
                argv.append(spec.flag)
                argv.append(val)
            else:
                argv.append(val)
        return argv, values

    # ── Launch ───────────────────────────────────────────────────

    def _on_launch(self):
        try:
            argv_args, values = self._collect_args()
        except ValueError as e:
            QtWidgets.QMessageBox.warning(self, "Missing argument", str(e))
            return
        if not os.path.isfile(self._script):
            QtWidgets.QMessageBox.critical(
                self, "Script not found", f"{self._script}")
            return

        cmd = [self._interpreter, self._script] + argv_args
        cmd_display = ' '.join(shlex.quote(c) for c in cmd)
        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                cwd=os.path.dirname(self._script) or None)
        except OSError as e:
            QtWidgets.QMessageBox.critical(self, "Launch failed", str(e))
            return

        rp = _RunningProcess(proc=proc, label=cmd_display)
        self._running.append(rp)
        item = QtWidgets.QListWidgetItem(f"PID {proc.pid}: {cmd_display}")
        item.setData(QtCore.Qt.UserRole, proc.pid)
        self._proc_list.addItem(item)
        self._status.setText(f"Started PID {proc.pid}")

        if self._open_url:
            QtCore.QTimer.singleShot(800, lambda: self._open_url_in_browser(values))

    def _open_url_in_browser(self, values: Optional[Dict[str, str]] = None):
        url = self._open_url
        if values:
            try:
                url = url.format(**values)
            except KeyError:
                pass
        import webbrowser
        webbrowser.open(url)

    def _poll_processes(self):
        if not self._running:
            return
        still_running: List[_RunningProcess] = []
        for rp in self._running:
            if rp.proc.poll() is None:
                still_running.append(rp)
            else:
                # Mark item as exited
                for r in range(self._proc_list.count()):
                    item = self._proc_list.item(r)
                    if item.data(QtCore.Qt.UserRole) == rp.proc.pid:
                        item.setText(f"PID {rp.proc.pid} (exit {rp.proc.returncode}): {rp.label}")
                        item.setForeground(QtCore.Qt.gray)
                        break
        self._running = still_running

    def _stop_all(self):
        for rp in self._running:
            try:
                rp.proc.terminate()
            except Exception:
                pass
        self._status.setText(f"Sent terminate to {len(self._running)} process(es)")
