# main_app.py
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
# from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as OriginalNavigationToolbar
from matplotlib.figure import Figure
import time
from pathlib import Path
import traceback # For detailed error reporting
import zarr
from zarr.storage import ZipStore # Correct import
from collections import OrderedDict
import subprocess # For HKL generation
import csv

from PyQt6.QtWidgets import (
    QMainWindow, QApplication, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QLineEdit, QCheckBox, QFileDialog, QStatusBar,
    QTabWidget, QGroupBox, QSizePolicy, QProgressBar, QMessageBox, QScrollArea,
    QListWidget, QAbstractItemView, QDialog, QDialogButtonBox, QTextEdit,
    QSpacerItem, QFormLayout
)
from PyQt6.QtCore import Qt, QThread, QObject, pyqtSignal, QTimer, QSettings, QSize, pyqtSlot
from PyQt6.QtGui import QIntValidator, QDoubleValidator, QFont, QIcon, QAction, QKeySequence

# --- Determine MIDAS_DIR ---
try:
    # Assumes this script is in $MIDAS_DIR/gui/ff/
    FF_GUI_DIR = Path(__file__).parent.resolve()
    GUI_DIR = FF_GUI_DIR.parent.resolve()
    MIDAS_DIR = GUI_DIR.parent.resolve()
    print(f"Determined MIDAS_DIR: {MIDAS_DIR}")
except NameError:
    # __file__ is not defined, e.g., when running in an interactive interpreter
    # Fallback to current working directory or prompt user?
    MIDAS_DIR = Path.cwd()
    print(f"Warning: Could not determine MIDAS_DIR from __file__. Using CWD: {MIDAS_DIR}")

# Import local modules
from utils import colors, try_parse_float, try_parse_int # Assuming utils.py is accessible
from data_handler import (read_parameters, load_image_frame, get_max_projection,
                         load_dark_frame, get_file_path, generate_big_detector_mask,
                         load_big_detector_mask)
from plotting import PlottingHandler # Now includes get_transform_matrix via utils

class CustomNavigationToolbar(OriginalNavigationToolbar):
    def __init__(self, canvas, parent, coordinates=True):
        super().__init__(canvas, parent, coordinates)
        self._coordinates_on = coordinates # Store the initial state

    def set_message(self, msg):
        # Override to do nothing, preventing default coordinate display
        # Or, if you want to allow it to be toggled:
        # if self._coordinates_on and self.parent().statusBar():
        #    self.parent().statusBar().showMessage(msg)
        pass # Effectively disables the toolbar's own status messages

    # Optional: if you want to be able to toggle the default display later
    def enable_default_coordinates(self, enable):
        self._coordinates_on = enable

# --- Worker Thread ---
class Worker(QObject):
    finished = pyqtSignal(object, str, object) # (result, task_id, worker_data)
    error = pyqtSignal(str, str)    # error message, task_id
    progress = pyqtSignal(int, str) # progress percentage, task_id

    def __init__(self, task_func, task_id, *args, **kwargs):
        super().__init__()
        self.task_func = task_func
        self.task_id = task_id
        self.args = args
        self.kwargs = kwargs
        self.is_running = True
        self.worker_data = {}

    def run(self):
        try:
            import inspect
            sig = inspect.signature(self.task_func)
            task_kwargs = self.kwargs.copy()
            if 'worker_data_dict' in sig.parameters:
                task_kwargs['worker_data_dict'] = self.worker_data

            result = self.task_func(*self.args, **task_kwargs)

            if self.is_running:
                self.finished.emit(result, self.task_id, self.worker_data)
        except Exception as e:
            if self.is_running:
                error_msg = f"Error in task '{self.task_id}': {traceback.format_exc()}"
                self.error.emit(error_msg, self.task_id)

    def stop(self):
        print(f"Worker task '{self.task_id}' received stop signal.")
        self.is_running = False


# --- Ring Material Parameter Dialog ---
class RingMaterialDialog(QDialog):
    def __init__(self, current_params_for_dialog, parent=None): # Renamed arg for clarity
        super().__init__(parent)
        self.setWindowTitle("Material & Ring Parameters")
        self.layout = QVBoxLayout(self)
        self.form_layout = QFormLayout()

        # Use current_params_for_dialog which should have Zarr values prioritized
        self.le_sg = QLineEdit(str(current_params_for_dialog.get('sg', 225)))
        self.le_sg.setValidator(QIntValidator(1, 230))
        self.le_sg.setToolTip("Crystallographic Space Group Number (1-230)")

        self.le_lat_const = [QLineEdit() for _ in range(6)]
        lc_values = current_params_for_dialog.get('LatticeConstant', np.zeros(6))
        lat_labels = ["a (Å)", "b (Å)", "c (Å)", "\u03B1 (\N{DEGREE SIGN})", "\u03B2 (\N{DEGREE SIGN})", "\u03B3 (\N{DEGREE SIGN})"]
        lat_const_layout = QGridLayout()
        for i, le in enumerate(self.le_lat_const):
             le.setText(f"{try_parse_float(lc_values[i]):.5f}")
             le.setValidator(QDoubleValidator())
             row, col = divmod(i, 3)
             lat_const_layout.addWidget(QLabel(lat_labels[i]), row, col*2)
             lat_const_layout.addWidget(le, row, col*2 + 1)
        lat_const_layout.setColumnStretch(1, 1); lat_const_layout.setColumnStretch(3, 1); lat_const_layout.setColumnStretch(5, 1)

        self.le_wl = QLineEdit(f"{try_parse_float(current_params_for_dialog.get('wl', 0.1729)):.6f}")
        self.le_wl.setValidator(QDoubleValidator(0.01, 10.0, 6))
        self.le_wl.setToolTip("X-ray Wavelength (Angstrom)")

        px_val = try_parse_float(current_params_for_dialog.get('px', 200.0))
        self.le_px = QLineEdit(f"{px_val:.2f}")
        self.le_px.setValidator(QDoubleValidator(0.1, 1000.0, 2))
        self.le_px.setToolTip("Detector Pixel Size (microns)")

        # For LSD in dialog, try DetParams[0] first, then general 'lsd', then default
        lsd_val = 1e6 # Default
        det_params_list_dialog = current_params_for_dialog.get('DetParams', [{}])
        if det_params_list_dialog and 'lsd' in det_params_list_dialog[0]:
            lsd_val = try_parse_float(det_params_list_dialog[0]['lsd'], lsd_val)
        elif 'lsd' in current_params_for_dialog: # Fallback to top-level 'lsd' if in Zarr
            lsd_val = try_parse_float(current_params_for_dialog.get('lsd'), lsd_val)

        self.le_lsd = QLineEdit(f"{lsd_val:.1f}")
        self.le_lsd.setValidator(QDoubleValidator(1.0, 10e7, 1))
        self.le_lsd.setToolTip("Sample-to-Detector Distance (microns)")

        max_rad_val = try_parse_float(current_params_for_dialog.get('maxRad', 2e6))
        self.le_max_rad = QLineEdit(f"{max_rad_val:.1f}")
        self.le_max_rad.setValidator(QDoubleValidator(1.0, 10e7, 1))
        self.le_max_rad.setToolTip("Maximum Ring Radius for HKL generation (microns)")

        self.form_layout.addRow("Space Group:", self.le_sg)
        self.form_layout.addRow("Lattice Constants:", lat_const_layout)
        self.form_layout.addRow("Wavelength (Å):", self.le_wl)
        self.form_layout.addRow("Pixel Size (\u00B5m):", self.le_px)
        self.form_layout.addRow("LSD (\u00B5m):", self.le_lsd)
        self.form_layout.addRow("Max Ring Rad (\u00B5m):", self.le_max_rad)
        self.layout.addLayout(self.form_layout)

        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.button_box.accepted.connect(self.accept); self.button_box.rejected.connect(self.reject)
        self.layout.addWidget(self.button_box)
    def get_values(self):
        try:
            lat_const = [try_parse_float(le.text()) for le in self.le_lat_const]
            return {'sg': try_parse_int(self.le_sg.text(), 225),
                    'LatticeConstant': np.array(lat_const),
                    'wl': try_parse_float(self.le_wl.text(), 0.1729),
                    'px': try_parse_float(self.le_px.text(), 200.0),
                    'lsd_dialog': try_parse_float(self.le_lsd.text(), 1e6), # Use a distinct key
                    'maxRad': try_parse_float(self.le_max_rad.text(), 2e6)}
        except Exception as e: print(f"Error getting values from dialog: {e}"); return {}

# --- Zarr Parameter Viewer Dialog ---
class ZarrViewerDialog(QDialog):
     def __init__(self, zarr_file_path, parent=None):
          super().__init__(parent)
          self.setWindowTitle(f"Zarr Parameters - {Path(zarr_file_path).name}")
          self.setMinimumSize(600, 400)
          self.layout = QVBoxLayout(self)

          self.text_edit = QTextEdit()
          self.text_edit.setReadOnly(True)
          self.text_edit.setFont(QFont("Monospace"))
          self.layout.addWidget(self.text_edit)

          self.populate_parameters(zarr_file_path)

          self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
          self.button_box.accepted.connect(self.accept)
          self.button_box.rejected.connect(self.reject)
          close_button = self.button_box.button(QDialogButtonBox.StandardButton.Close)
          if close_button:
              close_button.clicked.connect(self.accept)

          self.layout.addWidget(self.button_box)

     def populate_parameters(self, zarr_file_path):
          param_text = f"Parameters from: {zarr_file_path}\n" + "="*40 + "\n"
          store = None
          try:
               store = ZipStore(zarr_file_path, mode='r')
               z_root = zarr.open_group(store=store, mode='r')
               params_found = []

               def recursive_find_arrays(group, path_prefix=""):
                   # Iterate through arrays in the current group
                   for name, arr in group.arrays():
                       full_path = f"{path_prefix}/{name}" if path_prefix else name
                       # Check if path matches desired parameter locations
                       if ('analysis/process/analysis_parameters' in full_path or \
                           'measurement/process/scan_parameters' in full_path or \
                           'measurement/instrument/' in full_path):
                           try:
                                value = arr[...] # Read data
                                if value.size == 1:
                                     val_str = str(value.item())
                                     if isinstance(value.item(), bytes):
                                          try:
                                              val_str = value.item().decode('utf-8', errors='replace')
                                          except Exception:
                                              pass # Keep raw bytes str on decode error
                                elif value.ndim == 1 and value.size <= 10:
                                    val_str = np.array2string(value, separator=', ')
                                else:
                                    val_str = f"[Array shape={value.shape}, dtype={value.dtype}]"
                                params_found.append(f"{full_path}: {val_str}")
                           except Exception as read_err:
                                params_found.append(f"{full_path}: [Error reading: {read_err}]")
                   # Recursively visit subgroups
                   for name, subgroup in group.groups():
                       new_prefix = f"{path_prefix}/{name}" if path_prefix else name
                       recursive_find_arrays(subgroup, new_prefix)

               recursive_find_arrays(z_root) # Start traversal from root

               if params_found:
                    param_text += "\n".join(sorted(params_found))
               else:
                    param_text += "\nNo parameters found in standard locations."

               param_text += "\n\n" + "="*40 + "\nFull Tree:\n" + str(z_root.tree())

          except Exception as e:
               param_text += f"\n\nError reading Zarr file: {e}"
               print(f"Error processing Zarr parameters: {traceback.format_exc()}")
          finally:
               if store is not None and hasattr(store, 'mode') and store.mode != 'closed':
                    store.close()
          self.text_edit.setText(param_text)


# --- Main Application Window ---
class FFViewerApp(QMainWindow):
    ORG_NAME = "MIDAS-HEDM"
    APP_NAME = "FFViewer"

    @pyqtSlot(str) # Add this import from PyQt6.QtCore if not already there
    def do_nothing_with_toolbar_message(self, message):
        """Dummy slot to intercept and ignore toolbar's own messages."""
        pass

    def set_hydra_controls_enabled(self, enabled):
        """Enable/disable Hydra controls based on parameter loading."""
        is_param_loaded = hasattr(self, 'params') and bool(self.params.get('paramFN'))
        param_file_exists = is_param_loaded and Path(self.params['paramFN']).exists() if self.params.get('paramFN') else False
        hydra_active = hasattr(self, 'params') and self.params.get('HydraActive', False)

        if hasattr(self, 'btn_make_big_det'):
             self.btn_make_big_det.setEnabled(enabled and param_file_exists and hydra_active)
        if hasattr(self, 'btn_load_multi'):
             self.btn_load_multi.setEnabled(enabled and param_file_exists and hydra_active)
        if hasattr(self, 'cb_sep_folders'):
             self.cb_sep_folders.setEnabled(enabled and param_file_exists and hydra_active)
        if hasattr(self, 'btn_load_params'):
             self.btn_load_params.setEnabled(enabled and bool(self.current_param_file))

    def set_ui_enabled(self, enabled):
        """Enable/disable relevant UI elements during processing."""
        widgets_to_toggle = [
            self.btn_quit, self.btn_browse_data, self.btn_browse_dark, self.btn_browse_param,
            self.le_frame_nr, self.btn_prev_frame, self.btn_next_frame,
            self.btn_update_thresh, self.cb_log_scale, self.cb_hflip, self.cb_vflip,
            self.cb_transpose, self.cb_plot_rings, self.btn_select_rings,
            self.le_det_num_single, self.le_lsd_single, self.le_bcx_single,
            self.le_bcy_single, self.btn_update_ring_geom,
            self.cb_use_max_proj, self.le_max_frames, self.le_max_start_frame,
            self.btn_load_max_proj, self.btn_load_params,
            self.btn_make_big_det, self.btn_load_multi, self.cb_sep_folders,
            self.cb_dark_correct, self.btn_view_zarr
        ]
        for widget in widgets_to_toggle:
             if hasattr(self, widget.objectName()) and widget: # Check it exists
                 widget.setEnabled(enabled)

        # Special handling when re-enabling UI
        if enabled:
            # Hydra controls depend on whether params are loaded
            self.set_hydra_controls_enabled(True) # Let the method check internal state
            # Zarr view button depends on mode
            self.btn_view_zarr.setEnabled(self.is_zarr_mode)
            # Frame navigation depends on max proj mode
            if hasattr(self, 'cb_use_max_proj'):
                is_max_mode = self.cb_use_max_proj.isChecked()
                self.le_frame_nr.setEnabled(not is_max_mode)
                self.btn_prev_frame.setEnabled(not is_max_mode)
                self.btn_next_frame.setEnabled(not is_max_mode)
            else: # Fallback if max proj checkbox doesn't exist yet
                if hasattr(self, 'le_frame_nr'):
                    self.le_frame_nr.setEnabled(True)
                if hasattr(self, 'btn_prev_frame'):
                    self.btn_prev_frame.setEnabled(True)
                if hasattr(self, 'btn_next_frame'):
                    self.btn_next_frame.setEnabled(True)

    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{self.APP_NAME} - Far-Field Data Viewer")
        self.setGeometry(100, 100, 1400, 850)

        # --- State Variables ---
        self.params = {}
        self.current_data_single = None
        self.current_data_multi = None
        self.current_file_info = {}
        self.dark_frame_cache = {}
        self.current_frame_nr = 0
        self.current_frame_nr_pending = 0
        self.total_frames_available = 0
        self.current_data_file = ""
        self.current_dark_file = ""
        self.current_param_file = ""
        self.is_zarr_mode = False
        self.worker_thread = None
        self.worker = None
        self.midas_dir = MIDAS_DIR
        self.last_browse_dirs = {}

        # --- Initialize UI ---
        self.plotting_handler = None # Initialize plotting handler ref
        self._init_ui()
        self._connect_signals()

        # --- Initialize Plotting ---
        if self.plotting_handler is None: # Should be None here
            self.plotting_handler = PlottingHandler(
                 self.ax_single, self.ax_multi, self.canvas, self.toolbar, self.update_status_bar
             )

        # --- Default Values & Load Settings---
        self.zarr_params_cache = {}
        self._set_default_values()
        self._load_settings() # Load AFTER UI elements are created

        print("Initialization complete.")
        self.update_status_bar("Ready. Select a file or parameter file to begin.")
        if self.is_zarr_mode and self.current_data_file and Path(self.current_data_file).exists():
             self._read_params_from_zarr_to_cache()
             self._merge_zarr_cache_into_params() 
             self._update_gui_from_params() # Update GUI after potential Zarr param read


    def _init_ui(self):
        """Creates the main UI layout and widgets."""
        print("Initializing UI...")
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # --- Top Controls ---
        top_controls_layout = QGridLayout()
        self._create_file_controls(top_controls_layout) # Row 0 widgets
        self._create_frame_controls(top_controls_layout) # Row 1 widgets
        self.btn_quit = QPushButton("Quit")
        self.btn_quit.setToolTip("Exit the application (saves settings)")
        top_controls_layout.addWidget(self.btn_quit, 0, 9)
        self.btn_view_zarr = QPushButton("View Zarr Params")
        self.btn_view_zarr.setToolTip("View parameters stored inside the loaded Zarr file")
        self.btn_view_zarr.setEnabled(False)
        top_controls_layout.addWidget(self.btn_view_zarr, 1, 7)
        main_layout.addLayout(top_controls_layout)

        # --- Tab Widget for Plots ---
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)

        # --- Matplotlib Canvas Setup ---
        self.figure = Figure(figsize=(12, 6), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.canvas.updateGeometry()
        self.ax_single = None # Axes created/managed by _switch_axes
        self.ax_multi = None

        # --- Canvas Container ---
        self.canvas_container = QWidget()
        layout_canvas = QVBoxLayout(self.canvas_container)
        layout_canvas.setContentsMargins(0,0,0,0)
        layout_canvas.addWidget(self.canvas)
        # IMPORTANT: Create toolbar here
        # self.toolbar = NavigationToolbar(self.canvas, self)
        self.toolbar = CustomNavigationToolbar(self.canvas, self) # Changed here
        # self.toolbar.message.connect(self.do_nothing_with_toolbar_message)
        layout_canvas.addWidget(self.toolbar)

        # --- Tabs ---
        tab_single = QWidget()
        layout_single = QVBoxLayout(tab_single)
        # Don't add canvas here initially, _switch_axes handles placement
        self.tab_widget.addTab(tab_single, "Single Detector")

        tab_multi = QWidget()
        layout_multi = QVBoxLayout(tab_multi)
        self.multi_placeholder_label = QLabel("Load parameters using the 'Hydra Controls' to enable this view.")
        self.multi_placeholder_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout_multi.addWidget(self.multi_placeholder_label) # Add placeholder initially
        self.tab_widget.addTab(tab_multi, "Multiple Detectors (Hydra)")

        # --- Bottom Control Panel ---
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFixedHeight(280)
        main_layout.addWidget(scroll_area)
        control_panel_widget = QWidget()
        scroll_area.setWidget(control_panel_widget)
        control_panel_layout = QHBoxLayout(control_panel_widget)
        gb_display = QGroupBox("Display Options")
        gb_rings = QGroupBox("Ring Analysis")
        gb_hydra = QGroupBox("Hydra Controls")
        gb_max = QGroupBox("Max Projection")
        control_panel_layout.addWidget(gb_display, 1)
        control_panel_layout.addWidget(gb_rings, 1)
        control_panel_layout.addWidget(gb_max, 1)
        control_panel_layout.addWidget(gb_hydra, 1)
        self._create_display_controls(gb_display)
        self._create_ring_controls(gb_rings)
        self._create_max_projection_controls(gb_max)
        self._create_hydra_controls(gb_hydra) # Calls set_hydra_controls_enabled

        # --- Status Bar ---
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumSize(200, 15)
        self.progress_bar.setVisible(False)
        self.progress_bar.setToolTip("Progress of background tasks")
        self.status_bar.addPermanentWidget(self.progress_bar)

        # Initial UI state - IMPORTANT: Call *after* all UI elements created
        self._on_tab_changed(0)


    def _create_file_controls(self, layout):
        row = 0
        layout.addWidget(QLabel("Data File:"), row, 0)
        self.le_data_file = QLineEdit()
        self.le_data_file.setReadOnly(True)
        self.le_data_file.setToolTip("Path to the loaded data file (GE, raw, Zarr)")
        layout.addWidget(self.le_data_file, row, 1, 1, 3)
        self.btn_browse_data = QPushButton("Browse...")
        self.btn_browse_data.setToolTip("Browse for a data file (.ge*, .raw, .bin, .zip)")
        layout.addWidget(self.btn_browse_data, row, 4)
        layout.addWidget(QLabel("Dark File:"), row, 5)
        self.le_dark_file = QLineEdit()
        self.le_dark_file.setReadOnly(True)
        self.le_dark_file.setToolTip("Path to the dark file (optional)")
        layout.addWidget(self.le_dark_file, row, 6)
        self.btn_browse_dark = QPushButton("Browse...")
        self.btn_browse_dark.setToolTip("Browse for a dark correction file")
        layout.addWidget(self.btn_browse_dark, row, 7)
        self.cb_dark_correct = QCheckBox("Apply Dark Corr.")
        self.cb_dark_correct.setToolTip("Subtract the dark file from the data file")
        layout.addWidget(self.cb_dark_correct, row, 8)

    def _create_frame_controls(self, layout):
        row = 1
        layout.addWidget(QLabel("Frame Nr:"), row, 0)
        self.le_frame_nr = QLineEdit("0")
        self.le_frame_nr.setValidator(QIntValidator(0, 999999))
        self.le_frame_nr.setFixedWidth(70)
        self.le_frame_nr.setToolTip("Current frame number being displayed (0-indexed)")
        layout.addWidget(self.le_frame_nr, row, 1)
        self.btn_prev_frame = QPushButton("<")
        self.btn_prev_frame.setFixedWidth(30)
        self.btn_prev_frame.setToolTip("Go to previous frame (Ctrl+Left)")
        self.btn_prev_frame.setShortcut(QKeySequence(Qt.Modifier.CTRL | Qt.Key.Key_Left))
        layout.addWidget(self.btn_prev_frame, row, 2)
        self.btn_next_frame = QPushButton(">")
        self.btn_next_frame.setFixedWidth(30)
        self.btn_next_frame.setToolTip("Go to next frame (Ctrl+Right)")
        self.btn_next_frame.setShortcut(QKeySequence(Qt.Modifier.CTRL | Qt.Key.Key_Right))
        layout.addWidget(self.btn_next_frame, row, 3)
        self.lbl_total_frames = QLabel("/ 0")
        self.lbl_total_frames.setToolTip("Total number of frames available in the current context")
        layout.addWidget(self.lbl_total_frames, row, 4)
        self.lbl_omega_display = QLabel("Omega: N/A")
        self.lbl_omega_display.setToolTip("Calculated Omega angle for the current frame (if available)")
        layout.addWidget(self.lbl_omega_display, row, 5, 1, 2)

    def _create_display_controls(self, group_box):
        layout = QGridLayout(group_box)
        layout.addWidget(QLabel("Min Thresh:"), 0, 0)
        self.le_min_thresh = QLineEdit("0")
        self.le_min_thresh.setValidator(QDoubleValidator())
        self.le_min_thresh.setToolTip("Minimum intensity value for display scaling")
        layout.addWidget(self.le_min_thresh, 0, 1)
        layout.addWidget(QLabel("Max Thresh:"), 1, 0)
        self.le_max_thresh = QLineEdit("2000")
        self.le_max_thresh.setValidator(QDoubleValidator())
        self.le_max_thresh.setToolTip("Maximum intensity value for display scaling")
        layout.addWidget(self.le_max_thresh, 1, 1)
        self.btn_update_thresh = QPushButton("Update Thresholds")
        self.btn_update_thresh.setToolTip("Apply the min/max threshold values to the current view")
        layout.addWidget(self.btn_update_thresh, 2, 0, 1, 2)
        self.cb_log_scale = QCheckBox("Log Scale")
        self.cb_log_scale.setToolTip("Display image intensity on a logarithmic scale")
        layout.addWidget(self.cb_log_scale, 3, 0)
        self.cb_hflip = QCheckBox("HFlip")
        self.cb_hflip.setToolTip("Flip image horizontally (left-right)")
        layout.addWidget(self.cb_hflip, 4, 0)
        self.cb_vflip = QCheckBox("VFlip")
        self.cb_vflip.setToolTip("Flip image vertically (up-down)")
        layout.addWidget(self.cb_vflip, 4, 1)
        self.cb_transpose = QCheckBox("Transpose")
        self.cb_transpose.setToolTip("Transpose image axes (swap rows and columns)")
        layout.addWidget(self.cb_transpose, 5, 0, 1, 2)
        layout.setRowStretch(6, 1)

    def _create_ring_controls(self, group_box):
        layout = QVBoxLayout(group_box)
        self.cb_plot_rings = QCheckBox("Plot Rings")
        self.cb_plot_rings.setToolTip("Overlay calculated diffraction rings on the image")
        layout.addWidget(self.cb_plot_rings)
        self.btn_select_rings = QPushButton("Select Rings/Material...")
        self.btn_select_rings.setToolTip("Define material parameters and select which diffraction rings to display")
        layout.addWidget(self.btn_select_rings)
        self.lbl_ring_info = QLabel("Selected Rings: None")
        self.lbl_ring_info.setWordWrap(True)
        self.lbl_ring_info.setToolTip("Information about the currently selected rings for plotting")
        layout.addWidget(self.lbl_ring_info)
        ring_geom_group = QGroupBox("Single Detector Geometry (for Rings)")
        ring_geom_layout = QFormLayout(ring_geom_group)
        self.le_det_num_single = QLineEdit("1")
        self.le_det_num_single.setValidator(QIntValidator(1, 99))
        self.le_det_num_single.setToolTip("Detector number (1-indexed) currently shown in the 'Single Detector' tab")
        ring_geom_layout.addRow("Det Num:", self.le_det_num_single)
        self.le_lsd_single = QLineEdit("1000000")
        self.le_lsd_single.setValidator(QDoubleValidator())
        self.le_lsd_single.setToolTip("Sample-to-Detector Distance (microns) for the current single detector")
        ring_geom_layout.addRow("LSD (\u00B5m):", self.le_lsd_single)
        bc_layout = QHBoxLayout()
        self.le_bcx_single = QLineEdit("1024")
        self.le_bcx_single.setValidator(QDoubleValidator())
        self.le_bcx_single.setToolTip("Beam center X coordinate (column, pixels)")
        bc_layout.addWidget(self.le_bcx_single)
        self.le_bcy_single = QLineEdit("1024")
        self.le_bcy_single.setValidator(QDoubleValidator())
        self.le_bcy_single.setToolTip("Beam center Y coordinate (row, pixels)")
        bc_layout.addWidget(self.le_bcy_single)
        ring_geom_layout.addRow("Beam Center (px):", bc_layout)
        self.le_px_display = QLineEdit()
        self.le_px_display.setReadOnly(True)
        self.le_px_display.setToolTip("Detector Pixel Size (microns) - Set via Ring/Material Dialog or Params")
        ring_geom_layout.addRow("Pixel Size (\u00B5m):", self.le_px_display)
        self.btn_update_ring_geom = QPushButton("Update Ring Geometry")
        self.btn_update_ring_geom.setToolTip("Apply the specified LSD and Beam Center to the current single detector parameters and redraw rings")
        layout.addWidget(ring_geom_group)
        layout.addWidget(self.btn_update_ring_geom)
        layout.addStretch(1)

    def _create_max_projection_controls(self, group_box):
        layout = QGridLayout(group_box)
        self.cb_use_max_proj = QCheckBox("Use Max Projection")
        self.cb_use_max_proj.setToolTip("Calculate and display the maximum intensity projection over multiple frames instead of a single frame")
        layout.addWidget(self.cb_use_max_proj, 0, 0, 1, 2)
        layout.addWidget(QLabel("# Frames:"), 1, 0)
        self.le_max_frames = QLineEdit("240")
        self.le_max_frames.setValidator(QIntValidator(1, 99999))
        self.le_max_frames.setToolTip("Number of frames to include in the maximum projection calculation")
        layout.addWidget(self.le_max_frames, 1, 1)
        layout.addWidget(QLabel("Start Frame:"), 2, 0)
        self.le_max_start_frame = QLineEdit("0")
        self.le_max_start_frame.setValidator(QIntValidator(0, 99999))
        self.le_max_start_frame.setToolTip("Starting frame number (0-indexed) for the maximum projection")
        layout.addWidget(self.le_max_start_frame, 2, 1)
        self.btn_load_max_proj = QPushButton("Load Max Projection")
        self.btn_load_max_proj.setToolTip("Perform the maximum projection calculation and display the result")
        layout.addWidget(self.btn_load_max_proj, 3, 0, 1, 2)
        layout.setRowStretch(4, 1)

    def _create_hydra_controls(self, group_box):
        layout = QVBoxLayout(group_box)
        param_layout = QHBoxLayout()
        param_layout.addWidget(QLabel("Param File:"))
        self.le_param_file = QLineEdit()
        self.le_param_file.setReadOnly(True)
        self.le_param_file.setToolTip("Path to the Hydra multi-detector parameter file")
        param_layout.addWidget(self.le_param_file)
        self.btn_browse_param = QPushButton("Browse...")
        self.btn_browse_param.setToolTip("Browse for a multi-detector parameter file (.txt)")
        param_layout.addWidget(self.btn_browse_param)
        layout.addLayout(param_layout)
        load_write_layout = QHBoxLayout()
        self.btn_load_params = QPushButton("Load Params")
        self.btn_load_params.setToolTip("Load geometry and settings from the selected parameter file")
        load_write_layout.addWidget(self.btn_load_params)
        layout.addLayout(load_write_layout)
        action_layout = QHBoxLayout()
        self.btn_make_big_det = QPushButton("Make Big Detector")
        self.btn_make_big_det.setToolTip(f"Run the MapMultipleDetectors binary (expected in {self.midas_dir}/bin) using the loaded parameter file")
        action_layout.addWidget(self.btn_make_big_det)
        layout.addLayout(action_layout)
        load_multi_layout = QHBoxLayout()
        self.btn_load_multi = QPushButton("Load Multi-Detector View")
        self.btn_load_multi.setToolTip("Generate and display the composite view from all detectors for the current frame")
        load_multi_layout.addWidget(self.btn_load_multi)
        self.cb_sep_folders = QCheckBox("Separate Folders")
        self.cb_sep_folders.setToolTip("Check if detector data files are in separate subfolders (e.g., ge1/, ge2/)")
        load_multi_layout.addWidget(self.cb_sep_folders)
        layout.addLayout(load_multi_layout)
        self.set_hydra_controls_enabled(False) # Call it here during init
        layout.addStretch(1)

    # --- Signal Connections ---
    def _connect_signals(self):
        """Connect widget signals to handler slots."""
        self.btn_quit.clicked.connect(self.close)
        self.btn_browse_data.clicked.connect(self._browse_data_file)
        self.btn_browse_dark.clicked.connect(self._browse_dark_file)
        self.btn_browse_param.clicked.connect(self._browse_param_file)
        self.btn_view_zarr.clicked.connect(self._view_zarr_parameters) # Ensure this exists
        self.le_frame_nr.editingFinished.connect(self._on_frame_enter)
        self.btn_prev_frame.clicked.connect(self._decrement_frame)
        self.btn_next_frame.clicked.connect(self._increment_frame)
        self.btn_update_thresh.clicked.connect(self._update_display)
        self.cb_log_scale.stateChanged.connect(self._update_display)
        self.cb_hflip.stateChanged.connect(self._reload_current_frame)
        self.cb_vflip.stateChanged.connect(self._reload_current_frame)
        self.cb_transpose.stateChanged.connect(self._reload_current_frame)
        self.cb_dark_correct.stateChanged.connect(self._reload_current_frame)
        self.cb_plot_rings.stateChanged.connect(self._toggle_rings)
        self.btn_select_rings.clicked.connect(self._open_ring_selection)
        self.btn_update_ring_geom.clicked.connect(self._update_ring_geometry)
        self.cb_use_max_proj.stateChanged.connect(self._toggle_max_proj_mode)
        self.btn_load_max_proj.clicked.connect(self._load_max_projection_action)
        self.btn_load_params.clicked.connect(self._load_parameters)
        self.btn_make_big_det.clicked.connect(self._make_big_detector)
        self.btn_load_multi.clicked.connect(self._load_multi_detector_action)
        self.cb_sep_folders.stateChanged.connect(lambda state: self.params.update({'sepfolderVar': bool(state)}))
        self.tab_widget.currentChanged.connect(self._on_tab_changed)

    def _get_combined_ring_material_params(self):
        """Combines params from Zarr cache, self.params, and defaults for RingMaterialDialog."""
        # Start with a copy of general defaults or self.params
        # Order of preference: Zarr cache -> self.params (from .txt or previous GUI) -> hardcoded defaults
        combined = self.params.copy() # Start with current self.params

        # Overlay Zarr cached values (which are top-level, not DetParams specific for material props)
        # These zarr_params_cache keys match the keys used in RingMaterialDialog.get_values()
        # and the keys read in _read_params_from_zarr_to_cache
        for key in ['sg', 'LatticeConstant', 'wl', 'px', 'maxRad', 'lsd']: # 'lsd' can be top-level in Zarr
            if key in self.zarr_params_cache and self.zarr_params_cache[key] is not None:
                combined[key] = self.zarr_params_cache[key]
        
        # Ensure DetParams exists for LSD lookup if top-level 'lsd' wasn't in zarr_params_cache
        if 'lsd' not in combined or combined['lsd'] is None: # If Zarr cache didn't have a top-level LSD
            det_params_list = combined.get('DetParams', [{}])
            if det_params_list and 'lsd' in det_params_list[0] and det_params_list[0]['lsd'] is not None:
                combined['lsd'] = det_params_list[0]['lsd'] # Use from DetParams[0]
            else:
                combined.setdefault('lsd', 1e6) # Ultimate fallback

        # Ensure other keys have defaults if not found
        combined.setdefault('sg', 225)
        combined.setdefault('LatticeConstant', np.array([5.41116, 5.41116, 5.41116, 90.0, 90.0, 90.0]))
        combined.setdefault('wl', 0.172979)
        combined.setdefault('px', 200.0)
        combined.setdefault('maxRad', 2000000.0)
        return combined

    # --- Settings ---
    def _set_default_values(self):
        self.params.setdefault('NrPixelsY', 2048)
        self.params.setdefault('NrPixelsZ', 2048)
        self.params.setdefault('Header', 8192)
        self.params.setdefault('BytesPerPixel', 2)
        self.params.setdefault('px', 200.0)
        self.params.setdefault('firstFileNumber', 1)
        self.params.setdefault('nFramesPerFile', 0)
        self.params.setdefault('nFilesPerLayer', 1)
        self.params.setdefault('Padding', 6)
        self.params.setdefault('DetParams', [{'lsd': 1000000.0, 'bc': [1024.0, 1024.0], 'tx': 0.0, 'ty': 0.0, 'tz': 0.0, 'p0': 0.0, 'p1': 0.0, 'p2': 0.0, 'RhoD': 0.0}])
        self.params.setdefault('nDetectors', 1)
        self.params.setdefault('StartDetNr', 1)
        self.params.setdefault('EndDetNr', 1)
        self.params.setdefault('LatticeConstant', np.array([5.41116, 5.41116, 5.41116, 90.0, 90.0, 90.0]))
        self.params.setdefault('sg', 225)
        self.params.setdefault('wl', 0.172979)
        self.params.setdefault('maxRad', 2000000.0)
        self.params.setdefault('omegaStart', 0.0)
        self.params.setdefault('omegaStep', 0.0)
        self.params.setdefault('sepfolderVar', False)
        self.params.setdefault('HydraActive', False)

    def _ensure_det_params_structure(self, det_idx=0):
        """Ensures self.params['DetParams'] has a valid structure up to det_idx."""
        if 'DetParams' not in self.params or not isinstance(self.params['DetParams'], list):
            self.params['DetParams'] = []
        
        while len(self.params['DetParams']) <= det_idx:
            # Add default-like structures for missing detector entries
            self.params['DetParams'].append({'lsd': None, 'bc': [None, None], 'tx': 0.0, 'ty': 0.0, 'tz': 0.0, 'p0': 0.0, 'p1': 0.0, 'p2': 0.0, 'RhoD': 0.0})
            
        if not isinstance(self.params['DetParams'][det_idx], dict):
            # If it exists but is not a dict, replace it with a default-like structure
            self.params['DetParams'][det_idx] = {'lsd': None, 'bc': [None, None], 'tx': 0.0, 'ty': 0.0, 'tz': 0.0, 'p0': 0.0, 'p1': 0.0, 'p2': 0.0, 'RhoD': 0.0}

    def _merge_zarr_cache_into_params(self):
        if not self.is_zarr_mode or not self.zarr_params_cache:
            # print("Debug: Not Zarr mode or Zarr cache is empty, skipping merge.")
            return False # No Zarr mode or cache is empty

        print("Merging Zarr cache into self.params...")
        changed = False
        # General params
        for key in ['sg', 'LatticeConstant', 'wl', 'px', 'omegaStart', 'omegaStep']:
            if key in self.zarr_params_cache and self.zarr_params_cache[key] is not None:
                current_val = self.params.get(key)
                new_val = self.zarr_params_cache[key]
                needs_update = False
                # Check if value is actually different before assignment
                if isinstance(current_val, np.ndarray) or isinstance(new_val, np.ndarray):
                    if current_val is None or not np.array_equal(current_val, new_val):
                        needs_update = True
                elif current_val != new_val:
                    needs_update = True
                
                if needs_update:
                    self.params[key] = new_val
                    print(f"  Updated self.params['{key}'] from Zarr cache to: {new_val}")
                    changed = True

        # Detector specific params (Zarr cache primarily affects detector 0)
        det_idx_for_zarr = 0 
        self._ensure_det_params_structure(det_idx_for_zarr) # Ensure DetParams[0] exists and is a dict

        # Handle LSD for DetParams[0]
        if 'lsd' in self.zarr_params_cache and self.zarr_params_cache['lsd'] is not None:
            if self.params['DetParams'][det_idx_for_zarr].get('lsd') != self.zarr_params_cache['lsd']:
                self.params['DetParams'][det_idx_for_zarr]['lsd'] = self.zarr_params_cache['lsd']
                print(f"  Updated self.params['DetParams'][{det_idx_for_zarr}]['lsd'] from Zarr cache to: {self.params['DetParams'][det_idx_for_zarr]['lsd']}")
                changed = True

        # Handle BC for DetParams[0]
        # self.zarr_params_cache['bc'] is [plotting_horizontal, plotting_vertical]
        if 'bc' in self.zarr_params_cache and self.zarr_params_cache['bc'] is not None:
            current_bc_param = self.params['DetParams'][det_idx_for_zarr].get('bc')
            zarr_bc = self.zarr_params_cache['bc']
            
            needs_bc_update = False
            if current_bc_param is None:
                needs_bc_update = True
            elif isinstance(current_bc_param, (list, np.ndarray)) and isinstance(zarr_bc, (list, np.ndarray)):
                if len(current_bc_param) != len(zarr_bc) or not np.array_equal(np.array(current_bc_param, dtype=float), np.array(zarr_bc, dtype=float)):
                    needs_bc_update = True
            elif current_bc_param != zarr_bc : # Fallback for other types
                needs_bc_update = True

            if needs_bc_update:
                self.params['DetParams'][det_idx_for_zarr]['bc'] = list(zarr_bc) # Store as list
                print(f"  Updated self.params['DetParams'][{det_idx_for_zarr}]['bc'] from Zarr cache to: {self.params['DetParams'][det_idx_for_zarr]['bc']}")
                changed = True
        
        if changed:
            print(f"self.params['DetParams'][{det_idx_for_zarr}] after Zarr merge: {self.params['DetParams'][det_idx_for_zarr]}")
        else:
            print("No changes made to self.params during Zarr merge.")
        return changed

    def _save_settings(self):
        print("Saving settings...")
        settings = QSettings(self.ORG_NAME, self.APP_NAME)
        settings.setValue("geometry", self.saveGeometry())
        settings.setValue("windowState", self.saveState())
        settings.setValue("dataFile", self.current_data_file)
        settings.setValue("darkFile", self.current_dark_file)
        settings.setValue("paramFile", self.current_param_file)
        settings.setValue("lastDataDir", self.last_browse_dirs.get("data", str(Path.home())))
        settings.setValue("lastDarkDir", self.last_browse_dirs.get("dark", str(Path.home())))
        settings.setValue("lastParamDir", self.last_browse_dirs.get("param", str(Path.home())))
        settings.setValue("minThresh", self.le_min_thresh.text())
        settings.setValue("maxThresh", self.le_max_thresh.text())
        settings.setValue("logScale", self.cb_log_scale.isChecked())
        settings.setValue("hFlip", self.cb_hflip.isChecked())
        settings.setValue("vFlip", self.cb_vflip.isChecked())
        settings.setValue("transpose", self.cb_transpose.isChecked())
        settings.setValue("darkCorrect", self.cb_dark_correct.isChecked())
        settings.setValue("plotRings", self.cb_plot_rings.isChecked())
        settings.setValue("ringDetNum", self.le_det_num_single.text())
        settings.setValue("ringLSD", self.le_lsd_single.text())
        settings.setValue("ringBCX", self.le_bcx_single.text())
        settings.setValue("ringBCY", self.le_bcy_single.text())
        settings.setValue("ringSG", self.params.get('sg', 225))
        # Use semicolon for list saving as comma is standard float separator
        settings.setValue("ringLatConst", ";".join(map(str, self.params.get('LatticeConstant', np.zeros(6)))))
        settings.setValue("ringWL", self.params.get('wl', 0.1729))
        settings.setValue("ringPX", self.params.get('px', 200.0))
        settings.setValue("ringMaxRad", self.params.get('maxRad', 2e6))
        # Use comma for list of numbers saving
        settings.setValue("selectedRingNrs", ",".join(map(str, self.params.get('ringNrs', []))))
        settings.setValue("useMaxProj", self.cb_use_max_proj.isChecked())
        settings.setValue("maxProjFrames", self.le_max_frames.text())
        settings.setValue("maxProjStart", self.le_max_start_frame.text())
        settings.setValue("sepFolders", self.cb_sep_folders.isChecked())
        print("Settings saved.")

    def _load_settings(self):
        print("Loading settings...")
        settings = QSettings(self.ORG_NAME, self.APP_NAME)
        geometry = settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)
        windowState = settings.value("windowState")
        if windowState:
            self.restoreState(windowState)
        self.current_data_file = settings.value("dataFile", "")
        self.le_data_file.setText(self.current_data_file)
        self.current_dark_file = settings.value("darkFile", "")
        self.le_dark_file.setText(self.current_dark_file)
        self.current_param_file = settings.value("paramFile", "")
        self.le_param_file.setText(self.current_param_file)
        self.last_browse_dirs["data"] = settings.value("lastDataDir", str(Path.home()))
        self.last_browse_dirs["dark"] = settings.value("lastDarkDir", str(Path.home()))
        self.last_browse_dirs["param"] = settings.value("lastParamDir", str(Path.home()))
        self.is_zarr_mode = self.current_data_file.lower().endswith(".zip")
        self.btn_view_zarr.setEnabled(self.is_zarr_mode)
        self.le_min_thresh.setText(settings.value("minThresh", "0"))
        self.le_max_thresh.setText(settings.value("maxThresh", "2000"))
        self.cb_log_scale.setChecked(settings.value("logScale", False, type=bool))
        self.cb_hflip.setChecked(settings.value("hFlip", False, type=bool))
        self.cb_vflip.setChecked(settings.value("vFlip", False, type=bool))
        self.cb_transpose.setChecked(settings.value("transpose", False, type=bool))
        self.cb_dark_correct.setChecked(settings.value("darkCorrect", False, type=bool))
        self.le_det_num_single.setText(settings.value("ringDetNum", "1"))
        self.le_lsd_single.setText(settings.value("ringLSD", "1000000"))
        self.le_bcx_single.setText(settings.value("ringBCX", "1024"))
        self.le_bcy_single.setText(settings.value("ringBCY", "1024"))
        self.params['sg'] = settings.value("ringSG", 225, type=int)
        try:
            lc_str = settings.value("ringLatConst", ";".join(map(str, np.zeros(6))))
            self.params['LatticeConstant'] = np.array([float(x) for x in lc_str.split(";")])
        except:
            self.params['LatticeConstant'] = np.zeros(6)
        self.params['wl'] = settings.value("ringWL", 0.1729, type=float)
        self.params['px'] = settings.value("ringPX", 200.0, type=float)
        self.params['maxRad'] = settings.value("ringMaxRad", 2e6, type=float)
        if hasattr(self, 'le_px_display'): # Check exists before setting
            self.le_px_display.setText(f"{self.params['px']:.2f}")
        selected_nrs_str = settings.value("selectedRingNrs", "")
        if selected_nrs_str:
             try:
                 # Store potentially restored ring numbers for later use if needed
                 self.params['_saved_ringNrs'] = [int(n) for n in selected_nrs_str.split(",")]
             except Exception:
                 pass # Ignore if parsing fails
        self.cb_use_max_proj.setChecked(settings.value("useMaxProj", False, type=bool))
        self.le_max_frames.setText(settings.value("maxProjFrames", "240"))
        self.le_max_start_frame.setText(settings.value("maxProjStart", "0"))
        self.cb_sep_folders.setChecked(settings.value("sepFolders", False, type=bool))
        self.params['sepfolderVar'] = self.cb_sep_folders.isChecked()
        # Enable load button only if file exists
        self.btn_load_params.setEnabled(bool(self.current_param_file and Path(self.current_param_file).exists()))
        self._toggle_max_proj_mode(self.cb_use_max_proj.isChecked()) # Update frame nav enabled state AFTER checkbox is set
        self.set_hydra_controls_enabled(True) # Let the method check param file existence

        print("Settings loaded.")


    # --- Helper Methods & Slots ---
    def update_status_bar(self, message):
        if message: # If there's a message, show it
            self.status_bar.showMessage(message) # No timeout, or a very short one like 100ms
        else: # If the message is empty (e.g., cursor left axes), clear it
            self.status_bar.clearMessage()

    def show_progress(self, visible, value=0):
        self.progress_bar.setVisible(visible)
        self.progress_bar.setValue(value)

    # --- Thread Management ---
    def run_in_thread(self, task_func, task_id, *args, **kwargs):
        if self.worker_thread is not None and self.worker_thread.isRunning():
            current_task_id = getattr(self.worker, 'task_id', 'Unknown')
            msg = f"Busy with task '{current_task_id}'. Please wait."
            print(f"Warning: Worker thread already running. Ignoring request for task '{task_id}'.")
            QMessageBox.warning(self, "Busy", msg)
            self.update_status_bar(msg)
            return

        self.show_progress(True, 0)
        self.update_status_bar(f"Running task: {task_id}...")
        self.set_ui_enabled(False)
        self.worker_thread = QThread()
        self.worker = Worker(task_func, task_id, *args, **kwargs)
        self.worker.moveToThread(self.worker_thread)
        self.worker.finished.connect(self._handle_worker_finished)
        self.worker.error.connect(self._handle_worker_error)
        self.worker.progress.connect(self._handle_worker_progress)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.worker_thread.quit) # Worker done -> Thread quits event loop
        self.worker.error.connect(self.worker_thread.quit)    # Worker error -> Thread quits event loop
        self.worker_thread.finished.connect(self._on_thread_finished) # Thread finished -> Cleanup slot
        self.worker_thread.start()

    @pyqtSlot()
    def _on_thread_finished(self):
        print("Worker thread finished signal received.")
        # Safely schedule deletion for worker and thread objects
        # Check if they exist before calling deleteLater
        if hasattr(self, 'worker') and self.worker:
             self.worker.deleteLater()
        if hasattr(self, 'worker_thread') and self.worker_thread:
             self.worker_thread.deleteLater()
        # Nullify references immediately
        self.worker = None
        self.worker_thread = None
        self.show_progress(False) # Hide progress bar
        self.set_ui_enabled(True) # Re-enable UI
        print("Thread references cleaned, UI re-enabled.")

    # --- File Browsing Methods ---
    def _browse_data_file(self):
        file_filter = "Data Files (*.ge* *.raw *.bin *.zip *.hdf5 *.h5);;Zarr Archives (*.zip);;All Files (*)"
        settings = QSettings(self.ORG_NAME, self.APP_NAME)
        last_dir = settings.value("lastDataDir", self.last_browse_dirs.get("data", str(Path.home())))
        filename, _ = QFileDialog.getOpenFileName(self, "Select Data File", last_dir, file_filter)
        if filename:
            self.current_data_file = filename
            self.le_data_file.setText(filename)
            new_dir = str(Path(filename).parent)
            settings.setValue("lastDataDir", new_dir)
            self.last_browse_dirs["data"] = new_dir
            self.is_zarr_mode = filename.lower().endswith(".zip")
            self.btn_view_zarr.setEnabled(self.is_zarr_mode)
            self.params = {}
            self.dark_frame_cache = {}
            self.zarr_params_cache = {} # Reset caches
            self._set_default_values() # Apply defaults first
            if self.is_zarr_mode:
                print("Zarr file selected, attempting to read parameters from it...")
                self._read_params_from_zarr_to_cache() # Read from Zarr *after* defaults
                self._merge_zarr_cache_into_params()
                for key in ['sg', 'LatticeConstant', 'wl', 'px', 'omegaStart', 'omegaStep']:
                     if key in self.zarr_params_cache and self.zarr_params_cache[key] is not None:
                         self.params[key] = self.zarr_params_cache[key]
                # Handle DetParams[0]['lsd'] specifically if top-level lsd was in Zarr
                if 'lsd' in self.zarr_params_cache and self.zarr_params_cache['lsd'] is not None:
                    if 'DetParams' not in self.params or not self.params['DetParams']: self.params['DetParams'] = [{}]
                    self.params['DetParams'][0]['lsd'] = self.zarr_params_cache['lsd']
                if 'bc' in self.zarr_params_cache and self.zarr_params_cache['bc'] is not None:
                    if 'DetParams' not in self.params or not self.params['DetParams']: 
                        self.params['DetParams'] = [{}]
                    elif not self.params['DetParams']: # Ensure list is not empty
                        self.params['DetParams'].append({})
                    self.params['DetParams'][0]['bc'] = self.zarr_params_cache['bc'] # Assign the cached BC
                    print(f"  _browse_data_file: Updated self.params['DetParams'][0]['bc'] from Zarr cache: {self.params['DetParams'][0]['bc']}")
            self._update_gui_from_params() # Update GUI based on combined params
            self.set_hydra_controls_enabled(False) # Disable hydra controls
            self.current_param_file = "" # Clear param file path
            self.le_param_file.clear()
            self.update_status_bar(f"Data file selected: {Path(filename).name}. Load frame 0.")
            self.current_frame_nr = 0
            self.current_frame_nr_pending = 0
            self.le_frame_nr.setText("0")
            self._load_frame(0) # Triggers threaded load

    def _browse_dark_file(self):
        file_filter = "Data Files (*.ge* *.raw *.bin);;All Files (*)"
        settings = QSettings(self.ORG_NAME, self.APP_NAME)
        last_dir = settings.value("lastDarkDir", self.last_browse_dirs.get("dark", str(Path.home())))
        filename, _ = QFileDialog.getOpenFileName(self, "Select Dark File", last_dir, file_filter)
        if filename:
            self.current_dark_file = filename
            self.le_dark_file.setText(filename)
            new_dir = str(Path(filename).parent)
            settings.setValue("lastDarkDir", new_dir)
            self.last_browse_dirs["dark"] = new_dir
            self.dark_frame_cache = {} # Clear cache
            self.cb_dark_correct.setChecked(True)
            if self.current_data_single is not None or self.current_data_multi is not None:
                self._reload_current_frame()

    def _browse_param_file(self):
        settings = QSettings(self.ORG_NAME, self.APP_NAME)
        last_dir = settings.value("lastParamDir", self.last_browse_dirs.get("param", str(Path.home())))
        filename, _ = QFileDialog.getOpenFileName(self, "Select Parameter File", last_dir, "Parameter Files (*.txt);;All Files (*)")
        if filename:
            self.current_param_file = filename
            self.le_param_file.setText(filename)
            new_dir = str(Path(filename).parent)
            settings.setValue("lastParamDir", new_dir)
            self.last_browse_dirs["param"] = new_dir
            self.btn_load_params.setEnabled(True)
            self._load_parameters() # Load immediately

    # --- Other Slot Methods ---
    def _view_zarr_parameters(self):
        if not self.is_zarr_mode or not self.current_data_file:
             QMessageBox.warning(self, "Not Available", "A Zarr file must be loaded first.")
             return
        if not Path(self.current_data_file).exists():
             QMessageBox.warning(self, "File Not Found", f"Cannot find Zarr file: {self.current_data_file}")
             return
        try:
             # Pass self as parent
             dialog = ZarrViewerDialog(self.current_data_file, self)
             dialog.exec()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open Zarr parameter viewer: {e}\n{traceback.format_exc()}")

    def _load_parameters(self):
        if not self.current_param_file:
            QMessageBox.warning(self, "No Parameter File", "Please browse for a parameter file first.")
            return
        if not Path(self.current_param_file).exists():
             QMessageBox.critical(self, "File Not Found", f"Parameter file not found:\n{self.current_param_file}")
             return
        param_file_copy = str(self.current_param_file)
        self.run_in_thread(read_parameters, "load_params", param_file_copy)

    def _handle_load_params_finished(self, params_result):
        if not isinstance(params_result, dict):
             error_msg = f"Failed to load parameters: {params_result}"
             print(error_msg)
             QMessageBox.critical(self, "Parameter Error", error_msg)
             self.update_status_bar("Error loading parameters.")
             self.set_hydra_controls_enabled(False) # Ensure hydra disabled on error
             # UI enabling handled by _on_thread_finished
             return

        self.params = params_result
        # Now, if a Zarr file was previously loaded and its params cached, merge/override
        # Zarr params take precedence for material/scan properties if they exist in cache
        if self.is_zarr_mode and self.zarr_params_cache:
            print("Merging .txt params with cached Zarr params (Zarr values take precedence for matched keys)...")
            self._merge_zarr_cache_into_params() 
            for key in ['sg', 'LatticeConstant', 'wl', 'px', 'omegaStart', 'omegaStep']:
                if key in self.zarr_params_cache and self.zarr_params_cache[key] is not None:
                    print(f"  Overriding '{key}' with Zarr value: {self.zarr_params_cache[key]}")
                    self.params[key] = self.zarr_params_cache[key]
            # Handle DetParams[0]['lsd'] and ['bc'] specifically if top-level lsd/bc was in Zarr
            if 'lsd' in self.zarr_params_cache and self.zarr_params_cache['lsd'] is not None:
                if 'DetParams' not in self.params or not self.params['DetParams']: self.params['DetParams'] = [{}]
                self.params['DetParams'][0]['lsd'] = self.zarr_params_cache['lsd']
                print(f"  Overriding DetParams[0]['lsd'] with Zarr value: {self.zarr_params_cache['lsd']}")
            if 'bc' in self.zarr_params_cache and self.zarr_params_cache['bc'] is not None: # bc from zarr is [x,y]
                if 'DetParams' not in self.params or not self.params['DetParams']: self.params['DetParams'] = [{}]
                self.params['DetParams'][0]['bc'] = self.zarr_params_cache['bc']
                print(f"  Overriding DetParams[0]['bc'] with Zarr value: {self.zarr_params_cache['bc']}")
        self.update_status_bar("Parameters loaded successfully.")
        # --- Call helper to update GUI ---
        self._update_gui_from_params()
        # -------------------------------
        if self.current_data_file and self.current_data_file.lower().endswith(".zip"):
             print("Parameter file pointed to a Zarr data file, checking its internal parameters...")
             self._read_params_from_zarr() # Read Zarr params, potentially overwriting some from .txt
             self._update_gui_from_params() # Update GUI again with potentially merged params

        # Auto-find Data/Dark only if no data file already loaded
        if not self.current_data_file:
            hydra_active = self.params.get('HydraActive', False) # Need this for path finding
            if self.params.get('fileStem') and self.params.get('folder'):
                num = self.params.get('firstFileNumber', 1)
                det_nr_for_path = self.params.get('StartDetNr', 1) if hydra_active else -1
                file0_path = get_file_path(self.params, num, det_nr=det_nr_for_path, is_dark=False)
                if file0_path and Path(file0_path).exists():
                    self.current_data_file = file0_path
                    self.le_data_file.setText(file0_path)
                    self.is_zarr_mode = file0_path.lower().endswith(".zip")
                    self.btn_view_zarr.setEnabled(self.is_zarr_mode)
                    if self.is_zarr_mode: 
                        self._read_params_from_zarr_to_cache() # Read to cache if new file found
                    print(f"Found data file based on parameters: {file0_path}")
                    self.current_frame_nr = 0
                    self.current_frame_nr_pending = 0
                    self.le_frame_nr.setText("0")
                    self._load_frame(0) # Load frame 0 automatically
                else:
                    self.update_status_bar("Parameters loaded, but could not auto-find data file.")
            else:
                 self.update_status_bar("Parameters loaded, but missing folder/stem to find data file.")

            if self.params.get('darkStem') and self.params.get('darkNum') is not None and self.params.get('folder'):
                det_nr_for_path = self.params.get('StartDetNr', 1) if hydra_active else -1
                dark_path = get_file_path(self.params, 0, det_nr=det_nr_for_path, is_dark=True) # file_nr=0 is dummy here
                if dark_path and Path(dark_path).exists():
                     self.current_dark_file = dark_path
                     self.le_dark_file.setText(dark_path)
                     print(f"Found dark file based on parameters: {dark_path}")
                     self.dark_frame_cache = {} # Clear cache when new dark found
                else:
                     print(f"Dark file not found based on parameters ({dark_path}).")
        # Reload current frame only if data is already loaded (to apply new params like flips)
        elif self.current_data_single is not None or self.current_data_multi is not None:
             self._reload_current_frame()
        # UI enabling handled by _on_thread_finished

    def _update_gui_from_params(self):
        """Updates GUI widgets based on values currently in self.params."""
        print("--- Entering _update_gui_from_params ---")
        det_params_list = self.params.get('DetParams', [{}])
        current_det_nr = try_parse_int(self.le_det_num_single.text(), self.params.get('StartDetNr', 1)) # Read current GUI det num
        det_idx = current_det_nr - self.params.get('StartDetNr', 1)

        if not det_params_list: det_params_list = [{}] # Ensure at least one empty dict

        if not (0 <= det_idx < len(det_params_list)):
            print(f"  Warning: Invalid det_idx {det_idx} for current GUI det num {current_det_nr}. Falling back to index 0.")
            det_idx = 0
            current_det_nr = self.params.get('StartDetNr', 1)
            if hasattr(self, 'le_det_num_single'): self.le_det_num_single.setText(str(current_det_nr)) # Correct GUI if index was bad

        det_to_display = det_params_list[det_idx]
        print(f"  Updating GUI for Det {current_det_nr} (Index {det_idx}) using params: {det_to_display}")

        # Update LSD
        lsd_val = try_parse_float(det_to_display.get('lsd'), default=None)
        if hasattr(self, 'le_lsd_single'):
            lsd_val_str = f"{lsd_val:.1f}" if lsd_val is not None else "N/A"
            self.le_lsd_single.setText(lsd_val_str)
            print(f"  Setting le_lsd_single text to: {lsd_val_str}")

        # Beam center from self.params is now assumed to be [plotting_horizontal, plotting_vertical]
        bc = det_to_display.get('bc', [None, None]) 
        bc_plotting_horizontal = bc[0] if isinstance(bc, (list, np.ndarray)) and len(bc) > 0 else None
        bc_plotting_vertical = bc[1] if isinstance(bc, (list, np.ndarray)) and len(bc) > 1 else None
        print(f"  BC values from self.params for GUI: [Plotting_H={bc_plotting_horizontal}, Plotting_V={bc_plotting_vertical}]")
        if hasattr(self, 'le_bcx_single'): # This QLineEdit should display the PLOTTING HORIZONTAL
            bc_x_text = f"{try_parse_float(bc_plotting_horizontal):.2f}" if bc_plotting_horizontal is not None else "N/A"
            self.le_bcx_single.setText(bc_x_text)
            print(f"  Setting le_bcx_single (Plotting Horizontal) text to: {bc_x_text}")
        if hasattr(self, 'le_bcy_single'): # This QLineEdit should display the PLOTTING VERTICAL
            bc_y_text = f"{try_parse_float(bc_plotting_vertical):.2f}" if bc_plotting_vertical is not None else "N/A"
            self.le_bcy_single.setText(bc_y_text)
            print(f"  Setting le_bcy_single (Plotting Vertical) text to: {bc_y_text}")

        # Update Pixel Size Display
        px_val = try_parse_float(self.params.get('px'), default=None)
        if hasattr(self, 'le_px_display'):
            px_text = f"{px_val:.2f}" if px_val is not None else "N/A"
            self.le_px_display.setText(px_text)
            print(f"  Setting le_px_display text to: {px_text}")

        # Update Hydra controls based on whether params indicate Hydra mode
        if hasattr(self, 'set_hydra_controls_enabled'):
             self.set_hydra_controls_enabled(True) # Let method check internal state (HydraActive, paramFN exists)

        # Update ring info label (if rings are selected)
        if hasattr(self, '_update_ring_info_label'):
            self._update_ring_info_label()

        print("--- Exiting _update_gui_from_params ---")
        QApplication.processEvents() # Force GUI refresh


    def _read_params_from_zarr_to_cache(self): # Modified to update a cache
        """Reads parameters from the loaded Zarr file and updates self.zarr_params_cache."""
        if not self.is_zarr_mode or not self.current_data_file: 
            print("_read_params_from_zarr_to_cache: Not Zarr mode or no file."); 
            return
        if not Path(self.current_data_file).exists(): 
            print(f"_read_params_from_zarr_to_cache: Zarr file not found: {self.current_data_file}"); 
            return

        print(f"Caching parameters from Zarr file: {self.current_data_file}")
        self.zarr_params_cache = {} # Clear previous cache
        store = None
        try:
            store = ZipStore(self.current_data_file, mode='r')
            z_root = zarr.open_group(store=store, mode='r')
            zarr_param_map = OrderedDict([
                ('analysis/process/analysis_parameters/Lsd', ('lsd', float, 0)), # Level 0 for DetParams[0]
                ('measurement/instrument/detector/distance', ('lsd', float, 0)),
                ('analysis/process/analysis_parameters/YCen', ('zarr_bc_x', float, 0)), # YCen in file
                ('measurement/instrument/detector/beam_center_x', ('zarr_bc_x', float, 0)), # beam_center_x in file
                ('analysis/process/analysis_parameters/ZCen', ('zarr_bc_y', float, 0)), # ZCen in file
                ('measurement/instrument/detector/beam_center_y', ('zarr_bc_y', float, 0)), # beam_center_y in file
                ('analysis/process/analysis_parameters/PixelSize', ('px', float, 1)), # Level 1 for self.params/cache
                ('measurement/instrument/detector/x_pixel_size', ('px', float, 1)),
                ('analysis/process/analysis_parameters/Wavelength', ('wl', float, 1)),
                ('measurement/instrument/monochromator/energy', ('energy_kev', float, 1)),
                ('analysis/process/analysis_parameters/LatticeParameter', ('LatticeConstant', float, 1)),
                ('analysis/process/analysis_parameters/SpaceGroup', ('sg', int, 1)),
                ('measurement/process/scan_parameters/start', ('omegaStart', float, 1)),
                ('measurement/process/scan_parameters/step', ('omegaStep', float, 1)),
            ])
            temp_zarr_bc_vals = {'x': None, 'y': None} # To hold values read directly as 'zarr_bc_x' and 'zarr_bc_y'
            energy_kev_val = None
            for z_path, (p_key, p_type, _) in zarr_param_map.items(): # Level not used for cache directly
                if z_path in z_root:
                    try:
                        value = z_root[z_path][...];
                        if p_key == 'LatticeConstant' and value.shape == (6,): 
                            self.zarr_params_cache['LatticeConstant'] = value.astype(p_type)
                        elif p_key == 'energy_kev' and value.size == 1: 
                            energy_kev_val = p_type(value.item())
                        elif p_key == 'zarr_bc_x' and value.size == 1: # Read as is from Zarr
                            if temp_zarr_bc_vals['x'] is None: # Prioritize first found
                                temp_zarr_bc_vals['x'] = p_type(value.item())
                        elif p_key == 'zarr_bc_y' and value.size == 1: # Read as is from Zarr
                             if temp_zarr_bc_vals['y'] is None: # Prioritize first found
                                temp_zarr_bc_vals['y'] = p_type(value.item())
                        elif value.size == 1: 
                            self.zarr_params_cache[p_key] = p_type(value.item())
                    except Exception as read_err: 
                        print(f"    Warning: Could not read/process Zarr param '{z_path}': {read_err}")

            if temp_zarr_bc_vals['x'] is not None and temp_zarr_bc_vals['y'] is not None:
                # Assuming:
                # temp_zarr_bc_vals['x'] was read from Zarr's "beam_center_x" or "YCen"
                # temp_zarr_bc_vals['y'] was read from Zarr's "beam_center_y" or "ZCen"

                # If Zarr's "beam_center_x" (or YCen) is actually your PLOTTING VERTICAL coordinate
                # and Zarr's "beam_center_y" (or ZCen) is actually your PLOTTING HORIZONTAL coordinate:
                plotting_horizontal = temp_zarr_bc_vals['y'] # e.g., Zarr's 'y' is our plotting X
                plotting_vertical = temp_zarr_bc_vals['x']   # e.g., Zarr's 'x' is our plotting Y
                self.zarr_params_cache['bc'] = [plotting_horizontal, plotting_vertical]
                print(f"  Zarr BC: Read zarr_x={temp_zarr_bc_vals['x']}, zarr_y={temp_zarr_bc_vals['y']}. Stored in cache as [plotting_H, plotting_V]: {self.zarr_params_cache['bc']}")
            elif temp_zarr_bc_vals['x'] is not None or temp_zarr_bc_vals['y'] is not None:
                 print(f"  Warning: Only one component of Zarr beam center found (x:{temp_zarr_bc_vals['x']}, y:{temp_zarr_bc_vals['y']}). BC not fully set from Zarr.")

            if 'wl' not in self.zarr_params_cache and energy_kev_val is not None and energy_kev_val > 0: 
                self.zarr_params_cache['wl'] = 12.3984 / energy_kev_val
            
            if self.zarr_params_cache: 
                print(f"Zarr params cached: {self.zarr_params_cache}")
            else: 
                print("No relevant parameters found in Zarr to cache.")

        except Exception as e: 
            print(f"Error reading Zarr parameters to cache: {traceback.format_exc()}")
        finally:
            if store is not None and hasattr(store, 'mode') and store.mode != 'closed':
                try: 
                    store.close()
                except Exception: 
                    pass


    # --- Frame Navigation and Loading ---
    def _on_frame_enter(self):
        try:
            frame = int(self.le_frame_nr.text())
            if self.total_frames_available > 0 and (frame < 0 or frame >= self.total_frames_available):
                 corrected_frame = max(0, min(frame, self.total_frames_available - 1))
                 self.le_frame_nr.setText(str(corrected_frame))
                 if frame != corrected_frame: QMessageBox.warning(self, "Frame Out of Bounds", f"Frame must be 0 to {self.total_frames_available - 1}. Adjusted to {corrected_frame}.")
                 frame = corrected_frame
            elif self.total_frames_available == 0 and frame != 0:
                 self.le_frame_nr.setText("0"); QMessageBox.warning(self, "No Frames Loaded", "Cannot navigate frames yet."); frame = 0
            if frame != self.current_frame_nr or (self.current_data_single is None and self.current_data_multi is None): self._load_frame(frame)
        except ValueError: self.le_frame_nr.setText(str(self.current_frame_nr))

    def _increment_frame(self):
        if self.total_frames_available > 0:
            next_frame = self.current_frame_nr + 1
            if next_frame < self.total_frames_available: self._load_frame(next_frame)
            else: self.update_status_bar("Already at the last available frame.")
        else: self.update_status_bar("Load data first to navigate frames.")

    def _decrement_frame(self):
        if self.total_frames_available > 0:
            prev_frame = self.current_frame_nr - 1
            if prev_frame >= 0: self._load_frame(prev_frame)
            else: self.update_status_bar("Already at the first frame.")
        else: self.update_status_bar("Load data first to navigate frames.")

    def _reload_current_frame(self):
        if self.current_frame_nr is not None and self.current_frame_nr >= 0:
            print(f"Reloading frame {self.current_frame_nr}")
            if self.total_frames_available > 0 and self.current_frame_nr < self.total_frames_available: self._load_frame(self.current_frame_nr)
            elif self.total_frames_available == 0: self._load_frame(0)
            else: print("Cannot reload, frame number potentially invalid."); self.current_frame_nr = 0; self.le_frame_nr.setText("0")
            if self.total_frames_available > 0: self._load_frame(0)
        else:
            print("Cannot reload, no valid current frame number."); self.current_frame_nr = 0; self.le_frame_nr.setText("0")
            if self.total_frames_available > 0: self._load_frame(0)

    def _update_display(self):
         active_tab_index = self.tab_widget.currentIndex()
         plot_key = 'single' if active_tab_index == 0 else 'multi'
         ax = self.ax_single if active_tab_index == 0 else self.ax_multi
         data = self.current_data_single if active_tab_index == 0 else self.current_data_multi
         if data is not None and ax and ax.get_visible() and self.plotting_handler:
              try:
                   min_t = try_parse_float(self.le_min_thresh.text(), 0.0); max_t = try_parse_float(self.le_max_thresh.text(), 2000.0)
                   clim = (min_t, max_t); log_scale = self.cb_log_scale.isChecked()
                   self.plotting_handler.update_plot(plot_key, data, self.params, clim, log_scale, self.current_file_info)
              except ValueError: self.update_status_bar("Invalid threshold value.")

    def _load_frame(self, frame_nr):
        self.current_frame_nr_pending = frame_nr
        active_tab_index = self.tab_widget.currentIndex()
        is_max_proj_mode = self.cb_use_max_proj.isChecked() if hasattr(self, 'cb_use_max_proj') else False
        if is_max_proj_mode: self.update_status_bar("Max projection mode active. Use 'Load Max Projection' button."); return
        data_source_available = self.current_data_file or self.params.get('HydraActive', False)
        if not data_source_available: QMessageBox.warning(self, "File Error", "Please select a data file or load Hydra parameters."); return
        if self.is_zarr_mode:
            self._read_params_from_zarr_to_cache() # Read/Update params for Zarr before load
            self._merge_zarr_cache_into_params()
            self._update_gui_from_params() # Update GUI immediately after reading
        load_task_id = ""; args = []; task_func = None
        params_copy = self.params.copy() # Pass copy to thread
        params_copy['darkCorrectChecked'] = self.cb_dark_correct.isChecked(); params_copy['hflipChecked'] = self.cb_hflip.isChecked(); params_copy['vflipChecked'] = self.cb_vflip.isChecked(); params_copy['transposeChecked'] = self.cb_transpose.isChecked()

        if active_tab_index == 0: # Single Detector Tab
            load_task_id = "load_single_frame"; task_func = self._load_and_correct_single_frame_task
            det_nr_single = try_parse_int(self.le_det_num_single.text(), 1); params_copy['current_det_nr_single'] = det_nr_single
            if self.is_zarr_mode:
                 file_path = self.current_data_file; frame_in_file = frame_nr
                 if not file_path or not Path(file_path).exists(): self.update_status_bar("Zarr file path is invalid."); return
            else: # Raw binary mode
                 if params_copy.get('nFramesPerFile', 0) <= 0 and self.current_data_file and Path(self.current_data_file).exists():
                     try:
                        fsize = os.path.getsize(self.current_data_file); hdr = params_copy.get('Header', 8192); ny = params_copy.get('NrPixelsY', 2048); nz = params_copy.get('NrPixelsZ', 2048); bpp = params_copy.get('BytesPerPixel', 2); frame_bytes = ny * nz * bpp
                        if frame_bytes > 0 and fsize >= hdr: params_copy['nFramesPerFile'] = (fsize - hdr) // frame_bytes; print(f"Calculated nFramesPerFile: {params_copy['nFramesPerFile']}")
                        else: params_copy['nFramesPerFile'] = 0
                     except Exception as e_calc: print(f"Could not calculate nFramesPerFile: {e_calc}"); params_copy['nFramesPerFile'] = 0
                 n_frames_per_file = max(1, try_parse_int(params_copy.get('nFramesPerFile', 1)))
                 first_file_nr = try_parse_int(params_copy.get('firstFileNumber', 1))
                 file_nr = first_file_nr + frame_nr // n_frames_per_file; frame_in_file = frame_nr % n_frames_per_file
                 det_nr_for_path = det_nr_single if params_copy.get('HydraActive', False) else -1
                 file_path = get_file_path(params_copy, file_nr, det_nr=det_nr_for_path, is_dark=False)
                 if not file_path or not Path(file_path).exists(): self.update_status_bar(f"Data file not found for frame {frame_nr}: {file_path}"); return
            args = [params_copy, file_path, frame_in_file, params_copy['hflipChecked'], params_copy['vflipChecked'], params_copy['transposeChecked']]
        elif active_tab_index == 1: # Multi Detector Tab
            if not self.params.get('HydraActive', False): self.update_status_bar("Load Hydra parameters to use the multi-detector view."); return
            load_task_id = "load_multi_frame"; task_func = self._construct_multi_detector_task
            args = [params_copy, frame_nr, self.dark_frame_cache.copy()]
        if task_func and load_task_id: self.run_in_thread(task_func, load_task_id, *args)

    # --- Worker Tasks ---
    def _load_and_correct_single_frame_task(self, params, file_path, frame_in_file, hflip, vflip, transpose, worker_data_dict):
        dark_correct_checked = params.get('darkCorrectChecked', False)
        print(f"Worker: Loading single frame {frame_in_file} from {Path(file_path).name} (dark_corr={dark_correct_checked})")
        data, file_info = load_image_frame(params, file_path, frame_in_file, hflip, vflip, transpose)
        if data is None: raise ValueError(f"Failed to load data from {file_path}, frame {frame_in_file}")
        corrected_data = data; dark_applied = False; newly_loaded_darks = {}
        if dark_correct_checked and not file_info.get('dark_corrected', False):
            print(f"Worker: Attempting dark correction for frame {frame_in_file}")
            dark_data = None; dark_key = params.get('current_det_nr_single', 1)
            if dark_key in self.dark_frame_cache: print(f"Worker: Found dark for det {dark_key} in main cache."); dark_data = self.dark_frame_cache[dark_key]
            else:
                 print(f"Worker: Dark for det {dark_key} not in cache. Trying to load."); dark_file_path = self.current_dark_file
                 if not dark_file_path or not Path(dark_file_path).exists():
                      det_nr_for_path = dark_key if params.get('HydraActive', False) else -1; dark_file_path = get_file_path(params, 0, det_nr=det_nr_for_path, is_dark=True)
                 if dark_file_path and Path(dark_file_path).exists():
                      print(f"Worker: Loading dark from {dark_file_path}"); dark_data = load_dark_frame(params, dark_file_path, hflip, vflip, transpose)
                      if dark_data is not None: newly_loaded_darks[dark_key] = dark_data
                 else: print(f"Worker: Dark file path not found or invalid: {dark_file_path}")
            if dark_data is not None:
                 if dark_data.shape == data.shape: print(f"Worker: Applying dark correction (shape {dark_data.shape})"); corrected_data = data - dark_data; corrected_data[corrected_data < 0] = 0; dark_applied = True
                 else: print(f"Worker: Dark frame shape mismatch {dark_data.shape} vs {data.shape}. Correction skipped.")
            else: print("Worker: Dark frame not available or specified. Correction skipped.")
        file_info['dark_corrected'] = dark_applied or file_info.get('dark_corrected', False)
        worker_data_dict.update(newly_loaded_darks)
        total_frames = file_info.get('total_frames', 1)
        return corrected_data, file_info, total_frames

    def _construct_multi_detector_task(self, params, frame_nr, dark_cache_copy, worker_data_dict):
        print(f"Worker: Constructing multi-detector view for frame {frame_nr}")
        big_size = params.get('bigdetsize', 2048); px_size_um = try_parse_float(params.get('px')); start_det = params.get('StartDetNr', 1)
        det_params = params.get('DetParams', []); n_detectors = len(det_params)
        if not det_params or px_size_um <= 0: raise ValueError("Invalid parameters for multi-detector construction.")
        mask_composite = np.zeros((big_size, big_size), dtype=float); newly_loaded_darks = {}
        first_file_nr = params.get('firstFileNumber', 1); n_frames_per_file = max(1, try_parse_int(params.get('nFramesPerFile', 1)))
        file_nr = first_file_nr + frame_nr // n_frames_per_file; frame_in_file = frame_nr % n_frames_per_file
        do_dark_corr = params.get('darkCorrectChecked', False); hflip = params.get('hflipChecked', False); vflip = params.get('vflipChecked', False); transpose = params.get('transposeChecked', False)
        total_start_time = time.time()
        for i, det_p in enumerate(det_params):
            det_nr = start_det + i; print(f"Worker: Processing Detector {det_nr}..."); det_start_time = time.time()
            file_path = get_file_path(params, file_nr, det_nr=det_nr, is_dark=False)
            if not file_path or not Path(file_path).exists(): print(f"  Worker: Skipping detector {det_nr}: Data file not found ({file_path})"); continue
            data, _ = load_image_frame(params, file_path, frame_in_file, hflip, vflip, transpose)
            if data is None: print(f"  Worker: Skipping detector {det_nr}: Failed to load data."); continue
            corrected_data = data; dark_applied_this_det = False
            if do_dark_corr:
                dark_data = dark_cache_copy.get(det_nr)
                if dark_data is None:
                     dark_fn = get_file_path(params, 0, det_nr=det_nr, is_dark=True)
                     if dark_fn and Path(dark_fn).exists(): print(f"  Worker: Loading dark frame for detector {det_nr}..."); dark_data = load_dark_frame(params, dark_fn, hflip, vflip, transpose);
                     if dark_data is not None: newly_loaded_darks[det_nr] = dark_data
                if dark_data is not None:
                     if dark_data.shape == data.shape: corrected_data = data - dark_data; corrected_data[corrected_data < 0] = 0; dark_applied_this_det = True
                     else: print(f"  Worker: Dark shape mismatch det {det_nr}.")
                else: print(f"  Worker: Dark not found/loaded det {det_nr}.")
            rows, cols = np.nonzero(corrected_data > 0)
            if rows.size == 0: print(f"  Worker: Skipping detector {det_nr}: No pixels > 0."); continue
            try:
                tx = det_p.get('tx',0.0); ty = det_p.get('ty',0.0); tz = det_p.get('tz',0.0); bc_x, bc_y = det_p.get('bc', [data.shape[1]/2.0, data.shape[0]/2.0])
                TRs = get_transform_matrix(tx, ty, tz); Yc = -(cols.astype(float) - bc_x) * px_size_um; Zc = (rows.astype(float) - bc_y) * px_size_um; Xc = np.zeros_like(Yc)
                ABC = np.vstack((Xc, Yc, Zc)); ABC_lab = np.dot(TRs, ABC)
                center_x_big = big_size / 2.0; center_y_big = big_size / 2.0; NewYs_physical = ABC_lab[1, :]; NewZs_physical = ABC_lab[2, :]
                NewCols_big = center_x_big + NewYs_physical / px_size_um; NewRows_big = center_y_big + NewZs_physical / px_size_um
                NewCols_idx = np.round(NewCols_big).astype(int); NewRows_idx = np.round(NewRows_big).astype(int); intensities = corrected_data[rows, cols]
                valid_mask = (NewCols_idx >= 0) & (NewCols_idx < big_size) & (NewRows_idx >= 0) & (NewRows_idx < big_size)
                valid_cols = NewCols_idx[valid_mask]; valid_rows = NewRows_idx[valid_mask]; valid_intensities = intensities[valid_mask]
                if valid_cols.size > 0: mask_composite[valid_rows, valid_cols] = valid_intensities
                print(f"  Worker: Det {det_nr} processing time: {time.time() - det_start_time:.3f}s, Mapped {valid_cols.size}/{rows.size} pixels.")
            except Exception as mapping_err: print(f"  Worker: Error mapping detector {det_nr}: {mapping_err}"); continue
        print(f"Worker: Total multi-detector construction time: {time.time() - total_start_time:.3f}s"); worker_data_dict.update(newly_loaded_darks)
        total_frames = n_frames_per_file * params.get('nFilesPerLayer', 1);
        if total_frames <= 0: total_frames = 1
        file_info = {'source_type': 'multi_detector', 'omega': None, 'dark_corrected': do_dark_corr}
        omega_start = params.get('omegaStart'); omega_step = params.get('omegaStep')
        if omega_start is not None and omega_step is not None: file_info['omega'] = omega_start + frame_nr * omega_step
        return mask_composite, file_info, total_frames

    def _load_max_projection_task(self, params, file_path, num_frames, start_frame, hflip, vflip, transpose, worker_data_dict):
         print(f"Worker: Calculating max projection ({num_frames} frames from {start_frame}) from {Path(file_path).name}")
         data, file_info = get_max_projection(params, file_path, num_frames, start_frame, hflip, vflip, transpose)
         if data is None: raise ValueError("Max projection calculation failed in get_max_projection.")
         corrected_data = data; dark_applied = False; newly_loaded_darks = {}
         if params.get('darkCorrectChecked', False): # Use passed state
              print("Worker: Applying dark correction to max projection.")
              dark_data = None; dark_file_path = self.current_dark_file; dark_key = "explicit_dark"
              if not dark_file_path or not Path(dark_file_path).exists():
                  dark_det_nr = params.get('StartDetNr', 1); dark_key = dark_det_nr
                  if dark_key in self.dark_frame_cache: dark_data = self.dark_frame_cache[dark_key]
                  else: dark_file_path = get_file_path(params, 0, det_nr=dark_det_nr, is_dark=True)
              else: dark_file_path = self.current_dark_file
              if dark_data is None and dark_file_path and Path(dark_file_path).exists():
                   print(f"Worker: Loading dark from {dark_file_path} for max proj."); dark_data = load_dark_frame(params, dark_file_path, hflip, vflip, transpose)
                   if dark_data is not None and dark_key != "explicit_dark": newly_loaded_darks[dark_key] = dark_data
              if dark_data is not None:
                   if dark_data.shape == data.shape: corrected_data = data - dark_data; corrected_data[corrected_data < 0] = 0; dark_applied = True; print("Worker: Dark correction applied to max projection.")
                   else: print(f"Worker: Dark shape mismatch {dark_data.shape} vs {data.shape} for max proj.")
              else: print("Worker: Dark frame not available for max proj.")
         file_info['dark_corrected'] = dark_applied
         worker_data_dict.update(newly_loaded_darks)
         total_frames = file_info.get('total_frames', 1)
         return corrected_data, file_info, total_frames

    # --- Handler for Worker Results ---
    def _handle_worker_finished(self, result, task_id, worker_data):
        if worker_data: print(f"Main thread: Updating dark cache with {len(worker_data)} frames from worker."); self.dark_frame_cache.update(worker_data)
        if task_id == "load_params": self._handle_load_params_finished(result); return # Don't re-enable UI here
        elif task_id == "make_big_det":
             success, message = result
             if success: QMessageBox.information(self, "Success", f"Big Detector Mask generation finished.\n{message}"); self.update_status_bar("Big Detector Mask generated.")
             else: QMessageBox.critical(self, "Error", f"Big Detector Mask generation failed.\n{message}"); self.update_status_bar("Big Detector Mask generation failed.")
             if self.params.get('big_det_full_path'): print(f"Big detector mask path updated to: {self.params['big_det_full_path']}")
             return # Don't re-enable UI here

        # Handle Frame Loading Results
        try:
            if not isinstance(result, tuple) or len(result) != 3: raise TypeError(f"Unexpected result format for task '{task_id}'")
            data, file_info, total_frames = result
            if data is None: raise ValueError("Loaded data is None.")
            self.total_frames_available = total_frames if total_frames > 0 else 1; self.current_frame_nr = self.current_frame_nr_pending
            self.le_frame_nr.setText(str(self.current_frame_nr)); self.lbl_total_frames.setText(f"/ {self.total_frames_available - 1}")
            self.current_file_info = file_info; omega = file_info.get('omega'); self.lbl_omega_display.setText(f"\u03C9: {omega:.4f}\N{DEGREE SIGN}" if omega is not None else "\u03C9: N/A")
            clim = (try_parse_float(self.le_min_thresh.text()), try_parse_float(self.le_max_thresh.text())); log_scale = self.cb_log_scale.isChecked()
            self._update_gui_from_params()


            if task_id == "load_single_frame" or task_id == "load_max_proj":
                self.current_data_single = data; self.current_data_multi = None
                if self.tab_widget.currentIndex() != 0: self.tab_widget.setCurrentIndex(0)
                else: self._switch_axes('single')
                if self.plotting_handler: self.plotting_handler.update_plot('single', data, self.params, clim, log_scale, file_info)
                status_msg = f"Loaded frame {self.current_frame_nr}" if task_id == "load_single_frame" else "Loaded Max Projection"; self.update_status_bar(f"{status_msg}. Source: {file_info.get('source_type', 'N/A')}")
            elif task_id == "load_multi_frame":
                self.current_data_multi = data; self.current_data_single = None
                if self.tab_widget.currentIndex() != 1: self.tab_widget.setCurrentIndex(1)
                else: self._switch_axes('multi')
                if self.plotting_handler: self.plotting_handler.update_plot('multi', data, self.params, clim, log_scale, file_info)
                self.update_status_bar(f"Loaded multi-detector frame {self.current_frame_nr}.")
        except Exception as e:
            error_msg = f"Error processing worker result for task '{task_id}': {traceback.format_exc()}"; print(error_msg)
            QMessageBox.critical(self, "Processing Error", f"Failed to process results for {task_id}.\n{e}"); self.update_status_bar(f"Error processing result for {task_id}.")
        # UI enabling handled by _on_thread_finished

    def _handle_worker_error(self, error_message, task_id):
        print(f"Worker Error ({task_id}): {error_message}")
        # UI enabling handled by _on_thread_finished
        self.show_progress(False)
        error_dialog = QMessageBox(self); error_dialog.setIcon(QMessageBox.Icon.Critical); error_dialog.setWindowTitle(f"Task Error ({task_id})"); error_dialog.setText(f"An error occurred during task '{task_id}'."); error_dialog.setInformativeText("Details below:"); error_dialog.setDetailedText(error_message); error_dialog.exec()
        self.update_status_bar(f"Error during task: {task_id}")

    def _handle_worker_progress(self, value, task_id):
        self.progress_bar.setValue(value)

    # --- Ring Control Methods ---
    def _toggle_rings(self, state):
         is_checked = bool(state); self.params['plotRingsVar'] = is_checked
         active_tab_index = self.tab_widget.currentIndex(); plot_key = 'single' if active_tab_index == 0 else 'multi'
         if is_checked:
              if not self.params.get('ringRads'): # ringRads are populated after HKL selection
                  QMessageBox.information(self, "Select Rings", "Please use 'Select Rings/Material...' first to generate and select rings.")
                  self.cb_plot_rings.setChecked(False)
                  self.params['plotRingsVar'] = False
                  return
              print("Plotting rings...")
              if self.plotting_handler: 
                  self.plotting_handler.draw_rings(plot_key, force_redraw=True)
         else:
              print("Clearing rings...")
              if self.plotting_handler:
                  if self.plotting_handler.clear_rings(plot_key): 
                      self.canvas.draw_idle()
         self._update_ring_info_label()

    def _update_ring_info_label(self):
        if self.params.get('plotRingsVar', False) and self.params.get('ringNrs'):
             hkls = self.params.get('hkls', []); rads_um = self.params.get('ringRads', []); px = try_parse_float(self.params.get('px', 1.0)); ring_nrs = self.params.get('ringNrs', [])
             display_texts = []
             for i, ring_nr in enumerate(ring_nrs): hkl_str = ",".join(map(str, hkls[i])) if i < len(hkls) else "N/A"; display_texts.append(f"Nr:{ring_nr}({hkl_str})")
             info_text = "Selected Rings: " + "; ".join(display_texts); self.lbl_ring_info.setText(info_text); self.lbl_ring_info.setToolTip(info_text)
        else: self.lbl_ring_info.setText("Selected Rings: None"); self.lbl_ring_info.setToolTip("")

    def _open_ring_selection(self):
         # Get combined parameters for the dialog (Zarr > self.params > defaults)
         params_for_dialog = self._get_combined_ring_material_params()
         material_dialog = RingMaterialDialog(params_for_dialog, self)

         if not material_dialog.exec(): 
            self.update_status_bar("Ring selection cancelled.")
            return
         new_material_params = material_dialog.get_values()
         if not new_material_params: 
            QMessageBox.warning(self, "Input Error", "Could not retrieve valid material parameters.")
            return

         # Update self.params with values from the dialog
         self.params.update({k: v for k, v in new_material_params.items() if k != 'lsd_dialog'})
         # Specifically update DetParams[0]['lsd'] if it was changed in dialog
         if 'lsd_dialog' in new_material_params:
             if 'DetParams' not in self.params or not self.params['DetParams']: 
                 self.params['DetParams'] = [{}]
             self.params['DetParams'][0]['lsd'] = new_material_params['lsd_dialog']
             # Also update the GUI display for LSD directly
             self.le_lsd_single.setText(f"{new_material_params['lsd_dialog']:.1f}")

         self.le_px_display.setText(f"{self.params['px']:.2f}") # Update px display from self.params
         self.update_status_bar("Material parameters updated. Processing HKL list...")

         hkl_gen_path = self.midas_dir / "bin" / "GetHKLList"
         hkl_csv_path_cwd = Path.cwd() / "hkls.csv" # For fallback
         hkl_lines_data = []; header_line = ""
         hkl_source_message = ""

         if hkl_gen_path.is_file():
             temp_param_fn = Path.cwd() / "temp_ring_params_for_hkl.txt"
             try:
                 with open(temp_param_fn, 'w') as f:
                     f.write(f"Wavelength {self.params['wl']}\n"); 
                     f.write(f"SpaceGroup {self.params['sg']}\n")
                     # Use LSD from DetParams[0] for GetHKLList if available, else top-level, else default
                     lsd_for_hkl = self.params.get('DetParams', [{}])[0].get('lsd', self.params.get('lsd', 1e6))
                     f.write(f"Lsd {lsd_for_hkl}\n")
                     f.write(f"MaxRingRad {self.params['maxRad']}\n")
                     lc_str = " ".join(map(str, self.params['LatticeConstant']))
                     f.write(f"LatticeConstant {lc_str}\n")
                     f.write(f"px {self.params['px']}\n")
                 print(f"Running HKL generator: {hkl_gen_path} {temp_param_fn}")
                 result = subprocess.run([str(hkl_gen_path), str(temp_param_fn)], capture_output=True, text=True, encoding='utf-8', timeout=10) # Added timeout
                 print("HKL Gen Output:", result.stdout)
                 if result.stderr: 
                     print("HKL Gen Error:", result.stderr)
                 if result.returncode != 0: 
                     raise RuntimeError(f"GetHKLList failed (retcode {result.returncode}):\n{result.stderr}")
                 
                 hkl_csv_path_generated = Path.cwd() / "hkls.csv" # Standard output name
                 if not hkl_csv_path_generated.exists(): 
                     raise FileNotFoundError("hkls.csv not generated by GetHKLList.")
                 
                 with open(hkl_csv_path_cwd, 'r') as f:
                    # Read the header line first (if it exists and you want to use it)
                    header_content = f.readline().strip()
                    if header_content: # Check if header is not empty
                            # Split by one or more spaces for the header display
                        header_line_list = header_content.split()
                        header_line = "  ".join(header_line_list) # Rejoin with double space for display
                    else:
                        header_line = "" # Or your default header string

                    hkl_lines_data_all = [] # Clear any previous attempt
                    for line_num, line_content in enumerate(f): # Iterate over remaining lines
                        line_content = line_content.strip()
                        if not line_content or line_content.startswith('#'): # Skip empty or comment lines
                            continue
                        
                        parts = line_content.split() # Split by any whitespace

                        if len(parts) >= 11: # Check if we have enough parts after splitting
                            try:
                                hkl = [int(parts[0]), int(parts[1]), int(parts[2])]
                                ring_nr = int(parts[4])
                                ring_rad_um = float(parts[10])
                                # For display_line, we can just use the parts joined by a couple of spaces
                                display_line = "  ".join(parts) 
                                hkl_lines_data_all.append({'display': display_line, 'hkl': hkl, 'nr': ring_nr, 'rad': ring_rad_um})
                            except (ValueError, IndexError) as e_parse:
                                print(f"Warning: Skipping malformed HKL line {line_num+2} from file (parsing error: {e_parse}): {parts}")
                        else:
                            print(f"Warning: Skipping short HKL line {line_num+2} from file (parts found: {len(parts)}): {parts}")
                    hkl_lines_data_filtered = []
                    seen_ring_numbers = set()
                    if hkl_lines_data_all:
                        # Sort by RingNr, then by original order (implicitly by radius or how they appear in file)
                        # This ensures that if multiple entries for the same ring have slightly different radii,
                        # we consistently pick one (though GetHKLList should give same radius for same RingNr)
                        # Sorting is not strictly necessary if GetHKLList already groups/orders them,
                        # but it's safer.
                        # hkl_lines_data_all.sort(key=lambda x: (x['nr'], x['rad'])) # Optional sort

                        for item in hkl_lines_data_all:
                            if item['nr'] not in seen_ring_numbers:
                                hkl_lines_data_filtered.append((item['display'], item['hkl'], item['nr'], item['rad']))
                                seen_ring_numbers.add(item['nr'])
                    
                    hkl_lines_data = hkl_lines_data_filtered # Use the filtered list
                 hkl_source_message = "HKLs generated successfully."
             except subprocess.TimeoutExpired:
                 QMessageBox.warning(self, "HKL Generation Timeout", f"GetHKLList timed out. Trying to read existing hkls.csv...")
                 hkl_gen_path = None # Force fallback by "unfinding" the generator
             except Exception as e_gen:
                 QMessageBox.warning(self, "HKL Generation Error", f"Failed to generate HKL list: {e_gen}\nTrying to read existing hkls.csv...")
                 hkl_gen_path = None # Force fallback
             finally:
                 if temp_param_fn.exists():
                    try: os.remove(temp_param_fn)
                    except OSError: pass
         
         if not hkl_gen_path or not hkl_lines_data: # If generator not found OR generation failed and produced no data
            if hkl_csv_path_cwd.is_file():
                print(f"HKL generator not found or failed. Reading from existing: {hkl_csv_path_cwd}")
                try:
                    with open(hkl_csv_path_cwd, 'r') as f:
                        # Read the header line first (if it exists and you want to use it)
                        header_content = f.readline().strip()
                        if header_content: # Check if header is not empty
                             # Split by one or more spaces for the header display
                            header_line_list = header_content.split()
                            header_line = "  ".join(header_line_list) # Rejoin with double space for display
                        else:
                            header_line = "" # Or your default header string

                        hkl_lines_data_all = [] # Clear any previous attempt
                        for line_num, line_content in enumerate(f): # Iterate over remaining lines
                            line_content = line_content.strip()
                            if not line_content or line_content.startswith('#'): # Skip empty or comment lines
                                continue
                            
                            parts = line_content.split() # Split by any whitespace

                            if len(parts) >= 11: # Check if we have enough parts after splitting
                                try:
                                    hkl = [int(parts[0]), int(parts[1]), int(parts[2])]
                                    ring_nr = int(parts[4])
                                    ring_rad_um = float(parts[10])
                                    # For display_line, we can just use the parts joined by a couple of spaces
                                    display_line = "  ".join(parts) 
                                    hkl_lines_data_all.append({'display': display_line, 'hkl': hkl, 'nr': ring_nr, 'rad': ring_rad_um})
                                except (ValueError, IndexError) as e_parse:
                                    print(f"Warning: Skipping malformed HKL line {line_num+2} from file (parsing error: {e_parse}): {parts}")
                            else:
                                print(f"Warning: Skipping short HKL line {line_num+2} from file (parts found: {len(parts)}): {parts}")
                        hkl_lines_data_filtered = []
                        seen_ring_numbers = set()
                        if hkl_lines_data_all:
                            # Sort by RingNr, then by original order (implicitly by radius or how they appear in file)
                            # This ensures that if multiple entries for the same ring have slightly different radii,
                            # we consistently pick one (though GetHKLList should give same radius for same RingNr)
                            # Sorting is not strictly necessary if GetHKLList already groups/orders them,
                            # but it's safer.
                            # hkl_lines_data_all.sort(key=lambda x: (x['nr'], x['rad'])) # Optional sort

                            for item in hkl_lines_data_all:
                                if item['nr'] not in seen_ring_numbers:
                                    hkl_lines_data_filtered.append((item['display'], item['hkl'], item['nr'], item['rad']))
                                    seen_ring_numbers.add(item['nr'])
                        
                        hkl_lines_data = hkl_lines_data_filtered # Use the filtered list

                    if hkl_lines_data: 
                        hkl_source_message = f"HKLs read from: {hkl_csv_path_cwd.name}"
                    else: 
                        hkl_source_message = f"Found {hkl_csv_path_cwd.name}, but no valid HKLs read."
                except Exception as e_read:
                    QMessageBox.critical(self, "HKL Read Error", f"Failed to read HKL list from {hkl_csv_path_cwd.name}: {traceback.format_exc()}")
                    return # Critical error, cannot proceed
            else:
                QMessageBox.critical(self, "HKL Error", f"GetHKLList not found at {self.midas_dir / 'bin' / 'GetHKLList'}\nAND hkls.csv not found in current directory ({Path.cwd()}). Cannot proceed."); return

         if not hkl_lines_data: 
            QMessageBox.warning(self, "HKL Data", f"No valid HKL lines found from any source.\n({hkl_source_message})")
            return
         self.update_status_bar(hkl_source_message)

         dialog = QDialog(self)
         dialog.setWindowTitle("Select Rings to Display")
         dialog_layout = QVBoxLayout(dialog)
         dialog_layout.addWidget(QLabel("Available Rings (Select multiple using Ctrl/Shift):"))
         dialog_layout.addWidget(QLabel(header_line if header_line else "H K L ?? RingNr ?? ?? ?? ?? ?? Radius_um", font=QFont("Monospace"))) # Default header if none read
         list_widget = QListWidget()
         list_widget.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
         list_widget.setFont(QFont("Monospace"))
         list_widget_items_data = []
         for display_line, hkl, ring_nr, ring_rad_um in hkl_lines_data:
             list_widget.addItem(display_line)
             list_widget_items_data.append({'hkl': hkl, 'nr': ring_nr, 'rad': ring_rad_um})
         dialog_layout.addWidget(list_widget)
         button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
         button_box.accepted.connect(dialog.accept)
         button_box.rejected.connect(dialog.reject)
         dialog_layout.addWidget(button_box)
         if dialog.exec():
              selected_indices = [item.row() for item in list_widget.selectedIndexes()] # Corrected for PyQt6
              if selected_indices:
                   selected_rings_data = [list_widget_items_data[i] for i in selected_indices]
                   self.params['hkls'] = [item['hkl'] for item in selected_rings_data]
                   self.params['ringNrs'] = [item['nr'] for item in selected_rings_data]
                   self.params['ringRads'] = [item['rad'] for item in selected_rings_data]
                   self.params['RingsToShow'] = self.params['ringNrs'] # Legacy? Keep for now.
                   self.params['plotRingsVar'] = True
                   self.cb_plot_rings.setChecked(True)
                   self._update_ring_info_label()
                   active_tab_index = self.tab_widget.currentIndex()
                   plot_key = 'single' if active_tab_index == 0 else 'multi'
                   if self.plotting_handler: 
                       self.plotting_handler.draw_rings(plot_key, force_redraw=True)
              else: self._clear_ring_selection() # No rings selected
         else: self._clear_ring_selection() # Dialog cancelled

    def _clear_ring_selection(self):
         self.params['hkls'] = []; self.params['ringNrs'] = []; self.params['ringRads'] = []; self.params['RingsToShow'] = []; self.params['plotRingsVar'] = False
         self.cb_plot_rings.setChecked(False); self._update_ring_info_label();
         if self.plotting_handler: self.plotting_handler.clear_rings('single'); self.plotting_handler.clear_rings('multi'); self.canvas.draw_idle()

    def _update_ring_geometry(self):
        try:
             det_num = try_parse_int(self.le_det_num_single.text()); 
             lsd = try_parse_float(self.le_lsd_single.text()); 
             gui_bc_plotting_horizontal = try_parse_float(self.le_bcx_single.text())
             gui_bc_plotting_vertical = try_parse_float(self.le_bcy_single.text())
             det_idx = det_num - try_parse_int(self.params.get('StartDetNr', 1))
             det_idx = det_num - try_parse_int(self.params.get('StartDetNr', 1))
             if 'DetParams' in self.params and 0 <= det_idx < len(self.params['DetParams']):
                  print(f"Updating params from GUI for det {det_num}: LSD={lsd}, BC_Plotting=[{gui_bc_plotting_horizontal}, {gui_bc_plotting_vertical}]")
                  self.params['DetParams'][det_idx]['lsd'] = lsd
                  # Store in self.params in the [plotting_horizontal, plotting_vertical] order
                  self.params['DetParams'][det_idx]['bc'] = [gui_bc_plotting_horizontal, gui_bc_plotting_vertical]
                  self.params['current_det_nr_single'] = det_num
                  
                  if self.plotting_handler:
                      # PlottingHandler.current_single_bc_gui expects [plotting_horizontal, plotting_vertical]
                      self.plotting_handler.current_single_bc_gui = [gui_bc_plotting_horizontal, gui_bc_plotting_vertical]
                      self.plotting_handler.current_single_det_nr = det_num
             else:
                 QMessageBox.warning(self, "Update Error", f"Detector number {det_num} not found or invalid in current parameters."); return

             if self.cb_plot_rings.isChecked():
                 active_tab_index = self.tab_widget.currentIndex();
                 plot_key = 'single' if active_tab_index == 0 else 'multi'
                 if self.plotting_handler:
                     print(f"Redrawing rings for '{plot_key}' after geometry update.")
                     self.plotting_handler.draw_rings(plot_key, force_redraw=True)
             else:
                 print("Ring geometry updated in params, but rings are not plotted.")

        except (ValueError, TypeError): QMessageBox.warning(self, "Input Error", "Invalid number format for Detector Number, LSD or Beam Center.")
        except Exception as e: QMessageBox.warning(self, "Parameter Error", f"Could not update geometry: {e}\n{traceback.format_exc()}")

    # --- Max Projection Methods ---
    def _toggle_max_proj_mode(self, state):
         is_checked = bool(state)
         self.le_frame_nr.setEnabled(not is_checked); self.btn_prev_frame.setEnabled(not is_checked); self.btn_next_frame.setEnabled(not is_checked)
         if is_checked:
              self.update_status_bar("Max projection mode enabled. Use 'Load Max Projection' button.")
              if self.tab_widget.currentIndex() != 0: self.tab_widget.setCurrentIndex(0)
         else:
              self.update_status_bar("Max projection mode disabled. Switched to single frame view.")
              if self.current_frame_nr is not None and self.total_frames_available > 0:
                   self.current_frame_nr = max(0, min(self.current_frame_nr, self.total_frames_available - 1))
                   self.le_frame_nr.setText(str(self.current_frame_nr)); self._reload_current_frame()
              elif self.current_frame_nr is None: self._load_frame(0)

    def _load_max_projection_action(self):
         if not self.current_data_file: QMessageBox.warning(self, "File Error", "Select a data file (Raw or Zarr) first."); self.cb_use_max_proj.setChecked(False); self._toggle_max_proj_mode(False); return
         if not Path(self.current_data_file).exists(): QMessageBox.warning(self, "File Error", f"Data file not found: {self.current_data_file}"); self.cb_use_max_proj.setChecked(False); self._toggle_max_proj_mode(False); return
         try:
              num_frames = try_parse_int(self.le_max_frames.text()); start_frame = try_parse_int(self.le_max_start_frame.text())
              if num_frames <= 0: raise ValueError("Number of frames must be positive.")
              if start_frame < 0: raise ValueError("Start frame cannot be negative.")
              if self.tab_widget.currentIndex() != 0: self.tab_widget.setCurrentIndex(0)
              params_copy = self.params.copy()
              params_copy['darkCorrectChecked'] = self.cb_dark_correct.isChecked(); params_copy['hflipChecked'] = self.cb_hflip.isChecked(); params_copy['vflipChecked'] = self.cb_vflip.isChecked(); params_copy['transposeChecked'] = self.cb_transpose.isChecked()
              args = [params_copy, self.current_data_file, num_frames, start_frame, params_copy['hflipChecked'], params_copy['vflipChecked'], params_copy['transposeChecked']]
              self.current_frame_nr_pending = start_frame
              self.run_in_thread(self._load_max_projection_task, "load_max_proj", *args)
         except ValueError as e: QMessageBox.warning(self, "Input Error", f"Invalid input for max projection: {e}"); self.cb_use_max_proj.setChecked(False); self._toggle_max_proj_mode(False)

    # --- Hydra Methods ---
    def _make_big_detector(self):
         if not self.params.get('HydraActive', False) or not self.current_param_file: QMessageBox.warning(self, "Prerequisite Missing", "Load Hydra parameters from a file first."); return
         reply = QMessageBox.question(self, 'Confirmation', f"This will run the external 'MapMultipleDetectors'\n(expected in {self.midas_dir / 'bin'}). Proceed?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
         if reply == QMessageBox.StandardButton.Yes: self.run_in_thread(generate_big_detector_mask, "make_big_det", self.params.copy(), self.midas_dir)

    def _load_multi_detector_action(self):
        if not self.params.get('HydraActive', False): QMessageBox.warning(self, "Mode Error", "Load Hydra parameters first."); return
        frame_nr = self.current_frame_nr
        if self.tab_widget.currentIndex() != 1: self.tab_widget.setCurrentIndex(1)
        else: self._load_frame(frame_nr) # Trigger load if already on tab

    # --- Tab Management ---
    def _on_tab_changed(self, index):
        print(f"Switched to tab index: {index}")
        if self.plotting_handler: self.plotting_handler.clear_rings('single' if index == 1 else 'multi')
        if index == 0: # Single Detector
             self._switch_axes('single')
             if self.current_data_single is not None: self._update_display()
             else:
                  if hasattr(self, 'ax_single') and self.ax_single:
                    self.ax_single.clear()
                    if self.plotting_handler: self.plotting_handler.img_single = None
                    self.ax_single.set_title("Single Detector Display"); self.ax_single.set_xlabel("Lab Y (pixels - Horizontal)"); self.ax_single.set_ylabel("Lab Z (pixels - Vertical)")
                    self.canvas.draw_idle()
             status = "Switched to Single Detector view."
             if self.is_zarr_mode: status += " (Zarr Mode)"
             elif hasattr(self,'cb_use_max_proj') and self.cb_use_max_proj.isChecked(): status += " (Max Projection Mode)"
             self.update_status_bar(status)
        elif index == 1: # Multi Detector
             if self.params.get('HydraActive', False):
                  self._switch_axes('multi')
                  if self.current_data_multi is not None: self._update_display()
                  else:
                      if hasattr(self, 'ax_multi') and self.ax_multi:
                        self.ax_multi.clear()
                        if self.plotting_handler: self.plotting_handler.img_multi = None
                        self.ax_multi.text(0.5, 0.5, 'Use "Load Multi-Detector View" button', horizontalalignment='center', verticalalignment='center', transform=self.ax_multi.transAxes, fontsize=12, color='grey')
                        self.ax_multi.set_title("Multiple Detectors (Hydra)"); self.ax_multi.set_xlabel("Lab Y (pixels - Horizontal)"); self.ax_multi.set_ylabel("Lab Z (pixels - Vertical)")
                        self.canvas.draw_idle()
                  self.update_status_bar("Switched to Multi Detector view.")
             else: self._switch_axes('multi_placeholder'); self.update_status_bar("Load Hydra parameters to enable Multi Detector view.")

    def _switch_axes(self, mode):
        # Move canvas container
        if mode == 'single':
             if self.canvas_container.parent() != self.tab_widget.widget(0):
                  self.canvas_container.setParent(self.tab_widget.widget(0)); self.tab_widget.widget(0).layout().addWidget(self.canvas_container)
                  self.multi_placeholder_label.setParent(None); self.canvas_container.setVisible(True); self.multi_placeholder_label.setVisible(False)
        elif mode == 'multi':
             if self.canvas_container.parent() != self.tab_widget.widget(1):
                  self.canvas_container.setParent(self.tab_widget.widget(1)); self.tab_widget.widget(1).layout().addWidget(self.canvas_container)
                  self.multi_placeholder_label.setParent(None); self.canvas_container.setVisible(True); self.multi_placeholder_label.setVisible(False)
        elif mode == 'multi_placeholder':
             if self.multi_placeholder_label.parent() != self.tab_widget.widget(1):
                  self.canvas_container.setParent(None); self.multi_placeholder_label.setParent(self.tab_widget.widget(1))
                  self.tab_widget.widget(1).layout().addWidget(self.multi_placeholder_label); self.canvas_container.setVisible(False); self.multi_placeholder_label.setVisible(True)

        # Manage axes
        target_ax = None; target_ax_attr = None; handler_ax_attr = None; handler_img_attr = None
        if mode == 'single': target_ax_attr = 'ax_single'; handler_ax_attr = 'ax_single'; handler_img_attr = 'img_single'
        elif mode == 'multi': target_ax_attr = 'ax_multi'; handler_ax_attr = 'ax_multi'; handler_img_attr = 'img_multi'
        elif mode == 'multi_placeholder':
             self.figure.clear(); self.ax_single = None; self.ax_multi = None
             if self.plotting_handler: self.plotting_handler.ax_single = None; self.plotting_handler.ax_multi = None; self.plotting_handler.img_single = None; self.plotting_handler.img_multi = None
             print("Cleared figure for placeholder"); self.canvas.draw_idle(); return

        target_ax = getattr(self, target_ax_attr, None)
        if not target_ax or target_ax not in self.figure.axes:
            self.figure.clear(); target_ax = self.figure.add_subplot(111); setattr(self, target_ax_attr, target_ax)
            if self.plotting_handler: setattr(self.plotting_handler, handler_ax_attr, target_ax); setattr(self.plotting_handler, handler_img_attr, None)
            other_ax_attr = 'ax_multi' if mode == 'single' else 'ax_single'; other_handler_ax_attr = 'ax_multi' if mode == 'single' else 'ax_single'; other_handler_img_attr = 'img_multi' if mode == 'single' else 'img_single'
            setattr(self, other_ax_attr, None)
            if self.plotting_handler: setattr(self.plotting_handler, other_handler_ax_attr, None); setattr(self.plotting_handler, other_handler_img_attr, None)
            print(f"Recreated {target_ax_attr}")

        for ax in self.figure.get_axes(): # Ensure only target is present
            if ax != target_ax:
                try: self.figure.delaxes(ax)
                except ValueError: pass
        if target_ax and target_ax not in self.figure.get_axes():
            try: self.figure.add_axes(target_ax)
            except ValueError: pass

        if self.plotting_handler: setattr(self.plotting_handler, handler_ax_attr, target_ax)
        self.toolbar.update(); self.canvas.draw_idle()


    # --- Application Exit ---
    def closeEvent(self, event):
        if hasattr(self, 'worker_thread') and self.worker_thread is not None and self.worker_thread.isRunning():
             reply = QMessageBox.question(self, 'Confirm Exit', "A task is running. Stop and exit?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
             if reply == QMessageBox.StandardButton.Yes:
                  if hasattr(self, 'worker') and self.worker: self.worker.stop()
                  self.worker_thread.quit()
                  if not self.worker_thread.wait(500): print("Warning: Worker thread did not terminate gracefully.")
             else: event.ignore(); return
        self._save_settings()
        event.accept()

# --- Entry Point ---
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = FFViewerApp()
    window.show()
    sys.exit(app.exec())