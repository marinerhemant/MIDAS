# plotting.py
import matplotlib.pyplot as plt
import numpy as np
import time
from pathlib import Path

from utils import YZ4mREta, CalcEtaAngleRad, colors, rad2deg, try_parse_float, get_transform_matrix

class PlottingHandler:
    def __init__(self, axes_single, axes_multi, canvas, navigation_toolbar, status_bar_update_func):
        self.ax_single = axes_single
        self.ax_multi = axes_multi
        self.canvas = canvas
        self.toolbar = navigation_toolbar # Store toolbar reference
        self.status_bar_update = status_bar_update_func

        self.img_single = None
        self.img_multi = None
        self.rings_single = []
        self.rings_multi = []
        self.current_single_data = None
        self.current_multi_data = None
        self.current_params = {}
        self.current_single_bc_gui = [0, 0]
        self.current_single_det_nr = 1

        self._connect_events()

    def _connect_events(self):
        self.canvas.mpl_connect('motion_notify_event', self._on_motion)
        self.canvas.mpl_connect('button_press_event', self._on_double_click)

    def _on_double_click(self, event):
        if event.dblclick:
            print("Double click detected, resetting view.")
            ax_list = []
            if hasattr(self, 'ax_single') and self.ax_single and self.ax_single.get_visible(): ax_list.append(self.ax_single)
            if hasattr(self, 'ax_multi') and self.ax_multi and self.ax_multi.get_visible(): ax_list.append(self.ax_multi)
            if event.inaxes in ax_list:
                if self.toolbar: self.toolbar.home()
                else: event.inaxes.autoscale_view(); self.canvas.draw_idle()

    def _on_motion(self, event):
        active_ax = None; data_to_use = None; beam_center = [0, 0]
        single_ax_exists = hasattr(self, 'ax_single') and self.ax_single and self.ax_single.get_visible()
        multi_ax_exists = hasattr(self, 'ax_multi') and self.ax_multi and self.ax_multi.get_visible()

        if event.inaxes == self.ax_single and single_ax_exists: active_ax = self.ax_single; data_to_use = self.current_single_data; beam_center = self.current_single_bc_gui
        elif event.inaxes == self.ax_multi and multi_ax_exists:
             active_ax = self.ax_multi; data_to_use = self.current_multi_data
             if self.current_multi_data is not None: beam_center = [self.current_multi_data.shape[1] / 2.0, self.current_multi_data.shape[0] / 2.0]
        else: self.status_bar_update(""); return

        if event.xdata is None or event.ydata is None or data_to_use is None: self.status_bar_update(""); return

        col = int(round(event.xdata)); row = int(round(event.ydata))
        num_rows, num_cols = data_to_use.shape
        status_text = f"X: {event.xdata:.2f}, Y: {event.ydata:.2f}"
        rel_x_px = event.xdata - beam_center[0]; rel_y_px = event.ydata - beam_center[1]
        eta_deg, radius_px = CalcEtaAngleRad(-rel_x_px, rel_y_px)

        if 0 <= row < num_rows and 0 <= col < num_cols: intensity = data_to_use[row, col]; status_text += f"  |  Intensity: {intensity:.1f}"
        else: status_text += "  |  (Outside Image)"
        status_text += f"  |  Eta: {eta_deg:.2f}\N{DEGREE SIGN}  |  Radius (px): {radius_px:.2f}"
        self.status_bar_update(status_text)

    def update_plot(self, which_plot, data, params, clim, log_scale, file_info):
        if data is None: print(f"No data provided for {which_plot} plot."); return
        ax = self.ax_single if which_plot == 'single' else self.ax_multi
        if not ax: print(f"Axes for {which_plot} plot not available."); return
        img_artist = self.img_single if which_plot == 'single' else self.img_multi
        self.current_params = params

        if which_plot == 'single':
            self.current_single_data = data; self.current_single_det_nr = params.get('current_det_nr_single', 1)
            det_idx = self.current_single_det_nr - params.get('StartDetNr', 1)
            if 0 <= det_idx < len(params.get('DetParams',[])): self.current_single_bc_gui = params['DetParams'][det_idx].get('bc', [data.shape[1]/2.0, data.shape[0]/2.0])
            else: self.current_single_bc_gui = [data.shape[1]/2.0, data.shape[0]/2.0]
        else: self.current_multi_data = data

        valid_clim = (try_parse_float(clim[0], 0.0), try_parse_float(clim[1], 1000.0))
        if valid_clim[0] >= valid_clim[1]: valid_clim = (valid_clim[0], valid_clim[0] + 100.0)
        plot_data = data.copy(); plot_clim = valid_clim
        if log_scale:
            min_val = max(1.0, plot_clim[0]); max_val = max(min_val + 1, plot_clim[1])
            plot_data[plot_data < min_val] = min_val
            plot_data = np.log(plot_data); plot_clim = (np.log(min_val), np.log(max_val)); cmap = 'viridis'
        else: cmap = 'gray'

        xlim = ax.get_xlim(); ylim = ax.get_ylim(); first_plot = (img_artist is None)

        if first_plot:
            img_artist = ax.imshow(plot_data, cmap=cmap, vmin=plot_clim[0], vmax=plot_clim[1], interpolation='nearest', origin='lower', aspect='equal')
            if which_plot == 'single': ax.set_xlabel("Lab Y (pixels - Horizontal)"); ax.set_ylabel("Lab Z (pixels - Vertical)"); self.img_single = img_artist
            else: ax.set_xlabel("Lab Y (pixels - Horizontal)"); ax.set_ylabel("Lab Z (pixels - Vertical)"); self.img_multi = img_artist
        else: img_artist.set_data(plot_data); img_artist.set_cmap(cmap); img_artist.set_clim(plot_clim)

        frame_nr = params.get('current_frame_nr', 0); omega = file_info.get('omega')
        title = f"{which_plot.capitalize()} Detector | Frame: {frame_nr}"
        if omega is not None: title += f" | \u03C9: {omega:.4f}\N{DEGREE SIGN}"
        source_type = file_info.get('source_type', 'N/A'); title += f" | Source: {source_type}"
        ax.set_title(title)

        if not first_plot: ax.set_xlim(xlim); ax.set_ylim(ylim)
        else: ax.autoscale_view()
        self.draw_rings(which_plot)
        self.canvas.draw_idle()

    def clear_rings(self, which_plot):
        ring_list = self.rings_single if which_plot == 'single' else self.rings_multi
        rings_removed = False
        for line_set in ring_list[:]:
            for line in line_set:
                try: line.remove(); rings_removed = True
                except ValueError: pass
            if line_set in ring_list: ring_list.remove(line_set) # Should always be true here
        # Return flag indicating if anything was removed
        return rings_removed


    def draw_rings(self, which_plot, force_redraw=False):
        """Draws diffraction rings on the specified plot based on current params."""
        ax = self.ax_single if which_plot == 'single' else self.ax_multi
        ring_list = self.rings_single if which_plot == 'single' else self.rings_multi
        plot_rings_enabled = self.current_params.get('plotRingsVar', False)

        if not ax or not ax.get_visible(): return

        rings_cleared = False
        if not plot_rings_enabled or force_redraw:
             rings_cleared = self.clear_rings(which_plot)
             if not plot_rings_enabled:
                  if rings_cleared: self.canvas.draw_idle()
                  return

        if plot_rings_enabled and ring_list and not force_redraw: return

        ring_radii_um = self.current_params.get('ringRads', [])
        px_size_um = try_parse_float(self.current_params.get('px'))
        if not ring_radii_um or not px_size_um or px_size_um == 0: print("Ring radii/pixel size invalid."); return

        if which_plot == 'single':
             # --- Use BC associated with the *current* single detector ---
             det_idx = self.current_single_det_nr - self.current_params.get('StartDetNr', 1)
             if 0 <= det_idx < len(self.current_params.get('DetParams',[])):
                  det_p = self.current_params['DetParams'][det_idx]
                  # Get BC directly from the stored params for this detector
                  bc_x_px, bc_y_px = det_p.get('bc', self.current_single_bc_gui) # Fallback to GUI value if missing
                  lsd_um = det_p.get('lsd')
                  if lsd_um is None: print(f"Warning: LSD not found for det {self.current_single_det_nr}."); lsd_um = 1e6
             else: # Fallback if index invalid
                  bc_x_px, bc_y_px = self.current_single_bc_gui # Use GUI value
                  lsd_um = try_parse_float(self.current_params.get('lsdlocalvar_gui', 1e6)) # Less ideal fallback
             # ---------------------------------------------------------
        else: # Multi detector plot
             if self.current_multi_data is None: return
             rows, cols = self.current_multi_data.shape; bc_x_px, bc_y_px = cols / 2.0, rows / 2.0

        print(f"Drawing rings for '{which_plot}' using BC=[{bc_x_px:.2f}, {bc_y_px:.2f}]") # Debug print

        etas = np.linspace(-180, 180, num=181); new_ring_list = []
        for i, radius_um in enumerate(ring_radii_um):
            radius_px = radius_um / px_size_um
            plot_x_coords = []; plot_y_coords = []
            for eta in etas:
                Y_offset_px, Z_offset_px = YZ4mREta(radius_px * px_size_um, eta) # Get um offsets
                Y_offset_px /= px_size_um # Convert to pixel offsets
                Z_offset_px /= px_size_um
                plot_col = bc_x_px - Y_offset_px # Y left is positive phys -> negative col offset
                plot_row = bc_y_px + Z_offset_px # Z up is positive phys -> positive row offset
                plot_x_coords.append(plot_col); plot_y_coords.append(plot_row)

            hkl_info = self.current_params.get('hkls', [])
            hkl_str = ",".join(map(str, hkl_info[i])) if i < len(hkl_info) else "N/A"
            ring_nr_list = self.current_params.get('ringNrs', [])
            ring_nr_str = str(ring_nr_list[i]) if i < len(ring_nr_list) else "?"
            label = f"Ring {ring_nr_str} ({hkl_str})"
            line = ax.plot(plot_x_coords, plot_y_coords, color=colors[i % len(colors)], linestyle=':', linewidth=1.0, label=label)
            new_ring_list.append(line)

        if which_plot == 'single': self.rings_single = new_ring_list
        else: self.rings_multi = new_ring_list
        self.canvas.draw_idle() # Draw now that rings are added/redrawn
