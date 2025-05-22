# data_processing_dash.py
import numpy as np
import os
import time
import zarr
from zarr.storage import ZipStore
import h5py # For inspection, though zarr handles hdf5-backed stores if needed
from numba import jit, prange
from pathlib import Path
# from PIL import Image # Not directly used for data loading here
import traceback
import subprocess # Kept for generate_big_detector_mask if re-enabled

import utils_dash as utils

# --- Numba Max Projection (No changes from previous) ---
@jit(nopython=True, parallel=True)
def calculate_max_projection_numba_uint16(filename, header_size, bytes_per_pixel,
                                         rows, cols, num_frames, start_frame):
    if bytes_per_pixel != 2: return np.zeros((rows, cols), dtype=np.uint16)
    frame_size_bytes = rows * cols * bytes_per_pixel
    total_offset = header_size + start_frame * frame_size_bytes
    max_image = np.zeros((rows, cols), dtype=np.uint16)
    actual_frames_read = 0
    try:
        with open(filename, 'rb') as f_ptr:
            f_ptr.seek(total_offset)
            all_data_bytes = f_ptr.read(num_frames * frame_size_bytes)
        if len(all_data_bytes) < frame_size_bytes: return max_image
        actual_frames_read = len(all_data_bytes) // frame_size_bytes
        if actual_frames_read == 0: return max_image
        valid_bytes = actual_frames_read * frame_size_bytes
        all_data_arr = np.frombuffer(all_data_bytes[:valid_bytes], dtype=np.uint16)
        if all_data_arr.size % (rows * cols) != 0:
            num_valid_elements_for_reshape = (all_data_arr.size // (rows * cols)) * (rows * cols)
            if num_valid_elements_for_reshape == 0 : return max_image
            all_data_arr = all_data_arr[:num_valid_elements_for_reshape]
            actual_frames_read = all_data_arr.size // (rows*cols)
            if actual_frames_read == 0: return max_image
        all_data = all_data_arr.reshape((actual_frames_read, rows, cols))
        for r in prange(rows):
            for c in prange(cols):
                max_val = np.uint16(0)
                for frame_idx in range(actual_frames_read):
                    val = all_data[frame_idx, r, c]
                    if val > max_val: max_val = val
                max_image[r, c] = max_val
    except Exception as e: return np.zeros((rows, cols), dtype=np.uint16)
    return max_image

@jit(nopython=True, parallel=True)
def calculate_max_projection_numba_int32(filename, header_size, bytes_per_pixel,
                                         rows, cols, num_frames, start_frame):
    if bytes_per_pixel != 4: return np.zeros((rows, cols), dtype=np.int32)
    frame_size_bytes = rows * cols * bytes_per_pixel
    total_offset = header_size + start_frame * frame_size_bytes
    max_image = np.zeros((rows, cols), dtype=np.int32)
    actual_frames_read = 0
    try:
        with open(filename, 'rb') as f_ptr:
            f_ptr.seek(total_offset)
            all_data_bytes = f_ptr.read(num_frames * frame_size_bytes)
        if len(all_data_bytes) < frame_size_bytes: return max_image
        actual_frames_read = len(all_data_bytes) // frame_size_bytes
        if actual_frames_read == 0: return max_image
        valid_bytes = actual_frames_read * frame_size_bytes
        all_data_arr = np.frombuffer(all_data_bytes[:valid_bytes], dtype=np.int32)
        if all_data_arr.size % (rows * cols) != 0:
            num_valid_elements_for_reshape = (all_data_arr.size // (rows * cols)) * (rows * cols)
            if num_valid_elements_for_reshape == 0: return max_image
            all_data_arr = all_data_arr[:num_valid_elements_for_reshape]
            actual_frames_read = all_data_arr.size // (rows*cols)
            if actual_frames_read == 0: return max_image
        all_data = all_data_arr.reshape((actual_frames_read, rows, cols))
        for r in prange(rows):
            for c in prange(cols):
                max_val = np.int32(-2147483647 - 1)
                for frame_idx in range(actual_frames_read):
                    val = all_data[frame_idx, r, c]
                    if val > max_val: max_val = val
                max_image[r, c] = max_val
    except Exception as e: return np.zeros((rows, cols), dtype=np.int32)
    return max_image

# --- Parameter Extraction and Data Loading (Zarr Focused) ---
def extract_params_from_zarr(zarr_file_path_str):
    params = {
        'DetParams': [{'lsd': None, 'bc': [None, None], 'tx': 0.0, 'ty': 0.0, 'tz': 0.0,
                       'p0': 0.0, 'p1': 0.0, 'p2': 0.0}],
        'px': None, 'wl': None, 'sg': None, 'LatticeConstant': None,
        'omegaStart': None, 'omegaStep': None,
        'dataLoc': 'exchange/data',
        'darkLoc': 'exchange/dark',
        'NrPixelsY_detector': None, # Horizontal detector dimension (e.g., columns if Y is fast scan)
        'NrPixelsZ_detector': None, # Vertical detector dimension (e.g., rows if Z is slow scan)
        'paramFN_path': zarr_file_path_str, # Keep track of source
        'HydraActive': False, 'nDetectors': 1, 'StartDetNr': 1, 'EndDetNr': 1,
    }
    store = None
    ap_base = 'analysis/process/analysis_parameters'
    scan_base = 'measurement/process/scan_parameters'

    try:
        store = ZipStore(zarr_file_path_str, mode='r')
        z_root = zarr.open_group(store=store, mode='r')
        print(f"DEBUG (extract_params): Zarr tree for '{Path(zarr_file_path_str).name}': \n{z_root.tree()}")

        # Infer dimensions from data array first
        try:
            data_array_for_shape = z_root[params['dataLoc']] # Default 'exchange/data'
            if data_array_for_shape.ndim == 3:
                # Zarr shape: (frames, Y_detector_dim, Z_detector_dim)
                # Y_detector_dim is Horizontal, Z_detector_dim is Vertical.
                params['NrPixelsY_detector'] = data_array_for_shape.shape[1]
                params['NrPixelsZ_detector'] = data_array_for_shape.shape[2]
                print(f"DEBUG (extract_params): Inferred Detector Y_dim(Horizontal)={params['NrPixelsY_detector']}, Z_dim(Vertical)={params['NrPixelsZ_detector']}")
            else:
                print(f"Warning (extract_params): Data array at {params['dataLoc']} is not 3D.")
        except KeyError:
            print(f"Warning (extract_params): Data array not found at '{params['dataLoc']}' for dimension inference.")
        except Exception as e:
            print(f"Warning (extract_params): Error inferring dimensions: {e}")

        # Pixel Size
        px_val = z_root.get(f'{ap_base}/PixelSize')
        if px_val is not None: params['px'] = float(px_val[...]) # Assuming µm
        print(f"DEBUG (extract_params): px: {params['px']}")

        # LSD
        lsd_val = z_root.get(f'{ap_base}/Lsd')
        if lsd_val is not None: params['DetParams'][0]['lsd'] = float(lsd_val[...]) # Assuming µm
        print(f"DEBUG (extract_params): LSD: {params['DetParams'][0]['lsd']}")

        # Beam Center
        ycen_val_detector = z_root.get(f'{ap_base}/YCen') # Horizontal coord on detector
        zcen_val_detector = z_root.get(f'{ap_base}/ZCen') # Vertical coord on detector

        if ycen_val_detector is not None and zcen_val_detector is not None:
            # bc_plot_H_cols: YCen (detector horizontal) maps to plot columns.
            # bc_plot_V_rows: ZCen (detector vertical) maps to plot rows.
            # FIXME / VERIFY: If ZCen is from TOP of detector, needs: params['NrPixelsZ_detector'] - ZCen
            # Assuming ZCen is 0-indexed from bottom, compatible with origin='lower' for plotting.
            bc_plot_H_cols = float(ycen_val_detector[...])
            bc_plot_V_rows = float(zcen_val_detector[...])
            params['DetParams'][0]['bc'] = [bc_plot_H_cols, bc_plot_V_rows]
            print(f"DEBUG (extract_params): Set BC (plot_cols_H, plot_rows_V) from YCen, ZCen: {params['DetParams'][0]['bc']}")
        elif params['NrPixelsY_detector'] is not None and params['NrPixelsZ_detector'] is not None:
            params['DetParams'][0]['bc'] = [params['NrPixelsY_detector'] / 2.0, params['NrPixelsZ_detector'] / 2.0]
            print(f"DEBUG (extract_params): YCen/ZCen not found. Set default BC to image center: {params['DetParams'][0]['bc']}")
        else:
            print(f"Warning (extract_params): Beam center (YCen/ZCen) not found and image dimensions unknown.")


        # Wavelength
        wl_val = z_root.get(f'{ap_base}/Wavelength')
        if wl_val is not None: params['wl'] = float(wl_val[...]) # Assuming Ångströms
        print(f"DEBUG (extract_params): Wavelength: {params['wl']}")

        # Lattice Constant
        lc_val = z_root.get(f'{ap_base}/LatticeParameter')
        if lc_val is not None: params['LatticeConstant'] = np.array(lc_val[...]).astype(float)
        print(f"DEBUG (extract_params): LatticeConstant: {params['LatticeConstant']}")

        # Space Group
        sg_val = z_root.get(f'{ap_base}/SpaceGroup')
        if sg_val is not None: params['sg'] = int(sg_val[...])
        print(f"DEBUG (extract_params): SpaceGroup: {params['sg']}")

        # Omega
        omega_start_val = z_root.get(f'{scan_base}/start')
        omega_step_val = z_root.get(f'{scan_base}/step')
        if omega_start_val is not None and omega_step_val is not None:
            params['omegaStart'] = float(omega_start_val[...])
            params['omegaStep'] = float(omega_step_val[...])
        print(f"DEBUG (extract_params): omegaStart: {params['omegaStart']}, omegaStep: {params['omegaStep']}")

        for p_key in ['tx', 'ty', 'tz', 'p0', 'p1', 'p2']: # Add other DetParams if needed
            val = z_root.get(f'{ap_base}/{p_key}')
            if val is not None: params['DetParams'][0][p_key] = float(val[...])
        
        params['maxRad_for_hkl_gen'] = float(z_root.get(f'{ap_base}/MaxRingRad', [2e6])[...]) # For potential HKL re-gen

    except Exception as e:
        print(f"ERROR (extract_params): Failed to open or process Zarr for params: {traceback.format_exc()}")
    finally:
        if store:
            try: store.close()
            except: pass
            
    # Fallbacks for essential plotting params if still None
    if params['DetParams'][0]['lsd'] is None: params['DetParams'][0]['lsd'] = 1e6
    if params['px'] is None: params['px'] = 75.0
    if params['wl'] is None: params['wl'] = 0.1729 # Default if not found
    if params['DetParams'][0]['bc'] == [None, None] and params['NrPixelsY_detector'] and params['NrPixelsZ_detector']:
         params['DetParams'][0]['bc'] = [params['NrPixelsY_detector'] / 2.0, params['NrPixelsZ_detector'] / 2.0]

    return params


def load_image_frame(params, zarr_file_path_str, frame_nr,
                     hflip=False, vflip=False, transpose_display=False): # transpose_display for final view
    """Loads a single image frame from the specified Zarr file."""
    if not zarr_file_path_str or not Path(zarr_file_path_str).exists():
         print(f"Error: Zarr file not found: {zarr_file_path_str}")
         return None, {}

    file_info = {'source_type': 'zarr', 'omega': None, 'raw_frame_nr': frame_nr, 'total_frames': 1,
                 'dark_corrected_in_zarr': False} # If darks were pre-subtracted in Zarr
    data = None
    store = None

    try:
        store = ZipStore(zarr_file_path_str, mode='r')
        z_root = zarr.open_group(store=store, mode='r')

        data_loc = params.get('dataLoc', 'exchange/data')
        if data_loc not in z_root:
            print(f"Error: Data path '{data_loc}' not found in Zarr: {zarr_file_path_str}")
            return None, file_info

        data_array = z_root[data_loc]
        if data_array.ndim != 3:
             raise ValueError(f"Expected 3D data in Zarr group '{data_loc}', found {data_array.ndim} dims.")

        n_frames_total, dim_Y_detector, dim_Z_detector = data_array.shape
        file_info['total_frames'] = n_frames_total

        if not (0 <= frame_nr < n_frames_total):
            print(f"Error: Frame number {frame_nr} out of bounds (0-{n_frames_total-1})")
            return None, file_info

        print(f"Reading Zarr frame {frame_nr} from {Path(zarr_file_path_str).name} [{data_loc}]")
        raw_frame_data = data_array[frame_nr, :, :] # Slice is (Y_detector_dim, Z_detector_dim)
        
        # Data orientation: Zarr (Y_det, Z_det) -> Plotly imshow (rows, cols) = (Z_det, Y_det)
        # So, a transpose is needed.
        data_for_plot = raw_frame_data.T # Now (Z_detector_dim, Y_detector_dim) = (plot_rows, plot_cols)
        print(f"Zarr data slice shape (Y_det, Z_det): {raw_frame_data.shape}, after T for plot (plot_rows, plot_cols): {data_for_plot.shape}")

        # Omega calculation
        if params.get('omegaStart') is not None and params.get('omegaStep') is not None:
            file_info['omega'] = params['omegaStart'] + frame_nr * params['omegaStep']
        
        # Heuristic for pre-corrected darks
        dark_loc_check = params.get('darkLoc', 'exchange/dark')
        if dark_loc_check not in z_root or np.all(z_root.get(dark_loc_check, default=np.array([0]))[:] == 0):
            file_info['dark_corrected_in_zarr'] = True

        data = data_for_plot.astype(float) # Work with float

        # Apply display transformations
        if transpose_display: # This transpose is AFTER the initial orientation transpose
            print("Applying display transpose.")
            data = np.transpose(data)
        if hflip:
            print("Applying horizontal flip.")
            data = np.flip(data, 1) # Flips plot columns
        if vflip:
            print("Applying vertical flip.")
            data = np.flip(data, 0) # Flips plot rows

    except Exception as e:
        print(f"Error reading Zarr file {zarr_file_path_str}, frame {frame_nr}: {traceback.format_exc()}")
        return None, file_info
    finally:
        if store:
            try: store.close()
            except: pass
    return data, file_info


def get_max_projection(params, zarr_file_path_str, num_frames_max, start_frame_nr,
                       hflip=False, vflip=False, transpose_display=False):
    """Gets max projection from the Zarr file."""
    if not zarr_file_path_str or not Path(zarr_file_path_str).exists():
         print(f"Error: Zarr file not found for max projection: {zarr_file_path_str}")
         return None, {}

    file_info = {'source_type': 'zarr_max', 'omega': None, 'dark_corrected_in_zarr': False, 'total_frames': 1}
    data_max_for_plot = None
    store = None

    try:
        store = ZipStore(zarr_file_path_str, mode='r')
        z_root = zarr.open_group(store=store, mode='r')
        data_loc = params.get('dataLoc', 'exchange/data')
        if data_loc not in z_root:
             raise KeyError(f"Data path '{data_loc}' not found in Zarr for max projection.")

        data_array = z_root[data_loc]
        if data_array.ndim != 3:
             raise ValueError(f"Expected 3D data in Zarr group '{data_loc}'.")

        n_frames_total_zarr, dim_Y_detector, dim_Z_detector = data_array.shape
        file_info['total_frames'] = n_frames_total_zarr

        actual_start = max(0, start_frame_nr)
        actual_end = min(n_frames_total_zarr, actual_start + num_frames_max)
        actual_num = actual_end - actual_start

        if actual_num <= 0:
             print(f"Error: No valid frames for Zarr max proj in range.")
             return None, {}
        print(f"Calculating Zarr max projection: {actual_num} frames from {actual_start}")

        zarr_slice = data_array[actual_start:actual_end, :, :] # (actual_num, Y_det, Z_det)
        max_along_frames = np.max(zarr_slice, axis=0) # Result is (Y_det, Z_det)
        
        data_max_for_plot = max_along_frames.T # Transpose to (Z_det, Y_det) = (plot_rows, plot_cols)
        print(f"Zarr max proj raw shape (Y_det,Z_det): {max_along_frames.shape}, after T for plot: {data_max_for_plot.shape}")

        if params.get('omegaStart') is not None and params.get('omegaStep') is not None:
            file_info['omega'] = params['omegaStart'] + actual_start * params['omegaStep'] # Omega of start frame
        
        dark_loc_check = params.get('darkLoc', 'exchange/dark')
        if dark_loc_check not in z_root or np.all(z_root.get(dark_loc_check, default=np.array([0]))[:] == 0):
            file_info['dark_corrected_in_zarr'] = True
            
        data_max_for_plot = data_max_for_plot.astype(float)
        if transpose_display: data_max_for_plot = np.transpose(data_max_for_plot)
        if hflip: data_max_for_plot = np.flip(data_max_for_plot, 1)
        if vflip: data_max_for_plot = np.flip(data_max_for_plot, 0)

    except Exception as e:
        print(f"Error calculating Zarr max projection: {traceback.format_exc()}")
        return None, {}
    finally:
        if store:
            try: store.close()
            except: pass
    return data_max_for_plot, file_info


def load_dark_frame_from_zarr(params_from_zarr, zarr_file_path_str,
                              hflip=False, vflip=False, transpose_display=False):
    """Loads dark frame from the *same* Zarr file used for data."""
    dark_data_for_plot = None
    store = None
    try:
        dark_loc = params_from_zarr.get('darkLoc', 'exchange/dark')
        if not dark_loc:
            print("DEBUG (load_dark_from_zarr): No darkLoc specified in params.")
            return None

        store = ZipStore(zarr_file_path_str, mode='r')
        z_root = zarr.open_group(store=store, mode='r')

        if dark_loc in z_root:
            dark_array = z_root[dark_loc] # Shape e.g. (N_dark_frames, Y_det, Z_det) or (Y_det, Z_det)
            print(f"DEBUG (load_dark_from_zarr): Found dark array at {dark_loc}, shape {dark_array.shape}")
            
            raw_dark_slice = None
            if dark_array.ndim == 2: # (Y_det, Z_det)
                raw_dark_slice = dark_array[:]
            elif dark_array.ndim == 3: # (N_dark_frames, Y_det, Z_det) -> average
                raw_dark_slice = np.mean(dark_array[:], axis=0)
            else:
                print(f"DEBUG (load_dark_from_zarr): Dark array at {dark_loc} has unexpected ndim {dark_array.ndim}")
                return None
            
            dark_data_for_plot = raw_dark_slice.T # Transpose to (Z_det, Y_det) for plotting
            dark_data_for_plot = dark_data_for_plot.astype(float)

            if transpose_display: dark_data_for_plot = np.transpose(dark_data_for_plot)
            if hflip: dark_data_for_plot = np.flip(dark_data_for_plot, 1)
            if vflip: dark_data_for_plot = np.flip(dark_data_for_plot, 0)
            print(f"DEBUG (load_dark_from_zarr): Loaded dark, final shape for plot: {dark_data_for_plot.shape}")
        else:
            print(f"DEBUG (load_dark_from_zarr): Dark location '{dark_loc}' not found in Zarr.")

    except Exception as e:
        print(f"ERROR (load_dark_from_zarr): Failed: {traceback.format_exc()}")
    finally:
        if store:
            try: store.close()
            except: pass
    return dark_data_for_plot