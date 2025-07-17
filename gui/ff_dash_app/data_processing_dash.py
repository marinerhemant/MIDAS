# data_processing_dash.py
import numpy as np
import os
import time
import zarr
from zarr.storage import ZipStore
import h5py
from numba import jit, prange
from pathlib import Path
import traceback
import io

import utils_dash as utils

# --- Numba Max Projection (No changes) ---
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

# --- Parameter Extraction ---

def _get_default_params():
    """Returns a default parameter dictionary."""
    return {
        'DetParams': [{'lsd': None, 'bc': [None, None], 'tx': 0.0, 'ty': 0.0, 'tz': 0.0,
                       'p0': 0.0, 'p1': 0.0, 'p2': 0.0}],
        'px': None, 'wl': None, 'sg': None, 'LatticeConstant': None,
        'omegaStart': None, 'omegaStep': None,
        'dataLoc': 'exchange/data', 'darkLoc': 'exchange/dark',
        'NrPixelsY_detector': None, 'NrPixelsZ_detector': None,
        'paramFN_path': None, 'HydraActive': False, 'nDetectors': 1,
        'StartDetNr': 1, 'EndDetNr': 1,
    }

def parse_params_from_file(param_file_path):
    """Parses parameters from a ps.txt-style file path."""
    params = _get_default_params()
    det_p = params['DetParams'][0]
    
    with open(param_file_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        if line.strip().startswith('#') or not line.strip():
            continue
        
        val_lsd = utils.parse_param_line(line, "Lsd", float)
        if val_lsd is not None: det_p['lsd'] = val_lsd

        val_bc = utils.parse_param_line(line, "BC", float, num_values=2)
        if val_bc is not None: det_p['bc'] = val_bc

        val_px = utils.parse_param_line(line, "px", float)
        if val_px is not None: params['px'] = val_px

        val_wl = utils.parse_param_line(line, "Wavelength", float)
        if val_wl is not None: params['wl'] = val_wl
        
        val_sg = utils.parse_param_line(line, "SpaceGroup", int)
        if val_sg is not None: params['sg'] = val_sg

        val_lc = utils.parse_param_line(line, "LatticeConstant", float, num_values=6)
        if val_lc is not None: params['LatticeConstant'] = val_lc

        val_os = utils.parse_param_line(line, "OmegaStart", float)
        if val_os is not None: params['omegaStart'] = val_os

        val_ostep = utils.parse_param_line(line, "OmegaStep", float)
        if val_ostep is not None: params['omegaStep'] = val_ostep
        
        for p_key in ['tx', 'ty', 'tz', 'p0', 'p1', 'p2']:
             val_p = utils.parse_param_line(line, p_key, float)
             if val_p is not None: det_p[p_key] = val_p

    return params

def extract_parameters(data_file_path, param_file_path=None):
    """
    Dispatcher function to extract parameters from Zarr or HDF5+txt file.
    """
    file_path_obj = Path(data_file_path)
    if not file_path_obj.exists():
        raise FileNotFoundError(f"Data file not found: {data_file_path}")

    if file_path_obj.suffix == '.zip':
        print(f"Extracting parameters from Zarr file: {file_path_obj.name}")
        return _extract_params_from_zarr(data_file_path)
    
    elif file_path_obj.suffix == '.h5':
        if not param_file_path:
            raise ValueError("HDF5 file requires a parameter file path.")
        print(f"Extracting parameters from HDF5 file '{file_path_obj.name}' and parameter file.")
        
        params = parse_params_from_file(param_file_path)
        params['paramFN_path'] = data_file_path
        
        # Inspect H5 file to get dimensions
        with h5py.File(data_file_path, 'r') as hf:
            if params['dataLoc'] in hf:
                dset_shape = hf[params['dataLoc']].shape
                if len(dset_shape) == 3:
                    params['NrPixelsZ_detector'] = dset_shape[1]
                    params['NrPixelsY_detector'] = dset_shape[2]
                    print(f"Inferred HDF5 Detector dims: Y(cols)={params['NrPixelsY_detector']}, Z(rows)={params['NrPixelsZ_detector']}")
                    if params['DetParams'][0]['bc'] == [None, None]:
                         params['DetParams'][0]['bc'] = [dset_shape[2] / 2.0, dset_shape[1] / 2.0]
                else:
                    print(f"Warning: Data at {params['dataLoc']} is not 3D.")
            else:
                print(f"Warning: Data not found at '{params['dataLoc']}' in HDF5 file.")
        return params
    else:
        raise ValueError(f"Unsupported file type: {file_path_obj.suffix}")


def _extract_params_from_zarr(zarr_file_path_str):
    """Extracts parameters from a Zarr file (internal helper)."""
    params = _get_default_params()
    params['paramFN_path'] = zarr_file_path_str
    
    store = None
    ap_base = 'analysis/process/analysis_parameters'
    scan_base = 'measurement/process/scan_parameters'

    try:
        store = ZipStore(zarr_file_path_str, mode='r')
        z_root = zarr.open_group(store=store, mode='r')

        try:
            data_array_for_shape = z_root[params['dataLoc']]
            if data_array_for_shape.ndim == 3:
                params['NrPixelsY_detector'] = data_array_for_shape.shape[1]
                params['NrPixelsZ_detector'] = data_array_for_shape.shape[2]
        except KeyError:
            print(f"Warning: Data array not found at '{params['dataLoc']}'.")
        
        params['px'] = float(z_root.get(f'{ap_base}/PixelSize', [75.0])[...])
        params['DetParams'][0]['lsd'] = float(z_root.get(f'{ap_base}/Lsd', [1e6])[...])
        params['wl'] = float(z_root.get(f'{ap_base}/Wavelength', [0.1729])[...])
        
        ycen_val = z_root.get(f'{ap_base}/YCen')
        zcen_val = z_root.get(f'{ap_base}/ZCen')
        if ycen_val is not None and zcen_val is not None:
            params['DetParams'][0]['bc'] = [float(ycen_val[...]), float(zcen_val[...])]
        elif params['NrPixelsY_detector'] and params['NrPixelsZ_detector']:
            params['DetParams'][0]['bc'] = [params['NrPixelsY_detector'] / 2.0, params['NrPixelsZ_detector'] / 2.0]

        # Add other parameters as needed
        val = z_root.get(f'{ap_base}/SpaceGroup'); params['sg'] = int(val[...]) if val is not None else None
        val = z_root.get(f'{ap_base}/LatticeParameter'); params['LatticeConstant'] = np.array(val[...]).astype(float) if val is not None else None
        val = z_root.get(f'{scan_base}/start'); params['omegaStart'] = float(val[...]) if val is not None else None
        val = z_root.get(f'{scan_base}/step'); params['omegaStep'] = float(val[...]) if val is not None else None

    except Exception as e:
        print(f"ERROR (_extract_params_from_zarr): Failed to process Zarr: {traceback.format_exc()}")
    finally:
        if store:
            try: store.close()
            except: pass
    return params


def load_image_frame(params, data_file_path_str, frame_nr,
                     hflip=False, vflip=False, transpose_display=False):
    """Loads a single image frame from Zarr or HDF5 file."""
    if not data_file_path_str or not Path(data_file_path_str).exists():
         return None, {}

    file_info = {'source_type': 'data_file', 'omega': None, 'raw_frame_nr': frame_nr, 'total_frames': 1,
                 'dark_corrected_in_zarr': False}
    data_for_plot = None
    file_path_obj = Path(data_file_path_str)

    try:
        if file_path_obj.suffix == '.h5':
            with h5py.File(data_file_path_str, 'r') as hf:
                dset = hf[params['dataLoc']]
                n_frames_total = dset.shape[0]
                if not (0 <= frame_nr < n_frames_total):
                    print(f"Error: Frame {frame_nr} out of bounds (0-{n_frames_total-1})")
                    return None, file_info
                file_info['total_frames'] = n_frames_total
                data_for_plot = dset[frame_nr, :, :]
                
        elif file_path_obj.suffix == '.zip':
            with ZipStore(data_file_path_str, mode='r') as store:
                z_root = zarr.open_group(store=store, mode='r')
                data_array = z_root[params['dataLoc']]
                n_frames_total = data_array.shape[0]
                if not (0 <= frame_nr < n_frames_total):
                    return None, file_info
                file_info['total_frames'] = n_frames_total
                raw_frame_data = data_array[frame_nr, :, :]
                data_for_plot = raw_frame_data.T
        else:
            raise ValueError(f"Unsupported file type for loading: {file_path_obj.suffix}")

        if params.get('omegaStart') is not None and params.get('omegaStep') is not None:
            file_info['omega'] = params['omegaStart'] + frame_nr * params['omegaStep']
        
        data = data_for_plot.astype(float)
        if transpose_display: data = np.transpose(data)
        if hflip: data = np.flip(data, 1)
        if vflip: data = np.flip(data, 0)

        return data, file_info

    except Exception as e:
        print(f"Error reading frame {frame_nr} from {file_path_obj.name}: {traceback.format_exc()}")
        return None, {}


def get_max_projection(params, data_file_path_str, num_frames_max, start_frame_nr,
                       hflip=False, vflip=False, transpose_display=False):
    """Gets max projection from Zarr or HDF5 file."""
    if not data_file_path_str or not Path(data_file_path_str).exists():
         return None, {}

    file_info = {'source_type': 'max_proj', 'omega': None, 'total_frames': 1}
    data_max_for_plot = None
    file_path_obj = Path(data_file_path_str)

    try:
        actual_start, actual_num = 0, 0
        
        if file_path_obj.suffix == '.h5':
             with h5py.File(data_file_path_str, 'r') as hf:
                dset = hf[params['dataLoc']]
                n_frames_total, actual_start, actual_end = dset.shape[0], max(0, start_frame_nr), 0
                actual_end = min(n_frames_total, actual_start + num_frames_max)
                actual_num = actual_end - actual_start
                if actual_num <= 0: return None, {}
                file_info['total_frames'] = n_frames_total
                h5_slice = dset[actual_start:actual_end, :, :]
                data_max_for_plot = np.max(h5_slice, axis=0)

        elif file_path_obj.suffix == '.zip':
             with ZipStore(data_file_path_str, mode='r') as store:
                z_root = zarr.open_group(store=store, mode='r')
                data_array = z_root[params['dataLoc']]
                n_frames_total, actual_start, actual_end = data_array.shape[0], max(0, start_frame_nr), 0
                actual_end = min(n_frames_total, actual_start + num_frames_max)
                actual_num = actual_end - actual_start
                if actual_num <= 0: return None, {}
                file_info['total_frames'] = n_frames_total
                zarr_slice = data_array[actual_start:actual_end, :, :]
                max_along_frames = np.max(zarr_slice, axis=0)
                data_max_for_plot = max_along_frames.T
        
        else:
             raise ValueError(f"Unsupported file type: {file_path_obj.suffix}")

        print(f"Calculated max projection: {actual_num} frames from {actual_start}")
        if params.get('omegaStart') is not None and params.get('omegaStep') is not None:
            file_info['omega'] = params['omegaStart'] + actual_start * params['omegaStep']
        
        data = data_max_for_plot.astype(float)
        if transpose_display: data = np.transpose(data)
        if hflip: data = np.flip(data, 1)
        if vflip: data = np.flip(data, 0)
        return data, file_info

    except Exception as e:
        print(f"Error calculating max projection: {traceback.format_exc()}")
        return None, {}


def load_dark_frame(params, data_file_path_str,
                     hflip=False, vflip=False, transpose_display=False):
    """Loads dark frame from the specified Zarr or HDF5 file."""
    dark_data_for_plot = None
    file_path_obj = Path(data_file_path_str)
    dark_loc = params.get('darkLoc', 'exchange/dark')

    try:
        raw_dark_slice = None
        if file_path_obj.suffix == '.h5':
             with h5py.File(data_file_path_str, 'r') as hf:
                 if dark_loc in hf:
                     dark_array = hf[dark_loc]
                     if dark_array.ndim == 2:
                         raw_dark_slice = dark_array[:]
                     elif dark_array.ndim == 3:
                         if dark_array.shape[0] > 1:
                            raw_dark_slice = np.mean(dark_array[1:], axis=0)
                         else:
                            raw_dark_slice = dark_array[0, :, :]
                     if raw_dark_slice is not None:
                         dark_data_for_plot = raw_dark_slice

        elif file_path_obj.suffix == '.zip':
            with ZipStore(data_file_path_str, mode='r') as store:
                z_root = zarr.open_group(store=store, mode='r')
                if dark_loc in z_root:
                    dark_array = z_root[dark_loc]
                    if dark_array.ndim == 2:
                        raw_dark_slice = dark_array[:]
                    elif dark_array.ndim == 3:
                        raw_dark_slice = np.mean(dark_array[:], axis=0)
                    if raw_dark_slice is not None:
                        dark_data_for_plot = raw_dark_slice.T
        
        if dark_data_for_plot is None:
            return None

        data = dark_data_for_plot.astype(float)
        if transpose_display: data = np.transpose(data)
        if hflip: data = np.flip(data, 1)
        if vflip: data = np.flip(data, 0)
        return data

    except Exception as e:
        print(f"ERROR (load_dark_frame): Failed: {traceback.format_exc()}")
        return None