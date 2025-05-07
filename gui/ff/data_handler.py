# data_handler.py
import numpy as np
import os
import time
import zarr
from zarr.storage import ZipStore
from numcodecs import Blosc # Make sure blosc is available via numcodecs
import h5py
try:
    import hdf5plugin
except ImportError:
    print("hdf5plugin not found, support for some HDF5 compressions might be limited.")
from numba import jit, prange
from pathlib import Path
from PIL import Image
import traceback
import subprocess

# Assuming utils.py is in the same directory or accessible in PYTHONPATH
from utils import parse_param_line, try_parse_float, try_parse_int, get_transform_matrix


# --- Numba Max Projection ---
@jit(nopython=True, parallel=True)
def calculate_max_projection_numba_uint16(filename, header_size, bytes_per_pixel,
                                         rows, cols, num_frames, start_frame):
    """Calculates max projection using Numba for uint16 data."""
    if bytes_per_pixel != 2:
        return np.zeros((rows, cols), dtype=np.uint16) # Return empty/zero

    frame_size_bytes = rows * cols * bytes_per_pixel
    total_offset = header_size + start_frame * frame_size_bytes
    max_image = np.zeros((rows, cols), dtype=np.uint16)
    actual_frames_read = 0

    try:
        f_ptr = open(filename, 'rb')
        f_ptr.seek(total_offset)
        all_data_bytes = f_ptr.read(num_frames * frame_size_bytes)
        f_ptr.close() # Ensure file is closed

        if len(all_data_bytes) < frame_size_bytes: # Check if at least one frame read
             return max_image # No frames read

        actual_frames_read = len(all_data_bytes) // frame_size_bytes
        if actual_frames_read == 0:
             return max_image

        valid_bytes = actual_frames_read * frame_size_bytes
        all_data = np.frombuffer(all_data_bytes[:valid_bytes], dtype=np.uint16).reshape((actual_frames_read, rows, cols))

        for r in prange(rows):
            for c in prange(cols):
                max_val = np.uint16(0)
                for frame_idx in range(actual_frames_read):
                    val = all_data[frame_idx, r, c]
                    if val > max_val:
                        max_val = val
                max_image[r, c] = max_val

    except Exception as e:
        return np.zeros((rows, cols), dtype=np.uint16)

    return max_image

@jit(nopython=True, parallel=True)
def calculate_max_projection_numba_int32(filename, header_size, bytes_per_pixel,
                                         rows, cols, num_frames, start_frame):
    """Calculates max projection using Numba for int32 data."""
    if bytes_per_pixel != 4:
        return np.zeros((rows, cols), dtype=np.int32)

    frame_size_bytes = rows * cols * bytes_per_pixel
    total_offset = header_size + start_frame * frame_size_bytes
    max_image = np.zeros((rows, cols), dtype=np.int32)
    actual_frames_read = 0

    try:
        f_ptr = open(filename, 'rb')
        f_ptr.seek(total_offset)
        all_data_bytes = f_ptr.read(num_frames * frame_size_bytes)
        f_ptr.close()

        if len(all_data_bytes) < frame_size_bytes:
            return max_image

        actual_frames_read = len(all_data_bytes) // frame_size_bytes
        if actual_frames_read == 0:
            return max_image

        valid_bytes = actual_frames_read * frame_size_bytes
        all_data = np.frombuffer(all_data_bytes[:valid_bytes], dtype=np.int32).reshape((actual_frames_read, rows, cols))

        for r in prange(rows):
            for c in prange(cols):
                max_val = np.int32(-2147483647 - 1) # Min int32
                for frame_idx in range(actual_frames_read):
                    val = all_data[frame_idx, r, c]
                    if val > max_val:
                        max_val = val
                max_image[r, c] = max_val
    except Exception as e:
        return np.zeros((rows, cols), dtype=np.int32)

    return max_image


# --- Data Loading ---
def load_image_frame(params, file_path, frame_nr,
                     hflip=False, vflip=False, transpose=False):
    """Loads a single image frame from GE or Zarr file."""
    if not file_path or not Path(file_path).exists():
         print(f"Error: File not found or path is invalid: {file_path}")
         return None, {} # Return None data and empty info

    file_info = {'source_type': None, 'omega': None, 'raw_frame_nr': frame_nr, 'total_frames': 1, 'dark_corrected': False}
    data = None # Initialize data to None

    if file_path.lower().endswith(".zip"): # Zarr ZIP Store
        file_info['source_type'] = 'zarr'
        store = None
        try:
            store = ZipStore(file_path, mode='r')
            z_root = zarr.open_group(store=store, mode='r')

            data_path = params.get('dataLoc', 'exchange/data')
            if data_path not in z_root:
                if 'exchange/data' not in z_root:
                    print(f"Error: Data ({data_path} or exchange/data) not found in Zarr file: {file_path}")
                    if store: store.close()
                    return None, file_info
                else:
                    data_path = 'exchange/data'

            data_array = z_root[data_path]
            if data_array.ndim != 3:
                 raise ValueError(f"Expected 3D data in Zarr group '{data_path}', found {data_array.ndim} dimensions.")

            n_frames_total, nz_zarr, ny_zarr = data_array.shape
            file_info['total_frames'] = n_frames_total

            if frame_nr < 0 or frame_nr >= n_frames_total:
                print(f"Error: Frame number {frame_nr} out of bounds (0-{n_frames_total-1})")
                if store: store.close()
                return None, file_info

            print(f"Reading Zarr frame {frame_nr} from {file_path} [{data_path}]")
            start_time = time.time()
            data = data_array[frame_nr, :, :].T # Transpose to get (y, z) or (row, col)
            print(f"Zarr frame read time: {time.time() - start_time:.3f}s")

            # Try to get Omega
            try:
                scan_params_group = 'measurement/process/scan_parameters'
                start_omega = z_root[f'{scan_params_group}/start'][0]
                step_omega = z_root[f'{scan_params_group}/step'][0]
                file_info['omega'] = start_omega + frame_nr * step_omega
            except KeyError:
                 print("Omega start/step info not found in standard Zarr location.")
                 try:
                     omega_array_path = 'measurement/instrument/detector/omega' # Example path
                     if omega_array_path in z_root:
                         omega_array = z_root[omega_array_path]
                         if omega_array.ndim == 1 and omega_array.shape[0] == n_frames_total:
                             file_info['omega'] = omega_array[frame_nr]
                             print("Found omega value in array.")
                         else:
                             print(f"Omega array found but shape mismatch {omega_array.shape} vs expected ({n_frames_total},)")
                     else:
                          print("Omega array not found.")
                 except Exception as e_alt_omega:
                      print(f"Could not read omega array: {e_alt_omega}")
            except Exception as e_omega:
                 print(f"Error reading omega from Zarr: {e_omega}")

            # Dark correction heuristic
            dark_path = params.get('darkLoc', 'exchange/dark')
            if dark_path in z_root and np.any(z_root[dark_path][:] != 0):
                 file_info['dark_corrected'] = False
                 print("Non-zero dark found in Zarr, assuming dark correction might be needed.")
            else:
                 file_info['dark_corrected'] = True
                 print("No significant dark found in Zarr, assuming data is dark corrected.")

        except Exception as e:
            print(f"Error reading Zarr file {file_path}: {traceback.format_exc()}")
            return None, file_info # Return immediately on error
        finally:
            if store is not None and hasattr(store, 'mode') and store.mode != 'closed':
                 store.close() # Ensure store is closed

    else: # Assume GE or similar raw binary file
        file_info['source_type'] = 'raw_binary'
        file_info['dark_corrected'] = False # Needs correction if dark is available

        header = try_parse_int(params.get('Header', 8192))
        bytes_pp = try_parse_int(params.get('BytesPerPixel', 2))
        ny = try_parse_int(params.get('NrPixelsY', 2048)) # Rows
        nz = try_parse_int(params.get('NrPixelsZ', 2048)) # Columns
        n_frames_file = try_parse_int(params.get('nFramesPerFile', 0))

        if ny <= 0 or nz <= 0 or bytes_pp <= 0:
             print(f"Error: Invalid dimensions or bytes per pixel (Y={ny}, Z={nz}, BPP={bytes_pp})")
             return None, file_info

        if n_frames_file <= 0:
            try:
                file_size = os.path.getsize(file_path)
                frame_bytes = ny * nz * bytes_pp
                if frame_bytes > 0 and file_size >= header:
                    n_frames_file = (file_size - header) // frame_bytes
                else:
                    n_frames_file = 0
                params['nFramesPerFile'] = n_frames_file # Store calculated value for future use
                print(f"Calculated nFramesPerFile for {Path(file_path).name}: {n_frames_file}")
            except FileNotFoundError:
                 print(f"Error: File not found while calculating frames: {file_path}")
                 return None, file_info
            except Exception as e:
                 print(f"Error getting file size or calculating frames for {Path(file_path).name}: {e}")
                 return None, file_info

        file_info['total_frames'] = n_frames_file # Total frames in this specific file

        if n_frames_file == 0:
             print(f"Error: Calculated 0 frames for file {Path(file_path).name}.")
             return None, file_info
        if frame_nr < 0 or frame_nr >= n_frames_file:
            print(f"Error: Frame number {frame_nr} out of bounds for file {Path(file_path).name} (0-{n_frames_file-1})")
            return None, file_info

        bytes_to_skip = header + frame_nr * (ny * nz * bytes_pp)
        count = ny * nz # Total pixels per frame

        print(f"Reading GE frame {frame_nr} from {Path(file_path).name}")
        start_time = time.time()
        try:
            with open(file_path, 'rb') as f:
                f.seek(bytes_to_skip)
                dtype = np.uint16 if bytes_pp == 2 else np.int32 if bytes_pp == 4 else None
                if dtype is None:
                    print(f"Error: Unsupported BytesPerPixel: {bytes_pp}")
                    return None, file_info

                raw_data = np.fromfile(f, dtype=dtype, count=count)

            if raw_data.size < count:
                 print(f"Error: Could not read full frame {frame_nr} from {Path(file_path).name}. Expected {count} pixels, got {raw_data.size}.")
                 return None, file_info

            data = raw_data.reshape((ny, nz)) # Reshape to (rows, columns)
            print(f"Raw frame read time: {time.time() - start_time:.3f}s")

        except FileNotFoundError:
             print(f"Error: File not found during read: {file_path}")
             return None, file_info
        except Exception as e:
            print(f"Error reading raw file {file_path}, frame {frame_nr}: {traceback.format_exc()}")
            return None, file_info

    # --- Common Post-Processing ---
    if data is None: # Check if data loading failed
         return None, file_info

    data = data.astype(float) # Work with float
    if transpose:
        print("Applying transpose.")
        data = np.transpose(data)
    if hflip: # Horizontal flip (flips along axis 1 - columns)
        print("Applying horizontal flip.")
        data = np.flip(data, 1)
    if vflip: # Vertical flip (flips along axis 0 - rows)
        print("Applying vertical flip.")
        data = np.flip(data, 0)

    return data, file_info


def get_max_projection(params, file_path, num_frames_max, start_frame_nr,
                        hflip=False, vflip=False, transpose=False):
    """Gets max projection from either raw binary (using Numba) or Zarr (using numpy)."""

    if not file_path or not Path(file_path).exists():
         print(f"Error: File not found for max projection: {file_path}")
         return None, {}

    data_max = None
    file_info = {'source_type': None, 'omega': None, 'dark_corrected': False, 'total_frames': 1}
    start_time = time.time()

    if file_path.lower().endswith(".zip"): # Zarr ZIP Store
        print(f"Calculating max projection from Zarr: {num_frames_max} frames from {start_frame_nr} in {Path(file_path).name}")
        file_info['source_type'] = 'zarr_max'
        store = None
        try:
            store = ZipStore(file_path, mode='r')
            z_root = zarr.open_group(store=store, mode='r')
            data_path = params.get('dataLoc', 'exchange/data')
            if data_path not in z_root:
                 if 'exchange/data' not in z_root:
                     raise KeyError(f"Data ({data_path} or exchange/data) not found in Zarr file.")
                 else:
                     data_path = 'exchange/data'

            data_array = z_root[data_path]
            if data_array.ndim != 3:
                 raise ValueError(f"Expected 3D data in Zarr group '{data_path}', found {data_array.ndim} dimensions.")

            n_frames_total, nz_zarr, ny_zarr = data_array.shape
            file_info['total_frames'] = n_frames_total

            actual_start_frame = max(0, start_frame_nr)
            actual_end_frame = min(n_frames_total, actual_start_frame + num_frames_max)
            actual_num_frames = actual_end_frame - actual_start_frame

            if actual_num_frames <= 0:
                 print(f"Error: No valid frames in range [{start_frame_nr} - {start_frame_nr + num_frames_max -1}]")
                 if store: store.close()
                 return None, {}
            if actual_start_frame != start_frame_nr or actual_num_frames != num_frames_max:
                 print(f"Adjusted Zarr max proj range: Reading frames [{actual_start_frame} - {actual_end_frame-1}]")

            zarr_slice = data_array[actual_start_frame:actual_end_frame, :, :]
            data_max_zarr_raw = np.max(zarr_slice, axis=0) # Max along frame axis -> (nz, ny)
            data_max = data_max_zarr_raw.T # Transpose to (ny, nz) / (rows, cols)

            print(f"Zarr max projection calculation time: {time.time() - start_time:.3f}s")

        except Exception as e:
            print(f"Error calculating Zarr max projection: {traceback.format_exc()}")
            return None, {}
        finally:
            if store is not None and hasattr(store, 'mode') and store.mode != 'closed':
                 store.close()

    else: # Raw Binary File
        print(f"Calculating max projection from Raw Binary: {num_frames_max} frames from {start_frame_nr} in {Path(file_path).name}")
        file_info['source_type'] = 'raw_binary_max'
        header = try_parse_int(params.get('Header', 8192))
        bytes_pp = try_parse_int(params.get('BytesPerPixel', 2))
        ny = try_parse_int(params.get('NrPixelsY', 2048)) # Rows
        nz = try_parse_int(params.get('NrPixelsZ', 2048)) # Cols

        if ny <= 0 or nz <= 0 or bytes_pp not in [2, 4]:
             print("Error: Invalid dimensions or bytes per pixel for raw max projection.")
             return None, {}

        actual_frames_to_process = num_frames_max
        n_frames_in_file = 0
        try:
            file_size = os.path.getsize(file_path)
            frame_bytes = ny * nz * bytes_pp
            if frame_bytes > 0:
                 byte_offset = header + start_frame_nr * frame_bytes
                 if file_size >= header: n_frames_in_file = (file_size - header) // frame_bytes
                 if byte_offset < file_size:
                      available_bytes_after_start = file_size - byte_offset
                      available_frames = available_bytes_after_start // frame_bytes
                      if available_frames < num_frames_max:
                           print(f"Warning: Requested {num_frames_max} frames, but only {available_frames} available. Using {available_frames} frames.")
                           actual_frames_to_process = available_frames
                      elif available_frames == 0: print(f"Error: No full frames available after start frame {start_frame_nr}."); return None, {}
                 else: print(f"Error: Start frame {start_frame_nr} is beyond file size ({n_frames_in_file} frames)."); return None, {}
            else: print("Error: Frame size is zero for raw file."); return None, {}
        except Exception as e: print(f"Error checking file size for raw max projection: {e}"); return None, {}

        if actual_frames_to_process <= 0: print("Error: No frames to process for raw max projection."); return None, {}

        file_info['total_frames'] = n_frames_in_file

        data_max_raw = None
        if bytes_pp == 2: data_max_raw = calculate_max_projection_numba_uint16(file_path, header, bytes_pp, ny, nz, actual_frames_to_process, start_frame_nr)
        elif bytes_pp == 4: data_max_raw = calculate_max_projection_numba_int32(file_path, header, bytes_pp, ny, nz, actual_frames_to_process, start_frame_nr)

        print(f"Raw Binary max projection calculation time: {time.time() - start_time:.3f}s")

        if data_max_raw is None or data_max_raw.size == 0: print("Error: Max projection calculation failed (Numba returned empty)."); return None, {}
        if np.count_nonzero(data_max_raw) == 0: print("Warning: Raw max projection result is all zeros.")
        data_max = data_max_raw # Numba returns (rows, cols)

    # --- Common Post-Processing ---
    if data_max is None: return None, {}

    data_max = data_max.astype(float) # Convert to float
    if transpose: print("Applying transpose."); data_max = np.transpose(data_max)
    if hflip: print("Applying horizontal flip."); data_max = np.flip(data_max, 1) # axis 1 (cols)
    if vflip: print("Applying vertical flip."); data_max = np.flip(data_max, 0) # axis 0 (rows)

    return data_max, file_info


def load_dark_frame(params, dark_file_path,
                    hflip=False, vflip=False, transpose=False):
    """Loads the dark frame (assumes it's a single effective frame, frame 0)."""
    if not dark_file_path or not Path(dark_file_path).exists():
         print(f"Dark file path not specified or file not found: {dark_file_path}")
         return None

    print(f"Loading dark frame from: {Path(dark_file_path).name}")
    dark_data, _ = load_image_frame(params, dark_file_path, 0, hflip, vflip, transpose)
    if dark_data is None:
         print(f"Failed to load dark frame data from {dark_file_path}")
    return dark_data


def read_parameters(param_fn):
    """Reads parameters from the FF-HEDM style parameter file."""
    if not Path(param_fn).exists():
        raise FileNotFoundError(f"Parameter file not found: {param_fn}")

    params = {
        'paramFN': param_fn, 'DetParams': [], 'RingsToShow': [], 'hkls': [],
        'ringRads': [], 'ringNrs': [], 'LatticeConstant': np.zeros(6),
        'nDetectors': 0, 'Padding': 6, 'NrPixelsY': 2048, 'NrPixelsZ': 2048,
        'Header': 8192, 'BytesPerPixel': 2, 'px': 200.0, 'wl': 0.172979,
        'sg': 225, 'bigdetsize': 2048, 'nFramesPerFile': 0, 'nFilesPerLayer': 1,
        'StartDetNr': 1, 'EndDetNr': 1, 'omegaStart': 0.0, 'omegaStep': 0.0,
        'HydraActive': False, 'folder': None, 'fileStem': None, 'darkStem': None,
        'darkNum': None, 'maxRad': 2000000.0, '_temp_rings_info': []
    }
    det_params_list = []

    print(f"Reading parameters from: {param_fn}")
    try:
        with open(param_fn, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        raise IOError(f"Could not read parameter file {param_fn}: {e}")

    for line_num, line in enumerate(lines):
        line = line.strip()
        if not line or line.startswith('#'): continue
        parts = line.split()
        if not parts: continue
        keyword = parts[0]
        try: # Process each keyword
            if keyword == 'RawFolder': params['folder'] = parts[1]
            elif keyword == 'FileStem': params['fileStem'] = parts[1]
            # ... (rest of the parameter parsing as before, ENSURING NO SEMICOLONS) ...
            elif keyword == 'DarkStem': params['darkStem'] = parts[1]
            elif keyword == 'Padding': params['Padding'] = try_parse_int(parts[1], 6)
            elif keyword == 'Ext': params['fnext'] = parts[1]
            elif keyword == 'DarkNum': params['darkNum'] = try_parse_int(parts[1])
            elif keyword == 'FirstFileNumber' or keyword == 'StartFileNrFirstLayer': params['firstFileNumber'] = try_parse_int(parts[1], 1)
            elif keyword == 'NumDetectors': params['nDetectors'] = try_parse_int(parts[1], 0)
            elif keyword == 'StartDetNr': params['StartDetNr'] = try_parse_int(parts[1], 1)
            elif keyword == 'EndDetNr': params['EndDetNr'] = try_parse_int(parts[1], 1)
            elif keyword == 'Wedge': params['wedge'] = try_parse_float(parts[1])
            elif keyword == 'px': params['px'] = try_parse_float(parts[1], 200.0)
            elif keyword == 'Wavelength': params['wl'] = try_parse_float(parts[1], 0.172979)
            elif keyword == 'BigDetSize': params['bigdetsize'] = try_parse_int(parts[1], 2048)
            elif keyword == 'NrPixelsY': params['NrPixelsY'] = try_parse_int(parts[1], 2048)
            elif keyword == 'NrPixelsZ': params['NrPixelsZ'] = try_parse_int(parts[1], 2048)
            elif keyword == 'HeadSize': params['Header'] = try_parse_int(parts[1], 8192)
            elif keyword == 'BytesPerPixel': params['BytesPerPixel'] = try_parse_int(parts[1], 2)
            elif keyword == 'nFramesPerFile': params['nFramesPerFile'] = try_parse_int(parts[1], 0)
            elif keyword == 'OmegaStep': params['omegaStep'] = try_parse_float(parts[1])
            elif keyword == 'OmegaFirstFile' or keyword == 'OmegaStart': params['omegaStart'] = try_parse_float(parts[1])
            elif keyword == 'NrFilesPerSweep': params['nFilesPerLayer'] = try_parse_int(parts[1], 1)
            elif keyword == 'Lsd':
                if not det_params_list: det_params_list.append({})
                det_params_list[0]['lsd'] = try_parse_float(parts[1])
            elif keyword == 'BC':
                 if not det_params_list: det_params_list.append({})
                 det_params_list[0]['bc'] = [try_parse_float(parts[1]), try_parse_float(parts[2])]
            elif keyword == 'DetParams':
                if len(parts) < 11: raise ValueError("DetParams line needs at least 10 values.")
                params['HydraActive'] = True
                det_data = {'lsd': try_parse_float(parts[1]), 'bc': [try_parse_float(parts[2]), try_parse_float(parts[3])],
                            'tx': try_parse_float(parts[4]), 'ty': try_parse_float(parts[5]), 'tz': try_parse_float(parts[6]),
                            'p0': try_parse_float(parts[7]), 'p1': try_parse_float(parts[8]), 'p2': try_parse_float(parts[9]),
                            'RhoD': try_parse_float(parts[10])}
                det_params_list.append(det_data)
            elif keyword == 'LatticeParameter' or keyword == 'LatticeConstant':
                 if len(parts) < 7: raise ValueError("LatticeConstant/Parameter line needs 6 values.")
                 params['LatticeConstant'] = np.array([try_parse_float(p) for p in parts[1:7]])
            elif keyword == 'SpaceGroup': params['sg'] = try_parse_int(parts[1], 225)
            elif keyword == 'RingThresh':
                 if len(parts) < 3: raise ValueError("RingThresh line needs RingNr and Threshold.")
                 params['_temp_rings_info'].append({'nr': try_parse_int(parts[1]), 'thresh': try_parse_float(parts[2])})
            elif keyword == 'MaxRingRad': params['maxRad'] = try_parse_float(parts[1], 2000000.0)
            elif keyword == 'BorderToExclude': params['border'] = try_parse_int(parts[1])
            elif keyword == 'tolTilts': params['tolTilts'] = try_parse_float(parts[1])
            elif keyword == 'tolBC': params['tolBC'] = try_parse_float(parts[1])
            elif keyword == 'tolLsd': params['tolLsd'] = try_parse_float(parts[1])
            elif keyword == 'tolP': params['tolP'] = try_parse_float(parts[1])
            elif keyword == 'ImTransOpt': params['imTransOpt'] = try_parse_int(parts[1])

        except (IndexError, ValueError) as e:
            print(f"Warning: Could not parse line {line_num+1}: '{line}'. Error: {e}")
            continue

    # --- Finalize Parameter Setup ---
    if not params['HydraActive']:
        if not det_params_list: det_params_list.append({})
        def_bc = [params['NrPixelsZ']/2.0, params['NrPixelsY']/2.0] if params['NrPixelsY'] > 0 and params['NrPixelsZ'] > 0 else [1024.0, 1024.0]
        det_params_list[0].setdefault('lsd', params.get('lsd', 1000000.0))
        det_params_list[0].setdefault('bc', params.get('bc', def_bc))
        for key in ['tx', 'ty', 'tz', 'p0', 'p1', 'p2', 'RhoD']: det_params_list[0].setdefault(key, params.get(key, 0.0))
        params['nDetectors'] = 1; params['StartDetNr'] = 1; params['EndDetNr'] = 1
    else:
        if params['nDetectors'] == 0: params['nDetectors'] = len(det_params_list)
        if params['nDetectors'] != len(det_params_list):
            print(f"Warning: NumDetectors ({params['nDetectors']}) mismatch DetParams lines ({len(det_params_list)}). Using count from DetParams.")
            params['nDetectors'] = len(det_params_list)
        if params['nDetectors'] > 0: params['EndDetNr'] = params['StartDetNr'] + params['nDetectors'] - 1
        else: params['StartDetNr'] = 1; params['EndDetNr'] = 1

    params['DetParams'] = det_params_list

    # Expand ~ in paths
    for key in ['folder', 'darkStem', 'fileStem']:
        if key in params and params[key] and isinstance(params[key], str) and params[key].startswith('~'):
            try: params[key] = os.path.expanduser(params[key])
            except Exception as e_expand: print(f"Warning: Could not expand path for {key} ('{params[key]}'): {e_expand}")

    # Add path to BigDetector mask file if Hydra active
    if params['HydraActive']:
        params['bigFN'] = f"BigDetectorMaskEdgeSize{params['bigdetsize']}x{params['bigdetsize']}Unsigned16Bit.bin"
        param_dir = Path(param_fn).parent
        big_det_path = param_dir / params['bigFN']
        if not big_det_path.exists(): big_det_path = Path.cwd() / params['bigFN']
        params['big_det_full_path'] = str(big_det_path) if big_det_path.exists() else None
        print(f"Expected Big Detector Mask Path: {params.get('big_det_full_path', 'Not Found')}")

    print("Parameter reading finished.")
    return params


def get_file_path(params, file_nr, det_nr=-1, is_dark=False):
    """Constructs the full path for a data or dark file."""
    if not params: print("Warning: get_file_path called with empty params."); return None
    stem_key = 'darkStem' if is_dark else 'fileStem'
    num_key = 'darkNum' if is_dark else None
    stem = params.get(stem_key)
    num = params.get(num_key) if num_key else file_nr
    folder = params.get('folder') # Should be expanded path already
    padding = try_parse_int(params.get('Padding', 6))
    sep_folders = params.get('sepfolderVar', False)

    if stem is None or num is None or folder is None:
        print(f"Warning: Missing path components (stem={stem}, num={num}, folder={folder})")
        return None

    if params.get('HydraActive', False) and det_nr != -1: # Multi-detector
        ext = f"ge{det_nr}"
        base_folder = Path(folder)
        fldr = base_folder / f"ge{det_nr}" if sep_folders else base_folder
    else: # Single detector or non-Hydra
         ext_from_param = params.get('fnext')
         if ext_from_param: ext = ext_from_param[1:] if ext_from_param.startswith('.') else ext_from_param
         else: print(f"Warning: No file extension ('Ext') found in params."); ext = "" # Try without
         fldr = Path(folder)

    try:
        filename = f"{stem}_{str(num).zfill(padding)}"
        if ext: filename += f".{ext}"
        full_path = Path(fldr) / filename
    except Exception as e: print(f"Error constructing filename: {e}"); return None

    return str(full_path)


def load_big_detector_mask(params):
    """Loads the pre-generated big detector mask."""
    big_fn = params.get('big_det_full_path') # Path should be pre-calculated
    big_size = params.get('bigdetsize', 2048)
    if big_fn and Path(big_fn).exists():
        print(f"Loading big detector mask: {big_fn}")
        try:
            with open(big_fn, 'rb') as bigf:
                mask_data = np.fromfile(bigf, dtype=np.uint16, count=big_size*big_size)
            if mask_data.size != big_size*big_size: print(f"Warning: Big detector mask file size mismatch."); return None
            mask = mask_data.reshape((big_size, big_size)).astype(float)
            print("Big detector mask loaded successfully.")
            return mask
        except Exception as e: print(f"Error loading big detector mask {big_fn}: {traceback.format_exc()}"); return None
    else:
        print(f"Big detector mask file not found: {big_fn}")
        return np.zeros((big_size, big_size), dtype=float) # Default to dark background


def generate_big_detector_mask(params, midas_dir):
     """Calls the external MapMultipleDetectors binary. Returns (success_bool, message_str)."""
     map_cmd_path = midas_dir / "bin" / "MapMultipleDetectors"
     param_fn = params.get('paramFN')
     if not map_cmd_path.is_file(): msg = f"MapMultipleDetectors not found at {map_cmd_path}"; print(f"Error: {msg}"); return False, msg
     if not param_fn or not Path(param_fn).exists(): msg = f"Parameter file path not found ('{param_fn}')"; print(f"Error: {msg}"); return False, msg
     print(f"Running: {map_cmd_path} {param_fn}")
     try:
          result = subprocess.run([str(map_cmd_path), str(param_fn)], check=True, capture_output=True, text=True, encoding='utf-8')
          print("MapMultipleDetectors Output:\n", result.stdout)
          if result.stderr: print("MapMultipleDetectors Error Output:\n", result.stderr)
          print("Big detector mask generation command finished successfully.")
          big_fn_base = f"BigDetectorMaskEdgeSize{params['bigdetsize']}x{params['bigdetsize']}Unsigned16Bit.bin"
          param_dir = Path(param_fn).parent; big_det_path = param_dir / big_fn_base
          if not big_det_path.exists(): big_det_path = Path.cwd() / big_fn_base
          if big_det_path.exists():
               params['big_det_full_path'] = str(big_det_path); print(f"Big detector mask path updated: {params['big_det_full_path']}"); return True, f"Success.\nOutput:\n{result.stdout}"
          else: msg = f"Command finished, but mask file not found."; print(f"Error: {msg}"); params['big_det_full_path'] = None; return False, f"{msg}\nOutput:\n{result.stdout}"
     except FileNotFoundError: msg = f"Error: Command '{map_cmd_path}' could not be executed."; print(msg); return False, msg
     except subprocess.CalledProcessError as e: msg = f"Error running MapMultipleDetectors (code {e.returncode}):\nOutput:\n{e.stdout}\nError:\n{e.stderr}"; print(msg); return False, msg
     except Exception as e: msg = f"Unexpected error during MakeBigDetector: {traceback.format_exc()}"; print(msg); return False, msg