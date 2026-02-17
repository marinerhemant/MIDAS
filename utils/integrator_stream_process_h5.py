#!/usr/bin/env python3
"""
integrator_stream_process_h5.py - Convert binary output files from
IntegratorFitPeaksGPUStream to HDF5 and GSAS-II–compatible zarr.zip

This script reads the binary output files produced by the CUDA integrator 
(lineout.bin, and optionally fit.bin, Int2D.bin, and fit_curves.bin) and 
organizes them into a single HDF5 file with appropriate groups and datasets.

Optionally, it also creates a .zarr.zip file that can be imported directly
into GSAS-II using the 'MIDAS zarr' reader. The zarr file contains the
REtaMap, OmegaSumFrame, and InstrumentParameters groups.

The order of datasets and the grouping for summed frames is determined by the
sorted filenames in the provided mapping file.

OPTIMIZATION:
This script uses a streaming approach for large data (Int2D.bin, Lineout.bin) to
avoid loading entire files into RAM. It reads and writes data frame-by-frame 
(or group-by-group) based on the sorted order.
"""

import numpy as np
import h5py
import os
import argparse
import json
import datetime
import struct

try:
    import zarr
    from zarr.storage import ZipStore
except ImportError:
    zarr = None

# Default instrument parameters (matching G2pwd_MIDAS.py / IntegratorZarr.c)
DEFAULT_INST_PARAMS = {
    'Lam': 0.413263,      # Wavelength in Angstroms
    'Polariz': 0.99,
    'SH_L': 0.002,
    'U': 1.163,
    'V': -0.126,
    'W': 0.063,
    'X': 0.0,
    'Y': 0.0,
    'Z': 0.0,
    'Distance': 1000000.0, # Sample-detector distance in µm
}


def get_frame_dimensions(params_file):
    """
    Extracts dimensions, peak count, and instrument parameters from the
    integrator's parameter file.
    
    Args:
        params_file: Path to the parameter file
        
    Returns:
        Tuple (nRBins, nEtaBins, OSF, num_peaks, do_peak_fit, inst_params)
        where inst_params is a dict with keys matching InstrumentParameters.
    """
    # Default values
    RMax, RMin, RBinSize = 100, 10, 0.1
    EtaMax, EtaMin, EtaBinSize = 180, -180, 0.1
    OSF = None
    num_peaks = 0
    do_peak_fit = True  # Assume fitting is done by default
    
    # Instrument parameters (start with defaults)
    inst_params = dict(DEFAULT_INST_PARAMS)
    # Omega info for zarr output
    omega_start = None
    omega_step = None
    omega_fixed = None  # If 'Omega' key present, all frames share this value
    px = 200.0  # pixel size in µm

    with open(params_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            if len(parts) < 2:
                continue
                
            key = parts[0]
            value = parts[1]

            # Binning parameters
            if key == 'RMax': RMax = float(value)
            elif key == 'RMin': RMin = float(value)
            elif key == 'RBinSize': RBinSize = float(value)
            elif key == 'EtaMax': EtaMax = float(value)
            elif key == 'EtaMin': EtaMin = float(value)
            elif key == 'EtaBinSize': EtaBinSize = float(value)
            elif key == 'OmegaSumFrames': OSF = int(value)
            elif key == 'PeakLocation': num_peaks += 1
            elif key == 'DoPeakFit' and value == '0': do_peak_fit = False
            # Instrument parameters
            elif key == 'Wavelength': inst_params['Lam'] = float(value)
            elif key == 'Lsd': inst_params['Distance'] = float(value)  # already in µm
            elif key == 'Polariz': inst_params['Polariz'] = float(value)
            elif key == 'SHpL': inst_params['SH_L'] = float(value)
            elif key == 'U': inst_params['U'] = float(value)
            elif key == 'V': inst_params['V'] = float(value)
            elif key == 'W': inst_params['W'] = float(value)
            elif key == 'X': inst_params['X'] = float(value)
            elif key == 'Y': inst_params['Y'] = float(value)
            elif key == 'Z': inst_params['Z'] = float(value)
            # Omega info
            elif key == 'OmegaStart': omega_start = float(value)
            elif key == 'OmegaStep': omega_step = float(value)
            elif key == 'Omega': omega_fixed = float(value)
            # Pixel size
            elif key in ('PixelSize', 'PixelSizeY'): px = float(value)

    nRBins = int(np.ceil((RMax - RMin) / RBinSize))
    nEtaBins = int(np.ceil((EtaMax - EtaMin) / EtaBinSize))
    
    # Store omega and pixel info in inst_params for zarr creation
    inst_params['_omega_start'] = omega_start
    inst_params['_omega_step'] = omega_step
    inst_params['_omega_fixed'] = omega_fixed
    inst_params['_px'] = px
    
    return nRBins, nEtaBins, OSF, num_peaks, do_peak_fit, inst_params

def load_mapping_file(mapping_file, start_index=0):
    if not os.path.exists(mapping_file):
        print(f"Warning: Mapping file {mapping_file} does not exist")
        return None
    with open(mapping_file, 'r') as f:
        mapping = json.load(f)
    adjusted_mapping = {}
    for k, v in mapping.items():
        if isinstance(k, str) and k.isdigit(): k = int(k)
        if isinstance(k, int) and k >= start_index:
            adjusted_mapping[k - start_index] = v
    return adjusted_mapping

def extract_dataset_mapping_from_server_log(log_file):
    if not os.path.exists(log_file):
        print(f"Warning: Log file {log_file} does not exist"); return None
    mapping = {}; current_dataset = None; current_filename = None; frame_idx = 0
    with open(log_file, 'r') as f:
        for line in f:
            if "Sent dataset #" in line:
                import re; match = re.search(r"#(\d+)", line)
                if match:
                    current_dataset = int(match.group(1))
                    if frame_idx not in mapping: mapping[frame_idx] = {}
                    mapping[frame_idx]['dataset_num'] = current_dataset
                    if current_filename: mapping[frame_idx]['filename'] = current_filename
                    frame_idx += 1
            elif "Processing TIF frame #" in line:
                match = re.search(r"#(\d+)", line)
                if match: current_dataset = int(match.group(1))
            elif "Processing" in line and "file:" in line:
                match = re.search(r"Processing .* file: (.+)", line)
                if match: current_filename = os.path.basename(match.group(1))
            elif "Processed" in line and "frame" in line and not "Sent dataset" in line:
                if current_filename is not None:
                    if frame_idx not in mapping: mapping[frame_idx] = {}
                    mapping[frame_idx]['filename'] = current_filename
                    frame_idx += 1
    for k, v in list(mapping.items()):
        if not isinstance(v, dict):
            mapping[k] = {'filename': v} if isinstance(v, str) else {'dataset_num': v}
    return mapping

def read_fit_curves(filename):
    """
    Reads the fit_curves.bin file roughly into memory (it's small usually).
    Returns dict: frame_index -> (start_bin, curve_array)
    """
    if not os.path.exists(filename):
        return {}
    
    curves_data = {}
    header_fmt = 'iii'  # 3 integers
    header_size = struct.calcsize(header_fmt)

    try:
        file_size = os.path.getsize(filename)
        with open(filename, 'rb') as f:
            while f.tell() < file_size:
                header_bytes = f.read(header_size)
                if len(header_bytes) < header_size:
                    break
                
                frame_idx, start_bin, num_points = struct.unpack(header_fmt, header_bytes)
                
                data_fmt = f'{num_points}d'
                data_size = struct.calcsize(data_fmt)
                data_bytes = f.read(data_size)
                
                if len(data_bytes) < data_size:
                    print("Warning: Truncated fit_curves.bin")
                    break
                    
                curve_array = np.array(struct.unpack(data_fmt, data_bytes))
                curves_data[frame_idx] = (start_bin, curve_array)
    except Exception as e:
        print(f"Error reading fit_curves.bin: {e}")
                
    return curves_data

def read_frame_chunk(file_obj, frame_index, frame_size_bytes, dtype=np.float64, shape=None):
    """
    Reads a single frame's data from an already open binary file at the correct offset.
    """
    itemsize = np.dtype(dtype).itemsize
    offset = frame_index * frame_size_bytes
    file_obj.seek(offset)
    data = np.fromfile(file_obj, dtype=dtype, count=(frame_size_bytes // itemsize))
    
    if len(data) * itemsize != frame_size_bytes:
        # End of file or partial read
        return None
        
    if shape:
        return data.reshape(shape)
    return data

def create_hdf5_file_streamed(output_file, 
                              lineout_file, 
                              lineout_sm_file,
                              fit_file, 
                              int2d_file, 
                              fit_curves_file, 
                              map_data_file,
                              mapping, 
                              nRBins, nEtaBins, osf, num_peaks):
    
    # --- Open Binary Files ---
    # --- Open Binary Files with Large Buffering ---
    buffer_size = 10 * 1024 * 1024  # 10 MB buffer
    f_lineout = open(lineout_file, 'rb', buffering=buffer_size) if lineout_file and os.path.exists(lineout_file) else None
    f_lineout_sm = open(lineout_sm_file, 'rb', buffering=buffer_size) if lineout_sm_file and os.path.exists(lineout_sm_file) else None
    f_fit = open(fit_file, 'rb', buffering=buffer_size) if fit_file and os.path.exists(fit_file) else None
    f_int2d = open(int2d_file, 'rb', buffering=buffer_size) if int2d_file and os.path.exists(int2d_file) else None
    
    # Calculate sizes
    lineout_frame_size = nRBins * 2 * 8 # doubles
    int2d_frame_size = nRBins * nEtaBins * 8 # doubles
    fit_frame_size = num_peaks * 7 * 8 # doubles (7 params per peak)
    
    # Determine total frames from lineout file size
    if f_lineout:
        f_lineout.seek(0, 2)
        total_bytes = f_lineout.tell()
        num_frames = total_bytes // lineout_frame_size
        f_lineout.seek(0)
    else:
        num_frames = 0
        
    print(f"Total Frames Detected: {num_frames}")
    
    # Load fit curves fully (usually small enough)
    fit_curves_data = read_fit_curves(fit_curves_file) if fit_curves_file else {}
    
    # --- Create Output Groups ---
    with h5py.File(output_file, 'w') as h5f:
        h5f.attrs['creation_date'] = np.bytes_(datetime.datetime.now().isoformat())
        h5f.attrs['num_frames'] = num_frames
        
        # Structure
        grp_geom = h5f.create_group('geometry_maps')
        grp_line = h5f.create_group('lineouts') if f_lineout else None
        grp_line_sm = h5f.create_group('lineouts_simple_mean') if f_lineout_sm else None
        grp_fit = h5f.create_group('fit') if f_fit else None
        grp_curves = h5f.create_group('fit_curves') if fit_curves_data and f_lineout else None
        grp_omega = h5f.create_group('OmegaSumFrame') if f_int2d else None
        
        # --- Write Geometry Maps (Small, Single Read) ---
        if map_data_file and os.path.exists(map_data_file):
            raw_map = np.fromfile(map_data_file, dtype=np.float64)
            if len(raw_map) == 4 * nRBins * nEtaBins:
                map_reshaped = raw_map.reshape(4, nRBins, nEtaBins)
                
                ds_r = grp_geom.create_dataset('R_map', data=map_reshaped[0], track_times=False)
                ds_r.attrs['description'] = np.bytes_("R-center for each bin")
                ds_r.attrs['units'] = np.bytes_("pixels")
                
                ds_tth = grp_geom.create_dataset('TTh_map', data=map_reshaped[1], track_times=False)
                ds_tth.attrs['description'] = np.bytes_("TwoTheta-center for each bin")
                ds_tth.attrs['units'] = np.bytes_("degrees")
                
                ds_eta = grp_geom.create_dataset('Eta_map', data=map_reshaped[2], track_times=False)
                ds_eta.attrs['description'] = np.bytes_("Eta-center for each bin")
                ds_eta.attrs['units'] = np.bytes_("degrees")
                
                ds_area = grp_geom.create_dataset('Area_map', data=map_reshaped[3], track_times=False)
                ds_area.attrs['description'] = np.bytes_("Effective pixel area for each bin")
                ds_area.attrs['units'] = np.bytes_("fractional pixels")
                print("Geometry maps written.")
        
        # --- Sorting Logic ---
        frame_indices = list(range(num_frames))
        if mapping:
            def get_sort_key(frame_index):
                map_value = mapping.get(frame_index)
                is_mapped = map_value is not None
                if is_mapped:
                    sort_string = map_value.get('filename') or str(map_value.get('uniqueId', ''))
                    return (0, sort_string)
                return (1, frame_index)
            frame_indices.sort(key=get_sort_key)
            print("Sorted frames based on mapping.")

        def get_dataset_name(frame_index):
            if not mapping: return str(frame_index)
            map_value = mapping.get(frame_index)
            if map_value is None: return str(frame_index)
            name = map_value.get('filename')
            if name: return os.path.splitext(name)[0]
            return str(map_value.get('uniqueId') or map_value.get('dataset_num', frame_index))

        # --- Streaming Loop ---
        # We iterate through the SORTED frame indices.
        # For OSF grouping, we group contiguous items in the SORTED list.
        
        # Determine grouping
        if osf is not None and osf == -1:
            actual_osf = num_frames  # Sum all frames
        elif osf is None or osf <= 0:
            actual_osf = 1
        else:
            actual_osf = osf
            
        # Process in chunks of 'actual_osf'
        # This ensures we have the frames needed for summation together in the loop
        
        total_groups = (len(frame_indices) + actual_osf - 1) // actual_osf
        
        print(f"Starting conversion... ({total_groups} groups to process)")
        
        for group_idx in range(total_groups):
            start_i = group_idx * actual_osf
            end_i = min(start_i + actual_osf, len(frame_indices))
            
            current_group_frames = frame_indices[start_i:end_i]
            
            # --- Per-Frame Data (Lineouts, Fits, Curves) ---
            for frame_idx in current_group_frames:
                ds_name = get_dataset_name(frame_idx)
                
                # Lineout
                if grp_line:
                    data = read_frame_chunk(f_lineout, frame_idx, lineout_frame_size, shape=(nRBins, 2))
                    if data is not None:
                        ds = grp_line.create_dataset(ds_name, data=data, track_times=False)
                        ds.attrs['frame_index'] = frame_idx
                        
                        # Curves (depend on lineout for R axis)
                        if grp_curves and frame_idx in fit_curves_data:
                            start_bin, curve = fit_curves_data[frame_idx]
                            ds_c = grp_curves.create_dataset(ds_name, data=curve, track_times=False)
                            ds_c.attrs['original_frame_index'] = frame_idx
                            ds_c.attrs['start_bin_index'] = start_bin
                            # Extract R axis slice from lineout data we just read
                            num_pts = len(curve)
                            if start_bin + num_pts <= data.shape[0]:
                                ds_c.attrs['R_values'] = data[start_bin : start_bin + num_pts, 0]

                # Lineout Simple Mean
                if grp_line_sm:
                    data = read_frame_chunk(f_lineout_sm, frame_idx, lineout_frame_size, shape=(nRBins, 2))
                    if data is not None:
                        ds = grp_line_sm.create_dataset(ds_name, data=data, track_times=False) 
                        ds.attrs['frame_index'] = frame_idx

                # Fit
                if grp_fit:
                    data = read_frame_chunk(f_fit, frame_idx, fit_frame_size, shape=(num_peaks, 7))
                    if data is not None:
                        ds = grp_fit.create_dataset(ds_name, data=data, track_times=False)
                        ds.attrs['frame_index'] = frame_idx

            # --- Summed Data (Int2D Omega Sum) ---
            if grp_omega:
                # We need to read all frames in this group, sum them, and write one dataset
                sum_buffer = None
                valid_frames_count = 0
                
                for frame_idx in current_group_frames:
                    data = read_frame_chunk(f_int2d, frame_idx, int2d_frame_size, shape=(nRBins, nEtaBins))
                    if data is not None:
                        if sum_buffer is None:
                            sum_buffer = np.zeros_like(data)
                        sum_buffer += data
                        valid_frames_count += 1
                
                if sum_buffer is not None:
                    first_name = get_dataset_name(current_group_frames[0])
                    last_name = get_dataset_name(current_group_frames[-1])
                    ds_name_sum = f"Summed_from_{first_name}_to_{last_name}"
                    
                    ds = grp_omega.create_dataset(ds_name_sum, data=sum_buffer, track_times=False)
                    ds.attrs['Number Of Frames Summed'] = valid_frames_count
                    ds.attrs['original_frame_indices'] = current_group_frames
                    if mapping:
                        ds.attrs['last_frame_identifier'] = np.bytes_(str(last_name))

            if group_idx % 10 == 0:
                print(f"Processed group {group_idx}/{total_groups}", end='\r', flush=True)

        print(f"\nCompleted processing {total_groups} groups.")

    # --- Close Files ---
    if f_lineout: f_lineout.close()
    if f_lineout_sm: f_lineout_sm.close()
    if f_fit: f_fit.close()
    if f_int2d: f_int2d.close()


# =========================================================================
# Zarr.zip output for GSAS-II
# =========================================================================

def create_zarr_zip(zarr_output,
                    int2d_file,
                    map_data_file,
                    mapping,
                    nRBins, nEtaBins, osf,
                    inst_params,
                    num_frames=None):
    """
    Create a GSAS-II–compatible .zarr.zip file from the streaming integrator's
    binary outputs.

    The zarr file contains three top-level groups that the MIDAS zarr reader
    (G2pwd_MIDAS.py) requires:

        REtaMap              — (4, nRBins, nEtaBins)  geometry maps
        OmegaSumFrame        — group of summed 2D frames with omega attrs
        InstrumentParameters — group of scalar instrument parameters

    Args:
        zarr_output:   Path for the output .zarr.zip file
        int2d_file:    Path to Int2D.bin (per-frame 2D integrated arrays)
        map_data_file: Path to RTthEtaAreaMap.bin (4 × nRBins × nEtaBins)
        mapping:       Frame mapping dict (or None)
        nRBins:        Number of radial bins
        nEtaBins:      Number of azimuthal bins
        osf:           OmegaSumFrames value
        inst_params:   Dict of instrument parameters (from get_frame_dimensions)
        num_frames:    Total number of frames (auto-detected from int2d_file)
    """
    if zarr is None:
        print("WARNING: zarr module not available. Skipping zarr.zip creation.")
        print("Install with: pip install zarr==2.18.3")
        return

    # --- Validate inputs ---
    if not int2d_file or not os.path.exists(int2d_file):
        print("WARNING: Int2D.bin not found — cannot create zarr.zip without 2D data.")
        return
    if not map_data_file or not os.path.exists(map_data_file):
        print("WARNING: RTthEtaAreaMap.bin not found — cannot create zarr.zip.")
        return

    int2d_frame_size = nRBins * nEtaBins * 8  # doubles

    # Detect total frames from Int2D.bin size
    if num_frames is None:
        total_bytes = os.path.getsize(int2d_file)
        num_frames = total_bytes // int2d_frame_size

    if num_frames == 0:
        print("WARNING: No frames found in Int2D.bin.")
        return

    print(f"Creating zarr.zip: {zarr_output}")
    print(f"  {num_frames} frames, {nRBins} R bins × {nEtaBins} Eta bins")

    # --- Open zarr zip store (zarr v2) ---
    store = ZipStore(zarr_output, mode='w')
    root = zarr.group(store, overwrite=True)

    # --- 1. REtaMap (4, nRBins, nEtaBins) from RTthEtaAreaMap.bin ---
    raw_map = np.fromfile(map_data_file, dtype=np.float64)
    expected_size = 4 * nRBins * nEtaBins
    if len(raw_map) != expected_size:
        print(f"WARNING: Map file size mismatch: got {len(raw_map)}, expected {expected_size}")
        store.close()
        return

    remap = raw_map.reshape(4, nRBins, nEtaBins)
    root.array('REtaMap', data=remap, dtype='float64', chunks=False)
    print("  Written: REtaMap")

    # --- 2. OmegaSumFrame group ---
    osf_grp = root.create_group('OmegaSumFrame')

    # Sorting logic (same as HDF5 path)
    frame_indices = list(range(num_frames))
    if mapping:
        def get_sort_key(fi):
            mv = mapping.get(fi)
            if mv is not None:
                sk = mv.get('filename') or str(mv.get('uniqueId', ''))
                return (0, sk)
            return (1, fi)
        frame_indices.sort(key=get_sort_key)

    # Determine grouping
    if osf is not None and osf == -1:
        actual_osf = num_frames
    elif osf is None or osf <= 0:
        actual_osf = 1
    else:
        actual_osf = osf

    total_groups = (len(frame_indices) + actual_osf - 1) // actual_osf

    # Omega calculation
    omega_start = inst_params.get('_omega_start')
    omega_step = inst_params.get('_omega_step')
    omega_fixed = inst_params.get('_omega_fixed')

    f_int2d = open(int2d_file, 'rb', buffering=10 * 1024 * 1024)

    for group_idx in range(total_groups):
        start_i = group_idx * actual_osf
        end_i = min(start_i + actual_osf, len(frame_indices))
        current_group_frames = frame_indices[start_i:end_i]

        # Sum frames in this group
        sum_buffer = np.zeros((nRBins, nEtaBins), dtype=np.float64)
        valid_count = 0

        for fi in current_group_frames:
            data = read_frame_chunk(f_int2d, fi, int2d_frame_size,
                                   shape=(nRBins, nEtaBins))
            if data is not None:
                sum_buffer += data
                valid_count += 1

        if valid_count == 0:
            continue

        # Dataset name matches IntegratorZarr.c convention
        last_frame = current_group_frames[-1]
        ds_name = f"LastFrameNumber_{last_frame}"
        ds = osf_grp.array(ds_name, data=sum_buffer, dtype='float64', chunks=False)

        # Attributes expected by G2pwd_MIDAS.py
        ds.attrs['Number Of Frames Summed'] = valid_count
        ds.attrs['LastFrameNumber'] = int(last_frame)

        # Omega attributes
        first_frame = current_group_frames[0]
        if omega_fixed is not None:
            ds.attrs['FirstOme'] = omega_fixed
            ds.attrs['LastOme'] = omega_fixed
        elif omega_start is not None and omega_step is not None:
            ds.attrs['FirstOme'] = omega_start + first_frame * omega_step
            ds.attrs['LastOme'] = omega_start + last_frame * omega_step
        else:
            # No omega info — use frame index as placeholder
            ds.attrs['FirstOme'] = float(first_frame)
            ds.attrs['LastOme'] = float(last_frame)

        if group_idx % 50 == 0:
            print(f"  OmegaSumFrame: group {group_idx+1}/{total_groups}",
                  end='\r', flush=True)

    f_int2d.close()
    print(f"  Written: OmegaSumFrame ({total_groups} groups)")

    # --- 3. InstrumentParameters group ---
    ip_grp = root.create_group('InstrumentParameters')
    # Write each parameter as a 1-element array (matching IntegratorZarr.c HDF5 layout)
    for key in ('Lam', 'Polariz', 'SH_L', 'U', 'V', 'W', 'X', 'Y', 'Z', 'Distance'):
        val = inst_params.get(key, DEFAULT_INST_PARAMS.get(key, 0.0))
        ip_grp.array(key, data=np.array([val], dtype='float64'), chunks=False)

    print("  Written: InstrumentParameters")

    store.close()
    print(f"Successfully created zarr.zip: {zarr_output}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert binary output files from IntegratorFitPeaksGPUStream "
                    "to HDF5 and optionally GSAS-II–compatible zarr.zip")
    parser.add_argument('--lineout', type=str, default='lineout.bin', help='Lineout binary file')
    parser.add_argument('--lineout-simple-mean', type=str, default='lineout_simple_mean.bin', help='Lineout binary file (simple mean)')
    parser.add_argument('--fit', type=str, default='fit.bin', help='Fit binary file')
    parser.add_argument('--int2d', type=str, default='Int2D.bin', help='2D integrated binary file')
    parser.add_argument('--fit-curves', type=str, default='fit_curves.bin', help='Fitted curves binary file')
    parser.add_argument('--params', type=str, required=True, help='Parameter file used for integration')
    parser.add_argument('--map-data', type=str, default='RTthEtaAreaMap.bin', help='R, TTh, Eta, Area map binary file')
    parser.add_argument('--mapping', type=str, help='JSON file mapping frame indices to dataset IDs')
    parser.add_argument('--server-log', type=str, help='Server log file to extract dataset IDs')
    parser.add_argument('--output', type=str, default='integrator_output.h5', help='Output HDF5 file')
    parser.add_argument('--zarr-output', type=str, default=None,
                        help='Output zarr.zip file for GSAS-II (default: <output>.zarr.zip)')
    parser.add_argument('--no-zarr', action='store_true',
                        help='Skip zarr.zip creation')
    parser.add_argument('--start', type=int, default=0, help='Starting frame index (deprecated/unused in streaming mode)')
    parser.add_argument('--count', type=int, help='Number of frames to process (deprecated/unused in streaming mode)')
    parser.add_argument('--omega-sum-frames', type=int, help='Override OmegaSumFrames value')
    args = parser.parse_args()
    
    nRBins, nEtaBins, osf, num_peaks, do_peak_fit, inst_params = get_frame_dimensions(args.params)
    print(f"Frame dimensions: {nRBins} R bins, {nEtaBins} Eta bins")
    
    if args.omega_sum_frames is not None:
        osf = args.omega_sum_frames
        print(f"Overriding OmegaSumFrames with value: {osf}")

    mapping = None
    if args.mapping: 
        mapping = load_mapping_file(args.mapping, args.start)
    elif args.server_log: 
        mapping = extract_dataset_mapping_from_server_log(args.server_log)
        
    # --- HDF5 output ---
    create_hdf5_file_streamed(args.output, 
                              args.lineout, 
                              args.lineout_simple_mean, 
                              args.fit if do_peak_fit else None, 
                              args.int2d, 
                              args.fit_curves if do_peak_fit else None, 
                              args.map_data, 
                              mapping, 
                              nRBins, nEtaBins, osf, num_peaks)
    print(f"Successfully created HDF5 file: {args.output}")

    # --- Zarr.zip output for GSAS-II ---
    if not args.no_zarr:
        zarr_out = args.zarr_output
        if zarr_out is None:
            # Derive from HDF5 output name
            base = os.path.splitext(args.output)[0]
            zarr_out = base + '.zarr.zip'

        create_zarr_zip(
            zarr_output=zarr_out,
            int2d_file=args.int2d,
            map_data_file=args.map_data,
            mapping=mapping,
            nRBins=nRBins,
            nEtaBins=nEtaBins,
            osf=osf,
            inst_params=inst_params,
        )


if __name__ == "__main__":
    main()