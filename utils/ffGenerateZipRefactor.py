#!/usr/bin/env python

import warnings
import h5py
import numpy as np
import os
import sys
import argparse
import zarr
from numcodecs import Blosc
from pathlib import Path
import shutil
from numba import jit
import re
from math import ceil

# --- Global Configuration ---
compressor = Blosc(cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE)

# --- Helper Functions ---

def parse_parameter_file(filename):
    """Reads the parameter file into a dictionary, handling multi-entry keys."""
    params = {}
    try:
        with open(filename, 'r') as f:
            for line in f:
                line_no_comment = line.split('#', 1)[0]
                
                cleaned_line = line_no_comment.strip()

                # If the line is empty now, skip it.
                if not cleaned_line:
                    continue

                parts = cleaned_line.split()
                if not parts: continue
                key, values = parts[0], parts[1:]
                processed_values = []
                for v in values:
                    # This regex handles integers and floats correctly
                    if re.match(r"^-?\d+$", v): processed_values.append(int(v))
                    elif re.match(r"^-?\d*\.\d+$", v) or re.match(r"^-?\d+\.\d*$", v): processed_values.append(float(v))
                    else: processed_values.append(v)
                
                final_value = processed_values if len(processed_values) > 1 else processed_values[0]
                
                # --- START OF CORRECTED LOGIC ---
                if key not in params:
                    # If we haven't seen this key before, just assign the value.
                    params[key] = final_value
                else:
                    # If the key already exists, we have a multi-value parameter.
                    # First, check if the existing value is already a list.
                    if not isinstance(params[key], list):
                        # If not, convert it into a list containing the original value.
                        params[key] = [params[key]]
                    # Now, append the new value to the list.
                    params[key].append(final_value)
                # --- END OF CORRECTED LOGIC ---

    except FileNotFoundError:
        print(f"Error: Parameter file not found at '{filename}'")
        sys.exit(1)

    return params

@jit(nopython=True)
def apply_correction(img, dark_mean, pre_proc_thresh_val):
    """Applies dark correction. This function is now type-agnostic."""
    result = np.empty(img.shape, dtype=img.dtype)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for k in range(img.shape[2]):
                if img[i, j, k] < pre_proc_thresh_val[j, k]:
                    result[i, j, k] = 0
                else:
                    result[i, j, k] = max(0, int(img[i, j, k]) - int(dark_mean[j, k]))
    return result

def create_zarr_structure(zRoot):
    """Creates the basic Zarr group hierarchy."""
    groups = { 'exc': zRoot.create_group('exchange'), 'meas': zRoot.create_group('measurement') }
    groups['pro_meas'] = groups['meas'].create_group('process')
    groups['sp_pro_meas'] = groups['pro_meas'].create_group('scan_parameters')
    analysis_group = zRoot.create_group('analysis')
    pro_analysis_group = analysis_group.create_group('process')
    groups['sp_pro_analysis'] = pro_analysis_group.create_group('analysis_parameters')
    return groups

def write_analysis_parameters(z_groups, config):
    """
    Writes parameters from the config dictionary to the Zarr file,
    enforcing specific data types, array shapes, and default dataset existence.
    """

    # ALLOWED_MULTIVALUE_PARAMS = {
    #     'RingThresh', 'RingsToExclude', 'OmegaRanges', 'BoxSizes', 'ImTransOpt',
    #     'LatticeParameter', 'LatticeConstant'
    # }
    # for key, value in config.items():
    #     if key == 'BC':
    #         if not isinstance(value, list) or len(value) != 2:
    #             print(f"\nFATAL ERROR: The 'BC' parameter must have exactly 2 values. Found: {value}.")
    #             sys.exit(1)
    #         continue
    #     if isinstance(value, list) and key not in ALLOWED_MULTIVALUE_PARAMS:
    #         print(f"\nFATAL ERROR: The parameter '{key}' is not allowed to have multiple values. Found: {value}.")
    #         sys.exit(1)

    sp_pro_analysis, sp_pro_meas = z_groups['sp_pro_analysis'], z_groups['sp_pro_meas']
    print("\nWriting analysis parameters to Zarr file...")

    FORCE_DOUBLE_PARAMS = { "RMin", "RMax", "px", "PixelSize", "Completeness", "MinMatchesToAcceptFrac", "OverArea", "IntensityThresh", "MinS_N", "YPixelSize", "ZPixelSize", "BeamStopY", "BeamStopZ", "DetDist", "MaxDev", "OmegaStart", "OmegaFirstFile", "OmegaStep", "step", "BadPxIntensity", "GapIntensity", "FitWeightMean", "PixelSplittingRBin", "tolTilts", "tolBC", "tolLsd", "DiscArea", "OverlapLength", "ReferenceRingCurrent", "zDiffThresh", "GlobalPosition", "tolPanelFit", "tolP", "tolP0", "tolP1", "tolP2", "tolP3", "StepSizePos", "tInt", "tGap", "StepSizeOrient", "MarginRadius", "MarginRadial", "MarginEta", "MarginOme", "MargABG", "MargABC", "OmeBinSize", "EtaBinSize", "RBinSize", "EtaMin", "MinEta", "EtaMax", "X", "Y", "Z", "U", "V", "W", "SHpL", "Polariz", "MaxOmeSpotIDsToIndex", "MinOmeSpotIDsToIndex", "BeamThickness", "Wedge", "Rsample", "Hbeam", "Vsample", "RhoD", "MaxRingRad", "Lsd", "Wavelength", "Width", "WidthTthPx", "UpperBoundThreshold", "p3", "p2", "p1", "p0", "tz", "ty", "tx" }
    FORCE_INT_PARAMS = { "Twins", "MaxNFrames", "DoFit", "DiscModel", "UseMaximaPositions", "MaxNrPx", "MinNrPx", "MaxNPeaks", "PhaseNr", "NumPhases", "MinNrSpots", "UseFriedelPairs", "OverallRingToIndex", "SpaceGroup", "LayerNr", "DoFullImage", "SkipFrame", "SumImages", "Normalize", "SaveIndividualFrames", "OmegaSumFrames" }
    FORCE_STRING_PARAMS = { "GapFile", "BadPxFile", "ResultFolder" }
    RENAME_MAP = { "OmegaStep": "step", "Completeness": "MinMatchesToAcceptFrac", "px": "PixelSize", "LatticeConstant": "LatticeParameter", "OverallRingToIndex": "OverallRingToIndex", "resultFolder": "ResultFolder", "OmegaRange": "OmegaRanges" }

    for key, value in config.items():
        try:
            target_key = RENAME_MAP.get(key, key)
            target_group = sp_pro_analysis
            if key in ["OmegaStep", "start", "datatype"]: target_group = sp_pro_meas

            if key == 'BC':
                sp_pro_analysis.create_dataset('YCen', data=np.array([value[0]], dtype=np.double))
                sp_pro_analysis.create_dataset('ZCen', data=np.array([value[1]], dtype=np.double))
            elif key == 'LatticeConstant' or key == 'LatticeParameter':
                values = value if isinstance(value, list) else [value]
                padded_values = np.zeros(6, dtype=np.double)
                padded_values[:len(values)] = values
                target_group.create_dataset(target_key, data=padded_values.astype(np.double))
            elif key in ['RingThresh', 'RingsToExclude', 'OmegaRanges', 'BoxSizes', 'ImTransOpt']:
                print(key,value)
                # 1. Ensure the value is a list to prevent subscripting errors on scalars.
                temp_value = value if isinstance(value, list) else [value]

                # 2. Check if it's a list of lists (like RingThresh) or a flat list (like OmegaRange or ImTransOpt).
                # We can check this by seeing if the first element is *not* a list.
                if not temp_value or not isinstance(temp_value[0], list):
                    # It's a scalar (e.g., [0]) or a flat list (e.g., [-180, 180]).
                    # Wrap it in another list so NumPy treats it as a single row.
                    values_to_write = [temp_value]
                else:
                    # It's already a list of lists (e.g., [[1, 80], [2, 80]]). Use as is.
                    values_to_write = temp_value
                
                arr = np.array(values_to_write)
                dtype = np.int32 if key == 'ImTransOpt' else np.double
                target_group.create_dataset(target_key, data=arr.astype(dtype))
            else:
                arr = None
                if key in FORCE_STRING_PARAMS or target_key in FORCE_STRING_PARAMS:
                    arr = np.array([np.bytes_(str(value).encode('UTF-8'))])
                elif key in FORCE_DOUBLE_PARAMS or target_key in FORCE_DOUBLE_PARAMS:
                    arr = np.array(value if isinstance(value, list) else [value], dtype=np.double)
                elif key in FORCE_INT_PARAMS or target_key in FORCE_INT_PARAMS:
                    arr = np.array(value if isinstance(value, list) else [value], dtype=np.int32)
                else:
                    if isinstance(value, str): arr = np.array([np.bytes_(value.encode('UTF-8'))])
                    else: arr = np.array(value if isinstance(value, list) else [value])
                if arr is not None: target_group.create_dataset(target_key, data=arr)
        except Exception as e:
            print(f"  - Warning: Could not write parameter '{key}'. Reason: {e}")

    essential_datasets = {
        'RingThresh':   {'default': np.zeros((1, 2), dtype=np.double)},
        'RingsToExclude': {'default': np.zeros((1, 2), dtype=np.double)},
        'OmegaRanges':  {'default': np.array([[-10000, 0]], dtype=np.double)},
        'BoxSizes':     {'default': np.array([[-10000, 0, 0, 0]], dtype=np.double)},
        'ImTransOpt':   {'default': np.array([-1], dtype=np.int32)}
    }
    for key, props in essential_datasets.items():
        if key not in sp_pro_analysis:
            print(f"  - Info: Parameter '{key}' not found. Creating with default values.")
            sp_pro_analysis.create_dataset(key, data=props['default'])

    ome_ff = float(config.get('OmegaStart', config.get('OmegaFirstFile', 0)))
    ome_stp = float(config.get('OmegaStep', 0))
    skip_f = int(config.get('SkipFrame', 0))
    start_omega = ome_ff - (skip_f * ome_stp)
    sp_pro_meas.create_dataset('start', data=np.array([start_omega], dtype=np.double))

def process_hdf5_scan(config, z_groups):
    """Processes a scan from one or more HDF5 files, with correct SkipFrame logic."""
    print("\n--- Processing HDF5 File(s) ---")
    num_files, skip_frames = int(config.get('numFilesPerScan', 1)), int(config.get('SkipFrame', 0))
    data_loc, dark_loc = config['dataLoc'], config['darkLoc']
    pre_proc_active = int(config.get('preProcThresh', -1)) != -1

    file_list = []
    if num_files == 1: file_list.append(config['dataFN'])
    else:
        match = re.search(r'(\d+)(?=\.\w+$)', config['dataFN'])
        if not match: raise ValueError("Numeric sequence not found in HDF5 filename for multi-file scan.")
        fNr_orig_str, start_nr = match.group(1), int(match.group(1))
        for i in range(num_files):
            current_nr_str = str(start_nr + i).zfill(len(fNr_orig_str))
            file_list.append(config['dataFN'].replace(fNr_orig_str, current_nr_str, 1))

    with h5py.File(file_list[0], 'r') as hf:
        if data_loc not in hf: raise KeyError(f"Data location '{data_loc}' not found in HDF5 file.")
        frames_per_file, nZ, nY = hf[data_loc].shape
        output_dtype = hf[data_loc].dtype

        print("  - Checking for additional metadata inside HDF5 file...")
        hdf5_metadata_paths = {
            'startOmeOverride': 'startOmeOverride',
            '/measurement/instrument/GSAS2_PVS/Pressure': 'Pressure',
            '/measurement/instrument/GSAS2_PVS/Temperature': 'Temperature',
            '/measurement/instrument/GSAS2_PVS/I': 'I',
            '/measurement/instrument/GSAS2_PVS/I0': 'I0',
        }
        for h5_path, zarr_name in hdf5_metadata_paths.items():
            if h5_path in hf:
                try:
                    data_to_copy = hf[h5_path][()]
                    print(f"    - Found '{h5_path}'. Copying to measurement/process/scan_parameters/{zarr_name}.")
                    if not isinstance(data_to_copy, np.ndarray): data_to_copy = np.array([data_to_copy])
                    z_groups['sp_pro_meas'].create_dataset(zarr_name, data=data_to_copy)
                except Exception as e:
                    print(f"    - Warning: Could not copy metadata from '{h5_path}'. Reason: {e}")

    total_frames_to_write = frames_per_file + (frames_per_file - skip_frames) * (num_files - 1)
    print(f"HDF5 scan: {num_files} file(s), {frames_per_file} frames/file. Skipping {skip_frames} from files 2+. Total frames to write: {total_frames_to_write}. Dtype: {output_dtype}")

    z_data = z_groups['exc'].create_dataset('data', shape=(total_frames_to_write, nZ, nY), dtype=output_dtype, chunks=(1, nZ, nY), compression=compressor)
    z_offset = 0

    for i, fn in enumerate(file_list):
        print(f"  - Processing file {i+1}/{num_files}: {fn}")
        with h5py.File(fn, 'r') as hf:
            if data_loc not in hf: print(f"    Warning: '{data_loc}' not in {fn}, skipping."); continue

            # This logic now only runs once, for the first file
            if i == 0:
                dark_frames_found = False
                dark_frames = None

                # 1. Check for dark data in the main data file
                if dark_loc in hf:
                    dark_frames = hf[dark_loc][()]; dark_frames_found = True
                # 2. If not found, check for a separate dark file
                elif config.get('darkFN'):
                    with h5py.File(config['darkFN'], 'r') as hf_dark:
                        # Try the full path first
                        if dark_loc in hf_dark:
                            dark_frames = hf_dark[dark_loc][()]; dark_frames_found = True
                        else:
                            # Fallback: try just the dataset name at the root
                            dark_dataset_name = Path(dark_loc).name
                            if dark_dataset_name in hf_dark:
                                dark_frames = hf_dark[dark_dataset_name][()]; dark_frames_found = True

                if not dark_frames_found:
                    print("    Warning: No dark data found. Using temporary zeros for shape info.")
                    dark_frames = np.zeros((10, nZ, nY), dtype=output_dtype)

                # Decide what to write (real data or zeros) and write it only ONCE.
                dark_shape = dark_frames.shape
                if pre_proc_active:
                    print("  - Pre-processing is active. Writing zero arrays for dark/bright.")
                    z_groups['exc'].zeros('dark', shape=dark_frames.shape, dtype=output_dtype,chunks=(1, dark_shape[1], dark_shape[2]), compression=compressor)
                    z_groups['exc'].zeros('bright', shape=dark_frames.shape, dtype=output_dtype,chunks=(1, dark_shape[1], dark_shape[2]), compression=compressor)
                else:
                    print("  - Writing actual dark and bright frame data.")
                    z_groups['exc'].create_dataset('dark', data=dark_frames, dtype=output_dtype,chunks=(1, dark_shape[1], dark_shape[2]), compression=compressor)
                    z_groups['exc'].create_dataset('bright', data=dark_frames, dtype=output_dtype,chunks=(1, dark_shape[1], dark_shape[2]), compression=compressor)

            dark_mean = np.mean(dark_frames[skip_frames:], axis=0)
            pre_proc_val = (dark_mean + int(config['preProcThresh'])) if pre_proc_active else dark_mean

            start_frame_in_file = skip_frames if i > 0 else 0

            chunk_size = int(config.get('numFrameChunks', 100));
            if chunk_size == -1: chunk_size = frames_per_file

            for j in range(start_frame_in_file, frames_per_file, chunk_size):
                end_frame = min(j + chunk_size, frames_per_file)
                data_chunk = hf[data_loc][j:end_frame]
                if pre_proc_active:
                    data_chunk = apply_correction(data_chunk, dark_mean, pre_proc_val)
                z_data[z_offset : z_offset+len(data_chunk)] = data_chunk
                z_offset += len(data_chunk)
    return output_dtype

def process_multifile_scan(file_type, config, z_groups):
    """Processes a scan of multiple GE/TIFF files with correct SkipFrame logic."""
    print(f"\n--- Processing {file_type.upper()} File(s) ---")
    num_files, skip_frames = int(config.get('numFilesPerScan', 1)), int(config.get('SkipFrame', 0))
    header_size = int(config.get('HeadSize', 8192))
    numPxY, numPxZ = int(config['numPxY']), int(config['numPxZ'])
    pre_proc_active = int(config.get('preProcThresh', -1)) != -1

    output_dtype = np.uint32 if int(config.get('PixelValue', 2 if file_type=='ge' else 4)) == 4 else np.uint16
    bytes_per_pixel = np.dtype(output_dtype).itemsize
    print(f"Handling as {output_dtype.__name__}. Files per scan: {num_files}")

    match = re.search(r'(\d+)', Path(config['dataFN']).stem)
    if not match: raise ValueError("Could not find numeric sequence in data filename.")
    fNr_orig_str, start_nr = match.group(1), int(match.group(1))

    if file_type == 'ge': frames_per_file = (os.path.getsize(config['dataFN']) - header_size) // (bytes_per_pixel * numPxY * numPxZ)
    else: frames_per_file = 1

    total_frames_to_write = frames_per_file + (frames_per_file - skip_frames) * (num_files - 1)
    print(f"Scan: {num_files} file(s), {frames_per_file} frames/file. Skipping {skip_frames} from files 2+. Total frames to write: {total_frames_to_write}.")

    z_data = z_groups['exc'].create_dataset('data', shape=(total_frames_to_write, numPxZ, numPxY), dtype=output_dtype, chunks=(1, numPxZ, numPxY), compression=compressor)
    z_offset = 0

    dark_mean = np.zeros((numPxZ, numPxY), dtype=output_dtype)
    dark_frames_data = np.zeros((10, numPxZ, numPxY), dtype=output_dtype)
    if file_type == 'ge' and config.get('darkFN'):
        print(f"  - Reading GE dark file: {config['darkFN']}")
        dark_frames_data = np.fromfile(config['darkFN'], dtype=output_dtype, offset=header_size).reshape((-1, numPxZ, numPxY))
        dark_mean = np.mean(dark_frames_data[skip_frames:], axis=0)

    # Decide what to write (real data or zeros) and write it only ONCE.
    dark_shape = dark_frames_data.shape
    if pre_proc_active:
        print("  - Pre-processing is active. Writing zero arrays for dark/bright.")
        z_groups['exc'].zeros('dark', shape=dark_shape, dtype=output_dtype, chunks=(1, dark_shape[1], dark_shape[2]), compression=compressor)
        z_groups['exc'].zeros('bright', shape=dark_shape, dtype=output_dtype, chunks=(1, dark_shape[1], dark_shape[2]), compression=compressor)
    else:
        print("  - Writing actual dark and bright frame data.")
        z_groups['exc'].create_dataset('dark', data=dark_frames_data, dtype=output_dtype, chunks=(1, dark_shape[1], dark_shape[2]), compression=compressor)
        z_groups['exc'].create_dataset('bright', data=dark_frames_data, dtype=output_dtype, chunks=(1, dark_shape[1], dark_shape[2]), compression=compressor)

    pre_proc_val = (dark_mean + int(config['preProcThresh'])) if pre_proc_active else dark_mean

    for i in range(num_files):
        current_nr_str = str(start_nr + i).zfill(len(fNr_orig_str))
        current_fn = config['dataFN'].replace(fNr_orig_str, current_nr_str, 1)
        if not Path(current_fn).exists(): print(f"Warning: File not found, stopping: {current_fn}"); z_data.resize(z_offset, numPxZ, numPxY); break
        print(f"  - Reading file {i+1}/{num_files}: {current_fn}")

        start_frame_in_file = skip_frames if i > 0 else 0

        if file_type == 'ge':
            frames_in_this_file = frames_per_file - start_frame_in_file

            numFrameChunks = int(config.get('numFrameChunks', -1))
            if numFrameChunks == -1: numFrameChunks = frames_in_this_file
            num_chunks_in_file = int(ceil(frames_in_this_file / numFrameChunks))

            # Inner loop to process each chunk within the current file
            for chunk_idx in range(num_chunks_in_file):
                print(f"  - Processing chunk {chunk_idx + 1}/{num_chunks_in_file} in file {i + 1}/{num_files}")
                chunk_start_frame = chunk_idx * numFrameChunks
                chunk_end_frame = min((chunk_idx + 1) * numFrameChunks, frames_in_this_file)
                
                read_start_frame = start_frame_in_file + chunk_start_frame
                frames_to_read = chunk_end_frame - chunk_start_frame
                if frames_to_read <= 0: continue
                
                read_offset_bytes = header_size + (read_start_frame * bytes_per_pixel * numPxY * numPxZ)
                read_count_elements = frames_to_read * numPxY * numPxZ
                data_chunk = np.fromfile(current_fn, dtype=output_dtype, count=read_count_elements, offset=read_offset_bytes).reshape((frames_to_read, numPxZ, numPxY))
                
                if pre_proc_active:
                    data_chunk = apply_correction(data_chunk, dark_mean, pre_proc_val)
                
                z_data[z_offset : z_offset + len(data_chunk)] = data_chunk
                z_offset += len(data_chunk)
        else: # TIFF logic remains simple as it's one frame per file
            data_chunk = np.fromfile(current_fn, dtype=output_dtype, offset=8).reshape((1, numPxZ, numPxY))
            z_data[z_offset : z_offset + len(data_chunk)] = data_chunk
            z_offset += len(data_chunk)

    return output_dtype

def build_config(parser, args):
    """Builds a unified configuration dictionary with correct override priority."""
    # Load parameters from the file first.
    config = parse_parameter_file(args.paramFN)

    # If SkipFrame is not in the parameter file, add it with a default of 0
    if 'SkipFrame' not in config:
        print("Info: 'SkipFrame' not found in parameter file. Defaulting to 0.")
        config['SkipFrame'] = 0

    # If NrFilesPerSweep is not in the parameter file, add it with a default of 1
    if 'NrFilesPerSweep' not in config:
        print("Info: 'NrFilesPerSweep' not found in parameter file. Defaulting to 1.")
        config['NrFilesPerSweep'] = 1

    # Overlay command-line arguments.
    # A command-line arg overrides the param file. A default is used only if the key is not in the param file.
    for key, value in vars(args).items():
        if key not in config or value != parser.get_default(key):
            config[key] = value

    # Handle special dataFN construction logic
    if not args.dataFN or config.get('LayerNr', 1) > 1:
        try:
            layer = int(config['LayerNr'])
            print(config['StartFileNrFirstLayer'])
            fNr = int(config['StartFileNrFirstLayer']) + (layer - 1) * int(config['NrFilesPerSweep'])
            config['dataFN'] = str(Path(config['RawFolder']) / f"{config['FileStem']}_{str(fNr).zfill(int(config['Padding']))}{config['Ext']}")
        except KeyError as e:
            raise KeyError(f"Missing parameter for filename construction: {e}. Provide -dataFN.")
    else:
        config['dataFN'] = args.dataFN

    # Replicate original's implicit SkipFrame calculation
    # If HeadSize is > 8192 and SkipFrame was not explicitly set (is 0),
    # calculate SkipFrame from the excess header size.
    head_size = int(config.get('HeadSize', 0))
    if head_size > 8192 and int(config.get('SkipFrame', 0)) == 0:
        # Determine bytes per pixel from PixelValue, defaulting to 2 (for uint16)
        bytes_per_pixel = 4 if int(config.get('PixelValue', 2)) == 4 else 2
        num_px_y = int(config.get('numPxY', 2048))
        num_px_z = int(config.get('numPxZ', 2048))
        
        # This calculation must be integer division
        derived_skipf = (head_size - 8192) // (bytes_per_pixel * num_px_y * num_px_z)
        
        if derived_skipf > 0:
            print(f"Info: 'HeadSize' is {head_size}. Implicitly setting 'SkipFrame' to {derived_skipf}.")
            config['SkipFrame'] = derived_skipf
            # The original also reset HeadSize for the reader, which we should honor.
            config['HeadSize'] = 8192
            
    return config

# --- Main ---
def main():
    parser = argparse.ArgumentParser(description='Generate Zarr.zip dataset.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-resultFolder', type=str, required=True); 
    parser.add_argument('-paramFN', type=str, required=True)
    parser.add_argument('-dataFN', type=str, default=''); 
    parser.add_argument('-darkFN', type=str, default='')
    parser.add_argument('-dataLoc', type=str, default='exchange/data'); 
    parser.add_argument('-darkLoc', type=str, default='exchange/dark')
    parser.add_argument('-numFrameChunks', type=int, default=-1); 
    parser.add_argument('-preProcThresh', type=int, default=-1)
    parser.add_argument('-numFilesPerScan', type=int, default=1); 
    parser.add_argument('-LayerNr', type=int, default=1)
    parser.add_argument('-numPxY', type=int, default=2048); 
    parser.add_argument('-numPxZ', type=int, default=2048)
    parser.add_argument('-omegaStep', type=float, default=0.0)
    args = parser.parse_args()

    config = build_config(parser, args)

    if not args.dataFN or config.get('LayerNr', 1) > 1:
        outfn_base = Path(args.resultFolder) / f"{config.get('FileStem', 'file')}_{str(config.get('StartFileNrFirstLayer', 0)).zfill(int(config.get('Padding',6)))}"
    else: outfn_base = Path(args.resultFolder) / f"{Path(config['dataFN']).name}.analysis"
    outfn_zip = Path(f"{outfn_base}.MIDAS.zip")

    print(f"Input Data: {config['dataFN']}\nParameter File: {args.paramFN}\nOutput Zarr: {outfn_zip}")
    if outfn_zip.exists(): print(f"Moving existing file to '{outfn_zip}.old'"); shutil.move(str(outfn_zip), str(outfn_zip) + '.old')

    zip_store = zarr.ZipStore(str(outfn_zip), mode='w'); zRoot = zarr.group(store=zip_store, overwrite=True)
    z_groups = create_zarr_structure(zRoot); file_ext = Path(config['dataFN']).suffix.lower()
    output_dtype = None

    try:
        if file_ext in ['.h5', '.hdf5']: output_dtype = process_hdf5_scan(config, z_groups)
        elif file_ext == '.zip':
            print("\n--- Processing Existing Zarr.zip File ---")
            with zarr.open(config['dataFN'], 'r') as zf_in:
                source_data = zf_in['exchange/data']; output_dtype = source_data.dtype
                print(f"Copying Zarr data with shape: {source_data.shape}, dtype: {output_dtype}")
                zarr.copy_all(zf_in['exchange'], z_groups['exc'])
        elif file_ext in ['.tif', '.tiff']: output_dtype = process_multifile_scan('tiff', config, z_groups)
        else: output_dtype = process_multifile_scan('ge', config, z_groups)
    except (FileNotFoundError, KeyError, ValueError, SystemExit) as e:
        print(f"\nFATAL ERROR: {e}"); zip_store.close();
        if outfn_zip.exists(): os.remove(outfn_zip)
        sys.exit(1)

    if 'MaskFN' in config and config['MaskFN']:
        print(f"Loading final mask from {config['MaskFN']}")
        from PIL import Image
        mask_data = np.array(Image.open(config['MaskFN'])).astype(output_dtype)
        z_groups['exc'].create_dataset('mask', data=mask_data.reshape(1, *mask_data.shape), chunks=(1, *mask_data.shape))

    if output_dtype is None: print("Error: Could not determine data type."); sys.exit(1)

    dtype_map = {np.uint16: 'uint16', np.uint32: 'uint32', np.float32: 'float32', np.float64: 'float64'}
    dtype_str = dtype_map.get(np.dtype(output_dtype).type, 'unknown')
    z_groups['sp_pro_meas'].create_dataset('datatype', data=np.bytes_(dtype_str.encode('UTF-8')))
    print(f"\nWritten datatype for C-code: '{dtype_str}'")

    write_analysis_parameters(z_groups, config)
    zip_store.close()

    print("\n--- Zarr File Structure Verification ---");
    with zarr.open(str(outfn_zip), 'r') as zf: print(zf.tree())
    print(f"\nSuccessfully created Zarr file: {outfn_zip}")
    print(f"OutputZipName: {outfn_zip}")

if __name__ == '__main__':
    main()