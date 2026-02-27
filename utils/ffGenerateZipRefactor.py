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
import subprocess
from concurrent.futures import ThreadPoolExecutor

# --- Global Configuration ---
from numcodecs import blosc as _blosc
_blosc.set_nthreads(8)
compressor = Blosc(cname='zstd', clevel=1, shuffle=Blosc.BITSHUFFLE)

# --- Helper Functions ---

def _inspect_recursive(group, path="/"):
    """A recursive helper function to manually walk the Zarr group structure."""
    for name, array in group.arrays():
        full_path = f"{path}{name}"
        print(f"Dataset: {full_path}")
        print(f"  - Shape:   {array.shape}")
        print(f"  - Chunks:  {array.chunks}")
        print(f"  - Dtype:   {array.dtype}")
        compressor_info = array.compressor.get_config() if array.compressor else 'None'
        print(f"  - Compressor: {compressor_info}")
        print("-" * 25)

    for name, subgroup in group.groups():
        _inspect_recursive(subgroup, path=f"{path}{name}/")

def print_zarr_chunk_details(filepath):
    """
    Opens a Zarr archive and prints detailed metadata for each array
    by manually walking the group hierarchy. This is the most robust method.
    
    Args:
        filepath (str or Path): The path to the Zarr.zip file.
    """
    print("\n--- Zarr Array Chunk Details ---")
    try:
        with zarr.open(str(filepath), mode='r') as zf:
            if not isinstance(zf, zarr.hierarchy.Group):
                print("Error: The root of the Zarr archive is not a Group.")
                return
            
            _inspect_recursive(zf)
            
    except Exception as e:
        print(f"An error occurred while inspecting the Zarr archive: {e}")
        print(f"Error Type: {type(e).__name__}")


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
                
                if key not in params:
                    # First time we see this key. Just assign the value.
                    # e.g., params['RingThresh'] = [1, 80]
                    params[key] = final_value
                else:
                    if not isinstance(params[key], list) or not any(isinstance(i, list) for i in params[key]):
                        params[key] = [params[key]]
                    params[key].append(final_value)

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
def _copy_hdf5_group_to_zarr(hf_group, z_group, path_prefix='', exclude_paths=None):
    """Recursively copy all datasets from an HDF5 group to a Zarr group.
    
    Args:
        hf_group: Source h5py Group
        z_group: Destination Zarr Group
        path_prefix: Current path for logging
        exclude_paths: Set of full paths to skip
    """
    if exclude_paths is None:
        exclude_paths = set()
    
    for key in hf_group.keys():
        full_path = f"{path_prefix}/{key}" if path_prefix else key
        
        # Check if this path should be excluded
        if any(full_path.startswith(ep) for ep in exclude_paths):
            continue
        
        item = hf_group[key]
        try:
            if isinstance(item, h5py.Group):
                sub_z = z_group.require_group(key)
                _copy_hdf5_group_to_zarr(item, sub_z, full_path, exclude_paths)
            elif isinstance(item, h5py.Dataset):
                data = item[()]
                if not isinstance(data, np.ndarray):
                    data = np.array([data])
                if key in z_group:
                    del z_group[key]
                z_group.create_dataset(key, data=data)
                print(f"    - Copied: {full_path} (shape={data.shape})")
        except Exception as e:
            print(f"    - Warning: Could not copy '{full_path}': {e}")

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

    sp_pro_analysis, sp_pro_meas = z_groups['sp_pro_analysis'], z_groups['sp_pro_meas']
    print("\nWriting analysis parameters to Zarr file...")

    FORCE_DOUBLE_PARAMS = { "RMin", "RMax", "px", "PixelSize", "Completeness", "MinMatchesToAcceptFrac", "OverArea", "IntensityThresh", "MinS_N", "YPixelSize", "ZPixelSize", "BeamStopY", "BeamStopZ", "DetDist", "MaxDev", "OmegaStart", "OmegaFirstFile", "OmegaStep", "step", "BadPxIntensity", "GapIntensity", "FitWeightMean", "PixelSplittingRBin", "tolTilts", "tolBC", "tolLsd", "DiscArea", "OverlapLength", "ReferenceRingCurrent", "zDiffThresh", "GlobalPosition", "tolPanelFit", "tolP", "tolP0", "tolP1", "tolP2", "tolP3", "tolP4", "tolShifts", "tolRotation", "tolLsdPanel", "tolP2Panel", "DoubletSeparation", "MultFactor", "StepSizePos", "tInt", "tGap", "StepSizeOrient", "MarginRadius", "MarginRadial", "MarginEta", "MarginOme", "MargABG", "MargABC", "OmeBinSize", "EtaBinSize", "RBinSize", "EtaMin", "MinEta", "EtaMax", "X", "Y", "Z", "U", "V", "W", "SHpL", "Polariz", "MaxOmeSpotIDsToIndex", "MinOmeSpotIDsToIndex", "BeamThickness", "Wedge", "Rsample", "Hbeam", "Vsample", "RhoD", "MaxRingRad", "Lsd", "Wavelength", "Width", "WidthTthPx", "UpperBoundThreshold", "p3", "p2", "p1", "p0", "tz", "ty", "tx" }
    FORCE_INT_PARAMS = { "Twins", "MaxNFrames", "DoFit", "DiscModel", "UseMaximaPositions", "UsePixelOverlap", "MaxNrPx", "MinNrPx", "MaxNPeaks", "PhaseNr", "NumPhases", "MinNrSpots", "UseFriedelPairs", "OverallRingToIndex", "SpaceGroup", "LayerNr", "DoFullImage", "SkipFrame", "SumImages", "Normalize", "SaveIndividualFrames", "OmegaSumFrames", "NrFilesPerSweep", "NPanelsY", "NPanelsZ", "Padding", "PanelSizeY", "PanelSizeZ", "PanelGapsY", "PanelGapsZ", "doPeakFit", "nIterations", "NormalizeRingWeights", "OutlierIterations", "WeightByRadius", "WeightByFitSNR", "L2Objective", "DistortionOrder", "PerPanelLsd", "PerPanelDistortion", "FixPanelID", "MinIndicesForFit" }
    FORCE_STRING_PARAMS = { "GapFile", "BadPxFile", "ResultFolder", "PanelShiftsFile", "MaskFile" }
    RENAME_MAP = { "OmegaStep": "step", "Completeness": "MinMatchesToAcceptFrac", "px": "PixelSize", "LatticeConstant": "LatticeParameter", "OverAllRingToIndex": "OverallRingToIndex", "resultFolder": "ResultFolder", "OmegaRange": "OmegaRanges", "BoxSize": "BoxSizes" }

    for key, value in config.items():
        try:
            target_key = RENAME_MAP.get(key, key)
            target_group = sp_pro_analysis
            if key in ["OmegaStep", "start", "datatype", "doPeakFit"]: target_group = sp_pro_meas

            if key == 'BC':
                sp_pro_analysis.create_dataset('YCen', data=np.array([value[0]], dtype=np.double))
                sp_pro_analysis.create_dataset('ZCen', data=np.array([value[1]], dtype=np.double))
            elif key == 'LatticeConstant' or key == 'LatticeParameter':
                values = value if isinstance(value, list) else [value]
                padded_values = np.zeros(6, dtype=np.double)
                padded_values[:len(values)] = values
                target_group.create_dataset(target_key, data=padded_values.astype(np.double))
            elif key == 'ImTransOpt':
                # Ensure the value from the param file is a list
                values_to_write = value if isinstance(value, list) else [value]
                
                # Create a numpy array. Use flatten() to guarantee it is 1D.
                arr = np.array(values_to_write, dtype=np.int32).flatten()
                
                print(f"  - Writing '{target_key}' as a 1D array with shape: {arr.shape}")
                target_group.create_dataset(target_key, data=arr)
            elif key in ['RingThresh', 'RingsToExclude', 'OmegaRange', 'BoxSize']:
                temp_value = value if isinstance(value, list) else [value]
                if not temp_value or not isinstance(temp_value[0], list):
                    values_to_write = [temp_value]
                else:
                    values_to_write = temp_value
                arr = np.array(values_to_write)
                print(key,target_key,arr,arr.ndim)
                if arr.ndim == 1:
                    print(f"  - Info: Reshaping 1D array for '{target_key}' to ensure 2D shape.")
                    arr = arr.reshape(1, -1)
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

class BZ2Context:
    """
    Context manager to handle .bz2 files.
    - If file ends in .bz2: Decompresses it (keeping original), yields temp path, deletes temp path on exit.
    - If file is normal: Yields original path, does nothing on exit.
    """
    def __init__(self, filepath):
        self.filepath = str(filepath)
        self.temp_path = None
        self.is_bz2 = self.filepath.endswith('.bz2')

    def __enter__(self):
        if not self.is_bz2:
            return self.filepath
        
        # Define expected uncompressed filename (remove .bz2)
        self.temp_path = self.filepath[:-4]
        
        # Call shell command: bzip2 -d (decompress) -k (keep original) -f (force overwrite output)
        try:
            # We use check=True to raise an error if bzip2 fails
            subprocess.run(['bzip2', '-d', '-k', '-f', self.filepath], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error uncompressing {self.filepath}: {e}")
            sys.exit(1)
            
        return self.temp_path

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Delete the uncompressed file after processing
        if self.is_bz2 and self.temp_path and os.path.exists(self.temp_path):
            os.remove(self.temp_path)

def process_hdf5_scan(config, z_groups, zRoot):
    """Processes a scan from one or more HDF5 files, with correct SkipFrame logic and .bz2 support."""
    print("\n--- Processing HDF5 File(s) ---")
    num_files, skip_frames = int(config.get('numFilesPerScan', 1)), int(config.get('SkipFrame', 0))
    data_loc, dark_loc = config['dataLoc'], config['darkLoc']
    pre_proc_active = int(config.get('preProcThresh', -1)) != -1

    file_list = []
    # Handle filename parsing for sequences. If .bz2, strip it temporarily to find the number.
    is_bz2 = config['dataFN'].endswith('.bz2')
    clean_fn_for_parsing = config['dataFN'][:-4] if is_bz2 else config['dataFN']

    if num_files == 1: 
        file_list.append(config['dataFN'])
    else:
        p_parsing = Path(clean_fn_for_parsing)
        filename_stem = p_parsing.stem
        final_ext = p_parsing.suffix
        
        # Find the LAST group of digits (the sequence number)
        all_digit_matches = list(re.finditer(r'\d+', filename_stem))
        if not all_digit_matches:
            raise ValueError(f"Numeric sequence not found in HDF5 filename: '{filename_stem}'")

        last_match = all_digit_matches[-1]
        fNr_orig_str = last_match.group()
        start_nr = int(fNr_orig_str)
        
        filename_base = filename_stem[:last_match.start()]
        filename_mid_ext = filename_stem[last_match.end():]
        
        bz2_suffix = ".bz2" if is_bz2 else ""
        parent_dir = Path(config['dataFN']).parent
        
        for i in range(num_files):
            current_nr_str = str(start_nr + i).zfill(len(fNr_orig_str))
            # Correct reconstruction: Base + Number + Mid-Ext + .h5 + .bz2
            current_fn = parent_dir / f"{filename_base}{current_nr_str}{filename_mid_ext}{final_ext}{bz2_suffix}"
            file_list.append(str(current_fn))

    # Open first file to check metadata (uncompressing if needed)
    with BZ2Context(file_list[0]) as first_fn:
        with h5py.File(first_fn, 'r') as hf:
            if data_loc not in hf: raise KeyError(f"Data location '{data_loc}' not found in HDF5 file.")
            frames_per_file, nZ, nY = hf[data_loc].shape
            output_dtype = hf[data_loc].dtype

            # Update config with actual dimensions found in file
            config['numPxY'] = nY
            config['numPxZ'] = nZ
            print(f"  - Inferred dimensions from HDF5: {nY} x {nZ}")

            print("  - Checking for additional metadata inside HDF5 file...")
            # Allow user to override paths in param file
            path_pressure = config.get('PressureDatasetPath', '/measurement/instrument/GSAS2_PVS/Pressure')
            path_temp = config.get('TemperatureDatasetPath', '/measurement/instrument/GSAS2_PVS/Temperature')
            path_I = config.get('IDatasetPath', '/measurement/instrument/GSAS2_PVS/I')
            path_I0 = config.get('I0DatasetPath', '/measurement/instrument/GSAS2_PVS/I0')

            hdf5_metadata_paths = {
                'startOmeOverride': 'startOmeOverride',
                path_pressure: 'Pressure',
                path_temp: 'Temperature',
                path_I: 'I',
                path_I0: 'I0',
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

            # Copy instrument/ group (all keys recursively)
            if 'instrument' in hf:
                print("  - Copying instrument/ group from HDF5...")
                inst_z = zRoot.require_group('instrument')
                _copy_hdf5_group_to_zarr(hf['instrument'], inst_z, 'instrument')

            # Copy measurement/ group (excluding scan_parameters which are handled above)
            if 'measurement' in hf:
                print("  - Copying measurement/ group from HDF5...")
                _copy_hdf5_group_to_zarr(
                    hf['measurement'], z_groups['meas'], 'measurement',
                    exclude_paths={'measurement/process/scan_parameters'}
                )

    total_frames_to_write = frames_per_file + (frames_per_file - skip_frames) * (num_files - 1)
    print(f"HDF5 scan: {num_files} file(s), {frames_per_file} frames/file. Skipping {skip_frames} from files 2+. Total frames to write: {total_frames_to_write}. Dtype: {output_dtype}")

    z_data = z_groups['exc'].create_dataset('data', shape=(total_frames_to_write, nZ, nY), dtype=output_dtype, chunks=(1, nZ, nY), compression=compressor)
    z_offset = 0

    for i, fn in enumerate(file_list):
        print(f"  - Processing file {i+1}/{num_files}: {fn}")
        
        # Use BZ2Context here to unzip, read, and delete temp file automatically
        with BZ2Context(fn) as uncompressed_fn:
            with h5py.File(uncompressed_fn, 'r') as hf:
                if data_loc not in hf: print(f"    Warning: '{data_loc}' not in {fn}, skipping."); continue

                # Logic for the first file (Dark/Bright init)
                if i == 0:
                    dark_frames_found = False
                    dark_frames = None

                    # 1. Check for separate dark file (handle .bz2 there too)
                    if config.get('darkFN'):
                        with BZ2Context(config['darkFN']) as uncompressed_dark:
                            with h5py.File(uncompressed_dark, 'r') as hf_dark:
                                if dark_loc in hf_dark:
                                    dark_frames = hf_dark[dark_loc][()]
                                    dark_frames_found = True
                                    print(f"Dark data was found in {dark_loc} in {config['darkFN']}.")
                                else:
                                    dark_dataset_name = Path(dark_loc).name
                                    if dark_dataset_name in hf_dark:
                                        dark_frames = hf_dark[dark_dataset_name][()]
                                        dark_frames_found = True
                    # 2. Check internal
                    elif dark_loc in hf:
                        dark_frames = hf[dark_loc][()][()]
                        # dark_frames = hf[dark_loc][()]
                        dark_frames_found = True
                        print(f"Dark data was found in {dark_loc} in {config['dataFN']}.")

                    if not dark_frames_found:
                        print("    Warning: No dark data found. Using temporary zeros for shape info.")
                        dark_frames = np.zeros((10, nZ, nY), dtype=output_dtype)

                    dark_shape = dark_frames.shape
                    print(f'dark_shape: {dark_shape}')
                    if pre_proc_active:
                        print("  - Pre-processing is active. Writing zero arrays for dark/bright.")
                        z_groups['exc'].create_dataset('dark', data=np.zeros(dark_frames.shape, dtype=output_dtype), chunks=(1, dark_shape[1], dark_shape[2]), compression=compressor)
                        z_groups['exc'].create_dataset('bright', data=np.zeros(dark_frames.shape, dtype=output_dtype), chunks=(1, dark_shape[1], dark_shape[2]), compression=compressor)
                    else:
                        print("  - Writing actual dark and bright frame data.")
                        z_groups['exc'].create_dataset('dark', data=dark_frames, dtype=output_dtype,chunks=(1, dark_shape[1], dark_shape[2]), compression=compressor)
                        z_groups['exc'].create_dataset('bright', data=dark_frames, dtype=output_dtype,chunks=(1, dark_shape[1], dark_shape[2]), compression=compressor)

                if dark_shape[0] > skip_frames:
                    dark_mean = np.mean(dark_frames[skip_frames:], axis=0)
                else:
                    dark_mean = dark_frames[0]
                pre_proc_val = (dark_mean + int(config['preProcThresh'])) if pre_proc_active else dark_mean
                print(f'Mean value of pre_proc_val: {np.mean(pre_proc_val)}, darkMean: {np.mean(dark_mean)}')

                start_frame_in_file = skip_frames if i > 0 else 0

                chunk_size = int(config.get('numFrameChunks', 100));
                if chunk_size == -1: chunk_size = frames_per_file

                # Pipeline: overlap HDF5 reads with zarr writes
                import time as _time
                t_start_pipeline = _time.time()
                frames_written_this_file = 0
                total_frames_this_file = frames_per_file - start_frame_in_file
                j = start_frame_in_file
                # Initial read
                end_frame = min(j + chunk_size, frames_per_file)
                data_chunk = hf[data_loc][j:end_frame]
                if pre_proc_active:
                    data_chunk = apply_correction(data_chunk, dark_mean, pre_proc_val)
                j += chunk_size

                with ThreadPoolExecutor(max_workers=1) as prefetcher:
                    while True:
                        # Submit prefetch for the next chunk (if any)
                        if j < frames_per_file:
                            next_end = min(j + chunk_size, frames_per_file)
                            future = prefetcher.submit(lambda s, e: hf[data_loc][s:e], j, next_end)
                        else:
                            future = None

                        # Write current chunk to zarr (overlaps with HDF5 read above)
                        z_data[z_offset : z_offset + len(data_chunk)] = data_chunk
                        z_offset += len(data_chunk)
                        frames_written_this_file += len(data_chunk)

                        # Progress report
                        elapsed = _time.time() - t_start_pipeline
                        bytes_done = frames_written_this_file * nZ * nY * np.dtype(output_dtype).itemsize
                        throughput = bytes_done / elapsed / 1e6 if elapsed > 0 else 0
                        print(f"    Progress: {frames_written_this_file}/{total_frames_this_file} frames "
                              f"({elapsed:.1f}s, {throughput:.0f} MB/s)", flush=True)

                        if future is None:
                            break

                        # Get prefetched data for next iteration
                        data_chunk = future.result()
                        if pre_proc_active:
                            data_chunk = apply_correction(data_chunk, dark_mean, pre_proc_val)
                        j += chunk_size

                total_elapsed = _time.time() - t_start_pipeline
                total_bytes = total_frames_this_file * nZ * nY * np.dtype(output_dtype).itemsize
                print(f"    File {i+1} done: {total_frames_this_file} frames in {total_elapsed:.1f}s "
                      f"({total_bytes/total_elapsed/1e6:.0f} MB/s)", flush=True)
    return output_dtype

def process_multifile_scan(file_type, config, z_groups):
    """Processes a scan of multiple GE/TIFF files with SkipFrame logic and .bz2 support."""
    print(f"\n--- Processing {file_type.upper()} File(s) ---")
    num_files, skip_frames = int(config.get('numFilesPerScan', 1)), int(config.get('SkipFrame', 0))
    header_size = int(config.get('HeadSize', 8192))
    # numPxY, numPxZ will be determined/verified from the file if possible (for TIFF) or config (for GE)

    # Ensure tifffile is available if needed (for both data and dark files)
    if file_type != 'ge':
        import tifffile

    # --- Correction Logic ---
    dark_file_path = config.get('darkFN')
    dark_file_provided = dark_file_path and Path(dark_file_path).exists()
    pre_proc_thresh_value = int(config.get('preProcThresh', -1))
    correction_active = dark_file_provided or pre_proc_thresh_value != -1
    if correction_active:
        print("Info: Dark correction/processing is active.")

    output_dtype = np.uint32 if int(config.get('PixelValue', 2 if file_type=='ge' else 4)) == 4 else np.uint16
    bytes_per_pixel = np.dtype(output_dtype).itemsize
    print(f"Handling as {output_dtype.__name__}. Files per scan: {num_files}")

    # 1. Parsing filename. 
    original_path = Path(config['dataFN'])
    is_bz2_sequence = original_path.name.endswith('.bz2')
    path_for_parsing = original_path.with_suffix('') if is_bz2_sequence else original_path
    
    filename_stem = path_for_parsing.stem 
    
    # Find the LAST group of digits
    all_digit_matches = list(re.finditer(r'\d+', filename_stem))
    if not all_digit_matches:
        raise ValueError(f"Could not find a numeric sequence in the filename stem: '{filename_stem}'")

    last_match = all_digit_matches[-1]
    fNr_orig_str = last_match.group()
    start_nr = int(fNr_orig_str)
    
    # Identify parts: e.g. "park_0MPa_" + "000294" + ".edf"
    filename_base = filename_stem[:last_match.start()]
    filename_mid_ext = filename_stem[last_match.end():]
    
    parent_dir = original_path.parent
    raw_ext = path_for_parsing.suffix 
    full_ext = raw_ext + ".bz2" if is_bz2_sequence else raw_ext

    # 2. Pre-scan to find all existing files
    file_list = []
    for i in range(num_files):
        current_nr_str = str(start_nr + i).zfill(len(fNr_orig_str))
        # Reconstruct exactly: Base + Number + MiddleExt (.edf) + FinalExt (.ge5)
        current_fn = parent_dir / f"{filename_base}{current_nr_str}{filename_mid_ext}{full_ext}"

        if current_fn.exists():
            file_list.append(str(current_fn))
        else:
            print(f"Info: File sequence ended. Could not find {current_fn}")
            break
    
    if not file_list:
        raise FileNotFoundError(f"No data files found for the sequence starting with {config['dataFN']}")

    # Check size of first file
    with BZ2Context(file_list[0]) as first_fn_uncompressed:
        if file_type == 'ge':
            # For GE files, we MUST have valid numPxY/numPxZ from config/defaults as they have no header info
            numPxY = int(config.get('numPxY', 2048))
            numPxZ = int(config.get('numPxZ', 2048))
            
            frames_per_file = (os.path.getsize(first_fn_uncompressed) - header_size) // (bytes_per_pixel * numPxY * numPxZ)
        else: # tiff
            # For TIFF, we ignore config dims and read from file (overwriting config)
            test_img = tifffile.imread(first_fn_uncompressed)
            if test_img.ndim == 2:
                numPxZ, numPxY = test_img.shape
                frames_per_file = 1
            elif test_img.ndim == 3:
                frames_per_file, numPxZ, numPxY = test_img.shape
            else:
                 # This might happen if it's 4D or something else, default to last 2 dims
                 numPxZ, numPxY = test_img.shape[-2:]
                 frames_per_file = 1
                 if test_img.ndim > 2:
                    frames_per_file = test_img.shape[0]

            config['numPxY'] = numPxY
            config['numPxZ'] = numPxZ
            print(f"  - Inferred dimensions from TIFF: {numPxY} x {numPxZ}")

    num_files_found = len(file_list)
    total_frames_to_write = frames_per_file
    if num_files_found > 1:
        total_frames_to_write += (frames_per_file - skip_frames) * (num_files_found - 1)

    print(f"Scan: {num_files_found} file(s) found, {frames_per_file} frames/file. Skipping {skip_frames} from files 2+. Total frames to write: {total_frames_to_write}.")

    z_data = z_groups['exc'].create_dataset('data', shape=(total_frames_to_write, numPxZ, numPxY), dtype=output_dtype, chunks=(1, numPxZ, numPxY), compression=compressor)
    z_offset = 0

    dark_mean = np.zeros((numPxZ, numPxY), dtype=output_dtype)
    dark_shape = (1, numPxZ, numPxY) # Default
    
    # --- Handle Dark File (supports .bz2 for GE and TIFF) ---
    if dark_file_provided:
        print(f"  - Reading Dark file: {config['darkFN']}")
        with BZ2Context(config['darkFN']) as dark_fn_uncompressed:
            if file_type == 'ge':
                # GE Logic: Raw binary read
                dark_frames_data = np.fromfile(dark_fn_uncompressed, dtype=output_dtype, offset=header_size).reshape((-1, numPxZ, numPxY))
                dark_mean = np.mean(dark_frames_data[skip_frames:], axis=0)
                dark_shape = dark_frames_data.shape
            else:
                # TIFF Logic: Use tifffile
                dark_frames_data = tifffile.imread(dark_fn_uncompressed)
                # Handle cases where TIFF might be 2D (single frame) or 3D (stack)
                if dark_frames_data.ndim == 2:
                    # Single frame, reshape to (1, Y, X) for consistency in shape logic
                    dark_frames_data = dark_frames_data.reshape(1, *dark_frames_data.shape)
                    dark_mean = dark_frames_data[0]
                else:
                    # Multi-frame stack, apply skip logic
                    dark_mean = np.mean(dark_frames_data[skip_frames:], axis=0)
                
                dark_shape = dark_frames_data.shape
                # Ensure dtype matches output
                dark_mean = dark_mean.astype(output_dtype)

    if pre_proc_thresh_value != -1:
        print("  - Pre-processing threshold is set. Writing zero arrays for dark/bright.")
        z_groups['exc'].create_dataset('dark', data=np.zeros(dark_shape, dtype=output_dtype), chunks=(1, dark_shape[1], dark_shape[2]), compression=compressor)
        z_groups['exc'].create_dataset('bright', data=np.zeros(dark_shape, dtype=output_dtype), chunks=(1, dark_shape[1], dark_shape[2]), compression=compressor)
    else:
        # If we have real dark data (from GE or TIFF), write it. Otherwise it writes zeros.
        print("  - Writing actual dark and bright frame data.")
        # If dark_frames_data wasn't loaded (no file provided), we need a placeholder
        if not dark_file_provided:
             dark_frames_data = np.zeros(dark_shape, dtype=output_dtype)

        z_groups['exc'].create_dataset('dark', data=dark_frames_data, dtype=output_dtype, chunks=(1, dark_shape[1], dark_shape[2]), compression=compressor)
        z_groups['exc'].create_dataset('bright', data=dark_frames_data, dtype=output_dtype, chunks=(1, dark_shape[1], dark_shape[2]), compression=compressor)

    pre_proc_threshold = dark_mean + (pre_proc_thresh_value if pre_proc_thresh_value != -1 else 0)

    if file_type == 'ge':
        for i, current_fn in enumerate(file_list):
            print(f"  - Reading file {i+1}/{num_files_found}: {current_fn}")
            
            with BZ2Context(current_fn) as uncompressed_fn:
                start_frame_in_file = skip_frames if i > 0 else 0
                frames_in_this_file = (os.path.getsize(uncompressed_fn) - header_size) // (bytes_per_pixel * numPxY * numPxZ)
                readable_frames = frames_in_this_file - start_frame_in_file

                numFrameChunks = int(config.get('numFrameChunks', -1))
                if numFrameChunks == -1: numFrameChunks = readable_frames
                if numFrameChunks <= 0: continue
                
                num_chunks_in_file = int(ceil(readable_frames / numFrameChunks))

                for chunk_idx in range(num_chunks_in_file):
                    chunk_start_frame = chunk_idx * numFrameChunks
                    frames_to_read = min(numFrameChunks, readable_frames - chunk_start_frame)
                    if frames_to_read <= 0: continue
                    
                    read_start_offset_in_file = start_frame_in_file + chunk_start_frame
                    read_offset_bytes = header_size + (read_start_offset_in_file * bytes_per_pixel * numPxY * numPxZ)
                    
                    data_chunk = np.fromfile(uncompressed_fn, dtype=output_dtype, count=frames_to_read * numPxY * numPxZ, offset=read_offset_bytes).reshape((frames_to_read, numPxZ, numPxY))
                    
                    if correction_active:
                        data_chunk = apply_correction(data_chunk, dark_mean, pre_proc_threshold)
                    
                    z_data[z_offset : z_offset + len(data_chunk)] = data_chunk
                    z_offset += len(data_chunk)
    else: # TIFF reading in parallel
        import concurrent.futures
        
        def process_single_tiff(filepath):
            """Reads, corrects, and returns data for one TIFF file. Handles .bz2 automatically."""
            with BZ2Context(filepath) as uncompressed_path:
                data_chunk = tifffile.imread(uncompressed_path).reshape((1, numPxZ, numPxY))
                if correction_active:
                    return apply_correction(data_chunk, dark_mean, pre_proc_threshold)
                return data_chunk

        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = executor.map(process_single_tiff, file_list)

            for processed_data in results:
                z_data[z_offset : z_offset + len(processed_data)] = processed_data
                z_offset += len(processed_data)
                print(f"  - Wrote frame {z_offset}/{total_frames_to_write} to Zarr archive.")

    return output_dtype
    
def build_config(parser, args):
    """Builds a unified configuration dictionary with correct override priority."""
    # Load parameters from the file first.
    config = parse_parameter_file(args.paramFN)

    # Manually map the 'Dark' key from the param file to 'darkFN' for consistency.
    # The command-line argument '-darkFN' will still override this later if provided.
    if 'Dark' in config and not config.get('darkFN'):
        print(f"Info: Found 'Dark' key in parameter file: {config['Dark']}")
        config['darkFN'] = config['Dark']

    # --- Dimension Mapping ---
    # Map NrPixels family to numPxY/numPxZ if they exist in the parameter file
    # Priority: numPxY/Z (args/explicit) > numPxY/Z (param) > NrPixelsY/Z (param) > NrPixels (param)
    
    if 'NrPixels' in config:
        val = int(config['NrPixels'])
        if 'numPxY' not in config: config['numPxY'] = val
        if 'numPxZ' not in config: config['numPxZ'] = val
    
    if 'NrPixelsY' in config:
        config['numPxY'] = int(config['NrPixelsY'])
        
    if 'NrPixelsZ' in config:
        config['numPxZ'] = int(config['NrPixelsZ'])


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
        # Use the actual file number for this layer, not StartFileNrFirstLayer
        layer = int(config.get('LayerNr', 1))
        actual_fNr = int(config.get('StartFileNrFirstLayer', 0)) + (layer - 1) * int(config.get('NrFilesPerSweep', 1))
        outfn_base = Path(args.resultFolder) / f"{config.get('FileStem', 'file')}_{str(actual_fNr).zfill(int(config.get('Padding',6)))}"
    else: outfn_base = Path(args.resultFolder) / f"{Path(config['dataFN']).name}.analysis"
    outfn_zip = Path(f"{outfn_base}.MIDAS.zip")

    print(f"Input Data: {config['dataFN']}\nParameter File: {args.paramFN}\nOutput Zarr: {outfn_zip}")
    if outfn_zip.exists(): print(f"Moving existing file to '{outfn_zip}.old'"); shutil.move(str(outfn_zip), str(outfn_zip) + '.old')

    zip_store = zarr.ZipStore(str(outfn_zip), mode='w'); zRoot = zarr.group(store=zip_store, overwrite=True)
    z_groups = create_zarr_structure(zRoot); 
    # Check the extension. If it is .bz2, look at the extension BEFORE it (e.g., .h5.bz2 -> .h5)
    data_path = Path(config['dataFN'])
    file_ext = data_path.suffix.lower()
    
    if file_ext == '.bz2':
        # path.stem removes .bz2, so we get the suffix of the remaining name
        file_ext = Path(data_path.stem).suffix.lower()

    # file_ext = Path(config['dataFN']).suffix.lower()
    output_dtype = None

    try:
        if file_ext in ['.h5', '.hdf5']: output_dtype = process_hdf5_scan(config, z_groups, zRoot)
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
    with zarr.open(str(outfn_zip), 'r') as zf: 
        print(zf.tree())
    print_zarr_chunk_details(outfn_zip)
    print(f"\nSuccessfully created Zarr file: {outfn_zip}")
    print(f"OutputZipName: {outfn_zip}")

if __name__ == '__main__':
    main()
