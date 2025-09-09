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
                parts = line.strip().split()
                if not parts: continue
                key, values = parts[0], parts[1:]
                processed_values = []
                for v in values:
                    if re.match(r"^-?\d+$", v): processed_values.append(int(v))
                    elif re.match(r"^-?\d*\.\d+$", v) or re.match(r"^-?\d+\.\d*$", v): processed_values.append(float(v))
                    else: processed_values.append(v)
                final_value = processed_values if len(processed_values) > 1 else processed_values[0]
                if key not in params: params[key] = []
                params[key].append(final_value)
    except FileNotFoundError:
        print(f"Error: Parameter file not found at '{filename}'")
        sys.exit(1)
    for key, val in params.items():
        if len(val) == 1: params[key] = val[0]
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
    """Writes parameters from config dict to Zarr, handling all special cases."""
    sp_pro_analysis, sp_pro_meas = z_groups['sp_pro_analysis'], z_groups['sp_pro_meas']
    print("\nWriting analysis parameters to Zarr file...")
    RENAME_MAP = { "OmegaStep": "step", "Completeness": "MinMatchesToAcceptFrac", "px": "PixelSize",
                   "LatticeConstant": "LatticeParameter", "OverallRingToIndex": "OverallRingToIndex" }
    for key, value in config.items():
        try:
            target_key, target_group = RENAME_MAP.get(key, key), sp_pro_analysis
            if key in ["OmegaStep", "start", "datatype"]: target_group = sp_pro_meas
            if key == 'BC':
                values = value if isinstance(value, list) else [value]
                sp_pro_analysis.create_dataset('YCen', data=np.array([values[0]], dtype=np.double))
                sp_pro_analysis.create_dataset('ZCen', data=np.array([values[1]], dtype=np.double))
            elif key in ['RingThresh', 'RingsToExclude', 'OmegaRanges', 'BoxSizes', 'ImTransOpt']:
                arr, dtype = np.array(value), np.int32 if key == 'ImTransOpt' else np.double
                target_group.create_dataset(target_key, data=arr.astype(dtype))
            else:
                if isinstance(value, list): arr = np.array(value)
                elif isinstance(value, int): arr = np.array([value], dtype=np.int32)
                elif isinstance(value, float): arr = np.array([value], dtype=np.double)
                else: arr = np.bytes_(str(value).encode('UTF-8'))
                target_group.create_dataset(target_key, data=arr)
        except Exception as e:
            print(f"  - Warning: Could not write parameter '{key}'. Reason: {e}")
    ome_ff = float(config.get('OmegaStart', config.get('OmegaFirstFile', 0)))
    ome_stp = float(config.get('OmegaStep', 0))
    skip_f = int(config.get('SkipFrame', 0))
    start_omega = ome_ff - (skip_f * ome_stp)
    sp_pro_meas.create_dataset('start', data=np.array([start_omega], dtype=np.double))

def process_hdf5_scan(config, z_groups):
    """Processes a scan from one or more HDF5 files, with correct SkipFrame logic."""
    print("\n--- Processing HDF5 File(s) ---")
    num_files, skip_frames = int(config.get('numFilesPerScan', 1)), int(config['SkipFrame'])
    data_loc, dark_loc = config['dataLoc'], config['darkLoc']
    
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
    
    total_frames_to_write = frames_per_file + (frames_per_file - skip_frames) * (num_files - 1)
    print(f"HDF5 scan: {num_files} file(s), {frames_per_file} frames/file. Skipping {skip_frames} from files 2+. Total frames to write: {total_frames_to_write}. Dtype: {output_dtype}")

    z_data = z_groups['exc'].create_dataset('data', shape=(total_frames_to_write, nZ, nY), dtype=output_dtype, chunks=(1, nZ, nY), compression=compressor)
    z_offset = 0

    for i, fn in enumerate(file_list):
        print(f"  - Processing file {i+1}/{num_files}: {fn}")
        with h5py.File(fn, 'r') as hf:
            if data_loc not in hf: print(f"    Warning: '{data_loc}' not in {fn}, skipping."); continue
            
            if i == 0:
                if dark_loc in hf: dark_frames = hf[dark_loc][()]
                elif config['darkFN']:
                    with h5py.File(config['darkFN'], 'r') as hf_dark: dark_frames = hf_dark[dark_loc][()]
                else: print("    Warning: No dark data. Using zeros."); dark_frames = np.zeros((10, nZ, nY), dtype=output_dtype)
                z_groups['exc'].create_dataset('dark', data=dark_frames, dtype=output_dtype)
                z_groups['exc'].create_dataset('bright', data=dark_frames, dtype=output_dtype) # Assuming bright is same as dark
            
            dark_mean = np.mean(dark_frames[skip_frames:], axis=0)
            pre_proc_val = (dark_mean + int(config['preProcThresh'])) if int(config['preProcThresh']) != -1 else dark_mean
            
            start_frame_in_file = skip_frames if i > 0 else 0
            
            chunk_size = int(config.get('numFrameChunks', 100));
            if chunk_size == -1: chunk_size = frames_per_file

            for j in range(start_frame_in_file, frames_per_file, chunk_size):
                end_frame = min(j + chunk_size, frames_per_file)
                data_chunk = hf[data_loc][j:end_frame]
                if int(config['preProcThresh']) != -1:
                    data_chunk = apply_correction(data_chunk, dark_mean, pre_proc_val)
                z_data[z_offset : z_offset+len(data_chunk)] = data_chunk
                z_offset += len(data_chunk)
    return output_dtype

def process_multifile_scan(file_type, config, z_groups):
    """Processes a scan of multiple GE/TIFF files with correct SkipFrame logic."""
    print(f"\n--- Processing {file_type.upper()} File(s) ---")
    num_files, skip_frames = int(config.get('numFilesPerScan', 1)), int(config['SkipFrame'])
    header_size = int(config.get('HeadSize', 8192))
    numPxY, numPxZ = int(config['numPxY']), int(config['numPxZ'])
    
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
    
    z_groups['exc'].create_dataset('dark', data=dark_frames_data, dtype=output_dtype)
    z_groups['exc'].create_dataset('bright', data=dark_frames_data, dtype=output_dtype)

    pre_proc_val = (dark_mean + int(config['preProcThresh'])) if int(config['preProcThresh']) != -1 else dark_mean
    
    for i in range(num_files):
        current_nr_str = str(start_nr + i).zfill(len(fNr_orig_str))
        current_fn = config['dataFN'].replace(fNr_orig_str, current_nr_str, 1)
        if not Path(current_fn).exists(): print(f"Warning: File not found, stopping: {current_fn}"); z_data.resize(z_offset, numPxZ, numPxY); break
        print(f"  - Reading file {i+1}/{num_files}: {current_fn}")
        
        start_frame_in_file = skip_frames if i > 0 else 0
        
        if file_type == 'ge':
            offset = header_size + (start_frame_in_file * bytes_per_pixel * numPxY * numPxZ)
            frames_to_read = (os.path.getsize(current_fn) - offset) // (bytes_per_pixel * numPxY * numPxZ)
            data_chunk = np.fromfile(current_fn, dtype=output_dtype, offset=offset).reshape((frames_to_read, numPxZ, numPxY))
        else: # TIFF
            data_chunk = np.fromfile(current_fn, dtype=output_dtype, offset=8).reshape((1, numPxZ, numPxY))
        
        if int(config['preProcThresh']) != -1 and file_type == 'ge':
            data_chunk = apply_correction(data_chunk, dark_mean, pre_proc_val)
            
        z_data[z_offset : z_offset + len(data_chunk)] = data_chunk
        z_offset += len(data_chunk)
    return output_dtype
    
def build_config(parser, args):
    """Builds a unified configuration dictionary with correct override priority."""
    config = parse_parameter_file(args.paramFN)
    # If SkipFrame is not in the parameter file, add it with a default of 0
    if 'SkipFrame' not in config:
        print("Info: 'SkipFrame' not found in parameter file. Defaulting to 0.")
        config['SkipFrame'] = 0
    for key, value in vars(args).items():
        if key not in config or value != parser.get_default(key): 
            config[key] = value
    if not args.dataFN or config.get('LayerNr', 1) > 1:
        try:
            layer = int(config['LayerNr']); 
            fNr = int(config['StartFileNrFirstLayer']) + (layer - 1) * int(config['NrFilesPerSweep'])
            config['dataFN'] = str(Path(config['RawFolder']) / f"{config['FileStem']}_{str(fNr).zfill(int(config['Padding']))}{config['Ext']}")
        except KeyError as e: 
            raise KeyError(f"Missing parameter for filename construction: {e}. Provide -dataFN.")
    else: 
        config['dataFN'] = args.dataFN
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
    if outfn_zip.exists(): print(f"Moving existing file to '{outfn_zip}.old'"); shutil.move(outfn_zip, str(outfn_zip) + '.old')

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

    if int(config.get('preProcThresh', -1)) != -1:
        print("Pre-processing was active. Overwriting dark/bright arrays in Zarr with zeros.")
        for key in ['dark', 'bright']:
            if key in z_groups['exc']:
                shape = z_groups['exc'][key].shape; del z_groups['exc'][key]
                z_groups['exc'].zeros(key, shape=shape, dtype=output_dtype)

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

if __name__ == '__main__':
    main()