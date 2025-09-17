#!/usr/bin/env python3
"""
bin2hdf5.py - Convert binary output files from IntegratorFitPeaksGPUStream to HDF5

This script reads the binary output files produced by the CUDA integrator 
(lineout.bin, and optionally fit.bin and Int2D.bin) and organizes them into a 
single HDF5 file with appropriate groups and datasets.

The order of datasets and the grouping for summed frames is determined by the
sorted filenames in the provided mapping file.
"""

import numpy as np
import h5py
import os
import argparse
import json
import datetime
from collections import defaultdict

def read_binary_file(filename, dtype=np.float64, offset=0, count=None):
    """
    Read a binary file as a numpy array.
    
    Args:
        filename: Path to the binary file
        dtype: Data type of the binary file
        offset: Number of elements to skip at the beginning
        count: Number of elements to read (None for all)
        
    Returns:
        Numpy array of the binary data
    """
    if not os.path.exists(filename):
        print(f"Warning: File {filename} does not exist")
        return None
    
    data = np.fromfile(filename, dtype=dtype)
    
    if offset > 0:
        if offset >= len(data):
            print(f"Error: Offset {offset} exceeds file size {len(data)}")
            return None
        data = data[offset:]
    
    if count is not None and count < len(data):
        data = data[:count]
    
    return data

def get_frame_dimensions(params_file):
    """
    Extract dimensions from the parameter file used by the integrator.
    
    Args:
        params_file: Path to the parameter file
        
    Returns:
        Tuple (nRBins, nEtaBins, OSF) with the number of R and Eta bins,
        and the OmegaSumFrames value (or None if not specified)
    """
    # Default values
    RMax, RMin, RBinSize = 100, 10, 0.1
    EtaMax, EtaMin, EtaBinSize = 180, -180, 0.1
    OSF = None  # OmegaSumFrames value
    
    with open(params_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('RMax '):
                RMax = float(line.split()[1])
            elif line.startswith('RMin '):
                RMin = float(line.split()[1])
            elif line.startswith('RBinSize '):
                RBinSize = float(line.split()[1])
            elif line.startswith('EtaMax '):
                EtaMax = float(line.split()[1])
            elif line.startswith('EtaMin '):
                EtaMin = float(line.split()[1])
            elif line.startswith('EtaBinSize '):
                EtaBinSize = float(line.split()[1])
            elif line.startswith('OmegaSumFrames '):
                OSF = int(line.split()[1])
    
    nRBins = int(np.ceil((RMax - RMin) / RBinSize))
    nEtaBins = int(np.ceil((EtaMax - EtaMin) / EtaBinSize))
    
    return nRBins, nEtaBins, OSF

def reshape_lineout_data(lineout_data, nRBins):
    """
    Reshape lineout data into frames.
    
    Args:
        lineout_data: Raw lineout data array
        nRBins: Number of R bins
        
    Returns:
        Tuple (reshaped_data, num_frames) with the reshaped data and number of frames
    """
    if lineout_data is None:
        return None, 0
    
    frame_size = nRBins * 2  # Each frame has nRBins points with R and intensity
    num_frames = len(lineout_data) // frame_size
    
    if num_frames * frame_size != len(lineout_data):
        print(f"Warning: Lineout data size ({len(lineout_data)}) is not an exact multiple of frame size ({frame_size})")
        # Truncate to complete frames
        lineout_data = lineout_data[:num_frames * frame_size]
    
    return lineout_data.reshape(num_frames, nRBins, 2), num_frames

def reshape_fit_data(fit_data, num_frames):
    """
    Reshape fit data into frames.
    
    Args:
        fit_data: Raw fit data array
        num_frames: Number of frames
        
    Returns:
        Tuple (reshaped_data, num_frames) with the reshaped data and number of frames
    """
    if fit_data is None or num_frames == 0:
        return None, 0
    
    # Determine parameters per frame
    params_per_frame = len(fit_data) // num_frames
    
    if params_per_frame * num_frames != len(fit_data):
        print(f"Warning: Fit data size ({len(fit_data)}) is not an exact multiple of num_frames ({num_frames})")
        # Adjust for incomplete frames
        fit_data = fit_data[:params_per_frame * num_frames]
    
    # Each peak has 5 parameters (amplitude, background, mix, center, sigma)
    peaks_per_frame = params_per_frame // 5
    
    # Check if params_per_frame is a multiple of 5
    if params_per_frame % 5 != 0:
        print(f"Warning: Params per frame ({params_per_frame}) is not a multiple of 5")
        return fit_data.reshape(num_frames, params_per_frame), num_frames
    
    return fit_data.reshape(num_frames, peaks_per_frame, 5), num_frames

def reshape_int2d_data(int2d_data, nRBins, nEtaBins):
    """
    Reshape 2D intensity data into frames.
    
    Args:
        int2d_data: Raw 2D intensity data array
        nRBins: Number of R bins
        nEtaBins: Number of Eta bins
        
    Returns:
        Tuple (reshaped_data, num_frames) with the reshaped data and number of frames
    """
    if int2d_data is None:
        return None, 0
    
    frame_size = nRBins * nEtaBins
    num_frames = len(int2d_data) // frame_size
    
    if num_frames * frame_size != len(int2d_data):
        print(f"Warning: Int2D data size ({len(int2d_data)}) is not an exact multiple of frame size ({frame_size})")
        # Truncate to complete frames
        int2d_data = int2d_data[:num_frames * frame_size]
    
    return int2d_data.reshape(num_frames, nRBins, nEtaBins), num_frames

def load_mapping_file(mapping_file, start_index=0):
    """
    Load a mapping file that maps frame indices to dataset IDs.
    
    Args:
        mapping_file: Path to the mapping file (JSON format)
        start_index: Starting index to adjust mapping
        
    Returns:
        Dictionary mapping frame indices to dataset IDs
    """
    if not os.path.exists(mapping_file):
        print(f"Warning: Mapping file {mapping_file} does not exist")
        return None
    
    with open(mapping_file, 'r') as f:
        mapping = json.load(f)
    
    # Convert keys to integers and adjust for start_index
    adjusted_mapping = {}
    for k, v in mapping.items():
        if isinstance(k, str) and k.isdigit():
            k = int(k)
        if isinstance(k, int) and k >= start_index:
            adjusted_mapping[k - start_index] = v
    
    return adjusted_mapping

def estimate_fit_frame_size(fit_file, lineout_frame_size, num_sample_frames=10):
    """
    Estimate the frame size for fit data by analyzing the file.
    
    Args:
        fit_file: Path to the fit binary file
        lineout_frame_size: Size of a lineout frame (in elements)
        num_sample_frames: Number of frames to analyze
        
    Returns:
        Estimated size of a fit frame (in elements)
    """
    if not os.path.exists(fit_file):
        return 0
    
    # Get the total file size in number of doubles
    file_size = os.path.getsize(fit_file) // 8
    
    # If the file is small, just use the whole file
    if file_size < lineout_frame_size * num_sample_frames:
        num_sample_frames = max(1, file_size // lineout_frame_size)
    
    # Try to determine the fit frame size by looking at the common divisors
    # Most likely the fit frame size is a multiple of 5 (for peak parameters)
    if num_sample_frames > 0:
        for multiplier in range(1, 101):  # Try up to 100 peaks per frame
            if file_size % (5 * num_sample_frames * multiplier) == 0:
                return 5 * multiplier
    
    # Fallback: assume 1 peak per frame (5 parameters)
    return 5

def sum_frames(int2d_data, osf, sorted_original_indices):
    """
    Sum frames according to OmegaSumFrames parameter.
    
    This function assumes the int2d_data has already been sorted into the
    correct order for summing. It uses the sorted_original_indices list to
    report which original frames were part of each sum.
    
    Args:
        int2d_data: 3D array of 2D intensity data, pre-sorted into the desired
                    summing order.
        osf: OmegaSumFrames value
        sorted_original_indices: List of original frame indices, in the same
                                 sorted order as int2d_data.
        
    Returns:
        Tuple (summed_data, frame_groups, counts) where:
            - summed_data is a list of summed frame arrays
            - frame_groups is a list of lists, each containing the *original*
              frame indices that were summed.
            - counts is a list of the number of frames summed in each group.
    """
    if int2d_data is None:
        return [], [], []
    
    num_frames = int2d_data.shape[0]
    summed_data = []
    frame_groups = []
    counts = []
    
    # Special case: OSF = -1, sum all frames
    if osf == -1:
        summed = np.sum(int2d_data, axis=0)
        summed_data.append(summed)
        frame_groups.append(sorted_original_indices)
        counts.append(num_frames)
        return summed_data, frame_groups, counts
    
    # Handle regular OSF cases
    if osf is None or osf <= 0:
        # If OSF not specified or invalid, treat each frame individually
        for i in range(num_frames):
            summed_data.append(int2d_data[i])
            frame_groups.append([sorted_original_indices[i]])
            counts.append(1)
    else:
        # Group frames by OSF
        for i in range(0, num_frames, osf):
            end_idx = min(i + osf, num_frames)
            frames_to_sum = int2d_data[i:end_idx]
            summed = np.sum(frames_to_sum, axis=0)
            summed_data.append(summed)
            frame_groups.append(sorted_original_indices[i:end_idx])
            counts.append(end_idx - i)
    
    return summed_data, frame_groups, counts

def create_hdf5_file(output_file, lineout_data, fit_data, int2d_data, mapping=None, osf=None):
    """
    Create an HDF5 file with the organized data. Datasets and summed frames
    are sorted based on the filename found in the mapping file.
    
    Args:
        output_file: Path to the output HDF5 file
        lineout_data: Lineout data array
        fit_data: Fit data array
        int2d_data: 2D intensity data array
        mapping: Dictionary mapping frame indices to dataset IDs
        osf: OmegaSumFrames value (or None if not specified)
    """
    with h5py.File(output_file, 'w') as f:
        # Create attributes to store metadata
        f.attrs['creation_date'] = np.bytes_(datetime.datetime.now().isoformat())
        num_frames = lineout_data.shape[0] if lineout_data is not None else 0
        f.attrs['num_frames'] = num_frames
        
        # Determine the order for creating datasets. Default is chronological.
        frame_indices = list(range(num_frames))

        if mapping:
            # If a mapping file is provided, sort frame indices based on identifiers.
            def get_sort_key(frame_index):
                map_value = mapping.get(frame_index)
                is_mapped = map_value is not None
                if is_mapped:
                    if isinstance(map_value, dict):
                        sort_string = map_value.get('filename') or str(map_value.get('uniqueId', str(map_value)))
                    else:
                        sort_string = str(map_value)
                    return (0, sort_string)
                else:
                    return (1, frame_index)

            frame_indices.sort(key=get_sort_key)
            print("Sorted output datasets and summing groups based on mapping file.")
        
        # Helper to get a descriptive dataset name from the mapping
        def get_dataset_name(frame_index):
            if not mapping: return str(frame_index)
            map_value = mapping.get(frame_index)
            if map_value is None: return str(frame_index)
            
            if isinstance(map_value, dict):
                name = map_value.get('filename')
                if name: return os.path.splitext(name)[0]
                return str(map_value.get('uniqueId') or map_value.get('dataset_num', frame_index))
            else:
                return str(map_value)

        # Create groups for each data type if we have data for them
        if lineout_data is not None:
            lineout_group = f.create_group('lineouts')
            lineout_group.attrs['description'] = np.bytes_("Radially integrated data: R and intensity values")
            lineout_group.attrs['dimension_labels'] = np.bytes_("R[px], I[a.u.]")
            
            for i in frame_indices:
                dataset_name = get_dataset_name(i)
                ds = lineout_group.create_dataset(dataset_name, data=lineout_data[i])
                ds.attrs['frame_index'] = i
        
        if fit_data is not None:
            fit_group = f.create_group('fit')
            fit_group.attrs['description'] = np.bytes_("Peak fitting results: amplitude, background, mix, center, sigma")
            fit_group.attrs['parameter_labels'] = np.bytes_("amplitude, background, mix, center, sigma")
            
            num_fit_frames = fit_data.shape[0]
            for i in frame_indices:
                if i < num_fit_frames:
                    dataset_name = get_dataset_name(i)
                    ds = fit_group.create_dataset(dataset_name, data=fit_data[i])
                    ds.attrs['frame_index'] = i
                    if fit_data.ndim == 3:
                        ds.attrs['num_peaks'] = fit_data.shape[1]
        
        if int2d_data is not None:
            omega_group = f.create_group('OmegaSumFrame')
            omega_group.attrs['description'] = np.bytes_("2D intensity data: R vs. Eta, summed by sorted filename order")
            omega_group.attrs['dimension_labels'] = np.bytes_("R, Eta")

            # Reorder the int2d_data array according to the sorted frame_indices.
            # This ensures that adjacent frames in this new array are correct for summing.
            sorted_int2d_data = int2d_data[frame_indices]
            
            # Handle summing frames using the reordered data
            summed_data, frame_groups, counts = sum_frames(sorted_int2d_data, osf, frame_indices)
            
            for i, (summed, frames, count) in enumerate(zip(summed_data, frame_groups, counts)):
                last_frame = frames[-1]
                dataset_name = f"LastFrameNumber_{last_frame}"
                
                ds = omega_group.create_dataset(dataset_name, data=summed)
                ds.attrs['Number Of Frames Summed'] = count
                ds.attrs['frame_indices'] = frames
                
                if mapping:
                    identifier = "unknown"
                    if isinstance(mapping.get(last_frame, {}), dict):
                        identifier = mapping.get(last_frame, {}).get('filename', 
                                    mapping.get(last_frame, {}).get('uniqueId', last_frame))
                    else:
                        identifier = mapping.get(last_frame, last_frame)
                    ds.attrs['identifier'] = np.bytes_(str(identifier))

def extract_dataset_mapping_from_server_log(log_file):
    """
    Extract mapping between frame indices and dataset IDs from server log file.
    
    Args:
        log_file: Path to the server log file
        
    Returns:
        Dictionary mapping frame indices to dataset IDs
    """
    if not os.path.exists(log_file):
        print(f"Warning: Log file {log_file} does not exist")
        return None
    
    mapping = {}
    current_dataset = None
    current_filename = None
    frame_idx = 0
    
    with open(log_file, 'r') as f:
        for line in f:
            if "Sent dataset #" in line:
                import re
                match = re.search(r"#(\d+)", line)
                if match:
                    current_dataset = int(match.group(1))
                    if frame_idx not in mapping: mapping[frame_idx] = {}
                    mapping[frame_idx]['dataset_num'] = current_dataset
                    if current_filename:
                        mapping[frame_idx]['filename'] = current_filename
                    frame_idx += 1
            elif "Processing TIF frame #" in line:
                match = re.search(r"#(\d+)", line)
                if match:
                    current_dataset = int(match.group(1))
            elif "Processing" in line and "file:" in line:
                match = re.search(r"Processing .* file: (.+)", line)
                if match:
                    current_filename = os.path.basename(match.group(1))
            elif "Processed" in line and "frame" in line and not "Sent dataset" in line:
                if current_filename is not None:
                    if frame_idx not in mapping: mapping[frame_idx] = {}
                    mapping[frame_idx]['filename'] = current_filename
                    frame_idx += 1
    
    for k, v in list(mapping.items()):
        if not isinstance(v, dict):
            mapping[k] = {'filename': v} if isinstance(v, str) else {'dataset_num': v}
    
    return mapping

def main():
    parser = argparse.ArgumentParser(description="Convert binary output files from IntegratorFitPeaksGPUStream to HDF5")
    parser.add_argument('--lineout', type=str, default='lineout.bin', help='Lineout binary file (default: lineout.bin)')
    parser.add_argument('--fit', type=str, default='fit.bin', help='Fit binary file (optional, default: fit.bin)')
    parser.add_argument('--int2d', type=str, default='Int2D.bin', help='2D integrated binary file (optional, default: Int2D.bin)')
    parser.add_argument('--params', type=str, required=True, help='Parameter file used for integration')
    parser.add_argument('--mapping', type=str, help='JSON file mapping frame indices to dataset IDs')
    parser.add_argument('--server-log', type=str, help='Server log file to extract dataset IDs')
    parser.add_argument('--output', type=str, default='integrator_output.h5', help='Output HDF5 file (default: integrator_output.h5)')
    parser.add_argument('--start', type=int, default=0, help='Starting frame index (default: 0)')
    parser.add_argument('--count', type=int, help='Number of frames to process (default: all)')
    parser.add_argument('--peaks-per-frame', type=int, help='Number of peaks per frame for fit data (default: auto-detect)')
    parser.add_argument('--omega-sum-frames', type=int, help='Override OmegaSumFrames value from parameter file')
    
    args = parser.parse_args()
    
    nRBins, nEtaBins, osf = get_frame_dimensions(args.params)
    print(f"Frame dimensions: {nRBins} R bins, {nEtaBins} Eta bins")
    
    if args.omega_sum_frames is not None:
        osf = args.omega_sum_frames
        print(f"Overriding OmegaSumFrames with value: {osf}")
    
    if osf is not None:
        print(f"OmegaSumFrames: {osf}")
        if osf == -1: print("  All frames will be summed into a single output based on sorted order")
        elif osf > 0: print(f"  Every {osf} frames will be summed together based on sorted order")
    
    lineout_frame_size = nRBins * 2
    int2d_frame_size = nRBins * nEtaBins
    
    raw_lineout_data = read_binary_file(args.lineout, 
        offset=args.start * lineout_frame_size, 
        count=args.count * lineout_frame_size if args.count else None)
    
    if raw_lineout_data is None:
        print(f"Error: Could not read lineout file {args.lineout}")
        return
    
    lineout_data, lineout_frames = reshape_lineout_data(raw_lineout_data, nRBins)
    
    fit_frame_size = 0
    raw_fit_data = None
    if os.path.exists(args.fit):
        if args.peaks_per_frame:
            fit_frame_size = args.peaks_per_frame * 5
        else:
            fit_frame_size = estimate_fit_frame_size(args.fit, lineout_frame_size)
            print(f"Estimated fit frame size: {fit_frame_size} elements per frame ({fit_frame_size // 5} peaks per frame)")
        
        raw_fit_data = read_binary_file(args.fit, 
            offset=args.start * fit_frame_size, 
            count=args.count * fit_frame_size if args.count and fit_frame_size else None)
    else:
        print(f"Note: Fit file {args.fit} does not exist, continuing without fit data")
    
    raw_int2d_data = None
    if os.path.exists(args.int2d):
        raw_int2d_data = read_binary_file(args.int2d, 
            offset=args.start * int2d_frame_size, 
            count=args.count * int2d_frame_size if args.count else None)
    else:
        print(f"Note: Int2D file {args.int2d} does not exist, continuing without Int2D data")
    
    fit_data, _ = reshape_fit_data(raw_fit_data, lineout_frames)
    int2d_data, _ = reshape_int2d_data(raw_int2d_data, nRBins, nEtaBins)
    
    print(f"Found {lineout_frames} frames in lineout data")
    if fit_data is not None:
        print(f"Found {fit_data.shape[0]} frames in fit data")
        if fit_data.ndim == 3: print(f"  Each frame has {fit_data.shape[1]} peaks with {fit_data.shape[2]} parameters per peak")
    else: print("No fit data available")
        
    if int2d_data is not None: print(f"Found {int2d_data.shape[0]} frames in int2d data")
    else: print("No int2d data available")
    
    mapping = None
    if args.mapping:
        mapping = load_mapping_file(args.mapping, args.start)
        print(f"Loaded mapping from {args.mapping}")
    elif args.server_log:
        mapping = extract_dataset_mapping_from_server_log(args.server_log)
        print(f"Extracted mapping from server log {args.server_log}")
    
    create_hdf5_file(args.output, lineout_data, fit_data, int2d_data, mapping, osf)
    print(f"Created HDF5 file {args.output}")

if __name__ == "__main__":
    main()