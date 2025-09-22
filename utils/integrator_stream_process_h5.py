#!/usr/bin/env python3
"""
integrator_stream_process_h5.py - Convert binary output files from IntegratorFitPeaksGPUStream to HDF5

This script reads the binary output files produced by the CUDA integrator 
(lineout.bin, and optionally fit.bin, Int2D.bin, and fit_curves.bin) and 
organizes them into a single HDF5 file with appropriate groups and datasets.

The order of datasets and the grouping for summed frames is determined by the
sorted filenames in the provided mapping file.
"""

import numpy as np
import h5py
import os
import argparse
import json
import datetime
import struct

def read_binary_file(filename, dtype=np.float64, offset=0, count=None):
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
    Extracts dimensions and peak count from the integrator's parameter file.
    
    Args:
        params_file: Path to the parameter file
        
    Returns:
        Tuple (nRBins, nEtaBins, OSF, num_peaks, do_peak_fit)
    """
    # Default values
    RMax, RMin, RBinSize = 100, 10, 0.1
    EtaMax, EtaMin, EtaBinSize = 180, -180, 0.1
    OSF = None
    num_peaks = 0
    do_peak_fit = True  # Assume fitting is done by default

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

            if key == 'RMax': RMax = float(value)
            elif key == 'RMin': RMin = float(value)
            elif key == 'RBinSize': RBinSize = float(value)
            elif key == 'EtaMax': EtaMax = float(value)
            elif key == 'EtaMin': EtaMin = float(value)
            elif key == 'EtaBinSize': EtaBinSize = float(value)
            elif key == 'OmegaSumFrames': OSF = int(value)
            elif key == 'PeakLocation': num_peaks += 1
            elif key == 'DoPeakFit' and value == '0': do_peak_fit = False

    nRBins = int(np.ceil((RMax - RMin) / RBinSize))
    nEtaBins = int(np.ceil((EtaMax - EtaMin) / EtaBinSize))
    
    return nRBins, nEtaBins, OSF, num_peaks, do_peak_fit

def reshape_map_data(map_data, nRBins, nEtaBins):
    """Reshapes the flat R, TTh, Eta, Area map data."""
    if map_data is None:
        return None
    
    expected_size = nRBins * nEtaBins * 4
    if len(map_data) != expected_size:
        print(f"Warning: Map data size ({len(map_data)}) does not match expected size ({expected_size}).")
        return None
    
    # Reshape to (4, nRBins, nEtaBins) where the 4 maps are R, TTh, Eta, Area
    return map_data.reshape(4, nRBins, nEtaBins)

def reshape_lineout_data(lineout_data, nRBins):
    if lineout_data is None:
        return None, 0
    frame_size = nRBins * 2
    num_frames = len(lineout_data) // frame_size
    if num_frames * frame_size != len(lineout_data):
        print(f"Warning: Lineout data size ({len(lineout_data)}) is not an exact multiple of frame size ({frame_size})")
        lineout_data = lineout_data[:num_frames * frame_size]
    return lineout_data.reshape(num_frames, nRBins, 2), num_frames

def reshape_fit_data(fit_data, num_frames, peaks_per_frame):
    """Reshapes fit data using a known number of peaks per frame."""
    if fit_data is None or num_frames == 0:
        return None, 0
    
    if peaks_per_frame == 0:
        print("Warning: No 'PeakLocation' found in params file. Cannot process fit data.")
        return None, 0
    
    PARAMS_PER_PEAK = 7
    expected_params_per_frame = peaks_per_frame * PARAMS_PER_PEAK
    
    # Check if the total data size is a multiple of the expected size for one frame
    if len(fit_data) % expected_params_per_frame != 0:
        print(f"Warning: Total fit data size ({len(fit_data)}) is not a multiple of the expected "
              f"parameters for one frame ({expected_params_per_frame}). File may be from a different run.")
    
    # The CUDA code writes parameters for all frames, even if some fits failed.
    # The number of frames should match the lineout file.
    if len(fit_data) != num_frames * expected_params_per_frame:
         print(f"Warning: Fit data size implies a different number of frames than lineout data.")
         # We trust the lineout_frames count and truncate the fit data if necessary.
         fit_data = fit_data[:num_frames * expected_params_per_frame]

    return fit_data.reshape(num_frames, peaks_per_frame, PARAMS_PER_PEAK), num_frames

def reshape_int2d_data(int2d_data, nRBins, nEtaBins):
    if int2d_data is None: 
        return None, 0
    frame_size = nRBins * nEtaBins
    num_frames = len(int2d_data) // frame_size
    if num_frames * frame_size != len(int2d_data):
        print(f"Warning: Int2D data size ({len(int2d_data)}) is not an exact multiple of frame size ({frame_size})")
        int2d_data = int2d_data[:num_frames * frame_size]
    return int2d_data.reshape(num_frames, nRBins, nEtaBins), num_frames

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

def read_fit_curves(filename):
    """
    Reads the fit_curves.bin file with its variable-length record structure.
    
    File format per record:
    - C int (4 bytes): frame_index
    - C int (4 bytes): start_bin_index
    - C int (4 bytes): num_points
    - C double (8*num_points bytes): curve_data
    
    Returns:
        A dictionary mapping frame_index to a tuple of (start_bin, curve_data_array).
    """
    if not os.path.exists(filename):
        print(f"Warning: Fit curves file {filename} does not exist")
        return None
    
    curves_data = {}
    header_fmt = 'iii'  # 3 integers
    header_size = struct.calcsize(header_fmt)

    with open(filename, 'rb') as f:
        while True:
            header_bytes = f.read(header_size)
            if not header_bytes:
                break  # End of file
            
            try:
                frame_idx, start_bin, num_points = struct.unpack(header_fmt, header_bytes)
                
                data_fmt = f'{num_points}d' # num_points doubles
                data_size = struct.calcsize(data_fmt)
                data_bytes = f.read(data_size)
                
                curve_array = np.array(struct.unpack(data_fmt, data_bytes))
                curves_data[frame_idx] = (start_bin, curve_array)

            except struct.error:
                print("Warning: Could not parse a record from fit_curves.bin. File may be truncated.")
                break
                
    return curves_data

def sum_frames(int2d_data, osf, sorted_original_indices):
    if int2d_data is None: return [], [], []
    num_frames = int2d_data.shape[0]
    summed_data, frame_groups, counts = [], [], []
    if osf == -1:
        summed_data.append(np.sum(int2d_data, axis=0))
        frame_groups.append(sorted_original_indices)
        counts.append(num_frames)
        return summed_data, frame_groups, counts
    if osf is None or osf <= 0:
        for i in range(num_frames):
            summed_data.append(int2d_data[i])
            frame_groups.append([sorted_original_indices[i]])
            counts.append(1)
    else:
        for i in range(0, num_frames, osf):
            end_idx = min(i + osf, num_frames)
            summed_data.append(np.sum(int2d_data[i:end_idx], axis=0))
            frame_groups.append(sorted_original_indices[i:end_idx])
            counts.append(end_idx - i)
    return summed_data, frame_groups, counts

def create_hdf5_file(output_file, lineout_data, fit_data, int2d_data, fit_curves_data, map_data=None, mapping=None, osf=None):
    """
    Create an HDF5 file with the organized data, including fit curves.
    """
    with h5py.File(output_file, 'w') as f:
        f.attrs['creation_date'] = np.bytes_(datetime.datetime.now().isoformat())
        num_frames = lineout_data.shape[0] if lineout_data is not None else 0
        f.attrs['num_frames'] = num_frames
        
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
            print("Sorted output datasets and summing groups based on mapping file.")

        def get_dataset_name(frame_index):
            if not mapping: return str(frame_index)
            map_value = mapping.get(frame_index)
            if map_value is None: return str(frame_index)
            name = map_value.get('filename')
            if name: return os.path.splitext(name)[0]
            return str(map_value.get('uniqueId') or map_value.get('dataset_num', frame_index))

        if map_data is not None:
            geom_group = f.create_group('geometry_maps')
            geom_group.attrs['description'] = np.bytes_("Static geometry maps for each (R, Eta) bin")
            
            ds_r = geom_group.create_dataset('R_map', data=map_data[0])
            ds_r.attrs['description'] = np.bytes_("R-center for each bin")
            ds_r.attrs['units'] = np.bytes_("pixels")
            
            ds_tth = geom_group.create_dataset('TTh_map', data=map_data[1])
            ds_tth.attrs['description'] = np.bytes_("TwoTheta-center for each bin")
            ds_tth.attrs['units'] = np.bytes_("degrees")
            
            ds_eta = geom_group.create_dataset('Eta_map', data=map_data[2])
            ds_eta.attrs['description'] = np.bytes_("Eta-center for each bin")
            ds_eta.attrs['units'] = np.bytes_("degrees")
            
            ds_area = geom_group.create_dataset('Area_map', data=map_data[3])
            ds_area.attrs['description'] = np.bytes_("Effective pixel area for each bin")
            ds_area.attrs['units'] = np.bytes_("fractional pixels")


        if lineout_data is not None:
            lineout_group = f.create_group('lineouts')
            lineout_group.attrs['description'] = np.bytes_("Radially integrated data: R and intensity values")
            lineout_group.attrs['dimension_labels'] = np.bytes_("R[px], I[a.u.]")
            for i in frame_indices:
                ds = lineout_group.create_dataset(get_dataset_name(i), data=lineout_data[i])
                ds.attrs['frame_index'] = i
        
        if fit_data is not None:
            fit_group = f.create_group('fit')
            fit_group.attrs['description'] = np.bytes_("Peak fitting results")
            fit_group.attrs['parameter_labels'] = np.bytes_("amplitude, background, mix, center, sigma, goodness_of_fit, peak_area")
            num_fit_frames = fit_data.shape[0]
            for i in frame_indices:
                if i < num_fit_frames:
                    ds = fit_group.create_dataset(get_dataset_name(i), data=fit_data[i])
                    ds.attrs['frame_index'] = i
        
        if fit_curves_data is not None and lineout_data is not None:
            curves_group = f.create_group('fit_curves')
            curves_group.attrs['description'] = np.bytes_("Fitted model profiles for each fit job")
            
            # The keys in fit_curves_data are original frame indices. We need to iterate through
            # our sorted frame_indices to maintain order.
            for original_frame_index in frame_indices:
                if original_frame_index in fit_curves_data:
                    start_bin, curve = fit_curves_data[original_frame_index]
                    
                    dataset_name = get_dataset_name(original_frame_index)
                    ds = curves_group.create_dataset(dataset_name, data=curve)
                    
                    # Store metadata to make plotting easy
                    ds.attrs['original_frame_index'] = original_frame_index
                    ds.attrs['start_bin_index'] = start_bin
                    
                    # Store the corresponding R-axis values
                    num_points = len(curve)
                    r_axis = lineout_data[original_frame_index, start_bin : start_bin + num_points, 0]
                    ds.attrs['R_values'] = r_axis

        if int2d_data is not None:
            omega_group = f.create_group('OmegaSumFrame')
            omega_group.attrs['description'] = np.bytes_("2D intensity data: R vs. Eta, summed by sorted filename order")
            omega_group.attrs['dimension_labels'] = np.bytes_("R, Eta")
            sorted_int2d_data = int2d_data[frame_indices]
            summed_data, frame_groups, counts = sum_frames(sorted_int2d_data, osf, frame_indices)
            for i, (summed, frames, count) in enumerate(zip(summed_data, frame_groups, counts)):
                last_frame = frames[-1]
                ds_name = f"Summed_from_{get_dataset_name(frames[0])}_to_{get_dataset_name(last_frame)}"
                ds = omega_group.create_dataset(ds_name, data=summed)
                ds.attrs['Number Of Frames Summed'] = count
                ds.attrs['original_frame_indices'] = frames
                if mapping:
                    ds.attrs['last_frame_identifier'] = np.bytes_(str(get_dataset_name(last_frame)))

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

def main():
    parser = argparse.ArgumentParser(description="Convert binary output files from IntegratorFitPeaksGPUStream to HDF5")
    parser.add_argument('--lineout', type=str, default='lineout.bin', help='Lineout binary file')
    parser.add_argument('--fit', type=str, default='fit.bin', help='Fit binary file')
    parser.add_argument('--int2d', type=str, default='Int2D.bin', help='2D integrated binary file')
    parser.add_argument('--fit-curves', type=str, default='fit_curves.bin', help='Fitted curves binary file')
    parser.add_argument('--params', type=str, required=True, help='Parameter file used for integration')
    parser.add_argument('--map-data', type=str, default='RTthEtaAreaMap.bin', help='R, TTh, Eta, Area map binary file')
    parser.add_argument('--mapping', type=str, help='JSON file mapping frame indices to dataset IDs')
    parser.add_argument('--server-log', type=str, help='Server log file to extract dataset IDs')
    parser.add_argument('--output', type=str, default='integrator_output.h5', help='Output HDF5 file')
    parser.add_argument('--start', type=int, default=0, help='Starting frame index')
    parser.add_argument('--count', type=int, help='Number of frames to process')
    parser.add_argument('--omega-sum-frames', type=int, help='Override OmegaSumFrames value')
    args = parser.parse_args()
    
    nRBins, nEtaBins, osf, num_peaks, do_peak_fit = get_frame_dimensions(args.params)
    print(f"Frame dimensions: {nRBins} R bins, {nEtaBins} Eta bins")
    if num_peaks > 0:
        print(f"Found {num_peaks} 'PeakLocation' entries in parameter file.")

    if not do_peak_fit:
        print("Peak fitting is disabled. Skipping fit and fit_curves data.")

    if args.omega_sum_frames is not None:
        osf = args.omega_sum_frames
        print(f"Overriding OmegaSumFrames with value: {osf}")
    if osf is not None:
        print(f"OmegaSumFrames: {osf}")
    
    # --- Read Lineout Data ---
    lineout_frame_size = nRBins * 2
    raw_lineout_data = read_binary_file(args.lineout, 
        offset=args.start * lineout_frame_size, 
        count=args.count * lineout_frame_size if args.count else None)
    if raw_lineout_data is None:
        print("Error: Could not read lineout file. Aborting."); return
    lineout_data, lineout_frames = reshape_lineout_data(raw_lineout_data, nRBins)

    # --- Read Fit Data (using num_peaks) ---
    fit_data = None
    fit_curves_data = None
    if do_peak_fit:
        raw_fit_data = read_binary_file(args.fit)
        if raw_fit_data is not None:
            fit_data, _ = reshape_fit_data(raw_fit_data, lineout_frames, num_peaks)
        # --- Read Fit Curves Data ---
        fit_curves_data = read_fit_curves(args.fit_curves)
    
    # --- Read Int2D Data ---
    raw_int2d_data = read_binary_file(args.int2d)
    int2d_data = None
    if raw_int2d_data is not None:
        int2d_data, _ = reshape_int2d_data(raw_int2d_data, nRBins, nEtaBins)

    # --- Read Static Map Data ---
    raw_map_data = read_binary_file(args.map_data)
    map_data = None
    if raw_map_data is not None:
        map_data = reshape_map_data(raw_map_data, nRBins, nEtaBins)
    if map_data is not None:
        print(f"Found and processed geometry map data with shape {map_data.shape}")

    print(f"Found {lineout_frames} frames in lineout data")
    if fit_data is not None:
        print(f"Found {fit_data.shape[0]} frames in fit data")
        if fit_data.ndim == 3: 
            print(f"  Each frame has {fit_data.shape[1]} peaks with {fit_data.shape[2]} parameters per peak")
    if int2d_data is not None: 
        print(f"Found {int2d_data.shape[0]} frames in int2d data")
    if fit_curves_data is not None: 
        print(f"Found {len(fit_curves_data)} fitted curves")
    
    mapping = None
    if args.mapping: 
        mapping = load_mapping_file(args.mapping, args.start)
    elif args.server_log: 
        mapping = extract_dataset_mapping_from_server_log(args.server_log)

    create_hdf5_file(args.output, lineout_data, fit_data, int2d_data, fit_curves_data, map_data, mapping, osf)
    print(f"Successfully created HDF5 file: {args.output}")

if __name__ == "__main__":
    main()