#!/usr/bin/env python3
"""
Integrator Pipeline Control Script

This script orchestrates the X-ray diffraction data processing pipeline:
1. Starts the IntegratorFitPeaksGPUStream executable in the background
2. Once ready, starts the integrator_server.py to feed data to the integrator
3. Monitors processing progress
4. When complete, stops the processes and runs integrator_stream_process_h5.py
   to convert the binary output to HDF5 format

Usage:
  python integrator_control.py <param_file> <folder_to_process> [options]

Options:
  --extension=<ext>     File extension to process (default: tif)
  --pva                 Use PVA stream instead of files
  --dark=<dark_file>    Dark file for background subtraction
"""

import os
import sys
import time
import signal
import subprocess
import json
from pathlib import Path

# Check for psutil
try:
    import psutil
except ImportError:
    print("This script requires the psutil package. Install it with:")
    print("pip install psutil")
    sys.exit(1)

def read_parameters_from_file(param_file):
    """
    Extract relevant parameters from parameter file
    
    Returns:
        tuple: (nx, ny, omega_sum_frames)
    """
    nx = 0
    ny = 0
    omega_sum_frames = None
    
    with open(param_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('NrPixelsY'):
                nx = int(line.split()[1])
            elif line.startswith('NrPixelsZ'):
                ny = int(line.split()[1])
            elif line.startswith('NrPixels'):
                # If NrPixels is specified, use it for both dimensions
                nx = ny = int(line.split()[1])
            elif line.startswith('OmegaSumFrames'):
                omega_sum_frames = int(line.split()[1])
    
    return nx, ny, omega_sum_frames

def wait_for_server_ready(logfile, timeout=60):
    """Check logfile for indication that server is ready"""
    start_time = time.time()
    check_interval = 0.5
    
    while time.time() - start_time < timeout:
        if os.path.exists(logfile):
            with open(logfile, 'r') as f:
                content = f.read()
                if "Server listening on port 5000" in content:
                    print(f"IntegratorFitPeaksGPUStream is ready! (took {time.time() - start_time:.1f} seconds)")
                    return True
        
        time.sleep(check_interval)
    
    print(f"Timed out waiting for IntegratorFitPeaksGPUStream to start after {timeout} seconds")
    return False

def monitor_processing(mapping_file, expected_frames=None, check_interval=5):
    """
    Monitor the mapping file to track progress
    
    Args:
        mapping_file: Path to the frame mapping JSON file
        expected_frames: Number of frames expected to process
        check_interval: How often to check progress (in seconds)
        
    Returns:
        True if processing completed successfully, False otherwise
    """
    last_count = 0
    last_update_time = time.time()
    start_time = time.time()
    stalled_time = 0
    max_stall_time = 300  # 5 minutes
    
    while True:
        if os.path.exists(mapping_file):
            try:
                with open(mapping_file, 'r') as f:
                    mapping_data = json.load(f)
                    
                current_count = len(mapping_data)
                
                if current_count > last_count:
                    elapsed = time.time() - start_time
                    rate = current_count / elapsed if elapsed > 0 else 0
                    
                    if expected_frames:
                        remaining = expected_frames - current_count
                        eta = remaining / rate if rate > 0 else "unknown"
                        eta_str = f"{eta:.1f} seconds" if isinstance(eta, float) else eta
                        print(f"Processed {current_count}/{expected_frames} frames "
                              f"({current_count/expected_frames*100:.1f}%) - "
                              f"Rate: {rate:.2f} frames/sec - ETA: {eta_str}")
                    else:
                        print(f"Processed {current_count} frames - Rate: {rate:.2f} frames/sec")
                    
                    last_count = current_count
                    last_update_time = time.time()
                    stalled_time = 0
                else:
                    stalled_time = time.time() - last_update_time
                    
                if expected_frames and current_count >= expected_frames:
                    total_time = time.time() - start_time
                    print(f"All {expected_frames} frames processed in {total_time:.2f} seconds!")
                    return True
                    
                if stalled_time > max_stall_time:
                    print(f"WARNING: Processing appears to be stalled (no updates for {int(stalled_time)} seconds)")
                    choice = input("Options: [c]ontinue waiting, [f]inish anyway, [a]bort: ")
                    
                    if choice.lower().startswith('f'):
                        print("Finishing processing early...")
                        return True
                    elif choice.lower().startswith('a'):
                        print("Aborting processing...")
                        return False
                    else:
                        print("Continuing to wait...")
                        last_update_time = time.time()  # Reset stall timer
            except json.JSONDecodeError:
                print(f"Warning: Mapping file {mapping_file} exists but is not valid JSON (may be partially written)")
            except Exception as e:
                print(f"Error reading mapping file: {e}")
        
        time.sleep(check_interval)

def find_process_by_name(name):
    """Find process IDs by name pattern"""
    pids = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = " ".join(proc.info['cmdline'] or [])
            if name in cmdline:
                pids.append(proc.info['pid'])
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return pids

def kill_process(pid, timeout=5):
    """Kill a process by PID with timeout for graceful termination"""
    try:
        process = psutil.Process(pid)
        process.terminate()
        
        try:
            process.wait(timeout=timeout)
            print(f"Gracefully terminated process {pid}")
            return True
        except psutil.TimeoutExpired:
            print(f"Process {pid} did not terminate within {timeout} seconds, forcing...")
            process.kill()
            process.wait(timeout=2)
            print(f"Forcefully terminated process {pid}")
            return True
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        print(f"Could not terminate process {pid} - it may have already exited")
        return False
    except Exception as e:
        print(f"Error terminating process {pid}: {e}")
        return False

def count_files_in_folder(folder, extension):
    """Count files with given extension in folder"""
    pattern = f"*.{extension}"
    files = list(Path(folder).glob(pattern))
    return len(files)

def main():
    # Set up argument parser
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Orchestrate the X-ray diffraction data processing pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Required arguments
    parser.add_argument('--param-file', required=True, 
                        help='Parameter file (e.g., calib_file.txt)')
    
    # Input source options - mutually exclusive group
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--folder', help='Folder containing image files to process')
    input_group.add_argument('--pva', action='store_true', help='Use PVA stream instead of files')
    
    # Optional arguments
    parser.add_argument('--extension', default='tif', help='File extension to process (default: tif)')
    parser.add_argument('--dark', help='Dark file for background subtraction')
    parser.add_argument('--output-h5', default='integrator_output.h5', 
                        help='Output HDF5 filename (default: integrator_output.h5)')
    parser.add_argument('--pva-ip', default='10.54.105.139', 
                        help='PVA server IP address (default: 10.54.105.139)')
    parser.add_argument('--pva-channel', default='16pil-idb:Pva1:Image', 
                        help='PVA channel string (default: 16pil-idb:Pva1:Image)')
    parser.add_argument('--mapping-file', default='frame_mapping.json',
                        help='Output JSON file for frame mapping (default: frame_mapping.json)')
    parser.add_argument('--save-interval', type=int, default=10,
                        help='Save mapping every N frames (default: 10)')
    parser.add_argument('--h5-location', default='exchange/data',
                        help='Location within H5 files containing image data (default: exchange/data)')
    
    args = parser.parse_args()
    
    # Validation
    param_file = args.param_file
    if not os.path.exists(param_file):
        print(f"Error: Parameter file '{param_file}' not found")
        sys.exit(1)
    
    use_pva = args.pva
    folder_to_process = args.folder if not use_pva else None
    extension = args.extension
    mapping_file = args.mapping_file
    
    # Validate dark file if specified
    dark_file = args.dark
    if dark_file and not os.path.exists(dark_file):
        print(f"Warning: Dark file '{dark_file}' not found")
        dark_file = None
    
    # Set up file paths
    integrator_log = "integrator.log"
    server_log = "server.log"
    mapping_file = "frame_mapping.json"
    
    # Count number of files to process for progress tracking
    expected_frames = 0
    if not use_pva and folder_to_process:
        expected_frames = count_files_in_folder(folder_to_process, extension)
        if expected_frames == 0:
            print(f"Warning: No {extension} files found in {folder_to_process}")
            proceed = input("Continue anyway? (y/n): ")
            if proceed.lower() != 'y':
                sys.exit(0)
        else:
            print(f"Found {expected_frames} {extension} files to process")
    
    # Read parameters from parameter file
    nx, ny, omega_sum_frames = read_parameters_from_file(param_file)
    if nx == 0 or ny == 0:
        print("Error: Could not determine frame size from parameter file")
        sys.exit(1)
    
    print(f"Frame size: {nx} x {ny}")
    if omega_sum_frames is not None:
        print(f"OmegaSumFrames: {omega_sum_frames}")
    
    # Start IntegratorFitPeaksGPUStream in background
    print("Starting IntegratorFitPeaksGPUStream...")
    with open(integrator_log, 'w') as logfile:
        integrator_cmd = [os.path.expanduser("~/opt/MIDAS/FF_HEDM/bin/IntegratorFitPeaksGPUStream"), param_file]
        if dark_file:
            integrator_cmd.append(dark_file)
            print(f"Using dark correction file: {dark_file}")
        
        integrator_proc = subprocess.Popen(
            integrator_cmd,
            stdout=logfile,
            stderr=subprocess.STDOUT
        )
    
    print(f"Started IntegratorFitPeaksGPUStream with PID {integrator_proc.pid}")
    
    # Wait for server to be ready
    if not wait_for_server_ready(integrator_log):
        print("Error: Failed to start IntegratorFitPeaksGPUStream")
        print(f"Check {integrator_log} for details")
        integrator_proc.terminate()
        sys.exit(1)
    
    # Prepare server command
    server_cmd = [
        os.path.expanduser("~/opt/MIDAS/utils/integrator_server.py"),
        "--stream", "0" if not use_pva else "1",
        "--mapping-file", mapping_file,
        "--save-interval", str(args.save_interval)
    ]
    
    if not use_pva:
        server_cmd.extend([
            "--folder", folder_to_process,
            "--extension", extension
        ])
        
        # Add frame size if needed (for binary formats like GE)
        if extension.startswith("ge"):
            server_cmd.extend(["--frame-size", f"{nx}x{ny}"])
            
        # Add h5-location if needed
        if extension == "h5":
            server_cmd.extend(["--h5-location", args.h5_location])
    else:
        # PVA specific options
        server_cmd.extend([
            "--pva-ip", args.pva_ip,
            "--channel", args.pva_channel
        ])
    
    # Start integrator server
    print(f"Starting integrator_server.py with command: {' '.join(server_cmd)}")
    with open(server_log, 'w') as logfile:
        server_proc = subprocess.Popen(
            server_cmd,
            stdout=logfile,
            stderr=subprocess.STDOUT
        )
    
    print(f"Started integrator_server.py with PID {server_proc.pid}")
    
    # Monitor processing
    print("\nMonitoring processing progress...")
    completed = monitor_processing(mapping_file, expected_frames)
    
    # Check if server is still running
    server_running = server_proc.poll() is None
    
    if not server_running and not completed:
        print(f"Warning: Server process exited before completion. Check {server_log}")
    
    # Terminate processes
    print("Terminating server process...")
    if server_running:
        kill_process(server_proc.pid)
    
    # Give the integration server a moment to finish processing
    print("Waiting for IntegratorFitPeaksGPUStream to finish processing...")
    time.sleep(5)
    
    print("Terminating IntegratorFitPeaksGPUStream...")
    # Find all instances, sometimes the PID we have might not be correct
    pids = find_process_by_name("IntegratorFitPeaksGPUStream")
    for pid in pids:
        kill_process(pid)
    
    # Run integrator_stream_process_h5.py
    print("\nConverting binary output to HDF5...")
    h5_cmd = [
        os.path.expanduser("~/opt/MIDAS/utils/integrator_stream_process_h5.py"),
        "--lineout", "lineout.bin",
        "--fit", "fit.bin",
        "--int2d", "Int2D.bin",
        "--params", param_file,
        "--mapping", mapping_file,
        "--output", args.output_h5
    ]
    
    # Add omega-sum-frames if it was found in the parameter file
    if omega_sum_frames is not None:
        h5_cmd.extend(["--omega-sum-frames", str(omega_sum_frames)])
    
    print(f"Running: {' '.join(h5_cmd)}")
    try:
        result = subprocess.run(h5_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error converting to HDF5: {result.stderr}")
        else:
            print("HDF5 conversion completed successfully")
    except Exception as e:
        print(f"Error running HDF5 conversion: {e}")
    
    print("\nProcessing complete!")
    print("Output files:")
    print(f"  - lineout.bin (raw lineout data)")
    print(f"  - fit.bin (raw fit data, if peak fitting was enabled)")
    print(f"  - Int2D.bin (raw 2D integration data, if enabled)")
    print(f"  - {args.output_h5} (HDF5 formatted data)")
    print(f"  - {mapping_file} (frame mapping JSON)")
    print(f"  - {integrator_log} (log from IntegratorFitPeaksGPUStream)")
    print(f"  - {server_log} (log from integrator_server.py)")

if __name__ == "__main__":
    main()