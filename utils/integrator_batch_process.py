#!/usr/bin/env python3
"""
Integrator Pipeline Control Script

This script orchestrates the X-ray diffraction data processing pipeline:
0. Run DetectorMapper in case maps are not found in the current directory.
1. Starts the IntegratorFitPeaksGPUStream executable in the background
2. Once ready, starts the integrator_server.py to feed data to the integrator
3. Monitors processing progress
4. When complete, stops the processes and runs integrator_stream_process_h5.py
   to convert the binary output to HDF5 format

Usage:
  python integrator_control.py --param-file <param_file> [--folder <folder> | --pva] [options]

Options:
  --extension=<ext>     File extension to process (default: tif)
  --dark=<dark_file>    Dark file for background subtraction
"""

import os
import sys
import time
import signal
import subprocess
import json
import socket
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

def is_port_open(host, port, timeout=1):
    """Check if a port is open on the specified host"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    result = False
    try:
        result = sock.connect_ex((host, port)) == 0
    except socket.error:
        result = False
    finally:
        sock.close()
    return result

def wait_for_server_ready(logfile, process, timeout=180):
    """
    Check if server is ready by checking if port 60439 is open
    Also monitors process to ensure it's still running and checks log for errors
    
    Args:
        logfile: Path to the log file
        process: Subprocess object of the running process
        timeout: Maximum time to wait in seconds
        
    Returns:
        Boolean indicating if server is ready
    """
    start_time = time.time()
    check_interval = 0.5
    port_check_interval = 2  # Check port every 2 seconds
    last_port_check = 0
    last_log_size = 0
    last_log_check = 0
    
    print(f"Waiting for IntegratorFitPeaksGPUStream to initialize (timeout: {timeout}s)...")
    
    while time.time() - start_time < timeout:
        # Check if process is still running
        if process.poll() is not None:
            print(f"ERROR: IntegratorFitPeaksGPUStream process exited with code {process.returncode}")
            return False
        
        # Check if port is open (every port_check_interval seconds)
        current_time = time.time()
        if current_time - last_port_check > port_check_interval:
            if is_port_open('127.0.0.1', 60439):
                print(f"\nServer port 60439 is open! IntegratorFitPeaksGPUStream is ready! (took {current_time - start_time:.1f} seconds)")
                return True
            last_port_check = current_time
        
        # Still monitor log file for errors
        if os.path.exists(logfile):
            # Check for new content in log file
            current_size = os.path.getsize(logfile)
            
            # Print progress dots if no new log content in last 5 seconds
            if current_size == last_log_size and current_time - last_log_check > 5:
                print(".", end="", flush=True)
                last_log_check = current_time
            
            # Only read the file if it has changed
            if current_size > last_log_size:
                with open(logfile, 'r') as f:
                    content = f.read()
                
                # Print any new content to help with debugging
                if len(content) > last_log_size:
                    new_content = content[last_log_size:]
                    if "error" in new_content.lower() or "failed" in new_content.lower():
                        print("\nWARNING: Possible error in integrator log:")
                        print(new_content.strip())
                
                # Check for common errors that indicate the process won't succeed
                if "CUDA driver version is insufficient" in content or "NVML: Driver Not Loaded" in content:
                    print("\nERROR: CUDA driver issues detected. Check NVIDIA drivers are properly installed.")
                    return False
                
                if "error while loading shared libraries" in content:
                    print("\nERROR: Library loading issues detected. Check LD_LIBRARY_PATH.")
                    return False
                
                last_log_size = current_size
                last_log_check = current_time
        
        time.sleep(check_interval)
    
    print(f"\nTimed out waiting for IntegratorFitPeaksGPUStream to start after {timeout} seconds")
    print("Checking for any output in the log file:")
    
    # Show log file contents to help debugging
    if os.path.exists(logfile):
        with open(logfile, 'r') as f:
            content = f.read().strip()
            if content:
                print("\n----- Integrator Log Contents -----")
                print(content[-2000:] if len(content) > 2000 else content)  # Show last 2000 chars
                print("----- End of Log Contents -----")
            else:
                print("Log file exists but is empty.")
    else:
        print("No log file was created.")
    
    return False

def check_and_create_mapping_files(param_file, midas_env):
    """
    Check if mapping files exist (Map.bin, nMap.bin), and run DetectorMapper if they don't
    
    Args:
        param_file: Parameter file for the detector
        midas_env: Environment dictionary with LD_LIBRARY_PATH set
        
    Returns:
        Boolean indicating if mapping files exist or were successfully created
    """
    map_file = "Map.bin"
    nmap_file = "nMap.bin"
    
    # Check if mapping files already exist
    if os.path.exists(map_file) and os.path.exists(nmap_file):
        print(f"Mapping files ({map_file}, {nmap_file}) found in current directory.")
        return True
    
    # Files don't exist, need to run the mapper
    print(f"Mapping files not found. Running DetectorMapper to create them...")
    
    detector_mapper = os.path.expanduser("~/opt/MIDAS/FF_HEDM/bin/DetectorMapper")
    mapper_cmd = [detector_mapper, param_file]
    mapper_log = "detector_mapper.log"
    
    try:
        print(f"Running command: {' '.join(mapper_cmd)}")
        with open(mapper_log, 'w') as logfile:
            process = subprocess.Popen(
                mapper_cmd,
                stdout=logfile,
                stderr=subprocess.STDOUT,
                env=midas_env
            )
            
            # Poll for process completion with 1-second interval
            start_time = time.time()
            timeout = 300  # 5 minute timeout
            
            print("Waiting for DetectorMapper to complete...", end="", flush=True)
            while process.poll() is None:
                # Check if we've exceeded the timeout
                if time.time() - start_time > timeout:
                    print("\nERROR: DetectorMapper timed out after 300 seconds")
                    process.kill()
                    print(f"Check {mapper_log} for details")
                    return False
                
                # Print a dot every 5 seconds to show progress
                if int((time.time() - start_time) % 5) == 0:
                    print(".", end="", flush=True)
                
                # Sleep for a short period before checking again
                time.sleep(1)
            
            print("")  # New line after the progress dots
            
            # Check the return code
            if process.returncode != 0:
                print(f"ERROR: DetectorMapper exited with code {process.returncode}")
                print(f"Check {mapper_log} for details")
                return False
            
            print(f"DetectorMapper completed in {time.time() - start_time:.1f} seconds")
        
        # Check if files were created
        if os.path.exists(map_file) and os.path.exists(nmap_file):
            print(f"Successfully created mapping files.")
            return True
        else:
            print(f"ERROR: DetectorMapper completed but mapping files were not created.")
            print(f"Check {mapper_log} for details")
            return False
            
    except Exception as e:
        print(f"ERROR: Failed to run DetectorMapper: {e}")
        return False

def find_process_by_name(name):
    """Find process IDs by name pattern"""
    pids = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if proc.info['cmdline']:
                cmdline = " ".join(proc.info['cmdline'])
                if name in cmdline:
                    pids.append(proc.info['pid'])
            elif proc.info['name'] and name in proc.info['name']:
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

def monitor_processing(mapping_file, expected_frames=None, check_interval=0.5):
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
    
    # Get install dir from path of executable
    install_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Validate dark file if specified
    dark_file = args.dark
    if dark_file and not os.path.exists(dark_file):
        print(f"Warning: Dark file '{dark_file}' not found")
        dark_file = None
    
    # Set up file paths
    integrator_log = "integrator.log"
    server_log = "server.log"
    
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
    
    # Set up environment with the required LD_LIBRARY_PATH
    midas_env = os.environ.copy()
    # path should be set to the location of the MIDAS libraries: ${install_dir}/FF_HEDM/build/lib and ${install_dir}/FF_HEDM/build/include
    midas_env["LD_LIBRARY_PATH"] = os.path.join(install_dir, "FF_HEDM", "build", "lib") + ":" + midas_env.get("LD_LIBRARY_PATH", "")
    midas_env["LD_LIBRARY_PATH"] = os.path.join(install_dir, "FF_HEDM", "build", "include") + ":" + midas_env.get("LD_LIBRARY_PATH", "")
    
    # Print the environment for debugging
    print(f"Using LD_LIBRARY_PATH: {midas_env['LD_LIBRARY_PATH']}")
    
    # Add helpful environment variable that might be needed
    midas_env["CUDA_VISIBLE_DEVICES"] = "0"  # Use the first GPU
    
    # Check if mapping files exist and create them if needed
    if not check_and_create_mapping_files(param_file, midas_env):
        print("Error: Failed to create required mapping files.")
        sys.exit(1)
        
    # Start IntegratorFitPeaksGPUStream in background
    print("Starting IntegratorFitPeaksGPUStream...")
    
    integrator_executable = os.path.expanduser("~/opt/MIDAS/FF_HEDM/bin/IntegratorFitPeaksGPUStream")
    
    integrator_cmd = [integrator_executable, param_file]
    if dark_file:
        integrator_cmd.append(dark_file)
        print(f"Using dark correction file: {dark_file}")
    
    print(f"Running command: {' '.join(integrator_cmd)}")
    
    with open(integrator_log, 'w') as logfile:
        integrator_proc = subprocess.Popen(
            integrator_cmd,
            stdout=logfile,
            stderr=subprocess.STDOUT,
            env=midas_env
        )
    
    print(f"Started IntegratorFitPeaksGPUStream with PID {integrator_proc.pid}")
    
    # Wait for server to be ready with improved monitoring
    if not wait_for_server_ready(integrator_log, integrator_proc, timeout=180):
        print("Error: Failed to start IntegratorFitPeaksGPUStream")
        
        # Check if process is still running
        if integrator_proc.poll() is None:
            print("The process is still running. Terminating it now...")
            kill_process(integrator_proc.pid)
        else:
            print(f"Process already exited with code {integrator_proc.returncode}")
        
        print(f"Check {integrator_log} for details")
        sys.exit(1)
    
    # Prepare server command
    server_cmd = [
        sys.executable,  # Add this line to use the current Python interpreter
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
        sys.executable,  # Add this line to use the current Python interpreter
        os.path.expanduser("~/opt/MIDAS/utils/integrator_stream_process_h5.py"),
        "--lineout", "lineout.bin",
        "--fit", "fit.bin",
        "--int2d", "Int2D.bin",
        "--fit-curves", "fit_curves.bin",
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