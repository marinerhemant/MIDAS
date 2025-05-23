import socket
import time
import numpy as np
import struct
import ctypes
from PIL import Image
import matplotlib.pyplot as plt
import pvaccess
import os
import glob
import argparse
import re
import json
import atexit
import tifffile


def send_data_chunk(sock, dataset_num, data):
    t1 = time.time()
    
    # Check if data is already a numpy array with correct dtype
    if not isinstance(data, np.ndarray) or data.dtype != np.int64:
        # Convert data to numpy array
        data_array = np.array(data, dtype=np.int64)
    else:
        # Use the existing array
        data_array = data
    
    # Pack the dataset number
    header = struct.pack('H', dataset_num)
    
    # Use memoryview instead of tobytes() to avoid a copy
    data_view = memoryview(data_array).cast('B')
    
    # Combine and send in a single call
    sock.sendall(header + data_view)
    
    t2 = time.time()
    print(f"Sent dataset #{dataset_num} with {len(data_array)} int64_t values ({len(data_view)} bytes) in {t2 - t1:.4f} sec")


def process_image(x, sock, dataset_num, frame_mapping, frame_index):
    data = (x['value'][0]['intValue']).reshape(1679, 1475)
    data = data.flatten()
    t1 = time.time()
    
    # Update frame mapping
    frame_mapping[frame_index] = {
        'dataset_num': dataset_num,
        'uniqueId': x.get('uniqueId', dataset_num),
        'timestamp': time.time()
    }
    
    # Send the data with dataset number
    send_data_chunk(sock, dataset_num, data)
    
    # Return incremented dataset number (wrap around at 65535)
    return (dataset_num + 1) % 65536, frame_index + 1


def process_tif(file_path, sock, dataset_num, frame_mapping, frame_index):
    """
    Process a single TIF file
    Returns: Updated dataset_num and frame_index
    """
    t1 = time.time()
    # image = Image.open(file_path)
    print(f"Processing TIF file: {file_path}")
    # data = np.array(image).flatten()
    data = tifffile.imread(file_path)
    t_int = time.time()
    # Update frame mapping with filename
    frame_mapping[frame_index] = {
        'dataset_num': dataset_num,
        'filename': os.path.basename(file_path),
        'timestamp': time.time()
    }
    
    # Send the data with dataset number
    send_data_chunk(sock, dataset_num, data)
    t2 = time.time()
    print(f"Processed TIF frame #{dataset_num}, time: {t2 - t1:.4f} sec, {t2-t_int:.4f} sec to send data {t_int-t1:.4f} sec to read file.")
    
    # Return incremented dataset numbers
    return (dataset_num + 1) % 65536, frame_index + 1


def process_binary_ge(file_path, sock, dataset_num, frame_mapping, frame_index, frame_size):
    """
    Process binary .geX files with 8192-byte header
    X is a number between 1-5
    """
    print(file_path)
    with open(file_path, 'rb') as f:
        # Read the entire file
        data = f.read()
        
        # Calculate how many frames should be in the file
        header_size = 8192  # 8192 bytes
        bytes_per_pixel = 2  # uint16 = 2 bytes
        frame_pixels = frame_size[0] * frame_size[1]
        bytes_per_frame = frame_pixels * bytes_per_pixel
        total_size = len(data)
        
        # Calculate number of complete frames
        num_frames = (total_size - header_size) // bytes_per_frame
        
        print(f"Found {num_frames} frames in binary file")
        
        base_filename = os.path.basename(file_path)
        
        for frame_idx in range(num_frames):
            frame_start = header_size + frame_idx * bytes_per_frame
            frame_end = frame_start + bytes_per_frame
            
            # Extract the frame data as bytes
            frame_bytes = data[frame_start:frame_end]
            
            # Convert bytes to numpy array of uint16
            frame_data = np.frombuffer(frame_bytes, dtype=np.uint16)
            
            # Reshape to the specified frame size
            frame_data = frame_data.reshape(frame_size)
            
            # Convert to int64 for sending
            frame_data = frame_data.astype(np.int64).flatten()
            
            # Update frame mapping
            frame_mapping[frame_index] = {
                'dataset_num': dataset_num,
                'filename': f"{base_filename}:frame{frame_idx}",
                'timestamp': time.time()
            }
            
            # Send the data
            send_data_chunk(sock, dataset_num, frame_data)
            
            # Increment counters
            dataset_num = (dataset_num + 1) % 65536
            frame_index += 1
            
    # Return the updated dataset number and frame index
    return dataset_num, frame_index


def process_h5(file_path, sock, dataset_num, frame_mapping, frame_index, h5_location):
    """
    Process HDF5 files using h5py
    """
    try:
        import h5py
    except ImportError:
        print("Error: h5py library not installed. Install with 'pip install h5py'")
        return dataset_num, frame_index
    
    print(file_path)
    base_filename = os.path.basename(file_path)
    frames_processed = 0
    
    with h5py.File(file_path, 'r') as f:
        # Access the datasets at the user-provided location
        if h5_location in f:
            dataset = f[h5_location]
            
            # Check if it's a single dataset or a group with multiple datasets
            if isinstance(dataset, h5py.Dataset):
                # Single dataset
                print(f"Processing single dataset with shape {dataset.shape}")
                
                # If it's a 3D array, it contains multiple frames
                if len(dataset.shape) == 3:
                    num_frames = dataset.shape[0]
                    print(f"Dataset contains {num_frames} frames")
                    for i in range(num_frames):
                        frame_data = dataset[i].astype(np.int64).flatten()
                        
                        # Update frame mapping
                        frame_mapping[frame_index] = {
                            'dataset_num': dataset_num,
                            'filename': f"{base_filename}:{h5_location}:frame{i}",
                            'timestamp': time.time()
                        }
                        
                        send_data_chunk(sock, dataset_num, frame_data)
                        dataset_num = (dataset_num + 1) % 65536
                        frame_index += 1
                        frames_processed += 1
                else:
                    # Single frame
                    frame_data = dataset[:].astype(np.int64).flatten()
                    
                    # Update frame mapping
                    frame_mapping[frame_index] = {
                        'dataset_num': dataset_num,
                        'filename': f"{base_filename}:{h5_location}",
                        'timestamp': time.time()
                    }
                    
                    send_data_chunk(sock, dataset_num, frame_data)
                    dataset_num = (dataset_num + 1) % 65536
                    frame_index += 1
                    frames_processed += 1
            else:
                # Group with multiple datasets
                print(f"Processing group with keys: {list(dataset.keys())}")
                for key in dataset.keys():
                    if isinstance(dataset[key], h5py.Dataset):
                        frame_data = dataset[key][:].astype(np.int64).flatten()
                        
                        # Update frame mapping
                        frame_mapping[frame_index] = {
                            'dataset_num': dataset_num,
                            'filename': f"{base_filename}:{h5_location}/{key}",
                            'timestamp': time.time()
                        }
                        
                        send_data_chunk(sock, dataset_num, frame_data)
                        dataset_num = (dataset_num + 1) % 65536
                        frame_index += 1
                        frames_processed += 1
        else:
            print(f"Error: Location '{h5_location}' not found in H5 file")
    
    print(f"Processed {frames_processed} frames from H5 file")
    return dataset_num, frame_index


def save_frame_mapping(frame_mapping, mapping_file):
    """
    Save the frame mapping to a JSON file
    """
    # Convert integer keys to strings for JSON compatibility
    str_mapping = {str(k): v for k, v in frame_mapping.items()}
    
    try:
        with open(mapping_file, 'w') as f:
            json.dump(str_mapping, f, indent=2)
        print(f"Frame mapping saved to {mapping_file}")
    except Exception as e:
        print(f"Error saving frame mapping to {mapping_file}: {e}")


def save_mapping_at_exit(frame_mapping, mapping_file):
    """
    Function to be called at program exit to save mapping
    """
    if frame_mapping:
        print("Saving final frame mapping...")
        save_frame_mapping(frame_mapping, mapping_file)


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Image data processor and streamer')
    
    # General arguments
    parser.add_argument('--stream', type=int, choices=[0, 1], default=0,
                        help='Stream mode: 0=files, 1=PVA (default: 0)')
    
    # Arguments for stream=0 (file mode)
    parser.add_argument('--folder', type=str, 
                        help='Folder containing image files (required if stream=0)')
    parser.add_argument('--extension', type=str, default='tif',
                        help='File extension to process (default: tif, supports tif, ge1-5, h5)')
    parser.add_argument('--frame-size', type=str, 
                        help='Frame size for binary files, format: WIDTHxHEIGHT (e.g., 2048x2048)')
    parser.add_argument('--h5-location', type=str, default='exchange/data',
                        help='Location within H5 file containing image data (default: exchange/data)')
    
    # Arguments for stream=1 (PVA mode)
    parser.add_argument('--pva-ip', type=str, default='10.54.105.139',
                        help='PVA server IP address (default: 10.54.105.139)')
    parser.add_argument('--channel', type=str, default='16pil-idb:Pva1:Image',
                        help='PVA channel string (default: 16pil-idb:Pva1:Image)')
    
    # Add argument for mapping file
    parser.add_argument('--mapping-file', type=str, default='frame_mapping.json',
                        help='Output JSON file for frame-to-dataset mapping (default: frame_mapping.json)')
    parser.add_argument('--save-interval', type=int, default=10,
                        help='Save mapping file every N frames (default: 10)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.stream == 0 and not args.folder:
        parser.error("--folder is required when stream=0")
    
    # For .geX files, frame size is required
    if args.stream == 0 and args.extension.startswith('ge') and not args.frame_size:
        parser.error("--frame-size is required for .geX files (format: WIDTHxHEIGHT, e.g., 2048x2048)")
    
    # Parse frame size if provided
    frame_size = None
    if args.frame_size:
        match = re.match(r'(\d+)x(\d+)', args.frame_size)
        if not match:
            parser.error("--frame-size must be in format WIDTHxHEIGHT (e.g., 2048x2048)")
        frame_size = (int(match.group(2)), int(match.group(1)))  # Height x Width for numpy reshape
    
    # Set up socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    dataset_num = 0  # This will wrap around at 65536
    frame_index = 0  # This will not wrap around
    total_frames = 0  # This will count the actual total number of frames processed
    
    # Initialize frame mapping dictionary
    frame_mapping = {}
    
    # Register exit handler to save mapping on program exit
    atexit.register(save_mapping_at_exit, frame_mapping, args.mapping_file)
    
    # Set PVA server IP if in PVA mode
    if args.stream == 1:
        os.environ['EPICS_PVA_ADDR_LIST'] = args.pva_ip
    
    # Connect to C server - fixed address and port
    server_address = ('127.0.0.1', 60439)
    print(f"Connecting to {server_address[0]}:{server_address[1]}")
    
    try:
        sock.connect(server_address)
        t0 = time.time()
        
        if args.stream == 1:
            # Stream mode: PVA
            print(f"Connecting to PVA channel: {args.channel}")
            
            # Define a monitor function that captures dataset_num and frame_index
            def pva_monitor_callback(x):
                nonlocal dataset_num, frame_index, total_frames
                dataset_num, frame_index = process_image(x, sock, dataset_num, frame_mapping, frame_index)
                total_frames += 1
                
                # Save mapping at regular intervals
                if total_frames % args.save_interval == 0:
                    save_frame_mapping(frame_mapping, args.mapping_file)
                
                if total_frames % 100 == 0:  # Print status every 100 frames
                    elapsed = time.time() - t0
                    print(f"Processed {total_frames} frames, average FPS: {total_frames/elapsed:.2f}")
            
            # Set up PVA channel
            channel = pvaccess.Channel(args.channel)
            channel.monitor(pva_monitor_callback, 'field(uniqueId, value)')
            
            # Keep the program running
            while True:
                time.sleep(1)
                
        else:
            # Stream mode: Files
            print(f"Processing files with extension .{args.extension} from {args.folder}")
            
            # Handle different file types
            if args.extension.lower() == 'tif':
                # Process TIFF files
                files = glob.glob(os.path.join(args.folder, f'*.{args.extension}'))
                print(f"Found {len(files)} .{args.extension} files")
                
                for file in files:
                    dataset_num, frame_index = process_tif(file, sock, dataset_num, frame_mapping, frame_index)
                    total_frames += 1
                    
                    # Save mapping at regular intervals
                    if total_frames % args.save_interval == 0:
                        save_frame_mapping(frame_mapping, args.mapping_file)
                    
            elif args.extension.lower().startswith('ge') and args.extension[2:].isdigit():
                # Process .geX binary files
                files = glob.glob(os.path.join(args.folder, f'*.{args.extension}'))
                print(f"Found {len(files)} .{args.extension} files")
                
                for file in files:
                    frames_in_file_before = total_frames
                    dataset_num, frame_index = process_binary_ge(file, sock, dataset_num, frame_mapping, frame_index, frame_size)
                    # Calculate how many frames were in this file
                    frames_in_file = frame_index - total_frames
                    total_frames = frame_index
                    
                    # Save mapping at regular intervals
                    if total_frames % args.save_interval == 0:
                        save_frame_mapping(frame_mapping, args.mapping_file)
                    
            elif args.extension.lower() == 'h5':
                # Process H5 files
                files = glob.glob(os.path.join(args.folder, f'*.{args.extension}'))
                print(f"Found {len(files)} .{args.extension} files")
                
                for file in files:
                    frames_in_file_before = total_frames
                    dataset_num, frame_index = process_h5(file, sock, dataset_num, frame_mapping, frame_index, args.h5_location)
                    # Calculate how many frames were in this file
                    frames_in_file = frame_index - total_frames
                    total_frames = frame_index
                    
                    # Save mapping at regular intervals
                    if total_frames % args.save_interval == 0:
                        save_frame_mapping(frame_mapping, args.mapping_file)
                    
            else:
                print(f"Unsupported file extension: {args.extension}")
            
            # Final save of mapping
            save_frame_mapping(frame_mapping, args.mapping_file)
            
    except KeyboardInterrupt:
        print("Sending terminated by user")
        # Save mapping on interrupt
        save_frame_mapping(frame_mapping, args.mapping_file)
    except Exception as e:
        print(f"Error: {e}")
        # Try to save mapping even on error
        save_frame_mapping(frame_mapping, args.mapping_file)
    finally:
        # Close connection
        sock.close()
        print("Connection closed")
        if total_frames > 0:
            total_time = time.time() - t0
            print(f'Sent {total_frames} frames',
                  f"Total time: {total_time:.4f} sec", 
                  f'Average fps: {total_frames/total_time:.4f} fps')


if __name__ == "__main__":
    main()