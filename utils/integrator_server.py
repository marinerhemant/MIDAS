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

os.environ['EPICS_PVA_ADDR_LIST'] = '10.54.105.139'  # Enable PVA server

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
dataset_num = 0

def send_data_chunk(sock, dataset_num, data):
    t1 = time.time()
    
    # Check if data is already a numpy array with correct dtype
    if not isinstance(data, np.ndarray) or data.dtype != np.int32:
        # Convert data to numpy array
        data_array = np.array(data, dtype=np.int32)
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
    print(f"Sent dataset #{dataset_num} with {len(data_array)} int32_t values ({len(data_view)} bytes) in {t2 - t1:.4f} sec")

def processImage(x):
    global dataset_num
    data = (x['value'][0]['intValue']).reshape(1679,1475)
    # Image.fromarray(np.log(data+3)).save(f"frame_{dataset_num}.tif")  # Save the image for debugging
    data = data.flatten()
    t1 = time.time()
    # Send the data with dataset number
    send_data_chunk(sock, dataset_num, data)
    # Increment dataset number (wrap around at 65535)
    dataset_num = (dataset_num + 1) % 65536
    t2 = time.time()
    print(f"Time taken: {t2 - t1:.4f} sec")

def main():
    stream = 0
    # Connect to C server
    dataset_num = 0

    server_address = ('127.0.0.1', 5000)
    print(f"Connecting to {server_address[0]}:{server_address[1]}")
    try:
        sock.connect(server_address)
        if stream == 1:
            t0 = time.time()
            channel = pvaccess.Channel('16pil-idb:Pva1:Image')
            channel.monitor(processImage,'field(uniqueId, value)')
            while True:
                pass
        else:
            folder = '/gdata/dm/1ID/2024/AMsetup_nov24/data/pilatus/EXP030_sam4_Pos1_1ms_250Hz_line_P39p47_S0p04_StF2p5_FTopm50_Att000/'
            files = glob.glob(folder+'*.tif')
            print(len(files))
            print(folder)
            t0 = time.time()
            for file in files:
                image = Image.open(file)
                print(file)
                data = np.array(image).flatten()
                t1 = time.time()
                # Send the data with dataset number
                send_data_chunk(sock, dataset_num, data)
                # Increment dataset number (wrap around at 65535)
                dataset_num = (dataset_num + 1) % 65536
                t2 = time.time()
                print(f"Time taken: {t2 - t1:.4f} sec")
    except KeyboardInterrupt:
        print("Sending terminated by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Close connection
        sock.close()
        print("Connection closed")
        print(f'Sent {dataset_num} frames',f"Total time: {time.time() - t0:.4f} sec", f'Average fps: {dataset_num/(time.time() - t0):.4f} sec')

if __name__ == "__main__":
    main()