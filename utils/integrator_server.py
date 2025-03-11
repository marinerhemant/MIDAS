import socket
import time
import numpy as np
import struct
import ctypes
from PIL import Image
import matplotlib.pyplot as plt
class MyStruct(ctypes.Structure):
    _fields_ = [("field1", ctypes.c_uint16)]

def send_data_chunk(sock, dataset_num, data):
    t1 = time.time()
    
    # Pack the dataset number
    header = struct.pack('!H', dataset_num)
    
    format_string = '!' + 'H' * len(data)
    packed_data = struct.pack(format_string, *data)
    combined_data = header + packed_data
    sock.sendall(combined_data)
    
    t2 = time.time()
    print(f"Time taken to send data: {t2 - t1:.4f} sec")
    print(f"Sent dataset #{dataset_num} with {len(data)} uint16_t values ({len(packed_data)} bytes)")

def main():
    # Connect to C server
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_address = ('localhost', 5000)
    print(f"Connecting to {server_address[0]}:{server_address[1]}")
    
    try:
        sock.connect(server_address)
        
        # Number of uint16_t values to send in each chunk
        # Each uint16_t is 2 bytes, so 512 values = 1024 bytes
        num_values = 2048*2048  
        
        # Dataset counter
        dataset_num = 0
        
        # Generate random uint16_t values (0-12000)
        data = np.array(Image.open('test.tif')).astype(np.uint16).reshape((2048*2048))
        while True:
            t1 = time.time()
            # Send the data with dataset number
            send_data_chunk(sock, dataset_num, data)
            
            # Increment dataset number (wrap around at 65535)
            dataset_num = (dataset_num + 1) % 65536
            
            # Add a small delay between sends
            if dataset_num == 1:
                time.sleep(1)
            else:
                time.sleep(0.0014)
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

if __name__ == "__main__":
    main()