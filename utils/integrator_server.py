import socket
import time
import numpy as np
import struct

def send_data_chunk(sock, dataset_num, data):
    # 'H' format specifier is for unsigned short (uint16_t)
    # First pack the dataset number as uint16_t
    header = struct.pack('!H', dataset_num)
    # Then pack uint16_t values into bytes
    packed_data = struct.pack(f'!{len(data)}H', *data)
    # Send header followed by data
    sock.sendall(header + packed_data)
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
        
        # Generate random uint16_t values (0-65535)
        data = np.random.randint(0, 65536, num_values, dtype=np.uint16)
        while True:
            t1 = time.time()
            # Send the data with dataset number
            send_data_chunk(sock, dataset_num, data)
            
            # Increment dataset number (wrap around at 65535)
            dataset_num = (dataset_num + 1) % 65536
            
            # Add a small delay between sends
            if dataset_num == 0:
                time.sleep(0.1)
            else:
                time.sleep(0.001)
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